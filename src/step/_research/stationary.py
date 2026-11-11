from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch
from torch import nn
from sklearn.covariance import LedoitWolf

from .data import StageData, build_knn_adjacency
from .stationary_transport import unbalanced_relational_sinkhorn


@dataclass(frozen=True)
class V16Config:
    latent_dim: int = 24
    hidden_dim: int = 96
    representation_epochs: int = 160
    flow_epochs: int = 180
    batch_size: int = 256
    learning_rate: float = 2e-3
    mask_rate: float = 0.12
    single_gene_mask_probability: float | None = None
    mask_loss_weight: float = 3.0
    context_loss_weight: float = 0.5
    latent_consistency_weight: float = 0.1
    relation_weight: float = 0.35
    entropy: float = 0.08
    marginal_relaxation: float = 1.0
    pairs_per_source: int = 2
    neighbors: int = 8
    integration_step: float = 0.05
    projection_steps: int = 30
    projection_rate: float = 0.2
    projection_proximity: float = 0.05
    projection_target_weight: float = 1.0
    uncertainty_q95: float = 0.95
    identity_representation: bool | None = None
    seed: int = 6120
    device: str = "auto"


@dataclass(frozen=True)
class V16FitResult:
    representation_loss: float
    flow_loss: float
    latent_dimension: int
    wt_rows: int
    transition_pairs: int


@dataclass(frozen=True)
class V16InterventionResult:
    baseline: np.ndarray
    perturbed: np.ndarray
    effect: np.ndarray
    uncertainty_score: float
    uncertainty_label: str
    projection_residual: float
    method: str = "self_supervised_latent_intervention"


class _MaskedSpatialAutoencoder(nn.Module):
    def __init__(self, genes: int, latent: int, hidden: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3 * genes, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.SiLU(), nn.Linear(hidden, genes),
        )
        self.context_decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.SiLU(), nn.Linear(hidden, genes),
        )

    def encode(self, expression, context, mask):
        return self.encoder(torch.cat((expression, context, mask), dim=1))


class _LatentFlow(nn.Module):
    def __init__(self, latent: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * latent + 3, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, latent),
        )

    def forward(self, z, niche, time):
        if not isinstance(time, torch.Tensor):
            time = torch.as_tensor(time, dtype=z.dtype, device=z.device)
        if time.ndim == 0:
            time = time.expand(len(z), 1)
        elif time.ndim == 1:
            time = time[:, None]
        code = torch.cat((time, torch.sin(math.pi * time), torch.cos(math.pi * time)), 1)
        return self.net(torch.cat((z, niche, code), 1))


class CADENCEV16:
    """Masked spatial representation and stationary/conditional latent flow."""

    def __init__(self, genes: int, time_min: float, time_max: float, config: V16Config = V16Config()):
        if time_max <= time_min:
            raise ValueError("time_max must exceed time_min")
        self.genes = int(genes)
        self.time_min = float(time_min)
        self.time_scale = float(time_max - time_min)
        self.config = config
        self.fitted_ = False

    @property
    def device(self):
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.config.device)

    def _time(self, value):
        return (torch.as_tensor(value, dtype=torch.float32, device=self.device) - self.time_min) / max(self.time_scale, 1e-8)

    @staticmethod
    def _dense_graph(stage: StageData, neighbors: int):
        return build_knn_adjacency(stage.coordinates, n_neighbors=neighbors).cpu().numpy().astype(np.float32)

    @staticmethod
    def _context(expression, graph):
        return np.asarray(graph, dtype=np.float32) @ np.asarray(expression, dtype=np.float32) - np.asarray(expression, dtype=np.float32)

    def fit(self, stages: Sequence[StageData], adjacencies=None, section_masses=None) -> V16FitResult:
        if len(stages) < 2:
            raise ValueError("at least two ordered WT stages are required")
        if any(b.time <= a.time for a, b in zip(stages[:-1], stages[1:])):
            raise ValueError("WT stages must be strictly ordered")
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        graphs = [self._dense_graph(s, self.config.neighbors) for s in stages] if adjacencies is None else [np.asarray(g, dtype=np.float32) for g in adjacencies]
        values = np.vstack([s.expression for s in stages]).astype(np.float32)
        contexts = np.vstack([self._context(s.expression, g) for s, g in zip(stages, graphs)]).astype(np.float32)
        self.center_ = np.median(values, axis=0).astype(np.float32)
        self.scale_ = np.maximum(np.quantile(np.abs(values - self.center_), .75, axis=0), .05).astype(np.float32)
        x = (values - self.center_) / self.scale_
        c = contexts / self.scale_
        latent = min(self.config.latent_dim, self.genes, max(2, len(values) - 1))
        self.identity_representation_ = (self.genes <= 8 and self.genes <= self.config.latent_dim) if self.config.identity_representation is None else bool(self.config.identity_representation)
        if self.identity_representation_:
            latent = self.genes
            self.autoencoder_ = None
            representation_history = [0.0]
        else:
            self.autoencoder_ = _MaskedSpatialAutoencoder(self.genes, latent, self.config.hidden_dim).to(self.device)
            optimizer = torch.optim.AdamW(self.autoencoder_.parameters(), lr=self.config.learning_rate, weight_decay=1e-4)
            xt, ct = torch.as_tensor(x, device=self.device), torch.as_tensor(c, device=self.device)
            generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
            representation_history = []
            for _ in range(self.config.representation_epochs):
                order = torch.randperm(len(xt), generator=generator)
                total = 0.0; batches = 0
                for start in range(0, len(xt), self.config.batch_size):
                    take = order[start:start + self.config.batch_size].to(self.device)
                    clean, context = xt[take], ct[take]
                    mask = torch.rand(clean.shape, device=self.device) < self.config.mask_rate
                    single = torch.randint(self.genes, (len(clean),), device=self.device)
                    probability = min(1.0, max(0.25, self.genes / 32.0)) if self.config.single_gene_mask_probability is None else float(self.config.single_gene_mask_probability)
                    selected_rows = torch.rand(len(clean), device=self.device) < probability
                    mask[torch.arange(len(clean), device=self.device)[selected_rows], single[selected_rows]] = True
                    corrupted = clean.masked_fill(mask, 0.0)
                    z_mask = self.autoencoder_.encode(corrupted, context, mask.float())
                    z_clean = self.autoencoder_.encode(clean, context, torch.zeros_like(clean))
                    recon = self.autoencoder_.decoder(z_mask)
                    context_pred = self.autoencoder_.context_decoder(z_clean)
                    target = torch.as_tensor(values[take.cpu().numpy()], device=self.device)
                    masked = ((recon - target).square() * mask).sum() / mask.sum().clamp_min(1)
                    observed = ((recon - target).square() * (~mask)).sum() / (~mask).sum().clamp_min(1)
                    loss = self.config.mask_loss_weight * masked + observed + self.config.context_loss_weight * (context_pred - context).square().mean() + self.config.latent_consistency_weight * (z_mask - z_clean).square().mean()
                    optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.autoencoder_.parameters(), 5.0); optimizer.step()
                    total += float(loss.detach()); batches += 1
                representation_history.append(total / max(batches, 1))
            self.autoencoder_.eval()
        self.graphs_ = graphs
        self.stages_ = list(stages)
        latent_stages = [self.encode(s.expression, g) for s, g in zip(stages, graphs)]
        masses = np.ones(len(stages)) if section_masses is None else np.asarray(section_masses, dtype=float)
        rng = np.random.default_rng(self.config.seed)
        lefts, rights, nleft, nright, times, durations = [], [], [], [], [], []
        for k, (za, zb) in enumerate(zip(latent_stages[:-1], latent_stages[1:])):
            ra, rb = self._context(za, graphs[k]), self._context(zb, graphs[k + 1])
            plan, _ = unbalanced_relational_sinkhorn(za, zb, ra, rb, source_mass=masses[k], target_mass=masses[k + 1], relation_weight=self.config.relation_weight, entropy=self.config.entropy, marginal_relaxation=self.config.marginal_relaxation)
            conditional = plan / np.maximum(plan.sum(1, keepdims=True), 1e-12)
            conditional /= np.maximum(conditional.sum(1, keepdims=True), 1e-12)
            for _ in range(self.config.pairs_per_source):
                chosen = np.asarray([rng.choice(len(zb), p=row) for row in conditional])
                lefts.append(za); rights.append(zb[chosen]); nleft.append(ra); nright.append(rb[chosen])
                times.append(np.full(len(za), stages[k].time)); durations.append(np.full(len(za), stages[k + 1].time - stages[k].time))
        a, b, na, nb = [np.vstack(v).astype(np.float32) for v in (lefts, rights, nleft, nright)]
        t, dt = np.concatenate(times).astype(np.float32), np.concatenate(durations).astype(np.float32)
        self.flow_ = _LatentFlow(latent, self.config.hidden_dim).to(self.device)
        optimizer = torch.optim.AdamW(self.flow_.parameters(), lr=self.config.learning_rate, weight_decay=1e-4)
        tensors = [torch.as_tensor(v, device=self.device) for v in (a, b, na, nb, t, dt)]
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        flow_history = []
        for _ in range(self.config.flow_epochs):
            order = torch.randperm(len(a), generator=generator); total = 0.0; batches = 0
            for start in range(0, len(a), self.config.batch_size):
                q = order[start:start + self.config.batch_size].to(self.device)
                aa, bb, n0, n1, tt, dd = [v[q] for v in tensors]
                r = torch.rand((len(q), 1), device=self.device)
                state, niche = (1 - r) * aa + r * bb, (1 - r) * n0 + r * n1
                velocity = (bb - aa) / dd[:, None]
                pred = self.flow_(state, niche, self._time(tt + r.ravel() * dd))
                loss = (pred - velocity).square().mean()
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.flow_.parameters(), 5.0); optimizer.step()
                total += float(loss.detach()); batches += 1
            flow_history.append(total / max(batches, 1))
        self.flow_.eval(); self.wt_latent_ = np.vstack(latent_stages); self.latent_center_ = np.median(self.wt_latent_, axis=0)
        self.latent_precision_ = LedoitWolf().fit(self.wt_latent_).precision_; self.uncertainty_reference_ = np.sort(self._latent_distance(self.wt_latent_))
        self.representation_history_, self.flow_history_ = representation_history, flow_history; self.fitted_ = True
        return V16FitResult(min(representation_history[-10:]), min(flow_history[-10:]), latent, len(values), len(a))

    def _check(self):
        if not self.fitted_:
            raise RuntimeError("CADENCEV16 must be fitted on WT/NTC snapshots")

    def encode(self, expression, graph, mask=None):
        values = np.asarray(expression, dtype=np.float32)
        if self.identity_representation_:
            result = values.copy()
            if mask is not None: result[np.asarray(mask, dtype=bool)] = 0.0
            return result
        context = self._context(values, graph) / self.scale_; normalized = (values - self.center_) / self.scale_
        indicator = np.zeros_like(values) if mask is None else np.asarray(mask, dtype=np.float32)
        normalized = normalized.copy(); normalized[indicator.astype(bool)] = 0.0
        with torch.no_grad():
            z = self.autoencoder_.encode(torch.as_tensor(normalized, device=self.device), torch.as_tensor(context, device=self.device), torch.as_tensor(indicator, device=self.device))
        return z.cpu().numpy().astype(np.float32)

    def decode(self, z):
        if self.identity_representation_:
            return np.clip(np.asarray(z, dtype=np.float32), 0.0, None)
        with torch.no_grad():
            value = self.autoencoder_.decoder(torch.as_tensor(z, dtype=torch.float32, device=self.device))
        return np.clip(value.cpu().numpy(), 0.0, None).astype(np.float32)

    def _latent_distance(self, z):
        delta = np.asarray(z) - self.latent_center_
        return np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", delta, self.latent_precision_, delta), 0))

    def _uncertainty(self, z):
        d = float(np.quantile(self._latent_distance(z), self.config.uncertainty_q95)); score = float((1 + np.sum(self.uncertainty_reference_ <= d)) / (len(self.uncertainty_reference_) + 1)); label = "supported" if score <= .95 else ("caution" if score <= .99 else "unsupported"); return score, label

    @staticmethod
    def _niche(z, graph):
        return torch.as_tensor(graph, dtype=z.dtype, device=z.device) @ z - z

    def _flow(self, z, graph, time):
        return self.flow_(z, self._niche(z, graph), self._time(time))

    def simulate_wt(self, expression, graph, start_time, end_time):
        self._check(); z = torch.as_tensor(self.encode(expression, graph), device=self.device); g = torch.as_tensor(graph, dtype=torch.float32, device=self.device); duration = float(end_time - start_time); steps = max(1, int(math.ceil(abs(duration) / self.config.integration_step))); h = duration / steps
        with torch.no_grad():
            for i in range(steps):
                t = start_time + i * h; k1 = self._flow(z, g, t); middle = z + .5 * h * k1; z = z + h * self._flow(middle, g, t + .5 * h)
        return self.decode(z.cpu().numpy())

    def _project_gene(self, z, rows, gene, target):
        if self.identity_representation_:
            value = z.detach().clone(); value[torch.as_tensor(rows, dtype=torch.bool, device=self.device), gene] = float(target); return value, 0.0
        anchor = z.detach(); value = anchor.clone().requires_grad_(True); mask = torch.as_tensor(rows, dtype=torch.bool, device=self.device)
        for _ in range(self.config.projection_steps):
            decoded = self.autoencoder_.decoder(value); target_loss = (decoded[mask, gene] - float(target)).square().mean(); prox = (value - anchor).square().mean(); loss = self.config.projection_target_weight * target_loss + self.config.projection_proximity * prox; gradient = torch.autograd.grad(loss, value)[0]; value = (value - self.config.projection_rate * gradient).detach().requires_grad_(True)
        residual = float(torch.sqrt((self.autoencoder_.decoder(value)[mask, gene] - float(target)).square().mean()).detach().cpu()); return value.detach(), residual

    def intervene(self, expression, graph, gene_index, node_mask, start_time, end_time, target_value=0.0):
        self._check(); values = np.asarray(expression, dtype=np.float32); rows = np.asarray(node_mask, dtype=bool); indicator = np.zeros_like(values, dtype=np.float32); indicator[rows, gene_index] = 1.; masked = values.copy(); masked[rows, gene_index] = target_value
        z0 = torch.as_tensor(self.encode(values, graph), device=self.device); zu = torch.as_tensor(self.encode(masked, graph, indicator), device=self.device); zu, residual = self._project_gene(zu, rows, gene_index, target_value); g = torch.as_tensor(graph, dtype=torch.float32, device=self.device); duration = float(end_time - start_time); steps = max(1, int(math.ceil(duration / self.config.integration_step))); h = duration / steps
        for i in range(steps):
            t = start_time + i * h
            with torch.no_grad(): z0 = z0 + h * self._flow(z0, g, t); zu = zu + h * self._flow(zu, g, t)
            zu, residual = self._project_gene(zu, rows, gene_index, target_value)
        baseline = self.decode(z0.cpu().numpy()); perturbed = self.decode(zu.cpu().numpy()); perturbed[rows, gene_index] = target_value; score, label = self._uncertainty(zu.cpu().numpy()); return V16InterventionResult(baseline, perturbed, (perturbed - baseline).astype(np.float32), score, label, residual)

    def dynamic_grn(self, expression, graph, time, epsilon=1e-2):
        self._check(); x = np.asarray(expression, dtype=np.float32); base_z = torch.as_tensor(self.encode(x, graph), device=self.device); g = torch.as_tensor(graph, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            next_base = base_z + self.config.integration_step * self._flow(base_z, g, time); base = self.decode(next_base.cpu().numpy()).mean(0)
        matrix = np.empty((self.genes, self.genes), dtype=np.float32)
        for regulator in range(self.genes):
            altered = x.copy(); altered[:, regulator] += epsilon; z = torch.as_tensor(self.encode(altered, graph), device=self.device)
            with torch.no_grad(): out = self.decode((z + self.config.integration_step * self._flow(z, g, time)).cpu().numpy()).mean(0)
            matrix[:, regulator] = (out - base) / epsilon
        return matrix
