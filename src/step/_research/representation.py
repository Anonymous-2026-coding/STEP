from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import torch
from torch import nn
from sklearn.covariance import LedoitWolf

from .data import StageData, build_knn_adjacency
from .transport import unbalanced_relational_sinkhorn


@dataclass(frozen=True)
class V17RepresentationFit:
    representation_loss: float
    flow_loss: float
    latent_dimension: int
    wt_rows: int
    transition_pairs: int


@dataclass(frozen=True)
class V17InterventionResult:
    baseline: np.ndarray
    perturbed: np.ndarray
    effect: np.ndarray
    uncertainty_score: float
    uncertainty_label: str
    projection_residual: float
    method: str = "self_supervised_latent_intervention"


class _MaskedSpatialAutoencoder(nn.Module):
    def __init__(self, genes, latent, hidden):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(3 * genes, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, hidden), nn.SiLU(), nn.Linear(hidden, genes))
        self.context_decoder = nn.Sequential(nn.Linear(latent, hidden), nn.SiLU(), nn.Linear(hidden, genes))

    def encode(self, expression, context, mask):
        return self.encoder(torch.cat((expression, context, mask), dim=1))


class _LatentFlow(nn.Module):
    def __init__(self, latent, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * latent + 3, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, latent))

    def forward(self, z, niche, time):
        time = torch.as_tensor(time, dtype=z.dtype, device=z.device)
        if time.ndim == 0: time = time.expand(len(z), 1)
        elif time.ndim == 1: time = time[:, None]
        code = torch.cat((time, torch.sin(math.pi * time), torch.cos(math.pi * time)), 1)
        return self.net(torch.cat((z, niche, code), 1))


class EditableRepresentation:
    """Self-contained v17 representation/latent-edit implementation."""
    def __init__(self, genes, time_min, time_max, config):
        if time_max <= time_min: raise ValueError("time_max must exceed time_min")
        self.genes = int(genes); self.time_min = float(time_min); self.time_scale = float(time_max - time_min); self.config = config; self.fitted_ = False

    @property
    def device(self):
        return torch.device("cuda" if self.config.device == "auto" and torch.cuda.is_available() else ("cpu" if self.config.device == "auto" else self.config.device))

    def _time(self, value):
        return (torch.as_tensor(value, dtype=torch.float32, device=self.device) - self.time_min) / max(self.time_scale, 1e-8)

    def _check(self):
        if not self.fitted_:
            raise RuntimeError("CADENCEV17 must be fitted on WT snapshots")

    @staticmethod
    def _context(expression, graph):
        return np.asarray(graph, dtype=np.float32) @ np.asarray(expression, dtype=np.float32) - np.asarray(expression, dtype=np.float32)

    def fit_representation(self, stages: Sequence[StageData], adjacencies=None, section_masses=None):
        if len(stages) < 2: raise ValueError("at least two ordered WT stages are required")
        if any(b.time <= a.time for a, b in zip(stages[:-1], stages[1:])): raise ValueError("WT stages must be strictly ordered")
        torch.manual_seed(self.config.seed); np.random.seed(self.config.seed)
        graphs = [np.asarray(g, dtype=np.float32) for g in adjacencies] if adjacencies is not None else [build_knn_adjacency(s.coordinates, n_neighbors=self.config.neighbors).numpy() for s in stages]
        values = np.vstack([s.expression for s in stages]).astype(np.float32); contexts = np.vstack([self._context(s.expression, g) for s, g in zip(stages, graphs)]).astype(np.float32)
        self.center_ = np.median(values, axis=0).astype(np.float32); self.scale_ = np.maximum(np.quantile(np.abs(values - self.center_), .75, axis=0), .05).astype(np.float32)
        x = (values - self.center_) / self.scale_; c = contexts / self.scale_; latent = min(self.config.latent_dim, self.genes, max(2, len(values) - 1))
        default_identity = self.genes <= 8 and self.genes <= self.config.latent_dim
        self.identity_representation_ = default_identity if getattr(self.config, "identity_representation", None) is None else bool(self.config.identity_representation)
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)
        history = [0.0]
        if self.identity_representation_: self.autoencoder_ = None; latent = self.genes
        else:
            self.autoencoder_ = _MaskedSpatialAutoencoder(self.genes, latent, self.config.hidden_dim).to(self.device); opt = torch.optim.AdamW(self.autoencoder_.parameters(), lr=self.config.learning_rate, weight_decay=1e-4); xt, ct = torch.as_tensor(x, device=self.device), torch.as_tensor(c, device=self.device); history = []
            for _ in range(self.config.representation_epochs):
                order = torch.randperm(len(xt), generator=generator); total = 0.; batches = 0
                for start in range(0, len(xt), self.config.batch_size):
                    take = order[start:start + self.config.batch_size].to(self.device); clean, context = xt[take], ct[take]; mask = torch.rand(clean.shape, device=self.device) < self.config.mask_rate; single = torch.randint(self.genes, (len(clean),), device=self.device); probability = min(1., max(.25, self.genes / 32.)); rows = torch.rand(len(clean), device=self.device) < probability; mask[torch.arange(len(clean), device=self.device)[rows], single[rows]] = True; corrupted = clean.masked_fill(mask, 0.); zm = self.autoencoder_.encode(corrupted, context, mask.float()); zc = self.autoencoder_.encode(clean, context, torch.zeros_like(clean)); recon = self.autoencoder_.decoder(zm); target = torch.as_tensor(values[take.cpu().numpy()], device=self.device); masked = ((recon - target).square() * mask).sum() / mask.sum().clamp_min(1); observed = ((recon - target).square() * (~mask)).sum() / (~mask).sum().clamp_min(1); loss = self.config.mask_loss_weight * masked + observed + self.config.context_loss_weight * (self.autoencoder_.context_decoder(zc) - context).square().mean() + self.config.latent_consistency_weight * (zm - zc).square().mean(); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.autoencoder_.parameters(), 5.); opt.step(); total += float(loss.detach()); batches += 1
                history.append(total / max(batches, 1))
            self.autoencoder_.eval()
        self.graphs_ = graphs; self.stages_ = list(stages)
        latent_stages = [self.encode(stage.expression, graph) for stage, graph in zip(stages, graphs)]

        # Preserve the original v17 initialization sequence. The historical
        # implementation fitted one provisional v16-style flow epoch before
        # replacing it with the mass-weighted v17 flow. This step affects RNG
        # state and therefore must remain inside the self-contained v17 code.
        masses = np.ones(len(stages)) if section_masses is None else np.asarray(section_masses, dtype=float)
        rng = np.random.default_rng(self.config.seed)
        lefts, rights, nleft, nright, times, durations = [], [], [], [], [], []
        for index, (left, right) in enumerate(zip(latent_stages[:-1], latent_stages[1:])):
            relation_left, relation_right = self._context(left, graphs[index]), self._context(right, graphs[index + 1])
            plan, _ = unbalanced_relational_sinkhorn(
                left, right, relation_left, relation_right,
                source_mass=masses[index], target_mass=masses[index + 1],
                relation_weight=self.config.relation_weight, entropy=self.config.entropy,
                marginal_relaxation=self.config.marginal_relaxation,
            )
            conditional = plan / np.maximum(plan.sum(1, keepdims=True), 1e-12)
            conditional /= np.maximum(conditional.sum(1, keepdims=True), 1e-12)
            chosen = np.asarray([rng.choice(len(right), p=row) for row in conditional])
            lefts.append(left); rights.append(right[chosen]); nleft.append(relation_left); nright.append(relation_right[chosen])
            times.append(np.full(len(left), stages[index].time)); durations.append(np.full(len(left), stages[index + 1].time - stages[index].time))
        a, b, na, nb = [np.vstack(items).astype(np.float32) for items in (lefts, rights, nleft, nright)]
        time, duration = np.concatenate(times).astype(np.float32), np.concatenate(durations).astype(np.float32)
        self.flow_ = _LatentFlow(latent, self.config.hidden_dim).to(self.device)
        optimizer = torch.optim.AdamW(self.flow_.parameters(), lr=self.config.learning_rate, weight_decay=1e-4)
        tensors = [torch.as_tensor(item, device=self.device) for item in (a, b, na, nb, time, duration)]
        order = torch.randperm(len(a), generator=generator); total = 0.; batches = 0
        for start in range(0, len(a), self.config.batch_size):
            take = order[start:start + self.config.batch_size].to(self.device)
            aa, bb, n0, n1, tt, dt = [item[take] for item in tensors]
            fraction = torch.rand((len(take), 1), device=self.device)
            state = (1-fraction)*aa + fraction*bb; niche = (1-fraction)*n0 + fraction*n1
            truth = (bb-aa)/dt[:, None]; prediction = self.flow_(state, niche, self._time(tt + fraction.ravel()*dt))
            loss = (prediction-truth).square().mean(); optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.flow_.parameters(), 5.); optimizer.step(); total += float(loss.detach()); batches += 1
        provisional_loss = total / max(batches, 1)

        self.wt_latent_ = np.vstack(latent_stages); self.latent_center_ = np.median(self.wt_latent_, axis=0); self.latent_precision_ = LedoitWolf().fit(self.wt_latent_).precision_; self.uncertainty_reference_ = np.sort(self._latent_distance(self.wt_latent_)); self.fitted_ = True
        return V17RepresentationFit(min(history[-10:]), provisional_loss, latent, len(values), len(a))

    def encode(self, expression, graph, mask=None):
        values = np.asarray(expression, dtype=np.float32)
        if self.identity_representation_:
            out = values.copy();
            if mask is not None: out[np.asarray(mask, dtype=bool)] = 0.; return out
            return out
        context = self._context(values, graph) / self.scale_; normalized = (values - self.center_) / self.scale_; indicator = np.zeros_like(values) if mask is None else np.asarray(mask, dtype=np.float32); normalized = normalized.copy(); normalized[indicator.astype(bool)] = 0.
        with torch.no_grad(): z = self.autoencoder_.encode(torch.as_tensor(normalized, device=self.device), torch.as_tensor(context, device=self.device), torch.as_tensor(indicator, device=self.device))
        return z.cpu().numpy().astype(np.float32)

    def decode(self, z):
        if self.identity_representation_: return np.clip(np.asarray(z, dtype=np.float32), 0., None)
        with torch.no_grad(): value = self.autoencoder_.decoder(torch.as_tensor(z, dtype=torch.float32, device=self.device))
        return np.clip(value.cpu().numpy(), 0., None).astype(np.float32)

    def _latent_distance(self, z):
        delta = np.asarray(z) - self.latent_center_; return np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", delta, self.latent_precision_, delta), 0))

    def _uncertainty(self, z):
        d = float(np.quantile(self._latent_distance(z), getattr(self.config, "uncertainty_q95", .95))); score = float((1 + np.sum(self.uncertainty_reference_ <= d)) / (len(self.uncertainty_reference_) + 1)); return score, ("supported" if score <= .95 else ("caution" if score <= .99 else "unsupported"))

    @staticmethod
    def _niche(z, graph): return torch.as_tensor(graph, dtype=z.dtype, device=z.device) @ z - z

    def _flow(self, z, graph, time): return self.flow_(z, self._niche(z, graph), self._time(time))

    def _project_gene(self, z, rows, gene, target):
        if self.identity_representation_:
            value = z.detach().clone(); value[torch.as_tensor(rows, dtype=torch.bool, device=self.device), gene] = float(target); return value, 0.
        anchor = z.detach(); value = anchor.clone().requires_grad_(True); mask = torch.as_tensor(rows, dtype=torch.bool, device=self.device)
        for _ in range(self.config.projection_steps):
            decoded = self.autoencoder_.decoder(value); loss = self.config.projection_target_weight * (decoded[mask, gene] - float(target)).square().mean() + self.config.projection_proximity * (value - anchor).square().mean(); grad = torch.autograd.grad(loss, value)[0]; value = (value - self.config.projection_rate * grad).detach().requires_grad_(True)
        residual = float(torch.sqrt((self.autoencoder_.decoder(value)[mask, gene] - float(target)).square().mean()).detach().cpu()); return value.detach(), residual

    def intervene(self, expression, graph, gene_index, node_mask, start_time, end_time, target_value=0.):
        values = np.asarray(expression, dtype=np.float32); rows = np.asarray(node_mask, dtype=bool); indicator = np.zeros_like(values, dtype=np.float32); indicator[rows, gene_index] = 1.; masked = values.copy(); masked[rows, gene_index] = target_value; z0 = torch.as_tensor(self.encode(values, graph), device=self.device); zu, residual = self._project_gene(torch.as_tensor(self.encode(masked, graph, indicator), device=self.device), rows, gene_index, target_value); g = torch.as_tensor(graph, dtype=torch.float32, device=self.device); duration = float(end_time - start_time); steps = max(1, int(math.ceil(duration / self.config.integration_step))); h = duration / steps
        for i in range(steps):
            with torch.no_grad(): z0 = z0 + h * self._flow(z0, g, start_time + i * h); zu = zu + h * self._flow(zu, g, start_time + i * h)
            zu, residual = self._project_gene(zu, rows, gene_index, target_value)
        baseline = self.decode(z0.cpu().numpy()); perturbed = self.decode(zu.cpu().numpy()); perturbed[rows, gene_index] = target_value; score, label = self._uncertainty(zu.cpu().numpy()); return V17InterventionResult(baseline, perturbed, (perturbed - baseline).astype(np.float32), score, label, residual)
