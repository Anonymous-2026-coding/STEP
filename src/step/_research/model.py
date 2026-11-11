from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .data import StageData
from .representation import EditableRepresentation
from .transport import unbalanced_relational_sinkhorn


@dataclass(frozen=True)
class V17Config:
    latent_dim: int = 24
    hidden_dim: int = 96
    representation_epochs: int = 160
    flow_epochs: int = 220
    batch_size: int = 256
    learning_rate: float = 2e-3
    mask_rate: float = 0.12
    mask_loss_weight: float = 3.0
    context_loss_weight: float = 0.5
    latent_consistency_weight: float = 0.1
    relation_weight: float = 0.35
    entropy: float = 0.08
    marginal_relaxation: float = 1.0
    pairs_per_source: int = 3
    neighbors: int = 8
    growth_weight: float = 0.2
    endpoint_weight: float = 0.4
    moment_weight: float = 0.1
    validation_fraction: float = 0.2
    use_transport_weights: bool = True
    integration_step: float = 0.05
    projection_steps: int = 30
    projection_rate: float = 0.2
    projection_proximity: float = 0.05
    projection_target_weight: float = 1.0
    identity_representation: bool | None = None
    seed: int = 6120
    device: str = "auto"


@dataclass(frozen=True)
class V17FitResult:
    representation_loss: float
    best_validation_loss: float
    best_epoch: int
    latent_dimension: int
    training_pairs: int
    validation_pairs: int
    transported_mass_range: tuple[float, float]


@dataclass(frozen=True)
class WeightedLatentForecast:
    expression: np.ndarray
    weights: np.ndarray
    total_mass: float


class _MassWeightedFlow(nn.Module):
    def __init__(self, latent: int, hidden: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(2 * latent + 3, hidden), nn.LayerNorm(hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.velocity = nn.Linear(hidden, latent)
        self.growth = nn.Linear(hidden, 1)

    def forward(self, z, niche, time):
        if not isinstance(time, torch.Tensor):
            time = torch.as_tensor(time, dtype=z.dtype, device=z.device)
        if time.ndim == 0:
            time = time.expand(len(z), 1)
        elif time.ndim == 1:
            time = time[:, None]
        code = torch.cat((time, torch.sin(math.pi * time), torch.cos(math.pi * time)), 1)
        hidden = self.backbone(torch.cat((z, niche, code), 1))
        return self.velocity(hidden), self.growth(hidden).squeeze(1)


class CADENCEV17(EditableRepresentation):
    """Self-contained mass-weighted, endpoint-consistent latent flow."""

    def __init__(self, genes: int, time_min: float, time_max: float, config: V17Config = V17Config()):
        self.v17_config = config
        super().__init__(genes, time_min, time_max, config)

    def fit(self, stages: Sequence[StageData], adjacencies=None, section_masses=None) -> V17FitResult:
        # Parent fits only the self-supervised representation; its one-epoch
        # provisional flow is immediately replaced below.
        representation = self.fit_representation(stages, adjacencies, section_masses)
        config = self.v17_config
        graphs = self.graphs_
        latent = [self.encode(stage.expression, graph) for stage, graph in zip(stages, graphs)]
        masses = np.ones(len(stages), dtype=float) if section_masses is None else np.asarray(section_masses, dtype=float)
        masses = masses / np.exp(np.mean(np.log(np.maximum(masses, 1e-8))))
        rng = np.random.default_rng(config.seed)
        arrays = {key: [] for key in ("a", "b", "na", "nb", "time", "dt", "weight", "growth", "transition", "source_row")}
        transported = []
        for index, (left, right) in enumerate(zip(latent[:-1], latent[1:])):
            relation_left = self._context(left, graphs[index])
            relation_right = self._context(right, graphs[index + 1])
            source_mass = np.full(len(left), masses[index] / len(left))
            target_mass = np.full(len(right), masses[index + 1] / len(right))
            plan, diagnostic = unbalanced_relational_sinkhorn(
                left, right, relation_left, relation_right,
                source_weights=source_mass, target_weights=target_mass,
                source_mass=masses[index], target_mass=masses[index + 1],
                relation_weight=config.relation_weight, entropy=config.entropy,
                marginal_relaxation=config.marginal_relaxation,
            )
            transported.append(float(diagnostic["transported_mass"]))
            row_mass = plan.sum(1)
            conditional = plan / np.maximum(row_mass[:, None], 1e-12)
            conditional /= np.maximum(conditional.sum(1, keepdims=True), 1e-12)
            duration = float(stages[index + 1].time - stages[index].time)
            # KL-relaxed transported mass is not itself the observed target
            # population size. Preserve its relative source-cell allocation,
            # but calibrate the total to the next section's design mass before
            # constructing the log-growth target.
            growth_mass = row_mass * (masses[index + 1] / max(row_mass.sum(), 1e-12))
            growth = np.log(np.maximum(growth_mass, 1e-8) / np.maximum(source_mass, 1e-8)) / duration
            sample_weight = (
                row_mass / max(row_mass.mean(), 1e-12)
                if config.use_transport_weights else np.ones_like(row_mass)
            )
            for _ in range(config.pairs_per_source):
                chosen = np.asarray([rng.choice(len(right), p=value) for value in conditional])
                arrays["a"].append(left); arrays["b"].append(right[chosen])
                arrays["na"].append(relation_left); arrays["nb"].append(relation_right[chosen])
                arrays["time"].append(np.full(len(left), stages[index].time))
                arrays["dt"].append(np.full(len(left), duration))
                arrays["weight"].append(sample_weight); arrays["growth"].append(growth)
                arrays["transition"].append(np.full(len(left), index)); arrays["source_row"].append(np.arange(len(left)))
        matrix = {
            key: np.vstack(value).astype(np.float32) if key in {"a", "b", "na", "nb"} else np.concatenate(value).astype(np.float32)
            for key, value in arrays.items()
        }
        # Hold out the same deterministic source rows in every sampled copy,
        # stratified within each temporal transition.
        validation = np.zeros(len(matrix["a"]), dtype=bool)
        for transition in np.unique(matrix["transition"]):
            local_rows = np.unique(matrix["source_row"][matrix["transition"] == transition])
            rng_local = np.random.default_rng(config.seed + int(transition) + 101)
            selected = rng_local.choice(local_rows, max(1, int(round(len(local_rows) * config.validation_fraction))), replace=False)
            validation |= (matrix["transition"] == transition) & np.isin(matrix["source_row"], selected)
        train_index = np.flatnonzero(~validation); validation_index = np.flatnonzero(validation)
        tensors = {key: torch.as_tensor(value, device=self.device) for key, value in matrix.items() if key not in {"transition", "source_row"}}
        self.flow_ = _MassWeightedFlow(representation.latent_dimension, config.hidden_dim).to(self.device)
        optimizer = torch.optim.AdamW(self.flow_.parameters(), lr=config.learning_rate, weight_decay=1e-4)
        generator = torch.Generator(device="cpu").manual_seed(config.seed)

        def loss_for(index, train):
            aa, bb, n0, n1, tt, dt, weight, growth = [tensors[key][index] for key in ("a", "b", "na", "nb", "time", "dt", "weight", "growth")]
            if train:
                r = torch.rand((len(index), 1), device=self.device)
            else:
                r = torch.full((len(index), 1), 0.5, device=self.device)
            state = (1-r)*aa+r*bb; niche=(1-r)*n0+r*n1
            velocity_truth=(bb-aa)/dt[:,None]
            velocity, growth_prediction=self.flow_(state,niche,self._time(tt+r.ravel()*dt))
            normalized_weight=weight/weight.mean().clamp_min(1e-8)
            velocity_loss=((velocity-velocity_truth).square().mean(1)*normalized_weight).mean()
            growth_loss=(torch.nn.functional.smooth_l1_loss(growth_prediction,growth,reduction="none")*normalized_weight).mean()
            velocity0,_=self.flow_(aa,n0,self._time(tt)); endpoint=aa+dt[:,None]*velocity0
            endpoint_loss=((endpoint-bb).square().mean(1)*normalized_weight).mean()
            mean_loss=(endpoint.mean(0)-bb.mean(0)).square().mean()
            variance_loss=(endpoint.var(0,unbiased=False)-bb.var(0,unbiased=False)).square().mean()
            return velocity_loss+config.growth_weight*growth_loss+config.endpoint_weight*endpoint_loss+config.moment_weight*(mean_loss+variance_loss)

        history=[]; best=float("inf"); best_epoch=-1; best_state=None
        for epoch in range(config.flow_epochs):
            order=train_index[torch.randperm(len(train_index),generator=generator).numpy()]
            self.flow_.train(); total=0.; count=0
            for start in range(0,len(order),config.batch_size):
                index=torch.as_tensor(order[start:start+config.batch_size],device=self.device)
                loss=loss_for(index,True);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.flow_.parameters(),5.);optimizer.step();total+=float(loss.detach());count+=1
            self.flow_.eval()
            with torch.no_grad(): validation_loss=float(loss_for(torch.as_tensor(validation_index,device=self.device),False).cpu())
            history.append({"epoch":epoch,"train_loss":total/max(count,1),"validation_loss":validation_loss})
            if validation_loss<best:
                best=validation_loss;best_epoch=epoch;best_state=copy.deepcopy(self.flow_.state_dict())
        self.flow_.load_state_dict(best_state);self.flow_.eval();self.v17_history_=history
        self.fitted_=True
        return V17FitResult(
            representation_loss=representation.representation_loss,
            best_validation_loss=best, best_epoch=best_epoch,
            latent_dimension=representation.latent_dimension,
            training_pairs=len(train_index), validation_pairs=len(validation_index),
            transported_mass_range=(min(transported),max(transported)),
        )

    def _flow(self, z, graph, time):
        velocity, _ = self.flow_(z, self._niche(z, graph), self._time(time))
        return velocity

    def rollout_weighted(self, expression, graph, start_time, end_time, initial_mass=1.0):
        self._check(); z=torch.as_tensor(self.encode(expression,graph),device=self.device);g=torch.as_tensor(graph,dtype=torch.float32,device=self.device);log_weight=torch.full((len(z),),math.log(max(initial_mass,1e-8)/len(z)),device=self.device);duration=float(end_time-start_time);steps=max(1,int(math.ceil(abs(duration)/self.v17_config.integration_step)));step=duration/steps
        with torch.no_grad():
            for index in range(steps):
                time=start_time+(index+.5)*step;niche=self._niche(z,g);velocity,growth=self.flow_(z,niche,self._time(time));middle=z+.5*step*velocity;middle_velocity,middle_growth=self.flow_(middle,self._niche(middle,g),self._time(time));z=z+step*middle_velocity;log_weight=log_weight+step*middle_growth
        weights=torch.exp(log_weight.clamp(-30,30)).cpu().numpy();return WeightedLatentForecast(self.decode(z.cpu().numpy()),weights,float(weights.sum()))

    def simulate_wt(self, expression, graph, start_time, end_time):
        return self.rollout_weighted(expression,graph,start_time,end_time).expression

    @staticmethod
    def deterministic_resample(forecast: WeightedLatentForecast, count: int):
        probability=forecast.weights/forecast.weights.sum();cumulative=np.cumsum(probability);positions=(np.arange(count)+.5)/count;index=np.minimum(np.searchsorted(cumulative,positions),len(probability)-1);return forecast.expression[index]
