"""Instrument the submitted Euler edit operator without final output overwrite.

Support distances are descriptive kNN diagnostics, not probabilities of validity.
Projection schedules are in physical time, separate from solver discretization.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class EditControl:
    name: str
    initialization: str = "masked"
    initial_projection: bool = True
    repeat_projection: bool = True
    temporal_flow: bool = True
    initial_budget_multiplier: int = 1
    solver_step: float | None = None
    correction_interval: float | None = None
    proximity: float | None = None
    magnitude_matched_random: bool = False


def default_controls():
    return [EditControl("direct_clamp", "direct", False, False),
            EditControl("mask_only", initial_projection=False, repeat_projection=False),
            EditControl("initial_only", repeat_projection=False),
            EditControl("full_STEP"),
            EditControl("initial_only_equal_budget", repeat_projection=False, initial_budget_multiplier=-1),
            EditControl("edit_only", repeat_projection=False, temporal_flow=False),
            EditControl("repeated_no_flow", temporal_flow=False),
            EditControl("matched_random", repeat_projection=False, magnitude_matched_random=True),
            EditControl("prox_0.05", proximity=.05),
            EditControl("prox_0.5", proximity=.5)]


class LatentSupport:
    def __init__(self, reference, k=5):
        self.reference = np.unique(np.asarray(reference, float), axis=0)
        self.k = min(k, len(self.reference)-1)
        if self.k < 1:
            raise ValueError("at least two support reference observations required")
        self.nn = NearestNeighbors(n_neighbors=self.k+1).fit(self.reference)
        # Explicit reference query allows exclusion of each observation itself.
        d = self.nn.kneighbors(self.reference)[0][:, 1:]
        self.loo = d.mean(1)
        self.q95 = max(float(np.quantile(self.loo, .95)), 1e-12)

    def distance(self, z):
        d = self.nn.kneighbors(np.asarray(z), n_neighbors=self.k+1)[0]
        # Apply the same exclusion rule to every group. A WT point identical
        # to a reference must not receive an artificial self-neighbor advantage.
        selected = np.where((d[:, :1] < 1e-8), d[:, 1:], d[:, :self.k])
        return selected.mean(1) / self.q95


def edit_rollout(model, x, graph, gene, rows, start, end, control, target=0., random_seed=0):
    """Return paired response, latent path, and raw decoder residual per correction."""
    x = np.asarray(x, np.float32)
    rows = np.asarray(rows, bool)
    if model.identity_representation_:
        raise ValueError("diagnostic requires the neural representation")
    config = model.config
    g = torch.as_tensor(graph, dtype=torch.float32, device=model.device)
    clean = torch.as_tensor(model.encode(x, graph), device=model.device)
    edited_x = x.copy(); edited_x[rows, gene] = target
    indicator = np.zeros_like(x); indicator[rows, gene] = 1.
    anchor = torch.as_tensor(model.encode(edited_x, graph,
        None if control.initialization == "direct" else indicator), device=model.device)
    z0, z = clean.clone(), anchor.clone()
    duration = float(end-start)
    if duration < 0:
        raise ValueError("forward edits require end >= start")
    hmax = control.solver_step or config.integration_step
    interval = control.correction_interval
    if interval is None:
        count = max(1, int(math.ceil(duration/hmax)))
        times = np.linspace(start, end, count+1)
        correction = np.ones(count+1, bool)
    else:
        boundaries = np.r_[np.arange(start, end, interval), end]
        grid = [float(start)]; marks = [True]
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            count = max(1, int(math.ceil((right-left)/hmax)))
            for j in range(1, count+1):
                grid.append(float(left+(right-left)*j/count)); marks.append(j == count)
        times, correction = np.asarray(grid), np.asarray(marks)
    total_corrections = 1 + int(correction[1:].sum())
    multiplier = total_corrections if control.initial_budget_multiplier == -1 else control.initial_budget_multiplier
    trace = []; path = []; clean_path = []
    mask = torch.as_tensor(rows, device=model.device)
    prox = config.projection_proximity if control.proximity is None else control.proximity

    def residual(state):
        with torch.no_grad():
            return float((model.autoencoder_.decoder(state)[mask, gene]-target).square().mean().sqrt().cpu())

    def project(state, time, budget):
        initial = state.detach(); value = initial.clone().requires_grad_(True)
        before = residual(initial)
        for _ in range(budget):
            d = model.autoencoder_.decoder(value)
            loss = config.projection_target_weight*(d[mask,gene]-target).square().mean() + prox*(value-initial).square().mean()
            gradient = torch.autograd.grad(loss, value)[0]
            value = (value-config.projection_rate*gradient).detach().requires_grad_(True)
        value = value.detach()
        trace.append({"time":float(time), "before":before, "after":residual(value),
                      "steps":budget, "correction_rms":float((value-initial).square().mean().sqrt().cpu())})
        return value

    if control.initial_projection:
        z = project(z, start, config.projection_steps*multiplier)
    if control.magnitude_matched_random:
        # Match full initial-edit norm relative to the clean state, without labels.
        rng = np.random.default_rng(random_seed)
        random = torch.as_tensor(rng.normal(size=z.shape), dtype=z.dtype, device=model.device)
        delta = z-clean
        random = random / random.norm(dim=1,keepdim=True).clamp_min(1e-12)*delta.norm(dim=1,keepdim=True)
        z = clean+random
    path.append(z.cpu().numpy().copy()); clean_path.append(z0.cpu().numpy().copy())
    for index in range(1, len(times)):
        h = float(times[index]-times[index-1])
        if control.temporal_flow:
            with torch.no_grad():
                z0 = z0+h*model._flow(z0,g,float(times[index-1]))
                z = z+h*model._flow(z,g,float(times[index-1]))
        if control.repeat_projection and correction[index]:
            z = project(z, times[index], config.projection_steps)
        path.append(z.cpu().numpy().copy()); clean_path.append(z0.cpu().numpy().copy())
    # This is deliberately before output-level hard clamping.
    raw = model.autoencoder_.decoder(z).detach().cpu().numpy()
    baseline = model.decode(z0.cpu().numpy())
    decoded = np.maximum(raw, 0.)
    response = decoded-baseline
    return {"response":response, "raw":raw, "baseline":baseline,
            "path":np.asarray(path), "clean_path":np.asarray(clean_path),
            "times":times, "trace":trace,
            "final_raw_residual":residual(z),
            "initial_edit_rms":float(np.sqrt(np.mean((path[0]-clean.cpu().numpy())**2))),
            "final_edit_rms":float((z-z0).square().mean().sqrt().cpu()),
            "gradient_steps":sum(row["steps"] for row in trace)}
