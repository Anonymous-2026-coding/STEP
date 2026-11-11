"""Public interfaces for WT dynamics, static interventions and GRN sensitivities."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from ._research.data import StageData, build_knn_adjacency
from ._research.model import CADENCEV17 as STEPModel, V17Config as STEPConfig
from ._research.stationary import CADENCEV16 as StationarySTEP, V16Config as StationaryConfig


def _expression(values):
    x = np.asarray(values, dtype=np.float32)
    if x.ndim != 2 or min(x.shape) < 1 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("expression must be a finite nonnegative (observations, genes) matrix")
    return x


def _graph(x, coordinates, adjacency, neighbors):
    if adjacency is None:
        xy = np.asarray(coordinates, dtype=np.float32)
        if xy.shape != (len(x), 2) or not np.isfinite(xy).all():
            raise ValueError("provide finite (observations, 2) coordinates or an adjacency")
        adjacency = build_knn_adjacency(xy, neighbors).numpy()
    graph = np.asarray(adjacency, dtype=np.float32)
    if graph.shape != (len(x), len(x)) or not np.isfinite(graph).all() or (graph < 0).any():
        raise ValueError("adjacency must be a finite nonnegative square matrix matching observations")
    return graph


class _EditableModel:
    def _check_fitted(self):
        if not hasattr(self, "core"):
            raise RuntimeError("fit a WT reference before prediction or intervention")

    def _set_genes(self, genes, count):
        names = [f"gene_{i}" for i in range(count)] if genes is None else list(map(str, genes))
        if len(names) != count or len(set(names)) != count:
            raise ValueError("genes must contain one unique name per expression column")
        self.genes = names

    def _gene_index(self, gene):
        if isinstance(gene, str):
            if gene not in self.genes:
                raise ValueError(f"unknown gene: {gene}")
            return self.genes.index(gene)
        if not isinstance(gene, (int, np.integer)) or not 0 <= int(gene) < self.core.genes:
            raise ValueError("gene must be a known name or an in-range integer index")
        return int(gene)

    def perturb(self, expression, coordinates, gene, rows, target_value=0.0,
                start_time=0.0, end_time=1.0, *, adjacency=None):
        """Edit selected rows and propagate with repeated decoder-target projection.

        Use the same genes and expression units as training. Coordinates may be
        None with an explicit adjacency. Static times are dimensionless
        propagation coordinates, not inferred biological observation times.
        """
        self._check_fitted()
        x = _expression(expression)
        if x.shape[1] != self.core.genes:
            raise ValueError("query gene columns must match training")
        mask = np.asarray(rows)
        if mask.dtype != np.bool_ or mask.shape != (len(x),) or not mask.any():
            raise ValueError("rows must be a boolean mask selecting at least one observation")
        if not np.isfinite([start_time, end_time, target_value]).all() or end_time < start_time or target_value < 0:
            raise ValueError("require finite times with end >= start and a nonnegative target value")
        graph = _graph(x, coordinates, adjacency, self.config.neighbors)
        return self.core.intervene(x, graph, self._gene_index(gene), mask,
                                   float(start_time), float(end_time), float(target_value))

    def extract_grn(self, expression, coordinates, time, *, regulators=None,
                    epsilon=0.01, adjacency=None, remove_self=True):
        """Sensitivity of a local decoded one-step map, (targets, regulators).

        Each regulator is increased by epsilon in all supplied observations;
        output changes are averaged over that spatial patch, in input units.
        """
        self._check_fitted()
        x = _expression(expression)
        if x.shape[1] != self.core.genes or not np.isfinite(time) or not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("match training genes and provide finite time and positive epsilon")
        graph = _graph(x, coordinates, adjacency, self.config.neighbors)
        indices = list(range(x.shape[1])) if regulators is None else [self._gene_index(g) for g in regulators]
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("regulators must be a nonempty list without duplicates")
        g = torch.as_tensor(graph, device=self.core.device)

        def local_map(values):
            z = torch.as_tensor(self.core.encode(values, graph), device=self.core.device)
            with torch.no_grad():
                evolved = z + self.config.integration_step * self.core._flow(z, g, float(time))
            return self.core.decode(evolved.cpu().numpy()).mean(0)

        base = local_map(x)
        matrix = np.empty((x.shape[1], len(indices)), dtype=np.float32)
        for column, regulator in enumerate(indices):
            altered = x.copy()
            altered[:, regulator] += epsilon
            matrix[:, column] = (local_map(altered) - base) / epsilon
            if remove_self:
                matrix[regulator, column] = 0.0
        return matrix

    def save(self, path):
        """Save a local PyTorch checkpoint including the WT training reference."""
        self._check_fitted()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"format": "step-neural-1", "model": self}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        """Load a trusted STEP/StaticSTEP checkpoint on the requested device."""
        payload = torch.load(path, map_location=device, weights_only=False)
        if payload.get("format") != "step-neural-1":
            raise ValueError("unrecognized STEP checkpoint")
        model = payload["model"]
        model.config = replace(model.config, device=device)
        model.core.config = model.config
        if hasattr(model.core, "v17_config"):
            model.core.v17_config = model.config
        if not hasattr(model, "genes"):
            model._set_genes(None, model.core.genes)
        return model


class STEP(_EditableModel):
    """Editable spatial representation, relational UOT and mass-weighted flow."""

    def __init__(self, config=None, **settings):
        if config is not None and settings:
            raise ValueError("supply config or keyword settings, not both")
        self.config = config or STEPConfig(**settings)

    def fit_wt(self, expression_by_time, coordinates_by_time, times,
               section_masses=None, adjacencies=None, *, genes=None, time_bounds=None):
        """Fit WT snapshots. Optional time_bounds fixes a known time axis.

        time_bounds may include scheduled evaluation times, not their expression.
        Multiple sections can use supplied block-diagonal spatial adjacencies.
        """
        if not (len(times) == len(expression_by_time) == len(coordinates_by_time)):
            raise ValueError("one expression matrix and coordinate matrix per time")
        self.times = np.asarray(times, dtype=float)
        if len(times) < 2 or not np.isfinite(self.times).all() or np.any(np.diff(self.times) <= 0):
            raise ValueError("at least two finite strictly increasing observed times required")
        xs = [_expression(x) for x in expression_by_time]
        self._set_genes(genes, xs[0].shape[1])
        self.stages = [StageData(x, np.asarray(xy, np.float32), float(t))
                       for x, xy, t in zip(xs, coordinates_by_time, times)]
        for stage in self.stages:
            stage.validate(len(self.genes))
            if not np.isfinite(stage.coordinates).all():
                raise ValueError("coordinates must be finite")
        if adjacencies is not None and len(adjacencies) != len(times):
            raise ValueError("provide one adjacency per training snapshot")
        graphs = [_graph(s.expression, s.coordinates, None if adjacencies is None else adjacencies[i],
                         self.config.neighbors) for i, s in enumerate(self.stages)]
        self.section_masses = np.ones(len(times)) if section_masses is None else np.asarray(section_masses, float)
        if self.section_masses.shape != (len(times),) or not np.isfinite(self.section_masses).all() or (self.section_masses <= 0).any():
            raise ValueError("section_masses must contain one finite positive value per snapshot")
        bounds = (float(self.times[0]), float(self.times[-1])) if time_bounds is None else tuple(map(float, time_bounds))
        if len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] > self.times[0] or bounds[1] < self.times[-1] or bounds[0] >= bounds[1]:
            raise ValueError("time_bounds must enclose all observed times and have positive width")
        self.core = STEPModel(len(self.genes), *bounds, self.config)
        self.fit_result = self.core.fit(self.stages, graphs, self.section_masses)
        return self

    def predict_next(self, source_time, target_time, source_index=None, *, resample=True):
        """Roll out a source snapshot; resample by learned population weights."""
        self._check_fitted()
        if source_index is None:
            matches = np.flatnonzero(np.isclose(self.times, source_time))
            if len(matches) != 1:
                raise ValueError("source_time must identify one observed training snapshot")
            source_index = int(matches[0])
        if not 0 <= source_index < len(self.stages):
            raise ValueError("source_index outside the observed snapshots")
        stage = self.stages[source_index]
        if not np.isclose(stage.time, source_time) or not np.isfinite(target_time) or target_time < source_time:
            raise ValueError("source_index/time must agree and target_time must be finite and >= source_time")
        masses = getattr(self, "section_masses", np.ones(len(self.stages)))
        forecast = self.core.rollout_weighted(stage.expression, self.core.graphs_[source_index],
                                              source_time, target_time, float(masses[source_index]))
        return self.core.deterministic_resample(forecast, len(stage.expression)) if resample else forecast.expression

    def interpolate(self, target_time):
        """Predict a missing intermediate stage using the preceding WT snapshot."""
        self._check_fitted()
        if not self.times[0] < target_time < self.times[-1] or np.isclose(self.times, target_time).any():
            raise ValueError("interpolation target must be unobserved and bracketed by training times")
        return self.predict_next(float(self.times[self.times < target_time][-1]), target_time)

    def forecast(self, target_time):
        """Forecast beyond the last WT training snapshot."""
        self._check_fitted()
        if not target_time > self.times[-1]:
            raise ValueError("forecast target must follow the last observed training time")
        return self.predict_next(float(self.times[-1]), target_time)


class StaticSTEP(_EditableModel):
    """Stationary STEP reference for assays without observed temporal snapshots."""

    def __init__(self, config=None, **settings):
        if config is not None and settings:
            raise ValueError("supply config or keyword settings, not both")
        self.config = config or StationaryConfig(**settings)

    def fit(self, expression, coordinates, *, adjacency=None, genes=None):
        """Fit control-only tissue with the stationary estimator's paired WT copies."""
        x = _expression(expression)
        xy = np.asarray(coordinates, dtype=np.float32)
        self._set_genes(genes, x.shape[1])
        stage = StageData(x, xy, 0.0)
        stage.validate(len(self.genes))
        if not np.isfinite(xy).all():
            raise ValueError("coordinates must be finite")
        graph = _graph(x, xy, adjacency, self.config.neighbors)
        self.core = StationarySTEP(len(self.genes), 0.0, 1.0, self.config)
        self.fit_result = self.core.fit([stage, StageData(x.copy(), xy.copy(), 1.0)], [graph, graph])
        return self
