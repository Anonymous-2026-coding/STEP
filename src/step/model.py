
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


@dataclass
class PerturbationResult:
    wt: np.ndarray
    edited: np.ndarray
    response: np.ndarray
    projection_trace: list[dict]
    support_distance: float
    support_percentile: float


class STEP:
    """Small WT-only latent path model for the public demonstration."""

    def __init__(self, latent_dim: int = 8, projection_steps: int = 24,
                 projection_rate: float = 0.08, proximity_weight: float = 0.05):
        self.latent_dim = latent_dim
        self.projection_steps = projection_steps
        self.projection_rate = projection_rate
        self.proximity_weight = proximity_weight
        self.fitted = False

    @staticmethod
    def _graph_context(x: np.ndarray, xy: np.ndarray, neighbors: int = 6) -> np.ndarray:
        if len(x) <= 1:
            return np.zeros_like(x)
        idx = NearestNeighbors(n_neighbors=min(neighbors + 1, len(x))).fit(xy).kneighbors(
            return_distance=False
        )[:, 1:]
        return x[idx].mean(axis=1) - x

    def fit_wt(self, expression_by_time: Sequence[np.ndarray], coordinates_by_time: Sequence[np.ndarray],
               times: Sequence[float]) -> "STEP":
        if len(expression_by_time) < 2:
            raise ValueError("at least two WT snapshots are required")
        self.times = np.asarray(times, dtype=float)
        self.expression = [np.asarray(x, dtype=float) for x in expression_by_time]
        self.coordinates = [np.asarray(x, dtype=float) for x in coordinates_by_time]
        genes = self.expression[0].shape[1]
        values = np.vstack(self.expression)
        self.center = np.median(values, axis=0)
        self.scale = np.maximum(np.quantile(np.abs(values - self.center), 0.75, axis=0), 0.05)
        normalized = (values - self.center) / self.scale
        self.pca = PCA(n_components=min(self.latent_dim, genes, len(values) - 1), random_state=0)
        self.pca.fit(normalized)
        self.fitted = True
        self.latent = [self.encode(x, xy) for x, xy in zip(self.expression, self.coordinates)]
        # A compact conditional velocity field: fit local latent displacement
        # from nearest-neighbor cross-snapshot pairs, with time as a covariate.
        rows = []
        targets = []
        for k in range(len(self.latent) - 1):
            source, target = self.latent[k], self.latent[k + 1]
            nearest = NearestNeighbors(n_neighbors=1).fit(target).kneighbors(source, return_distance=False)[:, 0]
            dt = self.times[k + 1] - self.times[k]
            for z, j in zip(source, nearest):
                rows.append(np.r_[z, self.times[k], 1.0])
                targets.append((target[j] - z) / dt)
        self.velocity_coef = np.linalg.lstsq(np.asarray(rows), np.asarray(targets), rcond=None)[0]
        self.support_center = np.median(np.vstack(self.latent), axis=0)
        support_delta = np.vstack(self.latent) - self.support_center
        self.support_scale = np.maximum(np.std(support_delta, axis=0), 0.05)
        self.support_reference = np.sort(np.sqrt(np.sum((support_delta / self.support_scale) ** 2, axis=1)))
        return self

    def encode(self, expression: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
        self._check()
        x = np.asarray(expression, dtype=float)
        context = self._graph_context(x, np.asarray(coordinates))
        normalized = (x - self.center) / self.scale
        # Context is used as a stable local-state correction before the low-
        # dimensional representation, matching the WT-only demo objective.
        return self.pca.transform(normalized + 0.15 * context / self.scale)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        self._check()
        return np.maximum(self.pca.inverse_transform(np.asarray(latent)) * self.scale + self.center, 0.0)

    def _flow(self, latent: np.ndarray, time: float) -> np.ndarray:
        features = np.c_[latent, np.full(len(latent), time), np.ones(len(latent))]
        return features @ self.velocity_coef

    def _rollout_latent(self, latent: np.ndarray, start_time: float, end_time: float) -> np.ndarray:
        steps = max(1, int(np.ceil(abs(end_time - start_time) / 0.25)))
        dt = (end_time - start_time) / steps
        z = np.asarray(latent, dtype=float).copy()
        for step in range(steps):
            t = start_time + step * dt
            midpoint = z + 0.5 * dt * self._flow(z, t)
            z = z + dt * self._flow(midpoint, t + 0.5 * dt)
        return z

    def predict_next(self, source_time: float, target_time: float, source_index: int | None = None) -> np.ndarray:
        """Predict an adjacent WT snapshot (interpolation or one-step extrapolation)."""
        self._check()
        if source_index is None:
            source_index = int(np.argmin(np.abs(self.times - source_time)))
        z = self._rollout_latent(self.latent[source_index], source_time, target_time)
        return self.decode(z)

    def _support(self, latent: np.ndarray) -> tuple[float, float]:
        d = float(np.quantile(np.sqrt(np.sum(((latent - self.support_center) / self.support_scale) ** 2, axis=1)), 0.95))
        percentile = float((1 + np.sum(self.support_reference <= d)) / (len(self.support_reference) + 1))
        return d, percentile

    def perturb(self, expression: np.ndarray, coordinates: np.ndarray, gene: int,
                rows: np.ndarray, target_value: float, start_time: float,
                end_time: float) -> PerturbationResult:
        """Run a WT-only latent edit followed by frozen-path propagation."""
        self._check()
        x = np.asarray(expression, dtype=float)
        mask = np.asarray(rows, dtype=bool)
        if mask.shape != (len(x),):
            raise ValueError("rows must be a boolean vector with one entry per observation unit")
        edited_input = x.copy()
        edited_input[mask, gene] = target_value
        anchor = self.encode(edited_input, coordinates)
        current = anchor.copy()
        trace = []
        for step in range(self.projection_steps + 1):
            decoded = self.decode(current)
            target_loss = float(np.mean((decoded[mask, gene] - target_value) ** 2))
            proximity = float(np.mean((current - anchor) ** 2))
            trace.append({"step": step, "objective": target_loss + self.proximity_weight * proximity,
                          "target_residual": float(np.sqrt(target_loss)),
                          "anchor_distance": float(np.sqrt(proximity))})
            if step == self.projection_steps:
                break
            # The PCA decoder is linear, so a finite-difference decoder
            # gradient gives a transparent proximal projection update.
            decoded_gene = decoded[mask, gene].mean()
            direction = self.pca.components_[:, gene] * self.scale[gene]
            gradient = (decoded_gene - target_value) * direction
            current -= self.projection_rate * gradient[None, :]
            current = anchor + (current - anchor) / max(1.0, np.linalg.norm(current - anchor) / 4.0)
        wt_latent = self.encode(x, coordinates)
        wt = self.decode(self._rollout_latent(wt_latent, start_time, end_time))
        edited = self.decode(self._rollout_latent(current, start_time, end_time))
        edited[mask, gene] = target_value
        support_distance, support_percentile = self._support(current)
        return PerturbationResult(wt, edited, edited - wt, trace, support_distance, support_percentile)

    def _check(self):
        if not self.fitted:
            raise RuntimeError("call fit_wt before prediction or perturbation")
