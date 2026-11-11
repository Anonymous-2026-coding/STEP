from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.spatial.distance import cdist


def _standardize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    center = np.median(x, axis=0)
    scale = np.quantile(np.abs(x - center), 0.75, axis=0)
    scale = np.maximum(1.4826 * scale, 1e-4)
    return (x - center) / scale


def unbalanced_relational_sinkhorn(
    source_latent: np.ndarray,
    target_latent: np.ndarray,
    source_relation: np.ndarray,
    target_relation: np.ndarray,
    *,
    source_weights=None,
    target_weights=None,
    source_mass: float = 1.0,
    target_mass: float = 1.0,
    relation_weight: float = 0.35,
    entropy: float = 0.08,
    marginal_relaxation: float = 1.0,
):
    """Compute the v17 KL-relaxed relational transport plan."""
    n, m = len(source_latent), len(target_latent)
    expression_cost = cdist(np.asarray(source_latent, dtype=float), np.asarray(target_latent, dtype=float), metric="sqeuclidean")
    relation_cost = cdist(np.asarray(source_relation, dtype=float), np.asarray(target_relation, dtype=float), metric="sqeuclidean")
    cost = expression_cost / max(float(np.median(expression_cost)), 1e-8)
    cost += float(relation_weight) * relation_cost / max(float(np.median(relation_cost)), 1e-8)
    a = np.asarray(source_weights if source_weights is not None else np.full(n, source_mass / max(n, 1)), dtype=np.float64)
    b = np.asarray(target_weights if target_weights is not None else np.full(m, target_mass / max(m, 1)), dtype=np.float64)
    a = np.maximum(a, 1e-12); b = np.maximum(b, 1e-12)
    a *= float(source_mass) / a.sum(); b *= float(target_mass) / b.sum()
    exponent = float(marginal_relaxation) / (float(marginal_relaxation) + float(entropy))
    log_kernel = -cost / max(float(entropy), 1e-5)
    log_a, log_b = np.log(a.clip(min=1e-300)), np.log(b.clip(min=1e-300))
    log_u, log_v = np.zeros(n), np.zeros(m)
    for _ in range(200):
        log_u = exponent * (log_a - logsumexp(log_kernel + log_v[None, :], axis=1))
        log_v = exponent * (log_b - logsumexp(log_kernel + log_u[:, None], axis=0))
    plan = np.exp(np.clip(log_kernel + log_u[:, None] + log_v[None, :], -745, 700))
    if not np.isfinite(plan).all() or plan.sum() <= 1e-12:
        raise RuntimeError("unbalanced transport returned no finite mass")
    diagnostic = {
        "transported_mass": float(plan.sum()),
        "source_mass": float(source_mass),
        "target_mass": float(target_mass),
        "mean_cost": float((plan * cost).sum() / max(plan.sum(), 1e-12)),
    }
    return plan.astype(np.float32), diagnostic
