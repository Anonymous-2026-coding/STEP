"""Historical v9 relational transport implementation."""
from __future__ import annotations

import numpy as np


def _standardize(x):
    x = np.asarray(x, dtype=np.float64)
    center = np.median(x, axis=0)
    scale = np.maximum(1.4826 * np.quantile(np.abs(x - center), 0.75, axis=0), 1e-6)
    return (x - center) / scale


def unbalanced_relational_sinkhorn(source_latent, target_latent, source_relation, target_relation, *, source_weights=None, target_weights=None, source_mass=1.0, target_mass=1.0, relation_weight=0.35, entropy=0.08, marginal_relaxation=1.0):
    left = _standardize(np.vstack((source_latent, target_latent))); right = _standardize(np.vstack((source_relation, target_relation))); n, m = len(source_latent), len(target_latent)
    cost = ((left[:n, None] - left[n:][None, :]) ** 2).mean(2) + float(relation_weight) * ((right[:n, None] - right[n:][None, :]) ** 2).mean(2)
    positive = cost[cost > 1e-12]; cost /= max(float(np.median(positive)) if positive.size else 1.0, 1e-8); eps = max(float(entropy), 1e-4); kernel = np.maximum(np.exp(-np.clip(cost / eps, 0., 80.)), 1e-30)
    a = np.asarray(source_weights if source_weights is not None else np.full(n, source_mass / max(n, 1)), dtype=np.float64); b = np.asarray(target_weights if target_weights is not None else np.full(m, target_mass / max(m, 1)), dtype=np.float64); a = np.maximum(a, 1e-12); b = np.maximum(b, 1e-12); a *= source_mass / a.sum(); b *= target_mass / b.sum(); tau = float(marginal_relaxation) / (float(marginal_relaxation) + eps); u = np.ones(n); v = np.ones(m)
    for _ in range(300):
        un = np.power(a / np.maximum(kernel @ v, 1e-30), tau); vn = np.power(b / np.maximum(kernel.T @ un, 1e-30), tau)
        if max(np.max(np.abs(un-u)), np.max(np.abs(vn-v))) < 1e-7: u, v = un, vn; break
        u, v = un, vn
    plan = np.nan_to_num(u[:, None] * kernel * v[None, :], nan=0., posinf=0., neginf=0.)
    return plan.astype(np.float32), {"transported_mass": float(plan.sum()), "source_mass": float(plan.sum(1).sum()), "target_mass": float(plan.sum(0).sum()), "mean_cost": float((plan * cost).sum() / max(plan.sum(), 1e-12))}
