"""Explicit metrics for new controlled analyses; constants have undefined correlation."""
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score


def effect_metrics(predicted, observed, exact_top_decile=True):
    p, y = np.asarray(predicted, float), np.asarray(observed, float)
    finite = np.isfinite(p) & np.isfinite(y)
    p, y = p[finite], y[finite]
    if len(y) < 2:
        raise ValueError("at least two finite scoring genes required")
    k = max(1, int(np.ceil(.1 * len(y))))
    labels = np.zeros(len(y), bool)
    if exact_top_decile:
        labels[np.argsort(-np.abs(y), kind="stable")[:k]] = True
    else:
        labels = np.abs(y) >= np.quantile(np.abs(y), .9)
    nonconstant = np.ptp(p) > 1e-12 and np.ptp(y) > 1e-12
    top = np.argsort(-np.abs(p), kind="stable")[:min(20, len(y))]
    return {"pearson": float(pearsonr(p, y).statistic) if nonconstant else float("nan"),
            "spearman": float(spearmanr(p, y).statistic) if nonconstant else float("nan"),
            "auprc": float(average_precision_score(labels, np.abs(p))),
            "precision20": float(labels[top].mean()), "prevalence": float(labels.mean()),
            "rmse": float(np.sqrt(np.mean((p-y)**2))),
            "norm_ratio": float(np.linalg.norm(p) / max(np.linalg.norm(y), 1e-12)),
            "n_genes": int(len(y))}


def sliced_wasserstein(x, y, seed=6120, directions=128):
    """Mean one-dimensional W2 on fixed Gaussian projections and quantile grid."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(x.shape[1], directions))
    direction /= np.linalg.norm(direction, axis=0, keepdims=True)
    q = (np.arange(max(len(x), len(y))) + .5) / max(len(x), len(y))
    a = np.quantile(np.asarray(x) @ direction, q, axis=0)
    b = np.quantile(np.asarray(y) @ direction, q, axis=0)
    return float(np.mean(np.sqrt(np.mean((a-b)**2, axis=0))))


def wt_metrics(prediction, observed, source):
    out = effect_metrics(np.mean(prediction, 0)-np.mean(source, 0),
                         np.mean(observed, 0)-np.mean(source, 0), False)
    out["swd"] = sliced_wasserstein(prediction, observed)
    return out
