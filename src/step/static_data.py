"""Submitted static benchmark readers and query construction."""
from __future__ import annotations
from pathlib import Path
import json, h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors
def read_h5(path: Path):
    with h5py.File(path, "r") as f:
        g = f["matrix"]
        shape = tuple(g["shape"][:])
        mat = sparse.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=shape)
        genes = np.asarray(g["features"]["name"][:]).astype(str)
        barcodes = np.asarray(g["barcodes"][:]).astype(str)
    return mat.T.tocsr(), genes, barcodes


def load_section(root: Path, sample: str):
    matrix, genes, barcodes = read_h5(root / f"{sample}_filtered_feature_bc_matrix.h5")
    ann = pd.read_csv(root / f"{sample}_spot_annotation.csv.gz")
    pos = pd.read_csv(root / f"{sample}_tissue_positions_list.csv.gz", header=None)
    pos.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
    ann = ann.set_index("barcode").reindex(barcodes)
    pos = pos.set_index("barcode").reindex(barcodes)
    keep = pos["in_tissue"].fillna(0).to_numpy(dtype=int) == 1
    keep &= np.isfinite(ann["nFeature_Spatial"].to_numpy(dtype=float))
    keep &= ann["nFeature_Spatial"].to_numpy(dtype=float) >= 100
    matrix = matrix[keep]
    labels = ann.loc[keep, "phenotypes"].fillna("NA").astype(str).to_numpy()
    coords = pos.loc[keep, ["pxl_col", "pxl_row"]].to_numpy(dtype=np.float32)
    # CPM-like normalization followed by log1p; the gene schema is intersected
    # across the four sections before HVG selection.
    totals = np.asarray(matrix.sum(1)).ravel()
    matrix = sparse.diags(1e4 / np.maximum(totals, 1.0)) @ matrix
    matrix = matrix.log1p().astype(np.float32)
    return {"expression": matrix, "genes": genes, "labels": labels, "coordinates": coords, "sample": sample}


def kp_control_mask(labels):
    """Author-annotated unperturbed KP lesions used as the WT reference."""
    return np.char.startswith(np.asarray(labels).astype(str), "KP_")
def spatial_graph(
    coordinates: np.ndarray, neighbors: int = 8, *, sparse_output: bool = False
) -> np.ndarray | sparse.csr_matrix:
    """Row-normalized distance graph built without expression outcomes."""
    xy = np.asarray(coordinates, dtype=np.float32)
    if len(xy) < 2:
        return np.ones((len(xy), len(xy)), dtype=np.float32)
    count = min(int(neighbors) + 1, len(xy))
    distance, index = NearestNeighbors(n_neighbors=count).fit(xy).kneighbors(xy)
    positive = distance[:, 1:]
    scale = max(float(np.median(positive)), 1e-6)
    weights = np.exp(-0.5 * (positive / scale) ** 2).astype(np.float32)
    if sparse_output:
        rows = np.repeat(np.arange(len(xy)), positive.shape[1])
        matrix = sparse.csr_matrix(
            (weights.ravel(), (rows, index[:, 1:].ravel())),
            shape=(len(xy), len(xy)),
            dtype=np.float32,
        )
        matrix = matrix.maximum(matrix.T).tocsr()
        row_sum = np.asarray(matrix.sum(1)).ravel().clip(min=1e-8)
        return sparse.diags(1.0 / row_sum) @ matrix
    matrix = np.zeros((len(xy), len(xy)), dtype=np.float32)
    matrix[np.repeat(np.arange(len(xy)), positive.shape[1]), index[:, 1:].ravel()] = weights.ravel()
    matrix = np.maximum(matrix, matrix.T)
    return matrix / np.maximum(matrix.sum(1, keepdims=True), 1e-8)


def spatial_matches(coordinates, target_rows, control_rows) -> np.ndarray:
    """One-to-one target/control matching using pre-outcome coordinates only."""
    if not len(control_rows):
        raise ValueError("control_rows must not be empty")
    if len(target_rows) > len(control_rows):
        raise ValueError("one-to-one matching requires at least as many controls as targets")
    cost = np.linalg.norm(
        coordinates[np.asarray(target_rows), None]
        - coordinates[np.asarray(control_rows)][None, :],
        axis=2,
    )
    target_index, control_index = linear_sum_assignment(cost)
    if not np.array_equal(target_index, np.arange(len(target_rows))):
        raise RuntimeError("Hungarian matching did not cover every target")
    return np.asarray(control_rows)[control_index]


def load_author_mixscape_logfc(
    path: str | Path,
    genes: tuple[str, ...],
    targets: list[str],
) -> tuple[dict[str, np.ndarray], dict]:
    """Read the authors' frozen Mixscape-selected cluster-vs-NT endpoint.

    This function is deliberately scoring-only: callers invoke it only after
    every model and comparator prediction has been materialized.
    """

    source = Path(path)
    required = {"feature", "group", "logFC"}
    frame = pd.read_csv(source, sep="\t", usecols=lambda name: name in required)
    if set(frame.columns) != required:
        raise ValueError(f"author DE table must contain {sorted(required)}")
    frame["feature"] = frame["feature"].astype(str)
    frame["group"] = frame["group"].astype(str)
    if frame.duplicated(["group", "feature"]).any():
        raise ValueError("author DE table has duplicate group/feature rows")
    missing_groups = sorted(set(targets).difference(frame["group"].unique()))
    if missing_groups:
        raise ValueError(f"author DE table lacks target groups: {missing_groups}")

    endpoint: dict[str, np.ndarray] = {}
    overlap: dict[str, int] = {}
    wanted = pd.Index(genes, dtype=str)
    for target in targets:
        local = frame.loc[frame.group == target, ["feature", "logFC"]].set_index("feature")
        values = local["logFC"].reindex(wanted).to_numpy(dtype=np.float32)
        endpoint[target] = values
        overlap[target] = int(np.isfinite(values).sum())
    audit = {
        "source": str(source),
        "semantic_name": "author Mixscape-selected perturbation-cluster vs NT logFC",
        "construction": (
            "authors ran CalcPerturbSig and RunMixscape split by expression cluster, "
            "clustered the perturbation-score embedding, labeled clusters by dominant "
            "perturbations, and ran presto::wilcoxauc for each labeled cluster versus NT"
        ),
        "role": "primary scoring-only transcriptomic endpoint",
        "genes_available_by_target": overlap,
        "warning": (
            "this is a Mixscape-selected perturbation-cluster contrast, not a randomized "
            "target-vs-explicit-NTC do-effect"
        ),
    }
    return endpoint, audit


def clean_patch(data, control_rows, anchor, neighbors=8):
    """Anchor plus nearby controls; guide-positive expression is never used."""
    candidates = control_rows[control_rows != anchor]
    nearest = NearestNeighbors(n_neighbors=min(neighbors, len(candidates))).fit(data["coordinates"][candidates])
    selected = candidates[nearest.kneighbors(data["coordinates"][anchor][None], return_distance=False)[0]]
    rows = np.concatenate(([anchor], selected))
    return data["expression"][rows], data["coordinates"][rows]


def prediction_anchors(matches: np.ndarray, maximum: int) -> np.ndarray:
    """Deterministic query anchors shared by IMPACT and comparator methods."""

    anchors = np.unique(np.asarray(matches, dtype=int))
    if len(anchors) > maximum:
        anchors = anchors[np.linspace(0, len(anchors) - 1, maximum).astype(int)]
    return anchors


def unscale_effect(values: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Return effects in the WT-only log-normalized expression units."""
    return np.asarray(values, dtype=np.float32) * np.asarray(scale, dtype=np.float32)
