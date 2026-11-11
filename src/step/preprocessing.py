"""Fold-local preprocessing preserved from the submitted benchmark helpers."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from scipy import sparse
from ._research.data import StageData
def systematic_indices(weights, count):
    probability = np.asarray(weights, dtype=float).copy()
    probability /= probability.sum()
    cumulative = np.cumsum(probability)
    positions = (np.arange(count) + 0.5) / count
    return np.minimum(np.searchsorted(cumulative, positions), len(weights) - 1)


def normalize_coordinates(coordinates):
    values = np.asarray(coordinates, dtype=np.float32)
    center = np.median(values, axis=0)
    scale = np.median(np.linalg.norm(values - center, axis=1))
    scale = max(float(scale), 1e-8)
    return (values - center) / scale, {
        "center": center.tolist(),
        "scale": scale,
        "rule": "median-centered coordinates divided by median radial distance",
    }


def batch_aware_hvg(sections, n_top_genes):
    section_scores = []
    support = []
    for section in sections:
        values = np.asarray(section, dtype=np.float64)
        mean = values.mean(axis=0)
        variance = np.maximum((values * values).mean(axis=0) - mean * mean, 0.0)
        detection = (values > 0).mean(axis=0)
        eligible = (detection >= 0.01) & (mean >= 1e-3)
        dispersion = variance / np.maximum(mean, 1e-6)
        finite = eligible & np.isfinite(dispersion)
        rank = np.zeros(values.shape[1], dtype=np.float64)
        if finite.any():
            order = np.argsort(dispersion[finite], kind="mergesort")
            finite_indices = np.flatnonzero(finite)
            rank[finite_indices[order]] = np.linspace(0.0, 1.0, len(order), endpoint=True)
        section_scores.append(rank)
        support.append(eligible.astype(np.float64))
    score = np.median(np.vstack(section_scores), axis=0) * np.mean(np.vstack(support), axis=0)
    order = np.argsort(score, kind="mergesort")[::-1]
    selected = order[: min(n_top_genes, len(order))]
    return selected.astype(int), {
        "sections": len(sections),
        "genes": int(len(score)),
        "selected": int(len(selected)),
        "minimum_section_detection": 0.01,
        "ranking": "median within-section dispersion percentile times cross-section support",
        "caller_must_supply_training_sections_only": True,
    }


def normalize_counts(counts, target_sum=10000.0):
    matrix = sparse.csr_matrix(counts, dtype=np.float32)
    library = np.asarray(matrix.sum(axis=1)).ravel().clip(min=1.0)
    matrix = matrix.multiply((target_sum / library)[:, None]).tocsr()
    matrix.data = np.log1p(matrix.data)
    return matrix


def load_compact(directory):
    audit = json.loads((directory / "preprocessing.json").read_text(encoding="utf-8"))
    stage_metadata = list(audit["stages"])
    if stage_metadata and "time" not in stage_metadata[0]:
        sequence = {name: float(index) for index, name in enumerate(audit.get("natural_sequence", []))}
        stage_metadata = [
            dict(item, time=sequence[item["name"]])
            for item in stage_metadata
            if item["name"] in sequence
        ]
    raw_records = []
    for index, item in enumerate(sorted(stage_metadata, key=lambda value: value["time"])):
        count_path = directory / f"stage_{index:02d}_counts.npz"
        compact_path = directory / f"stage_{index:02d}.npz"
        if count_path.exists():
            counts = sparse.load_npz(count_path)
        else:
            payload = np.load(compact_path, allow_pickle=True)
            counts = sparse.csr_matrix(payload["expression"].astype(np.float32))
        gene_path = directory / f"stage_{index:02d}_genes.npy"
        if not gene_path.exists() and index == 0:
            gene_path = directory / "gene_names.npy"
        if gene_path.exists():
            local_genes = np.load(gene_path, allow_pickle=True).astype(str)
        else:
            metadata_json = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            local_genes = np.asarray(metadata_json["gene_names"], dtype=str)
        metadata_path = directory / f"stage_{index:02d}_metadata.npz"
        if metadata_path.exists():
            metadata = np.load(metadata_path, allow_pickle=True)
            coordinates, coordinate_audit = normalize_coordinates(metadata["coordinates"])
            labels = metadata["labels"].astype(str)
            weights = metadata["design_weights"].astype(float)
        else:
            payload = np.load(compact_path, allow_pickle=True)
            coordinates, coordinate_audit = normalize_coordinates(payload["coordinates"])
            labels = payload["labels"].astype(str) if "labels" in payload.files else np.full(len(coordinates), "unknown")
            weights = np.ones(len(coordinates), dtype=float)
        raw_records.append(
            {
                "time": float(item["time"]),
                "name": item["name"],
                "counts": counts,
                "genes": local_genes,
                "coordinates": coordinates,
                "labels": labels,
                "weights": weights,
                "coordinate_audit": coordinate_audit,
            }
        )
    common = set(raw_records[0]["genes"])
    for record in raw_records[1:]:
        common.intersection_update(record["genes"])
    genes = np.asarray(sorted(common), dtype="U")
    if len(genes) < 1000:
        raise RuntimeError(f"Only {len(genes)} genes are shared across sections")
    records = []
    for record in raw_records:
        lookup = {gene: index for index, gene in enumerate(record.pop("genes"))}
        columns = np.asarray([lookup[gene] for gene in genes], dtype=int)
        record["normalized"] = normalize_counts(record.pop("counts")[:, columns])
        records.append(record)
    audit["gene_harmonization"] = {
        "rule": "gene-ID intersection across section assay schemas; no expression values used",
        "shared_genes": int(len(genes)),
    }
    return records, genes, audit


def fold_stages(records, training_indices, hvg_count, population_size):
    hvg_sections = []
    for index in training_indices:
        record = records[index]
        take = systematic_indices(record["weights"], population_size)
        hvg_sections.append(record["normalized"][take].toarray())
    selected, hvg_audit = batch_aware_hvg(hvg_sections, hvg_count)
    training_selected = np.vstack(
        [records[index]["normalized"][:, selected].toarray() for index in training_indices]
    )
    scale = np.quantile(training_selected, 0.99, axis=0).clip(min=1e-3)
    stages, weights = [], []
    for record in records:
        expression = np.clip(
            2.0 * record["normalized"][:, selected].toarray() / scale, 0.0, 2.0
        ).astype(np.float32)
        stages.append(
            StageData(
                expression=expression,
                coordinates=record["coordinates"],
                time=record["time"],
                labels=record["labels"],
                name=record["name"],
            )
        )
        weights.append(record["weights"])
    return stages, weights, selected, {
        **hvg_audit,
        "selection_stages": [records[index]["name"] for index in training_indices],
        "feature_scaling": "training-only gene-wise q99 mapped to 2 and clipped to [0,2]",
    }


def population_stage(stage, weights, count):
    take = systematic_indices(weights, count)
    return StageData(
        expression=stage.expression[take],
        coordinates=stage.coordinates[take],
        time=stage.time,
        labels=None if stage.labels is None else stage.labels[take],
        name=stage.name,
    )
