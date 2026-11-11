"""Native SPAC-seq data preparation, with no historical-package dependencies."""
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import re, json
import h5py
import numpy as np
from scipy import sparse
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors
@dataclass(frozen=True)
class LoaderConfig:
    filter_threshold: int = 1
    minimum_reference_bins: int = 10
    gene_count: int = 256


def decode(values):
    return np.asarray(values).astype("U")


def coordinate_alignment(expression_xy, guide_xy):
    multiplier = int(max(expression_xy[:, 1].max(), guide_xy[:, 1].max())) + 1
    expression_key = expression_xy[:, 0].astype(np.int64) * multiplier + expression_xy[:, 1]
    guide_key = guide_xy[:, 0].astype(np.int64) * multiplier + guide_xy[:, 1]
    order = np.argsort(guide_key); position = np.searchsorted(guide_key[order], expression_key)
    safe = np.minimum(position, len(order) - 1)
    if not np.all((position < len(order)) & (guide_key[order[safe]] == expression_key)):
        raise ValueError("not every expression coordinate has a guide-matrix coordinate")
    return order[position]


def csr_matrix(handle):
    group = handle["X"]
    if isinstance(group, h5py.Dataset): return sparse.csr_matrix(group[:])
    return sparse.csr_matrix((group["data"][:], group["indices"][:], group["indptr"][:]), shape=tuple(int(v) for v in group.attrs["shape"]))


def normalize_sparse(matrix):
    library = np.asarray(matrix.sum(1)).ravel().astype(np.float64).clip(min=1.0)
    normalized = matrix.astype(np.float32).multiply((1e4 / library)[:, None]).tocsr(); normalized.data = np.log1p(normalized.data)
    return normalized, library.astype(np.float32)


def tardis_max_filter(counts, threshold=1):
    values = np.asarray(counts)
    eligible = values > int(threshold)
    maximum = np.where(eligible, values, 0).max(1)
    return (eligible & (values == maximum[:, None])).astype(np.int8)


def load_native_tcell(directory, config):
    metadata, ranks, controls_log = [], [], []
    all_genes = None
    for path in sorted(Path(directory).glob("Day*_rep*.h5")):
        if ".guide." in path.name: continue
        with h5py.File(path) as handle, h5py.File(path.with_suffix(".guide.h5")) as guide:
            ids = decode(handle["obs/_index"][:]); xy = handle["obsm/spatial"][:].astype(np.float32)
            gene_names = decode(handle["var/_index"][:])
            if all_genes is not None and not np.array_equal(all_genes, gene_names): raise ValueError("gene columns differ")
            all_genes = gene_names
            counts = guide["X"][:][coordinate_alignment(xy, guide["obsm/spatial"][:])]
            guide_names = decode(guide["var/_index"][:]); filtered = tardis_max_filter(counts, config.filter_threshold)
            annotation = json.loads(path.with_suffix(".annotation.json").read_text(encoding="utf-8"))
            is_t = np.asarray([annotation.get(item) == "T" for item in ids]); unique = filtered.sum(1) == 1
            label = np.full(len(ids), "", dtype="U64"); label[unique] = guide_names[filtered[unique].argmax(1)]
            ntc = np.flatnonzero(is_t & (label == "sgnon-targeting")); retained = len(ntc) >= config.minimum_reference_bins
            if retained:
                raw = csr_matrix(handle); normalized, _ = normalize_sparse(raw[ntc]); mean = np.asarray(normalized.mean(0)).ravel(); variance = np.asarray(normalized.power(2).mean(0)).ravel() - mean**2
                ranks.append(rankdata(variance / np.maximum(mean, 1e-6))); controls_log.append(normalized)
        metadata.append({"path": path, "xy": xy, "ids": ids, "is_t": is_t, "label": label, "ntc": ntc, "retained": retained, "guide_positive": counts.sum(1) > 0, "guide_counts": filtered.sum(0), "guide_names": guide_names})
        print(path.stem, "native T-cell NTC", len(ntc), flush=True)
    score = np.median(ranks, axis=0); support = sum(np.asarray((item > 0).sum(0)).ravel() for item in controls_log)
    excluded = np.asarray([gene.lower().startswith(("mt-", "rpl", "rps")) for gene in all_genes]); score[(support < 3) | excluded] = -np.inf
    guide_names = metadata[0]["guide_names"]
    forced = [int(np.flatnonzero(all_genes == name.removeprefix("sg"))[0]) for name in guide_names if name != "sgnon-targeting" and np.any(all_genes == name.removeprefix("sg"))]
    selected = []
    for index in forced + np.argsort(score)[::-1].tolist():
        if index not in selected and np.isfinite(score[index]): selected.append(index)
        if len(selected) >= config.gene_count: break
    selected = np.asarray(selected); genes = tuple(all_genes[selected])
    records, audit_rows = [], []
    for item in metadata:
        path, xy, label = item["path"], item["xy"], item["label"]
        day = int(re.search(r"Day(\d+)", path.stem)[1]); audit_rows.append({"section": path.stem, "day": day, "unique_reference_bins": len(item["ntc"]), "retained": item["retained"]})
        if not item["retained"]: continue
        targets = np.flatnonzero(item["is_t"] & (label != "") & (label != "sgnon-targeting")); anchors = np.concatenate((item["ntc"], targets))
        neighbor_model = NearestNeighbors(n_neighbors=9).fit(xy); distance, neighbor = neighbor_model.kneighbors(xy[anchors]); neighbor, distance = neighbor[:, 1:], distance[:, 1:]
        union = np.unique(np.concatenate((anchors, neighbor.ravel()))); lookup = np.full(len(xy), -1, dtype=int); lookup[union] = np.arange(len(union))
        with h5py.File(path) as handle:
            raw = csr_matrix(handle)[union]; library = np.asarray(raw.sum(1)).ravel().clip(min=1); logged = np.log1p(1e4 * raw[:, selected].toarray() / library[:, None]).astype(np.float32)
        local_anchors = lookup[anchors]; physical = np.repeat(np.arange(len(union))[:, None], 8, axis=1); physical[local_anchors] = lookup[neighbor]
        physical_distance = np.zeros((len(union), 8), dtype=np.float32); physical_distance[local_anchors] = distance.astype(np.float32)
        anchor_labels = np.asarray(["control"] * len(item["ntc"]) + [value.removeprefix("sg") for value in label[targets]], dtype="U64")
        records.append({"name": path.stem, "day": day, "expression": logged, "evaluation_log_expression": logged.copy(), "coordinates": xy[union], "library": library.astype(np.float32), "anchor_indices": local_anchors, "anchor_labels": anchor_labels, "anchor_niches": np.full(len(anchors), "section", dtype="U16"), "full_guide_positive": item["guide_positive"][union], "full_physical_neighbor_indices": physical, "full_physical_neighbor_distances": physical_distance, "original_barcodes": item["ids"][union]})
    controls = np.vstack([record["expression"][record["anchor_indices"][record["anchor_labels"] == "control"]] for record in records])
    scale = np.maximum(np.quantile(controls, 0.99, axis=0), 0.25).astype(np.float32)
    for record in records:
        record["expression"] = np.clip(record["expression"] / scale, 0, 2).astype(np.float32); record["expression_scale"] = scale
    guide_counts = sum(item["guide_counts"] for item in metadata)
    audit = {"sections": audit_rows, "guide_filtered_bins_across_sections": {name: int(count) for name, count in zip(guide_names, guide_counts)}}
    return records, genes, audit


def graph(coordinates, neighbors=8, radius=None):
    count = min(neighbors + 1, len(coordinates)); distance, indices = NearestNeighbors(n_neighbors=count).fit(coordinates).kneighbors(coordinates)
    scale = max(float(np.median(distance[:, 1:])), 1e-6); matrix = np.zeros((len(coordinates), len(coordinates)), dtype=np.float32)
    for row in range(len(coordinates)): matrix[row, indices[row, 1:]] = np.exp(-0.5 * (distance[row, 1:] / scale) ** 2)
    matrix = np.maximum(matrix, matrix.T)
    if radius is not None: matrix[np.linalg.norm(coordinates[:, None] - coordinates[None], axis=2) > radius] = 0
    isolated = matrix.sum(1) == 0; matrix[isolated, isolated] = 1.0
    return matrix / np.maximum(matrix.sum(1, keepdims=True), 1e-8)


def control_positions(record): return np.flatnonzero(record["anchor_labels"] == "control")


def native_radius(record): return float(np.quantile(record["full_physical_neighbor_distances"], 0.95))


def capped_controls(record, maximum=120):
    positions = control_positions(record)
    if maximum > 0 and len(positions) > maximum:
        rows = record["anchor_indices"][positions]; order = np.lexsort((record["coordinates"][rows, 1], record["coordinates"][rows, 0])); positions = positions[order][np.linspace(0, len(positions) - 1, maximum).astype(int)]
    rows = record["anchor_indices"][positions]; mass = np.full(len(rows), len(control_positions(record)) / len(rows), dtype=np.float32)
    return record["expression"][rows], record["coordinates"][rows], mass


def metadata_matches(record, target, count=3):
    target_positions = np.flatnonzero(record["anchor_labels"] == target); controls = control_positions(record); coordinates = record["coordinates"]
    target_rows, control_rows = record["anchor_indices"][target_positions], record["anchor_indices"][controls]; result = []
    for row in target_rows:
        nearest = NearestNeighbors(n_neighbors=min(count, len(controls))).fit(coordinates[control_rows]).kneighbors(coordinates[row][None], return_distance=False)[0]
        selected = controls[nearest]
        if len(selected) < count: selected = np.pad(selected, (0, count - len(selected)), mode="edge")
        result.append(selected)
    return target_positions, np.asarray(result, dtype=int)


def observed_effect(record, target, target_positions, matches):
    expression = record["evaluation_log_expression"] / record["expression_scale"]
    target_rows = record["anchor_indices"][target_positions]; reference_rows = record["anchor_indices"][matches]
    autonomous = expression[target_rows].mean(0) - expression[reference_rows].mean(axis=(0, 1))
    return autonomous


def clean_patch(record, control_position, neighbors=8):
    anchor = int(record["anchor_indices"][control_position]); candidates = record["anchor_indices"][control_positions(record)]; candidates = candidates[candidates != anchor]
    nearest = NearestNeighbors(n_neighbors=min(neighbors, len(candidates))).fit(record["coordinates"][candidates]).kneighbors(record["coordinates"][anchor][None], return_distance=False)[0]
    patch = np.concatenate(([anchor], candidates[nearest])); return record["expression"][patch], record["coordinates"][patch]
