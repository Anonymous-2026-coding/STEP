from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


@dataclass
class StageData:
    expression: np.ndarray
    coordinates: np.ndarray
    time: float
    labels: Optional[np.ndarray] = None
    name: str = ""
    sample_ids: Optional[np.ndarray] = None

    def validate(self, n_genes: int) -> None:
        if self.expression.ndim != 2 or self.expression.shape[1] != n_genes:
            raise ValueError("expression must have shape (spots, genes)")
        if self.coordinates.shape != (self.expression.shape[0], 2):
            raise ValueError("coordinates must have shape (spots, 2)")
        if not np.isfinite(self.expression).all() or (self.expression < 0).any():
            raise ValueError("expression must be finite and non-negative")
        if self.labels is not None and len(self.labels) != len(self.expression):
            raise ValueError("labels must have one entry per spot")
        if self.sample_ids is not None and len(self.sample_ids) != len(self.expression):
            raise ValueError("sample_ids must have one entry per spot")


class SpatioTemporalDataset:
    def __init__(self, stages: Sequence[StageData], gene_names: Sequence[str]):
        self.stages = sorted(stages, key=lambda stage: stage.time)
        self.gene_names = tuple(str(gene) for gene in gene_names)
        if len(self.stages) < 2:
            raise ValueError("at least two stages are required")
        if len(set(self.gene_names)) != len(self.gene_names):
            raise ValueError("gene names must be unique")
        for stage in self.stages:
            stage.validate(len(self.gene_names))
        times = np.asarray(self.times)
        if np.any(np.diff(times) <= 0):
            raise ValueError("stage times must be strictly increasing")

    @property
    def times(self) -> list[float]:
        return [stage.time for stage in self.stages]

    def gene_index(self, gene: str) -> int:
        try:
            return self.gene_names.index(gene)
        except ValueError as exc:
            raise KeyError(f"gene {gene!r} is not in the dataset") from exc

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {"gene_names": list(self.gene_names), "stages": []}
        for index, stage in enumerate(self.stages):
            filename = f"stage_{index:02d}.npz"
            payload = {
                "expression": stage.expression.astype(np.float32),
                "coordinates": stage.coordinates.astype(np.float32),
            }
            if stage.labels is not None:
                payload["labels"] = np.asarray(stage.labels).astype(str)
            if stage.sample_ids is not None:
                payload["sample_ids"] = np.asarray(stage.sample_ids).astype(str)
            np.savez_compressed(directory / filename, **payload)
            metadata["stages"].append(
                {"file": filename, "time": stage.time, "name": stage.name}
            )
        (directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path) -> "SpatioTemporalDataset":
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        stages = []
        for item in metadata["stages"]:
            with np.load(directory / item["file"]) as data:
                labels = data["labels"] if "labels" in data else None
                sample_ids = data["sample_ids"] if "sample_ids" in data else None
                stages.append(
                    StageData(
                        expression=data["expression"],
                        coordinates=data["coordinates"],
                        labels=labels,
                        time=float(item["time"]),
                        name=item.get("name", ""),
                        sample_ids=sample_ids,
                    )
                )
        return cls(stages, metadata["gene_names"])


def build_knn_adjacency(
    coordinates: np.ndarray | torch.Tensor,
    n_neighbors: int = 8,
    self_loops: bool = False,
    device: Optional[torch.device] = None,
    groups: Optional[Sequence[str] | np.ndarray] = None,
) -> torch.Tensor:
    """Return a dense row-normalized spatial adjacency matrix.

    When ``groups`` is supplied, neighbors are constructed independently for
    each tissue section. This prevents artificial edges between biological
    replicates whose coordinates happen to overlap.
    """
    coords = (
        coordinates.detach().cpu().numpy()
        if isinstance(coordinates, torch.Tensor)
        else np.asarray(coordinates)
    )
    if len(coords) < 2:
        return torch.ones((len(coords), len(coords)), dtype=torch.float32, device=device)
    adjacency = np.zeros((len(coords), len(coords)), dtype=np.float32)
    group_values = (
        np.zeros(len(coords), dtype=np.int8) if groups is None else np.asarray(groups)
    )
    if len(group_values) != len(coords):
        raise ValueError("groups must have one entry per coordinate")
    for group in np.unique(group_values):
        members = np.flatnonzero(group_values == group)
        if len(members) == 1:
            adjacency[members[0], members[0]] = 1.0
            continue
        local = coords[members]
        k = min(n_neighbors + 1, len(local))
        distances, indices = NearestNeighbors(n_neighbors=k).fit(local).kneighbors(local)
        positive = distances[:, 1:][distances[:, 1:] > 0]
        scale = float(np.median(positive)) if positive.size else 1.0
        for local_row, global_row in enumerate(members):
            start = 0 if self_loops else 1
            for distance, local_column in zip(
                distances[local_row, start:], indices[local_row, start:]
            ):
                adjacency[global_row, members[local_column]] = np.exp(
                    -((distance / max(scale, 1e-8)) ** 2)
                )
    adjacency = np.maximum(adjacency, adjacency.T)
    if self_loops:
        np.fill_diagonal(adjacency, 1.0)
    row_sum = adjacency.sum(axis=1, keepdims=True)
    adjacency = adjacency / np.maximum(row_sum, 1e-8)
    return torch.as_tensor(adjacency, dtype=torch.float32, device=device)


def infer_correlation_mask(
    stages: Iterable[StageData],
    quantile: float = 0.8,
    include_diagonal: bool = True,
) -> np.ndarray:
    """Create a data-derived candidate support mask for exploratory pilots.

    This is not a causal prior. Confirmatory analyses should supply a motif or
    perturbation-supported mask instead.
    """
    expression = np.concatenate([stage.expression for stage in stages], axis=0)
    correlation = np.nan_to_num(np.corrcoef(expression, rowvar=False))
    off_diagonal = np.abs(correlation[~np.eye(correlation.shape[0], dtype=bool)])
    threshold = np.quantile(off_diagonal, quantile) if off_diagonal.size else 0.0
    mask = (np.abs(correlation) >= threshold).astype(np.float32)
    if include_diagonal:
        np.fill_diagonal(mask, 1.0)
    return mask
