"""Portable NumPy inputs; no object arrays or pickle data files are required."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from ._research.data import StageData


@dataclass
class Snapshots:
    expression: list
    coordinates: list
    times: np.ndarray
    genes: list
    names: list
    masses: np.ndarray
    adjacencies: list | None = None


def save_snapshots(path, expression, coordinates, times, genes, *, names=None, masses=None,
                   adjacencies=None):
    """Write variable-size snapshots with a shared ordered gene schema."""
    arrays = dict(times=np.asarray(times, float), genes=np.asarray(genes, dtype="U"),
                  names=np.asarray(names if names is not None else list(map(str, times)), dtype="U"),
                  masses=np.asarray(masses if masses is not None else np.ones(len(times)), float))
    if len(expression) != len(times) or len(coordinates) != len(times):
        raise ValueError("provide one expression and coordinate matrix per time")
    if adjacencies is not None and len(adjacencies) != len(times):
        raise ValueError("provide one adjacency per time")
    for i, (x, xy) in enumerate(zip(expression, coordinates)):
        arrays[f"expression_{i}"] = np.asarray(x)
        arrays[f"coordinates_{i}"] = np.asarray(xy, np.float32)
        if adjacencies is not None:
            arrays[f"adjacency_{i}"] = np.asarray(adjacencies[i], np.float32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_snapshots(path):
    """Read indexed matrices or stacked expression/coordinates benchmark arrays."""
    with np.load(path, allow_pickle=False) as data:
        times = np.asarray(data["times"], float)
        genes = data["genes"].astype(str).tolist()
        if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("times must be finite and strictly increasing")
        if not genes or len(set(genes)) != len(genes):
            raise ValueError("genes must be unique and nonempty")
        xs = [np.asarray(data["expression"][i] if "expression" in data else data[f"expression_{i}"])
              for i in range(len(times))]
        xy = [np.asarray(data["coordinates"][i] if "coordinates" in data else data[f"coordinates_{i}"], np.float32)
              for i in range(len(times))]
        graphs = ([np.asarray(data["graphs"][i], np.float32) for i in range(len(times))] if "graphs" in data else
                  [np.asarray(data[f"adjacency_{i}"], np.float32) for i in range(len(times))] if "adjacency_0" in data else None)
        for x, coords, t in zip(xs, xy, times):
            StageData(x, coords, float(t)).validate(len(genes))
            if len(x) == 0 or not np.isfinite(coords).all():
                raise ValueError("snapshots must be nonempty and coordinates must be finite")
        masses = np.asarray(data["masses"], float) if "masses" in data else np.ones(len(times))
        if masses.shape != times.shape or not np.isfinite(masses).all() or (masses <= 0).any():
            raise ValueError("masses must be positive and match times")
        names = data["names"].astype(str).tolist() if "names" in data else list(map(str, times))
        if len(names) != len(times):
            raise ValueError("provide one name per snapshot")
    return Snapshots(xs, xy, times, genes, names, masses, graphs)
