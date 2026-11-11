"""Single-run workflows used by the command line and bundled examples."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from . import STEP, STEPConfig, StaticSTEP, StationaryConfig
from .io import load_snapshots
from .metrics import wt_metrics


def read_config(path, device, static=False, seed=None):
    values = json.loads(Path(path).read_text()) if path else {}
    values["device"] = device
    if seed is not None:
        values["seed"] = seed
    return (StationaryConfig if static else STEPConfig)(**values)


def run_wt(data_path, output, config, *, task=None, target_index=None):
    """Fit one held-out stage, save its model/predictions, and score that stage."""
    data = load_snapshots(data_path)
    with np.load(data_path, allow_pickle=False) as raw:
        if target_index is None:
            if "target" not in raw:
                raise ValueError("target_index is required for an unsplit snapshot file")
            target = int(raw["target"])
            stored_task = str(raw["task"])
            if task is not None and stored_task != task:
                raise ValueError("requested task disagrees with the prepared fold")
            task = stored_task
            train = raw["train"].astype(int).tolist()
            source = int(raw["source"])
        else:
            target = int(target_index)
            if task not in {"interpolation", "extrapolation"}:
                raise ValueError("task must be interpolation or extrapolation")
            train = [i for i in range(len(data.times)) if i != target] if task == "interpolation" else list(range(target))
            source = target - 1
    if task not in {"interpolation", "extrapolation"}:
        raise ValueError("task must be interpolation or extrapolation")
    if not 0 <= target < len(data.times) or len(train) < 2 or source not in train or target in train:
        raise ValueError("split must contain >=2 WT training stages and exclude the target")
    if any(i < 0 or i >= len(data.times) for i in train) or train != sorted(set(train)):
        raise ValueError("training stage indices must be unique, valid and increasing")
    if task == "extrapolation" and max(train) >= target:
        raise ValueError("forecasting may only use preceding snapshots")
    if task == "interpolation" and not min(train) < target < max(train):
        raise ValueError("interpolation requires training snapshots on both sides")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = STEP(config).fit_wt([data.expression[i] for i in train], [data.coordinates[i] for i in train],
        data.times[train], section_masses=data.masses[train], genes=data.genes,
        adjacencies=None if data.adjacencies is None else [data.adjacencies[i] for i in train],
        time_bounds=(float(data.times[0]), float(data.times[-1])))
    prediction = model.predict_next(float(data.times[source]), float(data.times[target]))
    scores = wt_metrics(prediction, data.expression[target], data.expression[source])
    record = dict(task=task, seed=config.seed, source=data.names[source], target=data.names[target],
                  training_stages=[data.names[i] for i in train], **scores)
    model.save(output / "model.pt")
    np.savez_compressed(output / "predictions.npz", prediction=prediction, truth=data.expression[target],
                        source=data.expression[source], genes=np.asarray(data.genes, dtype="U"))
    (output / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    (output / "fit.json").write_text(json.dumps(asdict(model.fit_result), indent=2) + "\n")
    print(json.dumps(record), flush=True)
    return record


def fit_static(data_path, output, config):
    with np.load(data_path, allow_pickle=False) as data:
        model = StaticSTEP(config).fit(data["expression"], data["coordinates"], genes=data["genes"].astype(str),
                                      adjacency=data["adjacency"] if "adjacency" in data else None)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "model.pt")
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    (output / "fit.json").write_text(json.dumps(asdict(model.fit_result), indent=2) + "\n")
    return model
