"""Train a control-only mouse lung reference and run all bundled KO queries."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from step.workflows import read_config, fit_static
from step.metrics import effect_metrics
from step.diagnostics import EditControl, edit_rollout

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/perturb_map")
    p.add_argument("--smoke", action="store_true", help="short interface check; does not reproduce reference scores")
    args = p.parse_args()
    torch.set_num_threads(2)
    config = read_config(ROOT / "configs/static_mouse.json", args.device, static=True)
    if args.smoke:
        config = replace(config, representation_epochs=2, flow_epochs=2)
    model = fit_static(ROOT / "data/perturb_map/wt.npz", args.output, config)
    with np.load(ROOT / "data/perturb_map/queries.npz", allow_pickle=False) as data:
        xs, graphs, group = data["expression"], data["adjacency"], data["group_index"]
        targets, units = data["targets"].astype(str), data["units"].astype(str)
    effects, residuals = [], []
    for i, (x, graph, gi) in enumerate(zip(xs, graphs, group)):
        rows = np.arange(len(x)) == 0
        result = model.perturb(x, None, targets[gi], rows, adjacency=graph)
        effects.append(result.effect[0])
        residuals.append(result.projection_residual)
        if i == 0:
            np.savez_compressed(args.output / "query.npz", expression=x, adjacency=graph,
                                rows=rows, genes=np.asarray(model.genes, dtype="U"))
            diagnostic = edit_rollout(model.core, x, graph, model.genes.index(targets[gi]),
                                      rows, 0.0, 1.0, EditControl("full_STEP"))
            keep = np.arange(len(model.genes)) != model.genes.index(targets[gi])
            np.testing.assert_allclose(diagnostic["response"][:, keep], result.effect[:, keep], atol=2e-6)
            pd.DataFrame(diagnostic["trace"]).to_csv(args.output / "projection_trace.csv", index=False)
    effects = np.stack(effects)
    # Measured KO responses enter only here, after training and all predictions.
    with np.load(ROOT / "data/perturb_map/evaluation.npz", allow_pickle=False) as evaluation:
        observed, scale = evaluation["observed_effect"], evaluation["scale"]
    records, predictions = [], []
    for gi, (target, unit) in enumerate(zip(targets, units)):
        predicted = effects[group == gi].mean(0) * scale[gi]
        keep = np.arange(len(model.genes)) != model.genes.index(target)
        scores = effect_metrics(predicted[keep], observed[gi, keep], exact_top_decile=False)
        predictions.append(predicted)
        records.append(dict(target=target, unit=unit, seed=config.seed, queries=int((group == gi).sum()),
                            residual=float(np.asarray(residuals)[group == gi].mean()), **scores))
    frame = pd.DataFrame(records)
    frame.to_csv(args.output / "per_unit_metrics.csv", index=False)
    keys = ["pearson", "spearman", "rmse", "auprc", "residual", "norm_ratio"]
    per_target = frame.groupby("target", sort=False)[keys].mean()
    per_target.to_csv(args.output / "per_target_metrics.csv")
    summary = dict(seed=config.seed, smoke=args.smoke, query_count=len(xs),
                   aggregation="mean units within target, then equal mean over targets",
                   **{k: float(v) for k, v in per_target.mean().items()})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(args.output / "predictions.npz", effects=effects,
                        predicted_effect=np.stack(predictions), observed_effect=observed,
                        genes=np.asarray(model.genes, dtype="U"), targets=targets, units=units)
    regulators = list(dict.fromkeys(targets))
    matrix = model.extract_grn(xs[0], None, 0.0, regulators=regulators, adjacency=graphs[0])
    pd.DataFrame(matrix, index=model.genes, columns=regulators).rename_axis("target").to_csv(args.output / "grn.csv")
    print("Mouse lung: all queries, one seed\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
