"""Train STEP on the bundled Limb Xenium interpolation and forecast folds."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from step.workflows import read_config, run_wt

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/limb")
    p.add_argument("--task", choices=["both", "interpolation", "extrapolation"], default="both")
    p.add_argument("--smoke", action="store_true", help="short interface check; does not reproduce reference scores")
    args = p.parse_args()
    torch.set_num_threads(2)
    config = read_config(ROOT / "configs/wt_limb.json", args.device)
    if args.smoke:
        config = replace(config, representation_epochs=2, flow_epochs=2)
    records = []
    for index in range(6):
        task = "interpolation" if index < 4 else "extrapolation"
        if args.task not in {"both", task} or (args.smoke and index not in {0, 4}):
            continue
        record = run_wt(ROOT / f"data/limb_xenium/fold_{index}.npz",
                        args.output / task / f"fold_{index}", config)
        records.append(dict(fold=index, **record))
    frame = pd.DataFrame(records)
    frame.to_csv(args.output / "fold_metrics.csv", index=False)
    keys = ["pearson", "spearman", "rmse", "auprc", "swd"]
    summary = frame.groupby("task", sort=False)[keys].mean()
    summary.to_csv(args.output / "summary.csv")
    (args.output / "run.json").write_text(json.dumps(dict(seed=config.seed, smoke=args.smoke,
        interpolation_folds=int((frame.task == "interpolation").sum()),
        extrapolation_folds=int((frame.task == "extrapolation").sum())), indent=2) + "\n")
    print("\nLimb Xenium: one seed, mean over all requested folds\n" + summary.to_string())


if __name__ == "__main__":
    main()
