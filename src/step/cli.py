"""Small command-line entry points for user-prepared inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from . import STEP
from .workflows import read_config, run_wt, fit_static


def main():
    parser = argparse.ArgumentParser(description="STEP spatial dynamics, interventions and GRN extraction")
    parser.add_argument("--threads", type=int, default=2)
    commands = parser.add_subparsers(dest="command", required=True)
    wt = commands.add_parser("wt", help="interpolate or forecast a held-out WT snapshot")
    wt.add_argument("--task", choices=["interpolation", "extrapolation"])
    wt.add_argument("--target-index", type=int)
    static = commands.add_parser("fit-static", help="fit a control-only stationary reference")
    for p in (wt, static):
        p.add_argument("--config", type=Path)
        p.add_argument("--seed", type=int)
    perturb = commands.add_parser("perturb", help="edit selected rows and propagate with repeated projection")
    perturb.add_argument("--gene", required=True, help="gene name in training columns")
    perturb.add_argument("--target-value", type=float, default=0.0)
    perturb.add_argument("--start", type=float, default=0.0)
    perturb.add_argument("--end", type=float, default=1.0)
    grn = commands.add_parser("grn", help="extract a spatial patch's local gene sensitivities")
    grn.add_argument("--time", type=float, default=0.0)
    grn.add_argument("--regulators", nargs="+")
    grn.add_argument("--epsilon", type=float, default=0.01)
    for p in (wt, static, perturb, grn):
        p.add_argument("--data", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--device", default="cpu")
    for p in (perturb, grn):
        p.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.command == "wt":
        run_wt(args.data, args.output, read_config(args.config, args.device, seed=args.seed),
               task=args.task, target_index=args.target_index)
    elif args.command == "fit-static":
        fit_static(args.data, args.output, read_config(args.config, args.device, static=True, seed=args.seed))
    else:
        model = STEP.load(args.model, device=args.device)
        with np.load(args.data, allow_pickle=False) as data:
            x = data["expression"]
            xy = data["coordinates"] if "coordinates" in data else None
            adjacency = data["adjacency"] if "adjacency" in data else None
            if "genes" in data and list(data["genes"].astype(str)) != model.genes:
                raise ValueError("query genes/order do not match the checkpoint")
            args.output.mkdir(parents=True, exist_ok=True)
            if args.command == "perturb":
                result = model.perturb(x, xy, args.gene, data["rows"], args.target_value,
                                       args.start, args.end, adjacency=adjacency)
                np.savez_compressed(args.output / "perturbation.npz", baseline=result.baseline,
                    perturbed=result.perturbed, effect=result.effect, genes=np.asarray(model.genes, dtype="U"))
                (args.output / "summary.json").write_text(json.dumps(dict(gene=args.gene,
                    projection_residual=result.projection_residual, uncertainty_score=result.uncertainty_score), indent=2)+"\n")
            else:
                regulators = args.regulators or model.genes
                matrix = model.extract_grn(x, xy, args.time, regulators=regulators,
                                          epsilon=args.epsilon, adjacency=adjacency)
                pd.DataFrame(matrix, index=model.genes, columns=regulators).rename_axis("target").to_csv(args.output / "grn.csv")
        print(f"Saved {args.command} outputs to {args.output}")


if __name__ == "__main__":
    main()
