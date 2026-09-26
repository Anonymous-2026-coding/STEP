# Demo contents

`data/wt_subset.npz` contains four unperturbed snapshots with shape
`(time=4, units=32, genes=24)` and matching 2D coordinates. The subset is derived from the controlled
WT replay and is intentionally small enough to run on a laptop.

`01_wt_dynamics.py` fits the WT path and writes an adjacent interpolation and a one-step extrapolation.
`02_in_silico_perturbation.py` edits gene index 3 in the left spatial region, performs the decoder-
consistent latent projection, and propagates the edited state through the frozen WT path.

The bundled `output/` files are example artifacts. They can be regenerated with `../scripts/run_demo.sh`.
