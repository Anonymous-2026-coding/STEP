# STEP

**Self-Supervised Editable Transport for Counterfactual Spatiotemporal Gene Perturbations**

STEP learns from unpaired wild-type spatial transcriptomic snapshots. It combines
an editable spatial representation, relation-aware unbalanced optimal transport
and a learned temporal flow. The same representation supports WT interpolation,
forecasting, spatially localized in silico interventions with repeated
decoder-target projection, and context-specific GRN sensitivity extraction.

<img width="1546" height="656" alt="image" src="https://github.com/user-attachments/assets/88012f44-d8be-445b-864c-338379555218" />


## Installation

Use Python 3.10 for the reference environment. From this repository's root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU build:
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
# For the CUDA reference runs, use this line instead of the CPU line:
# python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install -e .
```

The tested reference environment uses Python 3.10, PyTorch 2.2.2, NumPy 1.26.4,
SciPy 1.11.4 and scikit-learn 1.5.2. Full example configurations are included;
GPU execution is recommended for training. Commands below run from the repository root.

## Datasets and accessions

| Dataset | Application | Official data / publication | Evaluated stages or samples |
| --- | --- | --- | --- |
| ARTISTA, axolotl development | WT interpolation and forecasting | [ARTISTA](https://db.cngb.org/stomics/artista/), [Development.h5ad](https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000056/stomics/Development.h5ad) | stage44, stage54, stage57, juvenile, adult |
| MOSTA, mouse organogenesis | WT interpolation and forecasting | [MOSTA](https://db.cngb.org/stomics/mosta/), [Chen et al., Cell, 2022](https://doi.org/10.1016/j.cell.2022.04.003) | E9.5–E16.5 |
| **Limb Xenium, axolotl regeneration** | WT interpolation and forecasting; bundled quick start | [GSE294212](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE294212), [García-García et al., bioRxiv, 2025](https://doi.org/10.1101/2025.03.30.645595) | uninjured; 3, 5, 7, 10, 14 days post-amputation |
| **Zebrafish heart Stereo-seq** | WT interpolation and forecasting | [Li et al., Nature Communications, 2025](https://doi.org/10.1038/s41467-025-59070-0); non-imputed Stereo-seq regeneration data linked in the publication | uninjured; 6/12 hours and 1/3/7/14/28 days post-amputation |
| Perturb-DBiT, HT29 | Static spatial perturbation | [GSE319277](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE319277), [official data/code](https://github.com/abaysoy/Perturb_DBiT) | GSM9514752 / Bru.LC.6 |
| **Perturb-map, mouse lung cancer** | Static spatial perturbation; bundled quick start | [GSE193460](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE193460), Dhainaut et al., Cell, 2022 | GSM5808054–GSM5808057 |
| SPAC-seq | Dynamic spatial perturbation | [CRA023189](https://ngdc.cncb.ac.cn/gsa/browse/CRA023189), [data portal](https://spac.pku-genomics.org), [official code](https://github.com/zenglab-pku/SPAC-seq) | T-cell Day 4 and Day 7 |
| ZESTA, zebrafish embryogenesis | Spatiotemporal perturbation / GRN analysis | [ZESTA](https://db.cngb.org/stomics/zesta/), [Liu et al., Developmental Cell, 2022](https://doi.org/10.1016/j.devcel.2022.04.009) | 3.3, 5.25, 10, 12, 18, 24 hours post-fertilization |

Only the compact, prepared **Limb Xenium** and **Perturb-map** example inputs are
bundled. Download the other assays from their original sources and prepare them
using the common input format below. Cite the original studies when using their data.

## Preparing your data

The Python API accepts NumPy arrays; observations need not correspond across
snapshots, and each snapshot may contain a different number of cells or spots.

| Input | Shape / meaning |
| --- | --- |
| Expression at each time | `(n_observations, n_genes)`, finite nonnegative normalized expression |
| Spatial coordinates | `(n_observations, 2)`, in the same row order as expression |
| Times | One strictly increasing model-time value per snapshot |
| Gene names | The same ordered, unique names for every snapshot |
| Optional adjacency | `(n_observations, n_observations)`, nonnegative row-normalized spatial graph |
| Optional masses | One positive design mass per snapshot; defaults to equal masses |
| Intervention mask | Boolean `(n_observations,)`; `True` selects edited cells/spots |

Start with library-size normalization and `log1p`. Harmonize gene columns across
snapshots, then select genes and fit any expression scaling using **training WT
snapshots only**. Apply that fixed transformation to held-out snapshots and
queries. A knockout target of zero is specified in these nonnegative input units.
Coordinate scales should be comparable across snapshots. For multiple tissue
sections, construct graphs within each section and pass block-diagonal
adjacencies so unrelated sections are not connected.

For command-line workflows, save a portable snapshot file:

```python
from step.io import save_snapshots

save_snapshots(
    "my_snapshots.npz",
    expression=expression_by_time,      # list of (n_t, G) arrays
    coordinates=coordinates_by_time,    # list of (n_t, 2) arrays
    times=observed_times,
    genes=gene_names,
)
```

See [the data format guide](docs/data_format.md) for preprocessing, NPZ schemas,
static inputs and the bundled data descriptions.

## Model interfaces

### WT interpolation and forecasting

Fit the allowed training snapshots, then call the appropriate prediction method:

```python
from step import STEP, STEPConfig

model = STEP(STEPConfig(seed=6121, device="cpu")).fit_wt(
    expression_by_time, coordinates_by_time, observed_times, genes=gene_names,
)
# Example observed_times: [0, 3, 7, 10]. Neither target is used for fitting.
interpolated = model.interpolate(target_time=5.0)
forecast = model.forecast(target_time=14.0)
model.save("outputs/wt_model.pt")
```

Interpolation uses snapshots on both sides of a missing time. Forecasting uses
preceding snapshots only. Predictions have the source snapshot's number of
particles, resampled by the learned population weights; they are population
predictions, without a one-to-one correspondence to held-out cells.

The same operations are available from the command line. Indices are zero-based:

```bash
step wt --data my_snapshots.npz --task interpolation --target-index 2 \
  --config configs/wt_limb.json --output outputs/my_interpolation
step wt --data my_snapshots.npz --task extrapolation --target-index 5 \
  --config configs/wt_limb.json --output outputs/my_forecast
```

The command holds out the selected snapshot, fits the allowed training stages,
and saves the checkpoint, predictions and metrics. Use a configuration as a
starting point for your assay: `wt_limb.json`, `wt_heart.json`, `wt_artista.json`
or `wt_mosta.json`. Heart's supplied configuration uses ordinal model times
`0, 1, …, 7`; retain the physical stage names as metadata.

### Spatial in silico perturbation

For a fitted temporal model, edit a selected tissue region and evolve it from
`start_time` to `end_time`:

```python
result = model.perturb(
    expression, coordinates,
    gene="GENE_OF_INTEREST", rows=region_mask,
    target_value=0.0, start_time=3.0, end_time=7.0,
)
# Each array has shape (n_observations, n_genes).
wt = result.baseline
edited = result.perturbed
response = result.effect                 # edited - wt
raw_target_residual = result.projection_residual
```

Projection is applied at initialization and after each integration step. The
reported residual is measured before the final target-coordinate overwrite.
For an assay with a single control tissue reference, use the stationary model:

```python
from step import StaticSTEP, StationaryConfig

static = StaticSTEP(StationaryConfig(seed=6120, device="cpu")).fit(
    control_expression, control_coordinates, genes=gene_names,
)
result = static.perturb(query_expression, query_coordinates,
                        gene="Tgfbr2", rows=region_mask)
```

The stationary estimator uses a dimensionless propagation interval `[0, 1]`,
rather than estimating a biological time course from a single snapshot. KO
measurements are needed only to evaluate predictions, not to fit either model.

### GRN extraction

```python
grn = model.extract_grn(
    expression, coordinates, time=3.0,
    regulators=["TF_A", "TF_B"], epsilon=0.01,
)
```

Rows are target genes and columns are the requested regulators. Values are
finite-difference sensitivities of the local decoded one-step map, averaged
over the supplied spatial patch. Self-edges are zeroed by default. These model
sensitivities provide a context-specific regulatory hypothesis, not a causal
validation. `StaticSTEP` exposes the same interface.

The command-line equivalents use a saved model and a prepared query NPZ:

```bash
step fit-static --data my_controls.npz --config configs/static_mouse.json \
  --output outputs/my_static
step perturb --model outputs/my_static/model.pt --data my_query.npz \
  --gene Tgfbr2 --output outputs/my_edit
step grn --model outputs/my_static/model.pt --data my_query.npz \
  --time 0 --regulators Tgfbr2 Jak2 --output outputs/my_grn
```

## Quick start: two bundled real datasets

The following commands train the neural models from scratch using the supplied
single-seed configurations. No private paths, external downloads or saved
research checkpoints are required. Reference scores below use **CUDA** with the
tested environment above. Use `--device cpu` for a CPU run; CPU and CUDA random
draws differ, so the same seed can produce a different fitted model. Omitting
`--device` selects CUDA when available and CPU otherwise.

### 1. Limb Xenium: WT interpolation and forecasting

```bash
python examples/wt_limb.py --device cuda
```

This uses the prepared 98-gene panel, 180 representative particles per stage,
all four interpolation folds and both one-step forecast folds. One configuration
and seed **6121** are shared across the two tasks. The reference results are:

| Task | Pearson ↑ | Spearman ↑ | RMSE ↓ | DE AUPRC ↑ | SWD ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Interpolation, four-fold mean | 0.6757 | 0.6413 | 0.1203 | 0.5109 | 0.2261 |
| Forecasting, two-fold mean | 0.2913 | 0.2352 | 0.1285 | 0.2925 | 0.2137 |

Results appear in `outputs/limb/summary.csv`, with each fold's metrics,
predictions, resolved configuration and trained model in its own subdirectory.
To run only one task, add `--task interpolation` or `--task extrapolation`.
To run a single prepared fold through the generic interface:

```bash
step wt --data data/limb_xenium/fold_4.npz --config configs/wt_limb.json --device cuda \
  --output outputs/limb_single_fold
```

### 2. Perturb-map mouse lung: spatial KO responses

```bash
python examples/perturb_mouse.py --device cuda
```

This fits the stationary model on **1,061 KP control spots and 128 genes**, then
runs all **80 local queries** for `Tgfbr2` and `Jak2` across four eligible
target/section groups. Seed **6120** gives a reference response AUPRC of
**0.3087**, averaging sections within each target, then targets equally. The
reference raw decoder-target residual is **0.4452**. Evaluation excludes the
edited gene and uses measured KO responses only after predictions are complete.

Outputs in `outputs/perturb_map/` include per-section and per-target metrics,
paired WT/edited response predictions, a projection residual trace, a checkpoint,
and a two-regulator GRN example. The script also writes a query file that can be
used directly with the generic interfaces:

```bash
step perturb --model outputs/perturb_map/model.pt \
  --data outputs/perturb_map/query.npz --gene Tgfbr2 --output outputs/lung_edit
step grn --model outputs/perturb_map/model.pt --data outputs/perturb_map/query.npz \
  --time 0 --regulators Tgfbr2 Jak2 --output outputs/lung_grn
```

Both examples use selected reference seeds/configurations, rather than averages
over initializations. Numerical results can vary with hardware and library
versions. Data descriptions and checksums are in [data/README.md](data/README.md).

## Outputs and evaluation

WT Pearson/Spearman compare signed population-mean changes relative to the
source. DE AUPRC ranks absolute changes against the observed top decile; RMSE
measures population-mean error and SWD compares particle distributions using
128 fixed projections. Perturbation AUPRC uses the same absolute-response
ranking after excluding the edited gene. A high AUPRC alone does not establish
correct up/down direction. `src/step/metrics.py` provides these metrics.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/step/` | Public API, IO, CLI, preprocessing, metrics and neural implementation |
| `configs/` | Single-run model settings for the evaluated assays |
| `examples/` | Two complete examples using bundled real data |
| `data/` | Compact Limb Xenium and Perturb-map inputs, descriptions and checksums |
| `docs/data_format.md` | Preparing and loading your own data |
| `tests/` | Neural transport, editing, persistence and interface checks |

Run `python -m unittest discover -s tests -v` to check the implementation.
Generated outputs are written outside the source/data folders and ignored by Git.


