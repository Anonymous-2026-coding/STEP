# Preparing STEP inputs

STEP accepts unpaired spatial transcriptomic snapshots. Rows represent cells or
spots; columns represent the same ordered gene panel in every input. There is
no requirement to match individual observations across time. The neural model
uses dense arrays, so prepare a representative spatial subset for large assays.

## Expression, coordinates and training splits

Use finite, nonnegative expression in consistent units. For count-based assays,
library-size normalization followed by `log1p` is a suitable starting point.
Intersect gene identifiers across assay schemas before aligning columns. Fit
gene selection and expression scaling on the allowed WT training snapshots;
apply the resulting transformation to evaluation snapshots and intervention
queries. Retain a perturbation target in the selected panel to edit that gene.

For example, after gene alignment:

```python
import numpy as np
from step.preprocessing import normalize_counts, batch_aware_hvg

# counts_by_time: aligned (n_t, G) raw-count matrices.
# train_indices: preceding stages for forecasting, or all observed stages
# except the missing intermediate stage for interpolation.
normalized = [normalize_counts(x).toarray() for x in counts_by_time]
selected, _ = batch_aware_hvg(
    [normalized[i] for i in train_indices], n_top_genes=1000,
)
scale = np.quantile(
    np.vstack([normalized[i][:, selected] for i in train_indices]),
    0.99, axis=0,
).clip(min=1e-3)
expression_by_time = [
    np.clip(2.0 * x[:, selected] / scale, 0.0, 2.0).astype(np.float32)
    for x in normalized
]
gene_names = np.asarray(aligned_gene_names)[selected]
```

This is an example transformation; already normalized targeted panels need not
be normalized a second time. A requested knockout value of zero is in the
transformed nonnegative expression units.

Coordinates have shape `(n_t, 2)` and the same row order as expression. Use a
consistent coordinate convention across stages. `normalize_coordinates` in
`step.preprocessing` provides median centering and radial scaling. When several
tissue sections form one snapshot, build spatial graphs within each section:

```python
from step import build_knn_adjacency
adjacency = build_knn_adjacency(
    coordinates, n_neighbors=8, groups=section_ids,
).numpy()
```

Adjacencies are dense, nonnegative, row-normalized matrices, with shape
`(n_t, n_t)`. Passing them explicitly avoids edges between separate sections.
Without an adjacency, STEP builds a spatial k-nearest-neighbor graph.

## WT snapshot files

The public helper supports different observation counts at different times:

```python
from step.io import save_snapshots, load_snapshots

save_snapshots(
    "snapshots.npz", expression_by_time, coordinates_by_time,
    times=[0, 3, 5, 7, 10, 14], genes=gene_names,
    names=["uninjured", "3dpa", "5dpa", "7dpa", "10dpa", "14dpa"],
    # Optional: masses=design_mass_by_time, adjacencies=graphs_by_time
)
snapshots = load_snapshots("snapshots.npz")
```

| NPZ key | Shape / meaning |
| --- | --- |
| `expression_0`, `expression_1`, … | `(n_t, G)` nonnegative expression |
| `coordinates_0`, `coordinates_1`, … | `(n_t, 2)` coordinates |
| `times` | `(T,)` finite, strictly increasing numeric model times |
| `genes` | `(G,)` unique Unicode gene names, in column order |
| `names` | Optional `(T,)` Unicode stage names |
| `masses` | Optional `(T,)` positive design masses; default is one per stage |
| `adjacency_0`, `adjacency_1`, … | Optional `(n_t, n_t)` spatial graphs |

The loader also accepts stacked `expression (T,N,G)` and
`coordinates (T,N,2)` arrays when all snapshots have the same number of rows.
The bundled Limb files use this form. All files load with `allow_pickle=False`.

`step wt` accepts an unsplit file with `--task` and `--target-index`. Indices are
zero-based: interpolation fits all stages except the target; extrapolation fits
only stages preceding it. Both need at least two training stages. Held-out
expression is used only for scoring after fitting and prediction.

A prepared fold may instead include `train` (integer training indices),
`source` and `target` (scalar indices), and `task` (scalar Unicode string,
`interpolation` or `extrapolation`). These metadata let `step wt` run without
split arguments. Each bundled Limb fold has its own training-only scaling;
use each fold's arrays together.

Use physical numeric times for the Limb configuration. The supplied heart
configuration uses stage indices `0,1,…,7`, retaining physical stages in
`names`. Model time changes the vector-field fitting problem; use the same
convention during fitting and inference. The CLI uses the complete scheduled
time range for model-time normalization. The Python API offers the same control
with `fit_wt(..., time_bounds=(start, end))`; only scheduled times, rather than
held-out expression, enter that argument.

## Static control and intervention files

For `step fit-static`, save a single WT control reference:

```python
np.savez_compressed(
    "controls.npz", expression=control_expression,
    coordinates=control_coordinates, genes=np.asarray(gene_names, dtype="U"),
    # adjacency=control_adjacency,  # optional
)
```

For `step perturb` or `step grn`, save a local query patch in the same expression
units and gene order as the fitted reference:

```python
np.savez_compressed(
    "query.npz", expression=query_expression,
    coordinates=query_coordinates, genes=np.asarray(gene_names, dtype="U"),
    rows=np.asarray(region_mask, dtype=bool),
)
```

`expression` has shape `(N,G)`, `coordinates` has shape `(N,2)`, and `rows` is a
Boolean `(N,)` mask selecting at least one edited observation. A query can use
`adjacency (N,N)` instead of coordinates. GRN extraction does not require
`rows`: its regulator increments apply to every row of the supplied patch.
The CLI checks query gene names/order when `genes` is present.

Repeated projection occurs at initialization and after integration steps. The
reported `projection_residual` measures the raw decoder output before the
final target-coordinate overwrite. Saved effects equal edited minus paired
WT predictions. For biological response evaluation, exclude the directly
edited gene, aggregate comparable tissue contexts, and compare against
measured KO-minus-WT effects in the same units.

## Bundled examples

The [data README](../data/README.md) describes the two prepared datasets,
folds, query groups and reference metrics. Limb arrays are representative
particles from a processed targeted panel, rather than raw counts or
independent biological replicates. The Perturb-map example separates the
WT training reference, control-derived query patches and measured KO effects
into distinct files; KO effects are opened only after all predictions finish.
