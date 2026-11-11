# Bundled real-data examples

The repository includes compact prepared inputs for two complete training and
prediction workflows. Other assays listed in the main README can be prepared
using the [common data format](../docs/data_format.md). Cite the original data
studies when reusing these inputs.

## Limb Xenium: spatial regeneration snapshots

Source: García-García et al., *bioRxiv*, 2025,
[10.1101/2025.03.30.645595](https://doi.org/10.1101/2025.03.30.645595);
[GEO GSE294212](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE294212).

`limb_xenium/fold_0.npz` through `fold_5.npz` contain six snapshots of
axolotl limb regeneration: uninjured and 3, 5, 7, 10, 14 days post-amputation.
Each file has 180 representative particles per stage and the same 98-gene
panel, derived from the processed spatial dataset. The retained source panel
is already normalized. For each fold, gene-wise q99 is fitted using its WT
training stages, mapped to 2 and clipped to `[0,2]`. Spatial coordinates are
median-centered and divided by the median radial distance. The original
prepared stage masses are 900 per stage. Particles are a compact computational
representation and should not be counted as biological replicates.

| File | Task | Predicted stage | Source stage | WT training stages |
| --- | --- | --- | --- | --- |
| `fold_0.npz` | Interpolation | 3 dpa | uninjured | uninjured, 5, 7, 10, 14 dpa |
| `fold_1.npz` | Interpolation | 5 dpa | 3 dpa | uninjured, 3, 7, 10, 14 dpa |
| `fold_2.npz` | Interpolation | 7 dpa | 5 dpa | uninjured, 3, 5, 10, 14 dpa |
| `fold_3.npz` | Interpolation | 10 dpa | 7 dpa | uninjured, 3, 5, 7, 14 dpa |
| `fold_4.npz` | Forecasting | 10 dpa | 7 dpa | uninjured, 3, 5, 7 dpa |
| `fold_5.npz` | Forecasting | 14 dpa | 10 dpa | uninjured, 3, 5, 7, 10 dpa |

Use `configs/wt_limb.json`, seed **6121**, for both tasks. Reference fold means
are in `limb_xenium/reference_metrics.json` and in the main README. The example
trains all six models from scratch; it does not ship trained checkpoints.

## Perturb-map: spatial mouse lung KO responses

Source: Dhainaut et al., *Cell*, 2022,
[GEO GSE193460](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE193460).

Expression uses library-size normalization followed by `log1p`, with 128
retained genes. WT controls are author-annotated unperturbed KP lesions.

| File | Content |
| --- | --- |
| `perturb_map/wt.npz` | WT expression `(1061,128)`, coordinates `(1061,2)`, within-section adjacency `(1061,1061)` and ordered gene names |
| `perturb_map/queries.npz` | 80 WT-derived local patches, expression `(80,4,128)`, adjacency `(80,4,4)`, group indices, gene names and group metadata |
| `perturb_map/evaluation.npz` | Four measured KO-minus-WT pseudobulk effects `(4,128)`, unit scales and matching group/gene metadata |

Every patch contains an anchor and its local control context; only the first
row is edited. There are 20 queries per group:

| Target | Tissue section |
| --- | --- |
| Tgfbr2 | GSM5808054_KP_1 |
| Tgfbr2 | GSM5808055_KP_2 |
| Tgfbr2 | GSM5808057_KP_4 |
| Jak2 | GSM5808054_KP_1 |

Use `configs/static_mouse.json`, seed **6120**. The model fits `wt.npz` only,
then generates predictions from the control-derived patches. The example
opens `evaluation.npz` after prediction. It excludes the edited gene from
scoring, averages sections within each target and then weights targets equally.
The reference response AUPRC is **0.3087**; additional single-seed metrics are
in `perturb_map/reference_metrics.json`. Reference scores for both examples use
CUDA. CPU execution is supported; its seeded training draws differ from CUDA.

The quick start writes `query.npz` for the general perturbation/GRN CLI, as well
as predictions, a raw decoder-target projection trace, per-unit scores and a
two-regulator sensitivity matrix. The stationary propagation interval is
dimensionless; these data are not an observed temporal KO series.

## Integrity and reference settings

Each dataset's `manifest.json` records file sizes, SHA256 hashes and NPZ array
schemas. Run `python scripts/verify_release.py` from the repository root to
check them. Reference metrics use one selected configuration/seed per example,
rather than a mean over model initializations. Exact values can vary with
hardware and numerical library versions.
