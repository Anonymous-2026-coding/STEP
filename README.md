# STEP: Self-supervised Transport of Editable Perturbations
Trained on unpaired wild-type spatial snapshots, STEP is a generative framework that couples mask-supervised representation learning with mass-calibrated unbalanced optimal transport flow matching to learn an editable latent manifold alongside a continuous spatiotemporal path law. Through decoder-consistent proximal projections in latent space, STEP enforces manifold and spatial mass constraints during gene knockouts, enabling accurate spatiotemporal counterfactual forecasting and response-gene prioritization.

1. `fit_wt` learns a continuous latent wild-type path from unpaired spatial snapshots and supports
   one-step interpolation and extrapolation.
2. `perturb` masks a target gene, projects the masked state to the nearest decoder-consistent latent
   state, and propagates the edited state through the frozen WT path.


<img width="1602" height="679" alt="42f27962aaba9c8c3d406f5d85956f45" src="https://github.com/user-attachments/assets/42e5232a-7ed6-4724-949d-7e48d6c5006f" />


## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python demo/01_wt_dynamics.py
python demo/02_in_silico_perturbation.py
```

Outputs are written to `demo/output/` as CSV files and publication-style PNG figures.

## API

```python
from step import STEP

model = STEP(latent_dim=8)
model.fit_wt(expression_by_time, coordinates_by_time, times)
prediction = model.predict_next(source_time=1.0, target_time=2.0)
counterfactual = model.perturb(
    expression=expression_by_time[1],
    coordinates=coordinates_by_time[1],
    gene=3,
    rows=region_mask,
    target_value=0.0,
    start_time=1.0,
    end_time=2.0,
)
```

The perturbation result contains the WT trajectory, edited trajectory, response, projection trace, and
the latent support distance of the edited state relative to the WT training support. 


