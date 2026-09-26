from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from step import STEP

ROOT = Path(__file__).resolve().parents[1]
data = np.load(ROOT / "demo/data/wt_subset.npz")
model = STEP(latent_dim=8).fit_wt(data["expression"], data["coordinates"], data["times"])
rows = data["coordinates"][1, :, 0] <= np.quantile(data["coordinates"][1, :, 0], 0.4)
result = model.perturb(data["expression"][1], data["coordinates"][1], gene=3, rows=rows,
                       target_value=0.0, start_time=1.0, end_time=2.0)
np.savez_compressed(ROOT / "demo/output/perturbation.npz", wt=result.wt, edited=result.edited, response=result.response)

fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2), constrained_layout=True)
axes[0].plot([x["step"] for x in result.projection_trace], [x["target_residual"] for x in result.projection_trace], "o-", color="#E45756")
axes[0].set(title="Proximal projection", xlabel="iteration", ylabel="target residual")
axes[1].scatter(data["coordinates"][1, :, 0], data["coordinates"][1, :, 1], c=np.abs(result.response[:, 3]), s=22, cmap="magma")
axes[1].set(title="Spatial response", xlabel="x", ylabel="y")
axes[2].bar(np.arange(data["expression"].shape[2]), np.abs(result.response).mean(axis=0), color="#5DA3D9")
axes[2].set(title="Downstream response", xlabel="gene index", ylabel="mean |effect|")
fig.savefig(ROOT / "demo/output/in_silico_perturbation.png", dpi=240)
print(f"Perturbation demo complete: support percentile={result.support_percentile:.3f}")
