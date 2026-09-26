from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from step import STEP

ROOT = Path(__file__).resolve().parents[1]
data = np.load(ROOT / "demo/data/wt_subset.npz")
model = STEP(latent_dim=8).fit_wt(data["expression"], data["coordinates"], data["times"])

interpolation = model.predict_next(1.0, 1.5, source_index=1)
extrapolation = model.predict_next(2.0, 3.0, source_index=2)
np.savez_compressed(ROOT / "demo/output/wt_predictions.npz", interpolation=interpolation, extrapolation=extrapolation)

fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), constrained_layout=True)
axes[0].plot(data["times"], data["expression"].mean(axis=(1, 2)), "o-", color="#4C78A8")
axes[0].set(title="WT snapshot trajectory", xlabel="time", ylabel="mean expression")
axes[1].bar(["interpolation", "one-step\nextrapolation"], [interpolation.mean(), extrapolation.mean()], color=["#72B7B2", "#F2CF5B"])
axes[1].set(title="Predicted WT states", ylabel="mean expression")
fig.savefig(ROOT / "demo/output/wt_dynamics.png", dpi=240)
print("WT demo complete: demo/output/wt_dynamics.png")
