"""Render the main Gemma 4 causal results into one compact figure."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


root = Path("artifacts/gemma4_interpretability")
out = Path("results/figures")
out.mkdir(parents=True, exist_ok=True)

modules = pd.read_json(root / "module_ablation.jsonl", lines=True)
heads = pd.read_json(root / "head_ablation.jsonl", lines=True)
attention = pd.read_json(root / "attention_patterns.jsonl", lines=True)
patch = pd.read_json(root / "anchor_patching.jsonl", lines=True)
lens = pd.read_json(root / "logit_lens.jsonl", lines=True)

fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)

ax = axes[0, 0]
for kind, label in [
    ("attention_update", "attention update"),
    ("mlp_update", "MLP update"),
    ("per_layer_input_update", "per-layer token input"),
]:
    values = modules[modules.kind == kind].groupby("layer").delta_rhyme_mass.mean()
    ax.plot(values.index, 100 * values, marker="o", ms=3, label=label)
ax.axhline(0, color="black", lw=.8)
ax.set(title="A. Necessity by layer", xlabel="layer", ylabel="change in rhyme probability (points)")
ax.legend(frameon=False, fontsize=9)

ax = axes[0, 1]
matrix = heads.groupby(["layer", "head"]).delta_rhyme_mass.mean().unstack().to_numpy().T * 100
im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-70, vcenter=0, vmax=20), origin="lower")
ax.scatter([24], [3], facecolors="none", edgecolors="black", s=100, linewidths=1.5)
ax.text(24.4, 3, "−65", va="center", fontsize=8, fontweight="bold")
ax.set(title="B. Attention-head ablation", xlabel="layer", ylabel="head")
fig.colorbar(im, ax=ax, label="change in rhyme probability (points)")

ax = axes[1, 0]
values = patch.groupby(["mode", "start_layer"]).delta_original_family_mass.mean().unstack(0)
for mode in ["single", "cumulative"]:
    ax.plot(values.index, 100 * values[mode], marker="o", ms=3, label=mode)
ax.axvline(13.5, color="black", ls="--", lw=1, label="shared full K/V stored at L14")
ax.axhline(0, color="black", lw=.8)
ax.set(title="C. Correct-anchor state patched into counterfactual", xlabel="patched layer", ylabel="recovered source-family probability (points)")
ax.legend(frameon=False, fontsize=9)

ax = axes[1, 1]
trajectory = lens.groupby(["condition", "layer"]).target_minus_competitor.mean().unstack(0)
ax.plot(trajectory.index, trajectory.rhyming, marker="o", ms=3, label="ordered rhyming prefix")
ax.plot(trajectory.index, trajectory.shuffled, marker="o", ms=3, label="shuffled prefix")
ax.axhline(0, color="black", lw=.8)
ax.set(title="D. Layerwise rhyme target vs semantic competitor", xlabel="layer", ylabel="logit difference")
ax.legend(frameon=False, fontsize=9)

fig.suptitle("Gemma 4 E2B rhyme circuit: causal localization", fontsize=16)
fig.savefig(out / "gemma4_circuit_summary.png", dpi=180)
print(out / "gemma4_circuit_summary.png")
