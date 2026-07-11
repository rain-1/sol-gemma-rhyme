"""Summary figures for the phase-2 representation and routing analyses."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#2a78d6"
AQUA = "#1baf7a"
VIOLET = "#4a3aa7"
RED = "#e34948"
GRAY = "#8a8984"
LIGHT_GRAY = "#c9c8c2"
INK = "#0b0b0b"

REPRESENTATION = Path("artifacts/gemma4_representation")
SCHEMES = Path("artifacts/gemma4_schemes")
STEERING = Path("artifacts/gemma4_steering")
HEAD_OUTPUT = Path("artifacts/gemma4_head_output")
PLANNING = Path("artifacts/gemma4_planning")


def read(path):
    return pd.DataFrame([json.loads(line) for line in open(path)])


def style(ax, title):
    ax.set_title(title, fontsize=10, loc="left", color=INK)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(LIGHT_GRAY)
    ax.tick_params(colors="#52514e", labelsize=8)


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    fig.subplots_adjust(hspace=0.42, wspace=0.28, left=0.06, right=0.98, top=0.92, bottom=0.08)

    # A. Layerwise family decodability.
    ax = axes[0, 0]
    probe = read(REPRESENTATION / "layerwise_probe.jsonl")
    spelling = read(REPRESENTATION / "cross_spelling_probe.jsonl")
    numeric = spelling[spelling.layer.apply(lambda v: isinstance(v, int))]
    ax.plot(probe.layer, probe.probe_accuracy, color=BLUE, lw=2, label="30-family probe")
    ax.plot(probe.layer, probe.transfer_accuracy, color=AQUA, lw=2, label="scaffold transfer")
    ax.plot(numeric.layer, numeric.cross_spelling_accuracy, color=VIOLET, lw=2,
            label="cross-spelling")
    kv = read(REPRESENTATION / "shared_kv_probe.jsonl")
    value_acc = kv[(kv.scaffold == "final_word") & (kv.memory == "full_attention")
                   & (kv.part == "value")].probe_accuracy.iloc[0]
    key_acc = kv[(kv.scaffold == "final_word") & (kv.memory == "full_attention")
                 & (kv.part == "key")].probe_accuracy.iloc[0]
    ax.scatter([14], [value_acc], color=BLUE, marker="D", s=42, zorder=5)
    ax.annotate("L14 shared value", (14, value_acc), textcoords="offset points",
                xytext=(8, 2), fontsize=8, color=INK)
    ax.scatter([14], [key_acc], color=GRAY, marker="D", s=42, zorder=5)
    ax.annotate("L14 shared key", (14, key_acc), textcoords="offset points",
                xytext=(8, -4), fontsize=8, color="#52514e")
    ax.axhline(1 / 30, color=GRAY, lw=1, ls=":")
    ax.annotate("chance", (30, 1 / 30), textcoords="offset points", xytext=(0, 4),
                fontsize=7, color="#52514e")
    ax.axvline(13, color=LIGHT_GRAY, lw=1, ls="--")
    ax.set_xlabel("layer (-1 = embedding)", fontsize=8)
    ax.set_ylabel("family classification accuracy", fontsize=8)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    style(ax, "A. Rhyme family is decodable at the storage boundary (layer 13)")

    # B. Scheme behavior.
    ax = axes[0, 1]
    behavior = read(SCHEMES / "scheme_behavior.jsonl")
    grouped = behavior.groupby(["scheme", "demo_stanzas"])[
        ["correct_family_mass", "competing_family_mass"]
    ].mean()
    schemes = ["aabb", "abab", "abba"]
    x = np.arange(3)
    ax.bar(x - 0.28, [grouped.loc[(s, 0), "correct_family_mass"] for s in schemes],
           width=0.24, color=BLUE, alpha=0.45, label="correct family, no demos")
    ax.bar(x, [grouped.loc[(s, 2), "correct_family_mass"] for s in schemes],
           width=0.24, color=BLUE, label="correct family, scheme demos")
    ax.bar(x + 0.28, [grouped.loc[(s, 2), "competing_family_mass"] for s in schemes],
           width=0.24, color=GRAY, label="competing (closed) family")
    ax.set_xticks(x, [s.upper() for s in schemes])
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("mean probability mass", fontsize=8)
    ax.annotate("competing family ≈ 0 in every condition", (2.35, 0.03),
                fontsize=7.5, ha="right", color="#52514e")
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    style(ax, "B. Completion follows the rhyme scheme, not adjacency")

    # C. L24H3 attention by scheme (demos=2).
    ax = axes[0, 2]
    attention = read(SCHEMES / "scheme_attention.jsonl")
    h3 = attention[(attention.layer == 24) & (attention["head"] == 3)
                   & (attention.demo_stanzas == 2)]
    means = h3.groupby("scheme")[["cue_ending", "a2_ending", "a1_ending"]].mean()
    ax.bar(x - 0.28, [means.loc[s, "cue_ending"] for s in schemes], width=0.24,
           color=BLUE, label="open cue line")
    ax.bar(x, [means.loc[s, "a2_ending"] for s in schemes], width=0.24,
           color=GRAY, label="closed pair, 2nd line")
    ax.bar(x + 0.28, [means.loc[s, "a1_ending"] for s in schemes], width=0.24,
           color=LIGHT_GRAY, label="closed pair, 1st line")
    ax.set_xticks(x, [f"{s.upper()}\n(cue {d} back)" for s, d in zip(schemes, [1, 2, 3])])
    ax.set_ylabel("L24H3 attention from final token", fontsize=8)
    ax.legend(fontsize=8, frameon=False)
    style(ax, "C. L24H3 finds the cue ending wherever the scheme puts it")

    # D. Open-cue routing.
    ax = axes[1, 0]
    open_cue = read(SCHEMES / "open_cue_behavior.jsonl")
    conditions = ["none", "aabb", "abab", "abba"]
    predicted = {"aabb": 1, "abab": 2, "abba": 3}
    x4 = np.arange(4)
    for j, distance in enumerate([1, 2, 3]):
        values = [open_cue[open_cue.demo_scheme == c][f"cue_distance_{distance}_mass"].mean()
                  for c in conditions]
        colors = [BLUE if predicted.get(c) == distance else GRAY for c in conditions]
        ax.bar(x4 + (j - 1) * 0.26, values, width=0.22, color=colors,
               alpha=1.0 if distance != 3 else 0.9)
        for xi, v in zip(x4, values):
            ax.annotate(f"d{distance}", (xi + (j - 1) * 0.26, v), fontsize=6.5,
                        ha="center", va="bottom", color="#52514e")
    ax.set_xticks(x4, ["no demos", "AABB demos", "ABAB demos", "ABBA demos"])
    tallest = max(
        open_cue[open_cue.demo_scheme == c][f"cue_distance_{d}_mass"].mean()
        for c in conditions for d in (1, 2, 3)
    )
    ax.set_ylim(0, tallest * 1.35)
    ax.set_ylabel("mass on cue family", fontsize=8)
    ax.annotate("blue = scheme-predicted cue", (0.98, 0.93), xycoords="axes fraction",
                fontsize=8, ha="right", color=BLUE)
    style(ax, "D. Three open cues: demos steer which line gets closed")

    # E. Steering dose-response at each layer.
    ax = axes[1, 1]
    steering = read(STEERING / "steering.jsonl")
    for layer, color in [(11, LIGHT_GRAY), (12, AQUA), (13, BLUE), (14, GRAY)]:
        sub = steering[(steering.layer == layer) & (steering.variant == "full_mean")]
        curve = sub.groupby("strength").steered_target_mass.mean()
        ax.plot(curve.index, curve.values, color=color, lw=2, marker="o", ms=4,
                label=f"layer {layer}")
    holdout = steering[(steering.layer == 13) & (steering.variant == "holdout_mean")]
    curve = holdout.groupby("strength").steered_target_mass.mean()
    ax.plot(curve.index, curve.values, color=BLUE, lw=1.4, ls="--", marker="s", ms=4,
            label="layer 13, held-out words")
    random = steering[(steering.layer == 13) & (steering.variant == "random_words")]
    curve = random.groupby("strength").steered_target_mass.mean()
    ax.plot(curve.index, curve.values, color=RED, lw=1.4, ls=":", label="random-word vector")
    ax.axhline(steering.baseline_target_mass.mean(), color=GRAY, lw=1, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    ax.set_xlabel("steering strength", fontsize=8)
    ax.set_ylabel("mass on steered family", fontsize=8)
    ax.legend(fontsize=7.5, frameon=False)
    style(ax, "E. Family-mean vectors steer the rhyme only before storage")

    # F. Rank of the transferable constraint.
    ax = axes[1, 2]
    rank = read(HEAD_OUTPUT / "head_output_rank.jsonl")
    for control, color, label in [(False, BLUE, "top-k principal components"),
                                  (True, GRAY, "random k-dim subspace")]:
        sub = rank[rank.random_subspace == control]
        curve = sub.groupby("rank").recovery.mean()
        ax.plot(curve.index, curve.values, color=color, lw=2, marker="o", ms=4, label=label)
    ax.axhline(1.0, color=LIGHT_GRAY, lw=1, ls="--")
    ax.annotate("full transfer", (1, 1.0), textcoords="offset points", xytext=(2, 4),
                fontsize=7, color="#52514e")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 64, 256], ["1", "4", "16", "64", "256"])
    ax.set_xlabel("subspace rank (of 256)", fontsize=8)
    ax.set_ylabel("candidate-preference recovery", fontsize=8)
    ax.legend(fontsize=8, frameon=False)
    style(ax, "F. The constraint occupies a ~16-32 dim subspace of L24H3")

    fig.suptitle("Gemma 4 E2B rhyme representation and routing (phase 2)", fontsize=12,
                 x=0.06, ha="left", color=INK)
    out = Path("results/figures/gemma4_phase2_summary.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"wrote {out}")

    # Planning figure (separate, two panels).
    if (PLANNING / "planning_lens.jsonl").exists():
        lens = read(PLANNING / "planning_lens.jsonl")
        lens = lens[lens.offset <= 8]
        fig2, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
        fig2.subplots_adjust(wspace=0.28, left=0.08, right=0.98, top=0.86, bottom=0.14)
        curve = lens.groupby("offset").head_attention_to_anchor.mean()
        left.plot(curve.index, curve.values, color=BLUE, lw=2, marker="o", ms=4)
        left.set_xlabel("position in final line (0 = opening newline)", fontsize=8)
        left.set_ylabel("L24H3 attention to anchor", fontsize=8)
        style(left, "A. Retrieval-head engagement along the final line")
        for layer, color in [(13, AQUA), (24, VIOLET), (34, BLUE)]:
            curve = lens.groupby("offset")[f"family_mass_layer_{layer}"].mean()
            right.plot(curve.index, curve.values, color=color, lw=2, marker="o", ms=4,
                       label=f"layer {layer}")
        right.set_xlabel("position in final line", fontsize=8)
        right.set_ylabel("logit-lens family mass", fontsize=8)
        right.legend(fontsize=8, frameon=False)
        style(right, "B. When the rhyme family becomes readable")
        out2 = Path("results/figures/gemma4_phase2_planning.png")
        fig2.savefig(out2, dpi=160)
        print(f"wrote {out2}")


if __name__ == "__main__":
    main()
