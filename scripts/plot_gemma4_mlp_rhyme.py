"""Figure for the MLP-rhyme localization results (run_gemma4_mlp_rhyme.py)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ART = Path("artifacts/gemma4_mlp_rhyme/mlp_rhyme.jsonl")
OUT = Path("results/figures/gemma4_mlp_rhyme.png")

ATTN, MLP = "#56abff", "#2fbf9e"


def main():
    rows = [json.loads(l) for l in open(ART)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # E2: cumulative decodability by stream
    streams = [r for r in rows if r["experiment"] == "streams"]
    for comp, color, label in [("mlp_cumulative", MLP, "MLP stream"),
                               ("attention_cumulative", ATTN, "attention stream")]:
        pts = sorted((r["layer"], r["probe_accuracy"]) for r in streams if r["component"] == comp)
        ax1.plot([x for x, _ in pts], [y for _, y in pts], "-o", ms=3, color=color, label=label)
    emb = next(r["probe_accuracy"] for r in streams if r["component"] == "embedding")
    ax1.axhline(emb, ls=":", color="#8b93a2", lw=1, label=f"static embedding ({emb:.2f})")
    ax1.axvline(13, ls="--", color="#8b83ff", lw=1, alpha=.7)
    ax1.text(13.3, ax1.get_ylim()[1] * .96, "storage (L13)", color="#8b83ff", fontsize=8, va="top")
    ax1.set_xlabel("layer (cumulative up to)"); ax1.set_ylabel("rhyme-family probe accuracy")
    ax1.set_title("Where the rhyme code accumulates", fontsize=11)
    ax1.legend(fontsize=8, frameon=False); ax1.grid(alpha=.15)

    # E4: per-layer causal knockout at the anchor
    sweep = [r for r in rows if r["experiment"] == "layer_sweep"]
    base = next(r["mass"] for r in sweep if r["stream"] == "none")
    for stream, color, label in [("mlp", MLP, "zero MLP write"),
                                 ("attention", ATTN, "zero attention write")]:
        pts = sorted((r["layer"], r["mass"]) for r in sweep if r["stream"] == stream)
        ax2.plot([x for x, _ in pts], [y for _, y in pts], "-o", ms=3, color=color, label=label)
    ax2.axhline(base, ls=":", color="#8b93a2", lw=1, label=f"intact ({base:.2f})")
    ax2.set_xlabel("ablated layer (at anchor position)")
    ax2.set_ylabel("rhyme-family probability at the blank")
    ax2.set_title("Which layers causally hold the code", fontsize=11)
    ax2.legend(fontsize=8, frameon=False); ax2.grid(alpha=.15)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
