"""Stage 3 of the MLP-13 SAE: are the learned features monosemantic phonemes?

For each SAE feature we look at the words that activate it most and ask whether
they share a coda or a vowel. If the dictionary has cleanly coda- and
vowel-selective features, the superposition report 09 found is resolved by an
overcomplete sparse basis where a change of coordinates could not.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np


def parse(label):
    if not label:
        return None, None
    parts = label.split("-")
    return parts[0], ("-".join(parts[1:]) or "OPEN")


def purity(words_idx, attr):
    vals = [attr[i] for i in words_idx if attr[i]]
    if not vals:
        return None, 0.0
    top, c = Counter(vals).most_common(1)[0]
    return top, c / len(vals)


def run(args):
    data = np.load(args.input, allow_pickle=True)
    H = data["features"].astype(np.float32)          # (words, features)
    words = [str(w) for w in data["words"]]
    labels = [str(l) for l in data["labels"]]
    vowel = [parse(l)[0] for l in labels]
    coda = [parse(l)[1] for l in labels]

    alive = np.where(H.max(0) > 1e-6)[0]
    l0 = (H > 1e-6).sum(1).mean()
    print(f"{H.shape[1]} features, {len(alive)} alive, mean L0 {l0:.1f}, {len(words)} words\n")

    topk = args.topk
    rows = []
    for j in alive:
        order = np.argsort(-H[:, j])[:topk]
        if H[order[-1], j] <= 1e-6:
            order = order[H[order, j] > 1e-6]
        if len(order) < 5:
            continue
        cod, cp = purity(order, coda)
        vow, vp = purity(order, vowel)
        rows.append((j, cod, cp, vow, vp, [words[i] for i in order[:6]]))

    coda_feats = sorted([r for r in rows if r[2] >= args.purity and r[2] >= r[4]], key=lambda r: -r[2])
    vowel_feats = sorted([r for r in rows if r[4] >= args.purity and r[4] > r[2]], key=lambda r: -r[4])

    print(f"coda-monosemantic features (>= {args.purity:.0%} of top-{topk} words share a coda): {len(coda_feats)}")
    seen = set()
    for j, cod, cp, vow, vp, ex in coda_feats:
        if cod in seen:
            continue
        seen.add(cod)
        print(f"  feat {j:5d}  coda -{cod:4s} {cp:.0%}  e.g. {', '.join(ex)}")
    print(f"\nvowel-monosemantic features: {len(vowel_feats)}")
    seen = set()
    for j, cod, cp, vow, vp, ex in vowel_feats:
        if vow in seen:
            continue
        seen.add(vow)
        print(f"  feat {j:5d}  vowel {vow:5s} {vp:.0%}  e.g. {', '.join(ex)}")

    n_coda_pure = sum(1 for r in rows if r[2] >= args.purity)
    n_vowel_pure = sum(1 for r in rows if r[4] >= args.purity)
    print(f"\nsummary: {n_coda_pure} coda-pure and {n_vowel_pure} vowel-pure features "
          f"of {len(rows)} live features covering {len(set(r[1] for r in coda_feats))} distinct codas, "
          f"{len(set(r[3] for r in vowel_feats))} distinct vowels")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("artifacts/gemma4_mlp_rhyme/sae_l13.npz"))
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--purity", type=float, default=0.8)
    run(p.parse_args())


if __name__ == "__main__":
    main()
