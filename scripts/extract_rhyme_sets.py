"""Read rhyme sets off the code the retrieval head actually consumes.

Report 07 shows rhyme is written by the layer-13 MLP but is noisy in the raw
residual stream (a probe there reaches only ~0.2-0.5). The retrieval head does
not read the raw residual: it reads the layer-14 shared full-attention VALUE
memory at the anchor, which is the value projection of that residual. Probing
*there* -- the representation the circuit analysis identifies as the rhyme code
-- gives a strong, precise readout.

This script consumes the value memory captured by run_gemma4_representation.py
(`artifacts/gemma4_representation/activations.npz`) and:

  1. fits a linear readout on a training half of the words and reports its
     accuracy on held-out words and across a held-out scaffold;
  2. pulls out rhyme sets by grouping held-out words by the readout's decision;
  3. runs a query tool -- give a word, get its rhyme set;
  4. compares against the raw residual stream (the control).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 0
QUERIES = ["light", "moon", "grace", "snow", "day", "small", "rain", "dream", "street", "time"]


def probe():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0, random_state=SEED))


def unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def purity(assign, truth):
    groups = defaultdict(list)
    for a, t in zip(assign, truth):
        groups[a].append(t)
    return sum(Counter(g).most_common(1)[0][1] for g in groups.values()) / len(truth)


def held_out_split(labels, rng):
    per = defaultdict(list)
    for i in rng.permutation(len(labels)):
        per[labels[i]].append(i)
    train, test = [], []
    for members in per.values():
        cut = max(3, round(len(members) * 0.6))
        train += members[:cut]
        test += members[cut:]
    return train, test


def run(args):
    data = np.load(args.activations, allow_pickle=True)
    words = [str(w) for w in data["words"]]
    labels = [str(l) for l in data["labels"]]
    # the code the retrieval head reads, and (for the control) the raw residual
    value = data["kv_line_end_full_attention_value"].astype(np.float32)
    value_alt = data["kv_final_word_full_attention_value"].astype(np.float32)  # other scaffold
    residual = data["states_line_end"][:, 13].astype(np.float32)  # layer-13 residual
    families = sorted(set(labels))
    chance = 1 / len(families)
    print(f"{len(words)} words, {len(families)} families, chance {chance:.3f}")

    rng = np.random.default_rng(SEED)
    tr, te = held_out_split(labels, rng)
    ytr, yte = [labels[i] for i in tr], [labels[i] for i in te]
    words_te = [words[i] for i in te]

    # (1) readout on held-out words, and across a held-out scaffold
    clf = probe().fit(value[tr], ytr)
    pred = list(clf.predict(value[te]))
    acc = np.mean([a == b for a, b in zip(pred, yte)])
    alt = probe().fit(value_alt[tr], ytr)
    acc_transfer = np.mean([alt.predict(value[te])[k] == yte[k] for k in range(len(te))])
    resid_clf = probe().fit(residual[tr], ytr)
    acc_resid = np.mean([a == b for a, b in zip(resid_clf.predict(residual[te]), yte)])
    print(f"\nheld-out-word family readout:")
    print(f"  layer-14 value memory (head reads here) : {acc:.2f}")
    print(f"  raw layer-13 residual (control)         : {acc_resid:.2f}")
    print(f"  value memory, across held-out scaffold  : {acc_transfer:.2f}")

    # (2) extracted rhyme sets
    print("\n--- rhyme sets pulled from held-out words (grouped by the readout) ---")
    by = defaultdict(list)
    for w, p, t in zip(words_te, pred, yte):
        by[p].append(w + ("" if p == t else "*"))
    for fam in sorted(by, key=lambda f: -len(by[f])):
        if len(by[fam]) >= 3:
            print(f"  {fam:9s}: {', '.join(by[fam])}")
    print("  (* = readout placed it here but CMUdict lists a different family)")

    # (3) unsupervised control
    k = len(set(yte))
    val_cl = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit(unit(value[te]))
    res_cl = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit(unit(residual[te]))
    print(f"\nunsupervised cluster purity on held-out words:")
    print(f"  layer-14 value memory : {purity(val_cl.labels_, yte):.2f}")
    print(f"  raw layer-13 residual : {purity(res_cl.labels_, yte):.2f}")

    # (4) query tool
    print("\n--- query: give a word, get its rhyme set (top-6 nearest in value space) ---")
    Vu = unit(StandardScaler().fit_transform(value))
    wi = {w: i for i, w in enumerate(words)}
    queries = {}
    for q in QUERIES:
        if q in wi:
            order = [j for j in np.argsort(-(Vu @ Vu[wi[q]])) if j != wi[q]][:6]
            queries[q] = [words[j] for j in order]
            print(f"  {q:7s} -> {', '.join(queries[q])}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "chance": chance, "n_words": len(words), "n_families": len(families),
            "readout_value_memory": float(acc), "readout_residual": float(acc_resid),
            "readout_cross_scaffold": float(acc_transfer),
            "purity_value": purity(val_cl.labels_, yte), "purity_residual": purity(res_cl.labels_, yte),
            "extracted_sets": {f: by[f] for f in by}, "queries": queries,
        }, indent=2))
        print(f"\nwrote {args.output}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--activations", type=Path,
                   default=Path("artifacts/gemma4_representation/activations.npz"),
                   help="value memory captured by run_gemma4_representation.py")
    p.add_argument("--output", type=Path, default=Path("artifacts/gemma4_mlp_rhyme/rhyme_sets.json"))
    run(p.parse_args())


if __name__ == "__main__":
    main()
