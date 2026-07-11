"""Is the stored rhyme code compositional (vowel x coda) or holistic?

A rhyme family is a (stressed vowel, coda) pair. If the layer-14 value memory
encodes families as unrelated identities, a probe for the vowel should fail on
words from a family it never saw. If the code is phonemic — a shared vowel
component reused across codas — the probe should transfer.

Leave-one-family-out (LOFO): for every family whose vowel also occurs in other
families, train a vowel probe on all other families' words and test on the
held-out family. Same for codas. Chance baselines use the training-label
distribution. Representational similarity of family centroids against
shared-vowel / shared-coda structure complements the probes.

Runs entirely from saved E1 activations; no GPU needed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_gemma4_interpretability import write_jsonl


def make_probe():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=0))


def lofo_accuracy(features, word_families, family_label):
    """Mean accuracy predicting each held-out family's label from the rest."""
    scores, evaluated = [], 0
    for family in sorted(set(word_families)):
        label = family_label[family]
        other_families = {f for f in family_label if f != family and family_label[f] == label}
        if not other_families:
            continue  # label unique to this family: transfer is undefined
        train = np.array([f != family for f in word_families])
        test = ~train
        labels = np.array([family_label[f] for f in word_families])
        probe = make_probe().fit(features[train], labels[train])
        scores.append(float(probe.score(features[test], labels[test])))
        evaluated += 1
    return float(np.mean(scores)), evaluated


def majority_baseline(family_label):
    values = list(family_label.values())
    return max(values.count(v) for v in set(values)) / len(values)


def run(args):
    data = np.load(args.representation / "activations.npz")
    words = [str(w) for w in data["words"]]
    families = json.loads((args.representation / "families.json").read_text())
    family_of = {w: name for name, members in families.items() for w in members}
    word_families = [family_of[w] for w in words]

    vowel_of = {name: name.split("-")[0].rstrip("0123456789") for name in families}
    coda_of = {name: "-".join(name.split("-")[1:]) or "OPEN" for name in families}

    representations = {
        "embedding": data["states_final_word"][:, 0].astype(np.float32),
        "residual_layer_13": data["states_final_word"][:, 14].astype(np.float32),
        "l14_value": data["kv_final_word_full_attention_value"].astype(np.float32),
        "l14_key": data["kv_final_word_full_attention_key"].astype(np.float32),
    }

    rows = []
    for name, features in representations.items():
        for part, mapping in [("vowel", vowel_of), ("coda", coda_of)]:
            accuracy, evaluated = lofo_accuracy(features, word_families, mapping)
            rows.append({
                "representation": name,
                "part": part,
                "lofo_accuracy": accuracy,
                "families_evaluated": evaluated,
                "n_classes": len(set(mapping.values())),
                "majority_baseline": majority_baseline(mapping),
            })
            print(f"{name:18s} {part}: LOFO {accuracy:.3f} "
                  f"({evaluated} families, {len(set(mapping.values()))} classes)")

    # RSA: does centroid similarity reflect shared phonemes?
    order = sorted(families)
    for name, features in representations.items():
        centroids = np.stack([
            features[[i for i, f in enumerate(word_families) if f == fam]].mean(0)
            for fam in order
        ])
        centered = centroids - centroids.mean(0)
        normed = centered / np.linalg.norm(centered, axis=1, keepdims=True)
        similarity = normed @ normed.T
        upper = np.triu_indices(len(order), k=1)
        same_vowel = np.array([[vowel_of[a] == vowel_of[b] for b in order] for a in order])
        same_coda = np.array([[coda_of[a] == coda_of[b] for b in order] for a in order])
        rows.append({
            "representation": name,
            "part": "rsa",
            "same_vowel_minus_different": float(
                similarity[upper][same_vowel[upper]].mean()
                - similarity[upper][~same_vowel[upper]].mean()
            ),
            "same_coda_minus_different": float(
                similarity[upper][same_coda[upper]].mean()
                - similarity[upper][~same_coda[upper]].mean()
            ),
        })
        print(f"{name:18s} RSA: vowel {rows[-1]['same_vowel_minus_different']:+.3f} "
              f"coda {rows[-1]['same_coda_minus_different']:+.3f}")
    write_jsonl(args.representation / "phoneme_probe.jsonl", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", type=Path, default=Path("artifacts/gemma4_representation"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
