"""Rhyme-set geometry: where and how Gemma 4 represents rhyme families.

Captures the residual stream at a line-final word position for ~360 words in
30 rhyme families, then asks:

1. At which layer does rhyme-family membership become linearly decodable?
2. Does the code generalize across rime spellings (phonology, not orthography)?
3. Does it generalize across scaffolds and across words?
4. Is the family readable from the layer-14 shared VALUE memory but not the
   shared KEY memory, as the phase-1 causal result predicts?
5. Does family structure appear as geometric clustering, not just probe
   accuracy?

Raw activations and family metadata are saved for the steering experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rhyme_interp.dataset import RHYME_DEMONSTRATION_LINES
from rhyme_interp.families import build_families, cross_spelling_families
from rhyme_interp.model import load_model

from run_gemma4_interpretability import MODEL, batch_inputs, write_jsonl


SCAFFOLDS = {
    "final_word": "The final word upon the page was {word}",
    "line_end": "Every line she wrote would end in {word}",
}

PROBE_SEED = 0


def scaffold_prompt(template: str, word: str) -> str:
    demos = "\n".join(RHYME_DEMONSTRATION_LINES)
    return f"{demos}\n{template.format(word=word)}"


@torch.inference_mode()
def capture_scaffold(words, template, bundle, batch_size=32):
    """Final-position residual states (embedding + every layer) and shared KV."""
    layer_states = []
    kv_states = {}
    for start in range(0, len(words), batch_size):
        chunk = words[start : start + batch_size]
        prompts = [scaffold_prompt(template, word) for word in chunk]
        inputs = batch_inputs(prompts, bundle)
        output = bundle.model(
            **inputs,
            logits_to_keep=1,
            use_cache=False,
            output_hidden_states=True,
            return_shared_kv_states=True,
        )
        # (words, embedding + n_layers, width)
        layer_states.append(
            torch.stack([h[:, -1].float() for h in output.hidden_states], dim=1).cpu()
        )
        for memory_name, (key, value) in output.shared_kv_states.items():
            store = kv_states.setdefault(memory_name, {"key": [], "value": []})
            store["key"].append(key[:, :, -1].float().flatten(1).cpu())
            store["value"].append(value[:, :, -1].float().flatten(1).cpu())
    return (
        torch.cat(layer_states).numpy(),
        {
            name: {part: torch.cat(chunks).numpy() for part, chunks in parts.items()}
            for name, parts in kv_states.items()
        },
    )


def make_probe():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, random_state=PROBE_SEED),
    )


def cv_accuracy(features, labels, folds=4, seed=PROBE_SEED):
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for train, test in splitter.split(features, labels):
        probe = make_probe().fit(features[train], labels[train])
        scores.append(float(probe.score(features[test], labels[test])))
    return float(np.mean(scores))


def transfer_accuracy(train_features, test_features, labels, folds=4, seed=PROBE_SEED):
    """Word-held-out transfer: train on scaffold A, test unseen words on B."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = []
    for train, test in splitter.split(train_features, labels):
        probe = make_probe().fit(train_features[train], labels[train])
        scores.append(float(probe.score(test_features[test], labels[test])))
    return float(np.mean(scores))


def geometry(features, labels):
    """Mean within-family and between-family cosine after centering."""
    centered = features - features.mean(0)
    normed = centered / np.linalg.norm(centered, axis=1, keepdims=True).clip(1e-8)
    similarity = normed @ normed.T
    same = labels[:, None] == labels[None, :]
    off_diagonal = ~np.eye(len(labels), dtype=bool)
    return {
        "within_family_cosine": float(similarity[same & off_diagonal].mean()),
        "between_family_cosine": float(similarity[~same].mean()),
    }


def cross_spelling_split(families):
    """Train on the dominant rime spelling; test on differently spelled members."""
    train_words, test_words = [], []
    for family in families:
        groups = sorted(family.spelling_groups().items(), key=lambda kv: -len(kv[1]))
        train_words.extend((word, family.name) for word in groups[0][1])
        for _suffix, words in groups[1:]:
            test_words.extend((word, family.name) for word in words)
    return train_words, test_words


def load_saved(output: Path):
    from rhyme_interp.families import RhymeFamily

    data = np.load(output / "activations.npz")
    words = [str(word) for word in data["words"]]
    labels = np.array([str(label) for label in data["labels"]])
    captured = {name: data[f"states_{name}"].astype(np.float32) for name in SCAFFOLDS}
    kv_captured: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for file in data.files:
        if not file.startswith("kv_"):
            continue
        body, _, part = file[3:].rpartition("_")
        scaffold, _, memory = next(
            (body[: len(s)], None, body[len(s) + 1 :]) for s in SCAFFOLDS if body.startswith(s)
        )
        kv_captured.setdefault(scaffold, {}).setdefault(memory, {})[part] = data[file].astype(np.float32)
    saved_families = json.loads((output / "families.json").read_text())
    families = [RhymeFamily(tuple(name.split("-")), tuple(members)) for name, members in saved_families.items()]
    return words, labels, captured, kv_captured, families


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if args.from_saved:
        words, labels, captured, kv_captured, families = load_saved(args.output)
    else:
        bundle = load_model(MODEL, load_in_4bit=not args.bf16, attn_implementation="eager")
        families = build_families(bundle.token_words)
        words = [word for family in families for word in family.words]
        family_of = {word: family.name for family in families for word in family.words}
        labels = np.array([family_of[word] for word in words])
        (args.output / "families.json").write_text(
            json.dumps({family.name: list(family.words) for family in families}, indent=1)
        )
        captured = {}
        kv_captured = {}
        for scaffold, template in SCAFFOLDS.items():
            captured[scaffold], kv_captured[scaffold] = capture_scaffold(words, template, bundle)
            print(f"captured {scaffold}: {captured[scaffold].shape}")
        np.savez_compressed(
            args.output / "activations.npz",
            words=np.array(words),
            labels=labels,
            **{f"states_{name}": states.astype(np.float16) for name, states in captured.items()},
            **{
                f"kv_{scaffold}_{memory}_{part}": arrays[part].astype(np.float16)
                for scaffold, memories in kv_captured.items()
                for memory, arrays in memories.items()
                for part in ("key", "value")
            },
        )

    rng = np.random.default_rng(PROBE_SEED)
    shuffled = labels.copy()
    rng.shuffle(shuffled)

    n_layers = captured["final_word"].shape[1]
    layer_rows = []
    for layer in range(n_layers):
        row = {"layer": layer - 1}  # -1 is the embedding row
        primary = captured["final_word"][:, layer]
        row["probe_accuracy"] = cv_accuracy(primary, labels)
        row["shuffled_accuracy"] = cv_accuracy(primary, shuffled)
        row["scaffold_b_accuracy"] = cv_accuracy(captured["line_end"][:, layer], labels)
        row["transfer_accuracy"] = transfer_accuracy(
            primary, captured["line_end"][:, layer], labels
        )
        row.update(geometry(primary, labels))
        layer_rows.append(row)
        print(
            f"layer {row['layer']:>2}: probe {row['probe_accuracy']:.3f} "
            f"transfer {row['transfer_accuracy']:.3f} "
            f"within-between {row['within_family_cosine'] - row['between_family_cosine']:.3f}"
        )
    write_jsonl(args.output / "layerwise_probe.jsonl", layer_rows)

    # Phonology versus orthography: train on one rime spelling, test on others.
    spelling_families = cross_spelling_families(families)
    train_pairs, test_pairs = cross_spelling_split(spelling_families)
    word_index = {word: i for i, word in enumerate(words)}
    train_index = [word_index[word] for word, _ in train_pairs]
    test_index = [word_index[word] for word, _ in test_pairs]
    train_labels = np.array([name for _, name in train_pairs])
    test_labels = np.array([name for _, name in test_pairs])
    # A matched within-spelling holdout (a third of the train-spelling words)
    # separates "generalizes to new words" from "generalizes to new spellings".
    rng_within = np.random.default_rng(PROBE_SEED)
    by_family: dict[str, list[str]] = {}
    for word, name in train_pairs:
        by_family.setdefault(name, []).append(word)
    within_train, within_test = [], []
    for name, members in by_family.items():
        members = list(members)
        rng_within.shuffle(members)
        held = max(1, len(members) // 3)
        within_test.extend((word, name) for word in members[:held])
        within_train.extend((word, name) for word in members[held:])

    def spelling_metrics(features):
        cross = make_probe().fit(features[train_index], train_labels)
        w_train = [word_index[w] for w, _ in within_train]
        w_test = [word_index[w] for w, _ in within_test]
        within = make_probe().fit(features[w_train], np.array([n for _, n in within_train]))
        return {
            "cross_spelling_accuracy": float(cross.score(features[test_index], test_labels)),
            "within_spelling_accuracy": float(
                within.score(features[w_test], np.array([n for _, n in within_test]))
            ),
        }

    spelling_rows = []
    for layer in range(n_layers):
        spelling_rows.append({
            "layer": layer - 1,
            "n_train": len(train_index),
            "n_test": len(test_index),
            "n_classes": len(spelling_families),
            **spelling_metrics(captured["final_word"][:, layer]),
        })
    for memory, arrays in kv_captured["final_word"].items():
        for part in ("key", "value"):
            spelling_rows.append({
                "layer": f"{memory}_{part}",
                "n_train": len(train_index),
                "n_test": len(test_index),
                "n_classes": len(spelling_families),
                **spelling_metrics(arrays[part]),
            })
    write_jsonl(args.output / "cross_spelling_probe.jsonl", spelling_rows)

    # Shared-memory probes: the causal result predicts the layer-14 VALUE
    # carries family content. Keys should be closer to structural/positional.
    kv_rows = []
    for scaffold, memories in kv_captured.items():
        for memory, arrays in memories.items():
            for part in ("key", "value"):
                kv_rows.append({
                    "scaffold": scaffold,
                    "memory": memory,
                    "part": part,
                    "width": int(arrays[part].shape[1]),
                    "probe_accuracy": cv_accuracy(arrays[part], labels),
                })
                print(f"{scaffold} {memory} {part}: {kv_rows[-1]['probe_accuracy']:.3f}")
    write_jsonl(args.output / "shared_kv_probe.jsonl", kv_rows)

    print(f"Wrote representation analysis to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/gemma4_representation"))
    parser.add_argument("--bf16", action="store_true", help="Run in BF16 instead of NF4")
    parser.add_argument("--from-saved", action="store_true",
                        help="Reuse saved activations instead of running the model")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
