"""Portable rhyme-family steering vectors.

Phase 1 transferred exact activations between matched prompts. This experiment
asks a stronger question: is the stored rhyme constraint an *abstract family
code*? A difference of family means — computed from scaffold contexts and from
words other than the prompt's anchor or its likely completions — is added at
the anchor position of a natural couplet prompt.

If the completion switches to the steered family, the layer-13/14 code is a
portable family-level direction, not a copy of any particular token's state.

Variants per (couplet, layer, strength):

- `full_mean`: target/source means over all lexicon words of each family;
- `holdout_mean`: means excluding each family's four most frequent words
  (the plausible completions themselves); and
- `random_words`: a matched difference vector between two means of shuffled,
  family-mixed word sets — the placebo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rhyme_interp.dataset import build_elicitation_dataset
from rhyme_interp.model import load_model
from rhyme_interp.rhyme import rhyme_token_ids, rhymes

from run_gemma4_interpretability import (
    MODEL,
    anchor_positions,
    batch_inputs,
    forward_logits,
    write_jsonl,
)

STEER_LAYERS = [11, 12, 13, 14]
STRENGTHS = [1.0, 2.0, 4.0, 8.0]
HOLDOUT = 4  # drop each family's most frequent words in the holdout variant


def family_lookup(families: dict[str, list[str]], word: str) -> str | None:
    for name, members in families.items():
        if rhymes(word, members[0]) or word in members:
            return name
    return None


def build_vectors(states, words, families, layer_index):
    """Family mean vectors at one captured layer (scaffold A contexts)."""
    word_index = {word: i for i, word in enumerate(words)}
    vectors = {}
    for name, members in families.items():
        rows = [word_index[w] for w in members]
        vectors[name] = {
            "full": states[rows, layer_index].mean(0),
            "holdout": states[rows[HOLDOUT:], layer_index].mean(0),
        }
    return vectors


def run(args):
    representation = Path(args.representation)
    data = np.load(representation / "activations.npz")
    words = [str(w) for w in data["words"]]
    states = data["states_final_word"].astype(np.float32)  # (words, layers+1, width)
    families = json.loads((representation / "families.json").read_text())

    bundle = load_model(MODEL, load_in_4bit=True, attn_implementation="eager")
    layers = bundle.model.model.language_model.layers

    examples = build_elicitation_dataset("rhyming")
    qualifying = []
    for example in examples:
        source = family_lookup(families, example.anchor)
        if source is not None:
            qualifying.append((example, source))
    # Steer each qualifying couplet toward the next qualifying couplet's family.
    plan = []
    for i, (example, source) in enumerate(qualifying):
        target = qualifying[(i + 1) % len(qualifying)][1]
        if target != source:
            plan.append((example, source, target))
    print(f"steering {len(plan)} couplets across {len({t for _, _, t in plan})} target families")

    prompts = [example.prompt for example, _, _ in plan]
    anchors = [example.anchor for example, _, _ in plan]
    inputs = batch_inputs(prompts, bundle)
    positions = anchor_positions(prompts, anchors, bundle, inputs["input_ids"].shape[1])
    batch_index = torch.arange(len(plan), device=bundle.device)

    mass_ids = {}
    for _, source, target in plan:
        for name in (source, target):
            if name not in mass_ids:
                mass_ids[name] = rhyme_token_ids(families[name][0], bundle.token_words)

    def family_mass(logits, name):
        return logits.float().softmax(-1)[:, mass_ids[name]].sum(-1)

    baseline_logits = forward_logits(inputs, bundle)

    rng = np.random.default_rng(0)
    all_words = list(words)
    rows = []
    for layer in STEER_LAYERS:
        vectors = build_vectors(states, words, families, layer + 1)  # +1 skips embedding row
        random_a = [words[i] for i in rng.choice(len(all_words), 12, replace=False)]
        random_b = [words[i] for i in rng.choice(len(all_words), 12, replace=False)]
        word_index = {word: i for i, word in enumerate(words)}
        random_vector = (
            states[[word_index[w] for w in random_a], layer + 1].mean(0)
            - states[[word_index[w] for w in random_b], layer + 1].mean(0)
        )
        for variant in ["full_mean", "holdout_mean", "random_words"]:
            deltas = []
            for _, source, target in plan:
                if variant == "random_words":
                    deltas.append(random_vector)
                else:
                    kind = "full" if variant == "full_mean" else "holdout"
                    deltas.append(vectors[target][kind] - vectors[source][kind])
            delta = torch.tensor(np.stack(deltas), device=bundle.device)
            for strength in STRENGTHS:
                def hook(_module, _args, output):
                    hidden = output.clone()
                    hidden[batch_index, positions] += (
                        strength * delta.to(hidden.dtype)
                    )
                    return hidden

                handle = layers[layer].register_forward_hook(hook)
                steered_logits = forward_logits(inputs, bundle)
                handle.remove()
                for i, (example, source, target) in enumerate(plan):
                    top_id = int(steered_logits[i].argmax())
                    top_word = bundle.token_words.get(
                        top_id, bundle.tokenizer.decode([top_id])
                    )
                    rows.append({
                        "id": example.id,
                        "anchor": example.anchor,
                        "source_family": source,
                        "target_family": target,
                        "layer": layer,
                        "strength": strength,
                        "variant": variant,
                        "vector_norm": float(np.linalg.norm(deltas[i])),
                        "baseline_source_mass": float(family_mass(baseline_logits, source)[i]),
                        "baseline_target_mass": float(family_mass(baseline_logits, target)[i]),
                        "steered_source_mass": float(family_mass(steered_logits, source)[i]),
                        "steered_target_mass": float(family_mass(steered_logits, target)[i]),
                        "top1": top_word,
                        "top1_in_target": rhymes(families[target][0], top_word)
                        or top_word in families[target],
                    })
                summary = [r for r in rows if r["layer"] == layer and r["strength"] == strength
                           and r["variant"] == variant]
                print(f"layer {layer} alpha {strength} {variant}: "
                      f"target mass {np.mean([r['steered_target_mass'] for r in summary]):.3f} "
                      f"(baseline {np.mean([r['baseline_target_mass'] for r in summary]):.3f}) "
                      f"top1-in-target {np.mean([r['top1_in_target'] for r in summary]):.2f}")
    write_jsonl(args.output / "steering.jsonl", rows)
    print(f"Wrote steering results to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation", type=Path, default=Path("artifacts/gemma4_representation"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gemma4_steering"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
