"""Rhyme-scheme routing: does the retrieval circuit follow AABB, ABAB, ABBA?

Each scheme presents the same four target-stanza lines in a different order.
The incomplete final line is identical, and the correct completion always
rhymes with couplet B's cue line; only the cue line's distance changes
(adjacent, two back, three back). The competing family A closes its own
couplet inside the stanza.

Measured per scheme, with and without in-scheme demonstration stanzas:

1. probability mass on the scheme-correct family B versus the competing
   family A, and the greedy word;
2. every head's final-token attention to the three line-ending positions
   (cue ending, adjacent closed ending, far closed ending); and
3. rhyme mass after ablating L24H3.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rhyme_interp.dataset import build_scheme_dataset
from rhyme_interp.model import load_model, target_token_id
from rhyme_interp.rhyme import rhyme_token_ids, rhymes

from run_gemma4_interpretability import (
    MODEL,
    anchor_positions,
    batch_inputs,
    forward_logits,
    write_jsonl,
    zero_head_final,
)

SCHEMES = ["aabb", "abab", "abba"]
CANDIDATE_LAYER, CANDIDATE_HEAD = 24, 3


def family_masses(logits, examples, bundle):
    probs = logits.float().softmax(-1)
    rows = []
    for i, example in enumerate(examples):
        mass_b = float(probs[i, bundle.rhyme_cache[example.anchor_b]].sum())
        mass_a = float(probs[i, bundle.rhyme_cache[example.anchor_a]].sum())
        top_id = int(logits[i].argmax())
        top_word = bundle.token_words.get(top_id, bundle.tokenizer.decode([top_id]))
        rows.append({
            "id": example.id,
            "scheme": example.scheme,
            "anchor_a": example.anchor_a,
            "anchor_b": example.anchor_b,
            "correct_family_mass": mass_b,
            "competing_family_mass": mass_a,
            "top1": top_word,
            "top1_in_correct_family": rhymes(example.anchor_b, top_word),
            "top1_in_competing_family": rhymes(example.anchor_a, top_word),
            "target_b_probability": float(probs[i, target_token_id(bundle.tokenizer, example.target_b)]),
        })
    return rows


def ending_positions(examples, prompts, bundle, padded_length):
    """Positions of the three completed line endings in each target stanza."""
    endings = {}
    for name, words in [
        ("cue_ending", [e.anchor_b for e in examples]),
        ("a2_ending", [e.target_a for e in examples]),
        ("a1_ending", [e.anchor_a for e in examples]),
    ]:
        endings[name] = anchor_positions(prompts, words, bundle, padded_length)
    return endings


def run(args):
    bundle = load_model(MODEL, load_in_4bit=not args.bf16, attn_implementation="eager")
    model = bundle.model
    layers = model.model.language_model.layers
    heads = model.config.text_config.num_attention_heads

    behavior_rows, attention_rows, ablation_rows = [], [], []
    for scheme in SCHEMES:
        for demo_stanzas in [0, 2]:
            examples = build_scheme_dataset(scheme, demo_stanzas=demo_stanzas)
            prompts = [example.prompt for example in examples]
            for example in examples:
                for anchor in (example.anchor_a, example.anchor_b):
                    bundle.rhyme_cache = getattr(bundle, "rhyme_cache", {})
                    if anchor not in bundle.rhyme_cache:
                        bundle.rhyme_cache[anchor] = rhyme_token_ids(anchor, bundle.token_words)
            inputs = batch_inputs(prompts, bundle)
            logits = forward_logits(inputs, bundle)
            for row in family_masses(logits, examples, bundle):
                behavior_rows.append({**row, "demo_stanzas": demo_stanzas})

            # Head attention from the final token to each completed line ending.
            with torch.inference_mode():
                outputs = model(**inputs, logits_to_keep=1, use_cache=False, output_attentions=True)
            positions = ending_positions(examples, prompts, bundle, inputs["input_ids"].shape[1])
            for layer_index, attn in enumerate(outputs.attentions):
                final_row = attn[:, :, -1]  # (batch, heads, seq)
                for i, example in enumerate(examples):
                    for head in range(heads):
                        attention_rows.append({
                            "scheme": scheme,
                            "demo_stanzas": demo_stanzas,
                            "id": example.id,
                            "layer": layer_index,
                            "head": head,
                            **{name: float(final_row[i, head, pos[i]]) for name, pos in positions.items()},
                        })
            del outputs

            # Necessity of the couplet retrieval head under each scheme.
            candidate = layers[CANDIDATE_LAYER]
            with zero_head_final(candidate, CANDIDATE_HEAD, candidate.self_attn.head_dim):
                ablated = forward_logits(inputs, bundle)
            for clean_row, ablated_row in zip(
                family_masses(logits, examples, bundle), family_masses(ablated, examples, bundle)
            ):
                ablation_rows.append({
                    "scheme": scheme,
                    "demo_stanzas": demo_stanzas,
                    "id": clean_row["id"],
                    "clean_correct_mass": clean_row["correct_family_mass"],
                    "ablated_correct_mass": ablated_row["correct_family_mass"],
                    "clean_competing_mass": clean_row["competing_family_mass"],
                    "ablated_competing_mass": ablated_row["competing_family_mass"],
                })
            print(f"{scheme} demos={demo_stanzas}: "
                  f"correct {sum(r['correct_family_mass'] for r in behavior_rows if r['scheme'] == scheme and r['demo_stanzas'] == demo_stanzas) / len(examples):.3f}")

    write_jsonl(args.output / "scheme_behavior.jsonl", behavior_rows)
    write_jsonl(args.output / "scheme_attention.jsonl", attention_rows)
    write_jsonl(args.output / "scheme_head_ablation.jsonl", ablation_rows)
    print(f"Wrote scheme routing analysis to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/gemma4_schemes"))
    parser.add_argument("--bf16", action="store_true", help="Run in BF16 instead of NF4")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
