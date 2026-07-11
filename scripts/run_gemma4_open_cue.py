"""Scheme-controlled retrieval with three open cue lines.

The target stanza is identical in every condition: three completed lines that
do not rhyme with each other (three open families at distances 1, 2, 3 from
the incomplete line), then a neutral incomplete line. Only the demonstration
stanzas differ. If in-context scheme structure controls the retrieval head's
addressing, the completed family should follow the demonstrated scheme:
AABB -> adjacent cue, ABAB -> two back, ABBA -> three back.

A no-demonstration condition reveals the model's default policy among several
open lines.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rhyme_interp.dataset import build_open_cue_dataset
from rhyme_interp.model import load_model
from rhyme_interp.rhyme import rhyme_token_ids, rhymes

from run_gemma4_interpretability import (
    MODEL,
    anchor_positions,
    batch_inputs,
    forward_logits,
    write_jsonl,
)

CANDIDATE_LAYER, CANDIDATE_HEAD = 24, 3
CUES = ["cue_distance_1", "cue_distance_2", "cue_distance_3"]


def run(args):
    bundle = load_model(MODEL, load_in_4bit=True, attn_implementation="eager")
    model = bundle.model
    heads = model.config.text_config.num_attention_heads
    bundle.rhyme_cache = {}

    behavior_rows, attention_rows = [], []
    conditions = [("none", 0)] + [(scheme, 2) for scheme in ["aabb", "abab", "abba"]]
    for demo_scheme, demo_stanzas in conditions:
        examples = build_open_cue_dataset(
            demo_scheme if demo_stanzas else "aabb", demo_stanzas=demo_stanzas
        )
        prompts = [example.prompt for example in examples]
        for example in examples:
            for cue in CUES:
                word = getattr(example, cue)
                if word not in bundle.rhyme_cache:
                    bundle.rhyme_cache[word] = rhyme_token_ids(word, bundle.token_words)
        inputs = batch_inputs(prompts, bundle)
        logits = forward_logits(inputs, bundle)
        probs = logits.float().softmax(-1)

        positions = {
            cue: anchor_positions(
                prompts, [getattr(e, cue) for e in examples], bundle, inputs["input_ids"].shape[1]
            )
            for cue in CUES
        }
        with torch.inference_mode():
            outputs = model(**inputs, logits_to_keep=1, use_cache=False, output_attentions=True)

        for i, example in enumerate(examples):
            top_id = int(logits[i].argmax())
            top_word = bundle.token_words.get(top_id, bundle.tokenizer.decode([top_id]))
            row = {
                "id": example.id,
                "demo_scheme": demo_scheme,
                "top1": top_word,
            }
            for cue in CUES:
                word = getattr(example, cue)
                row[f"{cue}_word"] = word
                row[f"{cue}_mass"] = float(probs[i, bundle.rhyme_cache[word]].sum())
                row[f"top1_rhymes_{cue}"] = rhymes(word, top_word)
            behavior_rows.append(row)

        for layer_index, attn in enumerate(outputs.attentions):
            final_row = attn[:, :, -1]
            for i, example in enumerate(examples):
                for head in range(heads):
                    attention_rows.append({
                        "demo_scheme": demo_scheme,
                        "id": example.id,
                        "layer": layer_index,
                        "head": head,
                        **{cue: float(final_row[i, head, positions[cue][i]]) for cue in CUES},
                    })
        del outputs
        mean = lambda key: sum(r[key] for r in behavior_rows if r["demo_scheme"] == demo_scheme) / len(examples)
        print(f"{demo_scheme}: d1 {mean('cue_distance_1_mass'):.3f} "
              f"d2 {mean('cue_distance_2_mass'):.3f} d3 {mean('cue_distance_3_mass'):.3f}")

    write_jsonl(args.output / "open_cue_behavior.jsonl", behavior_rows)
    write_jsonl(args.output / "open_cue_attention.jsonl", attention_rows)
    print(f"Wrote open-cue analysis to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/gemma4_schemes"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
