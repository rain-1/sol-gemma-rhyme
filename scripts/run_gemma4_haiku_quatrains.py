"""External validity: scheme routing on fresh Claude-generated ABAB quatrains.

The controlled scheme experiments recycle 25 hand-written couplets. Here the
same questions run on independently generated ABAB quatrains (CMUdict-validated
exact rhymes, novel vocabulary): does the model complete line 4 with line 2's
family (the open cue two lines back) rather than the closed family of lines
1 and 3, and does L24H3 still address the cue ending?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from rhyme_interp.dataset import SCHEME_DEMONSTRATION_COUPLETS
from rhyme_interp.model import load_model, target_token_id
from rhyme_interp.rhyme import rhyme_token_ids, rhymes

from run_gemma4_interpretability import (
    MODEL,
    anchor_positions,
    batch_inputs,
    forward_logits,
    write_jsonl,
)

CANDIDATE_LAYER, CANDIDATE_HEAD = 24, 3


def abab_demo_prefix() -> str:
    def split(pair):
        first, second = pair
        return first, second

    lines = []
    for i in range(2):
        a1, a2 = split(SCHEME_DEMONSTRATION_COUPLETS[2 * i])
        b1, b2 = split(SCHEME_DEMONSTRATION_COUPLETS[2 * i + 1])
        lines.extend([a1, b1, a2, b2])
    return "\n".join(lines) + "\n\n"


def run(args):
    bundle = load_model(MODEL, load_in_4bit=True, attn_implementation="eager")
    model = bundle.model
    quatrains = [json.loads(line) for line in open(args.data)]

    usable = []
    for i, poem in enumerate(quatrains):
        endings = poem["endings"]
        if any(target_token_id(bundle.tokenizer, w) is None for w in endings[:3]):
            continue
        prefix, _, target = poem["lines"][3].rpartition(" ")
        usable.append({"index": i, "lines": poem["lines"], "endings": endings,
                       "prefix": prefix, "target": target})
    print(f"usable quatrains: {len(usable)} / {len(quatrains)}")

    bundle.rhyme_cache = {}
    for poem in usable:
        for word in poem["endings"][:2]:
            if word not in bundle.rhyme_cache:
                bundle.rhyme_cache[word] = rhyme_token_ids(word, bundle.token_words)

    rows = []
    for demos in [0, 2]:
        prefix = abab_demo_prefix() if demos else ""
        prompts = [prefix + "\n".join(p["lines"][:3]) + "\n" + p["prefix"] for p in usable]
        inputs = batch_inputs(prompts, bundle)
        logits = forward_logits(inputs, bundle)
        probs = logits.float().softmax(-1)
        with torch.inference_mode():
            outputs = model(**inputs, logits_to_keep=1, use_cache=False, output_attentions=True)
        attention = outputs.attentions[CANDIDATE_LAYER][:, CANDIDATE_HEAD]
        positions = {
            name: anchor_positions(prompts, [p["endings"][line] for p in usable],
                                   bundle, inputs["input_ids"].shape[1])
            for name, line in [("line1", 0), ("cue_line2", 1), ("line3", 2)]
        }
        del outputs
        for i, poem in enumerate(usable):
            cue, closed = poem["endings"][1], poem["endings"][0]
            top_id = int(logits[i].argmax())
            top_word = bundle.token_words.get(top_id, bundle.tokenizer.decode([top_id]))
            rows.append({
                "index": poem["index"],
                "demos": demos,
                "cue_word": cue,
                "closed_word": closed,
                "cue_family_mass": float(probs[i, bundle.rhyme_cache[cue]].sum()),
                "closed_family_mass": float(probs[i, bundle.rhyme_cache[closed]].sum()),
                "target_probability": float(
                    probs[i, target_token_id(bundle.tokenizer, poem["target"])]
                ) if target_token_id(bundle.tokenizer, poem["target"]) else None,
                "top1": top_word,
                "top1_rhymes_cue": rhymes(cue, top_word) or top_word == poem["target"],
                "top1_rhymes_closed": rhymes(closed, top_word),
                **{f"attention_{name}": float(attention[i, -1, pos[i]])
                   for name, pos in positions.items()},
            })
        import numpy as np
        sub = [r for r in rows if r["demos"] == demos]
        print(f"demos={demos}: cue mass {np.mean([r['cue_family_mass'] for r in sub]):.3f} "
              f"closed mass {np.mean([r['closed_family_mass'] for r in sub]):.3f} "
              f"top1-rhymes-cue {np.mean([r['top1_rhymes_cue'] for r in sub]):.2f} "
              f"attn cue {np.mean([r['attention_cue_line2'] for r in sub]):.3f} "
              f"line3 {np.mean([r['attention_line3'] for r in sub]):.3f}")
    write_jsonl(args.output / "haiku_quatrains.jsonl", rows)
    print(f"Wrote Haiku quatrain analysis to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/haiku_quatrains.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gemma4_schemes"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
