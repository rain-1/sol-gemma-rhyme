"""Does Gemma 4 plan the rhyme before it reaches the final word?

Claude exhibits forward planning in rhymed poetry: candidate rhyme words are
already active at the start of the final line. This script asks the same
question for Gemma 4 E2B on the natural couplet benchmark.

At every token position of the incomplete final line (including the newline
that opens it):

1. **Logit-lens family mass.** Project each layer's residual through the
   final norm and unembedding; measure the probability mass (within the
   single-token word vocabulary) on the anchor's rhyme family. Early activity
   at line-start positions indicates planning rather than last-token lookup.
2. **Retrieval-head engagement.** L24H3's attention from that position to the
   anchor. Does the head retrieve the anchor throughout the line or only at
   the end?
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from rhyme_interp.dataset import build_elicitation_dataset
from rhyme_interp.model import load_model
from rhyme_interp.rhyme import rhyme_token_ids

from run_gemma4_interpretability import (
    MODEL,
    anchor_positions,
    batch_inputs,
    write_jsonl,
)

CANDIDATE_LAYER, CANDIDATE_HEAD = 24, 3
LENS_LAYERS = [10, 13, 16, 20, 24, 27, 30, 34]


def run(args):
    bundle = load_model(MODEL, load_in_4bit=True, attn_implementation="eager")
    model = bundle.model
    norm = model.model.language_model.norm
    examples = build_elicitation_dataset("rhyming")
    anchors = [example.anchor for example in examples]
    prompts = [example.prompt for example in examples]
    bundle.rhyme_cache = {a: rhyme_token_ids(a, bundle.token_words) for a in set(anchors)}

    eligible_ids = torch.tensor(sorted(bundle.token_words), device=bundle.device)
    unembed = model.lm_head.weight[eligible_ids]
    family_masks = {}
    id_position = {int(t): i for i, t in enumerate(eligible_ids.tolist())}
    for anchor in set(anchors):
        mask = torch.zeros(len(eligible_ids), dtype=torch.bool, device=bundle.device)
        mask[[id_position[t] for t in bundle.rhyme_cache[anchor]]] = True
        family_masks[anchor] = mask

    inputs = batch_inputs(prompts, bundle)
    padded = inputs["input_ids"].shape[1]
    anchor_pos = anchor_positions(prompts, anchors, bundle, padded)

    # The final line starts at the newline after the target couplet's first
    # line; find it as the last newline-containing token in each prompt.
    newline_positions = []
    for prompt in prompts:
        ids = bundle.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        texts = [bundle.tokenizer.decode([t]) for t in ids]
        hits = [i for i, t in enumerate(texts) if "\n" in t]
        newline_positions.append(hits[-1] + padded - len(ids))

    with torch.inference_mode():
        outputs = model(
            **inputs, use_cache=False, output_hidden_states=True, output_attentions=True
        )

    lens_rows = []
    attention = outputs.attentions[CANDIDATE_LAYER][:, CANDIDATE_HEAD]  # (batch, q, k)
    for i, example in enumerate(examples):
        start = newline_positions[i]
        mask = family_masks[example.anchor]
        for position in range(start, padded):
            offset = position - start  # 0 = the newline opening the final line
            head_attention = float(attention[i, position, anchor_pos[i]])
            row = {
                "id": example.id,
                "anchor": example.anchor,
                "offset": offset,
                "is_final": position == padded - 1,
                "token": bundle.tokenizer.decode([int(inputs["input_ids"][i, position])]),
                "head_attention_to_anchor": head_attention,
            }
            for layer in LENS_LAYERS:
                hidden = outputs.hidden_states[layer + 1][i, position]
                lens = F.linear(norm(hidden.unsqueeze(0)).to(unembed.dtype), unembed).float()[0]
                probs = lens.softmax(-1)
                row[f"family_mass_layer_{layer}"] = float(probs[mask].sum())
            lens_rows.append(row)
    write_jsonl(args.output / "planning_lens.jsonl", lens_rows)

    import numpy as np
    for offset in range(0, 8):
        rows = [r for r in lens_rows if r["offset"] == offset]
        if not rows:
            continue
        print(f"offset {offset}: n={len(rows)} attn {np.mean([r['head_attention_to_anchor'] for r in rows]):.3f} "
              f"mass@13 {np.mean([r['family_mass_layer_13'] for r in rows]):.3f} "
              f"mass@34 {np.mean([r['family_mass_layer_34'] for r in rows]):.3f}")
    print(f"Wrote planning analysis to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/gemma4_planning"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
