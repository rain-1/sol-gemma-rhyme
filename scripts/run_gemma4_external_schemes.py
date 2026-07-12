"""Replicate Gemma 4 rhyme retrieval on independent AABB/ABAB/ABBA poems.

The final word of line four is removed.  No demonstrations or instructions are
prepended: the model sees only the three complete lines and the incomplete
fourth line.  For every poem this script measures clean behavior, L24H3's
attention to each completed line ending, and behavior after ablating L24H3 at
the prediction position.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rhyme_interp.model import load_model, target_token_id
from rhyme_interp.model import DEFAULT_REVISIONS
from rhyme_interp.manifest import write_manifest
from rhyme_interp.rhyme import rhyme_token_ids, rhymes
from run_gemma4_interpretability import (
    MODEL, anchor_positions, batch_inputs, forward_logits, write_jsonl,
    zero_head_final,
)

CANDIDATE_LAYER, CANDIDATE_HEAD = 24, 3
CUE_LINE = {"aabb": 2, "abab": 1, "abba": 0}
CLOSED_LINE = {"aabb": 0, "abab": 0, "abba": 1}


def load_poems(path: Path) -> list[dict]:
    poems = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for i, poem in enumerate(poems):
        scheme = poem.get("scheme")
        lines, endings = poem.get("lines", []), poem.get("endings", [])
        if scheme not in CUE_LINE or len(lines) != 4 or len(endings) != 4:
            raise ValueError(f"Malformed external-scheme row {i}")
        actual = [line.rstrip(".,;:!?\"'").rsplit(" ", 1)[-1].lower() for line in lines]
        if actual != endings:
            raise ValueError(f"Ending mismatch in row {i}: {actual} != {endings}")
        cue = endings[CUE_LINE[scheme]]
        closed = endings[CLOSED_LINE[scheme]]
        if not rhymes(cue, endings[3]):
            raise ValueError(f"Final word does not rhyme with cue in row {i}")
        closed_partner = {"aabb": 1, "abab": 2, "abba": 2}[scheme]
        if not rhymes(closed, endings[closed_partner]):
            raise ValueError(f"Closed pair does not rhyme in row {i}")
        if rhymes(cue, closed):
            raise ValueError(f"Rhyme families overlap in row {i}")
    return poems


def prompt_for(poem: dict) -> str:
    prefix, separator, _ = poem["lines"][3].rpartition(" ")
    if not separator:
        raise ValueError("Final line has no removable final word")
    return "\n".join(poem["lines"][:3] + [prefix])


def behavioral_rows(logits, poems, bundle, condition: str) -> list[dict]:
    probs = logits.float().softmax(-1)
    rows = []
    for i, poem in enumerate(poems):
        scheme = poem["scheme"]
        cue = poem["endings"][CUE_LINE[scheme]]
        closed = poem["endings"][CLOSED_LINE[scheme]]
        top_id = int(logits[i].argmax())
        top_word = bundle.token_words.get(top_id, bundle.tokenizer.decode([top_id]).strip())
        target_id = target_token_id(bundle.tokenizer, poem["endings"][3])
        rows.append({
            "id": poem["id"], "scheme": scheme, "condition": condition,
            "cue_word": cue, "closed_word": closed,
            "target_word": poem["endings"][3], "top1": top_word,
            "top1_rhymes_cue": rhymes(cue, top_word),
            "top1_rhymes_closed": rhymes(closed, top_word),
            "cue_family_mass": float(probs[i, bundle.rhyme_cache[cue]].sum()),
            "closed_family_mass": float(probs[i, bundle.rhyme_cache[closed]].sum()),
            "target_probability": float(probs[i, target_id]) if target_id is not None else None,
        })
    return rows


def run(args):
    poems = load_poems(args.data)
    bundle = load_model(MODEL, load_in_4bit=not args.bf16, attn_implementation="eager")
    usable = [p for p in poems if all(
        target_token_id(bundle.tokenizer, word) is not None for word in p["endings"]
    )]
    print(f"usable: {len(usable)} / {len(poems)}")
    bundle.rhyme_cache = {}
    for poem in usable:
        for index in (CUE_LINE[poem["scheme"]], CLOSED_LINE[poem["scheme"]]):
            word = poem["endings"][index]
            bundle.rhyme_cache.setdefault(word, rhyme_token_ids(word, bundle.token_words))

    prompts = [prompt_for(p) for p in usable]
    inputs = batch_inputs(prompts, bundle)
    clean = forward_logits(inputs, bundle)
    behavior = behavioral_rows(clean, usable, bundle, "clean")

    with torch.inference_mode():
        outputs = bundle.model(**inputs, logits_to_keep=1, use_cache=False, output_attentions=True)
    head_attention = outputs.attentions[CANDIDATE_LAYER][:, CANDIDATE_HEAD, -1]
    positions = {
        line: anchor_positions(prompts, [p["endings"][line] for p in usable], bundle,
                               inputs["input_ids"].shape[1])
        for line in range(3)
    }
    attention = [{
        "id": poem["id"], "scheme": poem["scheme"],
        **{f"attention_line{line + 1}": float(head_attention[i, positions[line][i]])
           for line in range(3)},
        "cue_line": CUE_LINE[poem["scheme"]] + 1,
        "attention_cue": float(head_attention[i, positions[CUE_LINE[poem["scheme"]]][i]]),
    } for i, poem in enumerate(usable)]
    del outputs

    layer = bundle.model.model.language_model.layers[CANDIDATE_LAYER]
    with zero_head_final(layer, CANDIDATE_HEAD, layer.self_attn.head_dim):
        ablated = forward_logits(inputs, bundle)
    behavior.extend(behavioral_rows(ablated, usable, bundle, "l24h3_ablated"))

    write_jsonl(args.output / "external_scheme_behavior.jsonl", behavior)
    write_jsonl(args.output / "external_scheme_attention.jsonl", attention)
    write_manifest(
        args.output / "external_schemes.manifest.json", model=MODEL,
        revision=DEFAULT_REVISIONS.get(MODEL), precision="bf16" if args.bf16 else "nf4",
        seed=None, datasets=[args.data],
    )
    for scheme in CUE_LINE:
        clean_rows = [r for r in behavior if r["scheme"] == scheme and r["condition"] == "clean"]
        ablated_rows = [r for r in behavior if r["scheme"] == scheme and r["condition"] == "l24h3_ablated"]
        attn_rows = [r for r in attention if r["scheme"] == scheme]
        print(f"{scheme.upper()} n={len(clean_rows)} top1={np.mean([r['top1_rhymes_cue'] for r in clean_rows]):.3f} "
              f"mass={np.mean([r['cue_family_mass'] for r in clean_rows]):.3f} "
              f"ablated={np.mean([r['cue_family_mass'] for r in ablated_rows]):.3f} "
              f"cue-attn={np.mean([r['attention_cue'] for r in attn_rows]):.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/external_scheme_quatrains.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/gemma4_external_schemes"))
    parser.add_argument("--bf16", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
