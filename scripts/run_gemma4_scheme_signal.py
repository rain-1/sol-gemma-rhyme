"""Where does the scheme live: in the retrieval query or in the memory?

The open-cue prompts under AABB and ABAB demonstrations share an identical
target stanza and equally long demonstration prefixes (the same lines,
reordered), so activations align position-for-position. The two conditions
retrieve different cue lines (distance 1 versus distance 2). Swapping parts of
the computation between conditions localizes the routing signal:

- **query**: replace the final-token residual entering layer 24 with the other
  condition's. If attention and behavior follow the donor scheme, the routing
  decision is carried by the final-token state.
- **memory**: replace the layer-14 shared K/V at every target-stanza position
  with the other condition's. If routing follows the donor, the demonstrations
  instead mark stanza line-endings as addressable in memory.
- **both**: the two together (consistency check).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rhyme_interp.dataset import build_open_cue_dataset
from rhyme_interp.model import load_model
from rhyme_interp.rhyme import rhyme_token_ids

from run_gemma4_interpretability import (
    MODEL,
    anchor_positions,
    batch_inputs,
    forward_logits,
    write_jsonl,
)

CANDIDATE_LAYER, CANDIDATE_HEAD = 24, 3
CUES = ["cue_distance_1", "cue_distance_2", "cue_distance_3"]


@torch.inference_mode()
def capture_condition(inputs, bundle):
    """Layer-23 final-position output, shared KV, logits."""
    layers = bundle.model.model.language_model.layers
    captured = {}

    def save(_module, _args, output):
        captured["residual"] = output[:, -1].detach().clone()

    handle = layers[CANDIDATE_LAYER - 1].register_forward_hook(save)
    output = bundle.model(
        **inputs, logits_to_keep=1, use_cache=False, return_shared_kv_states=True
    )
    handle.remove()
    key, value = output.shared_kv_states["full_attention"]
    return captured["residual"], (key.detach().clone(), value.detach().clone()), \
        output.logits[:, -1].float()


def run(args):
    bundle = load_model(MODEL, load_in_4bit=True, attn_implementation="eager")
    model = bundle.model
    layers = model.model.language_model.layers
    bundle.rhyme_cache = {}

    conditions = {}
    for scheme in ["aabb", "abab"]:
        examples = build_open_cue_dataset(scheme, demo_stanzas=2)
        prompts = [e.prompt for e in examples]
        for e in examples:
            for cue in CUES:
                word = getattr(e, cue)
                if word not in bundle.rhyme_cache:
                    bundle.rhyme_cache[word] = rhyme_token_ids(word, bundle.token_words)
        inputs = batch_inputs(prompts, bundle)
        residual, kv, logits = capture_condition(inputs, bundle)
        conditions[scheme] = {
            "examples": examples, "prompts": prompts, "inputs": inputs,
            "residual": residual, "kv": kv, "logits": logits,
        }
    aabb_len = conditions["aabb"]["inputs"]["input_ids"].shape[1]
    abab_len = conditions["abab"]["inputs"]["input_ids"].shape[1]
    assert aabb_len == abab_len, "reordered demos should not change token count"
    # Stanza tokens: everything after the fixed demonstration prefix. The
    # prefix is identical lines reordered, so per-example stanza spans align.
    prefix_tokens = bundle.tokenizer(
        conditions["aabb"]["prompts"][0].rsplit("\n\n", 1)[0] + "\n\n",
        return_tensors="pt",
    )["input_ids"].shape[1]

    def stanza_slice(prompt_index, condition):
        ids = bundle.tokenizer(condition["prompts"][prompt_index], return_tensors="pt")["input_ids"][0]
        start = aabb_len - (len(ids) - prefix_tokens)
        return start  # stanza spans [start, padded_length)

    rows = []
    for dest_scheme, source_scheme in [("aabb", "abab"), ("abab", "aabb")]:
        dest = conditions[dest_scheme]
        source = conditions[source_scheme]
        examples = dest["examples"]
        prompts = dest["prompts"]
        inputs = dest["inputs"]
        positions = {
            cue: anchor_positions(prompts, [getattr(e, cue) for e in examples],
                                  bundle, aabb_len)
            for cue in CUES
        }
        starts = [stanza_slice(i, dest) for i in range(len(examples))]

        def query_hook(_module, _args, output):
            fixed = output.clone()
            fixed[:, -1] = source["residual"].to(fixed.dtype)
            return fixed

        def kv_hook(_module, _args, kwargs, output):
            key, value = kwargs["shared_kv_states"]["full_attention"]
            key, value = key.clone(), value.clone()
            source_key, source_value = source["kv"]
            for i, start in enumerate(starts):
                key[i, :, start:] = source_key[i, :, start:]
                value[i, :, start:] = source_value[i, :, start:]
            kwargs["shared_kv_states"]["full_attention"] = (key, value)
            return output

        for intervention in ["none", "query", "memory", "both"]:
            handles = []
            if intervention in {"query", "both"}:
                handles.append(layers[CANDIDATE_LAYER - 1].register_forward_hook(query_hook))
            if intervention in {"memory", "both"}:
                handles.append(layers[14].self_attn.register_forward_hook(kv_hook, with_kwargs=True))
            with torch.inference_mode():
                outputs = model(**inputs, logits_to_keep=1, use_cache=False,
                                output_attentions=True)
            for handle in handles:
                handle.remove()
            logits = outputs.logits[:, -1].float()
            probs = logits.softmax(-1)
            attention = outputs.attentions[CANDIDATE_LAYER][:, CANDIDATE_HEAD]
            for i, example in enumerate(examples):
                row = {
                    "dest_scheme": dest_scheme,
                    "source_scheme": source_scheme,
                    "intervention": intervention,
                    "id": example.id,
                }
                for cue in CUES:
                    word = getattr(example, cue)
                    row[f"{cue}_mass"] = float(probs[i, bundle.rhyme_cache[word]].sum())
                    row[f"{cue}_attention"] = float(attention[i, -1, positions[cue][i]])
                rows.append(row)
            del outputs
            import numpy as np
            sub = [r for r in rows if r["dest_scheme"] == dest_scheme
                   and r["intervention"] == intervention]
            print(f"{dest_scheme} <- {source_scheme} [{intervention}]: " + " ".join(
                f"d{d} attn {np.mean([r[f'cue_distance_{d}_attention'] for r in sub]):.3f}"
                f"/mass {np.mean([r[f'cue_distance_{d}_mass'] for r in sub]):.3f}"
                for d in (1, 2)
            ))
    write_jsonl(args.output / "scheme_signal.jsonl", rows)
    print(f"Wrote scheme-signal localization to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/gemma4_schemes"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
