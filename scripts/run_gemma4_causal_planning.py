"""Causally test when the rhyme constraint enters Gemma's incomplete final line.

For each natural prompt, a destination prompt changes only the earlier rhyme
anchor. We transfer source states at the anchor, the line-opening newline, and
successive final-line positions. If a rhyme plan exists at line start, patching
that state or its layer-14 shared memory should restore the source rhyme family.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rhyme_interp.dataset import build_elicitation_dataset
from rhyme_interp.manifest import write_manifest
from rhyme_interp.model import DEFAULT_REVISIONS, load_model
from rhyme_interp.rhyme import rhyme_token_ids
from run_gemma4_interpretability import (
    MODEL,
    anchor_positions,
    batch_inputs,
    forward_logits,
    make_counterfactual_prompts,
    metrics,
    write_jsonl,
)


def token_position_map(prompts, anchors, bundle, padded_length):
    anchor = anchor_positions(prompts, anchors, bundle, padded_length)
    result = {"anchor": anchor, "pre_anchor": anchor - 1}
    newline, first_word, middle, penultimate, final_input = [], [], [], [], []
    for prompt in prompts:
        ids = bundle.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        decoded = [bundle.tokenizer.decode([int(token)]) for token in ids]
        hits = [i for i, text in enumerate(decoded) if "\n" in text]
        start = hits[-1]
        shift = padded_length - len(ids)
        end = len(ids) - 1
        newline.append(start + shift)
        first_word.append(min(start + 1, end) + shift)
        middle.append((start + end) // 2 + shift)
        penultimate.append(max(start, end - 1) + shift)
        final_input.append(end + shift)
    for name, values in [
        ("line_start", newline),
        ("first_word", first_word),
        ("line_middle", middle),
        ("penultimate_input", penultimate),
        ("final_input", final_input),
    ]:
        result[name] = torch.tensor(values, device=bundle.device)
    return result


@torch.inference_mode()
def capture_source(inputs, bundle):
    layers = bundle.model.model.language_model.layers
    captured = {"residuals": [None] * 14}

    handles = []
    for layer_index in range(14):
        def save_residual(_module, _args, output, layer_index=layer_index):
            captured["residuals"][layer_index] = output.detach().clone()
        handles.append(layers[layer_index].register_forward_hook(save_residual))

    def save_l23(_module, _args, output):
        captured["l23_final"] = output[:, -1].detach().clone()

    handles.append(layers[23].register_forward_hook(save_l23))
    output = bundle.model(
        **inputs,
        logits_to_keep=1,
        use_cache=False,
        return_shared_kv_states=True,
    )
    for handle in handles:
        handle.remove()
    for memory in ["sliding_attention", "full_attention"]:
        key, value = output.shared_kv_states[memory]
        captured[f"{memory}_key"] = key.detach().clone()
        captured[f"{memory}_value"] = value.detach().clone()
    captured["logits"] = output.logits[:, -1].float()
    return captured


def run(args):
    bundle = load_model(
        MODEL,
        load_in_4bit=not args.bf16,
        attn_implementation="eager",
    )
    model = bundle.model
    layers = model.model.language_model.layers
    examples = build_elicitation_dataset("rhyming")
    source_prompts = [example.prompt for example in examples]
    source_anchors = [example.anchor for example in examples]
    targets = [example.target for example in examples]
    destination_prompts, destination_anchors = make_counterfactual_prompts(examples)
    source_inputs = batch_inputs(source_prompts, bundle)
    destination_inputs = batch_inputs(destination_prompts, bundle)
    assert source_inputs["input_ids"].shape == destination_inputs["input_ids"].shape
    padded = source_inputs["input_ids"].shape[1]
    source_positions = token_position_map(source_prompts, source_anchors, bundle, padded)
    destination_positions = token_position_map(
        destination_prompts, destination_anchors, bundle, padded
    )
    bundle.rhyme_cache = {
        anchor: rhyme_token_ids(anchor, bundle.token_words)
        for anchor in set(source_anchors + destination_anchors)
    }
    source = capture_source(source_inputs, bundle)
    destination_logits = forward_logits(destination_inputs, bundle)
    source_metrics = metrics(source["logits"], source_anchors, targets, bundle)
    destination_metrics = metrics(destination_logits, source_anchors, targets, bundle)
    batch = torch.arange(len(examples), device=bundle.device)
    rows = []

    def record(kind, position, patched_logits, part=None, layer=None, memory=None):
        patched = metrics(patched_logits, source_anchors, targets, bundle)
        for i, anchor in enumerate(source_anchors):
            denominator = source_metrics[i]["rhyme_mass"] - destination_metrics[i]["rhyme_mass"]
            rows.append({
                "kind": kind,
                "part": part,
                "layer": layer,
                "memory": memory,
                "position": position,
                "anchor": anchor,
                "counter_anchor": destination_anchors[i],
                "source_mass": source_metrics[i]["rhyme_mass"],
                "destination_mass": destination_metrics[i]["rhyme_mass"],
                "patched_mass": patched[i]["rhyme_mass"],
                "recovery": (
                    (patched[i]["rhyme_mass"] - destination_metrics[i]["rhyme_mass"])
                    / denominator
                    if abs(denominator) > 1e-6 else None
                ),
            })

    # Scan residual transfer through every layer that can still alter either
    # shared memory. This includes the sliding-memory boundary at layer 13 and
    # the full-memory boundary at layer 14.
    for layer_index in range(14):
        for position_name, destination_position in destination_positions.items():
            source_position = source_positions[position_name]

            def residual_hook(_module, _args, output, layer_index=layer_index):
                fixed = output.clone()
                fixed[batch, destination_position] = source["residuals"][layer_index][
                    batch, source_position
                ].to(fixed.dtype)
                return fixed

            handle = layers[layer_index].register_forward_hook(residual_hook)
            patched_logits = forward_logits(destination_inputs, bundle)
            handle.remove()
            record("residual", position_name, patched_logits, layer=layer_index)

    # Transfer the actual shared memory at one position, splitting address
    # (key) from content (value).
    for memory, storing_layer in [("sliding_attention", 13), ("full_attention", 14)]:
        for part in ["key", "value", "both"]:
            for position_name, destination_position in destination_positions.items():
                source_position = source_positions[position_name]

                def memory_hook(_module, _args, kwargs, output, memory=memory, part=part):
                    key, value = kwargs["shared_kv_states"][memory]
                    key, value = key.clone(), value.clone()
                    if part in {"key", "both"}:
                        key[batch, :, destination_position] = source[f"{memory}_key"][
                            batch, :, source_position
                        ]
                    if part in {"value", "both"}:
                        value[batch, :, destination_position] = source[f"{memory}_value"][
                            batch, :, source_position
                        ]
                    kwargs["shared_kv_states"][memory] = key, value
                    return output

                handle = layers[storing_layer].self_attn.register_forward_hook(
                    memory_hook, with_kwargs=True
                )
                patched_logits = forward_logits(destination_inputs, bundle)
                handle.remove()
                record(
                    "shared_memory", position_name, patched_logits,
                    part=part, layer=storing_layer, memory=memory,
                )

    # Positive upper-bound: replace the final query state immediately before
    # the retrieval layer.
    def query_hook(_module, _args, output):
        fixed = output.clone()
        fixed[:, -1] = source["l23_final"].to(fixed.dtype)
        return fixed

    handle = layers[23].register_forward_hook(query_hook)
    patched_logits = forward_logits(destination_inputs, bundle)
    handle.remove()
    record("l23_query", "final_input", patched_logits)

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "causal_planning.jsonl", rows)
    write_manifest(
        args.output / "causal_planning.manifest.json",
        model=MODEL,
        revision=DEFAULT_REVISIONS[MODEL],
        precision="bf16" if args.bf16 else "nf4",
        seed=0,
        datasets=[],
    )
    print(f"Wrote {len(rows)} rows to {args.output / 'causal_planning.jsonl'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/gemma4_phase3"))
    parser.add_argument("--bf16", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
