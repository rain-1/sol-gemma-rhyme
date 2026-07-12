"""Decompose Gemma 4 rhyme-scheme routing into query, key, value, and cue position.

The AABB- and ABAB-demonstration conditions have identical target stanzas and
equal-length prefixes. We transfer one state at a time from the donor scheme to
the destination and measure which cue L24H3 addresses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from rhyme_interp.dataset import build_open_cue_dataset
from rhyme_interp.manifest import write_manifest
from rhyme_interp.model import DEFAULT_REVISIONS, load_model
from rhyme_interp.rhyme import rhyme_token_ids
from run_gemma4_interpretability import (
    MODEL,
    anchor_positions,
    batch_inputs,
    write_jsonl,
)


LAYER, HEAD = 24, 3
CUES = ["cue_distance_1", "cue_distance_2", "cue_distance_3"]


@torch.inference_mode()
def capture(inputs, bundle):
    layers = bundle.model.model.language_model.layers
    residuals: list[torch.Tensor | None] = [None] * LAYER
    layer23_updates = {}
    handles = []
    for index in range(LAYER):
        def save(_module, _args, output, index=index):
            residuals[index] = output[:, -1].detach().clone()
        handles.append(layers[index].register_forward_hook(save))
    for name, module in [
        ("attention", layers[23].post_attention_layernorm),
        ("mlp", layers[23].post_feedforward_layernorm),
        ("per_layer_input", layers[23].post_per_layer_input_norm),
    ]:
        def save_update(_module, _args, output, name=name):
            layer23_updates[name] = output[:, -1].detach().clone()
        handles.append(module.register_forward_hook(save_update))
    output = bundle.model(
        **inputs,
        logits_to_keep=1,
        use_cache=False,
        output_attentions=True,
        return_shared_kv_states=True,
    )
    for handle in handles:
        handle.remove()
    key, value = output.shared_kv_states["full_attention"]
    return {
        "residuals": residuals,
        "layer23_updates": layer23_updates,
        "key": key.detach().clone(),
        "value": value.detach().clone(),
        "logits": output.logits[:, -1].float(),
        "attention": output.attentions[LAYER][:, HEAD, -1].detach(),
    }


def condition(scheme, bundle):
    examples = build_open_cue_dataset(scheme, demo_stanzas=2)
    prompts = [example.prompt for example in examples]
    inputs = batch_inputs(prompts, bundle)
    positions = {
        cue: anchor_positions(
            prompts,
            [getattr(example, cue) for example in examples],
            bundle,
            inputs["input_ids"].shape[1],
        )
        for cue in CUES
    }
    return {
        "examples": examples,
        "prompts": prompts,
        "inputs": inputs,
        "positions": positions,
        **capture(inputs, bundle),
    }


def measure(output, destination, bundle):
    probs = output.logits[:, -1].float().softmax(-1)
    attention = output.attentions[LAYER][:, HEAD, -1]
    rows = []
    for i, example in enumerate(destination["examples"]):
        row = {"id": example.id}
        for cue in CUES:
            word = getattr(example, cue)
            row[f"{cue}_mass"] = float(probs[i, bundle.rhyme_cache[word]].sum())
            row[f"{cue}_attention"] = float(attention[i, destination["positions"][cue][i]])
        rows.append(row)
    return rows


def run(args):
    bundle = load_model(
        MODEL,
        load_in_4bit=not args.bf16,
        attn_implementation="eager",
    )
    model = bundle.model
    layers = model.model.language_model.layers
    bundle.rhyme_cache = {}
    conditions = {}
    for scheme in ["aabb", "abab"]:
        examples = build_open_cue_dataset(scheme, demo_stanzas=2)
        for example in examples:
            for cue in CUES:
                word = getattr(example, cue)
                bundle.rhyme_cache.setdefault(word, rhyme_token_ids(word, bundle.token_words))
        conditions[scheme] = condition(scheme, bundle)
    assert conditions["aabb"]["inputs"]["input_ids"].shape == conditions["abab"]["inputs"]["input_ids"].shape

    rows = []
    batch = torch.arange(len(conditions["aabb"]["examples"]), device=bundle.device)
    for destination_name, source_name in [("aabb", "abab"), ("abab", "aabb")]:
        destination, source = conditions[destination_name], conditions[source_name]

        def record(intervention, output, **details):
            for row in measure(output, destination, bundle):
                rows.append({
                    "destination": destination_name,
                    "source": source_name,
                    "intervention": intervention,
                    **details,
                    **row,
                })

        # Baseline.
        with torch.inference_mode():
            output = model(
                **destination["inputs"], logits_to_keep=1, use_cache=False, output_attentions=True
            )
        record("none", output)
        del output

        # Query pathway: patch the final-position residual after each layer.
        for layer_index in range(LAYER):
            def query_hook(_module, _args, output, layer_index=layer_index):
                fixed = output.clone()
                fixed[:, -1] = source["residuals"][layer_index].to(fixed.dtype)
                return fixed

            handle = layers[layer_index].register_forward_hook(query_hook)
            with torch.inference_mode():
                output = model(
                    **destination["inputs"], logits_to_keep=1, use_cache=False,
                    output_attentions=True,
                )
            handle.remove()
            record("query", output, layer=layer_index)
            del output

        # Resolve the abrupt layer-23 query transition into its additive
        # attention, MLP, and per-layer-token-input updates.
        update_modules = {
            "attention": layers[23].post_attention_layernorm,
            "mlp": layers[23].post_feedforward_layernorm,
            "per_layer_input": layers[23].post_per_layer_input_norm,
        }
        update_sets = [
            ("attention",),
            ("mlp",),
            ("per_layer_input",),
            ("attention", "mlp"),
            ("attention", "mlp", "per_layer_input"),
        ]
        for selected in update_sets:
            handles = []
            for name in selected:
                def update_hook(_module, _args, output, name=name):
                    fixed = output.clone()
                    fixed[:, -1] = source["layer23_updates"][name].to(fixed.dtype)
                    return fixed
                handles.append(update_modules[name].register_forward_hook(update_hook))
            with torch.inference_mode():
                output = model(
                    **destination["inputs"], logits_to_keep=1, use_cache=False,
                    output_attentions=True,
                )
            for handle in handles:
                handle.remove()
            record("layer23_update", output, components="+".join(selected))
            del output

        # Memory pathway: split key/value and patch each cue alone, all cues,
        # or all target-stanza positions.
        all_cue_positions = torch.stack([destination["positions"][cue] for cue in CUES], dim=1)
        source_all_cue_positions = torch.stack([source["positions"][cue] for cue in CUES], dim=1)
        for part in ["key", "value", "both"]:
            for position_name in [*CUES, "all_cues"]:
                if position_name == "all_cues":
                    dest_positions = all_cue_positions
                    source_positions = source_all_cue_positions
                else:
                    dest_positions = destination["positions"][position_name][:, None]
                    source_positions = source["positions"][position_name][:, None]

                def memory_hook(_module, _args, kwargs, output):
                    key, value = kwargs["shared_kv_states"]["full_attention"]
                    key, value = key.clone(), value.clone()
                    for column in range(dest_positions.shape[1]):
                        dp = dest_positions[:, column]
                        sp = source_positions[:, column]
                        if part in {"key", "both"}:
                            key[batch, :, dp] = source["key"][batch, :, sp]
                        if part in {"value", "both"}:
                            value[batch, :, dp] = source["value"][batch, :, sp]
                    kwargs["shared_kv_states"]["full_attention"] = key, value
                    return output

                handle = layers[14].self_attn.register_forward_hook(memory_hook, with_kwargs=True)
                with torch.inference_mode():
                    output = model(
                        **destination["inputs"], logits_to_keep=1, use_cache=False,
                        output_attentions=True,
                    )
                handle.remove()
                record("memory", output, part=part, position=position_name)
                del output

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "routing_decomposition.jsonl", rows)
    write_manifest(
        args.output / "routing_decomposition.manifest.json",
        model=MODEL,
        revision=DEFAULT_REVISIONS[MODEL],
        precision="bf16" if args.bf16 else "nf4",
        seed=0,
        datasets=[],
    )
    print(f"Wrote {len(rows)} rows to {args.output / 'routing_decomposition.jsonl'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/gemma4_phase3"))
    parser.add_argument("--bf16", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
