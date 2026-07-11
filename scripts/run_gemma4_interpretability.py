"""Run causal rhyme analyses on quantized Gemma 4 E2B.

The script deliberately keeps the model loaded once. Outputs are JSON files in
one artifact directory so every table in the report can be recomputed.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import re

import pandas as pd
import torch
import torch.nn.functional as F

from rhyme_interp.dataset import COUPLETS, build_elicitation_dataset
from rhyme_interp.model import load_model, target_token_id
from rhyme_interp.rhyme import rhyme_token_ids, rhymes


MODEL = "google/gemma-4-E2B"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def batch_inputs(prompts, bundle):
    tokenizer = bundle.tokenizer
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    result = tokenizer(prompts, padding=True, return_tensors="pt").to(bundle.device)
    tokenizer.padding_side = old_side
    return result


def anchor_positions(prompts, anchors, bundle, padded_length):
    """Find the final occurrence of each anchor, accounting for left padding."""
    positions = []
    for prompt, anchor in zip(prompts, anchors):
        ids = bundle.tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        anchor_id = target_token_id(bundle.tokenizer, anchor)
        hits = (ids == anchor_id).nonzero().flatten()
        if not len(hits):
            raise ValueError(f"Could not find anchor {anchor!r}")
        positions.append(int(hits[-1]) + padded_length - len(ids))
    return torch.tensor(positions, device=bundle.device)


def metrics(logits, anchors, targets, bundle):
    probs = logits.float().softmax(-1)
    rows = []
    for i, (anchor, target) in enumerate(zip(anchors, targets)):
        rhyme_ids = bundle.rhyme_cache[anchor]
        top_id = int(logits[i].argmax())
        top_word = bundle.token_words.get(top_id, bundle.tokenizer.decode([top_id]))
        target_id = target_token_id(bundle.tokenizer, target)
        rows.append({
            "anchor": anchor,
            "target": target,
            "top1": top_word,
            "top1_rhymes": rhymes(anchor, top_word),
            "rhyme_mass": float(probs[i, rhyme_ids].sum()),
            "target_probability": float(probs[i, target_id]),
            "target_logit": float(logits[i, target_id]),
        })
    return rows


@torch.inference_mode()
def forward_logits(inputs, bundle):
    return bundle.model(**inputs, logits_to_keep=1, use_cache=False).logits[:, -1].float()


@contextmanager
def zero_module_final(module):
    def hook(_module, _args, output):
        if isinstance(output, tuple):
            hidden = output[0].clone()
            hidden[:, -1] = 0
            return (hidden, *output[1:])
        hidden = output.clone()
        hidden[:, -1] = 0
        return hidden

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def zero_head_final(layer, head, head_dim):
    def hook(_module, args):
        hidden = args[0].clone()
        hidden[:, -1, head * head_dim : (head + 1) * head_dim] = 0
        return (hidden, *args[1:])

    handle = layer.self_attn.o_proj.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def summarize_intervention(kind, layer, changed, baseline):
    rows = []
    for after, before in zip(changed, baseline):
        rows.append({
            "kind": kind,
            "layer": layer,
            "anchor": before["anchor"],
            "baseline_rhyme_mass": before["rhyme_mass"],
            "rhyme_mass": after["rhyme_mass"],
            "delta_rhyme_mass": after["rhyme_mass"] - before["rhyme_mass"],
            "baseline_target_probability": before["target_probability"],
            "target_probability": after["target_probability"],
            "delta_target_probability": after["target_probability"] - before["target_probability"],
            "top1_rhymes": after["top1_rhymes"],
        })
    return rows


def capture_layer_states(inputs, positions, bundle):
    layers = bundle.model.model.language_model.layers
    captured = [None] * len(layers)
    batch = torch.arange(inputs["input_ids"].shape[0], device=bundle.device)
    handles = []
    for index, layer in enumerate(layers):
        def hook(_module, _args, output, index=index):
            captured[index] = output[batch, positions].detach().clone()
        handles.append(layer.register_forward_hook(hook))
    forward_logits(inputs, bundle)
    for handle in handles:
        handle.remove()
    return captured


def make_counterfactual_prompts(examples):
    anchors = [example.anchor for example in examples]
    prompts = []
    counter_anchors = []
    for i, example in enumerate(examples):
        replacement = anchors[(i + 1) % len(anchors)]
        head, match, tail = example.prompt.rpartition(example.anchor)
        if not match:
            raise ValueError(example.anchor)
        prompts.append(head + replacement + tail)
        counter_anchors.append(replacement)
    return prompts, counter_anchors


def run(args):
    bundle = load_model(MODEL, load_in_4bit=True)
    model = bundle.model
    layers = model.model.language_model.layers
    examples = build_elicitation_dataset("rhyming")
    shuffled = build_elicitation_dataset("shuffled")
    anchors = [example.anchor for example in examples]
    targets = [example.target for example in examples]
    prompts = [example.prompt for example in examples]
    shuffled_prompts = [example.prompt for example in shuffled]
    # Gemma has a 262k vocabulary. Pronunciation-family construction is pure
    # Python, so cache it once rather than repeating it after every intervention.
    bundle.rhyme_cache = {anchor: rhyme_token_ids(anchor, bundle.token_words) for anchor in set(anchors)}
    inputs = batch_inputs(prompts, bundle)
    shuffled_inputs = batch_inputs(shuffled_prompts, bundle)

    baseline_logits = forward_logits(inputs, bundle)
    baseline = metrics(baseline_logits, anchors, targets, bundle)
    shuffled_logits = forward_logits(shuffled_inputs, bundle)
    shuffled_metrics = metrics(shuffled_logits, anchors, targets, bundle)
    write_jsonl(args.output / "baseline.jsonl", [
        {**row, "condition": "rhyming"} for row in baseline
    ] + [{**row, "condition": "shuffled"} for row in shuffled_metrics])

    # Layer-local final-position ablations distinguish attention, MLP, and the
    # architecture's unusual per-layer token-input pathway.
    ablations = []
    for layer_index, layer in enumerate(layers):
        for kind, module in [
            ("attention_update", layer.post_attention_layernorm),
            ("mlp_update", layer.post_feedforward_layernorm),
            ("per_layer_input_update", layer.post_per_layer_input_norm),
        ]:
            with zero_module_final(module):
                changed = metrics(forward_logits(inputs, bundle), anchors, targets, bundle)
            ablations.extend(summarize_intervention(kind, layer_index, changed, baseline))
    write_jsonl(args.output / "module_ablation.jsonl", ablations)

    # Causal head ablation. Run all 280 heads, not only visually attractive ones.
    head_rows = []
    for layer_index, layer in enumerate(layers):
        for head in range(model.config.text_config.num_attention_heads):
            with zero_head_final(layer, head, layer.self_attn.head_dim):
                changed = metrics(forward_logits(inputs, bundle), anchors, targets, bundle)
            for row in summarize_intervention("head", layer_index, changed, baseline):
                row["head"] = head
                head_rows.append(row)
    write_jsonl(args.output / "head_ablation.jsonl", head_rows)

    # Attention patterns: final query -> target anchor, aggregated only after all
    # examples are collected. Attention is descriptive; ablation is causal.
    model.set_attn_implementation("eager")
    with torch.inference_mode():
        outputs = model(**inputs, logits_to_keep=1, use_cache=False, output_attentions=True)
    anchor_pos = anchor_positions(prompts, anchors, bundle, inputs["input_ids"].shape[1])
    attention_rows = []
    for layer_index, attn in enumerate(outputs.attentions):
        for example_index, anchor in enumerate(anchors):
            for head in range(attn.shape[1]):
                attention_rows.append({
                    "layer": layer_index,
                    "head": head,
                    "anchor": anchor,
                    "anchor_attention": float(attn[example_index, head, -1, anchor_pos[example_index]]),
                    "attention_entropy": float(-(attn[example_index, head, -1].float().clamp_min(1e-12) *
                                                attn[example_index, head, -1].float().clamp_min(1e-12).log()).sum()),
                })
    write_jsonl(args.output / "attention_patterns.jsonl", attention_rows)
    model.set_attn_implementation("sdpa")

    # Logit lens at the final position for ordered and shuffled contexts. The
    # comparison is the reference rhyme against the strongest baseline non-rhyme.
    source_states = capture_layer_states(inputs, torch.full((len(examples),), inputs["input_ids"].shape[1] - 1,
                                                             device=bundle.device), bundle)
    shuffled_states = capture_layer_states(
        shuffled_inputs,
        torch.full((len(examples),), shuffled_inputs["input_ids"].shape[1] - 1, device=bundle.device),
        bundle,
    )
    eligible_ids = list(bundle.token_words)
    competitors = []
    for i, anchor in enumerate(anchors):
        rhyme_set = set(bundle.rhyme_cache[anchor])
        candidates = [token_id for token_id in eligible_ids if token_id not in rhyme_set]
        values = baseline_logits[i, candidates]
        competitors.append(candidates[int(values.argmax())])
    norm = model.model.language_model.norm
    softcap = model.config.text_config.final_logit_softcapping
    lens_rows = []
    for condition, states in [("rhyming", source_states), ("shuffled", shuffled_states)]:
        for layer_index, states_at_layer in enumerate(states):
            hidden = norm(states_at_layer)
            for i, (anchor, target) in enumerate(zip(anchors, targets)):
                target_id = target_token_id(bundle.tokenizer, target)
                pair = torch.tensor([target_id, competitors[i]], device=bundle.device)
                # Computing only the two selected rows avoids a full 262k-token
                # unembedding for every layer/example.
                pair_logits = F.linear(hidden[i : i + 1], model.lm_head.weight[pair]).float()[0]
                if softcap:
                    pair_logits = softcap * torch.tanh(pair_logits / softcap)
                lens_rows.append({
                    "condition": condition,
                    "layer": layer_index,
                    "anchor": anchor,
                    "target": target,
                    "competitor": bundle.token_words[competitors[i]],
                    "target_logit": float(pair_logits[0]),
                    "competitor_logit": float(pair_logits[1]),
                    "target_minus_competitor": float(pair_logits[0] - pair_logits[1]),
                })
    write_jsonl(args.output / "logit_lens.jsonl", lens_rows)

    # Exact anchor-token substitutions keep every other character fixed.
    counter_prompts, counter_anchors = make_counterfactual_prompts(examples)
    counter_inputs = batch_inputs(counter_prompts, bundle)
    counter_logits = forward_logits(counter_inputs, bundle)
    original_family_on_counter = metrics(counter_logits, anchors, targets, bundle)
    counter_family_on_counter = metrics(counter_logits, counter_anchors, targets, bundle)
    counter_rows = []
    for i, anchor in enumerate(anchors):
        counter_rows.append({
            "anchor": anchor,
            "counter_anchor": counter_anchors[i],
            "original_prompt_rhyme_mass": baseline[i]["rhyme_mass"],
            "counter_prompt_original_family_mass": original_family_on_counter[i]["rhyme_mass"],
            "counter_prompt_counter_family_mass": counter_family_on_counter[i]["rhyme_mass"],
            "original_top1": baseline[i]["top1"],
            "counter_top1": original_family_on_counter[i]["top1"],
            "counter_top1_rhymes_original": original_family_on_counter[i]["top1_rhymes"],
            "counter_top1_rhymes_counter": counter_family_on_counter[i]["top1_rhymes"],
        })
    write_jsonl(args.output / "anchor_counterfactual.jsonl", counter_rows)

    # Patch original anchor states into counterfactual prompts, one layer at a
    # time and cumulatively from a layer onward (important because Gemma injects
    # token-specific per-layer inputs at every block).
    original_anchor_pos = anchor_positions(prompts, anchors, bundle, inputs["input_ids"].shape[1])
    counter_anchor_pos = anchor_positions(counter_prompts, counter_anchors, bundle, counter_inputs["input_ids"].shape[1])
    original_anchor_states = capture_layer_states(inputs, original_anchor_pos, bundle)
    batch_index = torch.arange(len(examples), device=bundle.device)
    patch_rows = []
    for start_layer in range(len(layers)):
        for mode in ["single", "cumulative"]:
            handles = []
            selected = [start_layer] if mode == "single" else list(range(start_layer, len(layers)))
            for layer_index in selected:
                def hook(_module, _args, output, layer_index=layer_index):
                    hidden = output.clone()
                    hidden[batch_index, counter_anchor_pos] = original_anchor_states[layer_index]
                    return hidden
                handles.append(layers[layer_index].register_forward_hook(hook))
            patched = metrics(forward_logits(counter_inputs, bundle), anchors, targets, bundle)
            for handle in handles:
                handle.remove()
            for i, row in enumerate(patched):
                patch_rows.append({
                    "mode": mode,
                    "start_layer": start_layer,
                    "anchor": anchors[i],
                    "counter_anchor": counter_anchors[i],
                    "baseline_counter_original_family_mass": original_family_on_counter[i]["rhyme_mass"],
                    "patched_original_family_mass": row["rhyme_mass"],
                    "delta_original_family_mass": row["rhyme_mass"] - original_family_on_counter[i]["rhyme_mass"],
                })
    write_jsonl(args.output / "anchor_patching.jsonl", patch_rows)

    print(f"Wrote Gemma 4 causal analysis to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/gemma4_interpretability"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
