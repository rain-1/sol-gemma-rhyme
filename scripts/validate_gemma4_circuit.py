"""Targeted validation and falsification tests for the Gemma 4 rhyme circuit."""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from rhyme_interp.dataset import RHYME_DEMONSTRATION_LINES, build_elicitation_dataset
from rhyme_interp.model import load_model, target_token_id
from rhyme_interp.rhyme import rhyme_token_ids

from run_gemma4_interpretability import (
    MODEL,
    anchor_positions,
    batch_inputs,
    forward_logits,
    metrics,
    write_jsonl,
    zero_head_final,
    zero_module_final,
)


CONTROLLED_PAIRS = [
    # anchor A, anchor B, fixed final prefix, candidate A, candidate B
    ("moon", "long", "The old musician played a", "tune", "song"),
    ("rain", "star", "Beyond the fields there passed a", "train", "car"),
    ("snow", "night", "The cottage windows cast a", "glow", "light"),
    ("face", "roam", "He knew that he had found his", "place", "home"),
    ("deep", "west", "The tired village settled down to", "sleep", "rest"),
]

HOMOPHONES = [
    ("sea", "see", "free"),
    ("night", "knight", "light"),
    ("blue", "blew", "true"),
    ("air", "heir", "care"),
    ("road", "rode", "load"),
    ("right", "write", "night"),
    ("rain", "reign", "train"),
]

ORTHOGRAPHIC_FOILS = [
    ("love", "move", "dove", "prove"),
    ("cough", "though", "off", "glow"),
    ("food", "good", "mood", "wood"),
    ("heard", "beard", "word", "feared"),
    ("pint", "mint", "night", "hint"),
]


def controlled_prompt(anchor, final_prefix):
    demos = "\n".join(RHYME_DEMONSTRATION_LINES)
    return f"{demos}\nThe final word upon the page was {anchor}\n{final_prefix}"


def candidate_difference(logits, candidate_a, candidate_b, bundle):
    a = target_token_id(bundle.tokenizer, candidate_a)
    b = target_token_id(bundle.tokenizer, candidate_b)
    return float(logits[a] - logits[b])


def capture_head_input(inputs, layer, bundle):
    captured = {}

    def hook(_module, args):
        captured["value"] = args[0][:, -1].detach().clone()

    handle = layer.self_attn.o_proj.register_forward_pre_hook(hook)
    forward_logits(inputs, bundle)
    handle.remove()
    return captured["value"]


@contextmanager
def patch_head_final(layer, head, source, source_order=None):
    head_dim = layer.self_attn.head_dim

    def hook(_module, args):
        hidden = args[0].clone()
        values = source if source_order is None else source[source_order]
        hidden[:, -1, head * head_dim : (head + 1) * head_dim] = values[:, head * head_dim : (head + 1) * head_dim]
        return (hidden, *args[1:])

    handle = layer.self_attn.o_proj.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def run(output: Path):
    bundle = load_model(MODEL, load_in_4bit=True, attn_implementation="eager")
    model = bundle.model
    layers = model.model.language_model.layers
    candidate_layer, candidate_head = 24, 3
    candidate = layers[candidate_layer]

    # Necessity and selectivity across rhyme, shuffled, plain, and target-line-only
    # contexts. KL measures generic distribution damage alongside rhyme loss.
    condition_examples = {name: build_elicitation_dataset(name) for name in ["rhyming", "shuffled", "plain"]}
    condition_prompts = {name: [example.prompt for example in examples]
                         for name, examples in condition_examples.items()}
    condition_prompts["line_only"] = [example.final_prefix for example in condition_examples["plain"]]
    natural_rows = []
    for condition, prompts in condition_prompts.items():
        examples = condition_examples.get(condition, condition_examples["plain"])
        anchors = [example.anchor for example in examples]
        targets = [example.target for example in examples]
        bundle.rhyme_cache = {a: rhyme_token_ids(a, bundle.token_words) for a in set(anchors)}
        inputs = batch_inputs(prompts, bundle)
        clean_logits = forward_logits(inputs, bundle)
        clean = metrics(clean_logits, anchors, targets, bundle)
        with zero_head_final(candidate, candidate_head, candidate.self_attn.head_dim):
            ablated_logits = forward_logits(inputs, bundle)
        ablated = metrics(ablated_logits, anchors, targets, bundle)
        kl = (clean_logits.float().softmax(-1) *
              (clean_logits.float().log_softmax(-1) - ablated_logits.float().log_softmax(-1))).sum(-1)
        for i, anchor in enumerate(anchors):
            natural_rows.append({
                "condition": condition,
                "anchor": anchor,
                "clean_rhyme_mass": clean[i]["rhyme_mass"],
                "ablated_rhyme_mass": ablated[i]["rhyme_mass"],
                "delta_rhyme_mass": ablated[i]["rhyme_mass"] - clean[i]["rhyme_mass"],
                "clean_top1_rhymes": clean[i]["top1_rhymes"],
                "ablated_top1_rhymes": ablated[i]["top1_rhymes"],
                "kl_clean_to_ablated": float(kl[i]),
            })
    write_jsonl(output / "candidate_specificity.jsonl", natural_rows)

    # Exact paired candidate prompts and causal transfer of only L24H3's output.
    pair_rows = []
    source_prompts = [controlled_prompt(a, prefix) for a, _b, prefix, _ca, _cb in CONTROLLED_PAIRS]
    dest_prompts = [controlled_prompt(b, prefix) for _a, b, prefix, _ca, _cb in CONTROLLED_PAIRS]
    source_inputs = batch_inputs(source_prompts, bundle)
    dest_inputs = batch_inputs(dest_prompts, bundle)
    source_logits = forward_logits(source_inputs, bundle)
    dest_logits = forward_logits(dest_inputs, bundle)
    source_head = capture_head_input(source_inputs, candidate, bundle)
    order = torch.arange(len(CONTROLLED_PAIRS), device=bundle.device).roll(1)
    for head in range(model.config.text_config.num_attention_heads):
        with patch_head_final(candidate, head, source_head):
            patched_logits = forward_logits(dest_inputs, bundle)
        with patch_head_final(candidate, head, source_head, order):
            random_logits = forward_logits(dest_inputs, bundle)
        for i, (anchor_a, anchor_b, prefix, cand_a, cand_b) in enumerate(CONTROLLED_PAIRS):
            src = candidate_difference(source_logits[i], cand_a, cand_b, bundle)
            dst = candidate_difference(dest_logits[i], cand_a, cand_b, bundle)
            patched = candidate_difference(patched_logits[i], cand_a, cand_b, bundle)
            random = candidate_difference(random_logits[i], cand_a, cand_b, bundle)
            denominator = src - dst
            pair_rows.append({
                "head": head,
                "pair": f"{anchor_a}/{anchor_b}",
                "candidate_a": cand_a,
                "candidate_b": cand_b,
                "source_difference": src,
                "destination_difference": dst,
                "anchor_effect": src - dst,
                "patched_difference": patched,
                "random_source_difference": random,
                "recovery": (patched - dst) / denominator if abs(denominator) > 1e-6 else None,
                "random_recovery": (random - dst) / denominator if abs(denominator) > 1e-6 else None,
            })
    write_jsonl(output / "controlled_head_transfer.jsonl", pair_rows)

    # Directly patch the layer-14 full-attention shared memory at the anchor
    # position. All later full-attention layers (including L24H3) reuse this K/V.
    with torch.inference_mode():
        source_output = model(
            **source_inputs, logits_to_keep=1, use_cache=False, return_shared_kv_states=True
        )
    source_shared = source_output.shared_kv_states["full_attention"]
    source_positions = anchor_positions(
        source_prompts, [row[0] for row in CONTROLLED_PAIRS], bundle, source_inputs["input_ids"].shape[1]
    )
    dest_positions = anchor_positions(
        dest_prompts, [row[1] for row in CONTROLLED_PAIRS], bundle, dest_inputs["input_ids"].shape[1]
    )
    batch_index = torch.arange(len(CONTROLLED_PAIRS), device=bundle.device)
    kv_rows = []
    for mode in ["key", "value", "both"]:
        for random_source in [False, True]:
            source_order = batch_index.roll(1) if random_source else batch_index

            def patch_shared(_module, _args, kwargs, output):
                key, value = kwargs["shared_kv_states"]["full_attention"]
                key, value = key.clone(), value.clone()
                source_key, source_value = source_shared
                if mode in {"key", "both"}:
                    key[batch_index, :, dest_positions] = source_key[source_order, :, source_positions[source_order]]
                if mode in {"value", "both"}:
                    value[batch_index, :, dest_positions] = source_value[source_order, :, source_positions[source_order]]
                kwargs["shared_kv_states"]["full_attention"] = (key, value)
                return output

            handle = layers[14].self_attn.register_forward_hook(patch_shared, with_kwargs=True)
            patched_logits = forward_logits(dest_inputs, bundle)
            handle.remove()
            for i, (anchor_a, anchor_b, _prefix, cand_a, cand_b) in enumerate(CONTROLLED_PAIRS):
                src = candidate_difference(source_logits[i], cand_a, cand_b, bundle)
                dst = candidate_difference(dest_logits[i], cand_a, cand_b, bundle)
                patched = candidate_difference(patched_logits[i], cand_a, cand_b, bundle)
                kv_rows.append({
                    "mode": mode,
                    "random_source": random_source,
                    "pair": f"{anchor_a}/{anchor_b}",
                    "source_difference": src,
                    "destination_difference": dst,
                    "patched_difference": patched,
                    "recovery": (patched - dst) / (src - dst) if abs(src - dst) > 1e-6 else None,
                })
    write_jsonl(output / "shared_kv_transfer.jsonl", kv_rows)

    # Ordered source head -> shuffled destination on all natural prompts.
    clean_examples = condition_examples["rhyming"]
    corrupt_examples = condition_examples["shuffled"]
    anchors = [x.anchor for x in clean_examples]
    targets = [x.target for x in clean_examples]
    bundle.rhyme_cache = {a: rhyme_token_ids(a, bundle.token_words) for a in set(anchors)}
    clean_inputs = batch_inputs([x.prompt for x in clean_examples], bundle)
    corrupt_inputs = batch_inputs([x.prompt for x in corrupt_examples], bundle)
    clean_logits = forward_logits(clean_inputs, bundle)
    corrupt_logits = forward_logits(corrupt_inputs, bundle)
    source_head = capture_head_input(clean_inputs, candidate, bundle)
    transfer_rows = []
    for head in range(model.config.text_config.num_attention_heads):
        with patch_head_final(candidate, head, source_head):
            patched_logits = forward_logits(corrupt_inputs, bundle)
        clean_m = metrics(clean_logits, anchors, targets, bundle)
        corrupt_m = metrics(corrupt_logits, anchors, targets, bundle)
        patch_m = metrics(patched_logits, anchors, targets, bundle)
        for i, anchor in enumerate(anchors):
            denom = clean_m[i]["rhyme_mass"] - corrupt_m[i]["rhyme_mass"]
            transfer_rows.append({
                "head": head,
                "anchor": anchor,
                "clean_mass": clean_m[i]["rhyme_mass"],
                "corrupt_mass": corrupt_m[i]["rhyme_mass"],
                "patched_mass": patch_m[i]["rhyme_mass"],
                "recovery": ((patch_m[i]["rhyme_mass"] - corrupt_m[i]["rhyme_mass"]) / denom
                             if abs(denom) > 1e-6 else None),
            })
    write_jsonl(output / "ordered_to_shuffled_head_transfer.jsonl", transfer_rows)

    # Phonology vs orthography behavioral dissociations using identical scaffolds.
    phonology_rows = []
    for left, right, probe in HOMOPHONES:
        unrelated = HOMOPHONES[(HOMOPHONES.index((left, right, probe)) + 1) % len(HOMOPHONES)][0]
        prompts = [controlled_prompt(left, "The poet completed the rhyme with"),
                   controlled_prompt(right, "The poet completed the rhyme with"),
                   controlled_prompt(unrelated, "The poet completed the rhyme with")]
        logits = forward_logits(batch_inputs(prompts, bundle), bundle)
        probe_id = target_token_id(bundle.tokenizer, probe)
        phonology_rows.append({
            "kind": "homophone",
            "left": left,
            "right": right,
            "left_probe": probe,
            "right_probe": probe,
            "left_probe_logit": float(logits[0, probe_id]),
            "right_probe_logit": float(logits[1, probe_id]),
            "unrelated": unrelated,
            "unrelated_probe_logit": float(logits[2, probe_id]),
            "left_minus_right_probe_logit": float(logits[0, probe_id] - logits[1, probe_id]),
            "left_minus_unrelated_probe_logit": float(logits[0, probe_id] - logits[2, probe_id]),
        })
    for left, right, left_probe, right_probe in ORTHOGRAPHIC_FOILS:
        prompts = [controlled_prompt(left, "The poet completed the rhyme with"),
                   controlled_prompt(right, "The poet completed the rhyme with")]
        logits = forward_logits(batch_inputs(prompts, bundle), bundle)
        li = target_token_id(bundle.tokenizer, left_probe)
        ri = target_token_id(bundle.tokenizer, right_probe)
        phonology_rows.append({
            "kind": "orthographic_foil",
            "left": left,
            "right": right,
            "left_probe": left_probe,
            "right_probe": right_probe,
            "left_condition_preference": float(logits[0, li] - logits[0, ri]),
            "right_condition_preference": float(logits[1, li] - logits[1, ri]),
            "difference_in_differences": float((logits[0, li] - logits[0, ri]) - (logits[1, li] - logits[1, ri])),
        })
    write_jsonl(output / "phonology_controls.jsonl", phonology_rows)

    print(f"Wrote targeted validation to {output}")


if __name__ == "__main__":
    run(Path("artifacts/gemma4_validation"))
