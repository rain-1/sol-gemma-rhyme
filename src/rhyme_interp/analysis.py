"""Causal interventions for GPT-NeoX/Pythia."""

from __future__ import annotations

from contextlib import contextmanager

import torch

from .model import next_token_logits, score_logits


def _replace_output(output, hidden):
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


@contextmanager
def layer_update_ablation(model, layer_index: int):
    """Remove one block's update at the final sequence position."""
    layer = model.gpt_neox.layers[layer_index]
    state = {}

    def pre_hook(_module, args):
        state["input"] = args[0]

    def hook(_module, _args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        hidden[:, -1] = state["input"][:, -1]
        return _replace_output(output, hidden)

    handles = [layer.register_forward_pre_hook(pre_hook), layer.register_forward_hook(hook)]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def attention_head_ablation(model, layer_index: int, head_index: int):
    """Zero one head before the attention output projection at the final token."""
    layer = model.gpt_neox.layers[layer_index]
    n_heads = model.config.num_attention_heads
    head_size = model.config.hidden_size // n_heads

    def pre_hook(_module, args):
        hidden = args[0].clone()
        start, end = head_index * head_size, (head_index + 1) * head_size
        hidden[:, -1, start:end] = 0
        return (hidden, *args[1:])

    handle = layer.attention.dense.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


@torch.inference_mode()
def scan_layer_ablation(example, bundle) -> list[dict]:
    baseline = score_logits(next_token_logits(example.prompt, bundle), example.anchor, example.target, bundle)
    rows = []
    for layer in range(bundle.model.config.num_hidden_layers):
        with layer_update_ablation(bundle.model, layer):
            metrics = score_logits(next_token_logits(example.prompt, bundle), example.anchor, example.target, bundle)
        rows.append({
            "layer": layer,
            "rhyme_mass": metrics["rhyme_mass"],
            "delta_rhyme_mass": metrics["rhyme_mass"] - baseline["rhyme_mass"],
            "delta_rhyme_logit_advantage": metrics["rhyme_logit_advantage"] - baseline["rhyme_logit_advantage"],
        })
    return rows


@torch.inference_mode()
def scan_head_ablation(example, bundle, layers: list[int] | None = None) -> list[dict]:
    baseline = score_logits(next_token_logits(example.prompt, bundle), example.anchor, example.target, bundle)
    layers = layers or list(range(bundle.model.config.num_hidden_layers))
    rows = []
    for layer in layers:
        for head in range(bundle.model.config.num_attention_heads):
            with attention_head_ablation(bundle.model, layer, head):
                metrics = score_logits(next_token_logits(example.prompt, bundle), example.anchor, example.target, bundle)
            rows.append({
                "layer": layer,
                "head": head,
                "rhyme_mass": metrics["rhyme_mass"],
                "delta_rhyme_mass": metrics["rhyme_mass"] - baseline["rhyme_mass"],
                "delta_rhyme_logit_advantage": metrics["rhyme_logit_advantage"] - baseline["rhyme_logit_advantage"],
            })
    return rows


def _anchor_position(prompt: str, anchor: str, tokenizer) -> int:
    tokens = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    anchor_ids = tokenizer.encode(" " + anchor, add_special_tokens=False)
    if len(anchor_ids) != 1:
        raise ValueError(f"Anchor must be one token: {anchor}")
    hits = (tokens == anchor_ids[0]).nonzero().flatten()
    if not len(hits):
        raise ValueError(f"Anchor token not found in prompt: {anchor}")
    return int(hits[-1])


@torch.inference_mode()
def counterfactual_anchor_patch(source, destination, bundle) -> list[dict]:
    """Patch the source anchor's residual state over the destination anchor by layer."""
    model, tokenizer = bundle.model, bundle.tokenizer
    source_inputs = tokenizer(source.prompt, return_tensors="pt").to(bundle.device)
    dest_inputs = tokenizer(destination.prompt, return_tensors="pt").to(bundle.device)
    source_pos = _anchor_position(source.prompt, source.anchor, tokenizer)
    dest_pos = _anchor_position(destination.prompt, destination.anchor, tokenizer)
    baseline_logits = model(**dest_inputs).logits[0, -1]
    baseline_source = score_logits(baseline_logits, source.anchor, source.target, bundle)
    baseline_dest = score_logits(baseline_logits, destination.anchor, destination.target, bundle)
    rows = []
    for layer_index, layer in enumerate(model.gpt_neox.layers):
        captured = {}

        def capture(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured["value"] = hidden[:, source_pos].detach().clone()

        handle = layer.register_forward_hook(capture)
        model(**source_inputs)
        handle.remove()

        def patch(_module, _args, output):
            hidden = (output[0] if isinstance(output, tuple) else output).clone()
            hidden[:, dest_pos] = captured["value"]
            return _replace_output(output, hidden)

        handle = layer.register_forward_hook(patch)
        patched_logits = model(**dest_inputs).logits[0, -1]
        handle.remove()
        src = score_logits(patched_logits, source.anchor, source.target, bundle)
        dst = score_logits(patched_logits, destination.anchor, destination.target, bundle)
        rows.append({
            "layer": layer_index,
            "source_anchor": source.anchor,
            "destination_anchor": destination.anchor,
            "source_rhyme_mass_delta": src["rhyme_mass"] - baseline_source["rhyme_mass"],
            "destination_rhyme_mass_delta": dst["rhyme_mass"] - baseline_dest["rhyme_mass"],
        })
    return rows

