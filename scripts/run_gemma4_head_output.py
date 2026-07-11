"""What does L24H3 actually write into the residual stream?

Three questions about the retrieval head's output at the final token:

1. **Rhyme-set readout.** Project each head's residual update directly onto the
   unembedding (frozen-norm direct logit attribution). If L24H3 carries the
   rhyme constraint in output space, its top boosted words should *be* the
   anchor's rhyme family — the rhyme set read out of one head's activation.
2. **Direct versus indirect.** Ablate the head while replaying every later
   final-position update from the clean run. The remaining loss is the head's
   direct-to-logits effect; the rest travels through later MLPs/attention.
3. **Rank.** Transfer the head output between controlled anchor pairs, but
   restricted to the top-k principal components of the head's output
   distribution over 359 anchor words. How many dimensions carry the
   constraint? Random subspaces of the same rank are the control.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from rhyme_interp.families import build_families
from rhyme_interp.model import load_model, target_token_id
from rhyme_interp.rhyme import rhyme_token_ids, rhymes

from run_gemma4_interpretability import (
    MODEL,
    batch_inputs,
    forward_logits,
    metrics,
    write_jsonl,
    zero_head_final,
)
from rhyme_interp.dataset import build_elicitation_dataset
from validate_gemma4_circuit import CONTROLLED_PAIRS, candidate_difference, controlled_prompt

CANDIDATE_LAYER, CANDIDATE_HEAD = 24, 3


def frozen_rms(x, reference, weight, eps):
    """Gemma4RMSNorm (`x/rms * weight`) with the scale computed from `reference`."""
    scale = torch.rsqrt(reference.float().pow(2).mean(-1, keepdim=True) + eps)
    return x.float() * scale * weight.float()


@torch.inference_mode()
def capture_final(inputs, bundle, layer):
    """o_proj input at the final position, plus the pre-norm last residual."""
    captured = {}

    def save_o_proj(_module, args):
        captured["o_proj_input"] = args[0][:, -1].detach().clone()

    def save_last(_module, _args, output):
        captured["last_residual"] = output[:, -1].detach().clone()

    layers = bundle.model.model.language_model.layers
    handles = [
        layer.self_attn.o_proj.register_forward_pre_hook(save_o_proj),
        layers[-1].register_forward_hook(save_last),
    ]
    logits = forward_logits(inputs, bundle)
    for handle in handles:
        handle.remove()
    return captured["o_proj_input"], captured["last_residual"], logits


@torch.inference_mode()
def head_residual_updates(o_proj_input, layer):
    """Per-head residual updates with the joint (frozen) post-attention norm."""
    head_dim = layer.self_attn.head_dim
    heads = o_proj_input.shape[-1] // head_dim
    full = layer.self_attn.o_proj(o_proj_input)
    norm = layer.post_attention_layernorm
    updates = []
    for head in range(heads):
        masked = torch.zeros_like(o_proj_input)
        s = slice(head * head_dim, (head + 1) * head_dim)
        masked[:, s] = o_proj_input[:, s]
        raw = layer.self_attn.o_proj(masked)
        updates.append(frozen_rms(raw, full, norm.weight, norm.eps))
    return updates  # list of (batch, width) float32


def run(args):
    bundle = load_model(MODEL, load_in_4bit=True, attn_implementation="eager")
    model = bundle.model
    layers = model.model.language_model.layers
    candidate = layers[CANDIDATE_LAYER]
    final_norm = model.model.language_model.norm
    eligible_ids = torch.tensor(sorted(bundle.token_words), device=bundle.device)
    unembed = model.lm_head.weight[eligible_ids]

    examples = build_elicitation_dataset("rhyming")
    anchors = [example.anchor for example in examples]
    targets = [example.target for example in examples]
    bundle.rhyme_cache = {a: rhyme_token_ids(a, bundle.token_words) for a in set(anchors)}
    inputs = batch_inputs([example.prompt for example in examples], bundle)

    # 1. Frozen-norm direct logit attribution for all eight heads at layer 24.
    o_proj_input, last_residual, clean_logits = capture_final(inputs, bundle, candidate)
    updates = head_residual_updates(o_proj_input, candidate)
    id_list = eligible_ids.tolist()
    eligible_words = [bundle.token_words[i] for i in id_list]
    position_of = {token_id: i for i, token_id in enumerate(id_list)}
    dla_rows = []
    for head, update in enumerate(updates):
        normed = frozen_rms(update, last_residual, final_norm.weight, final_norm.eps)
        contribution = normed.to(unembed.dtype) @ unembed.T  # (batch, eligible)
        contribution = contribution.float()
        for i, (anchor, target) in enumerate(zip(anchors, targets)):
            family_positions = torch.tensor(
                [position_of[t] for t in bundle.rhyme_cache[anchor] if t in position_of],
                device=bundle.device,
            )
            family_mask = torch.zeros(len(id_list), dtype=torch.bool, device=bundle.device)
            family_mask[family_positions] = True
            top = torch.topk(contribution[i], 20).indices
            top_words = [eligible_words[int(j)] for j in top]
            dla_rows.append({
                "head": head,
                "anchor": anchor,
                "family_mean_contribution": float(contribution[i][family_mask].mean()),
                "nonfamily_mean_contribution": float(contribution[i][~family_mask].mean()),
                "target_contribution": float(
                    contribution[i][position_of[target_token_id(bundle.tokenizer, target)]]
                ),
                "anchor_contribution": float(
                    contribution[i][position_of[target_token_id(bundle.tokenizer, anchor)]]
                ),
                "top20_words": top_words,
                "top20_family_fraction": float(np.mean([rhymes(anchor, w) for w in top_words])),
            })
    write_jsonl(args.output / "head_dla.jsonl", dla_rows)
    h3 = [r for r in dla_rows if r["head"] == CANDIDATE_HEAD]
    print(f"L24H3 mean top-20 family fraction: {np.mean([r['top20_family_fraction'] for r in h3]):.3f}")

    # 2. Direct versus indirect: ablate the head with later updates replayed.
    replay_modules = []
    for layer_index in range(CANDIDATE_LAYER, len(layers)):
        block = layers[layer_index]
        replay_modules.append((layer_index, "mlp", block.post_feedforward_layernorm))
        replay_modules.append((layer_index, "per_layer", block.post_per_layer_input_norm))
        if layer_index > CANDIDATE_LAYER:
            replay_modules.append((layer_index, "attention", block.post_attention_layernorm))

    cache = {}
    handles = [
        module.register_forward_hook(
            lambda _m, _a, output, key=key: cache.__setitem__(key, output[:, -1].detach().clone())
        )
        for key, module in [((li, kind), module) for li, kind, module in replay_modules]
    ]
    forward_logits(inputs, bundle)
    for handle in handles:
        handle.remove()

    @contextmanager
    def replay_downstream():
        def make_hook(key):
            def hook(_module, _args, output):
                fixed = output.clone()
                fixed[:, -1] = cache[key]
                return fixed
            return hook

        handles = [module.register_forward_hook(make_hook((li, kind)))
                   for li, kind, module in replay_modules]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    clean = metrics(clean_logits, anchors, targets, bundle)
    with zero_head_final(candidate, CANDIDATE_HEAD, candidate.self_attn.head_dim):
        total_logits = forward_logits(inputs, bundle)
        with replay_downstream():
            direct_logits = forward_logits(inputs, bundle)
    total = metrics(total_logits, anchors, targets, bundle)
    direct = metrics(direct_logits, anchors, targets, bundle)
    path_rows = []
    for i, anchor in enumerate(anchors):
        path_rows.append({
            "anchor": anchor,
            "clean_rhyme_mass": clean[i]["rhyme_mass"],
            "total_ablated_mass": total[i]["rhyme_mass"],
            "direct_only_ablated_mass": direct[i]["rhyme_mass"],
        })
    write_jsonl(args.output / "direct_vs_indirect.jsonl", path_rows)
    print(
        "mean mass clean {:.3f} | total ablation {:.3f} | direct-only ablation {:.3f}".format(
            np.mean([r["clean_rhyme_mass"] for r in path_rows]),
            np.mean([r["total_ablated_mass"] for r in path_rows]),
            np.mean([r["direct_only_ablated_mass"] for r in path_rows]),
        )
    )

    # 3. Rank of the transferable constraint in the head's value space.
    families = build_families(bundle.token_words)
    words = [word for family in families for word in family.words]
    basis_inputs = [controlled_prompt(word, "The poet completed the rhyme with") for word in words]
    slices = []
    for start in range(0, len(basis_inputs), 64):
        chunk = batch_inputs(basis_inputs[start : start + 64], bundle)
        o_input, _, _ = capture_final(chunk, bundle, candidate)
        head_slice = o_input[:, CANDIDATE_HEAD * candidate.self_attn.head_dim :
                             (CANDIDATE_HEAD + 1) * candidate.self_attn.head_dim]
        slices.append(head_slice.float().cpu())
    basis = torch.cat(slices)
    mean = basis.mean(0)
    _, singular_values, v_matrix = torch.linalg.svd(basis - mean, full_matrices=False)
    explained = (singular_values ** 2 / (singular_values ** 2).sum()).cumsum(0)

    source_prompts = [controlled_prompt(a, prefix) for a, _b, prefix, _ca, _cb in CONTROLLED_PAIRS]
    dest_prompts = [controlled_prompt(b, prefix) for _a, b, prefix, _ca, _cb in CONTROLLED_PAIRS]
    source_inputs = batch_inputs(source_prompts, bundle)
    dest_inputs = batch_inputs(dest_prompts, bundle)
    source_o, _, source_logits = capture_final(source_inputs, bundle, candidate)
    dest_o, _, dest_logits = capture_final(dest_inputs, bundle, candidate)
    head_dim = candidate.self_attn.head_dim
    s = slice(CANDIDATE_HEAD * head_dim, (CANDIDATE_HEAD + 1) * head_dim)
    source_slice = source_o[:, s].float().cpu()
    dest_slice = dest_o[:, s].float().cpu()

    generator = torch.Generator().manual_seed(0)
    rank_rows = []
    for k in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        for control in [False, True]:
            if control:
                random = torch.randn(basis.shape[1], basis.shape[1], generator=generator)
                q_matrix, _ = torch.linalg.qr(random)
                projector = q_matrix[:, :k] @ q_matrix[:, :k].T
            else:
                projector = v_matrix[:k].T @ v_matrix[:k]
            patched_slice = dest_slice + (source_slice - dest_slice) @ projector
            patched_input = dest_o.clone()
            patched_input[:, s] = patched_slice.to(dest_o.dtype).to(bundle.device)

            def hook(_module, args):
                hidden = args[0].clone()
                hidden[:, -1] = patched_input
                return (hidden, *args[1:])

            handle = candidate.self_attn.o_proj.register_forward_pre_hook(hook)
            patched_logits = forward_logits(dest_inputs, bundle)
            handle.remove()
            for i, (anchor_a, anchor_b, _prefix, cand_a, cand_b) in enumerate(CONTROLLED_PAIRS):
                src = candidate_difference(source_logits[i], cand_a, cand_b, bundle)
                dst = candidate_difference(dest_logits[i], cand_a, cand_b, bundle)
                patched = candidate_difference(patched_logits[i], cand_a, cand_b, bundle)
                rank_rows.append({
                    "rank": k,
                    "random_subspace": control,
                    "pair": f"{anchor_a}/{anchor_b}",
                    "explained_variance": float(explained[k - 1]),
                    "recovery": (patched - dst) / (src - dst) if abs(src - dst) > 1e-6 else None,
                })
    write_jsonl(args.output / "head_output_rank.jsonl", rank_rows)
    for k in [1, 2, 4, 8]:
        rec = np.mean([r["recovery"] for r in rank_rows if r["rank"] == k and not r["random_subspace"]])
        print(f"rank {k}: mean recovery {rec:.3f}")
    print(f"Wrote head output analysis to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/gemma4_head_output"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
