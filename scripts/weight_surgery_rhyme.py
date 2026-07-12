"""Bake a fake rhyme into the weights of the layer-13 MLP (no training).

Report 07 localized the rhyme write to the layer-13 MLP; report 08 wrote a fake
rhyme in with an activation patch. This does it *permanently* with a rank-1 edit
to that MLP's down-projection, keyed on the anchor token -- a ROME-style edit.
After the edit the model completes with the fake rhyme with NO activation hook.

Gemma wrinkle: the MLP output passes through post_feedforward_layernorm (RMSNorm)
before joining the residual, which fixes the injected term's magnitude. So we
steer the *direction*: choose the down-projection output whose normed form points
along the target family's residual direction, u = dir / (1 + gamma).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rhyme_interp.dataset import RHYME_DEMONSTRATION_LINES
from rhyme_interp.model import load_model
from rhyme_interp.rhyme import rhyme_token_ids
from run_gemma4_interpretability import MODEL, batch_inputs

LAYER = 13
DEMO = "\n".join(RHYME_DEMONSTRATION_LINES)
SCAFFOLD = DEMO + "\nEvery line she wrote would end in {word}"

EDITS = [  # (anchor, first line, open second line, target family)
    ("month", "They had been gone for one whole month", "and then at last they found the", "AY1-T"),
    ("orange", "The lantern glowed a warm bright orange", "and every heart began to", "AY1-T"),
]
# unrelated prompts to check collateral damage
COLLATERAL = [
    (DEMO + "\nThe northern wind blew sharp and cold", "The cabin stood there, dark and", "cold"),
    (DEMO + "\nThe river hurried toward the sea", "The captive bird at last flew", "sea"),
    ("The capital of France is", None, None),
    ("Two plus two equals", None, None),
]


@torch.inference_mode()
def family_means(bundle):
    fam = json.loads(Path("artifacts/gemma4_representation/families.json").read_text())
    words, labels = [], []
    for name, members in fam.items():
        for w in members:
            if len(bundle.tokenizer(" " + w, add_special_tokens=False)["input_ids"]) == 1:
                words.append(w); labels.append(name)
    H = []
    for s in range(0, len(words), 32):
        prompts = [SCAFFOLD.format(word=w) for w in words[s:s + 32]]
        hs = bundle.model(**batch_inputs(prompts, bundle), use_cache=False,
                          output_hidden_states=True).hidden_states
        H.append(hs[LAYER + 1][:, -1].float())  # block-13 output
    H = torch.cat(H)
    means = {name: H[[i for i, l in enumerate(labels) if l == name]].mean(0)
             for name in set(labels)}
    return means, H.mean(0), fam


@torch.inference_mode()
def anchor_pos(first_line, prompt, bundle, padded):
    tok = bundle.tokenizer
    return len(tok(first_line)["input_ids"]) - 1 + padded - len(tok(prompt)["input_ids"])


def run(args):
    bundle = load_model(MODEL, load_in_4bit=False, attn_implementation="eager")
    model = bundle.model
    tok = bundle.tokenizer
    layer = model.model.language_model.layers[LAYER]
    down = layer.mlp.down_proj
    gamma = layer.post_feedforward_layernorm.weight.detach().float()  # (d_model,)
    print(f"down_proj weight {tuple(down.weight.shape)}, dtype {down.weight.dtype}")

    means, global_mean, fam = family_means(bundle)

    def greedy(prompt, second):
        full = prompt if second is None else f"{prompt}\n{second}"
        inp = batch_inputs([full], bundle)
        logits = model(**inp, use_cache=False, logits_to_keep=1).logits[0, -1].float()
        return tok.decode([logits.argmax()]).strip(), logits

    # collateral baseline
    coll_before = [greedy(p, s)[0] for p, s, _ in COLLATERAL]

    for anchor, l1, l2, target in EDITS:
        prompt = f"{DEMO}\n{l1}"
        full = f"{prompt}\n{l2}"
        inp = batch_inputs([full], bundle)
        pos = anchor_pos(prompt, full, bundle, inp["input_ids"].shape[1])

        # capture down_proj input k at the anchor position
        cap = {}
        h = down.register_forward_pre_hook(lambda m, a: cap.__setitem__("k", a[0][0, pos].detach().float()))
        base_word, base_logits = greedy(prompt, l2)
        h.remove()
        k = cap["k"]
        ids = rhyme_token_ids(fam[target][0], bundle.token_words)
        base_mass = base_logits.softmax(-1)[ids].sum().item()

        # rank-1 edit: set down_proj(k) so the normed MLP term points along dir
        dir_t = (means[target] - global_mean)
        u = dir_t / (1.0 + gamma)
        Wk = (down.weight.float() @ k)
        u = u / u.norm() * Wk.norm()  # natural output scale
        delta = torch.outer((u - Wk), k) / (k @ k)
        down.weight.data += delta.to(down.weight.dtype)

        edit_word, edit_logits = greedy(prompt, l2)
        edit_mass = edit_logits.softmax(-1)[ids].sum().item()

        print(f"\n=== {anchor!r}  ->  make it rhyme with {target} ({', '.join(fam[target][:3])}) ===")
        print(f"  before edit: {base_word!r:10}  ({target} mass {base_mass:.2f})")
        print(f"  AFTER  edit: {edit_word!r:10}  ({target} mass {edit_mass:.2f})   [no hooks; weights changed]")
        print(f"  ||delta_W|| = {delta.norm():.3f}, edited {(delta.abs()>0).any(0).sum().item()} of {delta.shape[1]} columns")

        down.weight.data -= delta.to(down.weight.dtype)  # restore for the next edit

    # collateral after (re-apply the first edit to measure damage while it's live)
    anchor, l1, l2, target = EDITS[0]
    prompt = f"{DEMO}\n{l1}"; full = f"{prompt}\n{l2}"
    inp = batch_inputs([full], bundle); pos = anchor_pos(prompt, full, bundle, inp["input_ids"].shape[1])
    cap = {}
    h = down.register_forward_pre_hook(lambda m, a: cap.__setitem__("k", a[0][0, pos].detach().float()))
    greedy(prompt, l2); h.remove(); k = cap["k"]
    dir_t = means[target] - global_mean; u = dir_t / (1.0 + gamma)
    Wk = down.weight.float() @ k; u = u / u.norm() * Wk.norm()
    delta = torch.outer((u - Wk), k) / (k @ k)
    down.weight.data += delta.to(down.weight.dtype)
    coll_after = [greedy(p, s)[0] for p, s, _ in COLLATERAL]
    down.weight.data -= delta.to(down.weight.dtype)

    print(f"\n=== collateral (month->AY1-T edit live) ===")
    for (p, s, tag), b, a in zip(COLLATERAL, coll_before, coll_after):
        flag = "" if b == a else "  <-- CHANGED"
        print(f"  {tag or p[:24]:26}: {b!r} -> {a!r}{flag}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bf16", action="store_true", default=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
