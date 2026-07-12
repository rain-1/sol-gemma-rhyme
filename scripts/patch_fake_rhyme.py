"""Silly but honest: write a fake rhyme identity into a word's slot.

If the layer-14 value memory is the rhyme code (report 07), then overwriting the
layer-13 residual at an anchor with a chosen family's mean should make the
retrieval head believe that word rhymes with that family -- even for a word with
no real rhyme, like "orange". We take unrhymable / arbitrary anchors, replace
their family code with a target family's mean at layer 13, and read the greedy
completion.
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

LAYER = 13  # inject at the output of block 13, just before layer-14 value storage
DEMO = "\n".join(RHYME_DEMONSTRATION_LINES)  # primes the model into "rhyme mode"

# first line ends on the target anchor; second line is semantically open so the
# rhyme pathway (not a fixed semantic completion) decides the final word
COUPLETS = [
    ("orange", "The lantern glowed a warm bright orange", "and every heart began to"),
    ("silver", "The winter moon was pale and silver", "and all at once we saw the"),
    ("month", "They had been gone for one whole month", "and then at last they found the"),
    ("engine", "The workshop held a broken engine", "and in the dark we heard the"),
]
# families we will try to make them rhyme with (rep word -> greedy should join it)
TARGETS = ["EY1", "AY1-T", "OW1-L-D", "IY1", "EH1-L"]


def anchor_last_token(first_line, prompt, bundle, padded_len):
    tok = bundle.tokenizer
    full = tok(prompt)["input_ids"]
    first = tok(first_line)["input_ids"]
    return len(first) - 1 + padded_len - len(full)


def run(args):
    data = np.load(Path(args.representation) / "activations.npz")
    words = [str(w) for w in data["words"]]
    labels = [str(l) for l in data["labels"]]
    states = data["states_final_word"].astype(np.float32)  # (words, emb+layers, width)
    fam_words = json.loads((Path(args.representation) / "families.json").read_text())

    # family mean residual at layer 13 (row LAYER+1 skips the embedding row)
    row = LAYER + 1
    means = {}
    for name in set(labels):
        idx = [i for i, l in enumerate(labels) if l == name]
        means[name] = states[idx, row].mean(0)

    bundle = load_model(MODEL, load_in_4bit=not args.bf16, attn_implementation="eager")
    layers = bundle.model.model.language_model.layers
    tok = bundle.tokenizer

    prompts = [f"{DEMO}\n{l1}\n{p2}" for _, l1, p2 in COUPLETS]
    first_lines = [f"{DEMO}\n{l1}" for _, l1, _ in COUPLETS]
    inputs = batch_inputs(prompts, bundle)
    padded = inputs["input_ids"].shape[1]
    positions = torch.tensor([anchor_last_token(fl, pr, bundle, padded)
                              for fl, pr in zip(first_lines, prompts)], device=bundle.device)
    batch = torch.arange(len(prompts), device=bundle.device)

    def greedy(delta=None, strength=0.0, overwrite=False):
        handle = None
        if delta is not None:
            def hook(_m, _a, output):
                h = output.clone()
                if overwrite:
                    h[batch, positions] = delta.to(h.dtype)
                else:
                    h[batch, positions] += strength * delta.to(h.dtype)
                return h
            handle = layers[LAYER].register_forward_hook(hook)
        with torch.inference_mode():
            logits = bundle.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        if handle:
            handle.remove()
        return logits

    # capture each anchor's own layer-13 code so we can overwrite (target - anchor)
    captured = {}
    def cap(_m, _a, output):
        captured["s"] = output[batch, positions].detach().float().cpu().numpy()
    h = layers[LAYER].register_forward_hook(cap)
    base_logits = greedy()
    h.remove()
    anchor_state = captured["s"]

    def word(logits, i):
        return tok.decode([logits[i].argmax()]).strip()

    strengths = [2.0, 4.0, 8.0]
    print(f"injecting at layer {LAYER}; strengths {strengths} then overwrite\n")
    for i, (anchor, l1, _) in enumerate(COUPLETS):
        print(f"=== anchor: {anchor!r} ===  normal completion: {word(base_logits, i)!r}")
        for t in TARGETS:
            delta = torch.tensor(np.stack([means[t] - anchor_state[j] for j in range(len(prompts))]),
                                 device=bundle.device)
            target_ids = rhyme_token_ids(fam_words[t][0], bundle.token_words)
            cells = []
            for s in strengths:
                lo = greedy(delta, s)
                cells.append(f"a{int(s)}:{word(lo, i)}({lo.softmax(-1)[i, target_ids].sum():.2f})")
            # overwrite: replace the anchor code entirely with the target mean
            omean = torch.tensor(np.stack([means[t] for _ in range(len(prompts))]), device=bundle.device)
            lo = greedy(omean, overwrite=True)
            cells.append(f"OW:{word(lo, i)}({lo.softmax(-1)[i, target_ids].sum():.2f})")
            print(f"  {t:8s} ({', '.join(fam_words[t][:3]):>18}): " + "  ".join(cells))
        print()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--representation", type=Path, default=Path("artifacts/gemma4_representation"))
    p.add_argument("--strength", type=float, default=1.5)
    p.add_argument("--bf16", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
