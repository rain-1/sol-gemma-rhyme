"""Stage 1 of the MLP-13 SAE: capture layer-13 residuals over a large lexicon.

Scales past the 30-family probe set to every single-token English word with a
CMUdict pronunciation, captured in the rhyme-mode scaffold. Saved for training a
sparse autoencoder on the layer-13 output, where the causal rhyme write is a
low-rank direction in superposition (report 09).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rhyme_interp.dataset import RHYME_DEMONSTRATION_LINES
from rhyme_interp.model import load_model
from rhyme_interp.families import unambiguous_rhyme_key
from rhyme_interp.rhyme import rhyme_keys
from run_gemma4_interpretability import MODEL, batch_inputs

LAYER = 13
SCAFFOLD = "\n".join(RHYME_DEMONSTRATION_LINES) + "\nEvery line she wrote would end in {word}"


def rime_label(word):
    key = unambiguous_rhyme_key(word)
    if key is None:
        keys = rhyme_keys(word)
        key = min(keys) if keys else None
    return "-".join(key) if key else ""


@torch.inference_mode()
def run(args):
    bundle = load_model(MODEL, load_in_4bit=True, attn_implementation="eager")
    tok = bundle.tokenizer
    # single-token, alphabetic, real (has a pronunciation) words
    words = []
    for w in bundle.token_words.values():
        if w.isalpha() and len(w) >= 2 and rhyme_keys(w):
            words.append(w)
    words = sorted(set(words))
    if args.limit:
        words = words[: args.limit]
    print(f"{len(words)} single-token real words")

    net = bundle.model.model.language_model.layers[LAYER]
    acts = []
    for s in range(0, len(words), 64):
        buf = {}
        h = net.register_forward_hook(
            lambda m, a, o: buf.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[:, -1].detach().float().cpu()))
        bundle.model(**batch_inputs([SCAFFOLD.format(word=w) for w in words[s:s + 64]], bundle),
                     use_cache=False)
        h.remove()
        acts.append(buf["h"])
        if s % 1024 == 0:
            print(f"  {s}/{len(words)}")
    acts = torch.cat(acts).numpy().astype(np.float16)
    labels = [rime_label(w) for w in words]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, activations=acts, words=np.array(words), labels=np.array(labels))
    n_lab = sum(1 for l in labels if l)
    print(f"wrote {args.output}: acts {acts.shape}, {n_lab} labelled, "
          f"{len(set(l for l in labels if l))} rime families")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/gemma4_mlp_rhyme/lexicon_l13.npz"))
    p.add_argument("--limit", type=int, default=6000)
    run(p.parse_args())


if __name__ == "__main__":
    main()
