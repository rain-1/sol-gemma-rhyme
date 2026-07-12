"""Open the layer-13 MLP: which neurons write the rhyme code?

Report 07 localized the rhyme write to the layer-13 MLP and a rank-1 edit proved
one direction in its down-projection is load-bearing. This asks what that
direction is made of. We capture the MLP-13 neuron activations (the 6144-d input
to down_proj) for every word and ask:

  1. How sparse is the code -- how few neurons read the rhyme family?
  2. Is it compositional -- do some neurons track the vowel and others the coda?
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rhyme_interp.dataset import RHYME_DEMONSTRATION_LINES, build_elicitation_dataset
from rhyme_interp.model import load_model
from rhyme_interp.rhyme import rhyme_token_ids
from run_gemma4_interpretability import MODEL, anchor_positions, batch_inputs

LAYER = 13
SCAFFOLD = "\n".join(RHYME_DEMONSTRATION_LINES) + "\nEvery line she wrote would end in {word}"


@torch.inference_mode()
def capture_neurons(bundle, words, batch_size=32):
    down = bundle.model.model.language_model.layers[LAYER].mlp.down_proj
    acts = []
    for s in range(0, len(words), batch_size):
        buf = {}
        h = down.register_forward_pre_hook(lambda m, a: buf.__setitem__("x", a[0][:, -1].detach().float().cpu()))
        bundle.model(**batch_inputs([SCAFFOLD.format(word=w) for w in words[s:s + batch_size]], bundle),
                     use_cache=False)
        h.remove()
        acts.append(buf["x"])
    return torch.cat(acts).numpy()  # (words, 6144)


def probe_topk(X, y, ks):
    out = {}
    F, _ = f_classif(X, y)
    order = np.argsort(-np.nan_to_num(F))
    for k in ks:
        cols = order[:k]
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        out[k] = float(cross_val_score(clf, X[:, cols], y, cv=5).mean())
    return out, order


def run(args):
    fam = json.loads(Path("artifacts/gemma4_representation/families.json").read_text())
    bundle = load_model(MODEL, load_in_4bit=True, attn_implementation="eager")
    words, labels = [], []
    for name, members in fam.items():
        for w in members:
            if len(bundle.tokenizer(" " + w, add_special_tokens=False)["input_ids"]) == 1:
                words.append(w); labels.append(name)
    X = capture_neurons(bundle, words)
    print(f"{len(words)} words, {X.shape[1]} MLP-13 neurons, {len(set(labels))} families")

    vowel_of = {n: n.split("-")[0].rstrip("0123456789") for n in set(labels)}
    coda_of = {n: "-".join(n.split("-")[1:]) or "OPEN" for n in set(labels)}
    yf = np.array([sorted(set(labels)).index(l) for l in labels])
    yv = np.array([sorted(set(vowel_of.values())).index(vowel_of[l]) for l in labels])
    yc = np.array([sorted(set(coda_of.values())).index(coda_of[l]) for l in labels])

    # (1) sparsity: family accuracy from the top-k most family-selective neurons
    ks = [4, 8, 16, 32, 64, 128, 512]
    accs, order = probe_topk(X, yf, ks)
    dense = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    full = float(cross_val_score(dense, X, yf, cv=5).mean())
    print("\nsparsity -- family readout from the top-k family-selective neurons:")
    for k in ks:
        print(f"  top {k:4d} neurons: {accs[k]:.2f}")
    print(f"  all {X.shape[1]} neurons: {full:.2f}   (chance {1/len(set(labels)):.3f})")

    # (2) compositionality: are neurons vowel-tuned or coda-tuned?
    Fv, _ = f_classif(X, yv); Fc, _ = f_classif(X, yc)
    Fv, Fc = np.nan_to_num(Fv), np.nan_to_num(Fc)
    print("\ncompositionality -- top coda-selective neurons (coda F >> vowel F):")
    coda_neurons = np.argsort(-(Fc - Fv))[:6]
    codas = sorted(set(coda_of.values()))
    for j in coda_neurons:
        means = [X[yc == c, j].mean() for c in range(len(codas))]
        pref = codas[int(np.argmax(means))]
        print(f"  neuron {j:5d}: coda-F {Fc[j]:5.1f} vowel-F {Fv[j]:4.1f}  fires most for coda -{pref}")
    print("\ntop vowel-selective neurons (vowel F >> coda F):")
    vowels = sorted(set(vowel_of.values()))
    for j in np.argsort(-(Fv - Fc))[:6]:
        means = [X[yv == v, j].mean() for v in range(len(vowels))]
        pref = vowels[int(np.argmax(means))]
        print(f"  neuron {j:5d}: vowel-F {Fv[j]:5.1f} coda-F {Fc[j]:4.1f}  fires most for vowel {pref}")

    # how separable are the two neuron populations
    top = np.argsort(-(Fv + Fc))[:200]
    coda_dom = int((Fc[top] > Fv[top]).sum())
    print(f"\namong the 200 most rhyme-selective neurons: {coda_dom} coda-dominant, {200-coda_dom} vowel-dominant")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, activations=X.astype(np.float16),
                            words=np.array(words), labels=np.array(labels))
        print(f"\nwrote {args.output}")

    # (3) causal: do the top family-selective neurons WRITE the code?
    causal_ablation(bundle, order, X.shape[1])


@torch.inference_mode()
def causal_ablation(bundle, order, n_neurons):
    """Zero the top-k family-selective MLP-13 neurons at the anchor and measure
    the collapse of the rhyme prediction, versus random neurons."""
    down = bundle.model.model.language_model.layers[LAYER].mlp.down_proj
    examples = build_elicitation_dataset("rhyming")
    prompts = [e.prompt for e in examples]
    inputs = batch_inputs(prompts, bundle)
    pos = anchor_positions(prompts, [e.anchor for e in examples], bundle, inputs["input_ids"].shape[1])
    batch = torch.arange(len(examples), device=bundle.device)
    ids = [rhyme_token_ids(e.anchor, bundle.token_words) for e in examples]

    def mass(neurons=None):
        handle = None
        if neurons is not None:
            cols = torch.tensor(neurons, device=bundle.device)
            def hook(_m, a):
                x = a[0].clone()
                x[batch[:, None], pos[:, None], cols[None, :]] = 0
                return (x,) + a[1:]
            handle = down.register_forward_pre_hook(hook)
        logits = bundle.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        if handle:
            handle.remove()
        probs = logits.softmax(-1)
        return float(np.mean([probs[i, ids[i]].sum().item() for i in range(len(examples))]))

    base = mass()
    rng = np.random.default_rng(0)
    print(f"\ncausal -- zero the top-k family-selective neurons at the anchor (baseline mass {base:.3f}):")
    for k in [8, 16, 32, 64]:
        top_mass = mass(order[:k].tolist())
        rand = np.mean([mass(rng.choice(n_neurons, k, replace=False).tolist()) for _ in range(3)])
        print(f"  top {k:3d} neurons: {top_mass:.3f} (-{100*(base-top_mass)/base:.0f}%)   "
              f"random {k}: {rand:.3f} (-{100*(base-rand)/base:.0f}%)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("artifacts/gemma4_mlp_rhyme/mlp13_neurons.npz"))
    p.add_argument("--bf16", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
