"""Where is the causal rhyme write in layer-13, if not the selective neurons?

Report 09: the 16 most family-selective MLP-13 neurons read the family at 0.79
but ablating them costs only 6% of the rhyme prediction, while the whole MLP is
worth 73%. So the causal signal is elsewhere. Two candidates:

  A. output-heavy neurons -- large down-projection weight into the family
     directions, even if their activation variance is unremarkable;
  B. a low-rank DIRECTION written in superposition across many neurons, so no
     neuron subset is causal but a few directions are.

We rank neurons by output-weighted contribution (A) and ablate them, and we
directly ablate the family subspace from the layer-13 output (B).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from rhyme_interp.dataset import RHYME_DEMONSTRATION_LINES, build_elicitation_dataset
from rhyme_interp.model import load_model
from rhyme_interp.rhyme import rhyme_token_ids
from run_gemma4_interpretability import MODEL, anchor_positions, batch_inputs
from sklearn.feature_selection import f_classif

LAYER = 13
SCAFFOLD = "\n".join(RHYME_DEMONSTRATION_LINES) + "\nEvery line she wrote would end in {word}"


@torch.inference_mode()
def capture(bundle, words, batch_size=32):
    """MLP-13 neuron activations (down_proj input) and block-13 output, last token."""
    net = bundle.model.model.language_model.layers[LAYER]
    acts, outs = [], []
    for s in range(0, len(words), batch_size):
        buf = {}
        h = net.mlp.down_proj.register_forward_pre_hook(
            lambda m, a: buf.__setitem__("x", a[0][:, -1].detach().float().cpu()))
        hs = bundle.model(**batch_inputs([SCAFFOLD.format(word=w) for w in words[s:s + batch_size]], bundle),
                          use_cache=False, output_hidden_states=True).hidden_states
        h.remove()
        acts.append(buf["x"]); outs.append(hs[LAYER + 1][:, -1].detach().float().cpu())
    return torch.cat(acts).numpy(), torch.cat(outs).numpy()


def run(args):
    fam = json.loads(Path("artifacts/gemma4_representation/families.json").read_text())
    bundle = load_model(MODEL, load_in_4bit=False, attn_implementation="eager")  # bf16 for real weights
    net = bundle.model.model.language_model.layers[LAYER]
    W_down = net.mlp.down_proj.weight.detach().float().cpu().numpy()  # (d_model, n_neurons)

    words, labels = [], []
    for name, members in fam.items():
        for w in members:
            if len(bundle.tokenizer(" " + w, add_special_tokens=False)["input_ids"]) == 1:
                words.append(w); labels.append(name)
    acts, outs = capture(bundle, words)
    families = sorted(set(labels))
    yf = np.array([families.index(l) for l in labels])
    global_mean = outs.mean(0)
    dirs = {f: outs[yf == i].mean(0) - global_mean for i, f in enumerate(families)}
    dir_hat = {f: d / (np.linalg.norm(d) + 1e-9) for f, d in dirs.items()}

    # --- rankings ---
    Ff, _ = f_classif(acts, yf); Ff = np.nan_to_num(Ff)
    rank_selective = np.argsort(-Ff)
    # output-weighted: for word i, neuron j contributes acts[i,j]*(W_down[:,j].dir_hat(fam_i))
    D = np.stack([dir_hat[l] for l in labels])            # (words, d_model)
    WdotD = D @ W_down                                    # (words, n_neurons)
    contrib = np.abs(acts * WdotD).mean(0)                # mean |signed contribution to correct family|
    rank_output = np.argsort(-contrib)

    # --- couplet setup for causal ablation ---
    examples = build_elicitation_dataset("rhyming")
    prompts = [e.prompt for e in examples]
    inputs = batch_inputs(prompts, bundle)
    pos = anchor_positions(prompts, [e.anchor for e in examples], bundle, inputs["input_ids"].shape[1])
    batch = torch.arange(len(examples), device=bundle.device)
    ids = [rhyme_token_ids(e.anchor, bundle.token_words) for e in examples]

    @torch.inference_mode()
    def rhyme_mass(neuron_hook=None, dir_hook=None):
        handles = []
        if neuron_hook is not None:
            cols = torch.tensor(neuron_hook, device=bundle.device)
            def nh(_m, a):
                x = a[0].clone(); x[batch[:, None], pos[:, None], cols[None, :]] = 0
                return (x,) + a[1:]
            handles.append(net.mlp.down_proj.register_forward_pre_hook(nh))
        if dir_hook is not None:
            P = torch.tensor(dir_hook, device=bundle.device, dtype=torch.float32)
            def dh(_m, _a, output):
                h = output[0] if isinstance(output, tuple) else output
                v = h[batch, pos].float()
                h[batch, pos] = (v - v @ P).to(h.dtype)
                return output
            handles.append(net.register_forward_hook(dh))
        logits = bundle.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        for h in handles:
            h.remove()
        probs = logits.softmax(-1)
        return float(np.mean([probs[i, ids[i]].sum().item() for i in range(len(examples))]))

    base = rhyme_mass()
    rng = np.random.default_rng(0)
    print(f"baseline rhyme-family mass {base:.3f}\n")

    print("A. ablate top-k neurons by ranking (mass, % drop):")
    print(f"{'k':>6} {'selective':>16} {'output-weighted':>18} {'random':>12}")
    for k in [16, 64, 256, 1024]:
        m_sel = rhyme_mass(neuron_hook=rank_selective[:k].tolist())
        m_out = rhyme_mass(neuron_hook=rank_output[:k].tolist())
        m_rnd = np.mean([rhyme_mass(neuron_hook=rng.choice(acts.shape[1], k, replace=False).tolist())
                         for _ in range(2)])
        d = lambda m: f"{m:.3f} (-{100*(base-m)/base:2.0f}%)"
        print(f"{k:>6} {d(m_sel):>16} {d(m_out):>18} {d(m_rnd):>12}")

    # B. directional ablation: project the family subspace out of the layer-13 output
    M = np.stack([dirs[f] for f in families])
    _, _, Vt = np.linalg.svd(M - M.mean(0), full_matrices=False)
    print("\nB. project out a rank-r subspace of the layer-13 output at the anchor:")
    print(f"{'rank':>6} {'family subspace':>18} {'random subspace':>18}")
    for r in [1, 4, 16, 64]:
        B = Vt[:r]
        P_fam = B.T @ B
        Q, _ = np.linalg.qr(rng.standard_normal((M.shape[1], r)))
        P_rnd = Q @ Q.T
        m_fam = rhyme_mass(dir_hook=P_fam.astype(np.float32))
        m_rnd = rhyme_mass(dir_hook=P_rnd.astype(np.float32))
        d = lambda m: f"{m:.3f} (-{100*(base-m)/base:2.0f}%)"
        print(f"{r:>6} {d(m_fam):>18} {d(m_rnd):>18}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bf16", action="store_true", default=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
