"""Where is rhyme written? Locate the phonological code in Gemma 4 E2B.

Three questions, building on the Phase 2 finding that rhyme family is linearly
decodable from the layer-13 anchor residual:

  E1. Is the rhyme code a fixed property of the word, or is it computed only
      when the context asks for a rhyme? (multiple neutral vs rhyming scaffolds)
  E2. Which sublayers write it: does the attention stream or the MLP stream
      carry the decodable rhyme code up to layer 13? (per-layer decomposition)
  E3. Is the MLP write causal? Zero each stream's contribution at the anchor
      position and measure the collapse of the final-token rhyme prediction.

In Gemma 4 each block adds  h += post_attention_layernorm(attn(h));
h += post_feedforward_layernorm(mlp(h)). Those two norm outputs are the additive
attention and MLP updates, so hooking them gives an exact residual decomposition
and an exact per-stream ablation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rhyme_interp.dataset import RHYME_DEMONSTRATION_LINES, build_elicitation_dataset
from rhyme_interp.model import DEFAULT_REVISIONS, load_model
from rhyme_interp.rhyme import rhyme_token_ids
from run_gemma4_interpretability import MODEL, anchor_positions, batch_inputs, write_jsonl

DEMO = "\n".join(RHYME_DEMONSTRATION_LINES)
RHYMING_SCAFFOLDS = {
    "final_word": f"{DEMO}\nThe final word upon the page was {{word}}",
    "line_end": f"{DEMO}\nEvery line she wrote would end in {{word}}",
    "couplet_rhyme": f"{DEMO}\nHe searched for a word that would rhyme, and chose {{word}}",
}
NEUTRAL_SCAFFOLDS = {
    "heard": "I heard the word {word}",
    "dictionary": "The dictionary contained an entry for the word {word}",
    "spelled": "She slowly spelled out the word {word}",
}
WRITE_LAYERS = list(range(14))  # blocks 0..13, up to the storage layer
PROBE_SEED = 0


def build_words(bundle, min_per_family=4, cap_per_family=8):
    fam = json.load(open("artifacts/gemma4_representation/families.json"))
    words, labels = [], []
    for family, members in fam.items():
        kept = []
        for w in members:
            ids = bundle.tokenizer(" " + w, add_special_tokens=False)["input_ids"]
            if len(ids) == 1:
                kept.append(w)
            if len(kept) >= cap_per_family:
                break
        if len(kept) >= min_per_family:
            words += kept
            labels += [family] * len(kept)
    return words, labels


def probe_accuracy(X, y, seed=PROBE_SEED):
    Xc = X - X.mean(0)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
    return float(cross_val_score(clf, Xc, y, cv=5).mean())


@torch.inference_mode()
def capture_layer(prompts, bundle, layer, batch_size=32):
    out = []
    for s in range(0, len(prompts), batch_size):
        inp = batch_inputs(prompts[s : s + batch_size], bundle)
        hs = bundle.model(**inp, use_cache=False, output_hidden_states=True).hidden_states
        out.append(hs[layer][:, -1].float().cpu().numpy())
    return np.concatenate(out)


@torch.inference_mode()
def capture_stream_updates(prompts, bundle, batch_size=32):
    """Per-layer additive attention and MLP updates at the final position."""
    layers = bundle.model.model.language_model.layers
    n = len(layers)
    attn_all, mlp_all = [], []
    for s in range(0, len(prompts), batch_size):
        attn = [None] * n
        mlp = [None] * n
        handles = []
        for i in range(n):
            handles.append(layers[i].post_attention_layernorm.register_forward_hook(
                lambda m, a, o, i=i: attn.__setitem__(i, o[:, -1].detach().float().cpu())))
            handles.append(layers[i].post_feedforward_layernorm.register_forward_hook(
                lambda m, a, o, i=i: mlp.__setitem__(i, o[:, -1].detach().float().cpu())))
        inp = batch_inputs(prompts[s : s + batch_size], bundle)
        bundle.model(**inp, use_cache=False)
        for h in handles:
            h.remove()
        attn_all.append(torch.stack(attn, 1).numpy())  # (batch, layers, width)
        mlp_all.append(torch.stack(mlp, 1).numpy())
    return np.concatenate(attn_all), np.concatenate(mlp_all)


def experiment_context(bundle, words, labels):
    """E1: rhyme decodability at layer 13 across neutral and rhyming scaffolds."""
    y = np.array([sorted(set(labels)).index(l) for l in labels])
    rows = []
    for kind, scaffolds in [("neutral", NEUTRAL_SCAFFOLDS), ("rhyming", RHYMING_SCAFFOLDS)]:
        for name, template in scaffolds.items():
            X = capture_layer([template.format(word=w) for w in words], bundle, 13)
            acc = probe_accuracy(X, y)
            rng = np.random.default_rng(PROBE_SEED)
            shuffled = probe_accuracy(X, rng.permutation(y))
            rows.append({"experiment": "context", "kind": kind, "scaffold": name,
                         "layer": 13, "probe_accuracy": acc, "shuffled_accuracy": shuffled})
            print(f"  E1 {kind:8s} {name:12s}: probe {acc:.3f} (shuffled {shuffled:.3f})")
    return rows


def experiment_streams(bundle, words, labels):
    """E2: cumulative attention vs MLP contribution to rhyme, layer by layer."""
    y = np.array([sorted(set(labels)).index(l) for l in labels])
    prompts = [RHYMING_SCAFFOLDS["line_end"].format(word=w) for w in words]
    attn, mlp = capture_stream_updates(prompts, bundle)  # (n, layers, width)
    emb = bundle.model.get_input_embeddings().weight
    EMB = np.array([emb[bundle.tokenizer(" " + w, add_special_tokens=False)["input_ids"][0]]
                    .detach().float().cpu().numpy() for w in words])
    rows = [{"experiment": "streams", "component": "embedding", "layer": -1,
             "probe_accuracy": probe_accuracy(EMB, y)}]
    for L in range(attn.shape[1]):
        a_cum = attn[:, : L + 1].sum(1)
        m_cum = mlp[:, : L + 1].sum(1)
        rows.append({"experiment": "streams", "component": "attention_cumulative",
                     "layer": L, "probe_accuracy": probe_accuracy(a_cum, y)})
        rows.append({"experiment": "streams", "component": "mlp_cumulative",
                     "layer": L, "probe_accuracy": probe_accuracy(m_cum, y)})
    for r in rows:
        if r["layer"] in (-1, 2, 5, 9, 13, 20, 27):
            print(f"  E2 {r['component']:22s} L{r['layer']:>2}: {r['probe_accuracy']:.3f}")
    return rows


@torch.inference_mode()
def family_mass(bundle, examples, ablate=None, layers=None):
    """Mean probability on each anchor's rhyme family at the blank, optionally
    zeroing a stream's contribution at the anchor position in `layers`."""
    prompts = [e.prompt for e in examples]
    inp = batch_inputs(prompts, bundle)
    positions = anchor_positions(prompts, [e.anchor for e in examples], bundle,
                                 inp["input_ids"].shape[1])
    batch = torch.arange(len(examples), device=bundle.device)
    net = bundle.model.model.language_model.layers
    handles = []
    if ablate is not None:
        module = "post_feedforward_layernorm" if ablate == "mlp" else "post_attention_layernorm"
        for L in layers:
            def hook(_m, _a, o, pos=positions):
                o = o.clone()
                o[batch, pos] = 0
                return o
            handles.append(getattr(net[L], module).register_forward_hook(hook))
    logits = bundle.model(**inp, use_cache=False, logits_to_keep=1).logits[:, -1].float()
    for h in handles:
        h.remove()
    probs = logits.softmax(-1)
    masses = []
    for i, e in enumerate(examples):
        ids = rhyme_token_ids(e.anchor, bundle.token_words)
        masses.append(float(probs[i, ids].sum()))
    return float(np.mean(masses)), masses


def experiment_ablation(bundle):
    """E3: causal MLP vs attention ablation at the anchor position."""
    examples = build_elicitation_dataset("rhyming")
    base, base_each = family_mass(bundle, examples)
    rows = [{"experiment": "ablation", "stream": "none", "layers": "-", "mass": base}]
    print(f"  E3 baseline anchor-family mass: {base:.3f}")
    bands = {"L13": [13], "L11-13": [11, 12, 13], "L0-13": WRITE_LAYERS}
    for band_name, band in bands.items():
        for stream in ("mlp", "attention"):
            mass, each = family_mass(bundle, examples, ablate=stream, layers=band)
            drop = 100 * (base - mass) / base
            rows.append({"experiment": "ablation", "stream": stream, "layers": band_name,
                         "mass": mass, "relative_drop_pct": drop})
            print(f"  E3 ablate {stream:9s} @ {band_name:7s}: mass {mass:.3f}  (-{drop:.0f}%)")
    return rows


def experiment_layer_sweep(bundle):
    """E4: knock out a single layer's MLP (or attention) write at the anchor and
    measure the collapse of the rhyme code, one layer at a time."""
    examples = build_elicitation_dataset("rhyming")
    base, _ = family_mass(bundle, examples)
    rows = [{"experiment": "layer_sweep", "stream": "none", "layer": -1, "mass": base}]
    print(f"  E4 baseline anchor-family mass: {base:.3f}")
    for stream in ("mlp", "attention"):
        for L in range(24):
            mass, _ = family_mass(bundle, examples, ablate=stream, layers=[L])
            rows.append({"experiment": "layer_sweep", "stream": stream, "layer": L,
                         "mass": mass, "relative_drop_pct": 100 * (base - mass) / base})
        worst = sorted((r for r in rows if r["stream"] == stream),
                       key=lambda r: r["mass"])[:4]
        print(f"  E4 {stream:9s} most damaging layers: " +
              ", ".join(f"L{r['layer']}({r['mass']:.2f})" for r in worst))
    return rows


def run(args):
    bundle = load_model(MODEL, load_in_4bit=not args.bf16, attn_implementation="eager")
    bundle.token_words = bundle.token_words  # single_token_words already attached
    words, labels = build_words(bundle)
    print(f"{len(words)} single-token words across {len(set(labels))} families "
          f"(>=4 members); chance = {1/len(set(labels)):.3f}")
    rows = []
    if not args.only or args.only == "context":
        print("E1 context-gating:")
        rows += experiment_context(bundle, words, labels)
    if not args.only or args.only == "streams":
        print("E2 stream decomposition:")
        rows += experiment_streams(bundle, words, labels)
    if not args.only or args.only == "ablation":
        print("E3 causal ablation:")
        rows += experiment_ablation(bundle)
    if not args.only or args.only == "sweep":
        print("E4 per-layer knockout sweep:")
        rows += experiment_layer_sweep(bundle)

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "mlp_rhyme.jsonl"
    if args.only and path.exists():  # keep the other experiments' rows
        kept = [json.loads(l) for l in open(path) if json.loads(l)["experiment"] != args.only]
        rows = kept + rows
    write_jsonl(path, rows)
    summary = {
        "n_words": len(words), "n_families": len(set(labels)),
        "context": {r["kind"] + "/" + r["scaffold"]: r["probe_accuracy"]
                    for r in rows if r["experiment"] == "context"},
        "streams_layer13": {r["component"]: r["probe_accuracy"] for r in rows
                            if r["experiment"] == "streams" and r["layer"] == 13},
        "ablation": {f"{r['stream']}@{r['layers']}": r["mass"] for r in rows
                     if r["experiment"] == "ablation"},
    }
    (args.output / "mlp_rhyme_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {len(rows)} rows to {args.output / 'mlp_rhyme.jsonl'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/gemma4_mlp_rhyme"))
    parser.add_argument("--only", choices=["context", "streams", "ablation", "sweep"],
                        help="run and refresh a single experiment, keeping the others")
    parser.add_argument("--bf16", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
