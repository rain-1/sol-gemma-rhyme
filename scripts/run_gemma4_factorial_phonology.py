"""Causal held-out vowel x coda recombination in Gemma 4's rhyme memory.

For each eligible target rhyme family, fit an additive model

    activation = intercept + vowel_effect + coda_effect

to E1 family centroids while excluding the target family.  Thus the target
vowel is learned only from other codas and the target coda only from other
vowels; those supporting family sets are necessarily disjoint.  Their sum is
an out-of-cell prediction of the unseen vowel+coda family.  We steer natural
couplet anchors from their observed source centroid toward that prediction at
the output of layer 13 (the input to causal layer-14 value storage).

The target-family split is fixed before model evaluation.  Alpha is selected
only on discovery families and frozen for confirmation families.  Controls
are norm-matched Gaussian vectors and label-shuffled additive models.
Cross-spelling target mass tests whether any effect extends beyond words with
the reference family's dominant written rime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rhyme_interp.dataset import build_elicitation_dataset
from rhyme_interp.families import rime_spelling
from rhyme_interp.model import load_model
from rhyme_interp.rhyme import rhyme_token_ids, rhymes
from run_gemma4_interpretability import (
    MODEL, anchor_positions, batch_inputs, forward_logits, write_jsonl,
)
from run_gemma4_steering import family_lookup

LAYER = 13
ALPHAS = (0.5, 1.0, 2.0, 4.0, 8.0)
SEED = 73021


def parts(family: str) -> tuple[str, str]:
    phones = family.split("-")
    return phones[0].rstrip("0123456789"), "-".join(phones[1:]) or "OPEN"


def additive_prediction(centroids: dict[str, np.ndarray], heldout: str,
                        labels: dict[str, tuple[str, str]] | None = None) -> np.ndarray:
    """Least-squares prediction for one excluded vowel×coda cell."""
    labels = labels or {f: parts(f) for f in centroids}
    train = [f for f in sorted(centroids) if f != heldout]
    vowels = sorted({labels[f][0] for f in train})
    codas = sorted({labels[f][1] for f in train})
    target_v, target_c = labels[heldout]
    if target_v not in vowels or target_c not in codas:
        raise ValueError(f"Held-out parts not represented for {heldout}")
    # Full one-hot design, solved by the minimum-norm pseudoinverse.  Despite
    # redundant intercept/effect columns, fitted cell predictions are unique.
    x = np.zeros((len(train), 1 + len(vowels) + len(codas)), np.float64)
    x[:, 0] = 1
    for i, family in enumerate(train):
        v, c = labels[family]
        x[i, 1 + vowels.index(v)] = 1
        x[i, 1 + len(vowels) + codas.index(c)] = 1
    y = np.stack([centroids[f] for f in train]).astype(np.float64)
    beta = np.linalg.pinv(x) @ y
    xt = np.zeros(x.shape[1]); xt[0] = 1
    xt[1 + vowels.index(target_v)] = 1
    xt[1 + len(vowels) + codas.index(target_c)] = 1
    return (xt @ beta).astype(np.float32)


def eligible_targets(families: dict[str, list[str]]) -> list[str]:
    labels = {f: parts(f) for f in families}
    return sorted(f for f, (v, c) in labels.items()
                  if sum(v == vv for vv, _ in labels.values()) > 1
                  and sum(c == cc for _, cc in labels.values()) > 1)


def fixed_split(targets: list[str]) -> dict[str, str]:
    """Stable preregistered split independent of activations/model outputs."""
    ordered = sorted(targets, key=lambda f: hashlib.sha256(("factorial-v1:" + f).encode()).hexdigest())
    return {f: ("discovery" if i % 2 == 0 else "confirmation") for i, f in enumerate(ordered)}


def bootstrap_ci(values, rng, reps=5000):
    values = np.asarray(values, float)
    if not len(values): return [float("nan"), float("nan")]
    means = values[rng.integers(0, len(values), (reps, len(values)))].mean(1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    rep = Path(args.representation)
    data = np.load(rep / "activations.npz")
    words = [str(x) for x in data["words"]]
    families = json.loads((rep / "families.json").read_text())
    states = data["states_final_word"].astype(np.float32)[:, LAYER + 1]
    word_i = {w: i for i, w in enumerate(words)}
    centroids = {f: states[[word_i[w] for w in members]].mean(0) for f, members in families.items()}
    targets = eligible_targets(families)
    split = fixed_split(targets)
    if args.phase != "all":
        targets = [f for f in targets if split[f] == args.phase]
    predictions = {f: additive_prediction(centroids, f) for f in targets}

    examples = [(e, family_lookup(families, e.anchor)) for e in build_elicitation_dataset("rhyming")]
    examples = [(e, f) for e, f in examples if f is not None]
    bundle = load_model(MODEL, load_in_4bit=not args.bf16, attn_implementation="eager")
    layers = bundle.model.model.language_model.layers
    rng = np.random.default_rng(SEED)
    rows = []

    # Evaluate target-by-target to keep GPU memory bounded.  Each batch contains
    # the same natural prompts, and only the causal anchor position is edited.
    for ti, target in enumerate(targets):
        batch = [(e, source) for e, source in examples if source != target]
        prompts = [e.prompt for e, _ in batch]
        anchors = [e.anchor for e, _ in batch]
        inputs = batch_inputs(prompts, bundle)
        pos = anchor_positions(prompts, anchors, bundle, inputs["input_ids"].shape[1])
        bix = torch.arange(len(batch), device=bundle.device)
        base_logits = forward_logits(inputs, bundle)
        target_ids = rhyme_token_ids(families[target][0], bundle.token_words)
        dominant = rime_spelling(families[target][0])
        cross_ids = [i for i in target_ids if rime_spelling(bundle.token_words[i]) != dominant]
        base_p = base_logits.softmax(-1)
        deltas = np.stack([predictions[target] - centroids[source] for _, source in batch])
        norms = np.linalg.norm(deltas, axis=1)

        variants = {"factorial": deltas}
        for k in range(args.control_repeats):
            z = rng.normal(size=deltas.shape).astype(np.float32)
            z *= (norms / np.linalg.norm(z, axis=1).clip(1e-8))[:, None]
            variants[f"random_{k}"] = z
            # Shuffle vowel/coda labels globally, refit, and norm-match its delta.
            fam_order = sorted(families)
            permuted = list(fam_order); rng.shuffle(permuted)
            shuffled_centroids = {label: centroids[donor] for label, donor in zip(fam_order, permuted)}
            placebo = additive_prediction(shuffled_centroids, target)
            sd = np.stack([placebo - centroids[source] for _, source in batch])
            sd *= (norms / np.linalg.norm(sd, axis=1).clip(1e-8))[:, None]
            variants[f"shuffled_{k}"] = sd

        alphas = ALPHAS if split[target] == "discovery" else (args.confirmation_alpha,)
        for variant, delta_np in variants.items():
            alphas_here = alphas
            delta = torch.tensor(delta_np, device=bundle.device)
            for alpha in alphas_here:
                def hook(_module, _args, output):
                    hidden = output.clone()
                    hidden[bix, pos] += alpha * delta.to(hidden.dtype)
                    return hidden
                handle = layers[LAYER].register_forward_hook(hook)
                logits = forward_logits(inputs, bundle)
                handle.remove()
                probs = logits.softmax(-1)
                for i, (example, source) in enumerate(batch):
                    top_id = int(logits[i].argmax())
                    top_word = bundle.token_words.get(top_id, bundle.tokenizer.decode([top_id]))
                    rows.append({
                        "target_family": target, "split": split[target], "source_family": source,
                        "prompt_id": example.id, "variant": variant, "alpha": alpha,
                        "vector_norm": float(norms[i]),
                        "baseline_target_mass": float(base_p[i, target_ids].sum()),
                        "target_mass": float(probs[i, target_ids].sum()),
                        "delta_target_mass": float(probs[i, target_ids].sum() - base_p[i, target_ids].sum()),
                        "baseline_cross_spelling_mass": float(base_p[i, cross_ids].sum()) if cross_ids else None,
                        "cross_spelling_mass": float(probs[i, cross_ids].sum()) if cross_ids else None,
                        "top1": top_word, "top1_in_target": bool(rhymes(top_word, families[target][0])),
                        "top1_cross_spelling": bool(rhymes(top_word, families[target][0]) and rime_spelling(top_word) != dominant),
                    })
        print(f"{ti+1}/{len(targets)} {target} ({split[target]})")
    write_jsonl(args.output / f"factorial_rows_{args.phase}.jsonl", rows)

    # Discovery-only alpha selection, then confirmation summary. Macro-average
    # family means so a family, rather than a prompt, is the unit of inference.
    disc = [r for r in rows if r["split"] == "discovery" and r["variant"] == "factorial"]
    scores = ({a: np.mean([r["delta_target_mass"] for r in disc if r["alpha"] == a]) for a in ALPHAS}
              if disc else {})
    selected = max(scores, key=scores.get) if scores else args.confirmation_alpha
    summary = {"seed": SEED, "layer": LAYER, "precision": "bf16" if args.bf16 else "nf4",
               "targets": targets, "split": split, "discovery_alpha_scores": scores,
               "selected_alpha": selected, "confirmation_alpha": args.confirmation_alpha,
               "n_prompts": len(examples), "control_repeats": args.control_repeats}
    eval_split = "confirmation" if any(r["split"] == "confirmation" for r in rows) else "discovery"
    eval_alpha = args.confirmation_alpha if eval_split == "confirmation" else selected
    fam_rows = []
    for variant_group, predicate in {
        "factorial": lambda v: v == "factorial",
        "random": lambda v: v.startswith("random_"),
        "shuffled": lambda v: v.startswith("shuffled_"),
    }.items():
        use = [r for r in rows if r["split"] == eval_split and predicate(r["variant"])
               and r["alpha"] == eval_alpha]
        byfam = {}
        for f in sorted({r["target_family"] for r in use}):
            vals = [r["delta_target_mass"] for r in use if r["target_family"] == f]
            row = {"variant": variant_group, "target_family": f, "n": len(vals),
                   "mean_delta_target_mass": float(np.mean(vals)),
                   "prompt_bootstrap_95ci": bootstrap_ci(vals, rng),
                   "top1_in_target_rate": float(np.mean([r["top1_in_target"] for r in use
                                                          if r["target_family"] == f]))}
            fam_rows.append(row); byfam[f] = row["mean_delta_target_mass"]
        vals = list(byfam.values())
        summary[variant_group] = {"family_macro_mean_delta_target_mass": float(np.mean(vals)),
                                  "family_bootstrap_95ci": bootstrap_ci(vals, rng), "n_families": len(vals)}
    fact_by = {r["target_family"]: r["mean_delta_target_mass"] for r in fam_rows if r["variant"] == "factorial"}
    for control in ("random", "shuffled"):
        ctrl_by = {r["target_family"]: r["mean_delta_target_mass"] for r in fam_rows if r["variant"] == control}
        paired = [fact_by[f] - ctrl_by[f] for f in fact_by]
        summary[f"factorial_minus_{control}"] = {
            "family_macro_mean": float(np.mean(paired)), "family_bootstrap_95ci": bootstrap_ci(paired, rng)
        }
    cross = [r["cross_spelling_mass"] - r["baseline_cross_spelling_mass"] for r in rows
             if r["split"] == eval_split and r["variant"] == "factorial"
             and r["alpha"] == eval_alpha and r["cross_spelling_mass"] is not None]
    summary["spelling_control"] = {"mean_delta_cross_spelling_mass": float(np.mean(cross)),
                                    "bootstrap_95ci_prompt": bootstrap_ci(cross, rng), "n": len(cross)}
    write_jsonl(args.output / f"per_family_{args.phase}.jsonl", fam_rows)
    (args.output / f"summary_{args.phase}.json").write_text(json.dumps(summary, indent=2))
    if args.phase == "discovery":
        (args.output / "selected_alpha.json").write_text(json.dumps({"selected_alpha": selected, "scores": scores}, indent=2))
    (args.output / "split.json").write_text(json.dumps({"seed": SEED, "targets": split}, indent=2))
    print(json.dumps(summary, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--representation", type=Path, default=Path("artifacts/gemma4_representation"))
    p.add_argument("--output", type=Path, default=Path("artifacts/gemma4_factorial_phonology"))
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--control-repeats", type=int, default=3)
    p.add_argument("--confirmation-alpha", type=float, default=2.0,
                   help="Frozen before confirmation evaluation; default inherited from E3")
    p.add_argument("--phase", choices=("discovery", "confirmation", "all"), default="discovery")
    run(p.parse_args())


if __name__ == "__main__": main()
