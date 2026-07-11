"""Pythia model loading and behavioral evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .rhyme import rhyme_token_ids, rhymes, single_token_words


DEFAULT_MODEL = "EleutherAI/pythia-410m-deduped"


@dataclass
class ModelBundle:
    model: object
    tokenizer: object
    device: torch.device
    token_words: dict[int, str]


def load_model(
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
    load_in_4bit: bool = False,
) -> ModelBundle:
    resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"dtype": "auto"}
    if load_in_4bit:
        kwargs.update(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            ),
            device_map={"": str(resolved)},
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if not load_in_4bit:
        model.to(resolved)
    model.eval()
    return ModelBundle(model, tokenizer, resolved, single_token_words(tokenizer))


def target_token_id(tokenizer, word: str) -> int | None:
    ids = tokenizer.encode(" " + word, add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


def score_logits(logits: torch.Tensor, anchor: str, target: str, bundle: ModelBundle, top_k: int = 10) -> dict:
    logits = logits.float()
    probs = logits.softmax(-1)
    rhyme_ids = rhyme_token_ids(anchor, bundle.token_words)
    rhyme_set = set(rhyme_ids)
    top = torch.topk(logits, top_k).indices.tolist()
    top_words = [bundle.token_words.get(i, bundle.tokenizer.decode([i])) for i in top]
    target_id = target_token_id(bundle.tokenizer, target)
    rank = None
    target_prob = None
    if target_id is not None:
        target_prob = float(probs[target_id])
        rank = int((logits > logits[target_id]).sum()) + 1
    rhyme_mass = float(probs[rhyme_ids].sum()) if rhyme_ids else 0.0
    # Normalize by family size to make anchors with many dictionary rhymes comparable.
    rhyme_mean_logit = float(logits[rhyme_ids].mean()) if rhyme_ids else math.nan
    eligible = list(bundle.token_words)
    non_rhyme = [i for i in eligible if i not in rhyme_set]
    non_rhyme_mean_logit = float(logits[non_rhyme].mean())
    return {
        "top1": top_words[0],
        "top1_rhymes": rhymes(anchor, top_words[0]),
        "top10": top_words,
        "top10_has_rhyme": any(i in rhyme_set for i in top),
        "rhyme_mass": rhyme_mass,
        "rhyme_family_size": len(rhyme_ids),
        "rhyme_logit_advantage": rhyme_mean_logit - non_rhyme_mean_logit,
        "target_probability": target_prob,
        "target_rank": rank,
    }


@torch.inference_mode()
def next_token_logits(prompt: str, bundle: ModelBundle) -> torch.Tensor:
    inputs = bundle.tokenizer(prompt, return_tensors="pt").to(bundle.device)
    return bundle.model(**inputs).logits[0, -1].detach()


def evaluate_examples(examples, bundle: ModelBundle, limit: int | None = None) -> list[dict]:
    rows = []
    for example in examples[:limit]:
        if target_token_id(bundle.tokenizer, example.anchor) is None:
            raise ValueError(f"Benchmark anchor is not one token: {example.anchor}")
        if target_token_id(bundle.tokenizer, example.target) is None:
            raise ValueError(f"Benchmark target is not one token: {example.target}")
        metrics = score_logits(next_token_logits(example.prompt, bundle), example.anchor, example.target, bundle)
        rows.append({**example.to_dict(), **metrics})
    return rows
