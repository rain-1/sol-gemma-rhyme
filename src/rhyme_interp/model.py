"""Pythia model loading and behavioral evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .rhyme import rhyme_token_ids, rhymes, single_token_words


DEFAULT_MODEL = "EleutherAI/pythia-410m-deduped"
GEMMA4_MODEL = "google/gemma-4-E2B"
# A model name is mutable. This is the exact snapshot used for the reported
# Gemma experiments and is applied unless a caller explicitly requests another.
DEFAULT_REVISIONS = {
    GEMMA4_MODEL: "19f17d3255f458aa49ebe8843d65ec7b7386db1f",
}


def is_gemma4(model_name: str) -> bool:
    """Return whether a Hub id or local path identifies Gemma 4."""
    return "gemma-4-" in model_name.lower()


def resolve_load_options(
    model_name: str, revision: str | None = None, attn_implementation: str | None = None,
) -> tuple[str | None, str | None]:
    """Resolve reproducible/safe defaults without requiring model weights."""
    resolved_revision = revision if revision is not None else DEFAULT_REVISIONS.get(model_name)
    # Gemma 4's SDPA path in the tested Transformers revision mishandles left
    # padding. Never leave correctness dependent on each script remembering this.
    resolved_attention = "eager" if is_gemma4(model_name) else attn_implementation
    return resolved_revision, resolved_attention


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
    attn_implementation: str | None = None,
    revision: str | None = None,
) -> ModelBundle:
    """Load a causal LM for analysis.

    Gemma 4 batch analyses automatically use `attn_implementation="eager"`: on the
    pinned Transformers development revision the sdpa path silently ignores
    the attention mask, so left-padded rows attend to their EOS padding and
    produce corrupted logits. Eager matches unbatched outputs exactly.
    """
    revision, attn_implementation = resolve_load_options(model_name, revision, attn_implementation)
    resolved = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    hub_kwargs = {"revision": revision} if revision else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **hub_kwargs)
    tokenizer.pad_token = tokenizer.eos_token
    kwargs = {"dtype": "auto"}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
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
    model = AutoModelForCausalLM.from_pretrained(model_name, **hub_kwargs, **kwargs)
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
