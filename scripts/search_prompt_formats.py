"""Search poem formatting on a development split and report held-out accuracy.

This is deliberately a universal-format search: a configuration is selected
using even-indexed poems and then evaluated without modification on odd-indexed
poems. It never selects a separate prompt for each target.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

import torch

from rhyme_interp.dataset import COUPLETS, DISTINCT_DEMONSTRATION_COUPLETS
from rhyme_interp.model import load_model
from rhyme_interp.rhyme import rhymes


@dataclass(frozen=True)
class Format:
    header: str
    demo_count: int
    demo_separator: str
    before_target: str
    first_suffix: str
    line_separator: str
    indent: str
    numbered: bool


HEADERS = [
    "",
    "Poem:\n",
    "A Poem\n",
    "A Rhyming Poem\n",
    "Rhyming Verse\n",
    "Verse:\n",
    "Poetry\n",
    "Untitled\n",
    "SONG\n",
    "Lyrics:\n",
]
DEMO_COUNTS = [0, 1, 3, 5, 10]
DEMO_SEPARATORS = ["\n", "\n\n"]
BEFORE_TARGETS = ["\n", "\n\n", "\n\n---\n\n"]
FIRST_SUFFIXES = ["", ",", ".", ";"]
LINE_SEPARATORS = ["\n", "\n\n"]
INDENTS = ["", "  ", "    "]


def formats() -> list[Format]:
    values = []
    for header in HEADERS:
        for demo_count in DEMO_COUNTS:
            for demo_separator in DEMO_SEPARATORS:
                for before_target in BEFORE_TARGETS:
                    for first_suffix in FIRST_SUFFIXES:
                        for line_separator in LINE_SEPARATORS:
                            for indent in INDENTS:
                                for numbered in [False, True]:
                                    # Avoid duplicate configs where separators cannot matter.
                                    if demo_count == 0 and (demo_separator != "\n" or before_target != "\n"):
                                        continue
                                    values.append(Format(header, demo_count, demo_separator, before_target,
                                                         first_suffix, line_separator, indent, numbered))
    return values


def render(config: Format, couplet_index: int) -> str:
    anchor, _target, first, second = COUPLETS[couplet_index]
    pieces = [config.header]
    if config.demo_count:
        demos = DISTINCT_DEMONSTRATION_COUPLETS[: config.demo_count]
        blocks = []
        for left, right in demos:
            blocks.append(config.indent + left + "\n" + config.indent + right)
        pieces.append(config.demo_separator.join(blocks))
        pieces.append(config.before_target)
    if config.numbered:
        pieces.append("1. " + first + config.first_suffix + config.line_separator + "2. " + second)
    else:
        pieces.append(config.indent + first + config.first_suffix + config.line_separator + config.indent + second)
    return "".join(pieces)


@torch.inference_mode()
def top_words(prompts: list[str], bundle, batch_size: int) -> list[str]:
    tokenizer = bundle.tokenizer
    original_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    words = []
    for start in range(0, len(prompts), batch_size):
        batch = tokenizer(prompts[start : start + batch_size], padding=True, return_tensors="pt").to(bundle.device)
        ids = bundle.model(**batch).logits[:, -1].argmax(-1).tolist()
        words.extend(bundle.token_words.get(token_id, tokenizer.decode([token_id])) for token_id in ids)
    tokenizer.padding_side = original_side
    return words


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/prompt_format_search.jsonl"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top", type=int, default=100, help="Dev-selected formats to evaluate on held-out poems")
    parser.add_argument("--model", default="EleutherAI/pythia-410m-deduped")
    parser.add_argument("--revision", help="Exact Hugging Face model revision")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--max-formats", type=int, default=None, help="Deterministically sample this many formats")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    bundle = load_model(args.model, revision=args.revision, load_in_4bit=args.load_in_4bit)
    configs = formats()
    if args.max_formats and args.max_formats < len(configs):
        configs = random.Random(args.seed).sample(configs, args.max_formats)
    dev_indices = list(range(0, len(COUPLETS), 2))
    test_indices = list(range(1, len(COUPLETS), 2))

    dev_prompts = [render(config, index) for config in configs for index in dev_indices]
    dev_words = top_words(dev_prompts, bundle, args.batch_size)
    scored = []
    width = len(dev_indices)
    for config_index, config in enumerate(configs):
        words = dev_words[config_index * width : (config_index + 1) * width]
        correct = sum(rhymes(COUPLETS[index][0], word) for index, word in zip(dev_indices, words))
        scored.append((correct, config_index, words))
    scored.sort(reverse=True)

    # Evaluate only dev-nominated formats on the untouched odd-indexed poems.
    nominated = scored[: args.top]
    test_prompts = [render(configs[config_index], index) for _, config_index, _ in nominated for index in test_indices]
    test_words = top_words(test_prompts, bundle, args.batch_size)
    rows = []
    width = len(test_indices)
    for rank, (dev_correct, config_index, dev_words_for_config) in enumerate(nominated, 1):
        words = test_words[(rank - 1) * width : rank * width]
        test_correct = sum(rhymes(COUPLETS[index][0], word) for index, word in zip(test_indices, words))
        rows.append({
            "dev_rank": rank,
            "config_index": config_index,
            "dev_correct": dev_correct,
            "dev_total": len(dev_indices),
            "test_correct": test_correct,
            "test_total": len(test_indices),
            "overall_correct": dev_correct + test_correct,
            "overall_total": len(COUPLETS),
            "format": asdict(configs[config_index]),
            "model": args.model,
            "dev_words": dev_words_for_config,
            "test_words": words,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    best_heldout = max(rows, key=lambda row: (row["test_correct"], row["overall_correct"]))
    print(f"Searched {len(configs)} universal formats on {len(dev_indices)} development poems")
    print(f"Best dev score: {rows[0]['dev_correct']}/{len(dev_indices)}; held-out: {rows[0]['test_correct']}/{len(test_indices)}")
    print("Best held-out score among nominated formats:")
    print(json.dumps(best_heldout, indent=2))


if __name__ == "__main__":
    main()
