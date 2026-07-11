"""Command-line entry points for the first milestone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .analysis import counterfactual_anchor_patch, scan_head_ablation, scan_layer_ablation
from .dataset import (
    build_counterfactual_dataset,
    build_dataset,
    build_distinct_elicitation_dataset,
    build_elicitation_dataset,
    write_jsonl,
)
from .model import load_model


def _write(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        with path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
    else:
        pd.DataFrame(rows).to_csv(path, index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["dataset", "evaluate", "elicit", "distinct", "layers", "heads", "patch"])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--example", type=int, default=0, help="Base couplet index (plain wrapper)")
    parser.add_argument("--source", type=int, default=1, help="Source base couplet for patching")
    parser.add_argument("--device")
    parser.add_argument("--model", default=None, help="Hugging Face model name (defaults to Pythia-410M deduped)")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load with bitsandbytes NF4 quantization")
    parser.add_argument("--counts", default="1,3,5,10,15,20", help="Comma-separated demo counts for distinct")
    parser.add_argument(
        "--condition",
        choices=["plain", "rhyming", "shuffled", "reversed", "all"],
        default="all",
        help="Long-context condition for the elicit command",
    )
    args = parser.parse_args(argv)
    examples = build_dataset()
    if args.command == "dataset":
        path = args.output or Path("data/controlled_couplets.jsonl")
        write_jsonl(path, examples)
        print(f"Wrote {len(examples)} examples to {path}")
        return
    bundle = load_model(
        model_name=args.model or "EleutherAI/pythia-410m-deduped",
        device=args.device,
        load_in_4bit=args.load_in_4bit,
    )
    if args.command in {"evaluate", "elicit", "distinct"}:
        from .model import evaluate_examples
        if args.command == "evaluate":
            rows = evaluate_examples(examples, bundle, args.limit)
            path = args.output or Path("artifacts/behavior.jsonl")
        elif args.command == "elicit":
            conditions = ["plain", "rhyming", "shuffled", "reversed"] if args.condition == "all" else [args.condition]
            rows = []
            for condition in conditions:
                for row in evaluate_examples(build_elicitation_dataset(condition), bundle, args.limit):
                    rows.append({**row, "condition": condition})
            path = args.output or Path("artifacts/elicitation.jsonl")
        else:
            rows = []
            for count in [int(value) for value in args.counts.split(",")]:
                for shuffled in [False, True]:
                    condition = f"distinct_{count}" + ("_shuffled" if shuffled else "")
                    for row in evaluate_examples(build_distinct_elicitation_dataset(count, shuffled), bundle, args.limit):
                        rows.append({**row, "condition": condition, "demonstration_count": count, "shuffled": shuffled})
            path = args.output or Path("artifacts/distinct_elicitation.jsonl")
    else:
        # Eight wrappers per base couplet; use the plain version.
        example = examples[args.example * 8]
        if args.command == "layers":
            rows = scan_layer_ablation(example, bundle)
            path = args.output or Path("artifacts/layer_ablation.csv")
        elif args.command == "heads":
            rows = scan_head_ablation(example, bundle)
            path = args.output or Path("artifacts/head_ablation.csv")
        else:
            counterfactuals = build_counterfactual_dataset()
            example = counterfactuals[args.example]
            source = counterfactuals[args.source]
            rows = counterfactual_anchor_patch(source, example, bundle)
            path = args.output or Path("artifacts/counterfactual_patch.csv")
    _write(rows, path)
    print(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
