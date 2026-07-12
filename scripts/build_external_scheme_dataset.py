"""Build the external scheme corpus from preserved Claude CLI responses.

This is intentionally deterministic: candidates are visited in source-file and
response order, CMUdict-invalid poems are logged, and the first 30 valid poems
per newly generated scheme are selected.  Existing ABAB poems are retained as
the independent comparison set, with their older provenance gap kept explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path

from rhyme_interp.rhyme import rhyme_keys, rhymes


PAIRINGS = {"aabb": ((0, 1), (2, 3)), "abab": ((0, 2), (1, 3)), "abba": ((0, 3), (1, 2))}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ending(line: str) -> str:
    return re.sub(r"[^a-z]", "", line.rsplit(" ", 1)[-1].lower())


def response_candidates(path: Path) -> tuple[list[dict], dict]:
    envelope = json.loads(path.read_text())
    text = envelope["result"].strip()
    if text.startswith("```json"):
        text = text[len("```json"):]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip()), envelope


def rejection(lines: list[str], scheme: str) -> str | None:
    if len(lines) != 4 or any(not isinstance(line, str) for line in lines):
        return "not four string lines"
    words = [ending(line) for line in lines]
    if any(not word or not rhyme_keys(word) for word in words):
        return "ending absent from CMUdict"
    (a, b), (c, d) = PAIRINGS[scheme]
    if not rhymes(words[a], words[b]) or not rhymes(words[c], words[d]):
        return "scheme pair fails exact CMUdict rhyme"
    if rhymes(words[a], words[c]):
        return "the two intended families overlap"
    return None


def run(args):
    spec = json.loads(args.spec.read_text())
    selected, sources, rejected = [], [], []
    for scheme in ("aabb", "abba"):
        accepted = 0
        for source in spec["sources"]:
            if source["scheme"] != scheme:
                continue
            path = args.raw_dir / source["file"]
            candidates, envelope = response_candidates(path)
            sources.append({
                **source, "sha256": sha256(path), "candidate_count": len(candidates),
                "resolved_models": sorted(envelope.get("modelUsage", {})),
                "claude_code_version": spec["claude_code_version"],
                "generation_date": spec["generation_date"],
                "selection": "CMUdict exact-rhyme validation, then response order",
            })
            for index, candidate in enumerate(candidates):
                reason = rejection(candidate.get("lines", []), scheme)
                if reason:
                    rejected.append({"source": source["file"], "index": index, "reason": reason})
                elif accepted < args.per_scheme:
                    lines = candidate["lines"]
                    selected.append({
                        "id": f"external-{scheme}-{accepted:02d}", "scheme": scheme,
                        "lines": lines, "endings": [ending(line) for line in lines],
                        "source_file": source["file"], "source_index": index,
                    })
                    accepted += 1
        if accepted < args.per_scheme:
            raise RuntimeError(f"Only {accepted} valid {scheme.upper()} poems")

    # The pre-existing independent ABAB set has 35 poems.  Select its first 30
    # CMUdict-valid entries so sample sizes match, without concealing its older
    # generation-provenance limitation.
    for index, poem in enumerate(
        json.loads(line) for line in args.abab.read_text().splitlines() if line.strip()
    ):
        if sum(row["scheme"] == "abab" for row in selected) >= args.per_scheme:
            break
        reason = rejection(poem["lines"], "abab")
        if reason:
            rejected.append({"source": args.abab.name, "index": index, "reason": reason})
            continue
        selected.append({
            "id": f"external-abab-{sum(row['scheme'] == 'abab' for row in selected):02d}",
            "scheme": "abab", "lines": poem["lines"],
            "endings": [ending(line) for line in poem["lines"]],
            "source_file": args.abab.name, "source_index": index,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in selected))
    provenance = {
        "schema_version": 1, "built_on": date.today().isoformat(),
        "builder": "scripts/build_external_scheme_dataset.py",
        "selection_target_per_scheme": args.per_scheme,
        "sources": sources + [{
            "file": args.abab.name, "scheme": "abab", "sha256": sha256(args.abab),
            "provenance_gap": "Original prompt, model version, raw response, and edits were not preserved.",
        }],
        "rejections": rejected,
        "final_dataset_sha256": sha256(args.output),
    }
    args.provenance.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote {len(selected)} poems; sha256={provenance['final_dataset_sha256']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--abab", type=Path, default=Path("data/haiku_quatrains.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/external_scheme_quatrains.jsonl"))
    parser.add_argument("--provenance", type=Path, default=Path("data/external_scheme_provenance.json"))
    parser.add_argument("--per-scheme", type=int, default=30)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
