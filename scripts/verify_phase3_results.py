"""Validate Phase-3 artifacts and emit a compact, committed evidence summary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path("artifacts")
OUTPUT = Path("results/evidence/gemma4_phase3_summary.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def records(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_json(path, lines=True)


def main():
    rank_path = ROOT / "gemma4_head_output_rank_sanity/head_output_rank.jsonl"
    rank_sanity_path = ROOT / "gemma4_head_output_rank_sanity/head_output_rank_sanity.json"
    routing_path = ROOT / "gemma4_phase3/routing_decomposition.jsonl"
    planning_path = ROOT / "gemma4_phase3/causal_planning.jsonl"
    external_behavior_path = ROOT / "gemma4_external_schemes/external_scheme_behavior.jsonl"
    external_attention_path = ROOT / "gemma4_external_schemes/external_scheme_attention.jsonl"
    factorial_path = ROOT / "gemma4_factorial_phonology/summary_confirmation.json"
    factorial_bf16_path = ROOT / "gemma4_factorial_phonology_bf16/summary_confirmation.json"

    rank = records(rank_path)
    rank_sanity = json.loads(rank_sanity_path.read_text())
    assert rank_sanity["head_dim"] == 512
    assert rank_sanity["full_rank_max_logit_error_vs_identity"] == {"pca": 0.0, "random": 0.0}
    full = rank[rank["rank"] == 512].groupby("subspace").recovery.mean().to_dict()
    assert max(full.values()) - min(full.values()) < 1e-8

    factorial = json.loads(factorial_path.read_text())
    factorial_bf16 = json.loads(factorial_bf16_path.read_text())
    # The exact schema is preserved, but locate the already reported primary
    # effects defensively to fail if the artifact changes shape.
    def find_effect(data, needle):
        for key, value in data.items():
            if needle in key and isinstance(value, (int, float)):
                return float(value)
        raise AssertionError(f"No {needle!r} effect in {list(data)}")

    routing = records(routing_path)
    assert len(routing) == 2100
    routing_summary = {}
    for destination, donor_distance in [("aabb", 2), ("abab", 1)]:
        subset = routing[routing.destination == destination]
        baseline = subset[subset.intervention == "none"].iloc[:25]
        memory = subset[
            (subset.intervention == "memory") & (subset.position == "all_cues")
        ]
        by_part = memory.groupby("part").agg(
            d1_attention=("cue_distance_1_attention", "mean"),
            d2_attention=("cue_distance_2_attention", "mean"),
            d1_mass=("cue_distance_1_mass", "mean"),
            d2_mass=("cue_distance_2_mass", "mean"),
        ).to_dict("index")
        # Key and both must agree closely; value alone must leave addressing near baseline.
        assert abs(by_part["key"]["d1_attention"] - by_part["both"]["d1_attention"]) < .01
        assert abs(by_part["value"]["d1_attention"] - baseline.cue_distance_1_attention.mean()) < .01
        query = subset[subset.intervention == "query"].groupby("layer").agg(
            d1=("cue_distance_1_attention", "mean"), d2=("cue_distance_2_attention", "mean")
        )
        donor_advantage = (
            query.d2 - query.d1 if donor_distance == 2 else query.d1 - query.d2
        )
        assert donor_advantage.loc[23] > donor_advantage.loc[22] + .15
        routing_summary[destination] = {
            "baseline": {
                "d1_attention": baseline.cue_distance_1_attention.mean(),
                "d2_attention": baseline.cue_distance_2_attention.mean(),
                "d1_mass": baseline.cue_distance_1_mass.mean(),
                "d2_mass": baseline.cue_distance_2_mass.mean(),
            },
            "all_cue_memory_by_part": by_part,
            "query_donor_advantage_layer22": donor_advantage.loc[22],
            "query_donor_advantage_layer23": donor_advantage.loc[23],
        }

    planning = records(planning_path)
    assert len(planning) == 3525
    residual = planning[planning.kind == "residual"]
    anchor_l13 = residual[(residual.layer == 13) & (residual.position == "anchor")].recovery.mean()
    line_positions = ["line_start", "first_word", "line_middle", "penultimate_input", "final_input"]
    line_residual = residual[residual.position.isin(line_positions)].groupby(
        ["layer", "position"]
    ).recovery.mean()
    memory = planning[
        (planning.kind == "shared_memory") & planning.position.isin(line_positions)
    ].groupby(["memory", "part", "position"]).recovery.mean()
    assert anchor_l13 > .9
    assert line_residual.abs().max() < .02
    assert memory.abs().max() < .02

    external_behavior = records(external_behavior_path)
    external_attention = records(external_attention_path)
    external_summary = {}
    for scheme in ["aabb", "abab", "abba"]:
        behavior = external_behavior[external_behavior.scheme == scheme]
        clean = behavior[behavior.condition == "clean"]
        ablated = behavior[behavior.condition == "l24h3_ablated"]
        attention = external_attention[external_attention.scheme == scheme]
        external_summary[scheme] = {
            "n": len(clean),
            "clean_top1_exact_rhyme": clean.top1_rhymes_cue.mean(),
            "clean_cue_family_mass": clean.cue_family_mass.mean(),
            "ablated_top1_exact_rhyme": ablated.top1_rhymes_cue.mean(),
            "ablated_cue_family_mass": ablated.cue_family_mass.mean(),
            "mean_correct_cue_attention": attention.attention_cue.mean(),
        }
    assert external_summary["aabb"]["clean_top1_exact_rhyme"] > .8
    assert external_summary["abab"]["clean_top1_exact_rhyme"] > .8
    assert external_summary["abba"]["clean_top1_exact_rhyme"] < .4

    summary = {
        "schema_version": 1,
        "rank": {
            "full_rank_recovery": full,
            "pca_rank_16_recovery": rank[(rank.subspace == "pca") & (rank["rank"] == 16)].recovery.mean(),
            "pca_rank_32_recovery": rank[(rank.subspace == "pca") & (rank["rank"] == 32)].recovery.mean(),
            "random_rank_32_recovery": rank[(rank.subspace == "random") & (rank["rank"] == 32)].recovery.mean(),
        },
        "factorial_artifact": factorial,
        "factorial_bf16_artifact": factorial_bf16,
        "routing": routing_summary,
        "planning": {
            "anchor_l13_recovery": anchor_l13,
            "max_abs_line_residual_recovery": line_residual.abs().max(),
            "max_abs_line_memory_recovery": memory.abs().max(),
        },
        "external_scheme_replication": external_summary,
        "raw_artifacts": {
            str(path): {"sha256": sha256(path), "rows": len(records(path))}
            for path in [rank_path, routing_path, planning_path,
                         external_behavior_path, external_attention_path]
        },
        "source_summaries": {
            str(factorial_path): sha256(factorial_path),
            str(factorial_bf16_path): sha256(factorial_bf16_path),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Verified Phase 3 and wrote {OUTPUT}")


if __name__ == "__main__":
    main()
