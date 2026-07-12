import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_gemma4_external_schemes.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("external_schemes", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_external_scheme_cue_locations():
    assert MODULE.CUE_LINE == {"aabb": 2, "abab": 1, "abba": 0}


def test_prompt_removes_only_final_word():
    poem = {
        "lines": ["One bright", "Two light", "Three slow", "Four glow"],
        "endings": ["bright", "light", "slow", "glow"],
        "scheme": "aabb",
    }
    assert MODULE.prompt_for(poem) == "One bright\nTwo light\nThree slow\nFour"


def test_committed_external_scheme_corpus_is_valid():
    path = Path(__file__).parents[1] / "data" / "external_scheme_quatrains.jsonl"
    if not path.exists():
        pytest.skip("generation in progress")
    poems = MODULE.load_poems(path)
    counts = {scheme: sum(p["scheme"] == scheme for p in poems) for scheme in MODULE.CUE_LINE}
    assert counts["aabb"] >= 30
    assert counts["abba"] >= 30
    assert counts["abab"] >= 30
    assert len({p["id"] for p in poems}) == len(poems)

    provenance = json.loads((path.parent / "external_scheme_provenance.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance["final_dataset_sha256"]
    for source in provenance["sources"]:
        if source["file"].endswith(".raw.json"):
            raw = path.parent / "external_scheme_raw" / source["file"]
            assert hashlib.sha256(raw.read_bytes()).hexdigest() == source["sha256"]
