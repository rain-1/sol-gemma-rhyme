import json

from rhyme_interp.manifest import sha256_file, write_manifest
from rhyme_interp.model import DEFAULT_REVISIONS, GEMMA4_MODEL, resolve_load_options


def test_gemma4_forces_eager_and_pinned_revision():
    revision, attention = resolve_load_options(GEMMA4_MODEL, attn_implementation="sdpa")
    assert revision == DEFAULT_REVISIONS[GEMMA4_MODEL]
    assert attention == "eager"


def test_explicit_revision_and_non_gemma_attention_are_preserved():
    assert resolve_load_options(GEMMA4_MODEL, "experiment", None) == ("experiment", "eager")
    assert resolve_load_options("EleutherAI/pythia-410m", None, "sdpa") == (None, "sdpa")


def test_manifest_records_dataset_hash(tmp_path):
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"x": 1}\n')
    output = tmp_path / "run.manifest.json"
    write_manifest(output, model="example/model", revision="abc", precision="bf16", seed=7,
                   datasets=[dataset], command=["experiment", "--seed", "7"])
    result = json.loads(output.read_text())
    assert result["schema_version"] == 1
    assert result["model"] == {"id": "example/model", "revision": "abc", "precision": "bf16"}
    assert result["datasets"] == [{"path": str(dataset), "sha256": sha256_file(dataset)}]
