import importlib.util
from pathlib import Path
import sys

import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_gemma4_head_output.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("run_gemma4_head_output", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_full_rank_projectors_are_explicit_identity():
    dim = 32
    random_basis = MODULE.orthogonal_random_basis(dim)
    pca_basis = torch.linalg.svd(torch.randn(64, dim), full_matrices=False).Vh
    identity = torch.eye(dim)

    assert torch.equal(MODULE.subspace_projector(random_basis, dim), identity)
    assert torch.equal(MODULE.subspace_projector(pca_basis, dim), identity)


def test_full_rank_transfers_are_exact_even_for_large_bf16_values():
    source = (torch.randn(5, 32) * 10_000).bfloat16().float()
    destination = (torch.randn(5, 32) * 10_000).bfloat16().float()
    identity = MODULE.subspace_projector(MODULE.orthogonal_random_basis(32), 32)

    transferred = MODULE.projected_transfer(source, destination, identity)

    assert torch.equal(transferred, source)


def test_random_controls_are_nested():
    basis = MODULE.orthogonal_random_basis(32)
    rank4 = MODULE.subspace_projector(basis, 4)
    rank8 = MODULE.subspace_projector(basis, 8)

    assert torch.allclose(rank4 @ rank8, rank4, atol=2e-6)
