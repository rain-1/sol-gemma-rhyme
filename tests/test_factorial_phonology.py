import importlib.util
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path("scripts").resolve()))
SPEC = importlib.util.spec_from_file_location("factorial", Path("scripts/run_gemma4_factorial_phonology.py"))
factorial = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(factorial)


def test_additive_prediction_recombines_heldout_cell():
    vowels = {"A": np.array([1., 0., 0.]), "B": np.array([0., 2., 0.]), "C": np.array([0., 0., 3.])}
    codas = {"X": np.array([.5, .25, 0.]), "Y": np.array([0., .5, .75])}
    labels = {f"{v}{c}": (v, c) for v in vowels for c in codas}
    centroids = {name: vowels[v] + codas[c] for name, (v, c) in labels.items()}
    got = factorial.additive_prediction(centroids, "AX", labels)
    np.testing.assert_allclose(got, centroids["AX"], atol=1e-6)


def test_targets_require_both_parts_elsewhere_and_split_is_stable():
    families = {"AA1-X": [], "AA1-Y": [], "BB1-X": [], "CC1-Z": []}
    assert factorial.eligible_targets(families) == ["AA1-X"]
    assert factorial.fixed_split(["BB1-X", "AA1-X"]) == factorial.fixed_split(["AA1-X", "BB1-X"])
