"""Independent retained Libxc gates for the default second-order XC API."""

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.method import resolve_method
from vibeqc_compiler.xc.program import build_program
from vibeqc_compiler.xc.spec import FunctionalSpec

FIXTURE = Path(__file__).parents[1] / "data/xc/p86-hessian.json"
DATA = json.loads(FIXTURE.read_text())


@pytest.mark.parametrize(
    "case", DATA["cases"], ids=lambda case: case["name"] + "-" + case["spin"]
)
def test_default_feature_hessian_matches_independent_libxc(case):
    assert DATA["schema"] == "vibeqc.independent-semilocal-hessian.v1"
    assert DATA["libxc"] == "7.0.0"
    name, spin = case["name"], case["spin"]
    if name.startswith(("LDA_", "GGA_")):
        spec = FunctionalSpec(name, ((name, Fraction(1)),), spin=spin)
    else:
        spec = resolve_method(name, spin=spin).primitives[0].functional
    program = build_program(spec)
    assert program.order == 2
    features = np.array(case["features"])
    expected = np.array(case["expected"])
    size = len(spec.features)
    assert expected.shape == (1 + size + size * (size + 1) // 2, 8)
    actual = program.evaluate(features)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)
