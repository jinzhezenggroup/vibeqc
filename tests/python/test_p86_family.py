"""P86-family MethodIR and independent Libxc scalar gates."""

from fractions import Fraction

import numpy as np
import pytest
from vibeqc_compiler.method import (
    ExactExchangePrimitive,
    SemilocalXCPrimitive,
    resolve_method,
)
from vibeqc_compiler.xc.program import build_program

POLARIZED = np.array([[0.3, 0.2, 0.015, 0.003, 0.01, 0.0, 0.0]]).T
UNPOLARIZED = np.array([[0.5, 0.031, 0.0]]).T


def test_b3p86_gaussian_alias_preserves_semantics():
    reference = resolve_method("B3P86")
    named = resolve_method("B3P86G")
    assert named.identity == reference.identity
    assert named.manifest_identity != reference.manifest_identity


def test_b3p86_vwn5_is_scientifically_distinct():
    assert resolve_method("B3P86").identity != resolve_method("B3P86V5").identity


@pytest.mark.parametrize(
    "name,exchange",
    [("BP86", Fraction(0)), ("B3P86", Fraction(1, 5)), ("B3P86V5", Fraction(1, 5))],
)
def test_p86_family_uses_typed_semilocal_and_exact_exchange(name, exchange):
    graph = resolve_method(name)
    assert isinstance(graph.primitives[0], SemilocalXCPrimitive)
    if exchange:
        assert isinstance(graph.primitives[1], ExactExchangePrimitive)
        assert graph.primitives[1].coefficient == exchange
    else:
        assert len(graph.primitives) == 1


ORACLES = {
    "BP86": (
        [
            -0.3281088565459033,
            -0.8932785416527623,
            -0.8082390581964841,
            -0.01118929400444907,
            0.01873338805859769,
            -0.025419756667903505,
            0.0,
            0.0,
        ],
        [-0.3257600023228303, -0.8541204488881198, -0.003669902461531198, 0.0],
    ),
    "B3P86": (
        [
            -0.2707917574869708,
            -0.7314978507867721,
            -0.6677188219896977,
            -0.007213289220566433,
            0.01517404432746413,
            -0.017459222338253623,
            0.0,
            0.0,
        ],
        [-0.2690370525251081, -0.7020560020716913, -0.0017899982130003744, 0.0],
    ),
    "B3P86V5": (
        [
            -0.26892856913391927,
            -0.7279173281412606,
            -0.6634892666206268,
            -0.007213289220566433,
            0.01517404432746413,
            -0.017459222338253623,
            0.0,
            0.0,
        ],
        [-0.26715763644791085, -0.6981762367384778, -0.0017899982130003744, 0.0],
    ),
}


@pytest.mark.parametrize("name", ORACLES)
def test_p86_family_scalar_values_match_pyscf_libxc_oracle(name):
    expected_p, expected_u = ORACLES[name]
    p_spec = resolve_method(name, spin="polarized").primitives[0].functional
    u_spec = resolve_method(name, spin="unpolarized").primitives[0].functional
    actual_p = build_program(p_spec, order=1).evaluate(POLARIZED).ravel()
    actual_u = build_program(u_spec, order=1).evaluate(UNPOLARIZED).ravel()
    np.testing.assert_allclose(actual_p, expected_p, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(actual_u, expected_u, rtol=2e-12, atol=2e-13)
