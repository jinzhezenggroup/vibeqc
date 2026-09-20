"""PW91-family MethodIR and independent Libxc scalar gates."""

from __future__ import annotations

import typing
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


@pytest.mark.parametrize(
    "alias,canonical",
    [("PW91PW91", "PW91"), ("X3LYPG", "X3LYP")],
)
def test_pw91_family_aliases_preserve_semantics(
    alias: typing.Any, canonical: typing.Any
) -> None:
    reference = resolve_method(canonical)
    named = resolve_method(alias)
    assert named.identity == reference.identity
    assert named.manifest_identity != reference.manifest_identity


def test_x3lyp_vwn5_is_scientifically_distinct() -> None:
    assert resolve_method("X3LYP").identity != resolve_method("X3LYP5").identity


@pytest.mark.parametrize(
    "name,exchange",
    [
        ("PW91", Fraction(0)),
        ("B3PW91", Fraction(1, 5)),
        ("X3LYP", Fraction(109, 500)),
        ("X3LYP5", Fraction(109, 500)),
    ],
)
def test_pw91_family_uses_typed_semilocal_and_exact_exchange(
    name: typing.Any, exchange: typing.Any
) -> None:
    graph = resolve_method(name)
    assert isinstance(graph.primitives[0], SemilocalXCPrimitive)
    if exchange:
        assert isinstance(graph.primitives[1], ExactExchangePrimitive)
        assert graph.primitives[1].coefficient == exchange
    else:
        assert len(graph.primitives) == 1


ORACLES = {
    "PW91": (
        [
            -0.3282121838488419,
            -0.8954942661475697,
            -0.8067117696168564,
            -0.007339659490082176,
            0.02173376863067051,
            -0.02517931481430781,
            0.0,
            0.0,
        ],
        [-0.3257974826183793, -0.8549024891343995, -0.00014467893897346026, 0.0],
    ),
    "B3PW91": (
        [
            -0.2691028702236862,
            -0.7296957529248795,
            -0.6626419375485588,
            -0.005998135088876942,
            0.017604352590843114,
            -0.016244068206564134,
            0.0,
            0.0,
        ],
        [-0.26726636770542345, -0.6987404269418774, -0.0006230403959361468, 0.0],
    ),
    "X3LYP": (
        [
            -0.25568408241100427,
            -0.6945769151788699,
            -0.6314773763033333,
            -0.01402605632606412,
            0.0017814304126332265,
            -0.02240296070310231,
            0.0,
            0.0,
        ],
        [-0.25394211498811364, -0.6653940377299119, -0.00804129634686297, 0.0],
    ),
    "X3LYP5": (
        [
            -0.2544190755818271,
            -0.6921459287511278,
            -0.6286057308159116,
            -0.01402605632606412,
            0.0017814304126332265,
            -0.02240296070310231,
            0.0,
            0.0,
        ],
        [-0.2526660903883324, -0.6627598812668355, -0.00804129634686297, 0.0],
    ),
}


@pytest.mark.parametrize("name", ORACLES)
def test_pw91_family_scalar_values_match_pyscf_libxc_oracle(name: typing.Any) -> None:
    expected_p, expected_u = ORACLES[name]
    p_spec = resolve_method(name, spin="polarized").primitives[0].functional
    u_spec = resolve_method(name, spin="unpolarized").primitives[0].functional
    actual_p = build_program(p_spec, order=1).evaluate(POLARIZED).ravel()
    actual_u = build_program(u_spec, order=1).evaluate(UNPOLARIZED).ravel()
    np.testing.assert_allclose(actual_p, expected_p, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(actual_u, expected_u, rtol=2e-12, atol=2e-13)
