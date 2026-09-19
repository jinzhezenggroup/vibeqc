"""Generated weighted first-gradient compiler contracts."""

import pytest
from vibeqc_compiler.integral.first_gradient import (
    FirstGradientTerm,
    FirstGradientWeight,
    emit_first_gradient,
    first_gradient_identity,
)
from vibeqc_compiler.integral.one_electron_derivatives import (
    build_one_electron_derivative_ir,
)
from vibeqc_compiler.integral.weighted_eri import build_weighted_eri_ir


def test_weighted_gradient_source_is_generic_and_accumulates_atomic_output():
    ir = build_one_electron_derivative_ir("overlap", (0, 0))
    terms = (FirstGradientTerm((FirstGradientWeight(1, (0, 1)),), -1.0),)
    identity = first_gradient_identity(ir, (0,), terms)
    source = emit_first_gradient(ir, (0,), terms, runtime_identity="a" * 64)
    assert len(identity) == 64
    assert "weights[1*nbf*nbf+ao[0]*nbf+ao[1]]" in source
    assert "atomicAdd(output+3*mapping.atoms[c]+axis" in source
    assert "rhf" not in source.lower()


def test_two_matrix_weight_product_is_declared_not_method_coded():
    ir = build_weighted_eri_ir((0, 0, 0, 0))
    terms = (
        FirstGradientTerm(
            (
                FirstGradientWeight(0, (0, 1)),
                FirstGradientWeight(2, (2, 3)),
            ),
            0.5,
        ),
    )
    source = emit_first_gradient(ir, (0,), terms, runtime_identity="b" * 64)
    assert "weights[0*nbf*nbf+ao[0]*nbf+ao[1]]" in source
    assert "weights[2*nbf*nbf+ao[2]*nbf+ao[3]]" in source
    assert "0x1.0000000000000p-1" in source


@pytest.mark.parametrize(
    "weight",
    [
        FirstGradientWeight(0, (0, 1)),
        FirstGradientWeight(7, (2, 3)),
    ],
)
def test_weight_contract_accepts_supported_slots(weight):
    assert weight.matrix_slot in range(8)


def test_invalid_weight_contract_fails_closed():
    with pytest.raises(ValueError):
        FirstGradientWeight(8, (0, 1))
    with pytest.raises(ValueError):
        FirstGradientTerm(())
