"""Matrix grouping must preserve arbitrary einsum label and batch placement."""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from tools.vibeqc_tensor import Index, IndexSpace, TensorSpec, einsum, input_tensor
from tools.vibeqc_tensor.cuda_gemm import fp64_coefficient, gemm_contract


def node_for(expression, dimensions):
    """Make explicitly typed operands for an independently written equation."""
    inputs, _ = expression.split("->")
    spaces = {label: IndexSpace(label, "batch", n) for label, n in dimensions.items()}
    nodes = [
        input_tensor(
            f"input_{i}",
            TensorSpec(
                tuple(
                    Index(f"axis_{axis}", spaces[label])
                    for axis, label in enumerate(labels)
                ),
                role="input",
            ),
        )
        for i, labels in enumerate(inputs.split(","))
    ]
    return einsum(expression, *nodes, coefficient=Fraction(-3, 4))


@pytest.mark.parametrize(
    "expression,dimensions",
    [
        ("ik,kj->ij", {"i": 3, "j": 5, "k": 7}),
        ("ki,jk->ji", {"i": 3, "j": 5, "k": 7}),
        ("bik,bkj->bij", {"b": 2, "i": 3, "j": 5, "k": 7}),
        ("ibk,jkb->jbi", {"b": 2, "i": 3, "j": 5, "k": 7}),
        ("abef,ijef->ijab", {"a": 3, "b": 2, "e": 3, "f": 2, "i": 2, "j": 3}),
        ("abik,abkj->bija", {"a": 2, "b": 3, "i": 2, "j": 3, "k": 5}),
        ("i,j->ij", {"i": 3, "j": 5}),
        ("i,i->", {"i": 7}),
        (",i->i", {"i": 5}),
    ],
)
def test_matrix_coordinates_reconstruct_independent_einsum(expression, dimensions):
    node = node_for(expression, dimensions)
    contract = gemm_contract(node)
    assert contract is not None
    rng = np.random.default_rng(146)
    arrays = [rng.normal(size=operand.spec.shape) for operand in node.inputs]
    result = np.zeros(node.spec.shape)
    for batch, row, column in product(
        range(contract.batch), range(contract.m), range(contract.n)
    ):
        for reduction in range(contract.k):
            a, b, c = contract.matrix_coordinates(batch, row, column, reduction)
            result[c] += contract.coefficient * arrays[0][a] * arrays[1][b]
    np.testing.assert_allclose(
        result, -0.75 * np.einsum(expression, *arrays), atol=2e-13, rtol=2e-13
    )


@pytest.mark.parametrize("expression", ["ik,j->i", "ii,ij->j", "ij,jk,kl->il"])
def test_non_gemm_einsums_explicitly_keep_the_general_path(expression):
    node = node_for(expression, {"i": 3, "j": 3, "k": 3, "l": 3})
    assert gemm_contract(node) is None


def test_empty_groups_and_bounded_partial_panel_storage():
    empty = gemm_contract(node_for("ik,kj->ij", {"i": 3, "k": 0, "j": 5}))
    assert empty.panel_bytes(2, 3, 4) == 0
    with pytest.raises(ValueError, match="outside"):
        empty.matrix_coordinates(0, 0, 0, 0)
    ordinary = gemm_contract(node_for("bik,bkj->bij", {"b": 9, "i": 3, "k": 7, "j": 5}))
    assert ordinary.panel_bytes(2, 3, 4) == 8 * (2 * 4 + 4 * 3 + 2 * 3)
    assert ordinary.panel_bytes(128, 128, 128) == 8 * (3 * 7 + 7 * 5 + 3 * 5)
    with pytest.raises(ValueError, match="positive"):
        ordinary.panel_bytes(0, 3, 4)


def test_coefficient_conversion_does_not_overflow_separate_integer_terms():
    assert fp64_coefficient((10**400, 10**400)) == 1.0
    with pytest.raises(ValueError, match="finite FP64"):
        fp64_coefficient((10**400, 1))
