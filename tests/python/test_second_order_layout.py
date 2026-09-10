"""Second-order mixed-pair factors and recovery on both coordinate indices."""

from dataclasses import replace

import numpy as np
import pytest
from vibeqc_compiler.integral.ir import FOUR_CENTER_ERI_OPERATOR, NuclearCoordinates
from vibeqc_compiler.integral.second_order_layout import (
    CenterRecovery,
    HessianLayout,
    second_center_recovery,
)


@pytest.mark.parametrize("centers", [(0,), (0, 1), (0, 1, 2, 3)])
def test_svec_preserves_frobenius_inner_products_and_dense_order(centers):
    packed = HessianLayout(centers, "svec")
    dense = HessianLayout(centers)
    random = np.random.default_rng(178)
    first, second = random.normal(size=(2, packed.dimension, packed.dimension))
    first, second = first + first.T, second + second.T
    x, y = packed.encode(first), packed.encode(second)
    np.testing.assert_allclose(x @ y, np.sum(first * second), atol=1e-12)
    np.testing.assert_allclose(packed.decode(x), first, atol=1e-14)
    np.testing.assert_array_equal(dense.decode(dense.encode(first)), first)
    assert (
        packed.tensor_layout.storage_elements
        == packed.dimension * (packed.dimension + 1) // 2
    )


def test_two_index_translation_recovery_matches_projected_hvp_and_same_atom_chain_rule():
    operator = FOUR_CENTER_ERI_OPERATOR
    recovery = second_center_recovery(operator, operator.nuclear_derivative(order=2))
    assert recovery.independent == (0, 1, 2)
    random = np.random.default_rng(178)
    matrix = random.normal(size=(9, 9))
    matrix += matrix.T
    full = recovery.recover_hessian(matrix)
    np.testing.assert_allclose(full.reshape(4, 3, 4, 3).sum(axis=0), 0, atol=2e-14)
    np.testing.assert_allclose(full.reshape(4, 3, 4, 3).sum(axis=2), 0, atol=2e-14)
    # Slots 0 and 1 share an atom. Expand its direction before the Hessian
    # contraction, then scatter both shell-center responses back to that atom.
    atoms = (0, 0, 1, 2)
    direction = random.normal(size=(3, 3))
    shell_direction = direction[list(atoms)]
    projected = recovery.project_direction(shell_direction)
    result = recovery.recover_vector((matrix @ projected.ravel()).reshape(3, 3))
    np.testing.assert_allclose(
        result.ravel(), full @ shell_direction.ravel(), atol=2e-14
    )
    atom_map = np.kron(np.eye(3)[list(atoms)], np.eye(3))
    scattered = np.zeros((3, 3))
    np.add.at(scattered, list(atoms), result)
    np.testing.assert_allclose(
        scattered.ravel(), atom_map.T @ full @ atom_map @ direction.ravel(), atol=2e-14
    )


def test_partial_center_request_does_not_assume_missing_translation_terms():
    operator = FOUR_CENTER_ERI_OPERATOR
    derivative = replace(
        operator.nuclear_derivative(order=2), parameters=NuclearCoordinates((0, 3))
    )
    recovery = second_center_recovery(operator, derivative)
    assert recovery.independent == (0, 3)
    np.testing.assert_array_equal(recovery.coordinate_matrix, np.eye(6))
    with pytest.raises(ValueError, match="order two"):
        second_center_recovery(operator, operator.nuclear_derivative())


def test_malformed_hessian_layouts_and_buffers_fail_explicitly():
    for centers, packing in [((0, 0), "dense"), ((), "dense"), ((0, 1), "triangular")]:
        with pytest.raises(ValueError):
            HessianLayout(centers, packing)
    packed = HessianLayout((0,), "svec")
    for matrix in (
        np.full((3, 3), np.inf),
        np.eye(3, dtype=complex),
        np.zeros((2, 2)),
        np.arange(9).reshape(3, 3),
    ):
        with pytest.raises(ValueError):
            packed.encode(matrix)
    with pytest.raises(ValueError):
        packed.decode(np.zeros(5))
    matrix = np.zeros((3, 3))
    matrix[0, 1] = matrix[1, 0] = 1.7e308
    with pytest.raises(FloatingPointError):
        packed.encode(matrix)


def test_invalid_recovery_cannot_silently_drop_or_relabel_a_coordinate():
    with pytest.raises(ValueError, match="basis"):
        CenterRecovery((0, 1), (0,), ((1,), (0,)))
    with pytest.raises(ValueError, match="at most one"):
        CenterRecovery((0, 1, 2), (0,), ((1,), (-1,), (-1,)))
    with pytest.raises(ValueError, match="at most one"):
        CenterRecovery((0, 1), (2,), ((1,), (-1,)))
