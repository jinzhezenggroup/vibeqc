"""Independent analytic and multistep numerical gates for partial Hessians/HVPs."""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest
from vibeqc_compiler.integral.expr import AlgebraForm
from vibeqc_compiler.integral.one_electron_derivatives import (
    build_one_electron_derivative_ir,
    build_one_electron_derivative_kernel,
    evaluate_one_electron_derivative_primitive,
)
from vibeqc_compiler.integral.second_derivatives import (
    build_eri_second_ir,
    build_one_electron_second_ir,
    build_second_derivative_kernel,
)
from vibeqc_compiler.integral.second_order_layout import HessianLayout
from vibeqc_compiler.integral.shell_spec import cartesian_components
from vibeqc_compiler.integral.weighted_eri import (
    build_weighted_eri_ir,
    build_weighted_eri_kernel,
)

from tools.vibeqc_validation.second_derivatives import (
    evaluate_second_primitive,
    libcint_eri_hessian,
    libcint_one_electron_hessian,
)
from tools.vibeqc_validation.weighted_eri import primitive_variables

CENTERS = np.array(
    [
        [0.13, -0.31, 0.24],
        [-0.43, 0.27, 0.51],
        [0.68, -0.14, -0.22],
        [-0.21, 0.48, -0.63],
    ]
)
EXPONENTS = (0.6, 0.8, 1.1, 0.9)


def check_hessian(hessian, first_derivatives, centers):
    """Separate scale-aware second-order gate, including three FD step sizes."""
    count = len(centers)
    scale = max(1.0, np.max(np.abs(hessian)))
    np.testing.assert_allclose(hessian, hessian.T, atol=5e-12 * scale, rtol=0)
    tensor = hessian.reshape(count, 3, count, 3)
    for axis in (0, 2):
        np.testing.assert_allclose(tensor.sum(axis=axis), 0, atol=5e-12 * scale)
    errors = []
    for step in (1e-3, 3e-4, 1e-4):
        finite = np.empty_like(hessian)
        for j in range(3 * count):
            displacement = np.zeros_like(centers)
            displacement.flat[j] = step
            finite[:, j] = (
                first_derivatives(centers + displacement).ravel()
                - first_derivatives(centers - displacement).ravel()
            ) / (2 * step)
        errors.append(np.max(np.abs(hessian - finite)))
    assert errors[-1] <= 2e-7 * scale, errors
    assert errors[-1] <= max(5e-11 * scale, errors[0] * 0.03), errors


@pytest.mark.parametrize("family", ["overlap", "kinetic", "nuclear_attraction"])
@pytest.mark.parametrize("angular,index", [((1, 2), 4), ((3, 0), 4)])
def test_stv_raw_against_analytic_libcint_and_first_gradient_fd(family, angular, index):
    pytest.importorskip("pyscf")
    ir = build_one_electron_second_ir(family, angular, charge=2.3)
    kernel = build_second_derivative_kernel(ir, (index,))
    count = len(ir.operator.centers)
    centers, exponents = CENTERS[:count], EXPONENTS[:2]
    actual = evaluate_second_primitive(kernel, exponents, centers).reshape(
        3 * count, 3 * count
    )
    reference = libcint_one_electron_hessian(family, angular, exponents, centers, 2.3)
    reference = reference.reshape(3 * count, 3 * count, -1)[:, :, index]
    np.testing.assert_allclose(actual, reference, atol=4e-11, rtol=2e-11)
    components = tuple(product(*(cartesian_components(l) for l in angular)))[index]
    first = build_one_electron_derivative_kernel(
        build_one_electron_derivative_ir(family, angular, charge=2.3), components
    )
    check_hessian(
        actual,
        lambda c: np.array(
            evaluate_one_electron_derivative_primitive(first, exponents, c)
        ),
        centers,
    )


@pytest.mark.parametrize(
    "angular", [(0, 0, 0, 0), (1, 0, 0, 0), (2, 1, 0, 1), (3, 0, 0, 0)]
)
@pytest.mark.parametrize("coincident", [False, True])
def test_eri_decay_response_and_both_translation_indices(angular, coincident):
    ir = build_eri_second_ir(angular, output="weighted_hessian")
    indices = tuple(range(min(2, ir.signature.component_count)))
    kernel = build_second_derivative_kernel(ir, indices)
    weights = np.random.default_rng(178).normal(size=ir.signature.component_count)
    centers = CENTERS.copy()
    if coincident:
        centers[1] = centers[0]
    actual = evaluate_second_primitive(kernel, EXPONENTS, centers, weights).reshape(
        12, 12
    )
    first = build_weighted_eri_kernel(build_weighted_eri_ir(angular), indices)

    def gradient(positions):
        variables = primitive_variables(
            EXPONENTS, positions, first.integral.maximum_coulomb_order
        )
        variables.update({f"component_weight_{i}": weights[i] for i in indices})
        return np.array(
            [
                [first.graph.evaluate(root, variables) for root in row]
                for row in first.gradients
            ]
        )

    check_hessian(actual, gradient, centers)


@pytest.mark.parametrize("family", ["overlap", "kinetic", "nuclear_attraction", "eri"])
def test_weighted_packed_hvp_tiling_and_algebra_commutation(family):
    def make(**kwargs):
        return (
            build_eri_second_ir((1, 0, 1, 0), **kwargs)
            if family == "eri"
            else build_one_electron_second_ir(family, (1, 2), **kwargs)
        )

    ir = make(output="weighted_hessian")
    count = len(ir.operator.centers)
    centers, exponents = CENTERS[:count], EXPONENTS[: len(ir.signature.shells)]
    weights = np.random.default_rng(178).normal(size=ir.signature.component_count)
    indices = (0, 2)
    matrix = evaluate_second_primitive(
        build_second_derivative_kernel(ir, indices), exponents, centers, weights
    ).reshape(3 * count, 3 * count)
    raw = sum(
        weights[i]
        * evaluate_second_primitive(
            build_second_derivative_kernel(make(output="raw_hessian"), (i,)),
            exponents,
            centers,
        )
        for i in indices
    )
    np.testing.assert_allclose(raw, matrix.ravel(), atol=3e-12, rtol=3e-12)
    packed = evaluate_second_primitive(
        build_second_derivative_kernel(
            make(output="weighted_hessian", packing="svec"), indices
        ),
        exponents,
        centers,
        weights,
    )
    np.testing.assert_allclose(
        HessianLayout(ir.operator.centers, "svec").decode(packed), matrix, atol=3e-12
    )
    direction = np.random.default_rng(17).normal(size=(count, 3))
    hvp_ir = make(output="weighted_hvp")
    hvp = build_second_derivative_kernel(hvp_ir, indices)
    actual = evaluate_second_primitive(hvp, exponents, centers, weights, direction)
    np.testing.assert_allclose(
        actual, matrix @ direction.ravel(), atol=3e-12, rtol=3e-12
    )
    # Distinct derivative schedules must retain the same scientific factors.
    alternative = build_second_derivative_kernel(
        hvp_ir,
        indices,
        primal_form=AlgebraForm.BINARY,
        output_form=AlgebraForm.FACTORED_NARY,
    )
    np.testing.assert_allclose(
        evaluate_second_primitive(alternative, exponents, centers, weights, direction),
        actual,
        atol=3e-12,
        rtol=3e-12,
    )
    selected = (0, 3 * count - 1)
    tile = build_second_derivative_kernel(hvp_ir, indices, output_indices=selected)
    np.testing.assert_allclose(
        evaluate_second_primitive(tile, exponents, centers, weights, direction),
        actual[list(selected)],
        atol=3e-12,
    )
    assert len(tile.outputs) == 2


def test_f_shell_second_moment_bound_and_nonfinal_recovery():
    ir = build_eri_second_ir((3, 3, 3, 3))
    kernel = build_second_derivative_kernel(ir, (0,), output_indices=(0,))
    assert kernel.boys_count == 15
    from vibeqc_compiler.integral.ir import TranslationInvariant

    invariant = TranslationInvariant(dependent_center=1)
    changed = replace(
        ir,
        operator=replace(ir.operator, invariants=(invariant,)),
        derivative=replace(ir.derivative, invariants=(invariant,)),
    )
    other = build_second_derivative_kernel(changed, (0,), output_indices=(0,))
    weights = np.zeros(ir.signature.component_count)
    weights[0] = 0.7
    direction = np.random.default_rng(178).normal(size=(4, 3))
    np.testing.assert_allclose(
        evaluate_second_primitive(kernel, EXPONENTS, CENTERS, weights, direction),
        evaluate_second_primitive(other, EXPONENTS, CENTERS, weights, direction),
        atol=2e-11,
        rtol=2e-11,
    )


@pytest.mark.parametrize("angular", [(2, 1, 0, 1), (3, 0, 0, 0)])
def test_eri_hessian_all_center_pairs_against_independent_libcint(angular):
    pytest.importorskip("pyscf")
    ir = build_eri_second_ir(angular, output="weighted_hessian")
    indices = (0, 4)
    weights = np.random.default_rng(178).normal(size=ir.signature.component_count)
    kernel = build_second_derivative_kernel(ir, indices)
    actual = evaluate_second_primitive(kernel, EXPONENTS, CENTERS, weights).reshape(
        12, 12
    )
    reference = libcint_eri_hessian(angular, EXPONENTS, CENTERS).reshape(12, 12, -1)
    expected = sum(reference[:, :, i] * weights[i] for i in indices)
    np.testing.assert_allclose(actual, expected, atol=5e-11, rtol=2e-11)
