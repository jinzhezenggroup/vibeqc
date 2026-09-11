"""Raw/fused arbitrary-weight derivatives, bounded subsets, and center semantics."""

from dataclasses import replace

import numpy as np
import pytest
from vibeqc_compiler.integral.ir import (
    DerivativeSpec,
    NuclearCoordinates,
    TranslationInvariant,
)
from vibeqc_compiler.integral.shell_class import build_shell_class_component_kernel
from vibeqc_compiler.integral.weighted_eri import (
    build_weighted_eri_ir,
    build_weighted_eri_kernel,
)

from tools.vibeqc_validation.weighted_eri import primitive_variables

EXPONENTS = (0.6, 0.8, 1.1, 0.9)
CENTERS = np.array(
    [
        [0.13, -0.31, 0.24],
        [-0.43, 0.27, 0.51],
        [0.68, -0.14, -0.22],
        [-0.21, 0.48, -0.63],
    ]
)


def evaluate(kernel, variables, weights):
    variables = {
        **variables,
        **{f"component_weight_{i}": w for i, w in enumerate(weights)},
    }
    gradient = np.array(
        [[kernel.graph.evaluate(x, variables) for x in row] for row in kernel.gradients]
    )
    return kernel.graph.evaluate(kernel.value, variables), gradient


@pytest.mark.parametrize(
    "angular",
    [
        (0, 0, 0, 0),
        (1, 0, 0, 0),
        (1, 0, 1, 0),
        (1, 1, 0, 0),
        (2, 0, 0, 0),
        (3, 0, 0, 0),
        (2, 1, 0, 1),
    ],
)
@pytest.mark.parametrize("coincident", [False, True])
def test_fused_external_weights_match_raw_component_derivatives(angular, coincident):
    integral = build_weighted_eri_ir(angular)
    kernel = build_weighted_eri_kernel(integral)
    assert not integral.consumers  # No hidden direct-HF category or density.
    centers = CENTERS.copy()
    if coincident:
        centers[1] = centers[0]
    variables = primitive_variables(EXPONENTS, centers, integral.maximum_coulomb_order)
    weights = np.random.default_rng(144).normal(size=kernel.spec.component_count)
    raw_value, raw_gradient = 0.0, np.zeros((4, 3))
    for weight, component in zip(weights, kernel.spec.components):
        raw = build_shell_class_component_kernel(
            kernel.spec, component, integral=integral
        )
        raw_value += weight * raw.graph.evaluate(raw.value, variables)
        raw_gradient += weight * np.array(
            [[raw.graph.evaluate(x, variables) for x in row] for row in raw.gradients]
        )
    value, gradient = evaluate(kernel, variables, weights)
    np.testing.assert_allclose(value, raw_value, atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(gradient, raw_gradient, atol=3e-12, rtol=3e-12)
    np.testing.assert_allclose(gradient.sum(axis=0), 0, atol=3e-14)


def test_partial_component_subsets_and_explicit_output_scale():
    integral = build_weighted_eri_ir((1, 0, 1, 0))
    consumer = integral.contractions[0]
    integral = replace(
        integral,
        contractions=(
            replace(
                consumer,
                output_sign=-1,
                weights=replace(consumer.weights, prefactor=0.7, sign=-1),
            ),
        ),
    )
    full = build_weighted_eri_kernel(integral)
    variables = primitive_variables(EXPONENTS, CENTERS, integral.maximum_coulomb_order)
    weights = np.arange(9) - 3.2
    value, gradient = evaluate(full, variables, weights)
    parts = [
        evaluate(build_weighted_eri_kernel(integral, subset), variables, weights)
        for subset in ((0, 4, 8), (1, 2, 3, 5, 6, 7))
    ]
    np.testing.assert_allclose(sum(v for v, g in parts), value, atol=2e-13)
    np.testing.assert_allclose(sum(g for v, g in parts), gradient, atol=2e-13)
    unscaled = build_weighted_eri_kernel(build_weighted_eri_ir((1, 0, 1, 0)))
    np.testing.assert_allclose(
        gradient, 0.7 * evaluate(unscaled, variables, weights)[1], atol=2e-13
    )


def test_nonfinal_translation_recovery_keeps_physical_shell_slots():
    integral = build_weighted_eri_ir((1, 0, 1, 0))
    invariant = TranslationInvariant(dependent_center=1)
    changed = replace(
        integral,
        operator=replace(integral.operator, invariants=(invariant,)),
        derivative=DerivativeSpec(1, NuclearCoordinates(), (invariant,)),
    )
    first, second = (
        build_weighted_eri_kernel(integral),
        build_weighted_eri_kernel(changed),
    )
    weights = np.random.default_rng(3).normal(size=9)
    variables = primitive_variables(EXPONENTS, CENTERS, integral.maximum_coulomb_order)
    assert changed.independent_derivative_centers == (0, 2, 3)
    np.testing.assert_allclose(
        evaluate(first, variables, weights)[1],
        evaluate(second, variables, weights)[1],
        atol=2e-13,
    )


def test_arbitrary_weight_atomic_gradient_matches_multistep_raw_scalar_differences():
    integral = build_weighted_eri_ir((1, 0, 1, 0))
    fused = build_weighted_eri_kernel(integral)
    weights = np.random.default_rng(77).normal(size=9)
    centers = CENTERS.copy()
    centers[1] = centers[0]
    atoms = (0, 0, 1, 2)
    components = [
        build_shell_class_component_kernel(fused.spec, c, integral=integral)
        for c in fused.spec.components
    ]

    def raw_scalar(coordinates):
        variables = primitive_variables(
            EXPONENTS, coordinates, integral.maximum_coulomb_order
        )
        return sum(
            w * c.graph.evaluate(c.value, variables)
            for w, c in zip(weights, components)
        )

    variables = primitive_variables(EXPONENTS, centers, integral.maximum_coulomb_order)
    gradient = evaluate(fused, variables, weights)[1]
    atom_gradient = np.zeros((3, 3))
    for slot, atom in enumerate(atoms):
        atom_gradient[atom] += gradient[slot]
    errors = []
    for step in (2e-3, 1e-3, 5e-4):
        finite = np.empty((3, 3))
        for atom in range(3):
            for axis in range(3):
                displacement = np.zeros((4, 3))
                displacement[np.array(atoms) == atom, axis] = step
                finite[atom, axis] = (
                    raw_scalar(centers + displacement)
                    - raw_scalar(centers - displacement)
                ) / (2 * step)
        errors.append(np.max(np.abs(finite - atom_gradient)))
    assert errors[1] < errors[0] * 0.3
    assert errors[2] < errors[1] * 0.3
    assert errors[2] < 2e-6


def test_large_classes_require_explicit_bounded_lowering_without_truncation():
    with pytest.raises(ValueError, match="64 explicit"):
        build_weighted_eri_kernel(build_weighted_eri_ir((3, 3, 3, 3)))
    with pytest.raises(ValueError, match="unique"):
        build_weighted_eri_kernel(build_weighted_eri_ir((1, 0, 0, 0)), (0, 0))
