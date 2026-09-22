"""Symbolic recurrence and finite-difference oracle tests for integral codegen."""

from __future__ import annotations

import math
import typing

import pytest
from codegen_fixtures import (
    boys_values,
    evaluate_value,
    factored_dppp_variables,
    sample_variables,
)
from vibeqc_compiler.integral import (
    DDDD_SPEC,
    FDDD_SPEC,
    FFPS_SPEC,
    PSSS_SPEC,
    SSSS_SPEC,
    ContractionSpec,
    OperatorFamily,
    OperatorSpec,
    TranslationInvariant,
    build_dppp_component_kernel,
    build_integral_ir,
    build_psss_kernel,
    build_shell_class_contraction_kernel,
    evaluate_fused_shell_component,
    evaluate_fused_shell_value,
)
from vibeqc_compiler.integral.shell_class import AXES, CENTERS


@pytest.mark.parametrize(
    ("spec", "component"),
    (
        (PSSS_SPEC, ("y", "", "", "")),
        (SSSS_SPEC, ("", "", "", "")),
    ),
)
def test_zero_order_pair_recurrence_matches_symbolic_oracle(
    spec: typing.Any, component: typing.Any
) -> None:
    """Verify generated low-order values and all-center force derivatives."""

    values = factored_dppp_variables(sample_variables())
    direct = build_shell_class_contraction_kernel(spec, component)
    fused = evaluate_fused_shell_component(spec, component, values)
    assert evaluate_fused_shell_value(spec, component, values) == pytest.approx(
        values["prefactor"] * direct.graph.evaluate(direct.value, values),
        rel=2.0e-12,
        abs=2.0e-12,
    )
    for center in range(4):
        for axis in range(3):
            assert fused[center][axis] == pytest.approx(
                direct.graph.evaluate(direct.gradients[center][axis], values),
                rel=3.0e-12,
                abs=3.0e-12,
            )


@pytest.mark.parametrize(
    "component",
    (
        ("xx", "xx", "xx", "xx"),
        ("xy", "xz", "yz", "zz"),
        ("zz", "zz", "zz", "zz"),
    ),
)
def test_dddd_tiled_recurrence_matches_symbolic_oracle(
    component: typing.Any,
) -> None:
    """Audit representative order-four/order-four Cartesian recurrences."""

    values = factored_dppp_variables(sample_variables())
    for order, value in enumerate(
        boys_values(
            values["rho"] * sum(values[f"difference_{axis}"] ** 2 for axis in AXES),
            10,
        )
    ):
        values[f"boys_{order}"] = value
    direct = build_shell_class_contraction_kernel(DDDD_SPEC, component)
    fused = evaluate_fused_shell_component(DDDD_SPEC, component, values)
    assert evaluate_fused_shell_value(DDDD_SPEC, component, values) == pytest.approx(
        values["prefactor"] * direct.graph.evaluate(direct.value, values),
        rel=2.0e-11,
        abs=2.0e-11,
    )
    for center in range(4):
        for axis in range(3):
            assert fused[center][axis] == pytest.approx(
                direct.graph.evaluate(direct.gradients[center][axis], values),
                rel=3.0e-11,
                abs=3.0e-11,
            )


@pytest.mark.parametrize(
    ("spec", "component"),
    (
        (FFPS_SPEC, ("xxx", "xyz", "z", "")),
        (FDDD_SPEC, ("xyz", "xy", "yz", "zz")),
    ),
)
def test_f_shell_recurrence_matches_symbolic_oracle(
    spec: typing.Any, component: typing.Any
) -> None:
    """Automate representative f-shell values and all-center gradients."""

    values = factored_dppp_variables(sample_variables())
    maximum_order = spec.maximum_force_coulomb_order
    argument = values["rho"] * sum(values[f"difference_{axis}"] ** 2 for axis in AXES)
    for order, value in enumerate(boys_values(argument, maximum_order + 1)):
        values[f"boys_{order}"] = value
    direct = build_shell_class_contraction_kernel(spec, component)
    fused = evaluate_fused_shell_component(spec, component, values)
    assert evaluate_fused_shell_value(spec, component, values) == pytest.approx(
        values["prefactor"] * direct.graph.evaluate(direct.value, values),
        rel=4.0e-11,
        abs=4.0e-11,
    )
    for center in range(4):
        for axis in range(3):
            assert fused[center][axis] == pytest.approx(
                direct.graph.evaluate(direct.gradients[center][axis], values),
                rel=5.0e-11,
                abs=5.0e-11,
            )


@pytest.mark.parametrize("argument", (0.0, 1.0e-10, 0.05, 1.0, 5.999))
@pytest.mark.parametrize("count", (1, 3, 7, 13))
def test_highest_order_boys_series_supports_downward_recurrence(
    argument: float, count: int
) -> None:
    """One highest-order series must reproduce every lower Boys value."""

    maximum_order = count - 1
    term = 1.0
    highest = 0.0
    for k in range(80):
        highest += term / (2 * maximum_order + 2 * k + 1)
        term *= -argument / (k + 1)
        if abs(term) < 1.0e-18:
            break
    candidate = [0.0] * count
    candidate[maximum_order] = highest
    exponential = math.exp(-argument)
    for order in range(maximum_order, 0, -1):
        candidate[order - 1] = (2.0 * argument * candidate[order] + exponential) / (
            2 * order - 1
        )

    reference = [
        sum(
            (-argument) ** k / (math.factorial(k) * (2 * order + 2 * k + 1))
            for k in range(80)
        )
        for order in range(count)
    ]
    assert candidate == pytest.approx(reference, rel=2.0e-12, abs=2.0e-14)


@pytest.mark.parametrize("p_axis", AXES)
def test_psss_symbolic_gradients_match_finite_difference(p_axis: str) -> None:
    kernel = build_psss_kernel(p_axis)
    values = sample_variables()
    argument = kernel.graph.evaluate(kernel.boys_argument, values)
    for order, value in enumerate(boys_values(argument)):
        values[f"boys_{order}"] = value

    step = 2.0e-6
    for center_index, center in enumerate(CENTERS[:3]):
        for axis_index, axis in enumerate(AXES):
            variable = f"{center}_{axis}"
            plus = dict(values)
            minus = dict(values)
            plus[variable] += step
            minus[variable] -= step
            numerical = (
                evaluate_value(kernel, plus) - evaluate_value(kernel, minus)
            ) / (2.0 * step)
            analytic = kernel.graph.evaluate(
                kernel.gradients[center_index][axis_index], values
            )
            assert analytic == pytest.approx(numerical, rel=2.0e-8, abs=2.0e-9)


def test_psss_fourth_center_uses_exact_translation_recovery() -> None:
    kernel = build_psss_kernel("x")
    values = sample_variables()
    argument = kernel.graph.evaluate(kernel.boys_argument, values)
    for order, value in enumerate(boys_values(argument)):
        values[f"boys_{order}"] = value
    for axis in range(3):
        total = sum(
            kernel.graph.evaluate(kernel.gradients[center][axis], values)
            for center in range(4)
        )
        assert total == pytest.approx(0.0, abs=2.0e-14)


def test_psss_oracle_uses_explicit_nonfinal_recovery_centers() -> None:
    """Keep the handwritten psss oracle aligned with derivative IR metadata."""

    operator = OperatorSpec(
        family=OperatorFamily.FOUR_CENTER_ERI,
        centers=(0, 1, 2, 3),
        invariants=(TranslationInvariant(dependent_center=1),),
    )
    force = ContractionSpec(
        consumer="direct_force",
        density="rhf|uhf",
        output="atomic_force",
    )
    integral = build_integral_ir(
        PSSS_SPEC,
        operator=operator,
        derivative=operator.nuclear_derivative(),
        contractions=(force,),
    )
    kernel = build_psss_kernel("x", integral=integral)
    values = sample_variables()
    argument = kernel.graph.evaluate(kernel.boys_argument, values)
    for order, value in enumerate(boys_values(argument)):
        values[f"boys_{order}"] = value

    # Center B is recovered, so the independent oracle roots are A/C/D and
    # the generated gradient tuple must retain the physical center positions.
    for center in (0, 2, 3):
        for axis, coordinate in enumerate(AXES):
            variable = f"{CENTERS[center]}_{coordinate}"
            plus = dict(values)
            minus = dict(values)
            plus[variable] += 2.0e-6
            minus[variable] -= 2.0e-6
            numerical = (
                evaluate_value(kernel, plus) - evaluate_value(kernel, minus)
            ) / 4.0e-6
            analytic = kernel.graph.evaluate(kernel.gradients[center][axis], values)
            assert analytic == pytest.approx(numerical, rel=2.0e-8, abs=2.0e-9)
    for axis in range(3):
        recovered = kernel.graph.evaluate(kernel.gradients[1][axis], values)
        independent_sum = sum(
            kernel.graph.evaluate(kernel.gradients[center][axis], values)
            for center in (0, 2, 3)
        )
        assert recovered == pytest.approx(-independent_sum, abs=2.0e-14)


@pytest.mark.parametrize(
    ("d_component", "p_components"),
    (("xx", "xxx"), ("xy", "xyz"), ("zz", "zyx")),
)
def test_dppp_symbolic_gradients_match_finite_difference(
    d_component: str, p_components: str
) -> None:
    kernel = build_dppp_component_kernel(d_component, tuple(p_components))
    values = sample_variables()
    argument = kernel.graph.evaluate(kernel.boys_argument, values)
    for order, value in enumerate(boys_values(argument, 7)):
        values[f"boys_{order}"] = value

    step = 1.0e-6
    for center_index, center in enumerate(CENTERS[:3]):
        for axis_index, axis in enumerate(AXES):
            variable = f"{center}_{axis}"
            plus = dict(values)
            minus = dict(values)
            plus[variable] += step
            minus[variable] -= step
            numerical = (
                evaluate_value(kernel, plus, 7) - evaluate_value(kernel, minus, 7)
            ) / (2.0 * step)
            analytic = kernel.graph.evaluate(
                kernel.gradients[center_index][axis_index], values
            )
            assert analytic == pytest.approx(numerical, rel=3.0e-7, abs=3.0e-8)


def test_dppp_translation_and_ket_pair_permutation_invariants() -> None:
    kernel = build_dppp_component_kernel("xy", tuple("xyz"))
    values = sample_variables()
    argument = kernel.graph.evaluate(kernel.boys_argument, values)
    for order, value in enumerate(boys_values(argument, 7)):
        values[f"boys_{order}"] = value
    for axis in range(3):
        total = sum(
            kernel.graph.evaluate(kernel.gradients[center][axis], values)
            for center in range(4)
        )
        assert total == pytest.approx(0.0, abs=2.0e-12)

    swapped_kernel = build_dppp_component_kernel("xy", tuple("xzy"))
    swapped_values = dict(values)
    swapped_values["gamma"], swapped_values["delta"] = (
        swapped_values["delta"],
        swapped_values["gamma"],
    )
    for axis in AXES:
        third = swapped_values[f"third_{axis}"]
        swapped_values[f"third_{axis}"] = swapped_values[f"fourth_{axis}"]
        swapped_values[f"fourth_{axis}"] = third
    assert evaluate_value(kernel, dict(values), 7) == pytest.approx(
        evaluate_value(swapped_kernel, swapped_values, 7),
        rel=2.0e-13,
        abs=2.0e-13,
    )
