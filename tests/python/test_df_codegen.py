"""Raw DF mathematics, normalization, and independent reference contracts."""

import math
from itertools import product

import pytest
from vibeqc_compiler.integral.df_values import (
    build_df_axis_moment,
    build_df_component_kernel,
    build_df_value_ir,
    evaluate_df_primitive,
)
from vibeqc_compiler.integral.ir import OperatorFamily
from vibeqc_compiler.integral.ir_serialization import integral_from_payload
from vibeqc_compiler.integral.shell_spec import cartesian_components


def test_raw_df_operator_roles_domain_and_permutations():
    for family, count in (
        (OperatorFamily.COULOMB_METRIC, 2),
        (OperatorFamily.THREE_CENTER_ERI, 3),
    ):
        for angular in product(range(4), repeat=count):
            integral = build_df_value_ir(family, angular)
            assert integral.derivative is None
            assert integral.maximum_coulomb_order == sum(angular)
            assert integral.consumers == frozenset()
            assert integral.operator.permutations == (
                tuple(range(count)),
                (1, 0, *range(2, count)),
            )
            roles = tuple(shell.role.value for shell in integral.signature.shells)
            assert roles == (
                ("auxiliary",) * 2
                if count == 2
                else ("orbital", "orbital", "auxiliary")
            )
            assert (
                integral.contractions[0].layout.shape
                == integral.signature.component_shape
            )
    with pytest.raises(ValueError, match="s/p/d/f"):
        build_df_value_ir("coulomb_metric", (3, 4))
    with pytest.raises(ValueError, match="basis roles"):
        build_df_value_ir("three_center_eri", (1, 2))
    with pytest.raises(ValueError, match="DF values"):
        build_df_value_ir("overlap", (0, 0))


def test_cuda_df_inventory_retains_all_operator_signatures_and_exact_root_counts():
    from vibeqc_compiler.integral.df_cuda import df_program_inventory

    inventory = df_program_inventory()
    assert len(inventory["programs"]) == 16 + 64
    root_counts = set()
    for payload in inventory["programs"]:
        integral = integral_from_payload(payload)
        count = sum(integral.signature.angular) // 2 + 1
        assert integral.recurrence == f"rys{count}"
        assert integral.required_rys_roots == count
        root_counts.add(count)
    assert root_counts == {1, 2, 3, 4, 5}
    with pytest.raises(ValueError, match="requires rys1"):
        build_df_value_ir("coulomb_metric", (0, 0), recurrence="rys2")


def test_metric_ss_and_pp_are_coulomb_integrals_at_coincident_centers():
    p, q = 0.7, 1.3
    prefactor = 2 * math.pi**2.5 / (p * q * math.sqrt(p + q))
    centers = ((0, 0, 0), (0, 0, 0))
    ss = build_df_component_kernel(
        build_df_value_ir("coulomb_metric", (0, 0)), ("", "")
    )
    assert evaluate_df_primitive(ss, (p, q), centers) == pytest.approx(
        prefactor, rel=2e-15
    )
    pp = build_df_value_ir("coulomb_metric", (1, 1))
    xx = build_df_component_kernel(pp, ("x", "x"))
    xy = build_df_component_kernel(pp, ("x", "y"))
    assert evaluate_df_primitive(xx, (p, q), centers) == pytest.approx(
        prefactor / (6 * (p + q)), rel=2e-15
    )
    assert evaluate_df_primitive(xy, (p, q), centers) == 0
    assert ss.coulomb_states == ((0, 0, 0),)
    assert xx.coulomb_states == ((2, 0, 0),)
    with pytest.raises(ValueError, match="positive"):
        evaluate_df_primitive(ss, (p, 0), centers)


@pytest.mark.parametrize("angular", [(1, 3), (2, 1, 3)])
def test_df_translation_and_only_declared_exchange_symmetry(angular):
    family = "coulomb_metric" if len(angular) == 2 else "three_center_eri"
    exponents = (0.6, 0.8, 1.0)[: len(angular)]
    centers = ((0.13, -0.31, 0.24), (-0.43, 0.27, 0.51), (0.68, -0.14, -0.22))[
        : len(angular)
    ]
    components = tuple(
        cartesian_components(l)[len(cartesian_components(l)) // 2] for l in angular
    )
    kernel = build_df_component_kernel(build_df_value_ir(family, angular), components)
    value = evaluate_df_primitive(kernel, exponents, centers)
    translated = tuple(
        tuple(x + shift for x, shift in zip(center, (1.1, -0.7, 0.4)))
        for center in centers
    )
    assert evaluate_df_primitive(kernel, exponents, translated) == pytest.approx(
        value, abs=2e-13, rel=2e-13
    )
    permutation = (1, 0, *range(2, len(angular)))
    swapped = build_df_component_kernel(
        build_df_value_ir(family, tuple(angular[i] for i in permutation)),
        tuple(components[i] for i in permutation),
    )
    assert evaluate_df_primitive(
        swapped,
        tuple(exponents[i] for i in permutation),
        tuple(centers[i] for i in permutation),
    ) == pytest.approx(value, abs=2e-13, rel=2e-13)


@pytest.mark.parametrize(
    "angular,reference",
    [
        ((0, 0), 16.658542388560555),
        ((1, 1), 4.497095761152227),
        ((3, 3), 6.81256088982111),
        ((0, 0, 0), 2.1386977635507054),
        ((1, 1, 1), -0.16691493963014886),
        ((3, 3, 3), -0.1296430096520732),
    ],
)
def test_raw_df_values_match_independent_libcint_vertical_slices(angular, reference):
    """Pin int2c2e/int3c2e values with unit Cartesian primitive normalization.

    PySCF/libcint reference geometry and basis exactly match the inputs below;
    shell-local first components are x**l in CCA order. The expected data are
    independent of both the Hermite DAG and the planned CUDA Rys lowering.
    """
    exponents = tuple(0.6 + 0.2 * i for i in range(len(angular)))
    centers = ((0.13, -0.31, 0.24), (-0.43, 0.27, 0.51), (0.68, -0.14, -0.22))[
        : len(angular)
    ]
    family = "coulomb_metric" if len(angular) == 2 else "three_center_eri"
    components = tuple("x" * l for l in angular)
    kernel = build_df_component_kernel(build_df_value_ir(family, angular), components)
    value = evaluate_df_primitive(kernel, exponents, centers)
    for a, l in zip(exponents, angular):
        odd_double_factorial = math.prod(range(1, 2 * l, 2))
        value *= (
            (2 * a / math.pi) ** 0.75
            * (4 * a) ** (0.5 * l)
            / math.sqrt(odd_double_factorial)
        )
    assert value == pytest.approx(reference, abs=2e-12, rel=2e-12)


def test_pruned_rys_axis_moments_include_all_same_and_cross_coordinate_pairings():
    graph, root = build_df_axis_moment(1, 1, 1)
    variables = {
        "mean_0": 0.2,
        "mean_1": -0.3,
        "mean_2": 0.4,
        "variance_x": 0.7,
        "variance_y": 0.8,
        "covariance_xy": 0.1,
    }
    reference = 0.2 * -0.3 * 0.4 + 0.7 * 0.4 + 0.1 * (0.2 - 0.3)
    assert graph.evaluate(root, variables) == pytest.approx(reference)
    graph, root = build_df_axis_moment(3, 0, 3)
    variables.update(mean_0=0, mean_1=0, mean_2=0)
    assert graph.evaluate(root, variables) == pytest.approx(
        9 * 0.7 * 0.8 * 0.1 + 6 * 0.1**3
    )
    assert graph.analyze_ssa((root,)).root_count == 1
