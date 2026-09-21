"""Integral-IR ownership and derivative-intent codegen tests."""

from __future__ import annotations

import pytest
from codegen_fixtures import boys_values, factored_dppp_variables, sample_variables
from vibeqc_compiler.integral import (
    DPPP_SPEC,
    FOUR_CENTER_ERI_OPERATOR,
    FUSED_SHELL_SPEC_BY_NAME,
    PSPS_SPEC,
    PSSS_SPEC,
    ContractionOutput,
    ContractionSpec,
    DerivativeSpec,
    KernelConsumer,
    NuclearCoordinates,
    OperatorFamily,
    OperatorSpec,
    ScheduleIR,
    ScheduleKind,
    TranslationInvariant,
    build_dppp_component_kernel,
    build_fused_shell_plan,
    build_integral_ir,
    build_psss_kernel,
    build_rys_force_program,
    build_shell_class_component_kernel,
    build_shell_class_contraction_kernel,
    build_weighted_shell_contraction_kernel,
    cuda_target_info,
    emit_ppps_resident_bra_rys3_cuda,
    emit_rys_force_root_body_cuda,
    emit_shell_class_fused_cuda,
    evaluate_fused_shell_observables,
    evaluate_rys_component,
)

TEST_CUDA_TARGET = cuda_target_info("sm_120")


def test_integral_ir_retains_higher_derivative_intent_until_cuda_boundary() -> None:
    """Keep Hessian intent explicit while rejecting the first-gradient ABI."""

    derivative = FOUR_CENTER_ERI_OPERATOR.nuclear_derivative(order=2)
    integral = build_integral_ir(DPPP_SPEC, derivative=derivative)

    assert integral.derivative == derivative
    assert integral.contractions[0].output == ContractionOutput.NUCLEAR_DERIVATIVE
    assert integral.maximum_coulomb_order == 7
    assert integral.required_rys_roots == 4
    with pytest.raises(
        ValueError,
        match="CUDA force result ABI currently exposes only order-one derivatives",
    ):
        build_fused_shell_plan(DPPP_SPEC, integral=integral, target=TEST_CUDA_TARGET)


def test_operator_invariant_selects_derivative_recovery_without_force_magic() -> None:
    """Drive Rys derivative centers from operator-declared translation semantics."""

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
        DPPP_SPEC,
        operator=operator,
        derivative=operator.nuclear_derivative(),
        contractions=(force,),
    )
    assert integral.independent_derivative_centers == (0, 2, 3)
    assert integral.recovered_derivative_centers == (1,)

    program = build_rys_force_program(DPPP_SPEC, integral=integral)
    assert program.operator == operator
    assert program.derivative == integral.derivative
    assert program.independent_derivative_centers == (0, 2, 3)
    assert program.recovered_derivative_centers == (1,)
    assert program.independent_force_centers == (0, 2, 3)
    assert program.recovered_force_centers == (1,)


def test_fused_shell_plan_preserves_an_explicit_integral_ir() -> None:
    """Carry derivative/contraction intent into scheduling without rebuilding it."""

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
        DPPP_SPEC,
        operator=operator,
        derivative=operator.nuclear_derivative(),
        contractions=(force,),
    )
    plan = build_fused_shell_plan(DPPP_SPEC, integral=integral, target=TEST_CUDA_TARGET)
    assert plan.kernel.integral is integral
    assert plan.kernel.integral.independent_derivative_centers == (0, 2, 3)
    assert plan.kernel.integral.recovered_derivative_centers == (1,)

    with pytest.raises(ValueError, match="consumer and integral"):
        build_fused_shell_plan(
            DPPP_SPEC,
            integral=integral,
            consumers=(KernelConsumer.FOCK,),
            target=TEST_CUDA_TARGET,
        )


def test_symbolic_kernel_builders_preserve_explicit_integral_ir() -> None:
    """Keep one mathematical request attached across every shell builder."""

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
    dppp_integral = build_integral_ir(
        DPPP_SPEC,
        operator=operator,
        derivative=operator.nuclear_derivative(),
        contractions=(force,),
    )
    component = DPPP_SPEC.components[0]

    assert (
        build_shell_class_component_kernel(
            DPPP_SPEC,
            component,
            integral=dppp_integral,
        ).integral
        is dppp_integral
    )
    assert (
        build_dppp_component_kernel(
            component[0],
            component[1:],
            integral=dppp_integral,
        ).integral
        is dppp_integral
    )
    assert (
        build_shell_class_contraction_kernel(
            DPPP_SPEC,
            component,
            integral=dppp_integral,
        ).integral
        is dppp_integral
    )
    assert (
        build_weighted_shell_contraction_kernel(
            DPPP_SPEC,
            component_indices=(0,),
            integral=dppp_integral,
        ).integral
        is dppp_integral
    )

    psss_integral = build_integral_ir(
        PSSS_SPEC,
        operator=operator,
        derivative=operator.nuclear_derivative(),
        contractions=(force,),
    )
    assert build_psss_kernel("x", integral=psss_integral).integral is psss_integral


def test_shell_contraction_kernel_uses_explicit_derivative_centers() -> None:
    """Generate direct center-D roots when the IR recovers center B."""

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
        DPPP_SPEC,
        operator=operator,
        derivative=operator.nuclear_derivative(),
        contractions=(force,),
    )
    component = DPPP_SPEC.components[0]
    full = build_shell_class_component_kernel(DPPP_SPEC, component)
    full_values = sample_variables()
    argument = full.graph.evaluate(full.boys_argument, full_values)
    for order, value in enumerate(
        boys_values(argument, DPPP_SPEC.maximum_force_coulomb_order + 1)
    ):
        full_values[f"boys_{order}"] = value
    factored_values = factored_dppp_variables(full_values)
    custom = build_shell_class_contraction_kernel(
        DPPP_SPEC,
        component,
        integral=integral,
    )
    custom_component = build_shell_class_component_kernel(
        DPPP_SPEC,
        component,
        integral=integral,
    )

    for center in (0, 2, 3):
        for axis in range(3):
            assert custom_component.graph.evaluate(
                custom_component.gradients[center][axis],
                full_values,
            ) == pytest.approx(
                full.graph.evaluate(full.gradients[center][axis], full_values),
                rel=1.0e-13,
                abs=1.0e-13,
            )
    for axis in range(3):
        recovered = custom_component.graph.evaluate(
            custom_component.gradients[1][axis],
            full_values,
        )
        independent_sum = sum(
            custom_component.graph.evaluate(
                custom_component.gradients[center][axis],
                full_values,
            )
            for center in (0, 2, 3)
        )
        assert recovered == pytest.approx(-independent_sum, rel=1.0e-13, abs=1.0e-13)

    for center in (0, 2, 3):
        for axis in range(3):
            assert custom.graph.evaluate(
                custom.gradients[center][axis], factored_values
            ) == pytest.approx(
                full.graph.evaluate(full.gradients[center][axis], full_values),
                rel=3.0e-11,
                abs=3.0e-11,
            )
    for axis in range(3):
        recovered = custom.graph.evaluate(custom.gradients[1][axis], factored_values)
        independent_sum = sum(
            custom.graph.evaluate(custom.gradients[center][axis], factored_values)
            for center in (0, 2, 3)
        )
        assert recovered == pytest.approx(-independent_sum, rel=1.0e-13, abs=1.0e-13)


def test_numeric_recurrence_oracles_follow_explicit_derivative_centers() -> None:
    """Keep fused and Rys host oracles aligned with non-final recovery."""

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
        DPPP_SPEC,
        operator=operator,
        derivative=operator.nuclear_derivative(),
        contractions=(force,),
    )
    values = factored_dppp_variables(sample_variables())
    component = DPPP_SPEC.components[0]

    fused = evaluate_fused_shell_observables(
        DPPP_SPEC,
        component,
        values,
        integral=integral,
    )
    rys = evaluate_rys_component(
        DPPP_SPEC,
        component,
        values,
        integral=integral,
    )
    for actual in (fused, rys):
        for center in (0, 2, 3):
            for axis in range(3):
                assert actual.gradients[center][axis] == pytest.approx(
                    fused.gradients[center][axis],
                    rel=2.0e-12,
                    abs=2.0e-12,
                )
        for axis in range(3):
            recovered = actual.gradients[1][axis]
            independent_sum = sum(
                actual.gradients[center][axis] for center in (0, 2, 3)
            )
            assert recovered == pytest.approx(
                -independent_sum,
                rel=2.0e-12,
                abs=2.0e-12,
            )


def test_rys_root_body_packs_nonfinal_recovery_centers_by_ir_order() -> None:
    """Keep force slots dense when translation recovers a non-final center."""

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
        DPPP_SPEC,
        operator=operator,
        derivative=operator.nuclear_derivative(),
        contractions=(force,),
    )
    body = emit_rys_force_root_body_cuda(
        DPPP_SPEC,
        component_group=1,
        integral=integral,
    )

    # Independent centers are A/C/D, so their force slots are 0..2, 3..5,
    # and 6..8.  The D derivative must use its own exponent instead of being
    # accidentally emitted at the old center*3 offset (or recovered as D).
    assert "force_3 += (gamma2 *" in body
    assert "force_6 += (delta2 *" in body
    assert "force_9" not in body


def test_ppps_resident_rys_lowering_uses_nonfinal_recovery_centers() -> None:
    """Keep resident Rys force slots and atomics aligned with explicit IR."""

    spec = FUSED_SHELL_SPEC_BY_NAME["ppps"]
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
        spec,
        operator=operator,
        derivative=operator.nuclear_derivative(),
        contractions=(force,),
        recurrence="rys3",
    )
    source = emit_ppps_resident_bra_rys3_cuda(integral=integral)

    # A/C/D are independent under center-B recovery and must occupy dense
    # slots 0..8; the resident path must not regress to a force_9 write.
    assert "force_3 += (gamma2 *" in source
    assert "force_6 += (delta2 *" in source
    assert "force_9" not in source
    assert "const double cdx = fourth.x - third.x;" in source
    assert "const double delta2 = 2.0 * q * fourth_product_scale;" in source

    recovery_begin = source.index("const double fourth_force_0")
    recovery = source[recovery_begin : recovery_begin + 900]
    assert "static_cast<std::size_t>(task.atom[1])" in recovery
    assert "static_cast<std::size_t>(task.atom[3])" not in recovery

    # Only independent bra center A is warp-reduced now; center B is the
    # recovered output and must not be treated as a resident bra slot.
    assert "context.ket_tasks[resident.ket_begin].atom[0]" in source
    assert "context.ket_tasks[resident.ket_begin].atom[1]" not in source


def test_fock_only_ir_has_no_implicit_derivative() -> None:
    """Avoid increasing Coulomb order when only a value contraction is requested."""

    integral = build_integral_ir(DPPP_SPEC, consumers=(KernelConsumer.FOCK,))
    assert integral.derivative is None
    assert integral.independent_derivative_centers == ()
    assert integral.recovered_derivative_centers == ()
    assert integral.maximum_coulomb_order == integral.value_coulomb_order

    # The symbolic builders consume the same boundary rather than silently
    # constructing the extra first-derivative Boys state for a value-only IR.
    component_kernel = build_shell_class_component_kernel(
        DPPP_SPEC,
        DPPP_SPEC.components[0],
        integral=integral,
    )
    contraction_kernel = build_shell_class_contraction_kernel(
        DPPP_SPEC,
        DPPP_SPEC.components[0],
        integral=integral,
    )
    for kernel in (component_kernel, contraction_kernel):
        boys_variables = {
            str(node.payload)
            for node in kernel.graph.nodes
            if node.operation == "variable"
            and isinstance(node.payload, str)
            and node.payload.startswith("boys_")
        }
        assert "boys_5" in boys_variables
        assert "boys_6" not in boys_variables

    with pytest.raises(ValueError, match="requires at least one contraction"):
        build_integral_ir(DPPP_SPEC, consumers=())


def test_derivative_cannot_invent_an_operator_invariant() -> None:
    """Require exact recovery relations to originate on the operator spec."""

    undeclared_recovery = DerivativeSpec(
        order=1,
        parameters=NuclearCoordinates(),
        invariants=(TranslationInvariant(dependent_center=0),),
    )
    with pytest.raises(ValueError, match="must be declared by the operator"):
        build_integral_ir(DPPP_SPEC, derivative=undeclared_recovery)


def test_subgroup_force_lowering_uses_explicit_nonfinal_recovery_slots() -> None:
    """Keep subgroup force output aligned with a center-B recovery IR."""

    spec = PSPS_SPEC
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
        spec,
        operator=operator,
        derivative=operator.nuclear_derivative(),
        contractions=(force,),
    )
    schedule = ScheduleIR(
        kind=ScheduleKind.SUBGROUP_TASKS,
        block_threads=128,
        component_tile=spec.component_count,
        tasks_per_warp=4,
        shared_coulomb=True,
    )
    source = emit_shell_class_fused_cuda(
        spec,
        build_fused_shell_plan(
            spec, integral=integral, schedule=schedule, target=TEST_CUDA_TARGET
        ),
    )

    assert "GeneratedPspsSubgroupForceStorage" in source
    assert "double decay_gradients[4][3]" in source
    assert "geometry.decay_gradients[3][coordinate]" in source
    assert "gradient[1][coordinate] = -gradient[0][coordinate]" in source
