"""Issue #673 Slice B: compile-time IntegralIR specialization pruning."""

from math import comb

import pytest
from vibeqc_compiler.integral import (
    DPPP_SPEC,
    PSSS_SPEC,
    KernelConsumer,
    build_fused_shell_plan,
    build_integral_ir,
    cuda_target_info,
    specialize_integral_ir,
)
from vibeqc_compiler.integral.lowering.dispatch import _specialize_fock_plan


def test_fock_output_pruning_removes_derivative_intent_without_mutating_source() -> (
    None
):
    mixed = build_integral_ir(
        DPPP_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
    )
    fock_contraction = next(
        item
        for item in mixed.contractions
        if item.kernel_consumer == KernelConsumer.FOCK
    )

    specialized = specialize_integral_ir(
        mixed,
        consumers=(KernelConsumer.FOCK,),
    )

    assert mixed.derivative is not None
    assert specialized.derivative is None
    assert specialized.consumers == frozenset((KernelConsumer.FOCK,))
    assert specialized.contractions == (fock_contraction,)
    assert specialized.maximum_coulomb_order == specialized.value_coulomb_order
    assert specialized.maximum_coulomb_order < mixed.maximum_coulomb_order


def test_rys_specialization_folds_root_count_from_derivative_to_value_order() -> None:
    force = build_integral_ir(
        PSSS_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
    )
    force = force.__class__(
        spec=force.spec,
        operator=force.operator,
        derivative=force.derivative,
        contractions=force.contractions,
        recurrence=f"rys{force.required_rys_roots}",
    )

    specialized = specialize_integral_ir(
        force,
        consumers=(KernelConsumer.FOCK,),
    )

    assert specialized.derivative is None
    assert specialized.recurrence == f"rys{specialized.required_rys_roots}"
    assert specialized.required_rys_roots < force.required_rys_roots


def test_specialization_rejects_unknown_empty_and_duplicate_output_demands() -> None:
    fock = build_integral_ir(DPPP_SPEC, consumers=(KernelConsumer.FOCK,))

    with pytest.raises(ValueError, match="unavailable"):
        specialize_integral_ir(fock, consumers=(KernelConsumer.FORCE,))
    with pytest.raises(ValueError, match="requested consumers"):
        specialize_integral_ir(fock, consumers=())
    with pytest.raises(ValueError, match="duplicate"):
        specialize_integral_ir(
            fock,
            consumers=(KernelConsumer.FOCK, KernelConsumer.FOCK),
        )


def test_generated_hf_fock_subplan_uses_value_only_coulomb_state_space() -> None:
    mixed = build_integral_ir(
        DPPP_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
    )
    plan = build_fused_shell_plan(
        DPPP_SPEC,
        integral=mixed,
        target=cuda_target_info("sm_120"),
    )

    fock_plan = _specialize_fock_plan(plan)

    fock_integral = fock_plan.kernel.integral
    assert fock_integral.consumers == frozenset((KernelConsumer.FOCK,))
    assert fock_integral.derivative is None
    assert len(fock_plan.coulomb_states) == comb(
        fock_integral.value_coulomb_order + 3,
        3,
    )
    assert len(fock_plan.coulomb_states) < len(plan.coulomb_states)
