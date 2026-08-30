"""Correctness tests for build-time shell-class symbolic code generation."""

from __future__ import annotations

import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.vibeqc_codegen import (
    DDDD_SPEC,
    DDPS_SPEC,
    DPDS_SPEC,
    DPPP_SPEC,
    FDDD_SPEC,
    FFPS_SPEC,
    FOUR_CENTER_ERI_OPERATOR,
    FUSED_SHELL_SPEC_BY_NAME,
    FUSED_SHELL_SPECS,
    PPSS_SPEC,
    PSPS_SPEC,
    PSSS_SPEC,
    SSSS_SPEC,
    AlgebraForm,
    AlgebraFusion,
    AlgebraOrdering,
    AlgebraPlacement,
    ContractionConsumer,
    ContractionSpec,
    CudaTargetInfo,
    DensityModel,
    DerivativeSpec,
    IntegralIR,
    KernelConsumer,
    NuclearCoordinates,
    NvrtcCacheSpec,
    OperatorFamily,
    OperatorSpec,
    PairOrientation,
    PairStorage,
    RysRecurrenceKind,
    RysState,
    ScheduleIR,
    ScheduleKind,
    ShellClassSpec,
    TranslationInvariant,
    build_dppp_component_kernel,
    build_dppp_contraction_kernel,
    build_dppp_fused_plan,
    build_fused_shell_plan,
    build_integral_ir,
    build_ppps_rys_force_program,
    build_psss_kernel,
    build_rys_axis_program,
    build_rys_force_program,
    build_shell_class_component_kernel,
    build_shell_class_contraction_kernel,
    build_weighted_shell_contraction_kernel,
    cartesian_components,
    cuda_target_info,
    dppp_components,
    emit_dppp_fused_cuda,
    emit_ppps_resident_bra_rys3_cuda,
    emit_ppps_rys3_root_body_cuda,
    emit_rys2_roots_cuda,
    emit_rys3_roots_cuda,
    emit_rys4_roots_cuda,
    emit_rys5_roots_cuda,
    emit_rys_force_root_body_cuda,
    emit_shell_class_fused_cuda,
    evaluate_dppp_fused_component,
    evaluate_fused_shell_component,
    evaluate_fused_shell_observables,
    evaluate_fused_shell_value,
    evaluate_ppps_rys_component,
    evaluate_rys_component,
    nvrtc_cache_key,
    rys2_table_roots_weights,
    rys3_roots_weights,
    rys3_table_roots_weights,
    rys4_roots_weights,
    rys4_table_roots_weights,
    rys5_roots_weights,
    rys5_table_roots_weights,
    rys_boys_values,
    schedule_candidates,
)
from tools.vibeqc_codegen.autotune import (
    StaticAlgebraModel,
    _analysis_roots,
    _compile_trial,
    _oracle_symbol_prefix,
    _packed_force_geometry_analysis,
    _run_autotune,
    emit_schedule_driver,
    emit_schedule_oracle_translation_unit,
    emit_schedule_resource_translation_unit,
    emit_schedule_translation_unit,
    schedule_payload,
    static_algebra_model,
    supported_schedule_trials,
    update_manifest_payload,
)
from tools.vibeqc_codegen.backend import TargetInfo, TargetScheduleShape
from tools.vibeqc_codegen.batch_benchmark import (
    DEFAULT_CANDIDATES,
    candidate_specs,
    emit_batch_driver,
    emit_candidate_translation_unit,
    rank_profiled_candidates,
)
from tools.vibeqc_codegen.benchmark import (
    emit_dppp_benchmark_cuda,
    emit_shell_class_benchmark_cuda,
    emit_shell_class_oracle_cuda,
)
from tools.vibeqc_codegen.production import (
    _PRODUCTION_PRELUDE,
    _partition_production_selections,
    _schedule_from_payload,
    emit_registry_header,
    load_production_fock_manifest,
    load_production_kernel_selections,
    load_production_manifest,
    resolve_production_profile,
    write_production_bundle,
    write_production_bundles,
)
from tools.vibeqc_codegen.shell_class import (
    AXES,
    CENTERS,
    emit_dppp_component_cuda,
    emit_dppp_contraction_cuda,
    emit_psss_cuda,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


RTX5090_DPPP_RESOURCE_LIMITS = {
    "generated_dppp_shell_class_force_rhf_kernel": (168, 40, 2072),
    "generated_dppp_shell_class_force_uhf_kernel": (168, 40, 2072),
    "generated_dppp_shell_class_force_rhf_persistent_kernel": (164, 40, 2080),
    "generated_dppp_shell_class_force_uhf_persistent_kernel": (164, 40, 2080),
}
RTX5090_DPPP_UNIFORM_RYS4_RESOURCE_LIMITS = {
    # Relocatable production shards retain the 1 KiB component-activity table
    # that a whole-program cubin compile may fold into another shared region.
    # Record the larger production-object envelope observed with CUDA 12.9.
    "generated_dppp_shell_class_force_rhf_kernel": (255, 168, 37896),
    "generated_dppp_shell_class_force_uhf_kernel": (255, 168, 37896),
    "generated_dppp_shell_class_force_rhf_persistent_kernel": (
        255,
        168,
        37896,
    ),
    "generated_dppp_shell_class_force_uhf_persistent_kernel": (
        255,
        168,
        37896,
    ),
}
RTX5090_DPDS_RESOURCE_LIMITS = {
    "generated_dpds_shell_class_force_rhf_kernel": (160, 40, 1880),
    "generated_dpds_shell_class_force_uhf_kernel": (160, 40, 1880),
    "generated_dpds_shell_class_force_rhf_persistent_kernel": (160, 40, 1888),
    "generated_dpds_shell_class_force_uhf_persistent_kernel": (160, 40, 1888),
}
RTX5090_DDPS_RESOURCE_LIMITS = {
    "generated_ddps_shell_class_force_rhf_kernel": (164, 64, 1880),
    "generated_ddps_shell_class_force_uhf_kernel": (164, 64, 1880),
    "generated_ddps_shell_class_force_rhf_persistent_kernel": (160, 64, 1888),
    "generated_ddps_shell_class_force_uhf_persistent_kernel": (160, 64, 1888),
}
RTX5090_PSPS_RESOURCE_LIMITS = {
    "generated_psps_shell_class_force_rhf_persistent_kernel": (246, 0, 3408),
    "generated_psps_shell_class_force_uhf_persistent_kernel": (246, 0, 3408),
}
RTX5090_PPSS_RESOURCE_LIMITS = {
    "generated_ppss_shell_class_force_rhf_persistent_kernel": (246, 0, 3408),
    "generated_ppss_shell_class_force_uhf_persistent_kernel": (246, 0, 3408),
}
RTX5090_PPPS_SCALAR_THREAD_RESOURCE_LIMITS = {
    "generated_ppps_shell_class_force_rhf_persistent_kernel": (168, 0, 27000),
    "generated_ppps_shell_class_force_uhf_persistent_kernel": (168, 0, 27000),
}
RTX5090_DPSS_SCALAR_RYS3_RESOURCE_LIMITS = {
    "generated_dpss_shell_class_force_rhf_persistent_kernel": (252, 0, 6224),
    "generated_dpss_shell_class_force_uhf_persistent_kernel": (252, 0, 6224),
}


@pytest.mark.parametrize(
    ("maximum_order", "series_threshold"),
    ((0, 1.0e-8), (1, 0.25), (2, 0.75), (3, 1.25), (4, 2.0)),
)
def test_generated_low_order_boys_thresholds_preserve_upward_recurrence(
    maximum_order: int, series_threshold: float
):
    """Keep the fast low-order branch accurate at its least stable point."""

    threshold_literal = "1.0e-8" if maximum_order == 0 else str(series_threshold)
    assert f"MaximumOrder == {maximum_order} ? {threshold_literal}" in (
        _PRODUCTION_PRELUDE
    )
    for argument in (
        series_threshold,
        series_threshold + 1.0e-6,
        0.5 * (series_threshold + 6.0),
        6.0,
    ):
        values = [0.5 * math.sqrt(math.pi / argument) * math.erf(math.sqrt(argument))]
        exponential = math.exp(-argument)
        for order in range(1, maximum_order + 1):
            values.append(
                ((2.0 * order - 1.0) * values[-1] - exponential) / (2.0 * argument)
            )
        reference = rys_boys_values(argument, maximum_order + 1)[maximum_order]
        assert values[maximum_order] == pytest.approx(
            reference, rel=5.0e-14, abs=1.0e-15
        )


def test_integral_ir_has_no_accelerator_schedule_fields():
    """Keep scientific intent independent of backend execution geometry."""

    assert set(IntegralIR.__dataclass_fields__) == {
        "spec",
        "operator",
        "derivative",
        "contractions",
        "recurrence",
    }
    synthetic = TargetInfo(
        backend="synthetic",
        architecture="wave64",
        subgroup_size=64,
        maximum_workgroup_threads=256,
        maximum_resident_workgroups=8,
    )
    TargetScheduleShape(128, 64).validate_for(synthetic)
    with pytest.raises(ValueError, match="subgroup size"):
        TargetScheduleShape(128, 32).validate_for(synthetic)


def test_generic_cuda_emitter_uses_backend_lowering_not_dppp_compatibility():
    """Keep generic compilation independent of historical shell adapters."""

    emitter = (
        REPOSITORY_ROOT / "tools" / "vibeqc_codegen" / "cuda_emitter.py"
    ).read_text(encoding="utf-8")
    compatibility = (
        REPOSITORY_ROOT / "tools" / "vibeqc_codegen" / "dppp_dispatch.py"
    ).read_text(encoding="utf-8")
    assert "from . import cuda_lowering as _implementation" in emitter
    assert "dppp_dispatch" not in emitter
    assert "from .cuda_lowering import" in compatibility
    assert "emit_shell_class_fused_cuda" not in compatibility


@pytest.mark.parametrize("architecture", ("sm_80", "sm_86", "sm_89", "sm_90", "sm_120"))
def test_cuda_target_catalog_covers_the_compile_matrix(architecture: str):
    """Expose target-derived scheduling and resource limits for supported SMs."""

    target = cuda_target_info(architecture)
    assert isinstance(target, CudaTargetInfo)
    assert target.architecture == architecture
    assert target.warp_size == 32
    assert target.maximum_threads_per_block == 1024
    assert target.tuning_maximum_shared_bytes <= target.shared_memory_per_block


def test_integral_and_schedule_irs_separate_math_from_cuda_mapping():
    """Expose derivative/contraction intent independently from CUDA mapping."""

    integral = build_integral_ir(
        DPPP_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
    )
    assert integral.operator == FOUR_CENTER_ERI_OPERATOR
    assert integral.derivative == DerivativeSpec(
        order=1,
        parameters=NuclearCoordinates(),
        invariants=(TranslationInvariant(),),
    )
    assert {item.consumer for item in integral.contractions} == {
        ContractionConsumer.DIRECT_FOCK,
        ContractionConsumer.DIRECT_FORCE,
    }
    assert all(
        item.density == frozenset(DensityModel) for item in integral.contractions
    )
    assert integral.value_coulomb_order == 5
    assert integral.maximum_coulomb_order == 6
    assert integral.requested_derivative_centers == (0, 1, 2, 3)
    assert integral.independent_force_centers == (0, 1, 2)
    assert integral.recovered_derivative_centers == (3,)
    candidates = schedule_candidates(integral)
    assert [item.kind for item in candidates] == [
        ScheduleKind.COMPONENT_LANES,
        ScheduleKind.TILED_COMPONENTS,
        ScheduleKind.TILED_COMPONENTS,
    ]
    assert candidates[0].block_threads == 192
    assert [item.component_tile for item in candidates[1:]] == [64, 128]


def test_operator_invariant_selects_derivative_recovery_without_force_magic():
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


def test_fused_shell_plan_preserves_an_explicit_integral_ir():
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
    plan = build_fused_shell_plan(DPPP_SPEC, integral=integral)
    assert plan.kernel.integral is integral
    assert plan.kernel.integral.independent_derivative_centers == (0, 2, 3)
    assert plan.kernel.integral.recovered_derivative_centers == (1,)

    with pytest.raises(ValueError, match="consumer and integral"):
        build_fused_shell_plan(
            DPPP_SPEC,
            integral=integral,
            consumers=(KernelConsumer.FOCK,),
        )


def test_symbolic_kernel_builders_preserve_explicit_integral_ir():
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

    assert build_shell_class_component_kernel(
        DPPP_SPEC,
        component,
        integral=dppp_integral,
    ).integral is dppp_integral
    assert build_dppp_component_kernel(
        component[0],
        component[1:],
        integral=dppp_integral,
    ).integral is dppp_integral
    assert build_shell_class_contraction_kernel(
        DPPP_SPEC,
        component,
        integral=dppp_integral,
    ).integral is dppp_integral
    assert build_weighted_shell_contraction_kernel(
        DPPP_SPEC,
        component_indices=(0,),
        integral=dppp_integral,
    ).integral is dppp_integral

    psss_integral = build_integral_ir(
        PSSS_SPEC,
        operator=operator,
        derivative=operator.nuclear_derivative(),
        contractions=(force,),
    )
    assert build_psss_kernel("x", integral=psss_integral).integral is psss_integral


def test_shell_contraction_kernel_uses_explicit_derivative_centers():
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
            assert custom.graph.evaluate(custom.gradients[center][axis], factored_values) == pytest.approx(
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


def test_numeric_recurrence_oracles_follow_explicit_derivative_centers():
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

def test_rys_root_body_packs_nonfinal_recovery_centers_by_ir_order():
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


def test_ppps_resident_rys_lowering_uses_nonfinal_recovery_centers():
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


def test_fock_only_ir_has_no_implicit_derivative():
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


def test_derivative_cannot_invent_an_operator_invariant():
    """Require exact recovery relations to originate on the operator spec."""

    undeclared_recovery = DerivativeSpec(
        order=1,
        parameters=NuclearCoordinates(),
        invariants=(TranslationInvariant(dependent_center=0),),
    )
    with pytest.raises(ValueError, match="must be declared by the operator"):
        build_integral_ir(DPPP_SPEC, derivative=undeclared_recovery)


def test_small_shell_schedule_space_includes_packed_and_cooperative_variants():
    """Allow tuning to choose task packing instead of one fixed mapping."""

    candidates = schedule_candidates(build_integral_ir(PSPS_SPEC))
    assert [item.kind for item in candidates[:5]] == [
        ScheduleKind.PACKED_TASKS,
        ScheduleKind.SHELL_TASK,
        ScheduleKind.SUBGROUP_TASKS,
        ScheduleKind.SUBGROUP_TASKS,
        ScheduleKind.COMPONENT_LANES,
    ]
    assert candidates[0].tasks_per_warp == 32
    assert candidates[2].subgroup_lanes == 16
    assert candidates[2].tasks_per_block == 16
    assert candidates[3].subgroup_lanes == 8
    assert candidates[3].tasks_per_block == 32


def test_subgroup_schedule_advances_independent_ppps_tasks_per_block():
    """Keep task-local barriers and reductions inside each lane subgroup."""

    spec = FUSED_SHELL_SPEC_BY_NAME["ppps"]
    schedule = next(
        item
        for item in schedule_candidates(
            build_integral_ir(
                spec,
                consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
            )
        )
        if item.kind == ScheduleKind.SUBGROUP_TASKS and item.tasks_per_warp == 4
    )
    plan = build_fused_shell_plan(
        spec,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=schedule,
    )
    source = emit_shell_class_fused_cuda(spec, plan)
    assert schedule.block_threads == 256
    assert schedule.subgroup_lanes == 8
    assert schedule.tasks_per_block == 32
    assert "GeneratedPppsSubgroupForceStorage" in source
    assert "GeneratedPppsSubgroupFockStorage" in source
    assert "state += 8U" in source
    assert "atomicAdd(task_head, 1U)" in source
    assert "__syncwarp(subgroup_mask)" in source
    assert (
        "__syncthreads()"
        not in source.split("GeneratedPppsSubgroupForceStorage", maxsplit=1)[1]
    )
    assert "blockIdx.x) * 32U + subgroup" in source


def test_subgroup_force_lowering_uses_explicit_nonfinal_recovery_slots():
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
        build_fused_shell_plan(spec, integral=integral, schedule=schedule),
    )

    assert "GeneratedPspsSubgroupForceStorage" in source
    assert "double decay_gradients[4][3]" in source
    assert "geometry.decay_gradients[3][coordinate]" in source
    assert "gradient[1][coordinate] = -gradient[0][coordinate]" in source


@pytest.mark.parametrize("name", ("dpss", "ppps", "dsps"))
def test_one_warp_component_schedule_strides_larger_coulomb_table(name):
    """Do not retain an idle second warp after a short Coulomb setup."""

    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    integral = build_integral_ir(
        spec,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
    )
    component_schedules = [
        item
        for item in schedule_candidates(integral)
        if item.kind == ScheduleKind.COMPONENT_LANES
    ]
    assert [item.block_threads for item in component_schedules] == [64, 32]
    compact = component_schedules[1]
    source = emit_shell_class_fused_cuda(
        spec,
        build_fused_shell_plan(
            spec,
            consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
            schedule=compact,
        ),
    )
    class_name = name[0].upper() + name[1:]
    assert f"kGenerated{class_name}BlockThreads = 32U" in source
    assert f"state += kGenerated{class_name}BlockThreads" in source
    assert "shared.coulomb[state] = generated_" in source
    assert "candidate_density_coefficient" not in source
    assert "fabs(candidate_density_coefficient)" not in source
    assert "__syncthreads_or(density_coefficient != 0.0)" in source
    assert "double component_force[9]{};" in source
    assert "double component_force[12]{};" not in source
    assert "double warp_sums[kGenerated" in source
    assert "WarpCount][9];" in source
    assert "const double fourth_value =" in source
    assert "shared.task.atom[3]" in source


def test_ppps_scalar_thread_schedule_emits_component_scoped_dag():
    """Keep every scalar recurrence inside a bounded no-spill helper."""

    spec = FUSED_SHELL_SPEC_BY_NAME["ppps"]
    schedule = ScheduleIR(
        kind=ScheduleKind.THREAD_TASKS,
        block_threads=32,
        component_tile=spec.component_count,
        tasks_per_warp=32,
        shared_coulomb=False,
        minimum_blocks_per_sm=8,
    )
    source = emit_shell_class_fused_cuda(
        spec,
        build_fused_shell_plan(spec, schedule=schedule),
    )
    assert "storage.component_weights[0] = 0.0;" in source
    assert "storage.component_weights[26] = 0.0;" in source
    assert "double force_0 = storage.task_force[0];" in source
    assert "double force_8 = storage.task_force[8];" in source
    assert "component_weights[component]" not in source
    assert "primitive_gradient[" not in source
    assert "generated_ppps_scalar_thread_accumulate_components_0_3" in source
    assert "generated_ppps_scalar_thread_accumulate_components_24_27" in source
    assert (
        source.count(
            "__device__ __noinline__ void "
            "generated_ppps_scalar_thread_accumulate_components_"
        )
        == 9
    )
    assert "__launch_bounds__(32, 8)" in source
    assert source.count("force_0 += primitive_scale") == 27


def test_ppps_scalar_thread_lowering_uses_explicit_derivative_center_slots():
    """Route scalar-thread force atomics through non-final IR recovery."""

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
    )
    schedule = ScheduleIR(
        kind=ScheduleKind.THREAD_TASKS,
        block_threads=32,
        component_tile=spec.component_count,
        tasks_per_warp=32,
        shared_coulomb=False,
        minimum_blocks_per_sm=8,
    )
    source = emit_shell_class_fused_cuda(
        spec,
        build_fused_shell_plan(spec, integral=integral, schedule=schedule),
    )

    assert "double decay_gradients[4][3];" in source
    assert "storage.primitive.decay_gradients[3][2]" in source
    recovery_begin = source.index(
        "const double fourth_force",
        source.index("generated_ppps_scalar_thread_force_task"),
    )
    recovery = source[recovery_begin : recovery_begin + 600]
    assert "static_cast<std::size_t>(task.atom[1])" in recovery
    assert "static_cast<std::size_t>(task.atom[3])" not in recovery


def test_ppps_rys_program_is_a_compact_unique_state_recurrence():
    """Keep the independent backend at recurrence-state granularity."""

    program = build_ppps_rys_force_program()
    assert program.spec == FUSED_SHELL_SPEC_BY_NAME["ppps"]
    assert program.nroots == 3
    assert program.independent_derivative_centers == (0, 1, 2)
    assert program.recovered_derivative_centers == (3,)
    assert program.independent_force_centers == (0, 1, 2)
    assert program.component_order == program.spec.components
    assert len(program.axis_program.requested_states) == 20
    assert len(program.axis_program.instructions) == 23
    states = [instruction.state for instruction in program.axis_program.instructions]
    assert len(states) == len(set(states))
    emitted: set[RysState] = set()
    for instruction in program.axis_program.instructions:
        assert set(instruction.dependencies) <= emitted
        emitted.add(instruction.state)
    assert {instruction.kind for instruction in program.axis_program.instructions} == {
        RysRecurrenceKind.SEED,
        RysRecurrenceKind.TRR_BRA,
        RysRecurrenceKind.TRR_KET,
        RysRecurrenceKind.HRR_BRA,
    }
    minimal = build_rys_axis_program((RysState(1, 1, 1, 0),))
    assert minimal.instructions[-1].state == RysState(1, 1, 1, 0)

    schedule = ScheduleIR(
        kind=ScheduleKind.THREAD_TASKS,
        block_threads=32,
        component_tile=program.spec.component_count,
        tasks_per_warp=32,
        shared_coulomb=False,
        minimum_blocks_per_sm=12,
    )
    plan = build_fused_shell_plan(
        program.spec,
        schedule=schedule,
        recurrence="rys3",
    )
    assert plan.kernel.integral.recurrence == "rys3"
    with pytest.raises(ValueError, match="requires rys4"):
        build_fused_shell_plan(DPPP_SPEC, recurrence="rys3")


def test_dddd_rys_program_exposes_five_root_backend_requirements():
    """Quantify the high-order state surface without emitting scalar algebra."""

    program = build_rys_force_program(DDDD_SPEC)
    assert program.nroots == 5
    assert len(program.component_order) == 1296
    assert len(program.axis_program.requested_states) == 162
    assert len(program.axis_program.instructions) == 216


def test_dddp_rys5_recurrence_matches_every_symbolic_component():
    """Lock the first promoted five-root class against symbolic lowering."""

    spec = FUSED_SHELL_SPEC_BY_NAME["dddp"]
    values = factored_dppp_variables(sample_variables())
    for component in spec.components:
        actual = evaluate_rys_component(spec, component, values)
        expected = evaluate_fused_shell_observables(spec, component, values)
        assert actual.value == pytest.approx(expected.value, rel=8.0e-13, abs=8.0e-13)
        for center in range(4):
            for axis in range(3):
                assert actual.gradients[center][axis] == pytest.approx(
                    expected.gradients[center][axis],
                    rel=2.0e-12,
                    abs=2.0e-12,
                )


def test_dddd_rys5_recurrence_matches_representative_symbolic_components():
    """Cover every Cartesian axis pattern without a 1296-case duplicate gate."""

    spec = DDDD_SPEC
    values = factored_dppp_variables(sample_variables())
    for component_index in (0, 1, 17, 215, 647, 648, 1024, 1295):
        component = spec.components[component_index]
        actual = evaluate_rys_component(spec, component, values)
        expected = evaluate_fused_shell_observables(spec, component, values)
        assert actual.value == pytest.approx(expected.value, rel=8.0e-13, abs=8.0e-13)
        for center in range(4):
            for axis in range(3):
                assert actual.gradients[center][axis] == pytest.approx(
                    expected.gradients[center][axis],
                    rel=2.0e-12,
                    abs=2.0e-12,
                )


def test_dppp_rys_program_bounds_four_root_state_groups():
    """Expose the exact DPPP Rys4 surface before production integration."""

    program = build_rys_force_program(DPPP_SPEC)
    assert program.nroots == 4
    assert len(program.component_order) == 162
    assert len(program.axis_program.requested_states) == 56
    assert len(program.axis_program.instructions) == 67
    body = emit_rys_force_root_body_cuda(DPPP_SPEC, component_group=3)
    assert body.count("const double component_density_weight") == 162
    assert body.count("double rys_state_") == 1375
    assert "boys_" not in body
    assert "component_gradient" not in body


@pytest.mark.parametrize("argument", (0.0, 1.0e-10, 0.05, 1.0, 25.0, 80.0))
def test_gpu4pyscf_rys2_table_matches_moment_oracle(argument: float):
    """Verify the attributed low-order table used by four production shells."""

    roots, weights = rys2_table_roots_weights(argument)
    moments = rys_boys_values(argument, 4)
    for order, expected in enumerate(moments):
        assert sum(
            weight * root**order for root, weight in zip(roots, weights, strict=True)
        ) == pytest.approx(expected, rel=3.0e-11, abs=3.0e-14)


@pytest.mark.parametrize("argument", (0.0, 1.0e-10, 0.05, 1.0, 5.999))
def test_rys3_rule_reproduces_first_six_boys_moments(argument: float):
    """Treat the host eigensolve only as a high-accuracy Rys3 oracle."""

    roots, weights = rys3_roots_weights(argument)
    assert all(0.0 < root < 1.0 for root in roots)
    assert all(weight > 0.0 for weight in weights)
    moments = rys_boys_values(argument, 6)
    for order, expected in enumerate(moments):
        assert sum(
            weight * root**order for root, weight in zip(roots, weights, strict=True)
        ) == pytest.approx(expected, rel=3.0e-13, abs=3.0e-14)


@pytest.mark.parametrize("argument", (0.0, 1.0e-10, 0.05, 1.0, 5.999, 25.0, 50.0, 80.0))
def test_gpu4pyscf_rys3_table_matches_moment_oracle(argument: float):
    """Verify the attributed nroots=3 table before CUDA emission."""

    roots, weights = rys3_table_roots_weights(argument)
    moments = rys_boys_values(argument, 6)
    for order, expected in enumerate(moments):
        assert sum(
            weight * root**order for root, weight in zip(roots, weights, strict=True)
        ) == pytest.approx(expected, rel=3.0e-11, abs=3.0e-14)


@pytest.mark.parametrize("argument", (0.0, 1.0e-10, 0.05, 1.0, 5.999))
def test_rys4_rule_reproduces_first_eight_boys_moments(argument: float):
    """Treat the host eigensolve only as a high-accuracy Rys4 oracle."""

    roots, weights = rys4_roots_weights(argument)
    assert all(0.0 < root < 1.0 for root in roots)
    assert all(weight > 0.0 for weight in weights)
    moments = rys_boys_values(argument, 8)
    for order, expected in enumerate(moments):
        assert sum(
            weight * root**order for root, weight in zip(roots, weights, strict=True)
        ) == pytest.approx(expected, rel=8.0e-12, abs=5.0e-14)


@pytest.mark.parametrize("argument", (0.0, 1.0e-10, 0.05, 1.0, 5.999, 25.0, 55.0, 80.0))
def test_gpu4pyscf_rys4_table_matches_moment_oracle(argument: float):
    """Verify the attributed nroots=4 slice before CUDA integration."""

    roots, weights = rys4_table_roots_weights(argument)
    moments = rys_boys_values(argument, 8)
    for order, expected in enumerate(moments):
        assert sum(
            weight * root**order for root, weight in zip(roots, weights, strict=True)
        ) == pytest.approx(expected, rel=4.0e-11, abs=5.0e-13)


@pytest.mark.parametrize("argument", (0.0, 1.0e-10, 0.05, 1.0, 5.999))
def test_rys5_rule_reproduces_first_ten_boys_moments(argument: float):
    """Treat the host eigensolve only as a high-accuracy Rys5 oracle."""

    roots, weights = rys5_roots_weights(argument)
    assert all(0.0 < root < 1.0 for root in roots)
    assert all(weight > 0.0 for weight in weights)
    moments = rys_boys_values(argument, 10)
    for order, expected in enumerate(moments):
        assert sum(
            weight * root**order for root, weight in zip(roots, weights, strict=True)
        ) == pytest.approx(expected, rel=3.0e-11, abs=8.0e-13)


@pytest.mark.parametrize(
    "argument", (0.0, 1.0e-10, 0.05, 1.0, 5.999, 25.0, 60.0, 80.0)
)
def test_gpu4pyscf_rys5_table_matches_moment_oracle(argument: float):
    """Verify the attributed nroots=5 slice before CUDA integration."""

    roots, weights = rys5_table_roots_weights(argument)
    moments = rys_boys_values(argument, 10)
    for order, expected in enumerate(moments):
        assert sum(
            weight * root**order for root, weight in zip(roots, weights, strict=True)
        ) == pytest.approx(expected, rel=5.0e-10, abs=5.0e-12)


@pytest.mark.parametrize(
    "argument",
    (
        3.0e-7 - 1.0e-14,
        3.0e-7,
        3.0e-7 + 1.0e-14,
        2.5 - 1.0e-12,
        2.5,
        2.5 + 1.0e-12,
        55.0 - 1.0e-10,
        55.0,
        55.0 + 1.0e-10,
    ),
)
def test_gpu4pyscf_rys4_table_is_accurate_across_branch_boundaries(
    argument: float,
):
    """Cover the small-x, interpolation-interval, and large-x boundaries."""

    roots, weights = rys4_table_roots_weights(argument)
    moments = rys_boys_values(argument, 8)
    for order, expected in enumerate(moments):
        assert sum(
            weight * root**order for root, weight in zip(roots, weights, strict=True)
        ) == pytest.approx(expected, rel=4.0e-11, abs=5.0e-13)


def test_dppp_rys4_cuda_emits_only_the_fixed_root_slice():
    """Keep Rys4 tables compact enough for generated CUDA compilation."""

    roots = emit_rys4_roots_cuda()
    assert "Copyright 2021-2024 The PySCF Developers" in roots
    assert "generated_dppp_rys4_rw[4480]" in roots
    assert "generated_dppp_rys4_roots" in roots
    assert "series < 8U" in roots
    assert "argument > 55.0" in roots


def test_dddp_rys5_cuda_emits_only_the_fixed_root_slice():
    """Keep Rys5 tables compact enough for generated CUDA compilation."""

    roots = emit_rys5_roots_cuda()
    assert "Copyright 2021-2024 The PySCF Developers" in roots
    assert "generated_dddp_rys5_rw[5600]" in roots
    assert "generated_dddp_rys5_roots" in roots
    assert "series < 10U" in roots
    assert "argument > 60.0" in roots


def test_low_order_rys2_cuda_emits_only_the_fixed_root_slice():
    """Keep the shared two-root table compact and attributed."""

    roots = emit_rys2_roots_cuda()
    assert "Copyright 2021-2024 The PySCF Developers" in roots
    assert "generated_low_order_rys2_rw[2240]" in roots
    assert "series < 4U" in roots
    assert "argument > 45.0" in roots


def test_ppps_rys_cuda_emits_compact_state_program_and_attributed_table():
    """Prevent the direct recurrence from regressing into a scalar DAG."""

    roots = emit_rys3_roots_cuda()
    body = emit_ppps_rys3_root_body_cuda()
    assert "Copyright 2021-2024 The PySCF Developers" in roots
    assert "generated_ppps_rys3_rw[3360]" in roots
    assert "generated_ppps_rys3_roots" in roots
    assert 80 <= body.count("double rys_state_") <= 93
    assert body.count("rys_state_") > 69
    assert body.count("const double component_density_weight") == 27
    assert body.count("force_") == 243
    assert "boys_" not in body
    assert "component_gradient" not in body

    spec = FUSED_SHELL_SPEC_BY_NAME["ppps"]
    schedule = ScheduleIR(
        kind=ScheduleKind.THREAD_TASKS,
        block_threads=32,
        component_tile=spec.component_count,
        tasks_per_warp=32,
        shared_coulomb=False,
        minimum_blocks_per_sm=12,
    )
    plan = build_fused_shell_plan(
        spec,
        schedule=schedule,
        recurrence="rys3",
    )
    source = emit_shell_class_fused_cuda(spec, plan)
    assert "generated_ppps_rys3_force_task" in source
    assert "generated_ppps_rys3_roots" in source
    assert "component_weights[kGeneratedPppsComponentCount][32]" in source
    assert "generated_ppps_scalar_thread" not in source

    dpss_spec = FUSED_SHELL_SPEC_BY_NAME["dpss"]
    dpss_schedule = ScheduleIR(
        kind=ScheduleKind.THREAD_TASKS,
        block_threads=32,
        component_tile=dpss_spec.component_count,
        tasks_per_warp=32,
        shared_coulomb=False,
        minimum_blocks_per_sm=8,
    )
    dpss_source = emit_shell_class_fused_cuda(
        dpss_spec,
        build_fused_shell_plan(
            dpss_spec,
            schedule=dpss_schedule,
            recurrence="rys3",
        ),
    )
    assert "generated_dpss_rys3_force_task" in dpss_source
    assert "generated_dpss_rys3_roots" in dpss_source
    assert "generated_ppps_rys3_roots" not in dpss_source


@pytest.mark.parametrize("name", ("psss", "psps", "ppss", "dsss"))
def test_low_order_shells_share_scalar_rys2_force_backend(name: str):
    """Emit each two-root shell with one complete quartet per CUDA lane."""

    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    schedule = ScheduleIR(
        kind=ScheduleKind.THREAD_TASKS,
        block_threads=32,
        component_tile=spec.component_count,
        tasks_per_warp=32,
        shared_coulomb=False,
        minimum_blocks_per_sm=8,
    )
    plan = build_fused_shell_plan(spec, schedule=schedule, recurrence="rys2")
    source = emit_shell_class_fused_cuda(spec, plan)
    assert f"generated_{name}_rys2_force_task" in source
    assert f"generated_{name}_rys2_roots" in source
    assert f"component_weights[kGenerated{name.title()}ComponentCount][32]" in source
    assert "root_index < 2U" in source


def test_ppps_rys_recurrence_matches_every_symbolic_component():
    """Lock component order, force signs, and translation recovery."""

    spec = FUSED_SHELL_SPEC_BY_NAME["ppps"]
    values = factored_dppp_variables(sample_variables())
    for component in spec.components:
        actual = evaluate_ppps_rys_component(component, values)
        expected = evaluate_fused_shell_observables(spec, component, values)
        assert actual.value == pytest.approx(expected.value, rel=3.0e-13, abs=3.0e-13)
        for center in range(4):
            for axis in range(3):
                assert actual.gradients[center][axis] == pytest.approx(
                    expected.gradients[center][axis],
                    rel=8.0e-13,
                    abs=8.0e-13,
                )


@pytest.mark.parametrize("name", ("dppp", "dpdp", "dpds", "ddpp", "ddps", "ddds"))
def test_cooperative_rys4_recurrence_matches_every_symbolic_component(
    name: str,
):
    """Lock each promoted four-root recurrence against symbolic lowering."""

    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    values = factored_dppp_variables(sample_variables())
    for component in spec.components:
        actual = evaluate_rys_component(spec, component, values)
        expected = evaluate_fused_shell_observables(spec, component, values)
        assert actual.value == pytest.approx(expected.value, rel=6.0e-13, abs=6.0e-13)
        for center in range(4):
            for axis in range(3):
                assert actual.gradients[center][axis] == pytest.approx(
                    expected.gradients[center][axis],
                    rel=1.5e-12,
                    abs=1.5e-12,
                )


@pytest.mark.parametrize(
    "name", ("dpps", "dpss", "dsps", "dspp", "dsds", "ddss", "pppp")
)
def test_cooperative_rys3_recurrence_matches_every_symbolic_component(
    name: str,
):
    """Lock each promoted three-root recurrence against symbolic lowering."""

    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    values = factored_dppp_variables(sample_variables())
    for component in spec.components:
        actual = evaluate_rys_component(spec, component, values)
        expected = evaluate_fused_shell_observables(spec, component, values)
        assert actual.value == pytest.approx(expected.value, rel=8.0e-13, abs=8.0e-13)
        for center in range(4):
            for axis in range(3):
                assert actual.gradients[center][axis] == pytest.approx(
                    expected.gradients[center][axis],
                    rel=2.0e-12,
                    abs=2.0e-12,
                )


def test_dppp_cooperative_rys4_uses_uniform_runtime_indexed_axis_recurrence():
    """Prevent regression to a divergent 162-way component dispatcher."""

    spec = FUSED_SHELL_SPEC_BY_NAME["dppp"]
    schedule = ScheduleIR(
        kind=ScheduleKind.COMPONENT_LANES,
        block_threads=192,
        component_tile=spec.component_count,
        tasks_per_warp=1,
        shared_coulomb=True,
        pair_orientation=PairOrientation.SWAPPED,
        pair_storage=PairStorage.MATERIALIZED,
        unroll_pair_terms=True,
        minimum_blocks_per_sm=2,
    )
    plan = build_fused_shell_plan(
        spec,
        schedule=schedule,
        recurrence="rys4",
    )
    source = emit_shell_class_fused_cuda(spec, plan)
    assert "generated_dppp_rys4_component_lane_task" in source
    assert "volatile double trr[5][4]" in source
    assert "generated_dppp_rys4_axis" in source
    assert "switch (component)" not in source
    assert "generated_dppp_rys4_fill_weights" not in source
    assert "component_weights[kGeneratedDpppComponentCount][32]" not in source
    assert "GeneratedDpppPrimitiveGeometry primitive" not in source
    assert "generated_dppp_rys4_roots" in source


def test_dppp_rys4_uniform_warps_advance_32_quartets_per_block():
    """Keep the 2111-style task and component coordinates explicit."""

    schedule = ScheduleIR(
        kind=ScheduleKind.SUBGROUP_TASKS,
        block_threads=256,
        component_tile=DPPP_SPEC.component_count,
        tasks_per_warp=4,
        shared_coulomb=True,
        pair_orientation=PairOrientation.SWAPPED,
        pair_storage=PairStorage.MATERIALIZED,
        unroll_pair_terms=True,
        minimum_blocks_per_sm=1,
    )
    plan = build_fused_shell_plan(
        DPPP_SPEC,
        schedule=schedule,
        recurrence="rys4",
    )
    source = emit_shell_class_fused_cuda(DPPP_SPEC, plan)
    assert schedule.tasks_per_block == 32
    assert schedule.subgroup_lanes == 8
    assert "kGeneratedDpppRys4TaskCount = 32U" in source
    assert "kGeneratedDpppRys4ComponentLanes = 8U" in source
    assert "const unsigned sq = thread & 31U" in source
    assert "const unsigned component_lane = thread >> 5U" in source
    assert "atomicAdd(task_head, kGeneratedDpppRys4TaskCount)" in source
    assert "generated_dppp_rys4_uniform_warp_roots" in source
    assert "switch (component_lane)" in source
    assert "generated_dppp_subgroup_force_task" not in source
    assert "generated_dppp_rys4_component_lane_task" not in source

    mixed_plan = build_fused_shell_plan(
        DPPP_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=schedule,
        recurrence="rys4",
    )
    mixed_source = emit_shell_class_fused_cuda(DPPP_SPEC, mixed_plan)
    assert "kGeneratedDpppBlockThreads = 256U" in mixed_source
    assert "kGeneratedDpppFockBlockThreads = 192U" in mixed_source
    assert "GeneratedDpppSubgroupFockStorage" not in mixed_source


@pytest.mark.parametrize("name", ("dddp", "dddd"))
def test_high_order_rys5_uniform_warps_advance_32_quartets_per_block(name: str):
    """Keep each five-root task/component mapping explicit."""

    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    schedule = ScheduleIR(
        kind=ScheduleKind.SUBGROUP_TASKS,
        block_threads=256,
        component_tile=spec.component_count,
        tasks_per_warp=4,
        shared_coulomb=True,
        pair_orientation=PairOrientation.SWAPPED,
        pair_storage=PairStorage.MATERIALIZED,
        unroll_pair_terms=True,
        minimum_blocks_per_sm=1,
    )
    plan = build_fused_shell_plan(
        spec,
        consumers=(KernelConsumer.FORCE,),
        schedule=schedule,
        recurrence="rys5",
    )
    source = emit_shell_class_fused_cuda(spec, plan)
    class_name = name[0].upper() + name[1:]
    assert f"kGenerated{class_name}Rys5TaskCount = 32U" in source
    assert f"kGenerated{class_name}Rys5ComponentLanes = 8U" in source
    assert f"generated_{name}_rys5_uniform_warp_roots" in source
    assert "root_index < 5U" in source
    assert f"generated_{name}_subgroup_force_task" not in source


@pytest.mark.parametrize(
    ("name", "fock_block_threads"),
    (("dpps", 64), ("dspp", 64), ("pppp", 96)),
)
def test_rys3_uniform_warps_split_components_without_scalar_spills(
    name: str, fock_block_threads: int
):
    """Reuse the 32-task geometry when one Rys3 thread owns too much state."""

    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    schedule = ScheduleIR(
        kind=ScheduleKind.SUBGROUP_TASKS,
        block_threads=256,
        component_tile=spec.component_count,
        tasks_per_warp=4,
        shared_coulomb=True,
        minimum_blocks_per_sm=1,
    )
    plan = build_fused_shell_plan(
        spec,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=schedule,
        recurrence="rys3",
    )
    source = emit_shell_class_fused_cuda(spec, plan)
    class_name = name[0].upper() + name[1:]
    assert f"kGenerated{class_name}Rys3TaskCount = 32U" in source
    assert f"kGenerated{class_name}Rys3ComponentLanes = 8U" in source
    assert f"generated_{name}_rys3_uniform_warp_roots" in source
    assert "root_index < 3U" in source
    assert f"kGenerated{class_name}FockBlockThreads = {fock_block_threads}U" in source
    assert f"generated_{name}_rys3_force_task" not in source
    assert f"generated_{name}_subgroup_force_task" not in source


@pytest.mark.parametrize(
    ("name", "block_threads", "trr_shape"),
    (
        ("dpps", 64, "volatile double trr[5][3]"),
        ("dsps", 32, "volatile double trr[4][3]"),
        ("dsds", 64, "volatile double trr[4][4]"),
        ("ddss", 64, "volatile double trr[6][2]"),
        ("pppp", 96, "volatile double trr[4][4]"),
    ),
)
def test_cooperative_rys3_hot_classes_use_uniform_component_lanes(
    name: str,
    block_threads: int,
    trr_shape: str,
):
    """Promote measured Rys3 hotspots without changing their direct Fock."""

    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    schedule = ScheduleIR(
        kind=ScheduleKind.COMPONENT_LANES,
        block_threads=block_threads,
        component_tile=spec.component_count,
        tasks_per_warp=1,
        shared_coulomb=True,
    )
    plan = build_fused_shell_plan(
        spec,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=schedule,
        recurrence="rys3",
    )
    source = emit_shell_class_fused_cuda(spec, plan)
    assert f"generated_{name}_rys3_component_lane_task" in source
    assert f"generated_{name}_rys3_roots" in source
    assert trr_shape in source
    assert "switch (component)" not in source
    assert f"generated_{name}_shell_class_fock_rhf_kernel" in source


@pytest.mark.parametrize(
    "spec",
    (
        PSPS_SPEC,
        PPSS_SPEC,
        FUSED_SHELL_SPEC_BY_NAME["dsss"],
    ),
)
def test_packed_schedule_models_low_order_fock_workers(spec):
    """Keep the accepted Fock topology while force moves to scalar Rys2."""

    selection = next(
        selection
        for selection in load_production_kernel_selections(
            REPOSITORY_ROOT
            / "tools"
            / "vibeqc_codegen"
            / "production_shell_classes.json"
        )
        if selection.spec == spec
    )
    schedule = selection.fock_schedule
    assert selection.recurrence == "rys2"
    assert selection.schedule.kind == ScheduleKind.THREAD_TASKS
    assert schedule is not None
    assert schedule.kind == ScheduleKind.PACKED_TASKS
    assert schedule.block_threads == 32
    assert schedule.tasks_per_warp == 32
    assert not schedule.shared_coulomb


def test_zero_order_pairs_lower_through_shell_task_schedule():
    """Generate low-order force/Fock code without handwritten psss algebra."""

    assert len(FUSED_SHELL_SPECS) == 55
    assert PSSS_SPEC.angular == (1, 0, 0, 0)
    assert SSSS_SPEC.angular == (0, 0, 0, 0)
    integral = build_integral_ir(
        PSSS_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
    )
    shell_schedule = next(
        item
        for item in schedule_candidates(integral)
        if item.kind == ScheduleKind.SHELL_TASK
    )
    plan = build_fused_shell_plan(
        PSSS_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=shell_schedule,
    )
    source = emit_shell_class_fused_cuda(PSSS_SPEC, plan)
    assert "kGeneratedPsssComponentCount = 3U" in source
    assert "generated_psss_pair_term<0U>" in source
    assert "const unsigned second_axes[1]" in source
    assert "generated_psss_component_gradient<false>" in source
    assert "generated_psss_component_value<false>" in source

    trials = supported_schedule_trials(PSSS_SPEC)
    assert any(trial.schedule.kind == ScheduleKind.PACKED_TASKS for trial in trials)
    assert any(trial.schedule.kind == ScheduleKind.SHELL_TASK for trial in trials)
    assert (
        sum(trial.schedule.kind == ScheduleKind.COMPONENT_LANES for trial in trials)
        == 8
    )

    packed_schedule = next(
        item
        for item in schedule_candidates(integral)
        if item.kind == ScheduleKind.PACKED_TASKS
    )
    packed_plan = build_fused_shell_plan(
        PSSS_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=packed_schedule,
    )
    packed = emit_shell_class_fused_cuda(PSSS_SPEC, packed_plan)
    assert "generated_psss_packed_force_lane" in packed
    assert "generated_psss_packed_fock_lane" in packed
    assert "generated_psss_weighted_component_gradient" in packed
    assert "candidate_density_coefficient" not in packed
    assert "fabs(candidate_density_coefficient)" not in packed
    assert "if (!any_component) return;" in packed
    assert "generated_psss_component_gradient<false>" not in packed
    assert "atomicAdd(task_head, 32U)" in packed
    assert "blockIdx.x) * 32U + threadIdx.x" in packed
    assert "generated_psss_shell_class_force_task" not in packed

    benchmark = emit_shell_class_benchmark_cuda(
        PSSS_SPEC,
        task_count=33,
        primitive_count=1,
        warmups=0,
        iterations=1,
        samples=1,
        schedule=packed_schedule,
    )
    assert "<<<(kTaskCount + 31U) / 32U," in benchmark


@pytest.mark.parametrize(
    ("spec", "component"),
    (
        (PSSS_SPEC, ("y", "", "", "")),
        (SSSS_SPEC, ("", "", "", "")),
    ),
)
def test_zero_order_pair_recurrence_matches_symbolic_oracle(spec, component):
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


def test_weighted_psss_graph_cse_matches_component_oracle():
    """Combine density-weighted components before CUDA primitive traversal."""

    weights = (0.7, -0.2, 1.1)
    variables = factored_dppp_variables(sample_variables())
    variables.update(
        {
            f"component_weight_{component}": weight
            for component, weight in enumerate(weights)
        }
    )
    weighted = build_weighted_shell_contraction_kernel(PSSS_SPEC)
    individual_node_count = sum(
        len(build_shell_class_contraction_kernel(PSSS_SPEC, component).graph.nodes)
        for component in PSSS_SPEC.components
    )
    assert len(weighted.graph.nodes) < individual_node_count
    expected_value = sum(
        weight * evaluate_fused_shell_value(PSSS_SPEC, component, variables)
        for weight, component in zip(weights, PSSS_SPEC.components, strict=True)
    )
    assert weighted.graph.evaluate(weighted.value, variables) == pytest.approx(
        expected_value,
        rel=2.0e-12,
        abs=2.0e-12,
    )
    for center in range(4):
        for axis in range(3):
            expected = sum(
                weight
                * evaluate_fused_shell_component(
                    PSSS_SPEC,
                    component,
                    variables,
                )[center][axis]
                for weight, component in zip(
                    weights,
                    PSSS_SPEC.components,
                    strict=True,
                )
            )
            assert weighted.graph.evaluate(
                weighted.gradients[center][axis],
                variables,
            ) == pytest.approx(expected, rel=3.0e-12, abs=3.0e-12)


def assert_rtx5090_resources(
    ptxas_output: str,
    limits: dict[str, tuple[int, int, int]],
) -> None:
    """Reject CUDA 12.9 resource regressions before production integration."""

    for function, (register_limit, stack_limit, shared_limit) in limits.items():
        match = re.search(
            rf"Function properties for {function}\n"
            r"\s+(\d+) bytes stack frame, (\d+) bytes spill stores, "
            r"(\d+) bytes spill loads\n"
            r"ptxas info\s+: Used (\d+) registers([^\n]*)",
            ptxas_output,
        )
        assert match is not None, f"missing ptxas resources for {function}"
        stack, spill_stores, spill_loads, registers = map(int, match.groups()[:4])
        shared_match = re.search(r"(\d+) bytes smem", match.group(5))
        shared = int(shared_match.group(1)) if shared_match is not None else 0
        assert registers <= register_limit
        assert stack <= stack_limit
        assert spill_stores == 0
        assert spill_loads == 0
        assert shared <= shared_limit


def test_shell_spec_generates_cca_components_and_compile_time_bounds():
    """Derive shell schedules without handwritten component tables."""

    assert cartesian_components(0) == ("",)
    assert cartesian_components(1) == AXES
    assert cartesian_components(2) == ("xx", "xy", "xz", "yy", "yz", "zz")
    assert cartesian_components(3) == (
        "xxx",
        "xxy",
        "xxz",
        "xyy",
        "xyz",
        "xzz",
        "yyy",
        "yyz",
        "yzz",
        "zzz",
    )
    assert DPPP_SPEC.pair_orders == (3, 2)
    assert DPDS_SPEC.pair_orders == (3, 2)
    assert DDPS_SPEC.pair_orders == (4, 1)
    assert DPPP_SPEC.maximum_force_coulomb_order == 6
    assert DPPP_SPEC.component_count == 162
    assert DPPP_SPEC.component_strides == (27, 9, 3, 1)


def test_large_dddd_class_defaults_to_tiled_lowering():
    """Keep AO products beyond CUDA's block limit in the generated catalog."""

    assert DDDD_SPEC.component_count == 1296
    assert DDDD_SPEC.pair_orders == (4, 4)
    integral = build_integral_ir(DDDD_SPEC)
    candidates = schedule_candidates(integral)
    assert [item.kind for item in candidates] == [
        ScheduleKind.TILED_COMPONENTS,
        ScheduleKind.TILED_COMPONENTS,
        ScheduleKind.TILED_COMPONENTS,
    ]
    assert [item.component_tile for item in candidates] == [64, 128, 256]
    plan = build_fused_shell_plan(DDDD_SPEC)
    assert plan.schedule.kind == ScheduleKind.TILED_COMPONENTS
    assert plan.block_threads == 64
    assert len(plan.coulomb_states) == 220
    source = emit_shell_class_fused_cuda(DDDD_SPEC, plan)
    assert "kGeneratedDdddComponentCount = 1296U" in source
    assert "__constant__ short generated_dddd_coulomb_indices[1000]" in source
    assert "state >> 4U" in source
    assert "state >> 8U" in source
    assert "component_tile_begin += 64U" in source

    trials = supported_schedule_trials(DDDD_SPEC)
    assert len(trials) == 24
    assert len({trial.schedule_id for trial in trials}) == len(trials)
    assert {
        (
            trial.schedule.component_tile,
            trial.schedule.pair_storage,
            trial.schedule.pair_orientation,
            trial.schedule.unroll_pair_terms,
        )
        for trial in trials
    } == {
        (tile, storage, orientation, unrolled)
        for tile in (64, 128, 256)
        for storage in PairStorage
        for orientation in PairOrientation
        for unrolled in (True, False)
    }


def test_large_pair_recompute_schedule_avoids_materialized_term_arrays():
    """Trade repeated pair algebra for bounded stack use in large tiled shells."""

    base = build_fused_shell_plan(
        DDDD_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
    ).schedule
    schedule = replace(
        base,
        component_tile=256,
        block_threads=256,
        pair_storage=PairStorage.RECOMPUTED,
        unroll_pair_terms=False,
    )
    plan = build_fused_shell_plan(
        DDDD_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=schedule,
    )
    source = emit_shell_class_fused_cuda(DDDD_SPEC, plan)
    assert "GeneratedDdddPairTerm first_terms" not in source
    assert "GeneratedDdddPairTerm second_terms" not in source
    assert "GeneratedDdddValueTerm first_terms" not in source
    assert "GeneratedDdddValueTerm second_terms" not in source
    assert "#pragma unroll 1" in source


@pytest.mark.parametrize(
    "component",
    (
        ("xx", "xx", "xx", "xx"),
        ("xy", "xz", "yz", "zz"),
        ("zz", "zz", "zz", "zz"),
    ),
)
def test_dddd_tiled_recurrence_matches_symbolic_oracle(component):
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
def test_f_shell_recurrence_matches_symbolic_oracle(spec, component):
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


def test_f_shell_cuda_lowering_emits_axes_triple_matchings_and_tiles():
    """Cover pair order six and a component product above the block limit."""

    ffps_source = emit_shell_class_fused_cuda(FFPS_SPEC)
    assert "generated_ffps_f_axes[10][3]" in ffps_source
    assert "if constexpr (PairOrder >= 6U)" in ffps_source
    assert "first_removed | second_removed | third_removed, 3U" in ffps_source
    assert "__constant__ short generated_ffps_coulomb_indices[729]" in ffps_source

    fddd_plan = build_fused_shell_plan(FDDD_SPEC)
    assert fddd_plan.schedule.kind == ScheduleKind.TILED_COMPONENTS
    assert fddd_plan.block_threads == 64
    assert FDDD_SPEC.component_count == 2160
    fddd_source = emit_shell_class_fused_cuda(FDDD_SPEC, fddd_plan)
    assert "generated_fddd_f_axes[10][3]" in fddd_source
    assert "component_tile_begin += 64U" in fddd_source


def test_shell_spec_component_schedule_round_trips_without_manual_decoding():
    for index, component in enumerate(DPPP_SPEC.components):
        assert DPPP_SPEC.component_index(component) == index
        assert DPPP_SPEC.component_from_index(index) == component
    assert DPPP_SPEC.component_quantums(("xy", "z", "x", "y")) == (
        (0, 0),
        (0, 1),
        (1, 2),
        (2, 0),
        (3, 1),
    )


def test_shell_spec_rejects_invalid_metadata_and_components():
    with pytest.raises(ValueError):
        ShellClassSpec("bad", (2, 1, 1))
    with pytest.raises(ValueError):
        ShellClassSpec("Bad", (2, 1, 1, 1))
    with pytest.raises(ValueError):
        DPPP_SPEC.validate_component(("xx", "x", "y", "xx"))
    with pytest.raises(IndexError):
        DPPP_SPEC.component_from_index(DPPP_SPEC.component_count)


@pytest.mark.parametrize(
    ("spec", "component"),
    (
        (DPDS_SPEC, ("xy", "z", "xz", "")),
        (DDPS_SPEC, ("xy", "xz", "z", "")),
    ),
)
def test_generic_shell_ad_matches_factored_lowering(spec, component):
    """Exercise pair orders 3+2 and 4+1 without handwritten builders."""

    full = build_shell_class_component_kernel(spec, component)
    factored = build_shell_class_contraction_kernel(spec, component)
    full_values = sample_variables()
    argument = full.graph.evaluate(full.boys_argument, full_values)
    for order, value in enumerate(
        boys_values(argument, spec.maximum_force_coulomb_order + 1)
    ):
        full_values[f"boys_{order}"] = value
    factored_values = factored_dppp_variables(full_values)

    full_value = full.graph.evaluate(full.value, full_values)
    factored_value = factored_values["prefactor"] * factored.graph.evaluate(
        factored.value, factored_values
    )
    assert factored_value == pytest.approx(full_value, rel=5.0e-13, abs=5.0e-13)
    for center in range(4):
        for axis in range(3):
            actual = factored.graph.evaluate(
                factored.gradients[center][axis], factored_values
            )
            expected = full.graph.evaluate(full.gradients[center][axis], full_values)
            assert actual == pytest.approx(expected, rel=3.0e-11, abs=3.0e-11)


@pytest.mark.parametrize("spec", (DPDS_SPEC, DDPS_SPEC))
def test_generic_fused_schedule_preserves_every_component_gradient(spec):
    """Audit generated lane schedules, including ddps double Wick matching."""

    plan = build_fused_shell_plan(spec)
    assert plan.components == spec.components
    assert plan.block_threads == 128
    assert plan.warp_count == 4
    assert len(plan.coulomb_states) == 84
    assert len(plan.coulomb_indices) == 7**3

    values = factored_dppp_variables(sample_variables())
    for component in plan.components:
        direct = build_shell_class_contraction_kernel(spec, component)
        fused = evaluate_fused_shell_component(spec, component, values)
        assert evaluate_fused_shell_value(spec, component, values) == pytest.approx(
            values["prefactor"] * direct.graph.evaluate(direct.value, values),
            rel=8.0e-12,
            abs=8.0e-12,
        )
        for center in range(4):
            for axis in range(3):
                expected = direct.graph.evaluate(direct.gradients[center][axis], values)
                assert fused[center][axis] == pytest.approx(
                    expected, rel=8.0e-12, abs=8.0e-12
                )


def boys_values(argument: float, count: int = 3) -> list[float]:
    """Reference Boys sequence sufficient for the psss pilot."""

    if argument < 1.0e-8:
        return [
            sum(
                (-argument) ** term
                / (math.factorial(term) * (2 * order + 2 * term + 1))
                for term in range(18)
            )
            for order in range(count)
        ]
    root = math.sqrt(argument)
    values = [0.5 * math.sqrt(math.pi / argument) * math.erf(root)]
    exponential = math.exp(-argument)
    for order in range(count - 1):
        values.append(((2 * order + 1) * values[-1] - exponential) / (2 * argument))
    return values


@pytest.mark.parametrize("argument", (0.0, 1.0e-10, 0.05, 1.0, 5.999))
@pytest.mark.parametrize("count", (1, 3, 7, 13))
def test_highest_order_boys_series_supports_downward_recurrence(
    argument: float, count: int
):
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


def sample_variables() -> dict[str, float]:
    values = {
        "alpha": 1.3,
        "beta": 0.7,
        "gamma": 0.9,
        "delta": 0.5,
        "kPi": math.pi,
    }
    positions = {
        "first": (0.1, -0.3, 0.2),
        "second": (-0.4, 0.2, 0.5),
        "third": (0.6, -0.1, -0.2),
        "fourth": (-0.2, 0.4, -0.6),
    }
    for center, position in positions.items():
        for axis, coordinate in zip(AXES, position, strict=True):
            values[f"{center}_{axis}"] = coordinate
    return values


def factored_dppp_variables(values: dict[str, float]) -> dict[str, float]:
    """Construct the common primitive geometry consumed by factored lowering."""

    alpha = values["alpha"]
    beta = values["beta"]
    gamma = values["gamma"]
    delta = values["delta"]
    p = alpha + beta
    q = gamma + delta
    mu = alpha * beta / p
    nu = gamma * delta / q
    result = {
        "inverse_two_p": 0.5 / p,
        "inverse_two_q": 0.5 / q,
        "rho": p * q / (p + q),
        "first_product_scale": alpha / p,
        "second_product_scale": beta / p,
        "third_product_scale": gamma / q,
        "fourth_product_scale": delta / q,
    }
    product_p = {}
    product_q = {}
    pair_distance_squared = 0.0
    for axis in AXES:
        first = values[f"first_{axis}"]
        second = values[f"second_{axis}"]
        third = values[f"third_{axis}"]
        fourth = values[f"fourth_{axis}"]
        product_p[axis] = (alpha * first + beta * second) / p
        product_q[axis] = (gamma * third + delta * fourth) / q
        result[f"pa_{axis}"] = product_p[axis] - first
        result[f"pb_{axis}"] = product_p[axis] - second
        result[f"qc_{axis}"] = product_q[axis] - third
        result[f"qd_{axis}"] = product_q[axis] - fourth
        result[f"difference_{axis}"] = product_p[axis] - product_q[axis]
        first_difference = first - second
        second_difference = third - fourth
        pair_distance_squared += (
            -mu * first_difference * first_difference
            - nu * second_difference * second_difference
        )
        result[f"decay_first_{axis}"] = -2.0 * mu * first_difference
        result[f"decay_second_{axis}"] = 2.0 * mu * first_difference
        result[f"decay_third_{axis}"] = -2.0 * nu * second_difference
        result[f"decay_fourth_{axis}"] = 2.0 * nu * second_difference
    result["prefactor"] = (
        2.0
        * math.pi**2.5
        / (p * q * math.sqrt(p + q))
        * math.exp(pair_distance_squared)
    )
    argument = result["rho"] * sum(result[f"difference_{axis}"] ** 2 for axis in AXES)
    # Order-eight shell quartets such as DDDD require the ninth Boys moment
    # for their first-derivative oracle.
    for order, value in enumerate(boys_values(argument, 10)):
        result[f"boys_{order}"] = value
    return result


def evaluate_value(kernel, values: dict[str, float], boys_count: int = 3) -> float:
    argument = kernel.graph.evaluate(kernel.boys_argument, values)
    for order, value in enumerate(boys_values(argument, boys_count)):
        values[f"boys_{order}"] = value
    return kernel.graph.evaluate(kernel.value, values)


@pytest.mark.parametrize("p_axis", AXES)
def test_psss_symbolic_gradients_match_finite_difference(p_axis: str):
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


def test_psss_fourth_center_uses_exact_translation_recovery():
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


def test_psss_oracle_uses_explicit_nonfinal_recovery_centers():
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
):
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


def test_dppp_translation_and_ket_pair_permutation_invariants():
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


@pytest.mark.parametrize(
    ("d_component", "p_components"),
    (("xx", "xxx"), ("xy", "xyz"), ("zz", "zyx")),
)
def test_factored_dppp_lowering_matches_full_symbolic_kernel(
    d_component: str, p_components: str
):
    full = build_dppp_component_kernel(d_component, tuple(p_components))
    factored = build_dppp_contraction_kernel(d_component, tuple(p_components))
    full_values = sample_variables()
    argument = full.graph.evaluate(full.boys_argument, full_values)
    for order, value in enumerate(boys_values(argument, 7)):
        full_values[f"boys_{order}"] = value
    factored_values = factored_dppp_variables(full_values)

    full_value = full.graph.evaluate(full.value, full_values)
    factored_value = factored_values["prefactor"] * factored.graph.evaluate(
        factored.value, factored_values
    )
    assert factored_value == pytest.approx(full_value, rel=3.0e-13, abs=3.0e-13)
    for center in range(4):
        for axis in range(3):
            assert factored.graph.evaluate(
                factored.gradients[center][axis], factored_values
            ) == pytest.approx(
                full.graph.evaluate(full.gradients[center][axis], full_values),
                rel=2.0e-11,
                abs=2.0e-11,
            )


def test_dppp_fused_plan_covers_components_and_shared_coulomb_states():
    plan = build_dppp_fused_plan()
    components = dppp_components()
    assert plan.components == components
    assert len(components) == 162
    assert len(plan.coulomb_states) == 84
    assert len(plan.coulomb_indices) == 7**3
    assert plan.block_threads == 192
    assert plan.warp_count == 6
    for index, (x_order, y_order, z_order) in enumerate(plan.coulomb_states):
        dense_index = (x_order * 7 + y_order) * 7 + z_order
        assert plan.coulomb_indices[dense_index] == index
        assert x_order + y_order + z_order <= 6


def test_dppp_fused_schedule_preserves_all_component_gradients():
    values = factored_dppp_variables(sample_variables())
    for component in dppp_components():
        direct = build_dppp_contraction_kernel(component[0], component[1:])
        fused = evaluate_dppp_fused_component(component, values)
        for center in range(4):
            for axis in range(3):
                expected = direct.graph.evaluate(direct.gradients[center][axis], values)
                actual = fused[center][axis]
                assert actual == pytest.approx(expected, rel=4.0e-12, abs=4.0e-12)


def test_dppp_fused_cuda_emits_one_shared_shell_class_schedule():
    source = emit_dppp_fused_cuda()
    assert "kGeneratedDpppComponentCount = 162U" in source
    assert "kGeneratedDpppCoulombStateCount = 84U" in source
    assert "kGeneratedDpppBlockThreads = 192U" in source
    assert "__shared__ Shared shared" in source
    assert "generated_dppp_density_coefficient" in source
    assert "generated_dppp_component_gradient" in source
    assert "generated_dppp_shell_class_force_rhf_kernel" in source
    assert "generated_dppp_shell_class_force_uhf_kernel" in source
    assert "generated_dppp_shell_class_force_rhf_persistent_kernel" in source
    assert "generated_dppp_shell_class_force_uhf_persistent_kernel" in source
    assert "GeneratedDpppPrimitivePairData" in source
    assert "primitive_pair_offsets" in source
    assert "reversed_shell_pair_mask" in source
    assert "primitive_exponents" not in source
    assert "retained_by_schwarz" in source
    assert source.count("boys_values<6>") == 1
    assert "__noinline__" not in source
    assert "generated_dppp_orbit_" not in source
    density_helper = source[
        source.index("double generated_dppp_density_coefficient(") : source.index(
            "/** Combine two reusable shell-pair records"
        )
    ]
    assert "orbit_scale" in density_helper
    assert "4.0 * density[offset + ij] * density[offset + kl]" in density_helper
    assert "for (unsigned permutation" not in density_helper


@pytest.mark.parametrize("unrestricted", (False, True))
def test_closed_density_orbit_matches_unique_permutations(unrestricted: bool):
    """Prove the closed force coefficient for every AO equality pattern."""

    order = 4
    alpha = [
        [
            float((min(row, column) + 1) * 7 + max(row, column))
            for column in range(order)
        ]
        for row in range(order)
    ]
    beta = [
        [
            float((min(row, column) + 2) * 11 - max(row, column))
            for column in range(order)
        ]
        for row in range(order)
    ]

    for i, j, k, l in itertools.product(range(order), repeat=4):
        permutations = (
            (i, j, k, l),
            (j, i, k, l),
            (i, j, l, k),
            (j, i, l, k),
            (k, l, i, j),
            (l, k, i, j),
            (k, l, j, i),
            (l, k, j, i),
        )
        old = 0.0
        seen: set[tuple[int, int, int, int]] = set()
        for a, b, c, d in permutations:
            if (a, b, c, d) in seen:
                continue
            seen.add((a, b, c, d))
            if unrestricted:
                old += 0.5 * (alpha[a][b] + beta[a][b]) * (alpha[c][d] + beta[c][d])
                old -= 0.5 * (alpha[a][c] * alpha[b][d] + beta[a][c] * beta[b][d])
            else:
                old += (
                    0.5 * alpha[a][b] * alpha[c][d] - 0.25 * alpha[a][c] * alpha[b][d]
                )

        orbit_scale = 0.5 if i == j else 1.0
        if k == l:
            orbit_scale *= 0.5
        if (i == k and j == l) or (i == l and j == k):
            orbit_scale *= 0.5
        if unrestricted:
            closed = orbit_scale * (
                4.0 * (alpha[i][j] + beta[i][j]) * (alpha[k][l] + beta[k][l])
                - 2.0
                * (
                    alpha[i][k] * alpha[j][l]
                    + alpha[i][l] * alpha[j][k]
                    + beta[i][k] * beta[j][l]
                    + beta[i][l] * beta[j][k]
                )
            )
        else:
            closed = orbit_scale * (
                4.0 * alpha[i][j] * alpha[k][l]
                - alpha[i][k] * alpha[j][l]
                - alpha[i][l] * alpha[j][k]
            )
        assert closed == pytest.approx(old, abs=1.0e-12)


def test_equal_shell_pair_component_domain_matches_active_tile_triangle():
    """Avoid double-counting (ij|kl) and (kl|ij) in shell-wide workers."""

    source = emit_shell_class_fused_cuda(FUSED_SHELL_SPEC_BY_NAME["pppp"])
    assert (
        "shared.task.shell_pair[0] != shared.task.shell_pair[1] || "
        "(first_p * 3U + second_p) >= (third_p * 3U + fourth_p)" in source
    )


def test_fused_cuda_can_emit_fock_values_and_force_gradients_together():
    """Generate both consumers from one integral and component schedule IR."""

    plan = build_fused_shell_plan(
        DPDS_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
    )
    source = emit_shell_class_fused_cuda(DPDS_SPEC, plan)
    assert "generated_dpds_component_value" in source
    assert "generated_dpds_component_gradient" in source
    assert "generated_dpds_shell_class_fock_rhf_kernel" in source
    assert "generated_dpds_shell_class_fock_uhf_persistent_kernel" in source
    assert "generated_dpds_shell_class_force_rhf_kernel" in source
    force_only = emit_shell_class_fused_cuda(DPDS_SPEC)
    assert "shell_class_fock" not in force_only
    assert "coordinate_gradient" not in source
    assert "Dual3" not in source


def test_tiled_component_schedule_covers_force_fock_and_benchmark_oracle():
    """Lower component tiles without silently dropping high-index AO quartets."""

    integral = build_integral_ir(
        DPPP_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
    )
    schedule = next(
        item
        for item in schedule_candidates(integral)
        if item.kind == ScheduleKind.TILED_COMPONENTS and item.component_tile == 64
    )
    plan = build_fused_shell_plan(
        DPPP_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=schedule,
    )
    source = emit_shell_class_fused_cuda(DPPP_SPEC, plan)
    assert "kGeneratedDpppBlockThreads = 64U" in source
    assert source.count("component_tile_begin += 64U") == 2
    assert source.count("state += kGeneratedDpppBlockThreads") == 1
    assert source.count("state += kGeneratedDpppFockBlockThreads") == 1
    assert "generated_dppp_component_gradient<true>" in source
    assert "generated_dppp_component_value<true>" in source

    benchmark = emit_shell_class_benchmark_cuda(
        DPPP_SPEC,
        task_count=2,
        primitive_count=1,
        warmups=0,
        iterations=1,
        samples=1,
        schedule=schedule,
    )
    # The generated candidate and independent recompute oracle must both walk
    # every tile; otherwise a partial-component benchmark can falsely pass.
    assert benchmark.count("component_tile_begin += 64U") == 2


def test_dpds_fused_cuda_is_generated_from_shell_spec():
    source = emit_shell_class_fused_cuda(DPDS_SPEC)
    assert "kGeneratedDpdsComponentCount = 108U" in source
    assert "kGeneratedDpdsBlockThreads = 128U" in source
    assert "const unsigned third_d = component % 6U" in source
    assert "const unsigned fourth_s = 0U" in source
    assert "generated_dpds_d_axes[third_d][1]" in source
    assert "GeneratedDpdsPairTerm second_terms[4]" in source
    assert "generated_dpds_shell_class_force_rhf_kernel" in source
    assert "generated_dpds_shell_class_force_uhf_persistent_kernel" in source
    assert "generated_dppp" not in source
    assert "__noinline__" not in source
    assert "Dual3" not in source


def test_pair_orientation_changes_the_materialized_contraction_pair():
    """Make pair orientation a measured CUDA code shape, not manifest metadata."""

    base = build_fused_shell_plan(
        DPDS_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
    ).schedule
    canonical = emit_shell_class_fused_cuda(
        DPDS_SPEC,
        build_fused_shell_plan(
            DPDS_SPEC,
            consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
            schedule=replace(base, pair_orientation=PairOrientation.CANONICAL),
        ),
    )
    swapped = emit_shell_class_fused_cuda(
        DPDS_SPEC,
        build_fused_shell_plan(
            DPDS_SPEC,
            consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
            schedule=replace(base, pair_orientation=PairOrientation.SWAPPED),
        ),
    )
    assert "GeneratedDpdsPairTerm second_terms[4]" in canonical
    assert "GeneratedDpdsValueTerm second_terms[4]" in canonical
    assert "GeneratedDpdsPairTerm first_terms[8]" in swapped
    assert "GeneratedDpdsValueTerm first_terms[8]" in swapped
    assert "GeneratedDpdsPairTerm first_terms[8]" not in canonical
    assert "GeneratedDpdsValueTerm first_terms[8]" not in canonical


def test_ddps_fused_cuda_generates_order_four_double_matchings():
    source = emit_shell_class_fused_cuda(DDPS_SPEC)
    assert "kGeneratedDdpsComponentCount = 108U" in source
    assert "kGeneratedDdpsBlockThreads = 128U" in source
    assert "PairOrder == 1U || PairOrder == 4U" in source
    assert "first_removed | second_removed, 2U" in source
    assert "first_d >= second_d" in source
    assert "GeneratedDdpsPairTerm second_terms[2]" in source
    assert "generated_ddps_shell_class_force_rhf_kernel" in source
    assert "generated_dppp" not in source
    assert "__noinline__" not in source
    assert "Dual3" not in source


def test_rys4_component_lanes_raise_a_second_center_d_shell():
    """Generate the exact b=3 HRR state needed by a d-center derivative."""

    schedule = ScheduleIR(
        kind=ScheduleKind.COMPONENT_LANES,
        block_threads=128,
        component_tile=DDPS_SPEC.component_count,
        tasks_per_warp=1,
        shared_coulomb=True,
        minimum_blocks_per_sm=1,
    )
    plan = build_fused_shell_plan(
        DDPS_SPEC,
        schedule=schedule,
        recurrence="rys4",
    )
    source = emit_shell_class_fused_cuda(DDPS_SPEC, plan)
    assert "if (b == 2U)" in source
    assert "trr, a + 3U, c, d, cd" in source
    assert "3.0 * ab * raised_twice" in source
    assert "__noinline__ void\ngenerated_ddps_rys4_component_lane_task" in source
    assert "generated_ddps_shell_class_force_rhf_kernel" in source


@pytest.mark.parametrize("name", ("psps", "ppss"))
def test_low_order_production_force_is_generated_by_common_rys2_pipeline(name: str):
    """Keep low-order production ownership in the shared IR and CUDA emitter."""

    selection = next(
        item
        for item in load_production_kernel_selections(
            REPOSITORY_ROOT
            / "tools"
            / "vibeqc_codegen"
            / "production_shell_classes.json",
            "sm_120",
        )
        if item.spec.name == name
    )
    plan = build_fused_shell_plan(
        selection.spec,
        consumers=selection.consumers,
        schedule=selection.schedule,
        recurrence=selection.recurrence,
    )
    source = emit_shell_class_fused_cuda(
        selection.spec,
        plan,
        fock_schedule=selection.fock_schedule,
    )
    assert selection.recurrence == "rys2"
    assert selection.schedule.kind == ScheduleKind.THREAD_TASKS
    assert selection.schedule.block_threads == 32
    assert f"generated_{name}_rys2_force_task" in source
    assert f"generated_{name}_shell_class_force_rhf_persistent_kernel" in source
    assert f"generated_{name}_shell_class_force_uhf_persistent_kernel" in source
    assert f"generated_{name}_shell_class_fock_rhf_persistent_kernel" in source
    assert "atomicAdd(task_head, 32U)" in source
    assert "VIBEQC_LOW_ORDER_TASK_BEGIN" not in source
    assert f"generated_{name}_contract_weighted_coulomb" not in source
    assert "Dual3" not in source


def test_packed_force_geometry_omits_component_coulomb_tables():
    """Keep packed-force shared storage limited to fields its CSE consumes."""

    source = emit_shell_class_fused_cuda(
        PSPS_SPEC,
        build_fused_shell_plan(
            PSPS_SPEC,
            consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
            schedule=ScheduleIR(
                kind=ScheduleKind.PACKED_TASKS,
                block_threads=32,
                component_tile=PSPS_SPEC.component_count,
                tasks_per_warp=32,
                shared_coulomb=False,
            ),
        ),
    )
    force_geometry = source.split(
        "struct GeneratedPspsPackedForceGeometry", maxsplit=1
    )[1].split("};", maxsplit=1)[0]
    assert "coordinate_powers" not in force_geometry
    assert "negative_two_rho_powers" not in force_geometry
    assert "pair_shifts[3][3]" in force_geometry
    assert (
        "pair_shifts[3][axis]"
        not in source.split("generated_psps_make_packed_force_geometry", maxsplit=1)[
            1
        ].split("/** Density-weighted shell gradient", maxsplit=1)[0]
    )
    assert "GeneratedPspsPackedForceLaneStorage" in source
    assert "GeneratedPspsPackedFockLaneStorage" in source


@pytest.mark.parametrize(
    ("spec", "pair_shift_rows"),
    ((PSPS_SPEC, 3), (DPPP_SPEC, 4)),
)
def test_packed_force_geometry_cuda_is_lowered_from_backend_neutral_algebra(
    spec, pair_shift_rows
):
    """Keep packed geometry setup derived from the shared scalar IR."""

    source = emit_shell_class_fused_cuda(
        spec,
        build_fused_shell_plan(
            spec,
            consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
            schedule=ScheduleIR(
                kind=ScheduleKind.PACKED_TASKS,
                block_threads=32,
                component_tile=spec.component_count,
                tasks_per_warp=32,
                shared_coulomb=False,
            ),
        ),
    )
    setup = source.split(
        f"generated_{spec.name}_make_packed_force_geometry", maxsplit=1
    )[1].split("/** Density-weighted shell gradient", maxsplit=1)[0]
    assert f"pair_shifts[{pair_shift_rows}][3]" in source
    assert "generated_dppp_axis(" not in setup
    assert "argument_squared_distance +=" not in setup
    assert "geometry.pair_shifts[0][0] =" in setup
    assert "geometry.decay_gradients[2][2] =" in setup
    assert "geometry.primitive_coefficient =" in setup
    assert f"boys_values<{spec.maximum_force_coulomb_order}>" in setup
    assert "sqrt(" in setup


def test_packed_force_lowering_uses_explicit_derivative_center_slots():
    """Route packed force atomics through non-final IR recovery metadata."""

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
        PSPS_SPEC,
        operator=operator,
        derivative=operator.nuclear_derivative(),
        contractions=(force,),
    )
    plan = build_fused_shell_plan(
        PSPS_SPEC,
        integral=integral,
        schedule=ScheduleIR(
            kind=ScheduleKind.PACKED_TASKS,
            block_threads=32,
            component_tile=PSPS_SPEC.component_count,
            tasks_per_warp=32,
            shared_coulomb=False,
        ),
    )
    source = emit_shell_class_fused_cuda(PSPS_SPEC, plan)

    # Independent slots are A/C/D, while the recovered force is accumulated
    # into B.  Differentiating center D also requires retaining its decay row.
    assert "decay_gradients[4][3]" in source
    assert "geometry.decay_gradients[3][2]" in source
    assert "0U, 2U, 3U};" in source
    recovery_begin = source.index(
        "const double fourth_force",
        source.index("generated_psps_packed_force_lane"),
    )
    recovery = source[recovery_begin : recovery_begin + 600]
    assert "static_cast<std::size_t>(task.atom[1])" in recovery
    assert "static_cast<std::size_t>(task.atom[3])" not in recovery


@pytest.mark.parametrize(
    ("name", "component_count", "state_count", "block_threads"),
    (
        ("ppps", 27, 20, 32),
        ("dpps", 54, 35, 64),
        ("dsps", 18, 20, 32),
    ),
)
def test_generated_fock_workers_use_value_only_shell_schedules(
    name, component_count, state_count, block_threads
):
    """Keep force-only gradients out of the generated SCF hot path."""

    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    plan = build_fused_shell_plan(
        spec,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
    )
    source = emit_shell_class_fused_cuda(spec, plan)
    class_name = name[0].upper() + name[1:]
    assert f"kGenerated{class_name}ComponentCount = {component_count}U" in source
    assert (
        f"kGenerated{class_name}FockCoulombStateCount =\n    {state_count}U" in source
    )
    assert f"kGenerated{class_name}FockBlockThreads = {block_threads}U" in source
    assert f"Generated{class_name}ValueTerm" in source
    assert f"generated_{name}_component_value" in source
    assert f"generated_{name}_shell_class_fock_rhf_persistent_kernel" in source
    assert f"generated_{name}_shell_class_fock_uhf_persistent_kernel" in source
    fock_fragment = source.split(
        "/** Coefficient-only pair term used by the SCF Fock recurrence. */",
        maxsplit=1,
    )[1]
    assert f"generated_{name}_density_coefficient" not in fock_fragment


def test_production_manifest_drives_generated_registry_and_shards(tmp_path: Path):
    """Keep machine CUDA out of Git while retaining deterministic builds."""

    manifest = (
        REPOSITORY_ROOT / "tools" / "vibeqc_codegen" / "production_shell_classes.json"
    )
    specifications = load_production_manifest(manifest)
    fock_specifications = load_production_fock_manifest(manifest)
    assert tuple(spec.name for spec in specifications) == (
        "dppp",
        "dpdp",
        "dddp",
        "dddd",
        "dpss",
        "dsds",
        "ddss",
        "ddpp",
        "ddds",
        "dpds",
        "ddps",
        "fpps",
        "ppps",
        "dpps",
        "dsps",
        "dspp",
        "pppp",
        "psps",
        "ppss",
        "dsss",
    )
    assert tuple(spec.name for spec in fock_specifications) == (
        "ssss",
        "psss",
        "dppp",
        "dpdp",
        "dddp",
        "dddd",
        "dpss",
        "dsds",
        "ddss",
        "ddpp",
        "ddds",
        "dpds",
        "ddps",
        "ppps",
        "dpps",
        "dsps",
        "dspp",
        "pppp",
        "psps",
        "ppss",
        "dsss",
    )
    selections = load_production_kernel_selections(manifest, "sm_120")
    assert tuple(
        selection.spec.name
        for selection in selections
        if KernelConsumer.FORCE in selection.consumers
    ) == tuple(
        spec.name for spec in specifications
    )
    assert all(selection.architecture == "sm_120" for selection in selections)
    assert all(
        selection.schedule.algebra_placement
        == AlgebraPlacement.MATERIALIZED_CSE
        for selection in selections
    )
    shards = _partition_production_selections(selections, shard_count=8)
    shard_by_name = {
        selection.spec.name: shard_index
        for shard_index, shard in enumerate(shards)
        for selection in shard
    }
    # Removing a manifest entry must not invalidate unrelated source shards.
    without_dppp = _partition_production_selections(
        tuple(selection for selection in selections if selection.spec.name != "dppp"),
        shard_count=8,
    )
    assert {
        selection.spec.name: shard_index
        for shard_index, shard in enumerate(without_dppp)
        for selection in shard
    } == {
        name: shard_index
        for name, shard_index in shard_by_name.items()
        if name != "dppp"
    }
    assert {
        selection.spec.name: selection.schedule.pair_storage for selection in selections
    } == {
        spec.name: (
            PairStorage.RECOMPUTED
            if spec.name in ("dddp", "dddd", "ddds")
            else PairStorage.MATERIALIZED
        )
        for spec in (selection.spec for selection in selections)
    }
    assert tuple(selection.consumers for selection in selections) == tuple(
        (KernelConsumer.FOCK,)
        if selection.spec.name in ("ssss", "psss")
        else (
            (KernelConsumer.FORCE,)
            if selection.spec.name == "fpps"
            else (KernelConsumer.FOCK, KernelConsumer.FORCE)
        )
        for selection in selections
    )
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    first = write_production_bundle(manifest, first_directory, shard_count=4)
    second = write_production_bundle(manifest, second_directory, shard_count=4)
    assert [path.name for path in first] == [path.name for path in second]
    for first_path, second_path in zip(first, second, strict=True):
        assert first_path.read_bytes() == second_path.read_bytes()
        assert b"\0" not in first_path.read_bytes()
    header = emit_registry_header(selections)
    assert '{"dppp", 12U, 5U, 128U, 3U, 162U}' in header
    assert '{"dpds", 13U, 5U, 256U, 3U, 108U}' in header
    assert '{"ddps", 16U, 5U, 256U, 3U, 108U}' in header
    assert '{"ppps", 4U, 3U, 256U, 3U, 27U}' in header
    assert '{"dsps", 7U, 3U, 32U, 3U, 18U}' in header
    assert '{"dpdp", 14U, 6U, 256U, 3U, 324U}' in header
    assert '{"dddp", 19U, 7U, 256U, 3U, 648U}' in header
    assert '{"dddd", 20U, 8U, 256U, 3U, 1296U}' in header
    assert '{"dpss", 10U, 3U, 32U, 3U, 18U}' in header
    assert '{"dsds", 9U, 4U, 64U, 3U, 36U}' in header
    assert '{"ddss", 15U, 4U, 64U, 3U, 36U}' in header
    assert '{"ddpp", 17U, 6U, 256U, 3U, 324U}' in header
    assert '{"ddds", 18U, 6U, 224U, 3U, 216U}' in header
    assert '{"dspp", 8U, 4U, 128U, 3U, 54U}' in header
    assert '{"dpps", 11U, 4U, 128U, 3U, 54U}' in header
    assert '{"pppp", 5U, 4U, 128U, 3U, 81U}' in header
    assert '{"psps", 2U, 2U, 32U, 3U, 9U}' in header
    assert '{"ppss", 3U, 2U, 32U, 3U, 9U}' in header
    assert '{"dsss", 6U, 2U, 32U, 3U, 6U}' in header
    assert "VIBEQC_AOT_SHELL_CLASSES" in header
    assert "VIBEQC_AOT_FOCK_SHELL_CLASSES" in header
    shards = "\n".join(
        path.read_text(encoding="utf-8") for path in first if "shard" in path.name
    )
    assert "offsetof(GeneratedDpppShellTask, shell_pair)" in shards
    assert "offsetof(GeneratedDpppPrimitivePairData, product_center)" in shards
    assert "const std::uint32_t* task_offset" in header
    generated_sources = [path.read_text(encoding="utf-8") for path in first]
    assert any("*task_offset + task_index" in source for source in generated_sources)
    assert any(
        "worker_blocks, tasks, task_offset" in source for source in generated_sources
    )


@pytest.mark.parametrize("architecture", ("sm_80", "sm_86", "sm_89", "sm_90"))
def test_unmeasured_cuda_targets_resolve_to_empty_portable_profile(
    architecture: str,
):
    """Never reuse the measured RTX 5090 schedule on another compute target."""

    manifest = (
        REPOSITORY_ROOT / "tools" / "vibeqc_codegen" / "production_shell_classes.json"
    )
    resolved = resolve_production_profile(manifest, architecture)
    assert resolved.profile == "portable_cuda"
    assert resolved.portable is True
    assert resolved.tuned is False
    assert resolved.selections == ()
    with pytest.raises(ValueError, match="incompatible"):
        resolve_production_profile(manifest, architecture, "sm_120")


def _small_multi_profile_manifest(path: Path) -> None:
    """Write two legal measured profiles for collision/link tests."""

    schedule = {
        "kind": "packed_tasks",
        "block_threads": 32,
        "component_tile": 6,
        "tasks_per_warp": 32,
        "shared_coulomb": False,
        "pair_orientation": "canonical",
        "pair_storage": "materialized",
        "unroll_pair_terms": True,
    }
    profile = {
        "kind": "tuned",
        "cuda_toolkit": "12.9.1",
        "generator_abi": 1,
        "kernels": [
            {
                "shell_class": "dsss",
                "consumers": ["force"],
                "schedule": schedule,
            }
        ],
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "default_architecture": "sm_120",
                "architectures": {
                    "sm_80": profile,
                    "sm_120": profile,
                    "portable_cuda": {"kind": "portable", "kernels": []},
                },
            }
        ),
        encoding="utf-8",
    )


def test_multi_profile_bundle_is_order_independent_and_collision_free(
    tmp_path: Path,
):
    """Generate separate symbols, metadata, and shards for every target."""

    manifest = tmp_path / "manifest.json"
    _small_multi_profile_manifest(manifest)
    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    first = write_production_bundles(manifest, first_directory, 1, ("sm_120", "sm_80"))
    second = write_production_bundles(
        manifest, second_directory, 1, ("sm_80", "sm_120")
    )
    assert [path.relative_to(first_directory) for path in first] == [
        path.relative_to(second_directory) for path in second
    ]
    for first_path, second_path in zip(first, second, strict=True):
        assert first_path.read_bytes() == second_path.read_bytes()

    registry = (first_directory / "vibeqc_generated_shell_registry.cu").read_text(
        encoding="utf-8"
    )
    assert "vibeqc_launch_sm80_generated_dsss" in registry
    assert "vibeqc_launch_sm120_generated_dsss" in registry
    header = (first_directory / "vibeqc_generated_shell_registry.hpp").read_text(
        encoding="utf-8"
    )
    assert header.index('"sm_80"') < header.index('"sm_120"')
    sm80 = next(path for path in first if "sm80_shard" in path.name).read_text(
        encoding="utf-8"
    )
    sm120 = next(path for path in first if "sm120_shard" in path.name).read_text(
        encoding="utf-8"
    )
    assert "namespace vibeqc::scf::generated::profile_sm80" in sm80
    assert "namespace vibeqc::scf::generated::profile_sm120" in sm120


def test_multi_profile_objects_compile_and_link_when_nvcc_is_configured(
    tmp_path: Path,
):
    """Verify two architecture bundles do not collide at host or device link."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the multi-profile compile/link test")
    manifest = tmp_path / "manifest.json"
    _small_multi_profile_manifest(manifest)
    output = tmp_path / "generated"
    write_production_bundles(manifest, output, 1, ("sm_80", "sm_120"))
    objects = []
    for architecture in ("sm_80", "sm_120"):
        source = next(
            (output / architecture).glob(
                f"vibeqc_generated_shell_{architecture.replace('_', '')}_shard_0.cu"
            )
        )
        obj = tmp_path / f"{architecture}.o"
        result = subprocess.run(
            [
                nvcc,
                "-std=c++20",
                f"-arch={architecture}",
                f"-I{REPOSITORY_ROOT / 'src'}",
                "-Xcompiler=-fPIC",
                "-c",
                str(source),
                "-o",
                str(obj),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        objects.append(obj)
    registry_object = tmp_path / "registry.o"
    result = subprocess.run(
        [
            nvcc,
            "-std=c++20",
            "-arch=sm_80",
            f"-I{output}",
            f"-I{REPOSITORY_ROOT / 'src'}",
            "-Xcompiler=-fPIC",
            "-c",
            str(output / "vibeqc_generated_shell_registry.cu"),
            "-o",
            str(registry_object),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    library = tmp_path / "libprofiles.so"
    result = subprocess.run(
        [
            nvcc,
            "-shared",
            str(registry_object),
            *(map(str, objects)),
            "-o",
            str(library),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_runtime_buckets_all_generated_classes_before_dispatch():
    """Prevent production promotion from restoring one scan per exact class."""

    source = (REPOSITORY_ROOT / "src" / "scf" / "cuda_rhf.cu").read_text(
        encoding="utf-8"
    )
    assert "classify_generated_shell_tasks_kernel" in source
    assert "prefix_generated_shell_task_counts_kernel" in source
    assert "materialize_generated_shell_tasks_kernel" in source
    assert "compact_generated_shell_tasks_kernel" not in source
    assert source.count("classify_generated_shell_tasks_kernel<<<") == 1


def test_one_electron_force_batches_point_charges_in_one_warp_per_ao_pair():
    """Keep nuclear centers device-batched with the scalar path as fallback."""

    source = (REPOSITORY_ROOT / "src" / "scf" / "cuda_rhf.cu").read_text(
        encoding="utf-8"
    )
    cooperative_begin = source.index(
        "void contracted_one_electron_force_pair_cooperative("
    )
    cooperative_end = source.index(
        "template <unsigned MaximumAngular, typename Scalar>",
        cooperative_begin,
    )
    cooperative = source[cooperative_begin:cooperative_end]
    assert "atom_base += warpSize" in cooperative
    assert "shared_coefficients[axis]" in cooperative
    assert "if (lane == 0U)" in cooperative
    assert "__shfl_down_sync" in cooperative
    assert source.count("one_electron_force_cooperative_kernel<<<") == 1
    assert 'std::getenv("VIBEQC_ONE_ELECTRON_FORCE_SCALAR")' in source
    assert "scalar_one_electron_force_environment == nullptr" in source


def test_batched_finalization_reuses_each_converged_raw_fock():
    """Reuse requested-accuracy peers and restore shared force metadata."""

    source = (REPOSITORY_ROOT / "src" / "scf" / "cuda_rhf.cu").read_text(
        encoding="utf-8"
    )
    assert "template <bool RetainConvergedDensity>" in source
    assert 'std::getenv("VIBEQC_FINAL_FOCK_REBUILD")' in source
    assert "select_final_fock_rebuild_kernel" in source
    assert "kTightConvergedFockReuseDensityRms = 1.0e-12" in source
    assert "kExpandedConvergedFockReuseDensityTolerance = 1.0e-8" in source
    assert "kExpandedConvergedFockReuseDensityRms = 2.0e-9" in source
    assert "converged_fock_reuse_density_rms(options.density_tolerance)" in source
    assert "copy_selected_matrices_kernel" in source
    assert "launch_direct_quartet_metadata(density)" in source
    # A resident dm0 is already normalized for its cached overlap matrix, so a
    # geometry change must re-run the warm-density normalization path.
    assert "plan.resident_warm_positions == host.positions" in source
    assert "plan.resident_warm_density == host.warm_density" in source
    assert "iteration > 1 || has_energy_baseline" in source
    assert "update_convergence_kernel<true>" in source
    assert "update_uhf_convergence_kernel<true>" in source


def test_ppps_queue_buckets_orientation_and_primitive_signature_on_device():
    """Keep Phase-3 bucketing on the compact production queue and A/B-able."""

    source = (REPOSITORY_ROOT / "src" / "scf" / "cuda_rhf.cu").read_text(
        encoding="utf-8"
    )
    assert "kPppsSignatureBucketCount" in source
    assert "resident_ppps_signature_bucket" in source
    assert "prefix_ppps_resident_signature_buckets_kernel" in source
    assert 'std::getenv("VIBEQC_PPPS_SIGNATURE_BUCKETING")' in source
    assert 'std::getenv("VIBEQC_PPPS_BLOCK_THREADS")' in source
    assert "ppps_resident_block_threads_requested" in source
    assert "resident_signature_offsets[bucket_index]" in source
    assert "atomicAdd(resident_signature_write_counts + bucket_index" in source
    assert "std::uint32_t* generated_ppps_resident_signatures =" in source
    assert "shell_class_profiling\n      ? arena_pointer<std::uint32_t>" in source
    assert "kBoundedForceSignatureShellClassMask" in source
    assert "bounded_force_signature_bucket" in source
    assert "scan_bounded_force_signature_counts_kernel" in source
    assert "prefix_bounded_force_signature_blocks_kernel" in source
    assert "bounded_paged_force_shell_class_mask" in source
    assert "bounded_force_signature_offsets, true" in source


def test_bounded_force_signature_mask_tracks_warp_uniform_schedules():
    """Keep page sorting aligned with every production lockstep task worker."""

    manifest = (
        REPOSITORY_ROOT / "tools" / "vibeqc_codegen" / "production_shell_classes.json"
    )
    selections = load_production_kernel_selections(manifest, "sm_120")
    lockstep_kinds = {
        ScheduleKind.PACKED_TASKS,
        ScheduleKind.THREAD_TASKS,
        ScheduleKind.SUBGROUP_TASKS,
    }
    expected_constants = {
        f"k{selection.spec.name.capitalize()}ShellClass"
        for selection in selections
        if selection.schedule.kind in lockstep_kinds
    }
    source = (REPOSITORY_ROOT / "src" / "scf" / "cuda_rhf.cu").read_text(
        encoding="utf-8"
    )
    mask_begin = source.index(
        "constexpr std::uint64_t kBoundedForceSignatureShellClassMask"
    )
    mask_end = source.index(";", mask_begin)
    configured_constants = set(
        re.findall(r"<< (k[A-Za-z0-9]+ShellClass)", source[mask_begin:mask_end])
    )
    assert configured_constants == expected_constants


def test_warm_density_validation_parallelizes_each_system_matrix():
    """Keep fixed-dm0 setup from regressing to one serial N^2 worker."""

    source = (REPOSITORY_ROOT / "src" / "scf" / "cuda_rhf.cu").read_text(
        encoding="utf-8"
    )
    assert "constexpr unsigned kWarmDensityThreads = 256" in source
    assert "warm_density_block_sum<kWarmDensityThreads>" in source
    for kernel in ("apply_warm_density_kernel", "apply_uhf_warm_density_kernel"):
        launch = rf"{kernel}<<<static_cast<unsigned>\(batch_size\),\s*"
        assert re.search(launch + r"kWarmDensityThreads", source)


def test_force_density_product_screening_is_force_only_and_conservative():
    """Keep the force queue optional without weakening the SCF Fock gate."""

    source = (REPOSITORY_ROOT / "src" / "scf" / "cuda_rhf.cu").read_text(
        encoding="utf-8"
    )
    assert "enum class DirectScreeningPurpose" in source
    assert "DirectScreeningPurpose::Fock" in source
    assert "DirectScreeningPurpose::Force" in source
    assert "kForceDensityProductScreeningTolerance = 1.0e-14" in source
    assert "fmin(screening_tolerance, kForceDensityProductScreeningTolerance)" in source
    assert 'std::getenv("VIBEQC_FORCE_DENSITY_PRODUCT_SCREENING")' in source
    assert "launch_direct_force_compaction();" in source


def test_cached_direct_plan_reuses_immutable_task_layout():
    """Keep quadratic shell-pair topology enumeration out of warm replay."""

    source = (REPOSITORY_ROOT / "src" / "scf" / "cuda_rhf.cu").read_text(
        encoding="utf-8"
    )
    layout_begin = source.index("detail::DirectQuartetTaskLayout direct_task_layout")
    layout_end = source.index(
        "// Direct consumers expand each compact logical tile", layout_begin
    )
    layout_setup = source[layout_begin:layout_end]
    assert "requested_quartet_direct && first_setup" in layout_setup
    assert "plan.total_shell_quartet_tiles" in layout_setup
    assert source.count("detail::make_direct_quartet_task_layout(") == 1
    assert "**plan, candidate, options" in source


def test_generated_order2_fock_masks_handwritten_fallback():
    """Prevent generated order-two Fock quartets from being scattered twice."""

    source = (REPOSITORY_ROOT / "src" / "scf" / "cuda_rhf.cu").read_text(
        encoding="utf-8"
    )
    task_begin = source.index("contract_fock_direct_order2_task(")
    task_end = source.index(
        "/** Fixed-capacity wrapper retained for high-register angular orders. */",
        task_begin,
    )
    task_source = source[task_begin:task_end]
    assert "generated_fock_shell_class_mask" in task_source
    assert "std::uint64_t{1} << shell_class" in task_source

    worker_begin = source.index("void build_fock_direct_order2_persistent_kernel(")
    worker_end = source.index(
        "/** Consume only the active compacted Fock domain from a device queue. */",
        worker_begin,
    )
    worker_source = source[worker_begin:worker_end]
    assert "generated_fock_shell_class_mask" in worker_source
    assert "contract_fock_direct_order2_task<Unrestricted>" in worker_source


def test_production_codegen_cmake_tracks_transitive_generator_inputs():
    """Regenerate production CUDA whenever shared compiler stages change."""

    source = (REPOSITORY_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    for dependency in (
        "tools/vibeqc_codegen/cuda.py",
        "tools/vibeqc_codegen/cuda_lowering.py",
        "tools/vibeqc_codegen/dppp_dispatch.py",
        "tools/vibeqc_codegen/expr.py",
        "tools/vibeqc_codegen/fused_schedule.py",
        "tools/vibeqc_codegen/ir.py",
        "tools/vibeqc_codegen/production.py",
        "tools/vibeqc_codegen/rys.py",
        "tools/vibeqc_codegen/rys3_data.py",
        "tools/vibeqc_codegen/rys5_data.py",
        "tools/vibeqc_codegen/shell_class.py",
        "tools/vibeqc_codegen/shell_spec.py",
    ):
        assert dependency in source
    assert "tools/vibeqc_codegen/low_order_force.py" not in source


def test_batch_screening_ranks_real_profile_and_emits_one_process_driver():
    with pytest.raises(ValueError, match="requires --profile"):
        candidate_specs()

    payload = {
        "shell_classes": [
            {"class": "dppp"},
            {"class": "ppps"},
            {"class": "ddps"},
            {"class": "psss"},
        ]
    }
    ranked = rank_profiled_candidates(payload, limit=2)
    assert tuple(spec.name for spec in ranked) == ("psss",)
    candidate = DEFAULT_CANDIDATES[0]
    source = emit_candidate_translation_unit(
        candidate,
        task_count=2,
        primitive_count=1,
        warmups=0,
        iterations=1,
        samples=1,
    )
    assert f"vibeqc_run_shell_class_{candidate.name}" in source
    driver = emit_batch_driver((candidate,))
    assert "cudaFree(nullptr)" in driver
    assert f"vibeqc_run_shell_class_{candidate.name}()" in driver


@pytest.mark.parametrize(
    ("name", "recurrence", "resource_limits"),
    (
        ("ppps", "rys3", None),
        ("dpss", "rys3", RTX5090_DPSS_SCALAR_RYS3_RESOURCE_LIMITS),
        ("psss", "rys2", None),
        ("psps", "rys2", None),
        ("ppss", "rys2", None),
        ("dsss", "rys2", None),
    ),
)
def test_scalar_rys_cuda_compiles_with_bounded_call_save_when_nvcc_is_configured(
    tmp_path: Path,
    name: str,
    recurrence: str,
    resource_limits: dict[str, tuple[int, int, int]] | None,
):
    """Bound scalar fixed-root resources before production promotion."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    schedule = ScheduleIR(
        kind=ScheduleKind.THREAD_TASKS,
        block_threads=32,
        component_tile=spec.component_count,
        tasks_per_warp=32,
        shared_coulomb=False,
        minimum_blocks_per_sm=8,
    )
    plan = build_fused_shell_plan(
        spec,
        schedule=schedule,
        recurrence=recurrence,
    )
    source = tmp_path / f"generated_{name}_{recurrence}.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(spec, plan),
        encoding="utf-8",
    )
    cubin = tmp_path / f"generated_{name}_{recurrence}.cubin"
    compile_started = time.perf_counter()
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-cubin",
            "-Xptxas=-v",
            str(source),
            "-o",
            str(cubin),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    compile_seconds = time.perf_counter() - compile_started
    if os.environ.get("VIBEQC_NVCC_VERBOSE"):
        print(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    print(
        json.dumps(
            {
                "compile_seconds": compile_seconds,
                "cubin_bytes": cubin.stat().st_size,
            },
            sort_keys=True,
        )
    )
    if cuda_architecture == "sm_120":
        output = result.stdout + result.stderr
        assert f"generated_{name}_shell_class_force_rhf_persistent_kernel" in output
        if resource_limits is not None:
            assert_rtx5090_resources(output, resource_limits)
        resource_records = re.findall(
            r"(\d+) bytes stack frame, (\d+) bytes spill stores, "
            r"(\d+) bytes spill loads",
            output,
        )
        assert resource_records
        numeric_records = tuple(tuple(map(int, record)) for record in resource_records)
        if name == "ppps":
            assert max(record[0] for record in numeric_records) <= 56
            assert max(record[1] for record in numeric_records) <= 64
            assert max(record[2] for record in numeric_records) <= 64
        else:
            assert all(record == (0, 0, 0) for record in numeric_records)


def test_dppp_cooperative_rys4_compiles_without_spills_when_nvcc_is_configured(
    tmp_path: Path,
):
    """Apply the sm_120 resource gate before any production promotion."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    schedule = ScheduleIR(
        kind=ScheduleKind.COMPONENT_LANES,
        block_threads=192,
        component_tile=DPPP_SPEC.component_count,
        tasks_per_warp=1,
        shared_coulomb=True,
        pair_orientation=PairOrientation.SWAPPED,
        pair_storage=PairStorage.MATERIALIZED,
        unroll_pair_terms=True,
        minimum_blocks_per_sm=2,
    )
    plan = build_fused_shell_plan(
        DPPP_SPEC,
        schedule=schedule,
        recurrence="rys4",
    )
    source = tmp_path / "generated_dppp_cooperative_rys4.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(DPPP_SPEC, plan),
        encoding="utf-8",
    )
    cubin = tmp_path / "generated_dppp_cooperative_rys4.cubin"
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-O3",
            "-cubin",
            "-Xptxas=-v",
            str(source),
            "-o",
            str(cubin),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    if os.environ.get("VIBEQC_NVCC_VERBOSE"):
        print(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    if cuda_architecture == "sm_120":
        assert_rtx5090_resources(
            result.stdout + result.stderr,
            {
                "generated_dppp_shell_class_force_rhf_kernel": (168, 160, 1024),
                "generated_dppp_shell_class_force_uhf_kernel": (168, 160, 1024),
                "generated_dppp_shell_class_force_rhf_persistent_kernel": (
                    168,
                    160,
                    1024,
                ),
                "generated_dppp_shell_class_force_uhf_persistent_kernel": (
                    168,
                    160,
                    1024,
                ),
            },
        )


@pytest.mark.parametrize(
    ("name", "schedule", "ordinary_limit", "persistent_limit"),
    (
        (
            "dpdp",
            ScheduleIR(
                kind=ScheduleKind.COMPONENT_LANES,
                block_threads=352,
                component_tile=324,
                tasks_per_warp=1,
                shared_coulomb=True,
                pair_orientation=PairOrientation.SWAPPED,
                pair_storage=PairStorage.RECOMPUTED,
                unroll_pair_terms=False,
                minimum_blocks_per_sm=1,
            ),
            (168, 200, 1312),
            (168, 200, 1320),
        ),
        (
            "dpds",
            ScheduleIR(
                kind=ScheduleKind.SUBGROUP_TASKS,
                block_threads=256,
                component_tile=108,
                tasks_per_warp=4,
                shared_coulomb=True,
                pair_orientation=PairOrientation.CANONICAL,
                pair_storage=PairStorage.MATERIALIZED,
                unroll_pair_terms=True,
                minimum_blocks_per_sm=1,
            ),
            (254, 112, 36872),
            (254, 112, 36872),
        ),
        (
            "ddpp",
            ScheduleIR(
                kind=ScheduleKind.COMPONENT_LANES,
                block_threads=352,
                component_tile=324,
                tasks_per_warp=1,
                shared_coulomb=True,
                pair_orientation=PairOrientation.SWAPPED,
                pair_storage=PairStorage.RECOMPUTED,
                unroll_pair_terms=False,
                minimum_blocks_per_sm=1,
            ),
            (168, 192, 1312),
            (167, 192, 1320),
        ),
        (
            "ddps",
            ScheduleIR(
                kind=ScheduleKind.SUBGROUP_TASKS,
                block_threads=256,
                component_tile=108,
                tasks_per_warp=4,
                shared_coulomb=True,
                pair_orientation=PairOrientation.CANONICAL,
                pair_storage=PairStorage.MATERIALIZED,
                unroll_pair_terms=True,
                minimum_blocks_per_sm=1,
            ),
            (254, 112, 36872),
            (254, 112, 36872),
        ),
        (
            "ddds",
            ScheduleIR(
                kind=ScheduleKind.COMPONENT_LANES,
                block_threads=224,
                component_tile=216,
                tasks_per_warp=1,
                shared_coulomb=True,
                pair_orientation=PairOrientation.SWAPPED,
                pair_storage=PairStorage.RECOMPUTED,
                unroll_pair_terms=False,
                minimum_blocks_per_sm=1,
            ),
            (254, 192, 1024),
            (254, 192, 1032),
        ),
    ),
)
def test_batched_rys4_hot_classes_compile_without_spills_when_nvcc_is_configured(
    tmp_path: Path,
    name: str,
    schedule: ScheduleIR,
    ordinary_limit: tuple[int, int, int],
    persistent_limit: tuple[int, int, int],
):
    """Lock the sm_120 resource envelope for batched Rys4 promotions."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    plan = build_fused_shell_plan(
        spec,
        schedule=schedule,
        recurrence="rys4",
    )
    source = tmp_path / f"generated_{name}_promoted_rys4.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(spec, plan),
        encoding="utf-8",
    )
    cubin = tmp_path / f"generated_{name}_promoted_rys4.cubin"
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-O3",
            "-cubin",
            "-Xptxas=-v",
            str(source),
            "-o",
            str(cubin),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    output = result.stdout + result.stderr
    if os.environ.get("VIBEQC_NVCC_VERBOSE"):
        print(output)
    assert result.returncode == 0, output
    assert cubin.exists()
    if cuda_architecture == "sm_120":
        function_prefix = f"generated_{name}_shell_class_force"
        assert_rtx5090_resources(
            output,
            {
                f"{function_prefix}_rhf_kernel": ordinary_limit,
                f"{function_prefix}_uhf_kernel": ordinary_limit,
                f"{function_prefix}_rhf_persistent_kernel": (persistent_limit),
                f"{function_prefix}_uhf_persistent_kernel": (persistent_limit),
            },
        )
        helper_records = re.findall(
            r"\d+ bytes stack frame, (\d+) bytes spill stores, "
            r"(\d+) bytes spill loads",
            output,
        )
        assert helper_records
        assert all(
            int(spill_stores) == 0 and int(spill_loads) == 0
            for spill_stores, spill_loads in helper_records
        )


def test_dppp_rys4_uniform_warps_compile_when_nvcc_is_configured(
    tmp_path: Path,
):
    """Compile the 32-task/eight-warp force worker before endpoint testing."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    schedule = ScheduleIR(
        kind=ScheduleKind.SUBGROUP_TASKS,
        block_threads=256,
        component_tile=DPPP_SPEC.component_count,
        tasks_per_warp=4,
        shared_coulomb=True,
        pair_orientation=PairOrientation.SWAPPED,
        pair_storage=PairStorage.MATERIALIZED,
        unroll_pair_terms=True,
        minimum_blocks_per_sm=1,
    )
    plan = build_fused_shell_plan(
        DPPP_SPEC,
        schedule=schedule,
        recurrence="rys4",
    )
    source = tmp_path / "generated_dppp_uniform_warp_rys4.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(DPPP_SPEC, plan),
        encoding="utf-8",
    )
    cubin = tmp_path / "generated_dppp_uniform_warp_rys4.cubin"
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-O3",
            "-cubin",
            "-Xptxas=-v",
            str(source),
            "-o",
            str(cubin),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    if os.environ.get("VIBEQC_NVCC_VERBOSE"):
        print(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert cubin.exists()
    if cuda_architecture == "sm_120":
        assert_rtx5090_resources(
            result.stdout + result.stderr,
            RTX5090_DPPP_UNIFORM_RYS4_RESOURCE_LIMITS,
        )


@pytest.mark.parametrize(
    ("name", "ordinary_limit", "persistent_limit"),
    (
        ("dpps", (216, 56, 36360), (218, 56, 36360)),
        ("dspp", (214, 56, 36360), (216, 56, 36360)),
        ("pppp", (230, 88, 36360), (232, 88, 36360)),
    ),
)
def test_rys3_uniform_warps_compile_without_spills_when_nvcc_is_configured(
    tmp_path: Path,
    name: str,
    ordinary_limit: tuple[int, int, int],
    persistent_limit: tuple[int, int, int],
):
    """Reject uniform Rys3 mappings that exceed their resource envelope."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    schedule = ScheduleIR(
        kind=ScheduleKind.SUBGROUP_TASKS,
        block_threads=256,
        component_tile=spec.component_count,
        tasks_per_warp=4,
        shared_coulomb=True,
        minimum_blocks_per_sm=1,
    )
    plan = build_fused_shell_plan(spec, schedule=schedule, recurrence="rys3")
    source = tmp_path / f"generated_{name}_uniform_warp_rys3.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(spec, plan),
        encoding="utf-8",
    )
    cubin = tmp_path / f"generated_{name}_uniform_warp_rys3.cubin"
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-O3",
            "-cubin",
            "-Xptxas=-v",
            str(source),
            "-o",
            str(cubin),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    if os.environ.get("VIBEQC_NVCC_VERBOSE"):
        print(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert cubin.exists()
    if cuda_architecture == "sm_120":
        function_prefix = f"generated_{name}_shell_class_force"
        assert_rtx5090_resources(
            result.stdout + result.stderr,
            {
                f"{function_prefix}_rhf_kernel": ordinary_limit,
                f"{function_prefix}_uhf_kernel": ordinary_limit,
                f"{function_prefix}_rhf_persistent_kernel": persistent_limit,
                f"{function_prefix}_uhf_persistent_kernel": persistent_limit,
            },
        )


@pytest.mark.parametrize(
    ("name", "block_threads", "resource_limit"),
    (
        ("dpps", 64, (168, 120, 656)),
        ("dsps", 32, (166, 96, 584)),
        ("pppp", 96, (168, 128, 728)),
    ),
)
def test_cooperative_rys3_hot_classes_compile_without_spills_when_nvcc_is_configured(
    tmp_path: Path,
    name: str,
    block_threads: int,
    resource_limit: tuple[int, int, int],
):
    """Apply a zero-spill sm_120 gate to every promoted Rys3 force class."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    schedule = ScheduleIR(
        kind=ScheduleKind.COMPONENT_LANES,
        block_threads=block_threads,
        component_tile=spec.component_count,
        tasks_per_warp=1,
        shared_coulomb=True,
    )
    plan = build_fused_shell_plan(
        spec,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=schedule,
        recurrence="rys3",
    )
    source = tmp_path / f"generated_{name}_cooperative_rys3.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(spec, plan),
        encoding="utf-8",
    )
    cubin = tmp_path / f"generated_{name}_cooperative_rys3.cubin"
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-O3",
            "-cubin",
            "-Xptxas=-v",
            str(source),
            "-o",
            str(cubin),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    output = result.stdout + result.stderr
    if os.environ.get("VIBEQC_NVCC_VERBOSE"):
        print(output)
    assert result.returncode == 0, output
    if cuda_architecture == "sm_120":
        class_name = name[0].upper() + name[1:]
        assert_rtx5090_resources(
            output,
            {
                f"generated_{name}_shell_class_force_rhf_kernel": resource_limit,
                f"generated_{name}_shell_class_force_uhf_kernel": resource_limit,
                f"generated_{name}_shell_class_force_rhf_persistent_kernel": (
                    resource_limit
                ),
                f"generated_{name}_shell_class_force_uhf_persistent_kernel": (
                    resource_limit
                ),
            },
        )
        assert f"Generated{class_name}Rys3Primitive" in source.read_text(
            encoding="utf-8"
        )


def test_ppps_rys3_benchmark_runs_against_component_lanes_when_nvcc_is_configured(
    tmp_path: Path,
):
    """Compare direct Rys recurrence with the current ppps task topology."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA benchmark gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    spec = FUSED_SHELL_SPEC_BY_NAME["ppps"]
    direct_schedule = ScheduleIR(
        kind=ScheduleKind.THREAD_TASKS,
        block_threads=32,
        component_tile=spec.component_count,
        tasks_per_warp=32,
        shared_coulomb=False,
        minimum_blocks_per_sm=8,
    )
    direct_plan = build_fused_shell_plan(
        spec,
        schedule=direct_schedule,
        recurrence="rys3",
    )
    production_plan = next(
        build_fused_shell_plan(
            spec,
            consumers=selection.consumers,
            schedule=selection.schedule,
        )
        for selection in load_production_kernel_selections(
            REPOSITORY_ROOT
            / "tools"
            / "vibeqc_codegen"
            / "production_shell_classes.json",
            "sm_120",
        )
        if selection.spec == spec
    )
    environment = dict(os.environ)

    def compile_and_run(label: str, plan) -> dict[str, object]:
        source = tmp_path / f"generated_ppps_{label}_benchmark.cu"
        source.write_text(
            emit_shell_class_benchmark_cuda(
                spec,
                task_count=512,
                primitive_count=2,
                warmups=1,
                iterations=3,
                samples=3,
                plan=plan,
                benchmark_kernel_only=True,
                persistent_kernel=True,
            ),
            encoding="utf-8",
        )
        executable = tmp_path / f"generated_ppps_{label}_benchmark"
        compile_result = subprocess.run(
            [
                nvcc,
                "-std=c++17",
                f"-arch={cuda_architecture}",
                "-O3",
                str(source),
                "-o",
                str(executable),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert compile_result.returncode == 0, (
            compile_result.stdout + compile_result.stderr
        )
        run_result = subprocess.run(
            [
                "srun",
                "--partition=main",
                "--gres=gpu:5090:1",
                "--nodes=1",
                "--ntasks=1",
                "--time=00:03:00",
                str(executable),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=210,
            env=environment,
        )
        assert run_result.returncode == 0, run_result.stdout + run_result.stderr
        payload = json.loads(run_result.stdout.strip().splitlines()[-1])
        assert payload["maximum_force_error"] <= (
            2.0e-10 * max(1.0, payload["maximum_force"])
        )
        return payload

    direct = compile_and_run("rys3", direct_plan)
    production = compile_and_run("component_lanes", production_plan)
    print(
        json.dumps(
            {
                "rys3": direct,
                "component_lanes": production,
                "speedup_vs_component_lanes": (
                    production["fused_ms"] / direct["fused_ms"]
                ),
            },
            sort_keys=True,
        )
    )


def test_dppp_cooperative_rys4_benchmark_runs_against_component_lanes_when_nvcc_is_configured(
    tmp_path: Path,
):
    """Measure 192-lane cooperative Rys4 against its subset-Wick predecessor."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA benchmark gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    rys4_schedule = ScheduleIR(
        kind=ScheduleKind.COMPONENT_LANES,
        block_threads=192,
        component_tile=DPPP_SPEC.component_count,
        tasks_per_warp=1,
        shared_coulomb=True,
        pair_orientation=PairOrientation.SWAPPED,
        pair_storage=PairStorage.MATERIALIZED,
        unroll_pair_terms=True,
        minimum_blocks_per_sm=2,
    )
    rys4_plan = build_fused_shell_plan(
        DPPP_SPEC,
        schedule=rys4_schedule,
        recurrence="rys4",
    )
    baseline_plan = build_fused_shell_plan(
        DPPP_SPEC,
        schedule=rys4_schedule,
        recurrence="subset_wick",
    )
    environment = dict(os.environ)

    def compile_and_run(label: str, plan) -> dict[str, object]:
        source = tmp_path / f"generated_dppp_{label}_benchmark.cu"
        source.write_text(
            emit_shell_class_benchmark_cuda(
                DPPP_SPEC,
                task_count=8192,
                primitive_count=3,
                warmups=1,
                iterations=3,
                samples=3,
                plan=plan,
                benchmark_kernel_only=True,
                persistent_kernel=True,
            ),
            encoding="utf-8",
        )
        executable = tmp_path / f"generated_dppp_{label}_benchmark"
        compiled = subprocess.run(
            [
                nvcc,
                "-std=c++17",
                f"-arch={cuda_architecture}",
                "-O3",
                str(source),
                "-o",
                str(executable),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert compiled.returncode == 0, compiled.stdout + compiled.stderr
        run = subprocess.run(
            [
                "srun",
                "--partition=main",
                "--gres=gpu:5090:1",
                "--nodes=1",
                "--ntasks=1",
                "--time=00:05:00",
                str(executable),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=330,
            env=environment,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        payload = json.loads(run.stdout.strip().splitlines()[-1])
        assert payload["maximum_force_error"] <= (
            2.0e-10 * max(1.0, payload["maximum_force"])
        )
        return payload

    rys4 = compile_and_run("cooperative_rys4", rys4_plan)
    baseline = compile_and_run("component_lanes", baseline_plan)
    result = {
        "cooperative_rys4": rys4,
        "component_lanes": baseline,
        "speedup_vs_component_lanes": (baseline["fused_ms"] / rys4["fused_ms"]),
    }
    print(json.dumps(result, sort_keys=True))
    assert result["speedup_vs_component_lanes"] > 1.0


def test_dppp_uniform_warp_rys4_benchmark_runs_against_component_lanes_when_nvcc_is_configured(
    tmp_path: Path,
):
    """Compare the 32-task mapping with the previously accepted force path."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA benchmark gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    uniform_schedule = ScheduleIR(
        kind=ScheduleKind.SUBGROUP_TASKS,
        block_threads=256,
        component_tile=DPPP_SPEC.component_count,
        tasks_per_warp=4,
        shared_coulomb=True,
        pair_orientation=PairOrientation.SWAPPED,
        pair_storage=PairStorage.MATERIALIZED,
        unroll_pair_terms=True,
        minimum_blocks_per_sm=1,
    )
    uniform_plan = build_fused_shell_plan(
        DPPP_SPEC,
        schedule=uniform_schedule,
        recurrence="rys4",
    )
    # Keep the comparison independent of the mutable production manifest.  If
    # the candidate is promoted, loading the manifest here would silently
    # benchmark the new kernel against itself and erase the rejection signal.
    component_lane_schedule = ScheduleIR(
        kind=ScheduleKind.COMPONENT_LANES,
        block_threads=192,
        component_tile=DPPP_SPEC.component_count,
        tasks_per_warp=1,
        shared_coulomb=True,
        pair_orientation=PairOrientation.SWAPPED,
        pair_storage=PairStorage.MATERIALIZED,
        unroll_pair_terms=True,
        minimum_blocks_per_sm=2,
    )
    component_lane_plan = build_fused_shell_plan(
        DPPP_SPEC,
        schedule=component_lane_schedule,
        recurrence="rys4",
    )
    environment = dict(os.environ)

    def compile_and_run(label: str, plan) -> dict[str, object]:
        source = tmp_path / f"generated_dppp_{label}_benchmark.cu"
        source.write_text(
            emit_shell_class_benchmark_cuda(
                DPPP_SPEC,
                task_count=8192,
                primitive_count=3,
                warmups=1,
                iterations=3,
                samples=3,
                plan=plan,
                benchmark_kernel_only=True,
                persistent_kernel=True,
            ),
            encoding="utf-8",
        )
        executable = tmp_path / f"generated_dppp_{label}_benchmark"
        compiled = subprocess.run(
            [
                nvcc,
                "-std=c++17",
                f"-arch={cuda_architecture}",
                "-O3",
                str(source),
                "-o",
                str(executable),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert compiled.returncode == 0, compiled.stdout + compiled.stderr
        run = subprocess.run(
            [
                "srun",
                "--partition=main",
                "--gres=gpu:5090:1",
                "--nodes=1",
                "--ntasks=1",
                "--time=00:05:00",
                str(executable),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=330,
            env=environment,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        payload = json.loads(run.stdout.strip().splitlines()[-1])
        assert payload["maximum_force_error"] <= (
            2.0e-10 * max(1.0, payload["maximum_force"])
        )
        return payload

    uniform = compile_and_run("uniform_warp_rys4", uniform_plan)
    component_lanes = compile_and_run("component_lane_rys4", component_lane_plan)
    result = {
        "uniform_warp_rys4": uniform,
        "component_lane_rys4": component_lanes,
        "speedup_vs_component_lanes": (
            component_lanes["fused_ms"] / uniform["fused_ms"]
        ),
    }
    print(json.dumps(result, sort_keys=True))
    assert result["speedup_vs_component_lanes"] > 1.0


@pytest.mark.parametrize("name", ("dpps", "dpss", "dsps", "dspp", "pppp"))
def test_cooperative_rys3_benchmark_runs_against_component_lanes_when_nvcc_is_configured(
    tmp_path: Path,
    name: str,
):
    """Gate each promoted Rys3 class against its accepted force recurrence."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA benchmark gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    selection = next(
        selection
        for selection in load_production_kernel_selections(
            REPOSITORY_ROOT
            / "tools"
            / "vibeqc_codegen"
            / "production_shell_classes.json",
            "sm_120",
        )
        if selection.spec == spec
    )
    rys3_plan = build_fused_shell_plan(
        spec,
        schedule=selection.schedule,
        recurrence="rys3",
    )
    baseline_plan = build_fused_shell_plan(
        spec,
        schedule=selection.schedule,
        recurrence="subset_wick",
    )
    environment = dict(os.environ)

    def compile_and_run(label: str, plan) -> dict[str, object]:
        source = tmp_path / f"generated_{name}_{label}_benchmark.cu"
        source.write_text(
            emit_shell_class_benchmark_cuda(
                spec,
                task_count=8192,
                primitive_count=3,
                warmups=1,
                iterations=3,
                samples=3,
                plan=plan,
                benchmark_kernel_only=True,
                persistent_kernel=True,
            ),
            encoding="utf-8",
        )
        executable = tmp_path / f"generated_{name}_{label}_benchmark"
        compiled = subprocess.run(
            [
                nvcc,
                "-std=c++17",
                f"-arch={cuda_architecture}",
                "-O3",
                str(source),
                "-o",
                str(executable),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert compiled.returncode == 0, compiled.stdout + compiled.stderr
        run = subprocess.run(
            [
                "srun",
                "--partition=main",
                "--gres=gpu:5090:1",
                "--nodes=1",
                "--ntasks=1",
                "--time=00:05:00",
                str(executable),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=330,
            env=environment,
        )
        assert run.returncode == 0, run.stdout + run.stderr
        payload = json.loads(run.stdout.strip().splitlines()[-1])
        assert payload["maximum_force_error"] <= (
            2.0e-10 * max(1.0, payload["maximum_force"])
        )
        return payload

    rys3 = compile_and_run("cooperative_rys3", rys3_plan)
    baseline = compile_and_run("component_lanes", baseline_plan)
    result = {
        "shell_class": name,
        "cooperative_rys3": rys3,
        "component_lanes": baseline,
        "speedup_vs_component_lanes": baseline["fused_ms"] / rys3["fused_ms"],
    }
    print(json.dumps(result, sort_keys=True))
    assert result["speedup_vs_component_lanes"] > 1.0


@pytest.mark.parametrize(
    ("spec", "resource_limits"),
    (
        (DPPP_SPEC, RTX5090_DPPP_RESOURCE_LIMITS),
        (DPDS_SPEC, RTX5090_DPDS_RESOURCE_LIMITS),
        (DDPS_SPEC, RTX5090_DDPS_RESOURCE_LIMITS),
    ),
)
def test_fused_cuda_compiles_when_nvcc_is_configured(
    tmp_path: Path, spec, resource_limits
):
    """Compile every generated shell class for explicit resource probes."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    source = tmp_path / f"generated_{spec.name}_fused.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(spec),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-cubin",
            "-Xptxas=-v",
            str(source),
            "-o",
            str(tmp_path / f"generated_{spec.name}_fused.cubin"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if os.environ.get("VIBEQC_NVCC_VERBOSE"):
        print(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    if cuda_architecture == "sm_120" and resource_limits is not None:
        assert_rtx5090_resources(result.stdout + result.stderr, resource_limits)


def test_ppps_scalar_thread_cuda_compiles_without_spills_when_nvcc_is_configured(
    tmp_path: Path,
):
    """Gate the scalar ppps prototype before any production routing."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    spec = FUSED_SHELL_SPEC_BY_NAME["ppps"]
    schedule = ScheduleIR(
        kind=ScheduleKind.THREAD_TASKS,
        block_threads=32,
        component_tile=spec.component_count,
        tasks_per_warp=32,
        shared_coulomb=False,
        minimum_blocks_per_sm=8,
    )
    source = tmp_path / "generated_ppps_scalar_thread.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(
            spec,
            build_fused_shell_plan(spec, schedule=schedule),
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-cubin",
            "-Xptxas=-v",
            str(source),
            "-o",
            str(tmp_path / "generated_ppps_scalar_thread.cubin"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    if os.environ.get("VIBEQC_NVCC_VERBOSE"):
        print(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    if cuda_architecture == "sm_120":
        assert_rtx5090_resources(
            result.stdout + result.stderr,
            RTX5090_PPPS_SCALAR_THREAD_RESOURCE_LIMITS,
        )
        resource_records = re.findall(
            r"(\d+) bytes stack frame, (\d+) bytes spill stores, "
            r"(\d+) bytes spill loads",
            result.stdout + result.stderr,
        )
        assert resource_records
        assert all(tuple(map(int, record)) == (0, 0, 0) for record in resource_records)


def test_ppps_scalar_thread_benchmark_runs_when_nvcc_is_configured(
    tmp_path: Path,
):
    """Execute the scalar persistent worker against the component oracle."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA benchmark gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    spec = FUSED_SHELL_SPEC_BY_NAME["ppps"]
    schedule = ScheduleIR(
        kind=ScheduleKind.THREAD_TASKS,
        block_threads=32,
        component_tile=spec.component_count,
        tasks_per_warp=32,
        shared_coulomb=False,
        minimum_blocks_per_sm=8,
    )
    environment = dict(os.environ)
    if environment.get("CUDA_VISIBLE_DEVICES") == "":
        environment.pop("CUDA_VISIBLE_DEVICES")

    def compile_and_run(
        label: str,
        selected_schedule: ScheduleIR,
    ) -> dict[str, object]:
        source = tmp_path / f"generated_ppps_{label}_benchmark.cu"
        source.write_text(
            emit_shell_class_benchmark_cuda(
                spec,
                task_count=512,
                primitive_count=2,
                warmups=1,
                iterations=3,
                samples=3,
                schedule=selected_schedule,
                benchmark_kernel_only=True,
                persistent_kernel=True,
            ),
            encoding="utf-8",
        )
        executable = tmp_path / f"generated_ppps_{label}_benchmark"
        compile_result = subprocess.run(
            [
                nvcc,
                "-std=c++17",
                f"-arch={cuda_architecture}",
                "-O3",
                str(source),
                "-o",
                str(executable),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=240,
        )
        assert compile_result.returncode == 0, (
            compile_result.stdout + compile_result.stderr
        )
        run_result = subprocess.run(
            [str(executable)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
            env=environment,
        )
        assert run_result.returncode == 0, run_result.stdout + run_result.stderr
        payload = json.loads(run_result.stdout.strip().splitlines()[-1])
        assert payload["consumer"] == "force"
        assert payload["topology"] == "persistent_shared"
        assert payload["maximum_force_error"] <= (
            2.0e-10 * max(1.0, payload["maximum_force"])
        )
        return payload

    production_schedule = next(
        selection.schedule
        for selection in load_production_kernel_selections(
            REPOSITORY_ROOT
            / "tools"
            / "vibeqc_codegen"
            / "production_shell_classes.json",
            "sm_120",
        )
        if selection.spec == spec
    )
    scalar_payload = compile_and_run("scalar_thread", schedule)
    production_payload = compile_and_run(
        "component_lanes",
        production_schedule,
    )
    print(
        json.dumps(
            {
                "scalar_thread_ms": scalar_payload["fused_ms"],
                "component_lanes_ms": production_payload["fused_ms"],
                "speedup_vs_component_lanes": (
                    production_payload["fused_ms"] / scalar_payload["fused_ms"]
                ),
            },
            sort_keys=True,
        )
    )


def test_joint_fock_force_cuda_compiles_when_nvcc_is_configured(tmp_path: Path):
    """Compile the dual-consumer pilot through the real CUDA frontend."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    plan = build_fused_shell_plan(
        DPDS_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
    )
    source = tmp_path / "generated_dpds_fock_force.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(DPDS_SPEC, plan),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-cubin",
            str(source),
            "-o",
            str(tmp_path / "generated_dpds_fock_force.cubin"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_tiled_joint_fock_force_cuda_compiles_when_nvcc_is_configured(
    tmp_path: Path,
):
    """Compile the tiled dual-consumer lowering through the real frontend."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    integral = build_integral_ir(
        DPPP_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
    )
    schedule = next(
        item
        for item in schedule_candidates(integral)
        if item.kind == ScheduleKind.TILED_COMPONENTS and item.component_tile == 64
    )
    plan = build_fused_shell_plan(
        DPPP_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=schedule,
    )
    source = tmp_path / "generated_dppp_tiled_fock_force.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(DPPP_SPEC, plan),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-cubin",
            str(source),
            "-o",
            str(tmp_path / "generated_dppp_tiled_fock_force.cubin"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_dddd_tiled_cuda_compiles_when_nvcc_is_configured(tmp_path: Path):
    """Compile a 1296-component class that cannot use one lane per quartet."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    schedule = replace(
        build_fused_shell_plan(DDDD_SPEC).schedule,
        block_threads=128,
        component_tile=128,
        pair_orientation=PairOrientation.SWAPPED,
        pair_storage=PairStorage.RECOMPUTED,
        unroll_pair_terms=True,
    )
    plan = build_fused_shell_plan(DDDD_SPEC, schedule=schedule)
    source = tmp_path / "generated_dddd_tiled.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(DDDD_SPEC, plan),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-cubin",
            "-Xptxas=-v",
            str(source),
            "-o",
            str(tmp_path / "generated_dddd_tiled.cubin"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("spec", "consumers"),
    (
        (FFPS_SPEC, (KernelConsumer.FOCK, KernelConsumer.FORCE)),
        (FDDD_SPEC, (KernelConsumer.FORCE,)),
    ),
)
def test_f_shell_cuda_compiles_when_nvcc_is_configured(tmp_path: Path, spec, consumers):
    """Compile pair-order-six and tiled f-shell gradients with CUDA 12.9."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    schedule = replace(build_fused_shell_plan(spec).schedule, unroll_pair_terms=False)
    if spec == FDDD_SPEC:
        schedule = replace(
            schedule,
            block_threads=128,
            component_tile=128,
            pair_storage=PairStorage.RECOMPUTED,
        )
    plan = build_fused_shell_plan(
        spec,
        consumers=consumers,
        schedule=schedule,
    )
    source = tmp_path / f"generated_{spec.name}.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(spec, plan),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-cubin",
            "-Xptxas=-v",
            str(source),
            "-o",
            str(tmp_path / f"generated_{spec.name}.cubin"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("name", "recurrence", "schedule"),
    (
        (
            "fsss",
            "rys3",
            ScheduleIR(
                kind=ScheduleKind.THREAD_TASKS,
                block_threads=32,
                component_tile=10,
                tasks_per_warp=32,
                shared_coulomb=False,
                minimum_blocks_per_sm=1,
            ),
        ),
        (
            "fpps",
            "rys4",
            ScheduleIR(
                kind=ScheduleKind.SUBGROUP_TASKS,
                block_threads=256,
                component_tile=90,
                tasks_per_warp=4,
                shared_coulomb=True,
                minimum_blocks_per_sm=1,
            ),
        ),
    ),
)
def test_structural_rys_capability_examples_compile_when_nvcc_is_configured(
    tmp_path: Path,
    name: str,
    recurrence: str,
    schedule: ScheduleIR,
):
    """Compile f-shell candidates admitted without shell-name allowlists."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    plan = build_fused_shell_plan(
        spec,
        schedule=schedule,
        recurrence=recurrence,
    )
    source = tmp_path / f"generated_{name}_{recurrence}_capability.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(spec, plan),
        encoding="utf-8",
    )
    cubin = tmp_path / f"generated_{name}_{recurrence}_capability.cubin"
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-O3",
            "-cubin",
            "-Xptxas=-v",
            str(source),
            "-o",
            str(cubin),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if os.environ.get("VIBEQC_NVCC_VERBOSE"):
        print(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert cubin.exists()
    if cuda_architecture == "sm_120":
        resource_records = re.findall(
            r"\d+ bytes stack frame, (\d+) bytes spill stores, "
            r"(\d+) bytes spill loads",
            result.stdout + result.stderr,
        )
        assert resource_records
        assert all(tuple(map(int, record)) == (0, 0) for record in resource_records)


def test_psss_shell_task_cuda_compiles_when_nvcc_is_configured(tmp_path: Path):
    """Compile a zero-order ket pair through generated Fock/force lowering."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    integral = build_integral_ir(
        PSSS_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
    )
    schedule = next(
        item
        for item in schedule_candidates(integral)
        if item.kind == ScheduleKind.SHELL_TASK
    )
    plan = build_fused_shell_plan(
        PSSS_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=schedule,
    )
    source = tmp_path / "generated_psss_shell_task.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(PSSS_SPEC, plan),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-cubin",
            str(source),
            "-o",
            str(tmp_path / "generated_psss_shell_task.cubin"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_psss_packed_cuda_compiles_when_nvcc_is_configured(tmp_path: Path):
    """Compile 32 independent low-order tasks per warp for Fock and force."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    integral = build_integral_ir(
        PSSS_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
    )
    schedule = next(
        item
        for item in schedule_candidates(integral)
        if item.kind == ScheduleKind.PACKED_TASKS
    )
    plan = build_fused_shell_plan(
        PSSS_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=schedule,
    )
    source = tmp_path / "generated_psss_packed.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(PSSS_SPEC, plan),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-cubin",
            "-Xptxas=-v",
            str(source),
            "-o",
            str(tmp_path / "generated_psss_packed.cubin"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("name", "resource_limits"),
    (
        ("psps", RTX5090_PSPS_RESOURCE_LIMITS),
        ("ppss", RTX5090_PPSS_RESOURCE_LIMITS),
    ),
)
def test_low_order_production_rys2_cuda_compiles_when_nvcc_is_configured(
    tmp_path: Path,
    name: str,
    resource_limits: dict[str, tuple[int, int, int]],
):
    """Compile the common production Rys2 source and reject spills."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    selection = next(
        item
        for item in load_production_kernel_selections(
            REPOSITORY_ROOT
            / "tools"
            / "vibeqc_codegen"
            / "production_shell_classes.json",
            "sm_120",
        )
        if item.spec.name == name
    )
    plan = build_fused_shell_plan(
        selection.spec,
        consumers=selection.consumers,
        schedule=selection.schedule,
        recurrence=selection.recurrence,
    )
    source = tmp_path / f"generated_{name}_production_rys2.cu"
    source.write_text(
        _PRODUCTION_PRELUDE.replace('#include "scf/generated_shell_task.hpp"\n', "")
        + "#include <cstddef>\n#include <cstdint>\n"
        + emit_shell_class_fused_cuda(
            selection.spec,
            plan,
            fock_schedule=selection.fock_schedule,
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-cubin",
            "-Xptxas=-v",
            str(source),
            "-o",
            str(tmp_path / f"generated_{name}_production_rys2.cubin"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if os.environ.get("VIBEQC_NVCC_VERBOSE"):
        print(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    if cuda_architecture == "sm_120":
        assert_rtx5090_resources(result.stdout + result.stderr, resource_limits)


@pytest.mark.parametrize("shared_coulomb", (True, False))
def test_schedule_knob_cuda_variants_compile_when_nvcc_is_configured(
    tmp_path: Path, shared_coulomb: bool
):
    """Compile both cooperative sharing policies through the real frontend."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    base = build_fused_shell_plan(DPDS_SPEC).schedule
    schedule = replace(
        base,
        shared_coulomb=shared_coulomb,
        unroll_pair_terms=False,
    )
    plan = build_fused_shell_plan(DPDS_SPEC, schedule=schedule)
    label = "shared" if shared_coulomb else "recomputed"
    source = tmp_path / f"generated_dpds_{label}.cu"
    source.write_text(
        """
template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) {
    values[order] = 1.0 / (2.0 * static_cast<double>(order) + 1.0 + argument);
  }
}
"""
        + emit_shell_class_fused_cuda(DPDS_SPEC, plan),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-cubin",
            str(source),
            "-o",
            str(tmp_path / f"generated_dpds_{label}.cubin"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_dppp_benchmark_compares_shared_and_recomputed_schedules():
    source = emit_dppp_benchmark_cuda(
        task_count=32,
        primitive_count=2,
        warmups=1,
        iterations=3,
        samples=5,
    )
    assert "constexpr unsigned kTaskCount = 32;" in source
    assert "constexpr unsigned kPrimitiveCount = 2;" in source
    assert "generated_dppp_component_gradient<true>" in source
    assert "generated_dppp_component_gradient<false>" in source
    assert "generated_dppp_component_recompute_rhf_kernel" in source
    assert "generated_dppp_make_primitive_geometry_uncached" in source
    assert "device_primitive_pairs" in source
    assert '\\"speedup\\"' in source


def test_fock_benchmark_compares_value_only_shared_and_recomputed_schedules():
    """Benchmark the SCF hot consumer with an independent value oracle."""

    source = emit_shell_class_benchmark_cuda(
        DPDS_SPEC,
        task_count=32,
        primitive_count=2,
        warmups=1,
        iterations=3,
        samples=5,
        consumer=KernelConsumer.FOCK,
    )
    assert "generated_dpds_shell_class_fock_rhf_kernel" in source
    assert "generated_dpds_component_recompute_fock_rhf_kernel" in source
    assert "generated_dpds_component_value<false>" in source
    assert (
        "generated_dpds_component_gradient<false>"
        not in source.split("/** Per-component Fock baseline", maxsplit=1)[1]
    )
    assert '\\"consumer\\":\\"fock\\"' in source
    assert '\\"maximum_fock_error\\"' in source


def test_packed_order2_fock_oracle_drops_force_wrappers():
    """Keep packed low-order schedules available to Fock autotuning."""

    trial = next(
        trial
        for trial in supported_schedule_trials(PSPS_SPEC, KernelConsumer.FOCK)
        if trial.schedule.kind == ScheduleKind.PACKED_TASKS
    )
    plan = build_fused_shell_plan(
        PSPS_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=trial.schedule,
    )
    source = emit_shell_class_oracle_cuda(PSPS_SPEC, plan, KernelConsumer.FOCK)
    assert "generated_psps_shell_class_fock_rhf_kernel" in source
    assert "generated_psps_shell_class_force_rhf_kernel" not in source


def test_fock_benchmark_runs_when_nvcc_is_configured(tmp_path: Path):
    """Execute the swapped value benchmark and its independent oracle."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA benchmark gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    schedule = replace(
        build_fused_shell_plan(DPDS_SPEC).schedule,
        pair_orientation=PairOrientation.SWAPPED,
    )
    source = tmp_path / "generated_dpds_fock_benchmark.cu"
    source.write_text(
        emit_shell_class_benchmark_cuda(
            DPDS_SPEC,
            task_count=2,
            primitive_count=1,
            warmups=0,
            iterations=1,
            samples=1,
            consumer=KernelConsumer.FOCK,
            schedule=schedule,
        ),
        encoding="utf-8",
    )
    executable = tmp_path / "generated_dpds_fock_benchmark"
    compile_result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-O3",
            str(source),
            "-o",
            str(executable),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr
    environment = dict(os.environ)
    if environment.get("CUDA_VISIBLE_DEVICES") == "":
        environment.pop("CUDA_VISIBLE_DEVICES")
    run_result = subprocess.run(
        [str(executable)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env=environment,
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
    payload = json.loads(run_result.stdout.strip().splitlines()[-1])
    assert payload["consumer"] == "fock"
    assert payload["maximum_fock_error"] <= (
        2.0e-10 * max(1.0, payload["maximum_fock"])
    )


def test_benchmark_accepts_an_explicit_schedule_or_lowered_plan():
    """Make the measured code shape an explicit autotuning input."""

    default_plan = build_fused_shell_plan(DPDS_SPEC)
    schedule = replace(
        default_plan.schedule,
        shared_coulomb=False,
        unroll_pair_terms=False,
    )
    source = emit_shell_class_benchmark_cuda(
        DPDS_SPEC,
        task_count=4,
        primitive_count=1,
        warmups=0,
        iterations=1,
        samples=1,
        schedule=schedule,
    )
    assert "double coulomb[1];" in source
    assert source.count("generated_dpds_component_gradient<false>") == 2
    assert "#pragma unroll 1" in source
    assert "#pragma unroll\n" in source
    assert "VIBEQC_PAIR_UNROLL" not in source
    plan = build_fused_shell_plan(DPDS_SPEC, schedule=schedule)
    assert source == emit_shell_class_benchmark_cuda(
        DPDS_SPEC,
        task_count=4,
        primitive_count=1,
        warmups=0,
        iterations=1,
        samples=1,
        plan=plan,
    )
    with pytest.raises(ValueError, match="either a fused plan or a schedule"):
        emit_shell_class_benchmark_cuda(
            DPDS_SPEC,
            task_count=1,
            primitive_count=1,
            warmups=0,
            iterations=1,
            samples=1,
            plan=plan,
            schedule=schedule,
        )


def test_autotune_compile_timeout_terminates_the_compiler_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Reject pathological large-shell compiles without orphaning NVCC children."""

    trial = supported_schedule_trials(DPDS_SPEC)[0]
    source = tmp_path / f"{trial.spec.name}_{trial.schedule_id}.cu"
    source.write_text("// fake CUDA input\n", encoding="utf-8")
    child_pid_file = tmp_path / "child.pid"
    fake_nvcc = tmp_path / "fake-nvcc"
    fake_nvcc.write_text(
        """#!/bin/sh
sleep 60 &
child_pid=$!
printf '%s\n' "$child_pid" > "$VIBEQC_TEST_CHILD_PID_FILE"
wait "$child_pid"
""",
        encoding="utf-8",
    )
    fake_nvcc.chmod(0o755)
    monkeypatch.setenv("VIBEQC_TEST_CHILD_PID_FILE", str(child_pid_file))

    row = _compile_trial(
        fake_nvcc,
        "sm_120",
        tmp_path,
        trial,
        compile_timeout=0.1,
    )
    assert row["timed_out"] is True
    assert row["returncode"] == 124
    assert "timed out after 0.1 seconds" in row["diagnostics"]

    child_pid = int(child_pid_file.read_text(encoding="utf-8"))
    child_proc = Path(f"/proc/{child_pid}")
    deadline = time.monotonic() + 2.0
    while child_proc.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not child_proc.exists()


def test_autotune_driver_probes_and_rejects_target_before_trials():
    """Fail an architecture/device mismatch before any benchmark entry runs."""

    target = cuda_target_info("sm_80")
    trial = supported_schedule_trials(DPDS_SPEC, target=target)[0]
    source = emit_schedule_driver((trial,), "sm_80")
    probe = source.index(r"\"target_probe\"")
    mismatch = source.index("properties.major != 8")
    trial_call = source.index(trial.entry_point, source.index("int failures = 0"))
    assert probe < mismatch < trial_call
    assert "maximum_blocks_per_sm" in source
    assert "compile target sm_80 does not match allocated" in source


def test_autotune_keeps_benchmark_executor_distinct_from_compile_pool():
    """Prevent parallel compilation from shadowing the GPU run adapter."""

    source = (REPOSITORY_ROOT / "tools" / "vibeqc_codegen" / "autotune.py").read_text(
        encoding="utf-8"
    )
    assert "benchmark_executor = CudaBenchmarkExecutor(" in source
    assert "as compile_pool:" in source
    assert "run = benchmark_executor.run(" in source
    assert "as executor:" not in source


def test_autotune_emits_unique_schedule_variants_and_manifest_records():
    """Keep same-class variants linkable and every winner reproducible."""

    trials = supported_schedule_trials(DPDS_SPEC)
    component_trials = tuple(
        trial for trial in trials if trial.schedule.kind == ScheduleKind.COMPONENT_LANES
    )
    assert len(component_trials) == 8
    assert any(
        trial.schedule.kind == ScheduleKind.TILED_COMPONENTS
        and trial.schedule.component_tile == 64
        for trial in trials
    )
    assert {
        (
            trial.schedule.pair_orientation,
            trial.schedule.shared_coulomb,
            trial.schedule.unroll_pair_terms,
        )
        for trial in component_trials
    } == {
        (orientation, shared, unrolled)
        for orientation in PairOrientation
        for shared in (True, False)
        for unrolled in (True, False)
    }
    sources = [
        emit_schedule_translation_unit(
            trial,
            task_count=2,
            primitive_count=1,
            warmups=0,
            iterations=1,
            samples=1,
        )
        for trial in component_trials[:2]
    ]
    for trial, source in zip(component_trials, sources, strict=False):
        assert trial.entry_point in source
        assert trial.symbol_prefix in source
        assert f'\\"schedule_id\\":\\"{trial.schedule_id}\\"' in source
        assert "shell_class_force_uhf_kernel" not in source
        assert "shell_class_force_rhf_kernel" not in source
        assert "shell_class_force_rhf_persistent_kernel" in source
        assert '\\"topology\\":\\"persistent_shared\\"' in source
        assert "device_task_head" in source
    assert component_trials[0].symbol_prefix not in sources[1]
    separate_source = emit_schedule_translation_unit(
        component_trials[0],
        task_count=2,
        primitive_count=1,
        warmups=0,
        iterations=1,
        samples=1,
        oracle_trial=component_trials[0],
    )
    oracle_source = emit_schedule_oracle_translation_unit(component_trials[0])
    oracle_kernel = (
        "vibeqc_oracle_dpds_force_component_lanes_b128_t108_w1_"
        "component_recompute_rhf_kernel"
    )
    assert separate_source.count(oracle_kernel) == 2
    assert oracle_source.count(oracle_kernel) == 1
    assert "Per-component recurrence baseline" not in separate_source
    assert "Per-component recurrence baseline" in oracle_source
    assert "center < 2U ? 1024U : 16U" in separate_source
    assert "const std::size_t force_count = 24U * 3U" in separate_source
    driver = emit_schedule_driver(component_trials[:2])
    assert component_trials[0].entry_point in driver
    assert component_trials[1].entry_point in driver

    manifest = {
        "schema_version": 2,
        "default_architecture": "sm_120",
        "architectures": {
            "sm_120": {
                "kernels": [
                    {
                        "shell_class": "dpds",
                        "consumers": ["force"],
                        "schedule": {},
                    }
                ]
            }
        },
    }
    updated = update_manifest_payload(
        manifest,
        "sm_120",
        {
            "dpds": component_trials[0].schedule,
            "ddps": component_trials[1].schedule,
        },
    )
    kernels = updated["architectures"]["sm_120"]["kernels"]
    assert kernels[0]["schedule"]["shared_coulomb"] is True
    assert kernels[0]["schedule"]["unroll_pair_terms"] is True
    assert kernels[0]["schedule"]["pair_storage"] == "materialized"
    assert kernels[1]["shell_class"] == "ddps"
    assert kernels[1]["consumers"] == ["force"]

    fock_trials = supported_schedule_trials(DPDS_SPEC, KernelConsumer.FOCK)
    assert all(trial.consumer == KernelConsumer.FOCK for trial in fock_trials)
    fock_source = emit_schedule_translation_unit(
        fock_trials[0],
        task_count=2,
        primitive_count=1,
        warmups=0,
        iterations=1,
        samples=1,
    )
    assert fock_trials[0].entry_point in fock_source
    assert fock_trials[0].symbol_prefix in fock_source
    assert '\\"consumer\\":\\"fock\\"' in fock_source
    assert "shell_class_force_task" not in fock_source
    assert "shell_class_fock_uhf_kernel" not in fock_source
    assert "shell_class_fock_rhf_kernel" not in fock_source
    assert "shell_class_fock_rhf_persistent_kernel" in fock_source
    assert "task.density_offset = 0U" in fock_source
    resource_source = emit_schedule_resource_translation_unit(fock_trials[0])
    assert "shell_class_fock_uhf_kernel" in resource_source
    assert "shell_class_fock_rhf_persistent_kernel" in resource_source

    fock_manifest = {
        "schema_version": 2,
        "default_architecture": "sm_120",
        "architectures": {
            "sm_120": {
                "kernels": [
                    {
                        "shell_class": "dpds",
                        "consumers": ["force"],
                        "schedule": {},
                    }
                ]
            }
        },
    }
    fock_updated = update_manifest_payload(
        fock_manifest,
        "sm_120",
        {"dpds": fock_trials[0].schedule},
        KernelConsumer.FOCK,
    )
    assert fock_updated["architectures"]["sm_120"]["kernels"][0]["consumers"] == [
        "fock",
        "force",
    ]


def test_autotune_analysis_roots_follow_declared_derivative_centers():
    """Exclude recovered centers from the static force root envelope."""

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
    kernel = build_shell_class_contraction_kernel(
        DPPP_SPEC,
        DPPP_SPEC.components[0],
    )
    roots = _analysis_roots(kernel, KernelConsumer.FORCE, integral=integral)
    expected = tuple(
        kernel.gradients[center][coordinate]
        for center in (0, 2, 3)
        for coordinate in range(3)
    )
    assert roots == expected


def test_autotune_trials_preserve_an_explicit_integral_ir():
    """Route non-final translation recovery through schedule trial emission."""

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
        DPDS_SPEC,
        operator=operator,
        derivative=operator.nuclear_derivative(),
        contractions=(force,),
    )
    trials = supported_schedule_trials(DPDS_SPEC, integral=integral)
    assert trials
    assert all(trial.integral is integral for trial in trials)
    assert trials[0].static_model.recurrence_state_count == 84

    source = emit_schedule_oracle_translation_unit(trials[0])
    assert "constexpr unsigned derivative_centers[3] = {0U, 2U, 3U};" in source


def test_autotune_trial_identity_includes_explicit_integral_intent():
    """Keep distinct recovery policies from sharing runtime or oracle symbols."""

    default_integral = build_integral_ir(DPDS_SPEC)
    center_one_operator = OperatorSpec(
        family=OperatorFamily.FOUR_CENTER_ERI,
        centers=(0, 1, 2, 3),
        invariants=(TranslationInvariant(dependent_center=1),),
    )
    center_one_integral = build_integral_ir(
        DPDS_SPEC,
        operator=center_one_operator,
        derivative=center_one_operator.nuclear_derivative(),
    )

    default_trial = supported_schedule_trials(
        DPDS_SPEC,
        integral=default_integral,
    )[0]
    center_one_trial = supported_schedule_trials(
        DPDS_SPEC,
        integral=center_one_integral,
    )[0]
    same_default_trial = supported_schedule_trials(
        DPDS_SPEC,
        integral=build_integral_ir(DPDS_SPEC),
    )[0]

    # Both trials use the same execution knobs; only mathematical recovery
    # intent differs, so the schedule ID remains equal while every emitted
    # identity that can index runtime/oracle artifacts stays disjoint.
    assert default_trial.schedule_id == center_one_trial.schedule_id
    assert default_trial.key != center_one_trial.key
    assert default_trial.entry_point != center_one_trial.entry_point
    assert default_trial.symbol_prefix != center_one_trial.symbol_prefix
    assert _oracle_symbol_prefix(default_trial) != _oracle_symbol_prefix(
        center_one_trial
    )
    assert default_trial.integral_suffix == same_default_trial.integral_suffix
    source = emit_schedule_translation_unit(
        center_one_trial,
        task_count=1,
        primitive_count=1,
        warmups=0,
        iterations=1,
        samples=1,
    )
    assert f'\\"trial_key\\":\\"{center_one_trial.key}\\"' in source


def test_autotune_static_model_records_operations_and_live_values():
    """Attach cached symbolic cost envelopes to every schedule candidate."""

    packed_force = next(
        trial
        for trial in supported_schedule_trials(PSPS_SPEC)
        if trial.schedule.kind == ScheduleKind.PACKED_TASKS
    )
    packed_model = packed_force.static_model
    assert isinstance(packed_model, StaticAlgebraModel)
    assert packed_model is static_algebra_model(packed_force)
    assert packed_model.scope == "weighted_shell_dag"
    assert packed_model.algebra_form == AlgebraForm.BINARY
    assert packed_model.algebra_fusion == AlgebraFusion.SEPARATE
    assert packed_model.algebra_placement == AlgebraPlacement.MATERIALIZED_CSE
    assert packed_model.component_count == 9
    assert packed_model.sampled_component_count == 9
    assert packed_model.recurrence_state_count == 20
    assert packed_model.root_count == 9
    assert packed_model.arithmetic_operation_count == (
        packed_model.materialized_value_count
    )
    assert packed_model.baseline_arithmetic_operation_count == (
        packed_model.arithmetic_operation_count
    )
    assert packed_model.baseline_materialized_value_count == (
        packed_model.materialized_value_count
    )
    assert packed_model.baseline_peak_live_values == packed_model.peak_live_values
    assert 0 < packed_model.peak_live_values < packed_model.materialized_value_count
    geometry_analysis, geometry_plan = _packed_force_geometry_analysis(3)
    assert dict(packed_model.operation_counts)["power"] >= (
        dict(geometry_analysis.operation_counts)["power"]
    )
    assert packed_model.arithmetic_operation_count >= (
        geometry_plan.arithmetic_operation_count
    )

    component_trials = tuple(
        trial
        for trial in supported_schedule_trials(DPDS_SPEC)
        if trial.schedule.kind == ScheduleKind.COMPONENT_LANES
    )
    component_model = component_trials[0].static_model
    assert component_model is component_trials[1].static_model
    assert component_model.scope == "balanced_component_sample_envelope"
    assert component_model.component_count == 108
    assert component_model.sampled_component_count == 27
    assert component_model.recurrence_state_count == 84
    assert component_model.root_count == 9
    assert component_model.arithmetic_operation_count > 0
    assert component_model.materialized_value_count > 0
    assert component_model.peak_live_values > 0
    payload = component_model.to_payload()
    assert payload["operation_counts"]["add"] > 0
    assert payload["estimated_peak_live_values"] == component_model.peak_live_values
    assert payload["pre_optimization"] == payload["post_optimization"]

    fock_trial = supported_schedule_trials(DPDS_SPEC, KernelConsumer.FOCK)[0]
    fock_model = fock_trial.static_model
    assert fock_model.root_count == 1
    assert fock_model.recurrence_state_count == 56


def test_packed_autotune_searches_real_algebra_placement_variants():
    """Tie schedule IDs, payloads, source lowering, and static models together."""

    trials = supported_schedule_trials(PSPS_SPEC)
    packed = tuple(
        trial for trial in trials if trial.schedule.kind == ScheduleKind.PACKED_TASKS
    )
    assert {
        (
            trial.schedule.algebra_placement,
            trial.schedule.algebra_ordering,
            trial.schedule.algebra_fusion,
            trial.schedule.algebra_form,
        )
        for trial in packed
    } == set(
        itertools.product(
            AlgebraPlacement,
            AlgebraOrdering,
            AlgebraFusion,
            AlgebraForm,
        )
    )
    assert all(
        trial.schedule.algebra_placement == AlgebraPlacement.MATERIALIZED_CSE
        and trial.schedule.algebra_ordering == AlgebraOrdering.TOPOLOGICAL
        and trial.schedule.algebra_fusion == AlgebraFusion.SEPARATE
        and trial.schedule.algebra_form == AlgebraForm.BINARY
        for trial in trials
        if trial.schedule.kind != ScheduleKind.PACKED_TASKS
    )
    assert len({trial.schedule_id for trial in trials}) == len(trials)
    assert all(
        schedule_payload(trial.schedule)["algebra_placement"]
        == trial.schedule.algebra_placement.value
        and schedule_payload(trial.schedule)["algebra_ordering"]
        == trial.schedule.algebra_ordering.value
        and schedule_payload(trial.schedule)["algebra_fusion"]
        == trial.schedule.algebra_fusion.value
        and schedule_payload(trial.schedule)["algebra_form"]
        == trial.schedule.algebra_form.value
        for trial in packed
    )

    comparable = {
        placement: next(
            trial
            for trial in packed
            if trial.schedule.algebra_placement == placement
            and trial.schedule.algebra_ordering == AlgebraOrdering.TOPOLOGICAL
            and trial.schedule.algebra_fusion == AlgebraFusion.SEPARATE
            and trial.schedule.algebra_form == AlgebraForm.BINARY
            and trial.schedule.minimum_blocks_per_sm == 2
            and trial.schedule.unroll_pair_terms
        )
        for placement in AlgebraPlacement
    }
    baseline = comparable[AlgebraPlacement.MATERIALIZED_CSE].static_model
    single_use = comparable[AlgebraPlacement.INLINE_SINGLE_USE].static_model
    pressure = comparable[AlgebraPlacement.PRESSURE_REMATERIALIZED].static_model
    assert single_use.baseline_arithmetic_operation_count == (
        baseline.arithmetic_operation_count
    )
    assert single_use.arithmetic_operation_count == baseline.arithmetic_operation_count
    assert single_use.materialized_value_count < baseline.materialized_value_count
    assert single_use.peak_live_values < baseline.peak_live_values
    assert pressure.arithmetic_operation_count <= int(
        baseline.arithmetic_operation_count * 1.2
    )
    assert pressure.materialized_value_count < single_use.materialized_value_count
    assert pressure.peak_live_values <= single_use.peak_live_values
    assert pressure.rematerialized_value_count > 0

    baseline_source = emit_schedule_translation_unit(
        comparable[AlgebraPlacement.MATERIALIZED_CSE],
        task_count=1,
        primitive_count=1,
        warmups=0,
        iterations=1,
        samples=1,
    )
    inline_source = emit_schedule_translation_unit(
        comparable[AlgebraPlacement.INLINE_SINGLE_USE],
        task_count=1,
        primitive_count=1,
        warmups=0,
        iterations=1,
        samples=1,
    )
    assert baseline_source != inline_source
    assert baseline_source.count("  const double v") > inline_source.count(
        "  const double v"
    )

    pressure_ordered = {
        placement: next(
            trial.static_model
            for trial in packed
            if trial.schedule.algebra_placement == placement
            and trial.schedule.algebra_ordering == AlgebraOrdering.PRESSURE_AWARE
            and trial.schedule.algebra_fusion == AlgebraFusion.SEPARATE
            and trial.schedule.algebra_form == AlgebraForm.BINARY
            and trial.schedule.minimum_blocks_per_sm == 2
            and trial.schedule.unroll_pair_terms
        )
        for placement in AlgebraPlacement
    }
    for placement, ordered_model in pressure_ordered.items():
        topological_model = comparable[placement].static_model
        assert ordered_model.arithmetic_operation_count == (
            topological_model.arithmetic_operation_count
        )
        assert ordered_model.materialized_value_count == (
            topological_model.materialized_value_count
        )
        assert ordered_model.peak_live_values < topological_model.peak_live_values
        assert ordered_model.reordered_value_count > 0

    fused = {
        placement: next(
            trial
            for trial in packed
            if trial.schedule.algebra_placement == placement
            and trial.schedule.algebra_ordering == AlgebraOrdering.TOPOLOGICAL
            and trial.schedule.algebra_fusion == AlgebraFusion.FMA
            and trial.schedule.algebra_form == AlgebraForm.BINARY
            and trial.schedule.minimum_blocks_per_sm == 2
            and trial.schedule.unroll_pair_terms
        )
        for placement in AlgebraPlacement
    }
    for placement, fused_trial in fused.items():
        separate_model = comparable[placement].static_model
        fused_model = fused_trial.static_model
        assert fused_model.arithmetic_operation_count < (
            separate_model.arithmetic_operation_count
        )
        assert fused_model.fma_operation_count > 0
        assert dict(fused_model.emitted_operation_counts)["fma"] == (
            fused_model.fma_operation_count
        )
    fused_source = emit_schedule_translation_unit(
        fused[AlgebraPlacement.MATERIALIZED_CSE],
        task_count=1,
        primitive_count=1,
        warmups=0,
        iterations=1,
        samples=1,
    )
    assert " = fma(" in fused_source

    forms = {
        form: next(
            trial
            for trial in packed
            if trial.schedule.algebra_placement
            == AlgebraPlacement.MATERIALIZED_CSE
            and trial.schedule.algebra_ordering == AlgebraOrdering.TOPOLOGICAL
            and trial.schedule.algebra_fusion == AlgebraFusion.SEPARATE
            and trial.schedule.algebra_form == form
            and trial.schedule.minimum_blocks_per_sm == 2
            and trial.schedule.unroll_pair_terms
        )
        for form in AlgebraForm
    }
    binary_model = forms[AlgebraForm.BINARY].static_model
    canonical_model = forms[AlgebraForm.CANONICAL_NARY].static_model
    factored_model = forms[AlgebraForm.FACTORED_NARY].static_model
    assert canonical_model.materialized_value_count < (
        binary_model.materialized_value_count
    )
    assert factored_model.arithmetic_operation_count < (
        canonical_model.arithmetic_operation_count
    )
    assert factored_model.peak_live_values < binary_model.peak_live_values
    canonical_source = emit_schedule_translation_unit(
        forms[AlgebraForm.CANONICAL_NARY],
        task_count=1,
        primitive_count=1,
        warmups=0,
        iterations=1,
        samples=1,
    )
    assert any(
        line.count(" + ") >= 2 or line.count(" * ") >= 2
        for line in canonical_source.splitlines()
        if "const double v" in line
    )


def test_algebra_placement_schedule_payload_is_backward_compatible():
    """Round-trip tuned placement while defaulting older manifests safely."""

    inline_schedule = next(
        trial.schedule
        for trial in supported_schedule_trials(PSPS_SPEC)
        if trial.schedule.kind == ScheduleKind.PACKED_TASKS
        and trial.schedule.algebra_placement == AlgebraPlacement.INLINE_SINGLE_USE
        and trial.schedule.algebra_ordering == AlgebraOrdering.PRESSURE_AWARE
        and trial.schedule.algebra_fusion == AlgebraFusion.FMA
        and trial.schedule.algebra_form == AlgebraForm.FACTORED_NARY
    )
    payload = schedule_payload(inline_schedule)
    assert _schedule_from_payload(payload) == inline_schedule

    del payload["algebra_placement"]
    del payload["algebra_ordering"]
    del payload["algebra_fusion"]
    del payload["algebra_form"]
    assert (
        _schedule_from_payload(payload).algebra_placement
        == AlgebraPlacement.MATERIALIZED_CSE
    )
    assert (
        _schedule_from_payload(payload).algebra_ordering
        == AlgebraOrdering.TOPOLOGICAL
    )
    assert (
        _schedule_from_payload(payload).algebra_fusion
        == AlgebraFusion.SEPARATE
    )
    assert _schedule_from_payload(payload).algebra_form == AlgebraForm.BINARY
    with pytest.raises(ValueError, match="packed tasks"):
        replace(
            build_fused_shell_plan(DPDS_SPEC).schedule,
            algebra_placement=AlgebraPlacement.INLINE_SINGLE_USE,
        )
    with pytest.raises(ValueError, match="packed tasks"):
        replace(
            build_fused_shell_plan(DPDS_SPEC).schedule,
            algebra_ordering=AlgebraOrdering.PRESSURE_AWARE,
        )
    with pytest.raises(ValueError, match="packed tasks"):
        replace(
            build_fused_shell_plan(DPDS_SPEC).schedule,
            algebra_fusion=AlgebraFusion.FMA,
        )
    with pytest.raises(ValueError, match="packed tasks"):
        replace(
            build_fused_shell_plan(DPDS_SPEC).schedule,
            algebra_form=AlgebraForm.CANONICAL_NARY,
        )


def test_autotune_candidate_artifact_includes_static_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Persist the static model even when compilation rejects a candidate."""

    trial = next(
        trial
        for trial in supported_schedule_trials(PSPS_SPEC)
        if trial.schedule.kind == ScheduleKind.PACKED_TASKS
    )

    def failed_compile(*args, **kwargs):
        selected_trial = args[3]
        return {
            "key": selected_trial.key,
            "object": tmp_path / "unused.o",
            "returncode": 1,
            "timed_out": False,
            "duration_seconds": 0.01,
            "diagnostics": "synthetic compiler rejection",
            "resources": (),
        }

    monkeypatch.setattr(
        "tools.vibeqc_codegen.autotune.supported_schedule_trials",
        lambda *args, **kwargs: (trial,),
    )
    monkeypatch.setattr(
        "tools.vibeqc_codegen.autotune._compile_trial",
        failed_compile,
    )
    arguments = SimpleNamespace(
        architecture="sm_120",
        nvcc=Path("nvcc"),
        compile_timeout=1.0,
        timeout=1,
        local=True,
        srun="srun",
        partition="main",
        gres="gpu:5090:1",
        slurm_time="00:01:00",
        max_registers=None,
        max_packed_registers=None,
        max_stack_bytes=None,
        max_shared_bytes=None,
        shell_class=["psps"],
        consumer=KernelConsumer.FORCE.value,
        work_directory=tmp_path,
        compile_jobs=1,
        tasks=1,
        primitives=1,
        warmups=0,
        iterations=1,
        samples=1,
        allow_experimental_subgroup_winner=True,
        absolute_tolerance=1.0e-12,
        relative_tolerance=1.0e-12,
        minimum_speedup=1.0,
        verbose=False,
        manifest_output=None,
        manifest=REPOSITORY_ROOT
        / "tools"
        / "vibeqc_codegen"
        / "production_shell_classes.json",
    )

    report = _run_autotune(arguments)
    assert report["winners"] == []
    assert len(report["candidates"]) == 1
    assert report["candidates"][0]["static_model"] == (
        trial.static_model.to_payload()
    )


def test_autotune_same_class_variants_link_when_nvcc_is_configured(
    tmp_path: Path,
):
    """Ensure symbol isolation lets one GPU process compare same-class code."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA link gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    trials = supported_schedule_trials(DPDS_SPEC)[:2]
    sources = []
    for trial in trials:
        path = tmp_path / f"{trial.schedule_id}.cu"
        path.write_text(
            emit_schedule_translation_unit(
                trial,
                task_count=1,
                primitive_count=1,
                warmups=0,
                iterations=1,
                samples=1,
            ),
            encoding="utf-8",
        )
        sources.append(path)
    driver = tmp_path / "driver.cu"
    driver.write_text(emit_schedule_driver(trials), encoding="utf-8")
    result = subprocess.run(
        [
            nvcc,
            "-std=c++17",
            f"-arch={cuda_architecture}",
            "-O3",
            str(driver),
            *(str(path) for path in sources),
            "-o",
            str(tmp_path / "autotune_link_gate"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("spec", "third_offset"),
    ((DPDS_SPEC, 9), (DDPS_SPEC, 12)),
)
def test_benchmark_is_generated_without_shell_specific_harness_code(spec, third_offset):
    source = emit_shell_class_benchmark_cuda(
        spec,
        task_count=32,
        primitive_count=2,
        warmups=1,
        iterations=3,
        samples=5,
    )
    assert "constexpr std::size_t n = 16U" in source
    assert f"task.ao_begin[2] = {third_offset}U" in source
    assert "task.ao_begin[3] = 15U" in source
    assert f"generated_{spec.name}_component_gradient<true>" in source
    assert f"generated_{spec.name}_component_gradient<false>" in source
    assert f"generated_{spec.name}_component_recompute_rhf_kernel" in source
    assert "generated_dppp" not in source


def test_cuda_emission_is_deterministic_and_runtime_ad_free():
    first = emit_psss_cuda(build_psss_kernel("z"))
    second = emit_psss_cuda(build_psss_kernel("z"))
    assert first == second
    assert "boys_values<2>" in first
    assert "generated_psss_z_gradient" in first
    assert "Dual3" not in first


def test_dppp_cuda_emission_is_deterministic_and_runtime_ad_free():
    kernel = build_dppp_component_kernel("xy", tuple("xyz"))
    first = emit_dppp_component_cuda(kernel)
    second = emit_dppp_component_cuda(build_dppp_component_kernel("xy", tuple("xyz")))
    assert first == second
    assert "boys_values<6>" in first
    assert "generated_dppp_xy_xyz_gradient" in first
    assert "Dual3" not in first

    factored = emit_dppp_contraction_cuda(
        build_dppp_contraction_kernel("xy", tuple("xyz"))
    )
    assert "GeneratedDpppGeometry" in factored
    assert "generated_dppp_xy_xyz_factored_gradient" in factored
    assert "boys_values" not in factored
    assert "Dual3" not in factored


def test_dppp_contraction_cuda_honors_explicit_nonfinal_recovery():
    """Pack factored decay rows by IR order when center B is recovered."""

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
    kernel = build_dppp_contraction_kernel(
        "xy",
        tuple("xyz"),
        integral=integral,
    )
    assert kernel.integral is integral
    source = emit_dppp_contraction_cuda(kernel)

    # The ket fourth-center derivative needs the complement of the stored
    # third-center product scale.  Decay rows are dense A/C/D slots, not
    # physical center indices A/B/C/D, so row 1 must remain present while
    # row 3 is not referenced by this three-independent-center geometry ABI.
    assert "1.0 - geometry.product_scales[2]" in source
    assert "geometry.decay_gradients[1]" in source
    assert "geometry.decay_gradients[2]" in source
    assert "geometry.decay_gradients[3]" not in source


def test_nvrtc_cache_key_covers_binary_compatibility_inputs():
    specification = NvrtcCacheSpec(
        generator_abi="1",
        shell_class=(2, 1, 2, 0),
        derivative_centers=(0, 1, 2),
        precision_policy="fp64",
        screening_policy="density-v1",
        source_digest="abc123",
        compute_capability="sm_120",
        nvrtc_version="12.9",
        driver_version="580.95.05",
    )
    key = nvrtc_cache_key(specification)
    assert len(key) == 64
    assert key == nvrtc_cache_key(specification)
    assert key != nvrtc_cache_key(
        replace(specification, precision_policy="mixed-fp32-control")
    )


def test_codegen_cli_writes_deterministic_aot_candidate(tmp_path: Path):
    output = tmp_path / "generated" / "psss_x.cuh"
    command = [
        sys.executable,
        "tools/generate_shell_kernels.py",
        "--shell-class",
        "psss",
        "--axis",
        "x",
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True)
    first = output.read_text(encoding="utf-8")
    subprocess.run(command, check=True)
    assert output.read_text(encoding="utf-8") == first
    assert "generated_psss_x_gradient" in first


def test_codegen_cli_writes_dppp_component_candidate(tmp_path: Path):
    output = tmp_path / "generated" / "dppp_xy_xyz.cuh"
    command = [
        sys.executable,
        "tools/generate_shell_kernels.py",
        "--shell-class",
        "dppp",
        "--d-component",
        "xy",
        "--p-components",
        "xyz",
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True)
    first = output.read_text(encoding="utf-8")
    subprocess.run(command, check=True)
    assert output.read_text(encoding="utf-8") == first
    assert "generated_dppp_xy_xyz_gradient" in first

    factored_output = tmp_path / "generated" / "dppp_xy_xyz_factored.cuh"
    factored_command = [
        *command[:-2],
        "--lowering",
        "factored",
        "--output",
        str(factored_output),
    ]
    subprocess.run(factored_command, check=True)
    factored = factored_output.read_text(encoding="utf-8")
    subprocess.run(factored_command, check=True)
    assert factored_output.read_text(encoding="utf-8") == factored
    assert "generated_dppp_xy_xyz_factored_gradient" in factored

    fused_output = tmp_path / "generated" / "dppp_fused.cuh"
    fused_command = [
        sys.executable,
        "tools/generate_shell_kernels.py",
        "--shell-class",
        "dppp",
        "--lowering",
        "fused",
        "--output",
        str(fused_output),
    ]
    subprocess.run(fused_command, check=True)
    fused = fused_output.read_text(encoding="utf-8")
    subprocess.run(fused_command, check=True)
    assert fused_output.read_text(encoding="utf-8") == fused
    assert "generated_dppp_shell_class_force_rhf_kernel" in fused


@pytest.mark.parametrize(
    "spec", (SSSS_SPEC, PSSS_SPEC, DPDS_SPEC, DDPS_SPEC, FDDD_SPEC)
)
def test_codegen_cli_writes_generated_fused_candidate(tmp_path: Path, spec):
    output = tmp_path / "generated" / f"{spec.name}_fused.cuh"
    command = [
        sys.executable,
        "tools/generate_shell_kernels.py",
        "--shell-class",
        spec.name,
        "--lowering",
        "fused",
        "--output",
        str(output),
    ]
    subprocess.run(command, check=True)
    first = output.read_text(encoding="utf-8")
    subprocess.run(command, check=True)
    assert output.read_text(encoding="utf-8") == first
    assert first == emit_shell_class_fused_cuda(spec)
    assert f"generated_{spec.name}_shell_class_force_rhf_kernel" in first
