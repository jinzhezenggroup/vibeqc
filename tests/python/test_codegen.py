"""Legacy cross-cutting codegen tests while compiler-domain suites are split out.

Add new domain-owned coverage to focused ``test_codegen_*.py`` modules instead of
growing this compatibility surface. See #489.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import re
import subprocess
import time
import typing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from codegen_fixtures import (
    boys_values,
    factored_dppp_variables,
    sample_variables,
)
from vibeqc_compiler.integral import (
    DDDD_SPEC,
    DDPS_SPEC,
    DPDS_SPEC,
    DPPP_SPEC,
    FDDD_SPEC,
    FFPS_SPEC,
    FUSED_SHELL_SPEC_BY_NAME,
    PSPS_SPEC,
    PSSS_SPEC,
    AlgebraForm,
    AlgebraFusion,
    AlgebraOrdering,
    AlgebraPlacement,
    ContractionSpec,
    CudaTargetInfo,
    KernelConsumer,
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
    build_rys_axis_program,
    build_rys_force_program,
    build_shell_class_component_kernel,
    build_shell_class_contraction_kernel,
    build_weighted_shell_contraction_kernel,
    cartesian_components,
    cuda_target_info,
    dppp_components,
    emit_rys_force_root_body_cuda,
    emit_shell_class_fused_cuda,
    evaluate_fused_shell_component,
    evaluate_fused_shell_observables,
    evaluate_fused_shell_value,
    evaluate_ppps_rys_component,
    evaluate_rys_component,
    rys_boys_values,
    schedule_candidates,
    supports_component_lane_rys,
)
from vibeqc_compiler.integral.autotune import (
    StaticAlgebraModel,
    _analysis_roots,
    _compile_trial,
    _oracle_symbol_prefix,
    _packed_force_geometry_analysis,
    _production_fock_schedule_index,
    _read_shell_class_file,
    _requested_shell_class_names,
    _resolve_specifications,
    _run_autotune,
    emit_schedule_driver,
    emit_schedule_oracle_translation_unit,
    emit_schedule_translation_unit,
    estimate_occupancy,
    schedule_payload,
    static_algebra_model,
    supported_schedule_trials,
    write_tuned_manifest,
)
from vibeqc_compiler.integral.batch_benchmark import (
    DEFAULT_CANDIDATES,
    KernelResources,
    benchmark_command,
    candidate_specs,
    discover_candidate_specs,
    emit_batch_driver,
    emit_candidate_translation_unit,
    parse_ptxas_resources,
    rank_profiled_candidates,
)
from vibeqc_compiler.integral.batch_benchmark import (
    _compile_candidate as _compile_batch_candidate,
)
from vibeqc_compiler.integral.benchmark import (
    benchmark_command as standalone_benchmark_command,
)
from vibeqc_compiler.integral.benchmark import (
    emit_shell_class_benchmark_cuda,
    emit_shell_class_oracle_cuda,
)
from vibeqc_compiler.integral.capabilities import (
    CAPABILITY_MIXED_FOCK,
    CAPABILITY_STREAMING_FOCK,
    build_capability_report,
)
from vibeqc_compiler.integral.production import (
    _PRODUCTION_PRELUDE,
    _partition_production_selections,
    emit_registry_header,
    emit_registry_source,
    load_production_fock_manifest,
    load_production_kernel_selections,
    load_production_manifest,
    resolve_production_profile,
    write_production_bundle,
    write_production_bundles,
)
from vibeqc_compiler.integral.shell_class import (
    AXES,
)
from vibeqc_compiler.integral.weighted_eri_cuda import emit_low_order_weighted_header

TEST_CUDA_TARGET = cuda_target_info("sm_120")

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


def _direct_cuda_source() -> typing.Any:
    """Read direct dispatch with its shared contracts and numerical owners."""
    root = REPOSITORY_ROOT / "src/scf"
    return "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in (
            "cuda/direct_constants.hpp",
            "cuda/integral_limits.hpp",
            "cuda/scalar_math.cuh",
            "cuda/gaussian_geometry.cuh",
            "cuda/cartesian_angular.cuh",
            "cuda/boys_table.cuh",
            "cuda/hermite_recurrence.cuh",
            "cuda/coulomb_auxiliary.cuh",
            "cuda/one_electron_force_workspace.hpp",
            "cuda/one_electron_native_overlap.cuh",
            "cuda/one_electron_native_attraction.cuh",
            "cuda/one_electron_native_attraction_gradient.cuh",
            "cuda/one_electron_native_contraction.cuh",
            "cuda/one_electron_native_force.cuh",
            "cuda/one_electron_reference.cu",
            "cuda/one_electron_force_reference.cu",
            "cuda/nuclear_kernels.cu",
            "cuda/direct_pair_cache.cu",
            "cuda/scf_constants.hpp",
            "cuda/scf_state_kernels.cu",
            "cuda/scf_matrix_kernels.cu",
            "cuda/scf_density_kernels.cu",
            "cuda/scf_diis_kernels.cu",
            "cuda/scf_convergence_kernels.cu",
            "cuda/basis_transform_kernels.cu",
            "cuda/launch_geometry.hpp",
            "cuda/direct_metadata.hpp",
            "cuda/direct_queue_index.cuh",
            "cuda/direct_screening.cuh",
            "cuda/direct_task_encoding.cuh",
            "cuda/direct_page_screening.cuh",
            "cuda/direct_queue_profile.cuh",
            "cuda/direct_tile_validation.cu",
            "cuda/direct_density_bounds.cu",
            "cuda/direct_tile_compaction.cu",
            "cuda/direct_generated_tasks.cu",
            "cuda/direct_resident_tasks.cu",
            "cuda/direct_bounded_pages.cu",
            "cuda/direct_bounded_tasks.cu",
            "cuda/direct_queue_scan.cu",
            "cuda/direct_queue_diagnostics.cu",
            "cuda/direct_native_cartesian.cuh",
            "cuda/direct_native_contraction.cuh",
            "cuda/direct_native_eri_order2.cuh",
            "cuda/direct_native_eri_order3.cuh",
            "cuda/direct_native_eri_order4.cuh",
            "cuda/direct_native_gradient_types.cuh",
            "cuda/direct_native_order2_gradient.cuh",
            "cuda/direct_native_order2_shell.cuh",
            "cuda/direct_native_order3_gradient.cuh",
            "cuda/direct_native_pair_order2.cuh",
            "cuda/direct_native_pair_order2_gradient.cuh",
            "cuda/direct_native_pair_order3.cuh",
            "cuda/direct_native_pair_order3_gradient.cuh",
            "cuda/direct_native_psss.cuh",
            "cuda/direct_native_shell_class.cuh",
            "cuda/direct_native_shell_pair_hermite.cuh",
            "cuda/direct_native_source_contraction.cuh",
            "cuda/eri_tensor_index.cuh",
            "cuda/direct_eri_symmetry.cuh",
            "cuda/direct_fock_accumulation.cuh",
            "cuda/direct_fock_quartet.cuh",
            "cuda/direct_fock_psss.cuh",
            "cuda/direct_fock_order2.cuh",
            "cuda/direct_force_density.cuh",
            "cuda/direct_force_low_order.cuh",
            "cuda/direct_force_order2.cuh",
            "cuda/direct_force_quartet.cuh",
            "cuda/direct_bounded_contraction.cuh",
            "cuda/direct_cached_tensor_kernels.cu",
            "cuda/direct_schwarz_kernels.cu",
            "cuda/direct_packed_fock_kernels.cu",
            "cuda/direct_angular_fock.cu",
            "cuda/direct_reference_force.cu",
            "cuda/direct_bounded_dddd.cu",
            "cuda/direct_bounded_exact_force.cu",
            "cuda/direct_bounded_fallback.cu",
            "cuda/direct_angular_force.cu",
            "cuda/direct_jk_kernels.cu",
            "cuda/weighted_eri_kernels.cu",
            "cuda_rhf.cpp",
        )
    )


@pytest.mark.parametrize(
    ("maximum_order", "series_threshold"),
    ((0, 1.0e-8), (1, 0.25), (2, 0.75), (3, 1.25), (4, 2.0)),
)
def test_generated_low_order_boys_thresholds_preserve_upward_recurrence(
    maximum_order: int, series_threshold: float
) -> None:
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


def test_generic_cuda_emitter_uses_backend_lowering_not_dppp_compatibility() -> None:
    """Keep generic compilation independent of historical shell adapters."""

    emitter = (
        REPOSITORY_ROOT / "python" / "vibeqc_compiler" / "integral" / "cuda_emitter.py"
    ).read_text(encoding="utf-8")
    compatibility = (
        REPOSITORY_ROOT / "python" / "vibeqc_compiler" / "integral" / "dppp_dispatch.py"
    ).read_text(encoding="utf-8")
    production = (
        REPOSITORY_ROOT / "python" / "vibeqc_compiler" / "integral" / "production.py"
    ).read_text(encoding="utf-8")
    benchmark = (
        REPOSITORY_ROOT / "python" / "vibeqc_compiler" / "integral" / "benchmark.py"
    ).read_text(encoding="utf-8")
    assert "from . import cuda_lowering as _implementation" in emitter
    assert "dppp_dispatch" not in emitter
    assert "from .cuda_lowering import" in compatibility
    assert "emit_shell_class_fused_cuda" not in compatibility
    assert "from .dppp_dispatch import" not in production
    assert "from .dppp_dispatch import" not in benchmark


@pytest.mark.parametrize("architecture", ("sm_80", "sm_86", "sm_89", "sm_90", "sm_120"))
def test_cuda_target_catalog_covers_the_compile_matrix(architecture: str) -> None:
    """Expose target-derived scheduling and resource limits for supported SMs."""

    target = cuda_target_info(architecture)
    assert isinstance(target, CudaTargetInfo)
    assert target.architecture == architecture
    assert target.warp_size == 32
    assert target.maximum_threads_per_block == 1024
    assert target.tuning_maximum_shared_bytes <= target.shared_memory_per_block


@pytest.mark.parametrize("name", ("fsss", "fsps"))
def test_scalar_thread_force_lowering_is_structural_for_f_shells(
    name: str,
) -> None:
    """Generate scalar subset/Wick force code without a shell-name allowlist.

    These classes intentionally are not production promotions.  Emitting them
    here proves that the compiler-owned fallback can cover a new f-shell
    derivative class from its component metadata and derivative IR alone.
    """

    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    schedule = ScheduleIR(
        kind=ScheduleKind.THREAD_TASKS,
        block_threads=32,
        component_tile=spec.component_count,
        tasks_per_warp=32,
        shared_coulomb=False,
        minimum_blocks_per_sm=1,
    )
    source = emit_shell_class_fused_cuda(
        spec,
        build_fused_shell_plan(
            spec, schedule=schedule, recurrence="subset_wick", target=TEST_CUDA_TARGET
        ),
    )

    class_name = name[0].upper() + name[1:]
    assert f"generated_{name}_scalar_thread_force_task" in source
    assert f"generated_{name}_scalar_thread_accumulate_components_" in source
    assert f"Generated{class_name}ScalarThreadStorage" in source
    assert "scalar thread-task force lowering is currently specialized" not in source


def test_ppps_scalar_thread_lowering_uses_explicit_derivative_center_slots() -> None:
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
        build_fused_shell_plan(
            spec, integral=integral, schedule=schedule, target=TEST_CUDA_TARGET
        ),
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


def test_ppps_rys_program_is_a_compact_unique_state_recurrence() -> None:
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
        program.spec, schedule=schedule, recurrence="rys3", target=TEST_CUDA_TARGET
    )
    assert plan.kernel.integral.recurrence == "rys3"
    with pytest.raises(ValueError, match="requires rys4"):
        build_fused_shell_plan(DPPP_SPEC, recurrence="rys3", target=TEST_CUDA_TARGET)


def test_dddd_rys_program_exposes_five_root_backend_requirements() -> None:
    """Quantify the high-order state surface without emitting scalar algebra."""

    program = build_rys_force_program(DDDD_SPEC)
    assert program.nroots == 5
    assert len(program.component_order) == 1296
    assert len(program.axis_program.requested_states) == 162
    assert len(program.axis_program.instructions) == 216


def test_dddp_rys5_recurrence_matches_every_symbolic_component() -> None:
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


def test_dddd_rys5_recurrence_matches_representative_symbolic_components() -> None:
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


def test_dppp_rys_program_bounds_four_root_state_groups() -> None:
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


@pytest.mark.parametrize("name", ("psss", "psps", "ppss", "dsss"))
def test_low_order_shells_share_scalar_rys2_force_backend(name: str) -> None:
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
    plan = build_fused_shell_plan(
        spec, schedule=schedule, recurrence="rys2", target=TEST_CUDA_TARGET
    )
    source = emit_shell_class_fused_cuda(spec, plan)
    assert f"generated_{name}_rys2_force_task" in source
    assert f"generated_{name}_rys2_roots" in source
    assert f"component_weights[kGenerated{name.title()}ComponentCount][32]" in source
    assert "root_index < 2U" in source


def test_ppps_rys_recurrence_matches_every_symbolic_component() -> None:
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
) -> None:
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
) -> None:
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


def test_dppp_cooperative_rys4_uses_uniform_runtime_indexed_axis_recurrence() -> None:
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
        spec, schedule=schedule, recurrence="rys4", target=TEST_CUDA_TARGET
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


def test_dppp_rys4_uniform_warps_advance_32_quartets_per_block() -> None:
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
        DPPP_SPEC, schedule=schedule, recurrence="rys4", target=TEST_CUDA_TARGET
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
        target=TEST_CUDA_TARGET,
    )
    mixed_source = emit_shell_class_fused_cuda(DPPP_SPEC, mixed_plan)
    assert "kGeneratedDpppBlockThreads = 256U" in mixed_source
    assert "kGeneratedDpppFockBlockThreads = 192U" in mixed_source
    assert "GeneratedDpppSubgroupFockStorage" not in mixed_source


@pytest.mark.parametrize("name", ("dddp", "dddd"))
def test_high_order_rys5_uniform_warps_advance_32_quartets_per_block(
    name: str,
) -> None:
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
        target=TEST_CUDA_TARGET,
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
) -> None:
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
        target=TEST_CUDA_TARGET,
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
) -> None:
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
        target=TEST_CUDA_TARGET,
    )
    source = emit_shell_class_fused_cuda(spec, plan)
    assert f"generated_{name}_rys3_component_lane_task" in source
    assert f"generated_{name}_rys3_roots" in source
    assert trr_shape in source
    assert "switch (component)" not in source
    assert f"generated_{name}_shell_class_fock_rhf_kernel" in source


@pytest.mark.parametrize(
    ("name", "recurrence", "block_threads"),
    (("dpss", "rys3", 32), ("ddss", "rys3", 64)),
)
def test_component_lane_rys_fock_lowering_uses_structural_capabilities(
    name: str, recurrence: str, block_threads: int
) -> None:
    """Use the fixed-root Fock worker for legal classes beyond the old list."""

    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    schedule = ScheduleIR(
        kind=ScheduleKind.COMPONENT_LANES,
        block_threads=block_threads,
        component_tile=spec.component_count,
        tasks_per_warp=1,
        shared_coulomb=True,
        minimum_blocks_per_sm=1,
    )
    plan = build_fused_shell_plan(
        spec,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=schedule,
        recurrence=recurrence,
        target=TEST_CUDA_TARGET,
    )
    assert supports_component_lane_rys(spec, schedule)
    source = emit_shell_class_fused_cuda(spec, plan)

    assert f"generated_{name}_rys3_value_axis" in source
    assert f"generated_{name}_shell_class_fock_rhf_kernel" in source
    fock_marker = f"generated_{name}_shell_class_fock_task("
    fock_source = source[source.index(fock_marker) :]
    assert f"generated_{name}_component_value" not in fock_source


def test_weighted_psss_graph_cse_matches_component_oracle() -> None:
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


def test_shell_spec_generates_cca_components_and_compile_time_bounds() -> None:
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


def test_large_dddd_class_defaults_to_tiled_lowering() -> None:
    """Keep AO products beyond CUDA's block limit in the generated catalog."""

    assert DDDD_SPEC.component_count == 1296
    assert DDDD_SPEC.pair_orders == (4, 4)
    integral = build_integral_ir(DDDD_SPEC)
    candidates = schedule_candidates(integral, target=TEST_CUDA_TARGET)
    # Search now exposes all target-legal mappings; the production default
    # below must still use the qualified 64-component tile.
    tiled = [item for item in candidates if item.kind == ScheduleKind.TILED_COMPONENTS]
    limit = min(
        TEST_CUDA_TARGET.maximum_threads_per_block,
        TEST_CUDA_TARGET.maximum_threads_per_sm,
    )
    expected_tiles = [
        TEST_CUDA_TARGET.warp_size * 2**power
        for power in range(1, limit.bit_length())
        if TEST_CUDA_TARGET.warp_size * 2**power <= limit
        and TEST_CUDA_TARGET.warp_size * 2**power < DDDD_SPEC.component_count
    ]
    assert [item.component_tile for item in tiled] == expected_tiles
    assert {item.kind for item in candidates} == {
        ScheduleKind.PACKED_TASKS,
        ScheduleKind.SHELL_TASK,
        ScheduleKind.SUBGROUP_TASKS,
        ScheduleKind.TILED_COMPONENTS,
    }
    plan = build_fused_shell_plan(DDDD_SPEC, target=TEST_CUDA_TARGET)
    assert plan.schedule.kind == ScheduleKind.TILED_COMPONENTS
    assert plan.block_threads == 64
    assert len(plan.coulomb_states) == 220
    source = emit_shell_class_fused_cuda(DDDD_SPEC, plan)
    assert "kGeneratedDdddComponentCount = 1296U" in source
    assert "__constant__ short generated_dddd_coulomb_indices[1000]" in source
    assert "state >> 4U" in source
    assert "state >> 8U" in source
    assert "component_tile_begin += 64U" in source

    trials = supported_schedule_trials(DDDD_SPEC, target=TEST_CUDA_TARGET)
    assert len({trial.schedule_id for trial in trials}) == len(trials)
    tiled_trials = [
        trial
        for trial in trials
        if trial.schedule.kind == ScheduleKind.TILED_COMPONENTS
    ]
    assert (
        len(tiled_trials)
        == len(expected_tiles) * len(PairStorage) * len(PairOrientation) * 2
    )
    assert {
        (
            trial.schedule.component_tile,
            trial.schedule.pair_storage,
            trial.schedule.pair_orientation,
            trial.schedule.unroll_pair_terms,
        )
        for trial in tiled_trials
    } == {
        (tile, storage, orientation, unrolled)
        for tile in expected_tiles
        for storage in PairStorage
        for orientation in PairOrientation
        for unrolled in (True, False)
    }


def test_f_shell_cuda_lowering_emits_axes_triple_matchings_and_tiles() -> None:
    """Cover pair order six and a component product above the block limit."""

    ffps_source = emit_shell_class_fused_cuda(FFPS_SPEC, target=TEST_CUDA_TARGET)
    assert "generated_ffps_f_axes[10][3]" in ffps_source
    assert "if constexpr (PairOrder >= 6U)" in ffps_source
    assert "first_removed | second_removed | third_removed, 3U" in ffps_source
    assert "__constant__ short generated_ffps_coulomb_indices[729]" in ffps_source

    fddd_plan = build_fused_shell_plan(FDDD_SPEC, target=TEST_CUDA_TARGET)
    assert fddd_plan.schedule.kind == ScheduleKind.TILED_COMPONENTS
    assert fddd_plan.block_threads == 64
    assert FDDD_SPEC.component_count == 2160
    fddd_source = emit_shell_class_fused_cuda(FDDD_SPEC, fddd_plan)
    assert "generated_fddd_f_axes[10][3]" in fddd_source
    assert "component_tile_begin += 64U" in fddd_source


def test_shell_spec_rejects_invalid_metadata_and_components() -> None:
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
def test_generic_shell_ad_matches_factored_lowering(
    spec: typing.Any, component: typing.Any
) -> None:
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


@pytest.mark.parametrize(
    ("d_component", "p_components"),
    (("xx", "xxx"), ("xy", "xyz"), ("zz", "zyx")),
)
def test_factored_dppp_lowering_matches_full_symbolic_kernel(
    d_component: str, p_components: str
) -> None:
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


def test_dppp_fused_plan_covers_components_and_shared_coulomb_states() -> None:
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


@pytest.mark.parametrize("unrestricted", (False, True))
def test_closed_density_orbit_matches_unique_permutations(
    unrestricted: bool,
) -> None:
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


def test_equal_shell_pair_component_domain_matches_active_tile_triangle() -> None:
    """Avoid double-counting (ij|kl) and (kl|ij) in shell-wide workers."""

    source = emit_shell_class_fused_cuda(
        FUSED_SHELL_SPEC_BY_NAME["pppp"], target=TEST_CUDA_TARGET
    )
    assert (
        "shared.task.shell_pair[0] != shared.task.shell_pair[1] || "
        "(first_p * 3U + second_p) >= (third_p * 3U + fourth_p)" in source
    )


def test_fused_cuda_can_emit_fock_values_and_force_gradients_together() -> None:
    """Generate both consumers from one integral and component schedule IR."""

    plan = build_fused_shell_plan(
        DPDS_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        target=TEST_CUDA_TARGET,
    )
    source = emit_shell_class_fused_cuda(DPDS_SPEC, plan)
    assert "generated_dpds_component_value" in source
    assert "generated_dpds_component_gradient" in source
    assert "generated_dpds_shell_class_fock_rhf_kernel" in source
    assert "generated_dpds_shell_class_fock_uhf_persistent_kernel" in source
    assert "generated_dpds_shell_class_force_rhf_kernel" in source
    force_only = emit_shell_class_fused_cuda(DPDS_SPEC, target=TEST_CUDA_TARGET)
    assert "shell_class_fock" not in force_only
    assert "coordinate_gradient" not in source
    assert "Dual3" not in source


def test_dpds_fused_cuda_is_generated_from_shell_spec() -> None:
    source = emit_shell_class_fused_cuda(DPDS_SPEC, target=TEST_CUDA_TARGET)
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


def test_pair_orientation_changes_the_materialized_contraction_pair() -> None:
    """Make pair orientation a measured CUDA code shape, not manifest metadata."""

    base = build_fused_shell_plan(
        DPDS_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        target=TEST_CUDA_TARGET,
    ).schedule
    canonical = emit_shell_class_fused_cuda(
        DPDS_SPEC,
        build_fused_shell_plan(
            DPDS_SPEC,
            consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
            schedule=replace(base, pair_orientation=PairOrientation.CANONICAL),
            target=TEST_CUDA_TARGET,
        ),
    )
    swapped = emit_shell_class_fused_cuda(
        DPDS_SPEC,
        build_fused_shell_plan(
            DPDS_SPEC,
            consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
            schedule=replace(base, pair_orientation=PairOrientation.SWAPPED),
            target=TEST_CUDA_TARGET,
        ),
    )
    assert "GeneratedDpdsPairTerm second_terms[4]" in canonical
    assert "GeneratedDpdsValueTerm second_terms[4]" in canonical
    assert "GeneratedDpdsPairTerm first_terms[8]" in swapped
    assert "GeneratedDpdsValueTerm first_terms[8]" in swapped
    assert "GeneratedDpdsPairTerm first_terms[8]" not in canonical
    assert "GeneratedDpdsValueTerm first_terms[8]" not in canonical


def test_ddps_fused_cuda_generates_order_four_double_matchings() -> None:
    source = emit_shell_class_fused_cuda(DDPS_SPEC, target=TEST_CUDA_TARGET)
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


def test_rys4_component_lanes_raise_a_second_center_d_shell() -> None:
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
        DDPS_SPEC, schedule=schedule, recurrence="rys4", target=TEST_CUDA_TARGET
    )
    source = emit_shell_class_fused_cuda(DDPS_SPEC, plan)
    assert "if (b == 2U)" in source
    assert "trr, a + 3U, c, d, cd" in source
    assert "3.0 * ab * raised_twice" in source
    assert "__noinline__ void\ngenerated_ddps_rys4_component_lane_task" in source
    assert "generated_ddps_shell_class_force_rhf_kernel" in source


@pytest.mark.parametrize("name", ("psps", "ppss"))
def test_low_order_production_force_is_generated_by_common_rys2_pipeline(
    name: str,
) -> None:
    """Keep low-order production ownership in the shared IR and CUDA emitter."""

    selection = next(
        item
        for item in load_production_kernel_selections(
            REPOSITORY_ROOT
            / "python"
            / "vibeqc_compiler"
            / "integral"
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
        target=TEST_CUDA_TARGET,
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


def test_packed_force_geometry_omits_component_coulomb_tables() -> None:
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
            target=TEST_CUDA_TARGET,
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
    spec: typing.Any, pair_shift_rows: typing.Any
) -> None:
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
            target=TEST_CUDA_TARGET,
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


def test_packed_force_lowering_uses_explicit_derivative_center_slots() -> None:
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
        target=TEST_CUDA_TARGET,
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


def test_explicit_component_lane_fock_width_reaches_streaming_wrapper() -> None:
    """Keep a wider tuned ppps Fock CTA consistent across generated entry points."""

    spec = FUSED_SHELL_SPEC_BY_NAME["ppps"]
    force_schedule = ScheduleIR(
        kind=ScheduleKind.SUBGROUP_TASKS,
        block_threads=256,
        component_tile=spec.component_count,
        tasks_per_warp=4,
        shared_coulomb=True,
    )
    fock_schedule = ScheduleIR(
        kind=ScheduleKind.COMPONENT_LANES,
        block_threads=64,
        component_tile=spec.component_count,
        shared_coulomb=True,
    )
    plan = build_fused_shell_plan(
        spec,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=force_schedule,
        recurrence="rys3",
        target=TEST_CUDA_TARGET,
    )
    source = emit_shell_class_fused_cuda(
        spec,
        plan,
        fock_schedule=fock_schedule,
        capabilities=("streaming_fock",),
    )
    assert "kGeneratedPppsFockBlockThreads = 64U" in source


def test_high_impact_fock_classes_emit_generated_mixed_capability() -> None:
    """Keep the profiled FP32 AOT set explicit and independently routed."""

    sources = {}
    for name in ("ppps", "dpps", "ddds", "dspp"):
        spec = FUSED_SHELL_SPEC_BY_NAME[name]
        plan = build_fused_shell_plan(
            spec,
            consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
            target=TEST_CUDA_TARGET,
        )
        sources[name] = emit_shell_class_fused_cuda(
            spec,
            plan,
            capabilities=(CAPABILITY_MIXED_FOCK,) if name != "dspp" else (),
        )

    dpps = sources["dpps"]
    assert "generated_dpps_shell_class_mixed_fock_rhf_persistent_kernel" in dpps
    assert "generated_dpps_shell_class_mixed_fock_uhf_persistent_kernel" in dpps
    assert "kGeneratedDppsMixedFockBlockThreads" in dpps
    assert "struct GeneratedDppsMixedValueTerm" in dpps
    assert "  float component_integral = 0.0F;" in dpps
    assert "const double* density" in dpps
    assert "double* fock" in dpps
    assert "generated_ppps_shell_class_mixed_fock" in sources["ppps"]
    assert "generated_ddds_shell_class_mixed_fock" in sources["ddds"]
    assert "generated_dspp_shell_class_mixed_fock" not in sources["dspp"]


def test_mixed_fock_recomputed_coulomb_scratch_uses_fp32() -> None:
    """Compile the mixed path when a Fock schedule recomputes Coulomb state."""

    spec = FUSED_SHELL_SPEC_BY_NAME["ddds"]
    force_schedule = ScheduleIR(
        kind=ScheduleKind.COMPONENT_LANES,
        block_threads=224,
        component_tile=spec.component_count,
        tasks_per_warp=1,
        shared_coulomb=True,
        pair_orientation=PairOrientation.SWAPPED,
        pair_storage=PairStorage.RECOMPUTED,
        unroll_pair_terms=False,
        minimum_blocks_per_sm=1,
    )
    fock_schedule = replace(
        force_schedule,
        shared_coulomb=False,
        pair_storage=PairStorage.MATERIALIZED,
        unroll_pair_terms=True,
        minimum_blocks_per_sm=0,
    )
    plan = build_fused_shell_plan(
        spec,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=force_schedule,
        recurrence="rys4",
        target=TEST_CUDA_TARGET,
    )
    source = emit_shell_class_fused_cuda(
        spec,
        plan,
        fock_schedule=fock_schedule,
        capabilities=(CAPABILITY_MIXED_FOCK,),
    )
    mixed = source.split("struct GeneratedDddsMixedPrimitiveGeometry", maxsplit=1)[1]
    assert "float coulomb[1];" in mixed
    assert "double coulomb[1];" not in mixed


def test_simple_registry_dispatches_profiled_mixed_fock_classes() -> None:
    """Expose a selectable capability mask for the profiled mixed workers."""

    manifest = (
        REPOSITORY_ROOT
        / "python"
        / "vibeqc_compiler"
        / "integral"
        / "production_shell_classes.json"
    )
    selections = load_production_kernel_selections(manifest, "sm_120")
    header = emit_registry_header(selections)
    source = emit_registry_source(selections)

    assert "kMixedFockShellKernels" in header
    mixed_rows = header.split("kMixedFockShellKernels", maxsplit=1)[1].split(
        "}};", maxsplit=1
    )[0]
    expected = {
        "ppps",
        "dpps",
        "dsps",
        "dsds",
        "ddss",
        "ddps",
        "ddds",
        "pppp",
    }
    for name in expected:
        assert f'"{name}"' in mixed_rows
        assert f"vibeqc_launch_generated_{name}_mixed_fock" in source
    assert '"dspp"' not in mixed_rows
    assert "enabled_mixed_fock_shell_class_mask" in header
    assert "launch_shell_class_mixed_fock" in header
    assert "VIBEQC_AOT_MIXED_FOCK_SHELL_CLASSES" in source
    assert "vibeqc_launch_generated_dspp_mixed_fock" not in source


def test_direct_tile_validation_is_opt_in_and_reports_descriptor_context() -> None:
    """Keep the large-AO queue validator diagnostic-only and actionable."""

    source = _direct_cuda_source()
    policy = (REPOSITORY_ROOT / "src" / "scf" / "cuda" / "rhf_policy.cpp").read_text(
        encoding="utf-8"
    )
    assert '"VIBEQC_DIRECT_TILE_VALIDATION"' in policy
    assert "validate_direct_tile_descriptors_kernel" in source
    assert "DirectTileValidationRecord" in source
    assert "direct-tile-validation error=" in source
    # The validator must stop before a consumer can turn a bad descriptor into
    # a secondary illegal access; normal runs never enter this branch.
    assert "if (direct_tile_validation &&" in source
    assert "return cudaSuccess;" in source


def test_graph_native_eigensolver_override_covers_all_solver_calls() -> None:
    """Keep the large-matrix escape hatch off cuSOLVER in every phase."""

    source = _direct_cuda_source()
    override_begin = source.index("if (requested_graph_native_eigensolver_override) {")
    override_end = source.index(
        "    }\n  }\n  const CudaEigensolverFamily", override_begin
    )
    override = source[override_begin:override_end]
    assert "plan.eigensolver_diagnostic.family =" in override
    assert "plan.eigensolver_diagnostic.ordinary_family =" in override
    assert "CudaEigensolverFamily::graph_native" in override
    probe_call = source.index(
        "probe_xsyev_batched_device_launch_graph(", override_begin
    )
    assert probe_call > override_begin
    assert (
        "Do not probe that provider first"
        in source[override_begin - 300 : override_begin]
    )
    # Finalization and split ordinary-stream iterations use ordinary_family;
    # an override that changes only family silently reintroduces XsyevBatched.
    assert (
        "ordinary_eigensolver_family == CudaEigensolverFamily::xsyev_batched" in source
    )


def test_large_matrix_stream_fallback_matches_gpu4pyscf_solver_contract() -> None:
    """Keep Graph-rejected large matrices on the standard Xsyevd provider."""

    source = _direct_cuda_source()
    assert "CudaEigensolverFamily::xsyevd" in source
    assert "cusolverDnXsyevd_bufferSize" in source
    eigensolver = (REPOSITORY_ROOT / "src/scf/cuda/eigensolver.cpp").read_text()
    assert "cusolverDnXsyevd(" in eigensolver
    probe_end = source.index(
        "const XsyevBatchedDispatch dispatch =",
        source.index("probe_xsyev_batched_device_launch_graph("),
    )
    assert "cudaGetLastError" in source[probe_end - 320 : probe_end]
    assert "dispatch.device_launch_graph_provider" in source
    assert "plan.eigensolver_diagnostic.ordinary_family =" in source
    assert "CudaEigensolverFamily::xsyevd" in source
    assert "Unlike XsyevBatched" in eigensolver


def test_bounded_force_registry_gaps_use_exact_runtime_fallback() -> None:
    """Prevent large-AO force runs from regressing to a hard CUDA error."""

    source = _direct_cuda_source()
    fallback = source.index("const auto launch_bounded_generic_force")
    dispatch = source.index("const auto launch_bounded_force")
    dispatch_boundary = re.search(
        r"if \((?:options\.compute_forces &&\s+)?quartet_direct &&\s+plan\.shell_quartet_tile_capacities",
        source[dispatch:],
    )
    assert dispatch_boundary is not None
    dispatch_end = dispatch + dispatch_boundary.start()
    assert fallback < dispatch < dispatch_end
    assert "bounded_direct_shell_quartet_kernel" in source[fallback:dispatch]
    assert "uncovered_force_shell_class_mask == 0U" in source[fallback:dispatch]
    assert "launch_bounded_generic_force" in source[dispatch:dispatch_end]
    # Registry incompleteness must not be converted into the old hard failure.
    assert "return cudaErrorNotSupported;" not in source[dispatch:dispatch_end]


def test_production_manifest_drives_generated_registry_and_shards(
    tmp_path: Path,
) -> None:
    """Keep machine CUDA out of Git while retaining deterministic builds."""

    manifest = (
        REPOSITORY_ROOT
        / "python"
        / "vibeqc_compiler"
        / "integral"
        / "production_shell_classes.json"
    )
    specifications = load_production_manifest(manifest)
    fock_specifications = load_production_fock_manifest(manifest)
    assert tuple(spec.name for spec in specifications) == (
        "ssss",
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
    ) == tuple(spec.name for spec in specifications)
    assert all(selection.architecture == "sm_120" for selection in selections)
    assert all(
        selection.schedule.algebra_placement == AlgebraPlacement.MATERIALIZED_CSE
        for selection in selections
    )
    canonical_spd = {
        "ssss",
        "psss",
        "psps",
        "ppss",
        "ppps",
        "pppp",
        "dsss",
        "dsps",
        "dspp",
        "dsds",
        "dpss",
        "dpps",
        "dppp",
        "dpds",
        "dpdp",
        "ddss",
        "ddps",
        "ddpp",
        "ddds",
        "dddp",
        "dddd",
    }
    generated_force = {
        selection.spec.name
        for selection in selections
        if KernelConsumer.FORCE in selection.consumers
    }
    # psss force reuses the exact bounded scheduler; every other canonical
    # s/p/d class has a production-selected generated force consumer.
    assert (generated_force | {"psss"}) & canonical_spd == canonical_spd
    direct_source = _direct_cuda_source()
    assert "unexpected_tuned_spd_fallback_mask" in direct_source
    assert "kCanonicalSpdShellClassMask" in direct_source
    assert "aot_shell_class_selection_override_requested()" in direct_source
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
        if selection.spec.name == "psss"
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
def test_unmeasured_cuda_targets_require_explicit_portable_profile(
    architecture: str,
) -> None:
    """Never hide a missing tuned profile behind an implicit generic build."""

    manifest = (
        REPOSITORY_ROOT
        / "python"
        / "vibeqc_compiler"
        / "integral"
        / "production_shell_classes.json"
    )
    with pytest.raises(ValueError, match="portable_cuda.*explicitly"):
        resolve_production_profile(manifest, architecture)
    resolved = resolve_production_profile(manifest, architecture, "portable_cuda")
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
) -> None:
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
) -> None:
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


def test_runtime_buckets_all_generated_classes_before_dispatch() -> None:
    """Prevent production promotion from restoring one scan per exact class."""

    source = _direct_cuda_source()
    assert "classify_generated_shell_tasks_kernel" in source
    assert "prefix_generated_shell_task_counts_kernel" in source
    assert "materialize_generated_shell_tasks_kernel" in source
    assert "compact_generated_shell_tasks_kernel" not in source
    assert source.count("classify_generated_shell_tasks_kernel<<<") == 1


def test_one_electron_force_batches_point_charges_in_retained_reference_warp() -> None:
    """Keep the explicit native exception batched without restoring scalar code."""

    source = _direct_cuda_source()
    force_source = (
        REPOSITORY_ROOT / "src/scf/cuda/one_electron_native_force.cuh"
    ).read_text(encoding="utf-8")
    cooperative_begin = force_source.index(
        "void contracted_one_electron_force_pair_cooperative("
    )
    cooperative = force_source[cooperative_begin:]
    assert "atom_base += warpSize" in cooperative
    assert "shared_coefficients[axis]" in cooperative
    assert "if (lane == 0U)" in cooperative
    assert "__shfl_down_sync" in cooperative
    assert source.count("one_electron_force_cooperative_kernel<<<") == 1
    assert "one_electron_force_scalar_kernel" not in source
    assert 'std::getenv("VIBEQC_ONE_ELECTRON_FORCE_SCALAR")' not in source


def test_generated_one_electron_derivatives_are_the_production_default() -> None:
    """Promote compiler-owned derivatives while retaining an explicit escape hatch."""

    policy = (REPOSITORY_ROOT / "src/scf/cuda/rhf_policy.cpp").read_text(
        encoding="utf-8"
    )
    begin = policy.index("bool generated_one_electron_derivatives_requested()")
    end = policy.index("bool resident_psss_bra_requested()", begin)
    selection = policy[begin:end]
    assert 'std::getenv("VIBEQC_ONE_ELECTRON_DERIVATIVES")' in selection
    assert "selection == nullptr" in selection
    assert 'std::strcmp(selection, "generated") == 0' in selection
    assert 'std::strcmp(selection, "reference") == 0' in selection
    assert 'std::strcmp(selection, "native") == 0' in selection
    assert 'std::strcmp(selection, "tensor") == 0' in selection
    assert "silently changing scientific owner" in selection
    assert selection.count("return true;") >= 2
    assert 'std::getenv("VIBEQC_ONE_ELECTRON_DERIVATIVE_MAPPING")' in selection
    assert (
        "if (selection == nullptr) return NucleusCooperativeSchedule::schedule_code;"
        in selection
    )
    assert 'std::strcmp(selection, "nucleus_cooperative") == 0' in selection
    assert "return NucleusCooperativeSchedule::schedule_code;" in selection


def test_batched_finalization_reuses_each_converged_raw_fock() -> None:
    """Reuse requested-accuracy peers and restore shared force metadata."""

    source = _direct_cuda_source()
    assert "template <bool RetainConvergedDensity>" in source
    assert 'std::getenv("VIBEQC_FINAL_FOCK_REBUILD")' in source
    assert "select_final_fock_rebuild_kernel" in source
    assert "kTightConvergedFockReuseDensityRms = 1.0e-12" in source
    assert "kExpandedConvergedFockReuseDensityTolerance = 1.0e-9" in source
    assert "kExpandedConvergedFockReuseDensityRms = 2.0e-9" in source
    assert "converged_fock_reuse_density_rms(options.density_tolerance)" in source
    assert "copy_selected_matrices_kernel" in source
    assert "launch_direct_quartet_metadata(density, false)" in source
    # A resident dm0 is already normalized for its cached overlap matrix, so a
    # geometry change must re-run the warm-density normalization path.
    assert "plan.resident_warm_positions == host.positions" in source
    assert "plan.resident_warm_density == host.warm_density" in source
    assert "iteration > 1 || has_energy_baseline" in source
    assert "update_convergence_kernel<true>" in source
    assert "update_uhf_convergence_kernel<true>" in source


def test_ppps_queue_buckets_orientation_and_primitive_signature_on_device() -> None:
    """Keep Phase-3 bucketing on the compact production queue and A/B-able."""

    source = _direct_cuda_source()
    assert "kPppsSignatureBucketCount" in source
    assert "resident_ppps_signature_bucket" in source
    assert "prefix_ppps_resident_signature_buckets_kernel" in source
    assert 'std::getenv("VIBEQC_PPPS_SIGNATURE_BUCKETING")' in source
    assert 'std::getenv("VIBEQC_PPPS_BLOCK_THREADS")' in source
    assert "ppps_resident_block_threads_requested" in source
    assert "resident_signature_offsets[bucket_index]" in source
    assert "atomicAdd(resident_signature_write_counts + bucket_index" in source
    assert "std::uint32_t* generated_ppps_resident_signatures =" in source
    assert re.search(
        r"shell_class_profiling\s*\?\s*arena_pointer<std::uint32_t>",
        source,
    )
    assert "kBoundedForceSignatureShellClassMask" in source
    assert "bounded_force_signature_bucket" in source
    assert "scan_bounded_force_signature_counts_kernel" in source
    assert "prefix_bounded_force_signature_blocks_kernel" in source
    assert "bounded_paged_force_shell_class_mask" in source
    assert "bounded_force_signature_offsets, true" in source
    # Force-only classes (for example fpps) are not part of the Fock registry;
    # bounded force paging must enumerate the force registry itself.
    assert "generated::selected_shell_kernels(bounded_force_kernel_count)" in source
    assert "bounded_page_density_tails" in source


def test_bounded_force_signature_mask_tracks_warp_uniform_schedules() -> None:
    """Keep page sorting aligned with every production lockstep task worker."""

    manifest = (
        REPOSITORY_ROOT
        / "python"
        / "vibeqc_compiler"
        / "integral"
        / "production_shell_classes.json"
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
    source = _direct_cuda_source()
    mask_begin = source.index(
        "constexpr std::uint64_t kBoundedForceSignatureShellClassMask"
    )
    mask_end = source.index(";", mask_begin)
    configured_constants = set(
        re.findall(r"<< (k[A-Za-z0-9]+ShellClass)", source[mask_begin:mask_end])
    )
    assert configured_constants == expected_constants


def test_bounded_dppp_force_uses_nonterminating_paged_screening() -> None:
    """Keep DPPP paged without assuming Schwarz-sorted ket segments."""

    source = _direct_cuda_source()
    mask_begin = source.index(
        "const std::uint64_t bounded_force_legacy_queue_shell_class_mask"
    )
    mask_end = source.index(
        "const std::uint64_t covered_force_shell_class_mask", mask_begin
    )
    mask_source = source[mask_begin:mask_end]
    assert "std::uint64_t{1} << kDpppShellClass" not in mask_source

    queue_source = (REPOSITORY_ROOT / "src/scf/cuda/queue_plan.cpp").read_text()
    order_begin = queue_source.index("bool make_bounded_stream_shell_pair_order")
    order_end = queue_source.index(
        "std::uint64_t bounded_lower_triangle_row", order_begin
    )
    assert "std::sort" not in queue_source[order_begin:order_end]

    # Compaction now has its own owner. Bound each assertion by the next
    # function in that owner rather than by a comment in a different file.
    compact_source = (
        REPOSITORY_ROOT / "src/scf/cuda/direct_bounded_pages.cu"
    ).read_text()
    compact_begin = compact_source.index(
        "__global__ void compact_bounded_exact_class_force_wave_kernel"
    )
    compact_end = compact_source.index(
        "void launch_compact_bounded_exact_class_force_wave_kernel", compact_begin
    )
    low_order_begin = source.index(
        "__global__ void contract_bounded_exact_low_order_force_page_kernel"
    )
    low_order_end = source.index(
        "__global__ __launch_bounds__(kBoundedDirectThreads, 1) void bounded_direct_shell_quartet_kernel",
        low_order_begin,
    )
    for page_source in (
        compact_source[compact_begin:compact_end],
        source[low_order_begin:low_order_end],
    ):
        force_gate = re.search(
            r"page_density_tails\.force < force_tolerance\) \{\s*(\w+);",
            page_source,
        )
        fock_gate = re.search(
            r"page_density_tails\.fock < screening_tolerance\) \{\s*(\w+);",
            page_source,
        )
        assert force_gate is not None
        assert force_gate.group(1) == "continue"
        assert fock_gate is not None
        assert fock_gate.group(1) == "continue"

    page_begin = source.index("const auto launch_bounded_overflow_force")
    page_end = source.index("const auto launch_bounded_native_force", page_begin)
    page_source = source[page_begin:page_end]
    assert "bounded_force_legacy_queue_shell_class_mask" in page_source
    assert page_source.index("bounded_force_legacy_queue_shell_class_mask") < (
        page_source.index("bounded_generated_page_range")
    )


def test_bounded_page_range_tracks_end_across_systems() -> None:
    """Do not stop a page at the first system that it intersects."""

    source = _direct_cuda_source()
    source = (REPOSITORY_ROOT / "src/scf/cuda/queue_plan.cpp").read_text()
    range_begin = source.index("BoundedGeneratedPageRange bounded_generated_page_range")
    range_source = source[range_begin:]
    assert "bool found_end = false;" in range_source
    assert "found_begin && !found_end && page_end <= system_end" in range_source
    assert "found_end = true;" in range_source
    assert "bra_end == 0U" not in range_source


def test_bounded_fock_pages_do_not_duplicate_streaming_consumers() -> None:
    """Run full generated pages before overflow-only streaming workers."""

    source = _direct_cuda_source()
    begin = source.index("const auto launch_bounded_paged_generated_fock")
    end = source.index("const auto launch_bounded_generated_fock", begin)
    page_source = source[begin:end]
    # Generated streaming consumers are overflow-only: they return immediately
    # when the per-class overflow flag is clear.  The paged exact consumer must
    # therefore not skip those classes, or normal tasks disappear from Fock.
    assert "host_generated_streaming_fock_shell_class_mask" not in page_source
    assert "host_native_streaming_fock_shell_class_mask" in page_source


def test_fixed_generated_task_arena_has_a_memory_admission_limit() -> None:
    """Route large grid-addressable buckets before a multi-GiB allocation."""

    source = _direct_cuda_source()
    assert "direct_schedule.fixed_topology.arena_maximum_bytes" in source
    assert "direct_jk_bounded_streaming_task_capacity_limit" in source
    assert "resolve_direct_jk_schedule_policy" in source
    assert "direct_task_layout.exact_tile_count >" in source
    assert "sizeof(GeneratedShellTask)" in source
    assert "requested_bounded_direct_streaming = true" in source


def test_direct_task_resource_domains_remain_separate() -> None:
    """Do not reuse fixed-topology storage to size bounded streaming pages."""

    source = (REPOSITORY_ROOT / "src/scf/cuda/rhf_policy.cpp").read_text()
    begin = source.index("direct_jk_bounded_streaming_task_capacity_limit")
    end = source.index("bool reuse_converged_fock_requested", begin)
    capacity_source = source[begin:end]
    assert "policy.fixed_topology" not in capacity_source
    assert "policy.bounded_streaming.task_capacity_ceiling" in capacity_source
    assert "policy.bounded_streaming.arena_maximum_bytes" in capacity_source


def test_bounded_force_keeps_fock_only_classes_out_of_force_dispatch() -> None:
    """Do not call the force registry for the remaining Fock-only psss entry."""

    source = _direct_cuda_source()
    begin = source.index(
        "const std::uint64_t explicit_generated_force_shell_class_mask"
    )
    end = source.index("const auto launch_bounded_generated_force", begin)
    mask_source = source[begin:end]
    assert "selected_fock_shell_kernels" not in mask_source
    assert "selected_shell_kernels(bounded_force_kernel_count)" in mask_source
    assert "~kBoundedNativePagedForceShellClassMask" in mask_source
    assert "cudaErrorNotSupported" in mask_source


def test_psss_force_codegen_emits_only_independent_gradient_roots() -> None:
    """Keep the Direct-HF psss candidate free of unused value/center-four roots."""

    source = emit_low_order_weighted_header(inline_single_use=True)
    begin = source.index("IndependentGradient psss_force(")
    end = source.index("IndependentGradient ssss_force(", begin)
    psss_force = source[begin:end]
    assert "Gradient psss(" in source
    assert "result.value" not in psss_force
    assert "result.center[3]" not in psss_force
    for center in range(3):
        for axis in range(3):
            assert f"result.center[{center}][{axis}]" in psss_force


def test_psss_force_math_is_unconditionally_generated() -> None:
    """Keep retired handwritten psss force math and its route selector absent."""

    native_source = (REPOSITORY_ROOT / "src/scf/cuda/direct_native_psss.cuh").read_text(
        encoding="utf-8"
    )
    low_order_source = (
        REPOSITORY_ROOT / "src/scf/cuda/direct_force_low_order.cuh"
    ).read_text(encoding="utf-8")
    policy_source = (REPOSITORY_ROOT / "src/scf/cuda/rhf_policy.cpp").read_text(
        encoding="utf-8"
    )
    assert "GeneratedMath" not in native_source
    assert "if constexpr (GeneratedMath)" not in native_source
    assert "generated_weighted_eri::Geometry geometry;" in native_source
    assert "generated_weighted_eri::Geometry geometry{};" not in native_source
    assert "generated_weighted_eri::psss_force" in native_source
    assert "generated_psss_weighted" not in low_order_source
    assert "VIBEQC_PSSS_WEIGHTED" not in policy_source
    assert (
        "contracted_eri_cartesian_source_psss_weighted_gradient<ResidentBra>"
        in low_order_source
    )


def test_ssss_force_codegen_emits_only_independent_gradient_roots() -> None:
    """Keep the native-adapter ssss helper free of unused value/center-four work."""

    source = emit_low_order_weighted_header(inline_single_use=True)
    begin = source.index("IndependentGradient ssss_force(")
    end = source.index("}  // namespace vibeqc::scf::generated_weighted_eri", begin)
    ssss_force = source[begin:end]
    assert "result.value" not in ssss_force
    assert "result.center[3]" not in ssss_force
    assert "geometry.product_scales[3]" not in ssss_force
    assert "geometry.decay[3]" not in ssss_force
    for center in range(3):
        for axis in range(3):
            assert f"result.center[{center}][{axis}]" in ssss_force


def test_ssss_force_retires_handwritten_math_and_selector() -> None:
    """Keep ssss force science compiler-owned on the qualified native scheduler."""

    manifest = load_production_kernel_selections(
        REPOSITORY_ROOT
        / "python"
        / "vibeqc_compiler"
        / "integral"
        / "production_shell_classes.json",
        "sm_120",
    )
    ssss = next(selection for selection in manifest if selection.spec.name == "ssss")
    assert KernelConsumer.FORCE in ssss.consumers

    types_source = (
        REPOSITORY_ROOT / "src/scf/cuda/direct_native_gradient_types.cuh"
    ).read_text(encoding="utf-8")
    low_order_source = (
        REPOSITORY_ROOT / "src/scf/cuda/direct_force_low_order.cuh"
    ).read_text(encoding="utf-8")
    assert "SsssWeightedGradient" not in types_source
    assert "contract_two_electron_force_ssss_task" in low_order_source
    assert "generated_weighted_eri::ssss_force" in low_order_source
    assert "direct_native_order01_gradient.cuh" not in low_order_source
    assert "generated_math" not in low_order_source
    assert "geometry.product_scales[3]" not in low_order_source
    assert "geometry.decay[3][axis]" not in low_order_source

    policy = (REPOSITORY_ROOT / "src/scf/cuda/rhf_policy.cpp").read_text(
        encoding="utf-8"
    )
    resources = (REPOSITORY_ROOT / "python/vibeqc/resources_hf.py").read_text(
        encoding="utf-8"
    )
    driver = _direct_cuda_source()
    assert "VIBEQC_SSSS_FORCE" not in policy
    assert "VIBEQC_SSSS_FORCE" not in resources
    assert "generated_ssss_force" not in driver
    assert "const std::uint64_t ssss_shell_class_mask" in driver
    assert "~ssss_shell_class_mask" in driver
    assert "~explicit_generated_force_shell_class_mask" in driver


def test_order01_force_retires_handwritten_generic_fallback() -> None:
    """Keep total-order-zero/one Direct-HF force mathematics compiler-owned."""

    assert not (
        REPOSITORY_ROOT / "src/scf/cuda/direct_native_order01_gradient.cuh"
    ).exists()

    quartet = (REPOSITORY_ROOT / "src/scf/cuda/direct_force_quartet.cuh").read_text(
        encoding="utf-8"
    )
    assert "direct_native_order01_gradient.cuh" not in quartet
    assert "contracted_eri_cartesian_source_order01_gradient" not in quartet
    assert "static_assert(AngularOrder >= 2U" in quartet

    bounded = (
        REPOSITORY_ROOT / "src/scf/cuda/direct_bounded_contraction.cuh"
    ).read_text(encoding="utf-8")
    assert "VIBEQC_BOUNDED_FORCE_CASE(0)" not in bounded
    assert "VIBEQC_BOUNDED_FORCE_CASE(1)" not in bounded


def test_order2_force_codegen_emits_only_independent_gradient_roots() -> None:
    """Keep PSPS/PPSS/DSSS native schedulers backed by force-only compiler roots."""

    source = emit_low_order_weighted_header(inline_single_use=True)
    names = ("psps_force", "ppss_force", "dsss_force")
    for index, name in enumerate(names):
        begin = source.index(f"IndependentGradient {name}(")
        if index + 1 < len(names):
            end = source.index(f"IndependentGradient {names[index + 1]}(", begin)
        else:
            end = source.index(
                "}  // namespace vibeqc::scf::generated_weighted_eri", begin
            )
        function = source[begin:end]
        assert "result.value" not in function
        assert "result.center[3]" not in function
        for center in range(3):
            for axis in range(3):
                assert f"result.center[{center}][{axis}]" in function


def test_order2_force_retires_handwritten_gradient_bodies() -> None:
    """Keep exact order-two Direct-HF force mathematics compiler-owned."""

    for name in ("dsss", "ppss", "psps"):
        assert not (
            REPOSITORY_ROOT / f"src/scf/cuda/direct_native_{name}_gradient.cuh"
        ).exists()

    source = (REPOSITORY_ROOT / "src/scf/cuda/direct_force_order2.cuh").read_text(
        encoding="utf-8"
    )
    assert (
        "contracted_eri_cartesian_source_order2_generated_weighted_gradient" in source
    )
    for name in ("psps", "ppss", "dsss"):
        assert f"generated_weighted_eri::{name}_force" in source
        assert f"direct_native_{name}_gradient.cuh" not in source
        assert f"contracted_eri_cartesian_source_{name}_weighted_gradient" not in source
    assert "generated_weighted_eri::Geometry geometry;" in source
    assert "generated_weighted_eri::Geometry geometry{};" not in source


def test_bounded_psss_resident_path_is_allocated_and_disjoint_from_page_fallback() -> (
    None
):
    """Use the validated resident-bra consumer before paging psss force work."""

    source = _direct_cuda_source()
    assert re.search(
        r"requested_quartet_direct\s*\?\s*host\.psss_resident_tasks\.size\(\)",
        source,
    )
    assert "requested_quartet_direct && !requested_bounded_direct_streaming" in source
    assert "launch_bounded_resident_psss_force" in source
    assert "~(bounded_resident_psss_force_enabled" in source


def test_warm_density_validation_parallelizes_each_system_matrix() -> None:
    """Keep fixed-dm0 setup from regressing to one serial N^2 worker."""

    source = _direct_cuda_source()
    assert "constexpr unsigned kWarmDensityThreads = 256" in source
    assert "warm_density_block_sum<kWarmDensityThreads>" in source
    for kernel in ("apply_warm_density_kernel", "apply_uhf_warm_density_kernel"):
        launch = rf"launch_{kernel}\(\s*static_cast<unsigned>\(batch_size\),\s*"
        assert re.search(launch + r"kWarmDensityThreads", source)
        assert f"{kernel}<<<grid, block, shared_bytes, stream>>>" in source


def test_force_density_product_screening_is_force_only_and_conservative() -> None:
    """Keep the force queue optional without weakening the SCF Fock gate."""

    source = _direct_cuda_source()
    assert "enum class DirectScreeningPurpose" in source
    assert "DirectScreeningPurpose::Fock" in source
    assert "DirectScreeningPurpose::Force" in source
    assert "kForceDensityProductScreeningTolerance = 1.0e-14" in source
    assert "fmin(screening_tolerance, kForceDensityProductScreeningTolerance)" in source
    assert 'std::getenv("VIBEQC_FORCE_DENSITY_PRODUCT_SCREENING")' in source
    assert "launch_direct_force_compaction();" in source


def test_cached_direct_plan_reuses_immutable_task_layout() -> None:
    """Keep quadratic shell-pair topology enumeration out of warm replay."""

    source = _direct_cuda_source()
    layout_begin = source.index("detail::DirectQuartetTaskLayout direct_task_layout")
    layout_end = source.index(
        "// Direct consumers expand each compact logical tile", layout_begin
    )
    layout_setup = source[layout_begin:layout_end]
    assert "requested_quartet_direct && first_setup" in layout_setup
    assert "plan.total_shell_quartet_tiles" in layout_setup
    assert source.count("detail::make_direct_quartet_task_layout(") == 1
    bucket_source = (
        REPOSITORY_ROOT / "src" / "scf" / "cuda" / "rhf_bucket.cpp"
    ).read_text(encoding="utf-8")
    assert "**plan, candidate, options" in bucket_source


def test_mixed_precision_is_budgeted_per_item_on_the_prepared_census() -> None:
    """Keep the mixed route budgeted on the prepared census and refined in FP64.

    The public ``auto`` policy resolves the FP32 cutoff from the accumulated-error
    budget and the mixed-capable tile census of this reference, so a route that
    cannot supply a census (bounded streaming) keeps the FP64 operator instead of
    accumulating an unbounded rounding error. The cutoff and the admission are
    resolved per item, so one batch keeps a cold item on the exact FP64 operator
    while a warm item runs the mixed route. The legacy diagnostic switch stays a
    separate, deliberately unbudgeted override.
    """

    source = _direct_cuda_source()
    threshold_begin = source.index(
        "const MixedPrecisionFockPolicy requested_precision_policy"
    )
    threshold_end = source.index(
        "const bool requested_mixed_precision_fock", threshold_begin
    )
    resolution = source[threshold_begin:threshold_end]
    assert re.search(
        r"requested_quartet_direct\s*\?\s*resolve_mixed_precision_fock_policy\(",
        resolution,
    )
    assert "mixed_precision_eligible_tile_count" in resolution
    # The per-item census is kept per system, uploaded per execution, and read
    # both by the tile gate and by the per-item refinement entry.
    assert "system_mixed_capable_tile_counts" in source
    assert "mixed_precision_system_census" in source
    assert "mixed_fock_item_cutoff(mixed_precision_cutoff_ceiling" in source
    assert "host_mixed_item_census.data()" in source
    assert "enter_target_refinement_kernel<<<" in source
    policy = (REPOSITORY_ROOT / "src" / "scf" / "cuda" / "rhf_policy.cpp").read_text(
        encoding="utf-8"
    )
    assert "admit_auto_mixed_precision_fock" in policy
    assert "kMixedPrecisionFloat32UnitRoundoff * eligible_tiles" in policy
    assert "resolve_mixed_precision_item" in policy
    assert "allow_mixed_precision && mixed_precision_fock" in source
    # The finalization path must explicitly disable the iterative mixed route.
    assert "launch_fock_builder(density, false)" in source


def test_bounded_streaming_uses_monotonic_system_density_tail() -> None:
    """Prune large class segments with a conservative density coarse bound."""

    topology = (REPOSITORY_ROOT / "src" / "scf" / "generated_shell_task.hpp").read_text(
        encoding="utf-8"
    )
    generator = (
        REPOSITORY_ROOT / "python" / "vibeqc_compiler" / "integral" / "production.py"
    ).read_text(encoding="utf-8")
    assert "const double* system_density_bounds" in topology
    assert "const double* system_pair_density_bounds" in topology
    assert "const std::uint32_t* generated_overflow" in topology
    assert "topology.system_pair_density_bounds[" in generator
    assert "system_density_bound < screening_tolerance" in generator
    assert "topology.generated_overflow[{shell_class}U]" in generator


def test_bounded_streaming_profiles_executed_precision_per_shell_class() -> None:
    """Count actual retained quartets without changing normal kernel work."""

    generator = (
        REPOSITORY_ROOT / "python" / "vibeqc_compiler" / "integral" / "production.py"
    ).read_text(encoding="utf-8")
    source = _direct_cuda_source()
    assert "record_fock_precision" in generator
    assert "fp64_work_count, fp32_work_count" in generator
    assert "bounded_fock_fp64_work_counts + shell_class" in source
    assert "bounded_fock_fp32_work_counts + shell_class" in source
    assert "fp64_quartets=%llu fp32_quartets=%llu" in source


def test_generated_order2_fock_masks_handwritten_fallback() -> None:
    """Prevent generated order-two Fock quartets from being scattered twice."""

    source = _direct_cuda_source()
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


def test_production_codegen_cmake_tracks_transitive_generator_inputs(
    tmp_path: typing.Any,
) -> None:
    """Regenerate production CUDA whenever shared compiler stages change."""

    # Dynamic dependencies exist only after the generator has run. Exercise the
    # real CPU-configurable pilot, then inspect its emitted depfile rather than
    # requiring the retired all-compiler glob in Ninja's pre-build graph.
    subprocess.run(
        [
            "cmake",
            "-S",
            str(REPOSITORY_ROOT),
            "-B",
            str(tmp_path),
            "-G",
            "Ninja",
            "-DVIBEQC_ENABLE_CUDA=OFF",
            "-DVIBEQC_BUILD_TESTS=OFF",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output = "generated/shell_kernels/eri_psss_x_gradient.cuh"
    subprocess.run(
        ["cmake", "--build", str(tmp_path), "--target", output],
        check=True,
        capture_output=True,
        text=True,
    )
    dependencies = (tmp_path / f"{output}.d").read_text(encoding="utf-8")
    for dependency in (
        "python/vibeqc_compiler/integral/blocks.py",
        "python/vibeqc_compiler/integral/cache.py",
        "python/vibeqc_compiler/integral/cuda.py",
        "python/vibeqc_compiler/integral/capabilities.py",
        "python/vibeqc_compiler/integral/cuda_lowering.py",
        "python/vibeqc_compiler/integral/expr.py",
        "python/vibeqc_compiler/integral/fused_schedule.py",
        "python/vibeqc_compiler/integral/ir.py",
        "python/vibeqc_compiler/integral/ir_serialization.py",
        "python/vibeqc_compiler/integral/production.py",
        "python/vibeqc_compiler/integral/rys.py",
        "python/vibeqc_compiler/integral/rys3_data.py",
        "python/vibeqc_compiler/integral/rys5_data.py",
        "python/vibeqc_compiler/integral/shell_class.py",
        "python/vibeqc_compiler/integral/shell_signature.py",
        "python/vibeqc_compiler/integral/shell_spec.py",
    ):
        assert dependency in dependencies
    # Production AOT uses the same depfile-enabled registration while retaining
    # its explicit non-Python manifest dependency. Unrelated compiler stages must
    # not be reintroduced as unconditional dependencies.
    assert "python/vibeqc_compiler/tensor/layout.py" not in dependencies
    cuda = (REPOSITORY_ROOT / "cmake/VibeQCCuda.cmake").read_text(encoding="utf-8")
    production = cuda.split("vibeqc_register_generated_sources(", 1)[1]
    production_dependencies = production.split("DEPENDS", 1)[1].split("ARGS", 1)[0]
    assert "${VIBEQC_AOT_SHELL_MANIFEST}" in production_dependencies
    assert "${VIBEQC_SCIENTIFIC_COMPILER_INPUTS}" not in production_dependencies


def test_cuda_target_request_is_resolved_before_language_enablement() -> None:
    """Do not let CMake/NVCC invent a compiler-default CUDA target."""

    cmake = (REPOSITORY_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    cuda_block = cmake.split("if(VIBEQC_ENABLE_CUDA)", 1)[1].split(
        "# Scoped VibeQC-owned GFN2 CPU runtime", 1
    )[0]
    target_error = cuda_block.index("CUDA target architecture is required")
    target_assignment = cuda_block.index(
        "set(CMAKE_CUDA_ARCHITECTURES ${_vibeqc_cuda_requested_architectures})"
    )
    language_enable = cuda_block.index("enable_language(CUDA)")
    assert target_error < target_assignment < language_enable
    assert "VIBEQC_CUDA_COMPILE_ARCHITECTURES" in cuda_block[:language_enable]
    assert "VIBEQC_CUDA_ARCHITECTURES" in cuda_block[:language_enable]
    assert "DEFINED CMAKE_CUDA_ARCHITECTURES" in cuda_block[:language_enable]
    assert "DEFINED ENV{CUDAARCHS}" in cuda_block[:language_enable]
    assert "set(CMAKE_CUDA_ARCHITECTURES 120)" not in cuda_block


def test_virtual_cuda_target_keeps_host_profile_portable() -> None:
    """Do not apply a measured host schedule to PTX that may JIT on a future GPU."""

    cmake = (REPOSITORY_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    profile_block = cmake.split("# A single real architecture may use", 1)[1].split(
        "vibeqc_register_cuda_generated_sources", 1
    )[0]
    virtual_guard = profile_block.index(
        'if(NOT _vibeqc_cuda_profile_architecture MATCHES "-virtual$")'
    )
    real_normalization = profile_block.index(
        'string(REGEX REPLACE "-real$" "" _vibeqc_cuda_profile_architecture'
    )
    profile_define = profile_block.index("VIBEQC_CUDA_PROFILE_ARCHITECTURE=")
    assert virtual_guard < real_normalization < profile_define
    assert 'REGEX REPLACE "-.*$"' not in profile_block


def test_batch_screening_ranks_real_profile_and_emits_one_process_driver() -> None:
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
        target=TEST_CUDA_TARGET,
    )
    assert f"vibeqc_run_shell_class_{candidate.name}" in source
    driver = emit_batch_driver((candidate,))
    assert "cudaFree(nullptr)" in driver
    assert f"vibeqc_run_shell_class_{candidate.name}()" in driver


def test_batch_screening_discovers_consumer_specific_manifest_gap() -> None:
    """Discover a bounded work-ranked gap without a hand-written name list."""

    force = discover_candidate_specs(consumer=KernelConsumer.FORCE, limit=3)
    assert tuple(spec.name for spec in force) == ("ffff", "fffd", "fdfd")
    fock = discover_candidate_specs(consumer=KernelConsumer.FOCK, limit=3)
    assert tuple(spec.name for spec in fock) == ("ffff", "fffd", "fdfd")
    assert FUSED_SHELL_SPEC_BY_NAME["fpps"] in discover_candidate_specs(
        consumer=KernelConsumer.FOCK
    )


def test_codegen_capability_report_covers_catalog_and_manifest() -> None:
    """Report structural backend reasons for all 55 canonical shell classes."""

    manifest = (
        REPOSITORY_ROOT
        / "python"
        / "vibeqc_compiler"
        / "integral"
        / "production_shell_classes.json"
    )
    report = build_capability_report(
        architecture="sm_120",
        manifest=manifest,
    )
    assert report["total_shell_classes"] == 55
    assert report["generic_fused_supported"] == 55
    assert report["backend"] == {
        "name": "cuda",
        "architecture": "sm_120",
        "compute_capability": "12.0",
        "generator_abi": 1,
        "schedule_source": "schedule_candidates",
        "emitter_validation": "emit_shell_class_fused_cuda",
    }
    assert report["recurrence_supported"] == {
        "subset_wick": 55,
        "rys2": 4,
        "rys3": 11,
        "rys4": 16,
        "rys5": 14,
    }
    assert report["force_derivative_supported"] == {"1": 55, "2": 0}
    rows = {row["shell_class"]: row for row in report["shell_classes"]}
    assert rows["psss"]["recurrences"]["rys2"]["supported"] is True
    assert rows["dppp"]["recurrences"]["rys4"]["supported"] is True
    assert rows["dppp"]["recurrences"]["rys3"]["supported"] is False
    assert rows["dppp"]["force_derivative_orders"]["1"]["supported"] is True
    second_force = rows["dppp"]["force_derivative_orders"]["2"]
    assert second_force["supported"] is False
    assert "order-one derivatives" in second_force["reasons"][0]
    assert rows["fsps"]["production"]["force"] is False
    assert rows["fsps"]["production"]["status"] == "manifest_gap"
    assert (
        rows["fsps"]["production"]["promotion_gate"]
        == "real_molecular_endpoint_and_resource_gates"
    )
    assert rows["dpps"]["production"]["force"] is True
    assert rows["dpps"]["production"]["status"] == "manifest_selected"
    assert CAPABILITY_STREAMING_FOCK in rows["dpps"]["production"]["capabilities"]


def test_batch_screening_sorts_unsorted_profile_work_and_deduplicates() -> None:
    """Choose f-shell candidates by measured work, not profile row order."""

    payload = {
        "shell_classes": [
            {"class": "fsss", "primitive_quartets": 10},
            {"class": "fsps", "primitive_work": 70},
            {"class": "fddd", "primitive_quartets": 600},
            # A duplicate row can occur when profiles combine orientations.
            {"class": "fsps", "primitive_work": 700},
        ]
    }

    ranked = rank_profiled_candidates(payload, limit=3)

    assert tuple(spec.name for spec in ranked) == ("fsps", "fddd", "fsss")


def test_batch_screening_excludes_production_classes_per_consumer() -> None:
    """Allow a force-only production class to enter the Fock screener."""

    # FPPS is promoted for force but not for coefficient-only Fock.  The
    # explicit and profiled paths must therefore agree that it is a Fock
    # candidate while still rejecting it from force screening.
    with pytest.raises(ValueError, match="fpps.*force production AOT"):
        candidate_specs(("fpps",), consumer=KernelConsumer.FORCE)
    assert tuple(
        spec.name
        for spec in candidate_specs(
            ("fpps", "fpps"),
            consumer=KernelConsumer.FOCK,
        )
    ) == ("fpps",)

    payload = {
        "shell_classes": [
            {"class": "ssss", "primitive_work": 900},
            {"class": "fpps", "primitive_work": 1000},
        ]
    }
    ranked = rank_profiled_candidates(
        payload,
        limit=2,
        consumer=KernelConsumer.FOCK,
    )
    assert tuple(spec.name for spec in ranked) == ("fpps",)


def test_batch_screening_can_emit_coefficient_only_fock_candidates() -> None:
    """Route Fock screening through the same generated task ABI."""

    source = emit_candidate_translation_unit(
        FUSED_SHELL_SPEC_BY_NAME["fsps"],
        task_count=2,
        primitive_count=1,
        warmups=0,
        iterations=1,
        samples=1,
        consumer=KernelConsumer.FOCK,
        target=TEST_CUDA_TARGET,
    )

    assert r"\"consumer\":\"fock\"" in source
    assert "generated_fsps_shell_class_fock_rhf_kernel" in source
    assert r"\"maximum_fock_error\"" in source


def test_batch_benchmark_command_has_finite_slurm_allocation() -> None:
    """Keep manual batch execution aligned with the CUDA adapter contract."""

    command = benchmark_command(
        Path("build/shell_batch_benchmark"),
        srun="srun",
        partition="main",
        gres="gpu:5090:1",
        slurm_time="00:07:00",
    )

    assert command == [
        "srun",
        "--partition=main",
        "--gres=gpu:5090:1",
        "--nodes=1",
        "--ntasks=1",
        "--time=00:07:00",
        "build/shell_batch_benchmark",
    ]
    with pytest.raises(ValueError, match="non-empty"):
        benchmark_command(Path("benchmark"), slurm_time=" ")


def test_standalone_benchmark_command_has_finite_slurm_allocation() -> None:
    """Keep the standalone CUDA benchmark under the same scheduler guard."""

    assert standalone_benchmark_command(
        Path("build/dppp_benchmark"),
        slurm_time="00:05:00",
    ) == [
        "srun",
        "--partition=main",
        "--gres=gpu:1",
        "--nodes=1",
        "--ntasks=1",
        "--time=00:05:00",
        "build/dppp_benchmark",
    ]


def test_ptxas_resource_parser_selects_fock_symbol_family() -> None:
    """Keep Fock resource gates independent from force resource records."""

    diagnostics = (
        "Function properties for generated_fsps_shell_class_fock_rhf_kernel\n"
        "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
        "ptxas info    : Used 64 registers, 0 bytes lmem, 0 bytes smem\n"
    )

    resources = parse_ptxas_resources(
        diagnostics,
        "fsps",
        consumer=KernelConsumer.FOCK,
    )

    assert len(resources) == 1
    assert resources[0].function.endswith("fock_rhf_kernel")


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
) -> None:
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
        spec, schedule=schedule, recurrence=recurrence, target=TEST_CUDA_TARGET
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
) -> None:
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
        DPPP_SPEC, schedule=schedule, recurrence="rys4", target=TEST_CUDA_TARGET
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
) -> None:
    """Lock the sm_120 resource envelope for batched Rys4 promotions."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    plan = build_fused_shell_plan(
        spec, schedule=schedule, recurrence="rys4", target=TEST_CUDA_TARGET
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
) -> None:
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
        DPPP_SPEC, schedule=schedule, recurrence="rys4", target=TEST_CUDA_TARGET
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
) -> None:
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
    plan = build_fused_shell_plan(
        spec, schedule=schedule, recurrence="rys3", target=TEST_CUDA_TARGET
    )
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
) -> None:
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
        target=TEST_CUDA_TARGET,
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
) -> None:
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
        spec, schedule=direct_schedule, recurrence="rys3", target=TEST_CUDA_TARGET
    )
    production_plan = next(
        build_fused_shell_plan(
            spec,
            consumers=selection.consumers,
            schedule=selection.schedule,
            target=TEST_CUDA_TARGET,
        )
        for selection in load_production_kernel_selections(
            REPOSITORY_ROOT
            / "python"
            / "vibeqc_compiler"
            / "integral"
            / "production_shell_classes.json",
            "sm_120",
        )
        if selection.spec == spec
    )
    environment = dict(os.environ)

    def compile_and_run(label: str, plan: typing.Any) -> dict[str, object]:
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
) -> None:
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
        DPPP_SPEC, schedule=rys4_schedule, recurrence="rys4", target=TEST_CUDA_TARGET
    )
    baseline_plan = build_fused_shell_plan(
        DPPP_SPEC,
        schedule=rys4_schedule,
        recurrence="subset_wick",
        target=TEST_CUDA_TARGET,
    )
    environment = dict(os.environ)

    def compile_and_run(label: str, plan: typing.Any) -> dict[str, object]:
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
) -> None:
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
        DPPP_SPEC, schedule=uniform_schedule, recurrence="rys4", target=TEST_CUDA_TARGET
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
        target=TEST_CUDA_TARGET,
    )
    environment = dict(os.environ)

    def compile_and_run(label: str, plan: typing.Any) -> dict[str, object]:
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
) -> None:
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
            / "python"
            / "vibeqc_compiler"
            / "integral"
            / "production_shell_classes.json",
            "sm_120",
        )
        if selection.spec == spec
    )
    rys3_plan = build_fused_shell_plan(
        spec, schedule=selection.schedule, recurrence="rys3", target=TEST_CUDA_TARGET
    )
    baseline_plan = build_fused_shell_plan(
        spec,
        schedule=selection.schedule,
        recurrence="subset_wick",
        target=TEST_CUDA_TARGET,
    )
    environment = dict(os.environ)

    def compile_and_run(label: str, plan: typing.Any) -> dict[str, object]:
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
    tmp_path: Path, spec: typing.Any, resource_limits: typing.Any
) -> None:
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
        + emit_shell_class_fused_cuda(spec, target=TEST_CUDA_TARGET),
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
) -> None:
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
            build_fused_shell_plan(spec, schedule=schedule, target=TEST_CUDA_TARGET),
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
) -> None:
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
                target=TEST_CUDA_TARGET,
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
            / "python"
            / "vibeqc_compiler"
            / "integral"
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


def test_joint_fock_force_cuda_compiles_when_nvcc_is_configured(
    tmp_path: Path,
) -> None:
    """Compile the dual-consumer pilot through the real CUDA frontend."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    plan = build_fused_shell_plan(
        DPDS_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        target=TEST_CUDA_TARGET,
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
) -> None:
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
        for item in schedule_candidates(integral, target=TEST_CUDA_TARGET)
        if item.kind == ScheduleKind.TILED_COMPONENTS and item.component_tile == 64
    )
    plan = build_fused_shell_plan(
        DPPP_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=schedule,
        target=TEST_CUDA_TARGET,
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


def test_dddd_tiled_cuda_compiles_when_nvcc_is_configured(tmp_path: Path) -> None:
    """Compile a 1296-component class that cannot use one lane per quartet."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    schedule = replace(
        build_fused_shell_plan(DDDD_SPEC, target=TEST_CUDA_TARGET).schedule,
        block_threads=128,
        component_tile=128,
        pair_orientation=PairOrientation.SWAPPED,
        pair_storage=PairStorage.RECOMPUTED,
        unroll_pair_terms=True,
    )
    plan = build_fused_shell_plan(DDDD_SPEC, schedule=schedule, target=TEST_CUDA_TARGET)
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
def test_f_shell_cuda_compiles_when_nvcc_is_configured(
    tmp_path: Path, spec: typing.Any, consumers: typing.Any
) -> None:
    """Compile pair-order-six and tiled f-shell gradients with CUDA 12.9."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    schedule = replace(
        build_fused_shell_plan(spec, target=TEST_CUDA_TARGET).schedule,
        unroll_pair_terms=False,
    )
    if spec == FDDD_SPEC:
        schedule = replace(
            schedule,
            block_threads=128,
            component_tile=128,
            pair_storage=PairStorage.RECOMPUTED,
        )
    plan = build_fused_shell_plan(
        spec, consumers=consumers, schedule=schedule, target=TEST_CUDA_TARGET
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
) -> None:
    """Compile f-shell candidates admitted without shell-name allowlists."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    plan = build_fused_shell_plan(
        spec, schedule=schedule, recurrence=recurrence, target=TEST_CUDA_TARGET
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


def test_psss_shell_task_cuda_compiles_when_nvcc_is_configured(
    tmp_path: Path,
) -> None:
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
        for item in schedule_candidates(integral, target=TEST_CUDA_TARGET)
        if item.kind == ScheduleKind.SHELL_TASK
    )
    plan = build_fused_shell_plan(
        PSSS_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=schedule,
        target=TEST_CUDA_TARGET,
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


def test_psss_packed_cuda_compiles_when_nvcc_is_configured(
    tmp_path: Path,
) -> None:
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
        for item in schedule_candidates(integral, target=TEST_CUDA_TARGET)
        if item.kind == ScheduleKind.PACKED_TASKS
    )
    plan = build_fused_shell_plan(
        PSSS_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=schedule,
        target=TEST_CUDA_TARGET,
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
) -> None:
    """Compile the common production Rys2 source and reject spills."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    selection = next(
        item
        for item in load_production_kernel_selections(
            REPOSITORY_ROOT
            / "python"
            / "vibeqc_compiler"
            / "integral"
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
        target=TEST_CUDA_TARGET,
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


def test_high_component_fock_oracle_block_covers_every_component() -> None:
    """Do not truncate the independent oracle for subgroup candidates."""

    trial = next(
        trial
        for trial in supported_schedule_trials(
            FUSED_SHELL_SPEC_BY_NAME["dppp"],
            KernelConsumer.FOCK,
            target=TEST_CUDA_TARGET,
        )
        if trial.schedule.kind == ScheduleKind.SUBGROUP_TASKS
        and trial.schedule.block_threads == 128
        and trial.schedule.tasks_per_warp == 4
        and trial.schedule.pair_orientation == PairOrientation.SWAPPED
        and trial.schedule.pair_storage == PairStorage.MATERIALIZED
        and trial.schedule.unroll_pair_terms
        and trial.schedule.minimum_blocks_per_sm == 0
    )
    source = emit_shell_class_benchmark_cuda(
        FUSED_SHELL_SPEC_BY_NAME["dppp"],
        task_count=1,
        primitive_count=1,
        warmups=0,
        iterations=1,
        samples=1,
        consumer=KernelConsumer.FOCK,
        schedule=trial.schedule,
        target=TEST_CUDA_TARGET,
    )
    baseline = source.split("/** Per-component Fock baseline", maxsplit=1)[1]
    assert "__launch_bounds__(192)" in baseline
    assert "<<<kTaskCount,\n        192>>>" in baseline


def test_packed_order2_fock_oracle_drops_force_wrappers() -> None:
    """Keep packed low-order schedules available to Fock autotuning."""

    trial = next(
        trial
        for trial in supported_schedule_trials(
            PSPS_SPEC, KernelConsumer.FOCK, target=TEST_CUDA_TARGET
        )
        if trial.schedule.kind == ScheduleKind.PACKED_TASKS
    )
    plan = build_fused_shell_plan(
        PSPS_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=trial.schedule,
        target=TEST_CUDA_TARGET,
    )
    source = emit_shell_class_oracle_cuda(PSPS_SPEC, plan, KernelConsumer.FOCK)
    assert "generated_psps_shell_class_fock_rhf_kernel" in source
    assert "generated_psps_shell_class_force_rhf_kernel" not in source


def test_fock_benchmark_runs_when_nvcc_is_configured(tmp_path: Path) -> None:
    """Execute the swapped value benchmark and its independent oracle."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA benchmark gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    schedule = replace(
        build_fused_shell_plan(DPDS_SPEC, target=TEST_CUDA_TARGET).schedule,
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
            target=TEST_CUDA_TARGET,
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


def test_autotune_compile_timeout_terminates_the_compiler_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject pathological large-shell compiles without orphaning NVCC children."""

    trial = supported_schedule_trials(DPDS_SPEC, target=TEST_CUDA_TARGET)[0]
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


def test_autotune_driver_probes_and_rejects_target_before_trials() -> None:
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


def test_autotune_keeps_benchmark_executor_distinct_from_compile_pool(
    tmp_path: typing.Any, monkeypatch: typing.Any
) -> None:
    """Exercise both compile pools before using the configured GPU adapter."""
    from vibeqc_compiler.integral.tuning import driver
    from vibeqc_compiler.integral.tuning.cli import argument_parser

    class ReachedBenchmark(Exception):
        pass

    trial = supported_schedule_trials(PSPS_SPEC, target=TEST_CUDA_TARGET)[0]
    compiled = []
    monkeypatch.setenv("VIBEQC_BENCHMARK_PARTITION", "test-partition")
    monkeypatch.setenv("VIBEQC_BENCHMARK_GRES", "gpu:environment:1")
    monkeypatch.setattr(driver, "supported_schedule_trials", lambda *a: (trial,))
    for name in (
        "emit_schedule_oracle_translation_unit",
        "emit_schedule_translation_unit",
        "emit_schedule_driver",
    ):
        monkeypatch.setattr(driver, name, lambda *a, **kw: "// test-only source")

    def compile_trial(*args: typing.Any) -> typing.Any:
        compiled.append(args)
        return {"returncode": 0, "object": tmp_path / "test-only.o"}

    monkeypatch.setattr(driver, "_compile_trial", compile_trial)
    monkeypatch.setattr(
        driver.CudaCompilerAdapter,
        "link",
        lambda *a: subprocess.CompletedProcess([], 0, "", ""),
    )

    def run_benchmark(
        self: typing.Any, executable: typing.Any, environment: typing.Any
    ) -> None:
        assert len(compiled) == 2  # Independent oracle and candidate pools.
        assert self.partition == "test-partition"
        assert self.gres == "gpu:explicit:1"
        assert executable.name == "shell_schedule_autotune"
        raise ReachedBenchmark

    monkeypatch.setattr(driver.CudaBenchmarkExecutor, "run", run_benchmark)
    arguments = argument_parser().parse_args(
        [
            "--architecture=sm_90",
            "--shell-class=psps",
            "--gres=gpu:explicit:1",
            "--compile-jobs=2",
            "--work-directory",
            str(tmp_path),
        ]
    )
    with pytest.raises(ReachedBenchmark):
        driver._run_autotune(arguments)


def test_autotune_expands_shell_class_list_files_for_batch_runs(
    tmp_path: Path,
) -> None:
    """Keep file-driven hotspot batches deterministic and comment-friendly."""

    classes = tmp_path / "hotspots.txt"
    classes.write_text(
        "# measured warm classes\nppps, psps\n\n dpps # second group\n",
        encoding="utf-8",
    )

    assert _read_shell_class_file(classes) == ("ppps", "psps", "dpps")
    arguments = SimpleNamespace(
        shell_class=["ppps"],
        shell_class_file=[classes],
    )
    assert _requested_shell_class_names(arguments) == (
        "ppps",
        "ppps",
        "psps",
        "dpps",
    )
    # Specification resolution removes the repeated command-line entry while
    # retaining the first appearance order used by the batch driver.
    resolved = tuple(
        spec.name for spec in _resolve_specifications(arguments.shell_class)
    )
    assert resolved == (
        "ppps",
        "psps",
        "dpps",
    )


@pytest.mark.parametrize("name", ("ssss", "psss", "psps", "ppss"))
def test_fock_autotune_includes_shared_production_baseline(name: str) -> None:
    """Treat a shared primary schedule as the shipped Fock baseline."""

    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    expected = dict(_production_fock_schedule_index("sm_120"))[name]
    trials = supported_schedule_trials(
        spec, KernelConsumer.FOCK, target=TEST_CUDA_TARGET
    )
    assert sum(trial.schedule == expected for trial in trials) == 1


def test_production_subgroup_fock_baseline_is_not_experimental() -> None:
    """The shipped subgroup mapping is evidence, not a new proposal."""

    from vibeqc_compiler.integral.tuning.driver import _experimental_subgroup_blocked

    expected = dict(_production_fock_schedule_index("sm_120"))["ppps"]
    trials = supported_schedule_trials(
        FUSED_SHELL_SPEC_BY_NAME["ppps"],
        KernelConsumer.FOCK,
        target=TEST_CUDA_TARGET,
    )
    baseline = next(trial for trial in trials if trial.schedule == expected)
    proposal = next(
        trial
        for trial in trials
        if trial.schedule.kind == ScheduleKind.SUBGROUP_TASKS
        and trial.schedule != expected
    )
    assert not _experimental_subgroup_blocked(
        baseline, is_production_baseline=True, allow_experimental=False
    )
    assert _experimental_subgroup_blocked(
        proposal, is_production_baseline=False, allow_experimental=False
    )


@pytest.mark.parametrize(
    "name",
    ("ppps", "pppp", "dpps", "dppp", "dpdp", "ddds", "dddp"),
)
def test_fock_autotune_includes_high_component_production_baseline(
    name: typing.Any,
) -> None:
    """Compare high-component proposals against the manifest Fock worker."""

    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    expected = dict(_production_fock_schedule_index("sm_120"))[name]
    trials = supported_schedule_trials(
        spec, KernelConsumer.FOCK, target=TEST_CUDA_TARGET
    )
    baselines = [trial for trial in trials if trial.schedule == expected]
    assert len(baselines) == 1, f"missing manifest baseline for {name}"


def test_autotune_manifest_replacement_is_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preserve an existing batch manifest when the final replace fails."""

    source = tmp_path / "source.json"
    output = tmp_path / "tuned.json"
    source.write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )
    output.write_text("existing production manifest\n", encoding="utf-8")
    trial = supported_schedule_trials(DPDS_SPEC, target=TEST_CUDA_TARGET)[0]

    def fail_replace(source_path: Path, output_path: Path) -> None:
        assert Path(source_path).parent == tmp_path
        assert Path(output_path) == output
        raise OSError("synthetic atomic-replace failure")

    monkeypatch.setattr(
        "vibeqc_compiler.integral.tuning.manifest.os.replace", fail_replace
    )
    with pytest.raises(OSError, match="synthetic atomic-replace failure"):
        write_tuned_manifest(
            source,
            output,
            "sm_120",
            {"dpds": trial.schedule},
        )

    assert output.read_text(encoding="utf-8") == "existing production manifest\n"
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_autotune_analysis_roots_follow_declared_derivative_centers() -> None:
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


def test_autotune_trials_preserve_an_explicit_integral_ir() -> None:
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
    trials = supported_schedule_trials(
        DPDS_SPEC, integral=integral, target=TEST_CUDA_TARGET
    )
    assert trials
    assert all(trial.integral is integral for trial in trials)
    assert trials[0].static_model.recurrence_state_count == 84

    source = emit_schedule_oracle_translation_unit(trials[0])
    assert re.search(
        r"constexpr unsigned derivative_centers\[3\]\s*=\s*\{\s*0U,\s*2U,\s*3U\s*\};",
        source,
    )


def test_autotune_trial_identity_includes_explicit_integral_intent() -> None:
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
        DPDS_SPEC, integral=default_integral, target=TEST_CUDA_TARGET
    )[0]
    center_one_trial = supported_schedule_trials(
        DPDS_SPEC, integral=center_one_integral, target=TEST_CUDA_TARGET
    )[0]
    same_default_trial = supported_schedule_trials(
        DPDS_SPEC, integral=build_integral_ir(DPDS_SPEC), target=TEST_CUDA_TARGET
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


def test_autotune_static_model_records_operations_and_live_values() -> None:
    """Attach cached symbolic cost envelopes to every schedule candidate."""

    packed_force = next(
        trial
        for trial in supported_schedule_trials(PSPS_SPEC, target=TEST_CUDA_TARGET)
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
    assert (
        dict(packed_model.operation_counts)["power"]
        >= (dict(geometry_analysis.operation_counts)["power"])
    )
    assert packed_model.arithmetic_operation_count >= (
        geometry_plan.arithmetic_operation_count
    )

    component_trials = tuple(
        trial
        for trial in supported_schedule_trials(DPDS_SPEC, target=TEST_CUDA_TARGET)
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

    fock_trial = supported_schedule_trials(
        DPDS_SPEC, KernelConsumer.FOCK, target=TEST_CUDA_TARGET
    )[0]
    fock_model = fock_trial.static_model
    assert fock_model.root_count == 1
    assert fock_model.recurrence_state_count == 56


@pytest.mark.parametrize("name", ["ppps", "dpps", "dddd"])
def test_value_only_native_helpers_use_the_pruned_coulomb_table_stride(
    name: typing.Any,
) -> None:
    """A Fock-only manifest must index each emitted state through its IR table."""
    import re

    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    integral = build_integral_ir(spec, consumers=(KernelConsumer.FOCK,))
    plan = build_fused_shell_plan(spec, integral=integral, target=TEST_CUDA_TARGET)
    source = emit_shell_class_fused_cuda(spec, plan)
    side = integral.maximum_coulomb_order + 1
    table = re.search(
        rf"generated_{spec.name}_coulomb_indices\[(\d+)\] = \{{(.*?)\}};",
        source,
        re.DOTALL,
    )
    assert table is not None
    values = [int(value) for value in re.findall(r"-?\d+", table.group(2))]
    assert int(table.group(1)) == len(values) == side**3
    assert f"(x_order * {side}U + y_order) * {side}U + z_order" in source
    # The common geometry helper still evaluates the derivative Boys order;
    # shrinking its scratch arrays with the lookup stride would overwrite it.
    geometry_side = spec.maximum_force_coulomb_order + 1
    assert f"double boys[{geometry_side}];" in source
    assert f"double coordinate_powers[3][{geometry_side}];" in source
    for index, (x, y, z) in enumerate(plan.coulomb_states):
        assert values[(x * side + y) * side + z] == index


def test_fock_static_model_handles_transformed_component_graphs() -> None:
    """Nonbinary Fock candidates must retain a usable model after normalization."""
    trials = supported_schedule_trials(
        PSPS_SPEC, KernelConsumer.FOCK, target=TEST_CUDA_TARGET
    )
    transformed = [
        trial for trial in trials if trial.schedule.algebra_form != AlgebraForm.BINARY
    ]
    assert transformed
    for trial in transformed:
        model = trial.static_model
        assert model.scope == "balanced_component_sample_envelope"
        assert model.root_count == 1
        assert model.arithmetic_operation_count > 0
        assert model.algebra_form == trial.schedule.algebra_form


def test_packed_autotune_searches_real_algebra_placement_variants() -> None:
    """Tie schedule IDs, payloads, source lowering, and static models together."""

    trials = supported_schedule_trials(PSPS_SPEC, target=TEST_CUDA_TARGET)
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
            if trial.schedule.algebra_placement == AlgebraPlacement.MATERIALIZED_CSE
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


def test_autotune_candidate_limit_samples_distinct_execution_geometries() -> None:
    """Quick tuning must not spend its budget on one enumeration prefix."""

    from vibeqc_compiler.integral.tuning.driver import (
        _diverse_bounded_trials,
        _schedule_geometry_key,
    )

    trials = supported_schedule_trials(
        PSPS_SPEC, KernelConsumer.FOCK, target=TEST_CUDA_TARGET
    )
    chosen = _diverse_bounded_trials(trials, 8)

    assert len(chosen) == 8
    assert len({_schedule_geometry_key(trial) for trial in chosen}) == len(chosen)
    assert {trial.schedule.kind for trial in chosen} >= {
        ScheduleKind.PACKED_TASKS,
        ScheduleKind.SHELL_TASK,
        ScheduleKind.SUBGROUP_TASKS,
        ScheduleKind.COMPONENT_LANES,
    }


def test_autotune_candidate_artifact_includes_static_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persist the static model even when compilation rejects a candidate."""

    trial = next(
        trial
        for trial in supported_schedule_trials(PSPS_SPEC, target=TEST_CUDA_TARGET)
        if trial.schedule.kind == ScheduleKind.PACKED_TASKS
    )

    def failed_compile(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
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
        "vibeqc_compiler.integral.tuning.driver.supported_schedule_trials",
        lambda *args, **kwargs: (trial,),
    )
    monkeypatch.setattr(
        "vibeqc_compiler.integral.tuning.driver._compile_trial",
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
        schedule_kind=[trial.schedule.kind.value],
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
        manifest_output=tmp_path / "manifest.json",
        require_all_winners=True,
        manifest=REPOSITORY_ROOT
        / "python"
        / "vibeqc_compiler"
        / "integral"
        / "production_shell_classes.json",
    )

    report = _run_autotune(arguments)
    assert report["winners"] == []
    assert len(report["candidates"]) == 1
    assert report["candidates"][0]["static_model"] == (trial.static_model.to_payload())
    profitability = report["candidates"][0]["profitability"]
    assert profitability["static"]["arithmetic_operation_count"] == (
        trial.static_model.arithmetic_operation_count
    )
    assert profitability["static"]["peak_live_values"] == (
        trial.static_model.peak_live_values
    )
    assert profitability["compiled"]["compiled_registers_per_thread"] is None
    assert report["candidates"][0]["source_bytes"] is None
    assert report["candidates"][0]["object_bytes"] is None
    assert report["candidates"][0]["occupancy"]["available"] is False
    assert report["artifacts"]["linked_executable_bytes"] is None
    assert report["artifacts"]["schedule_objects"] == {trial.key: None}
    assert report["search"] == {
        "schedule_kinds": [trial.schedule.kind.value],
        "bounded_trial_count": 1,
        "execution_dedup_enabled": True,
        "execution_deduplicated_count": 0,
        "execution_deduplicated": [],
        "candidate_limit_per_class": None,
        "candidate_limit_strategy": None,
        "trial_count": 1,
    }
    assert report["manifest"]["write_skipped"] is True
    assert not (tmp_path / "manifest.json").exists()


def test_fock_autotune_rejects_candidates_without_baseline_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never promote a Fock proposal when its shipped baseline was absent."""

    trials = supported_schedule_trials(
        PSPS_SPEC, KernelConsumer.FOCK, target=TEST_CUDA_TARGET
    )
    baseline, candidate = trials[:2]

    monkeypatch.setattr(
        "vibeqc_compiler.integral.tuning.driver._production_fock_schedule_index",
        lambda architecture: ((PSPS_SPEC.name, baseline.schedule),),
    )
    monkeypatch.setattr(
        "vibeqc_compiler.integral.tuning.driver.supported_schedule_trials",
        lambda *args, **kwargs: (baseline, candidate),
    )
    monkeypatch.setattr(
        "vibeqc_compiler.integral.tuning.driver.emit_schedule_oracle_translation_unit",
        lambda *args, **kwargs: "// oracle\n",
    )
    monkeypatch.setattr(
        "vibeqc_compiler.integral.tuning.driver.emit_schedule_translation_unit",
        lambda *args, **kwargs: "// candidate\n",
    )
    monkeypatch.setattr(
        "vibeqc_compiler.integral.tuning.driver.emit_schedule_driver",
        lambda *args, **kwargs: "// driver\n",
    )
    monkeypatch.setattr(
        "vibeqc_compiler.integral.tuning.driver.emit_schedule_resource_translation_unit",
        lambda *args, **kwargs: "// resource\n",
    )
    monkeypatch.setattr(
        "vibeqc_compiler.integral.tuning.driver._resource_rejections",
        lambda *args, **kwargs: [],
    )
    # The gate is independent of symbolic envelope construction; keep this
    # focused test from spending time building the large Fock component graph.
    monkeypatch.setattr(
        "vibeqc_compiler.integral.tuning.policy.static_algebra_model",
        lambda trial: SimpleNamespace(to_payload=dict),
    )

    def successful_compile(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        selected_trial = args[3]
        return {
            "key": selected_trial.key,
            "object": tmp_path / f"{selected_trial.schedule_id}.o",
            "returncode": 0,
            "timed_out": False,
            "duration_seconds": 0.01,
            "diagnostics": "",
            "resources": (),
        }

    monkeypatch.setattr(
        "vibeqc_compiler.integral.tuning.driver._compile_trial", successful_compile
    )
    monkeypatch.setattr(
        "vibeqc_compiler.integral.cuda_adapter.CudaCompilerAdapter.link",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        ),
    )
    runtime = {
        "shell_class": candidate.spec.name,
        "consumer": candidate.consumer.value,
        "schedule_id": candidate.schedule_id,
        "trial_key": candidate.key,
        "maximum_fock": 0.0,
        "maximum_fock_error": 0.0,
        "speedup": 1.2,
        "fused_ms": 1.0,
    }
    monkeypatch.setattr(
        "vibeqc_compiler.integral.cuda_adapter.CudaBenchmarkExecutor.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps(runtime) + "\n",
            stderr="",
        ),
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
        shell_class=[PSPS_SPEC.name],
        schedule_kind=[],
        consumer=KernelConsumer.FOCK.value,
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
        require_all_winners=False,
        manifest=REPOSITORY_ROOT
        / "python"
        / "vibeqc_compiler"
        / "integral"
        / "production_shell_classes.json",
    )

    report = _run_autotune(arguments)

    candidate_row = next(
        row for row in report["candidates"] if row["trial_key"] == candidate.key
    )
    assert candidate_row["accepted"] is False
    assert (
        "production baseline did not produce a runtime result"
        in candidate_row["rejection_reasons"]
    )
    assert report["winners"] == []


def test_autotune_occupancy_artifact_is_resource_bounded() -> None:
    """Record an auditable occupancy upper bound beside PTXAS resources."""

    trial = next(
        trial
        for trial in supported_schedule_trials(PSPS_SPEC, target=TEST_CUDA_TARGET)
        if trial.schedule.kind == ScheduleKind.PACKED_TASKS
    )
    target = cuda_target_info("sm_120")
    resources = (
        KernelResources(
            function="generated_psps_force_kernel",
            registers=128,
            stack_bytes=0,
            spill_store_bytes=0,
            spill_load_bytes=0,
            shared_bytes=4096,
        ),
    )

    payload = estimate_occupancy(resources, trial, target)

    assert payload["available"] is True
    assert payload["method"] == "resource_upper_bound"
    assert payload["block_threads"] == trial.schedule.block_threads
    kernel = payload["kernels"][0]
    assert kernel["resident_blocks_per_sm"] == 16
    assert kernel["active_threads_per_sm"] == 512
    assert kernel["estimated_occupancy"] == pytest.approx(1 / 3)
    assert kernel["limits"] == {
        "threads": 48,
        "registers": 16,
        "shared_memory": 25,
        "blocks": 24,
    }


def test_autotune_occupancy_artifact_preserves_missing_resource_records() -> None:
    """Rejected compiles still expose why occupancy could not be estimated."""

    trial = supported_schedule_trials(PSPS_SPEC, target=TEST_CUDA_TARGET)[0]
    payload = estimate_occupancy((), trial, cuda_target_info("sm_120"))

    assert payload == {
        "available": False,
        "method": "resource_upper_bound",
        "block_threads": trial.schedule.block_threads,
        "kernels": [],
    }


def test_batch_candidate_compile_records_artifact_provenance(
    tmp_path: Path,
) -> None:
    """Keep batch screening reports auditable even with a failed PTXAS parse."""

    source = tmp_path / f"{PSPS_SPEC.name}_candidate.cu"
    source.write_text("// generated source\n", encoding="utf-8")
    fake_nvcc = tmp_path / "fake-nvcc"
    fake_nvcc.write_text(
        """#!/bin/sh
output=''
while [ "$#" -gt 0 ]; do
  if [ "$1" = "-o" ]; then
    shift
    output="$1"
  fi
  shift
done
printf 'fake object' > "$output"
""",
        encoding="utf-8",
    )
    fake_nvcc.chmod(0o755)

    row = _compile_batch_candidate(fake_nvcc, "sm_120", tmp_path, PSPS_SPEC)

    assert row["returncode"] == 0
    assert row["compile_seconds"] >= 0.0
    assert row["source_bytes"] == source.stat().st_size
    assert row["object_bytes"] == len("fake object")


def test_autotune_same_class_variants_link_when_nvcc_is_configured(
    tmp_path: Path,
) -> None:
    """Ensure symbol isolation lets one GPU process compare same-class code."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA link gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    trials = supported_schedule_trials(DPDS_SPEC, target=TEST_CUDA_TARGET)[:2]
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
def test_benchmark_is_generated_without_shell_specific_harness_code(
    spec: typing.Any, third_offset: typing.Any
) -> None:
    source = emit_shell_class_benchmark_cuda(
        spec,
        task_count=32,
        primitive_count=2,
        warmups=1,
        iterations=3,
        samples=5,
        target=TEST_CUDA_TARGET,
    )
    assert "constexpr std::size_t n = 16U" in source
    assert f"task.ao_begin[2] = {third_offset}U" in source
    assert "task.ao_begin[3] = 15U" in source
    assert f"generated_{spec.name}_component_gradient<true>" in source
    assert f"generated_{spec.name}_component_gradient<false>" in source
    assert f"generated_{spec.name}_component_recompute_rhf_kernel" in source
    assert "generated_dppp" not in source
