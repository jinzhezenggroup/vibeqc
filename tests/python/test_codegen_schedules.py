"""Schedule selection, lowering, autotune, and schedule-manifest codegen tests."""

from __future__ import annotations

import os
import subprocess
import typing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from codegen_fixtures import factored_dppp_variables, sample_variables
from vibeqc_compiler.integral import (
    DDDD_SPEC,
    DDPS_SPEC,
    DPDS_SPEC,
    DPPP_SPEC,
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
    DensityModel,
    DerivativeSpec,
    IntegralIR,
    KernelConsumer,
    NuclearCoordinates,
    PairOrientation,
    PairStorage,
    ScheduleIR,
    ScheduleKind,
    TranslationInvariant,
    build_dppp_contraction_kernel,
    build_fused_shell_plan,
    build_integral_ir,
    build_shell_class_contraction_kernel,
    cuda_target_info,
    dppp_components,
    emit_dppp_fused_cuda,
    emit_shell_class_fused_cuda,
    evaluate_dppp_fused_component,
    evaluate_fused_shell_component,
    evaluate_fused_shell_value,
    schedule_candidates,
)
from vibeqc_compiler.integral.autotune import (
    _production_fock_schedule_index,
    _requested_schedule_kinds,
    emit_schedule_driver,
    emit_schedule_oracle_translation_unit,
    emit_schedule_resource_translation_unit,
    emit_schedule_translation_unit,
    schedule_payload,
    supported_schedule_trials,
    update_manifest_payload,
)
from vibeqc_compiler.integral.backend import TargetInfo, TargetScheduleShape
from vibeqc_compiler.integral.benchmark import (
    emit_dppp_benchmark_cuda,
    emit_shell_class_benchmark_cuda,
)
from vibeqc_compiler.integral.production import (
    _schedule_from_payload,
    load_production_kernel_selections,
)

TEST_CUDA_TARGET = cuda_target_info("sm_120")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_integral_ir_has_no_accelerator_schedule_fields() -> None:
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


def test_integral_and_schedule_irs_separate_math_from_cuda_mapping() -> None:
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
    candidates = schedule_candidates(integral, target=TEST_CUDA_TARGET)
    assert [item.kind for item in candidates] == [
        ScheduleKind.COMPONENT_LANES,
        ScheduleKind.TILED_COMPONENTS,
        ScheduleKind.TILED_COMPONENTS,
    ]
    assert candidates[0].block_threads == 192
    assert [item.component_tile for item in candidates[1:]] == [64, 128]


def test_fock_autotune_reuses_manifest_declared_baseline_schedules() -> None:
    """Keep high-component Fock baselines out of a second shell-name table."""

    schedules = dict(_production_fock_schedule_index("sm_120"))
    assert schedules["dppp"].tasks_per_warp == 4
    assert schedules["dpdp"].pair_storage == PairStorage.RECOMPUTED
    assert schedules["ddds"].tasks_per_warp == 2
    assert schedules["dddp"].minimum_blocks_per_sm == 2


def test_small_shell_schedule_space_includes_packed_and_cooperative_variants() -> None:
    """Allow tuning to choose task packing instead of one fixed mapping."""

    candidates = schedule_candidates(
        build_integral_ir(PSPS_SPEC), target=TEST_CUDA_TARGET
    )
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


def test_subgroup_schedule_advances_independent_ppps_tasks_per_block() -> None:
    """Keep task-local barriers and reductions inside each lane subgroup."""

    spec = FUSED_SHELL_SPEC_BY_NAME["ppps"]
    schedule = next(
        item
        for item in schedule_candidates(
            build_integral_ir(
                spec,
                consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
            ),
            target=TEST_CUDA_TARGET,
        )
        if item.kind == ScheduleKind.SUBGROUP_TASKS and item.tasks_per_warp == 4
    )
    plan = build_fused_shell_plan(
        spec,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=schedule,
        target=TEST_CUDA_TARGET,
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


@pytest.mark.parametrize("name", ("dpss", "ppps", "dsps"))
def test_one_warp_component_schedule_strides_larger_coulomb_table(
    name: typing.Any,
) -> None:
    """Do not retain an idle second warp after a short Coulomb setup."""

    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    integral = build_integral_ir(
        spec,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
    )
    component_schedules = [
        item
        for item in schedule_candidates(integral, target=TEST_CUDA_TARGET)
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
            target=TEST_CUDA_TARGET,
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


def test_ppps_scalar_thread_schedule_emits_component_scoped_dag() -> None:
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
        build_fused_shell_plan(spec, schedule=schedule, target=TEST_CUDA_TARGET),
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


@pytest.mark.parametrize(
    "spec",
    (
        PSPS_SPEC,
        PPSS_SPEC,
        FUSED_SHELL_SPEC_BY_NAME["dsss"],
    ),
)
def test_packed_schedule_models_low_order_fock_workers(spec: typing.Any) -> None:
    """Keep the accepted Fock topology while force moves to scalar Rys2."""

    selection = next(
        selection
        for selection in load_production_kernel_selections(
            REPOSITORY_ROOT
            / "python"
            / "vibeqc_compiler"
            / "integral"
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


def test_zero_order_pairs_lower_through_shell_task_schedule() -> None:
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
        for item in schedule_candidates(integral, target=TEST_CUDA_TARGET)
        if item.kind == ScheduleKind.SHELL_TASK
    )
    plan = build_fused_shell_plan(
        PSSS_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=shell_schedule,
        target=TEST_CUDA_TARGET,
    )
    source = emit_shell_class_fused_cuda(PSSS_SPEC, plan)
    assert "kGeneratedPsssComponentCount = 3U" in source
    assert "generated_psss_pair_term<0U>" in source
    assert "const unsigned second_axes[1]" in source
    assert "generated_psss_component_gradient<false>" in source
    assert "generated_psss_component_value<false>" in source

    trials = supported_schedule_trials(PSSS_SPEC, target=TEST_CUDA_TARGET)
    assert any(trial.schedule.kind == ScheduleKind.PACKED_TASKS for trial in trials)
    assert any(trial.schedule.kind == ScheduleKind.SHELL_TASK for trial in trials)
    assert (
        sum(trial.schedule.kind == ScheduleKind.COMPONENT_LANES for trial in trials)
        == 8
    )

    packed_schedule = next(
        item
        for item in schedule_candidates(integral, target=TEST_CUDA_TARGET)
        if item.kind == ScheduleKind.PACKED_TASKS
    )
    packed_plan = build_fused_shell_plan(
        PSSS_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=packed_schedule,
        target=TEST_CUDA_TARGET,
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
        target=TEST_CUDA_TARGET,
    )
    assert "<<<(kTaskCount + 31U) / 32U," in benchmark


def test_large_pair_recompute_schedule_avoids_materialized_term_arrays() -> None:
    """Trade repeated pair algebra for bounded stack use in large tiled shells."""

    base = build_fused_shell_plan(
        DDDD_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        target=TEST_CUDA_TARGET,
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
        target=TEST_CUDA_TARGET,
    )
    source = emit_shell_class_fused_cuda(DDDD_SPEC, plan)
    assert "GeneratedDdddPairTerm first_terms" not in source
    assert "GeneratedDdddPairTerm second_terms" not in source
    assert "GeneratedDdddValueTerm first_terms" not in source
    assert "GeneratedDdddValueTerm second_terms" not in source
    assert "#pragma unroll 1" in source


def test_shell_spec_component_schedule_round_trips_without_manual_decoding() -> None:
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


@pytest.mark.parametrize("spec", (DPDS_SPEC, DDPS_SPEC))
def test_generic_fused_schedule_preserves_every_component_gradient(
    spec: typing.Any,
) -> None:
    """Audit generated lane schedules, including ddps double Wick matching."""

    plan = build_fused_shell_plan(spec, target=TEST_CUDA_TARGET)
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


def test_dppp_fused_schedule_preserves_all_component_gradients() -> None:
    values = factored_dppp_variables(sample_variables())
    for component in dppp_components():
        direct = build_dppp_contraction_kernel(component[0], component[1:])
        fused = evaluate_dppp_fused_component(component, values)
        for center in range(4):
            for axis in range(3):
                expected = direct.graph.evaluate(direct.gradients[center][axis], values)
                actual = fused[center][axis]
                assert actual == pytest.approx(expected, rel=4.0e-12, abs=4.0e-12)


def test_dppp_fused_cuda_emits_one_shared_shell_class_schedule() -> None:
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


def test_tiled_component_schedule_covers_force_fock_and_benchmark_oracle() -> None:
    """Lower component tiles without silently dropping high-index AO quartets."""

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
        target=TEST_CUDA_TARGET,
    )
    # The generated candidate and independent recompute oracle must both walk
    # every tile; otherwise a partial-component benchmark can falsely pass.
    assert benchmark.count("component_tile_begin += 64U") == 2


@pytest.mark.parametrize(
    ("name", "component_count", "state_count", "block_threads"),
    (
        ("ppps", 27, 20, 32),
        ("dpps", 54, 35, 64),
        ("dsps", 18, 20, 32),
    ),
)
def test_generated_fock_workers_use_value_only_shell_schedules(
    name: typing.Any,
    component_count: typing.Any,
    state_count: typing.Any,
    block_threads: typing.Any,
) -> None:
    """Keep force-only gradients out of the generated SCF hot path."""

    spec = FUSED_SHELL_SPEC_BY_NAME[name]
    plan = build_fused_shell_plan(
        spec,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        target=TEST_CUDA_TARGET,
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


@pytest.mark.parametrize("shared_coulomb", (True, False))
def test_schedule_knob_cuda_variants_compile_when_nvcc_is_configured(
    tmp_path: Path, shared_coulomb: bool
) -> None:
    """Compile both cooperative sharing policies through the real frontend."""

    nvcc = os.environ.get("VIBEQC_NVCC")
    if nvcc is None:
        pytest.skip("set VIBEQC_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("VIBEQC_CUDA_ARCH", "sm_90")
    base = build_fused_shell_plan(DPDS_SPEC, target=TEST_CUDA_TARGET).schedule
    schedule = replace(
        base,
        shared_coulomb=shared_coulomb,
        unroll_pair_terms=False,
    )
    plan = build_fused_shell_plan(DPDS_SPEC, schedule=schedule, target=TEST_CUDA_TARGET)
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


def test_dppp_benchmark_compares_shared_and_recomputed_schedules() -> None:
    source = emit_dppp_benchmark_cuda(
        task_count=32,
        primitive_count=2,
        warmups=1,
        iterations=3,
        samples=5,
        target=TEST_CUDA_TARGET,
    )
    assert "constexpr unsigned kTaskCount = 32;" in source
    assert "constexpr unsigned kPrimitiveCount = 2;" in source
    assert "generated_dppp_component_gradient<true>" in source
    assert "generated_dppp_component_gradient<false>" in source
    assert "generated_dppp_component_recompute_rhf_kernel" in source
    assert "generated_dppp_make_primitive_geometry_uncached" in source
    assert "device_primitive_pairs" in source
    assert '\\"speedup\\"' in source


def test_fock_benchmark_compares_value_only_shared_and_recomputed_schedules() -> None:
    """Benchmark the SCF hot consumer with an independent value oracle."""

    source = emit_shell_class_benchmark_cuda(
        DPDS_SPEC,
        task_count=32,
        primitive_count=2,
        warmups=1,
        iterations=3,
        samples=5,
        consumer=KernelConsumer.FOCK,
        target=TEST_CUDA_TARGET,
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


def test_benchmark_accepts_an_explicit_schedule_or_lowered_plan() -> None:
    """Make the measured code shape an explicit autotuning input."""

    default_plan = build_fused_shell_plan(DPDS_SPEC, target=TEST_CUDA_TARGET)
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
        target=TEST_CUDA_TARGET,
    )
    assert "double coulomb[1];" in source
    assert source.count("generated_dpds_component_gradient<false>") == 2
    assert "#pragma unroll 1" in source
    assert "#pragma unroll\n" in source
    assert "VIBEQC_PAIR_UNROLL" not in source
    plan = build_fused_shell_plan(DPDS_SPEC, schedule=schedule, target=TEST_CUDA_TARGET)
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


def test_autotune_emits_unique_schedule_variants_and_manifest_records() -> None:
    """Keep same-class variants linkable and every winner reproducible."""

    trials = supported_schedule_trials(DPDS_SPEC, target=TEST_CUDA_TARGET)
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

    fock_trials = supported_schedule_trials(
        DPDS_SPEC, KernelConsumer.FOCK, target=TEST_CUDA_TARGET
    )
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
                        "schedule": {"force_marker": "preserve"},
                    },
                    {
                        "shell_class": "dpps",
                        "consumers": ["force"],
                        "schedule": {"force_marker": "preserve-dpps"},
                    },
                ]
            }
        },
    }
    fock_updated = update_manifest_payload(
        fock_manifest,
        "sm_120",
        {"dpds": fock_trials[0].schedule, "dpps": fock_trials[1].schedule},
        KernelConsumer.FOCK,
    )
    fock_rows = fock_updated["architectures"]["sm_120"]["kernels"]
    assert [row["consumers"] for row in fock_rows] == [
        ["fock", "force"],
        ["fock", "force"],
    ]
    assert fock_rows[0]["schedule"] == {"force_marker": "preserve"}
    assert fock_rows[0]["fock_schedule"] == schedule_payload(fock_trials[0].schedule)
    assert fock_rows[1]["schedule"] == {"force_marker": "preserve-dpps"}
    assert fock_rows[1]["fock_schedule"] == schedule_payload(fock_trials[1].schedule)


def test_autotune_deduplicates_batch_schedule_family_filters() -> None:
    """Keep a related-family batch small without changing request order."""

    arguments = SimpleNamespace(
        schedule_kind=[
            ScheduleKind.COMPONENT_LANES.value,
            ScheduleKind.COMPONENT_LANES.value,
            ScheduleKind.SUBGROUP_TASKS.value,
        ]
    )
    assert _requested_schedule_kinds(arguments) == (
        ScheduleKind.COMPONENT_LANES,
        ScheduleKind.SUBGROUP_TASKS,
    )
    assert _requested_schedule_kinds(SimpleNamespace()) == ()


def test_algebra_placement_schedule_payload_is_backward_compatible() -> None:
    """Round-trip tuned placement while defaulting older manifests safely."""

    inline_schedule = next(
        trial.schedule
        for trial in supported_schedule_trials(PSPS_SPEC, target=TEST_CUDA_TARGET)
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
        _schedule_from_payload(payload).algebra_ordering == AlgebraOrdering.TOPOLOGICAL
    )
    assert _schedule_from_payload(payload).algebra_fusion == AlgebraFusion.SEPARATE
    assert _schedule_from_payload(payload).algebra_form == AlgebraForm.BINARY
    with pytest.raises(ValueError, match="packed tasks"):
        replace(
            build_fused_shell_plan(DPDS_SPEC, target=TEST_CUDA_TARGET).schedule,
            algebra_placement=AlgebraPlacement.INLINE_SINGLE_USE,
        )
    with pytest.raises(ValueError, match="packed tasks"):
        replace(
            build_fused_shell_plan(DPDS_SPEC, target=TEST_CUDA_TARGET).schedule,
            algebra_ordering=AlgebraOrdering.PRESSURE_AWARE,
        )
    with pytest.raises(ValueError, match="packed tasks"):
        replace(
            build_fused_shell_plan(DPDS_SPEC, target=TEST_CUDA_TARGET).schedule,
            algebra_fusion=AlgebraFusion.FMA,
        )
    with pytest.raises(ValueError, match="packed tasks"):
        replace(
            build_fused_shell_plan(DPDS_SPEC, target=TEST_CUDA_TARGET).schedule,
            algebra_form=AlgebraForm.CANONICAL_NARY,
        )
