"""Correctness tests for build-time shell-class symbolic code generation."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from tools.qce_codegen import (
    DDDD_SPEC,
    DDPS_SPEC,
    DPDS_SPEC,
    DPPP_SPEC,
    FDDD_SPEC,
    FFPS_SPEC,
    FUSED_SHELL_SPEC_BY_NAME,
    FUSED_SHELL_SPECS,
    PPSS_BLOCK_THREADS,
    PPSS_SPEC,
    PSPS_BLOCK_THREADS,
    PSPS_SPEC,
    PSSS_SPEC,
    SSSS_SPEC,
    KernelConsumer,
    NvrtcCacheSpec,
    PairOrientation,
    PairStorage,
    ScheduleKind,
    ShellClassSpec,
    build_dppp_component_kernel,
    build_dppp_contraction_kernel,
    build_dppp_fused_plan,
    build_fused_shell_plan,
    build_integral_ir,
    build_psss_kernel,
    build_shell_class_component_kernel,
    build_shell_class_contraction_kernel,
    build_weighted_shell_contraction_kernel,
    cartesian_components,
    dppp_components,
    emit_dppp_fused_cuda,
    emit_ppss_weighted_force_cuda,
    emit_psps_weighted_force_cuda,
    emit_shell_class_fused_cuda,
    evaluate_dppp_fused_component,
    evaluate_fused_shell_component,
    evaluate_fused_shell_value,
    nvrtc_cache_key,
    schedule_candidates,
)
from tools.qce_codegen.autotune import (
    _compile_trial,
    emit_schedule_driver,
    emit_schedule_oracle_translation_unit,
    emit_schedule_resource_translation_unit,
    emit_schedule_translation_unit,
    supported_schedule_trials,
    update_manifest_payload,
)
from tools.qce_codegen.batch_benchmark import (
    DEFAULT_CANDIDATES,
    candidate_specs,
    emit_batch_driver,
    emit_candidate_translation_unit,
    rank_profiled_candidates,
)
from tools.qce_codegen.benchmark import (
    emit_dppp_benchmark_cuda,
    emit_shell_class_benchmark_cuda,
    emit_shell_class_oracle_cuda,
)
from tools.qce_codegen.production import (
    _PRODUCTION_PRELUDE,
    emit_registry_header,
    load_production_fock_manifest,
    load_production_kernel_selections,
    load_production_manifest,
    write_production_bundle,
)
from tools.qce_codegen.shell_class import (
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
    "generated_psps_shell_class_force_rhf_persistent_kernel": (220, 0, 0),
    "generated_psps_shell_class_force_uhf_persistent_kernel": (220, 0, 0),
}
RTX5090_PPSS_RESOURCE_LIMITS = {
    "generated_ppss_shell_class_force_rhf_persistent_kernel": (234, 0, 0),
    "generated_ppss_shell_class_force_uhf_persistent_kernel": (234, 0, 0),
}


def test_integral_and_schedule_irs_separate_math_from_cuda_mapping():
    """Expose value/force intent independently from schedule selection."""

    integral = build_integral_ir(
        DPPP_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
    )
    assert integral.value_coulomb_order == 5
    assert integral.maximum_coulomb_order == 6
    assert integral.independent_force_centers == (0, 1, 2)
    candidates = schedule_candidates(integral)
    assert [item.kind for item in candidates] == [
        ScheduleKind.COMPONENT_LANES,
        ScheduleKind.TILED_COMPONENTS,
        ScheduleKind.TILED_COMPONENTS,
    ]
    assert candidates[0].block_threads == 192
    assert [item.component_tile for item in candidates[1:]] == [64, 128]


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
        if item.kind == ScheduleKind.SUBGROUP_TASKS
        and item.tasks_per_warp == 4
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
    assert "__syncthreads()" not in source.split(
        "GeneratedPppsSubgroupForceStorage", maxsplit=1
    )[1]
    assert "blockIdx.x) * 32U + subgroup" in source


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
    assert (
        f"state += kGenerated{class_name}BlockThreads" in source
    )
    assert "shared.coulomb[state] = generated_" in source


@pytest.mark.parametrize(
    "spec",
    (
        PSPS_SPEC,
        PPSS_SPEC,
        FUSED_SHELL_SPEC_BY_NAME["dsss"],
    ),
)
def test_packed_schedule_models_low_order_fock_workers(spec):
    """Keep the accepted order-two topology at one shell task per lane."""

    schedule = next(
        selection.schedule
        for selection in load_production_kernel_selections(
            REPOSITORY_ROOT
            / "tools"
            / "qce_codegen"
            / "production_shell_classes.json"
        )
        if selection.spec == spec
    )
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
    assert any(
        trial.schedule.kind == ScheduleKind.PACKED_TASKS for trial in trials
    )
    assert any(trial.schedule.kind == ScheduleKind.SHELL_TASK for trial in trials)
    assert sum(
        trial.schedule.kind == ScheduleKind.COMPONENT_LANES for trial in trials
    ) == 8

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
                direct.graph.evaluate(
                    direct.gradients[center][axis], values
                ),
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
        weight
        * evaluate_fused_shell_value(PSSS_SPEC, component, variables)
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

    for function, (register_limit, stack_limit, shared_limit) in (
        limits.items()
    ):
        match = re.search(
            rf"Function properties for {function}\n"
            r"\s+(\d+) bytes stack frame, (\d+) bytes spill stores, "
            r"(\d+) bytes spill loads\n"
            r"ptxas info\s+: Used (\d+) registers([^\n]*)",
            ptxas_output,
        )
        assert match is not None, f"missing ptxas resources for {function}"
        stack, spill_stores, spill_loads, registers = map(
            int, match.groups()[:4]
        )
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
    for order, value in enumerate(boys_values(
        values["rho"]
        * sum(values[f"difference_{axis}"] ** 2 for axis in AXES),
        10,
    )):
        values[f"boys_{order}"] = value
    direct = build_shell_class_contraction_kernel(DDDD_SPEC, component)
    fused = evaluate_fused_shell_component(DDDD_SPEC, component, values)
    assert evaluate_fused_shell_value(
        DDDD_SPEC, component, values
    ) == pytest.approx(
        values["prefactor"] * direct.graph.evaluate(direct.value, values),
        rel=2.0e-11,
        abs=2.0e-11,
    )
    for center in range(4):
        for axis in range(3):
            assert fused[center][axis] == pytest.approx(
                direct.graph.evaluate(
                    direct.gradients[center][axis], values
                ),
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
    argument = values["rho"] * sum(
        values[f"difference_{axis}"] ** 2 for axis in AXES
    )
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
                direct.graph.evaluate(
                    direct.gradients[center][axis], values
                ),
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
    factored_value = (
        factored_values["prefactor"]
        * factored.graph.evaluate(factored.value, factored_values)
    )
    assert factored_value == pytest.approx(full_value, rel=5.0e-13, abs=5.0e-13)
    for center in range(4):
        for axis in range(3):
            actual = factored.graph.evaluate(
                factored.gradients[center][axis], factored_values
            )
            expected = full.graph.evaluate(
                full.gradients[center][axis], full_values
            )
            assert actual == pytest.approx(
                expected, rel=3.0e-11, abs=3.0e-11
            )


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
        assert evaluate_fused_shell_value(
            spec, component, values
        ) == pytest.approx(
            values["prefactor"] * direct.graph.evaluate(direct.value, values),
            rel=8.0e-12,
            abs=8.0e-12,
        )
        for center in range(4):
            for axis in range(3):
                expected = direct.graph.evaluate(
                    direct.gradients[center][axis], values
                )
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
        values.append(
            ((2 * order + 1) * values[-1] - exponential) / (2 * argument)
        )
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
        candidate[order - 1] = (
            2.0 * argument * candidate[order] + exponential
        ) / (2 * order - 1)

    reference = [
        sum(
            (-argument) ** k
            / (math.factorial(k) * (2 * order + 2 * k + 1))
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
    result["prefactor"] = (
        2.0
        * math.pi**2.5
        / (p * q * math.sqrt(p + q))
        * math.exp(pair_distance_squared)
    )
    argument = result["rho"] * sum(
        result[f"difference_{axis}"] ** 2 for axis in AXES
    )
    for order, value in enumerate(boys_values(argument, 7)):
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
                evaluate_value(kernel, plus, 7)
                - evaluate_value(kernel, minus, 7)
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
    factored_value = (
        factored_values["prefactor"]
        * factored.graph.evaluate(factored.value, factored_values)
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
                expected = direct.graph.evaluate(
                    direct.gradients[center][axis], values
                )
                actual = fused[center][axis]
                assert actual == pytest.approx(
                    expected, rel=4.0e-12, abs=4.0e-12
                )


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


def test_equal_shell_pair_component_domain_matches_active_tile_triangle():
    """Avoid double-counting (ij|kl) and (kl|ij) in shell-wide workers."""

    source = emit_shell_class_fused_cuda(FUSED_SHELL_SPEC_BY_NAME["pppp"])
    assert (
        "shared.task.shell_pair[0] != shared.task.shell_pair[1] || "
        "(first_p * 3U + second_p) >= (third_p * 3U + fourth_p)"
        in source
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
        if item.kind == ScheduleKind.TILED_COMPONENTS
        and item.component_tile == 64
    )
    plan = build_fused_shell_plan(
        DPPP_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
        schedule=schedule,
    )
    source = emit_shell_class_fused_cuda(DPPP_SPEC, plan)
    assert "kGeneratedDpppBlockThreads = 64U" in source
    assert source.count("component_tile_begin += 64U") == 2
    assert source.count("state += kGeneratedDpppBlockThreads") == 2
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
            schedule=replace(
                base, pair_orientation=PairOrientation.CANONICAL
            ),
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


def test_psps_weighted_cuda_uses_one_thread_per_complete_shell_task():
    """Keep low-order work dense instead of assigning three lanes per block."""

    source = emit_psps_weighted_force_cuda()
    assert PSPS_SPEC.angular == (1, 0, 1, 0)
    assert PSPS_BLOCK_THREADS == 256
    assert "kGeneratedPspsBlockThreads = 256U" in source
    assert "atomicAdd(task_head, 32U)" in source
    assert "*task_offset + task_index" in source
    assert "double component_weight[9]" in source
    assert "boys_values<3>" in source
    assert "generated_psps_density_coefficient<Unrestricted>" in source
    assert "generated_psps_shell_class_force_rhf_persistent_kernel" in source
    assert "generated_psps_shell_class_force_uhf_persistent_kernel" in source
    assert "__noinline__" not in source
    assert "Dual3" not in source


def test_ppss_weighted_cuda_reuses_the_low_order_worker_shape():
    """Generate ppss without routing its nine AO components through lanes."""

    source = emit_ppss_weighted_force_cuda()
    assert PPSS_SPEC.angular == (1, 1, 0, 0)
    assert PPSS_BLOCK_THREADS == 256
    assert "kGeneratedPpssBlockThreads = 256U" in source
    assert "atomicAdd(task_head, 32U)" in source
    assert "*task_offset + task_index" in source
    assert "double component_weight[9]" in source
    assert "generated_ppss_shell_class_force_rhf_persistent_kernel" in source
    assert "generated_ppss_shell_class_force_uhf_persistent_kernel" in source
    assert "__noinline__" not in source
    assert "Dual3" not in source


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
    assert (
        f"kGenerated{class_name}ComponentCount = {component_count}U"
        in source
    )
    assert (
        f"kGenerated{class_name}FockCoulombStateCount =\n"
        f"    {state_count}U"
        in source
    )
    assert (
        f"kGenerated{class_name}FockBlockThreads = {block_threads}U"
        in source
    )
    assert f"Generated{class_name}ValueTerm" in source
    assert f"generated_{name}_component_value" in source
    assert (
        f"generated_{name}_shell_class_fock_rhf_persistent_kernel"
        in source
    )
    assert (
        f"generated_{name}_shell_class_fock_uhf_persistent_kernel"
        in source
    )
    fock_fragment = source.split(
        "/** Coefficient-only pair term used by the SCF Fock recurrence. */",
        maxsplit=1,
    )[1]
    assert f"generated_{name}_density_coefficient" not in fock_fragment


def test_production_manifest_drives_generated_registry_and_shards(tmp_path: Path):
    """Keep machine CUDA out of Git while retaining deterministic builds."""

    manifest = (
        REPOSITORY_ROOT
        / "tools"
        / "qce_codegen"
        / "production_shell_classes.json"
    )
    specifications = load_production_manifest(manifest)
    fock_specifications = load_production_fock_manifest(manifest)
    assert tuple(spec.name for spec in specifications) == (
        "dppp",
        "dpdp",
        "dddp",
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
        "dppp",
        "dpds",
        "ddps",
        "ppps",
        "dpps",
        "dsps",
        "psps",
        "ppss",
        "dsss",
    )
    selections = load_production_kernel_selections(manifest, "sm_120")
    assert tuple(selection.spec.name for selection in selections) == tuple(
        spec.name for spec in specifications
    )
    assert all(selection.architecture == "sm_120" for selection in selections)
    assert {
        selection.spec.name: selection.schedule.pair_storage
        for selection in selections
    } == {
        spec.name: (
            PairStorage.RECOMPUTED
            if spec.name in ("dpdp", "dddp", "ddpp", "ddds")
            else PairStorage.MATERIALIZED
        )
        for spec in specifications
    }
    assert tuple(selection.consumers for selection in selections) == tuple(
        (KernelConsumer.FOCK, KernelConsumer.FORCE)
        if selection.spec.name
        in (
            "dppp",
            "dpds",
            "ddps",
            "ppps",
            "dpps",
            "dsps",
            "psps",
            "ppss",
            "dsss",
        )
        else (KernelConsumer.FORCE,)
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
    assert '{"dppp", 12U, 5U, 192U, 3U, 162U}' in header
    assert '{"dpds", 13U, 5U, 128U, 3U, 108U}' in header
    assert '{"ddps", 16U, 5U, 128U, 3U, 108U}' in header
    assert '{"ppps", 4U, 3U, 32U, 3U, 27U}' in header
    assert '{"dsps", 7U, 3U, 32U, 3U, 18U}' in header
    assert '{"dpdp", 14U, 6U, 64U, 2U, 64U}' in header
    assert '{"dddp", 19U, 7U, 64U, 2U, 64U}' in header
    assert '{"dpss", 10U, 3U, 32U, 2U, 18U}' in header
    assert '{"dsds", 9U, 4U, 64U, 2U, 36U}' in header
    assert '{"ddss", 15U, 4U, 64U, 2U, 36U}' in header
    assert '{"ddpp", 17U, 6U, 64U, 2U, 64U}' in header
    assert '{"ddds", 18U, 6U, 64U, 2U, 64U}' in header
    assert '{"dspp", 8U, 4U, 64U, 2U, 54U}' in header
    assert '{"dpps", 11U, 4U, 64U, 3U, 54U}' in header
    assert '{"pppp", 5U, 4U, 96U, 2U, 81U}' in header
    assert '{"psps", 2U, 2U, 32U, 3U, 9U}' in header
    assert '{"ppss", 3U, 2U, 32U, 3U, 9U}' in header
    assert '{"dsss", 6U, 2U, 32U, 3U, 6U}' in header
    assert "QCE_AOT_SHELL_CLASSES" in header
    assert "QCE_AOT_FOCK_SHELL_CLASSES" in header
    shards = "\n".join(
        path.read_text(encoding="utf-8")
        for path in first
        if "shard" in path.name
    )
    assert "offsetof(GeneratedDpppShellTask, shell_pair)" in shards
    assert (
        "offsetof(GeneratedDpppPrimitivePairData, product_center)"
        in shards
    )
    assert "const std::uint32_t* task_offset" in header
    generated_sources = [path.read_text(encoding="utf-8") for path in first]
    assert any("*task_offset + task_index" in source for source in generated_sources)
    assert any(
        "worker_blocks, tasks, task_offset" in source
        for source in generated_sources
    )


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

    worker_begin = source.index(
        "void build_fock_direct_order2_persistent_kernel("
    )
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
        "tools/qce_codegen/cuda.py",
        "tools/qce_codegen/dppp_dispatch.py",
        "tools/qce_codegen/expr.py",
        "tools/qce_codegen/fused_schedule.py",
        "tools/qce_codegen/ir.py",
        "tools/qce_codegen/low_order_force.py",
        "tools/qce_codegen/production.py",
        "tools/qce_codegen/shell_class.py",
        "tools/qce_codegen/shell_spec.py",
    ):
        assert dependency in source


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
    assert f'qce_run_shell_class_{candidate.name}' in source
    driver = emit_batch_driver((candidate,))
    assert "cudaFree(nullptr)" in driver
    assert f"qce_run_shell_class_{candidate.name}()" in driver


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

    nvcc = os.environ.get("QCE_NVCC")
    if nvcc is None:
        pytest.skip("set QCE_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("QCE_CUDA_ARCH", "sm_90")
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
    if os.environ.get("QCE_NVCC_VERBOSE"):
        print(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    if cuda_architecture == "sm_120" and resource_limits is not None:
        assert_rtx5090_resources(result.stdout + result.stderr, resource_limits)


def test_joint_fock_force_cuda_compiles_when_nvcc_is_configured(tmp_path: Path):
    """Compile the dual-consumer pilot through the real CUDA frontend."""

    nvcc = os.environ.get("QCE_NVCC")
    if nvcc is None:
        pytest.skip("set QCE_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("QCE_CUDA_ARCH", "sm_90")
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

    nvcc = os.environ.get("QCE_NVCC")
    if nvcc is None:
        pytest.skip("set QCE_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("QCE_CUDA_ARCH", "sm_90")
    integral = build_integral_ir(
        DPPP_SPEC,
        consumers=(KernelConsumer.FOCK, KernelConsumer.FORCE),
    )
    schedule = next(
        item
        for item in schedule_candidates(integral)
        if item.kind == ScheduleKind.TILED_COMPONENTS
        and item.component_tile == 64
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

    nvcc = os.environ.get("QCE_NVCC")
    if nvcc is None:
        pytest.skip("set QCE_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("QCE_CUDA_ARCH", "sm_90")
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
def test_f_shell_cuda_compiles_when_nvcc_is_configured(
    tmp_path: Path, spec, consumers
):
    """Compile pair-order-six and tiled f-shell gradients with CUDA 12.9."""

    nvcc = os.environ.get("QCE_NVCC")
    if nvcc is None:
        pytest.skip("set QCE_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("QCE_CUDA_ARCH", "sm_90")
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


def test_psss_shell_task_cuda_compiles_when_nvcc_is_configured(tmp_path: Path):
    """Compile a zero-order ket pair through generated Fock/force lowering."""

    nvcc = os.environ.get("QCE_NVCC")
    if nvcc is None:
        pytest.skip("set QCE_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("QCE_CUDA_ARCH", "sm_90")
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

    nvcc = os.environ.get("QCE_NVCC")
    if nvcc is None:
        pytest.skip("set QCE_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("QCE_CUDA_ARCH", "sm_90")
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


def test_psps_weighted_cuda_compiles_with_zero_stack_when_nvcc_is_configured(
    tmp_path: Path,
):
    """Keep the low-order AOT worker out of the giant-TU stack path."""

    nvcc = os.environ.get("QCE_NVCC")
    if nvcc is None:
        pytest.skip("set QCE_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("QCE_CUDA_ARCH", "sm_90")
    source = tmp_path / "generated_psps_weighted.cu"
    source.write_text(
        _PRODUCTION_PRELUDE.replace(
            '#include "scf/generated_shell_task.hpp"\n', ""
        )
        + "#include <cstddef>\n#include <cstdint>\n"
        + emit_psps_weighted_force_cuda(),
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
            str(tmp_path / "generated_psps_weighted.cubin"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if os.environ.get("QCE_NVCC_VERBOSE"):
        print(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    if cuda_architecture == "sm_120":
        assert_rtx5090_resources(
            result.stdout + result.stderr, RTX5090_PSPS_RESOURCE_LIMITS
        )


def test_ppss_weighted_cuda_compiles_when_nvcc_is_configured(tmp_path: Path):
    """Compile the ppss prototype and reject spills before benchmarking it."""

    nvcc = os.environ.get("QCE_NVCC")
    if nvcc is None:
        pytest.skip("set QCE_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("QCE_CUDA_ARCH", "sm_90")
    source = tmp_path / "generated_ppss_weighted.cu"
    source.write_text(
        _PRODUCTION_PRELUDE.replace(
            '#include "scf/generated_shell_task.hpp"\n', ""
        )
        + "#include <cstddef>\n#include <cstdint>\n"
        + emit_ppss_weighted_force_cuda(),
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
            str(tmp_path / "generated_ppss_weighted.cubin"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if os.environ.get("QCE_NVCC_VERBOSE"):
        print(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    if cuda_architecture == "sm_120":
        assert_rtx5090_resources(
            result.stdout + result.stderr, RTX5090_PPSS_RESOURCE_LIMITS
        )


@pytest.mark.parametrize("shared_coulomb", (True, False))
def test_schedule_knob_cuda_variants_compile_when_nvcc_is_configured(
    tmp_path: Path, shared_coulomb: bool
):
    """Compile both cooperative sharing policies through the real frontend."""

    nvcc = os.environ.get("QCE_NVCC")
    if nvcc is None:
        pytest.skip("set QCE_NVCC to run the generated CUDA compile gate")
    cuda_architecture = os.environ.get("QCE_CUDA_ARCH", "sm_90")
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
    assert "generated_dpds_component_gradient<false>" not in source.split(
        "/** Per-component Fock baseline", maxsplit=1
    )[1]
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
    source = emit_shell_class_oracle_cuda(
        PSPS_SPEC, plan, KernelConsumer.FOCK
    )
    assert "generated_psps_shell_class_fock_rhf_kernel" in source
    assert "generated_psps_shell_class_force_rhf_kernel" not in source


def test_fock_benchmark_runs_when_nvcc_is_configured(tmp_path: Path):
    """Execute the swapped value benchmark and its independent oracle."""

    nvcc = os.environ.get("QCE_NVCC")
    if nvcc is None:
        pytest.skip("set QCE_NVCC to run the generated CUDA benchmark gate")
    cuda_architecture = os.environ.get("QCE_CUDA_ARCH", "sm_90")
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
    assert compile_result.returncode == 0, (
        compile_result.stdout + compile_result.stderr
    )
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
    assert "QCE_PAIR_UNROLL" not in source
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
printf '%s\n' "$child_pid" > "$QCE_TEST_CHILD_PID_FILE"
wait "$child_pid"
""",
        encoding="utf-8",
    )
    fake_nvcc.chmod(0o755)
    monkeypatch.setenv("QCE_TEST_CHILD_PID_FILE", str(child_pid_file))

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


def test_autotune_emits_unique_schedule_variants_and_manifest_records():
    """Keep same-class variants linkable and every winner reproducible."""

    trials = supported_schedule_trials(DPDS_SPEC)
    component_trials = tuple(
        trial
        for trial in trials
        if trial.schedule.kind == ScheduleKind.COMPONENT_LANES
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
        "qce_oracle_dpds_force_component_lanes_b128_t108_w1_"
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
    assert fock_updated["architectures"]["sm_120"]["kernels"][0][
        "consumers"
    ] == ["fock", "force"]


def test_autotune_same_class_variants_link_when_nvcc_is_configured(
    tmp_path: Path,
):
    """Ensure symbol isolation lets one GPU process compare same-class code."""

    nvcc = os.environ.get("QCE_NVCC")
    if nvcc is None:
        pytest.skip("set QCE_NVCC to run the generated CUDA link gate")
    cuda_architecture = os.environ.get("QCE_CUDA_ARCH", "sm_90")
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
def test_benchmark_is_generated_without_shell_specific_harness_code(
    spec, third_offset
):
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
    second = emit_dppp_component_cuda(
        build_dppp_component_kernel("xy", tuple("xyz"))
    )
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
