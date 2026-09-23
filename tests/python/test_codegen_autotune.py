"""Focused autotune tests extracted from the legacy codegen suite.

See issue #489.
"""

from __future__ import annotations

import itertools
import json
import os
import re
import subprocess
import time
import typing
from pathlib import Path
from types import SimpleNamespace

import pytest
from vibeqc_compiler.integral import (
    DPDS_SPEC,
    DPPP_SPEC,
    FUSED_SHELL_SPEC_BY_NAME,
    PSPS_SPEC,
    AlgebraForm,
    AlgebraFusion,
    AlgebraOrdering,
    AlgebraPlacement,
    ContractionSpec,
    KernelConsumer,
    OperatorFamily,
    OperatorSpec,
    ScheduleKind,
    TranslationInvariant,
    build_integral_ir,
    build_shell_class_contraction_kernel,
    cuda_target_info,
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
from vibeqc_compiler.integral.batch_benchmark import KernelResources

TEST_CUDA_TARGET = cuda_target_info("sm_120")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


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
        "vibeqc_compiler.common.cuda_adapter.CudaCompilerAdapter.link",
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
        "vibeqc_compiler.common.cuda_adapter.CudaBenchmarkExecutor.run",
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
