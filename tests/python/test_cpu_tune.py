"""CPU schedule cost model and bounded measured autotuning."""

import shutil
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.cpu_dispatch import (
    cpu_target_supported,
    detect_cpu_features,
)
from vibeqc_compiler.common.cpu_target import (
    AVX2_FMA_TARGET,
    GENERIC_CPU_TARGET,
)
from vibeqc_compiler.integral.cpu_tune import (
    CpuTuneLimits,
    CpuTuneSchedule,
    cpu_static_cost,
    cpu_tune_candidates,
    tune_cpu_first_derivative_shell,
)
from vibeqc_compiler.integral.first_derivatives_execute import (
    FirstDerivativeShellEvaluator,
    compile_first_derivative_shell,
)
from vibeqc_compiler.integral.weighted_eri import build_weighted_eri_ir


@pytest.fixture(scope="module")
def compiler():
    executable = shutil.which("c++")
    if executable is None:
        pytest.skip("CPU C++ compiler required")
    return CppCompilerAdapter(Path(executable))


def _fixture():
    centers = np.array(
        [
            [0.13, -0.31, 0.24],
            [-0.43, 0.27, 0.51],
            [0.68, -0.14, -0.22],
            [-0.21, 0.48, -0.63],
        ]
    )
    primitives = (
        (
            (1.70, 0.30),
            (0.91, -0.24),
            (0.53, 0.61),
            (0.29, 0.18),
            (0.12, -0.08),
        ),
        ((0.80, 1.0),),
        ((1.10, 1.0),),
        ((0.90, 1.0),),
    )
    return primitives, centers


def test_cpu_tune_candidates_cover_targets_and_static_resources():
    ir = build_weighted_eri_ir((1, 0, 0, 0))
    limits = CpuTuneLimits(maximum_candidates=8)
    candidates = cpu_tune_candidates(
        ir,
        targets=(GENERIC_CPU_TARGET, AVX2_FMA_TARGET),
        limits=limits,
    )
    assert candidates[0].target == GENERIC_CPU_TARGET
    assert candidates[1].target == AVX2_FMA_TARGET
    assert len(candidates) <= limits.maximum_candidates
    assert len({str(candidate.to_payload()) for candidate in candidates}) == len(
        candidates
    )

    generic = CpuTuneSchedule(
        GENERIC_CPU_TARGET,
        candidates[0].lane,
        ir.signature.component_count,
    )
    cost = cpu_static_cost(ir, generic, record_count=5)
    assert cost["vector_width_fp64"] == 1
    assert cost["lane_utilization"] == 1.0
    assert cost["tail_fraction"] == 0.0
    assert cost["estimated_peak_live_scalar_values"] > 0
    assert cost["estimated_runtime_working_set_bytes"] > 0
    assert cost["estimated_logical_load_store_bytes"] > 0
    assert cost["generated_source_bytes"] > 0
    assert "l1_data_bytes" in cost["cache"]
    assert cost["register_spill_scope"]

    avx2 = cpu_static_cost(ir, candidates[1], record_count=5)
    assert avx2["vector_width_fp64"] == 4
    assert avx2["lane_utilization"] == pytest.approx(5 / 8)
    assert avx2["tail_fraction"] == pytest.approx(3 / 8)
    assert avx2["estimated_peak_live_vector_values"] > 0
    assert (
        avx2["aos_to_soa_bytes_per_lane_tile"] > cost["aos_to_soa_bytes_per_lane_tile"]
    )


def test_cpu_tune_dpss_space_includes_component_tiling():
    ir = build_weighted_eri_ir((2, 1, 0, 0))
    candidates = cpu_tune_candidates(
        ir,
        targets=(GENERIC_CPU_TARGET, AVX2_FMA_TARGET),
        limits=CpuTuneLimits(maximum_candidates=16),
    )
    tile_sizes = {candidate.component_tile_size for candidate in candidates}
    assert ir.signature.component_count in tile_sizes
    assert any(tile < ir.signature.component_count for tile in tile_sizes)


def test_cpu_autotune_numerical_gate_timing_and_parallel_evidence(compiler, tmp_path):
    ir = build_weighted_eri_ir((1, 0, 0, 0))
    primitives, centers = _fixture()
    reference_artifact = compile_first_derivative_shell(
        ir,
        compiler,
        tmp_path / "reference",
        tile_size=3,
    )
    reference = FirstDerivativeShellEvaluator(
        reference_artifact,
        record_capacity=7,
        budget_bytes=8 << 20,
    ).contract(primitives, centers)

    runtime = detect_cpu_features()
    targets = [GENERIC_CPU_TARGET]
    if cpu_target_supported(AVX2_FMA_TARGET, runtime):
        targets.append(AVX2_FMA_TARGET)
    limits = CpuTuneLimits(
        maximum_candidates=len(targets),
        repeats=5,
        parallel_tasks=2,
        parallel_workers=(1, 2),
    )
    result = tune_cpu_first_derivative_shell(
        ir,
        compiler,
        tmp_path / "tune",
        primitives=primitives,
        centers=centers,
        independent_reference=reference,
        reference_identity="unit-test-independent-scalar-path-v1",
        runtime=runtime,
        targets=tuple(targets),
        limits=limits,
    )
    assert result["schema"] == "vibeqc.cpu-autotune.v1"
    assert result["selected"]["selection_identity"]
    assert result["selected"]["program_identity"]
    assert result["selected"]["compiler_artifact_keys"]
    assert result["selected"]["candidate"]["target"]["name"] in {
        target.name for target in targets
    }

    ready = [row for row in result["candidates"] if row["status"] == "ready"]
    assert len(ready) == len(targets)
    assert all(row["numerical"]["passed"] for row in ready)
    assert all(row["compiled_resources"]["artifact_keys"] for row in ready)
    assert all(row["compiled_resources"]["cold_load_seconds"] >= 0 for row in ready)
    for row in ready:
        static = row["static_resources"]
        actual_bytes = row["compiled_resources"]["numeric_storage_bytes"]
        assert static["schema"] == "vibeqc.cpu.static-cost.v2"
        assert actual_bytes == static["estimated_numeric_storage_bytes"]
        assert actual_bytes <= static["estimated_runtime_working_set_bytes"]
        assert (
            static["estimated_runtime_working_set_bytes"]
            <= limits.maximum_working_set_bytes
        )
    if len(targets) > 1:
        nonbaseline = [
            row for row in ready if row["candidate"]["target"]["name"] != "generic"
        ]
        assert nonbaseline
        assert "comparison" in nonbaseline[0]
        assert len(nonbaseline[0]["timing_samples"]) == 10

    parallel = result["parallel_interaction"]
    assert [row["workers"] for row in parallel["baseline"]] == [1, 2]
    assert [row["workers"] for row in parallel["selected"]] == [1, 2]
    assert all(row["systems_per_second"] > 0 for row in parallel["selected"])
    if len(targets) > 1:
        assert (
            parallel["best_simd"]["candidate"]["target"]["name"] == AVX2_FMA_TARGET.name
        )
        assert [row["workers"] for row in parallel["best_simd"]["measurements"]] == [
            1,
            2,
        ]


def test_cpu_autotune_rejects_bad_reference_before_compilation(compiler, tmp_path):
    ir = build_weighted_eri_ir((1, 0, 0, 0))
    primitives, centers = _fixture()
    with pytest.raises(ValueError, match="independent reference"):
        tune_cpu_first_derivative_shell(
            ir,
            compiler,
            tmp_path,
            primitives=primitives,
            centers=centers,
            independent_reference=np.zeros((1, 1)),
            reference_identity="bad-shape",
            targets=(GENERIC_CPU_TARGET,),
            limits=CpuTuneLimits(maximum_candidates=1),
        )


def test_cpu_autotune_rejects_numerically_wrong_reference(compiler, tmp_path):
    ir = build_weighted_eri_ir((1, 0, 0, 0))
    primitives, centers = _fixture()
    wrong = np.zeros((ir.signature.component_count, 13))
    with pytest.raises(RuntimeError, match="scalar baseline"):
        tune_cpu_first_derivative_shell(
            ir,
            compiler,
            tmp_path,
            primitives=primitives,
            centers=centers,
            independent_reference=wrong,
            reference_identity="deliberately-wrong-reference-v1",
            targets=(GENERIC_CPU_TARGET,),
            limits=CpuTuneLimits(
                maximum_candidates=1,
                repeats=5,
                parallel_tasks=2,
                parallel_workers=(1,),
            ),
        )


def test_cpu_static_cost_accounts_for_record_storage():
    """The complete consumer owns its AoS record buffer, not only SIMD lanes."""
    ir = build_weighted_eri_ir((1, 0, 0, 0))
    schedule = cpu_tune_candidates(ir, targets=(GENERIC_CPU_TARGET,))[0]
    small = cpu_static_cost(ir, schedule, record_count=5)
    large = cpu_static_cost(ir, schedule, record_count=4096)
    assert (
        large["estimated_runtime_working_set_bytes"]
        > small["estimated_runtime_working_set_bytes"]
    )
    # Four exponents, twelve coordinates and one coefficient per record.
    assert large["estimated_runtime_working_set_bytes"] >= 4096 * 17 * 8


def test_cpu_autotune_rejects_record_budget_before_compilation(monkeypatch, tmp_path):
    import vibeqc_compiler.integral.cpu_tune as tuning

    ir = build_weighted_eri_ir((1, 0, 0, 0))
    primitives, centers = _fixture()

    def forbidden_compile(*args, **kwargs):
        raise AssertionError("insufficient record storage reached compilation")

    monkeypatch.setattr(
        tuning, "compile_first_derivative_cpu_lane_shell", forbidden_compile
    )
    with pytest.raises(RuntimeError, match="scalar baseline"):
        tune_cpu_first_derivative_shell(
            ir,
            None,
            tmp_path,
            primitives=primitives,
            centers=centers,
            independent_reference=np.zeros((ir.signature.component_count, 13)),
            reference_identity="preflight-must-precede-numerical-gate",
            targets=(GENERIC_CPU_TARGET,),
            limits=CpuTuneLimits(maximum_candidates=1, maximum_working_set_bytes=1000),
        )


@pytest.mark.parametrize(
    "field",
    [
        "maximum_candidates",
        "repeats",
        "maximum_source_bytes",
        "maximum_working_set_bytes",
        "parallel_tasks",
    ],
)
@pytest.mark.parametrize("value", [True, 8.5, float("nan"), float("inf")])
def test_cpu_tune_integer_limits_reject_lossy_or_nonfinite_values(field, value):
    with pytest.raises(ValueError):
        CpuTuneLimits(**{field: value})


@pytest.mark.parametrize(
    "value",
    [True, "30", float("nan"), float("inf"), -float("inf"), 0, -1],
)
def test_cpu_tune_compile_limit_must_be_finite_positive_real(value):
    with pytest.raises(ValueError):
        CpuTuneLimits(maximum_compile_seconds=value)


@pytest.mark.parametrize("value", [1, 0.5, 30.0])
def test_cpu_tune_accepts_finite_integer_or_fractional_compile_seconds(value):
    assert CpuTuneLimits(maximum_compile_seconds=value).maximum_compile_seconds == value
