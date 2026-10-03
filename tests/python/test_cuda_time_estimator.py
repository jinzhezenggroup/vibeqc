"""CPU-only contract and hand-computed oracles for calibrated CUDA timing."""

import json
from dataclasses import replace

import pytest
from generativeqc_compiler.common.cuda_cost_model import (
    StaticCudaCost,
    static_cuda_cost,
)
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.common.cuda_time_estimator import (
    CudaTimingCalibration,
    estimate_cuda_time,
)
from generativeqc_compiler.common.gpu_profitability import GpuProfitability


def _calibration(**overrides: object) -> CudaTimingCalibration:
    values: dict[str, object] = {
        "device": "synthetic-device",
        "architecture": "sm_120",
        "sm_count": 170,
        "workload": "synthetic FP64; FMA=2 ops; streaming semantic bytes",
        "provenance": "unit-test arithmetic, not measured GPU calibration",
        "effective_compute_ops_per_second": 1.0e12,
        "effective_memory_bytes_per_second": 1.0e11,
        "launch_seconds": 1.0e-5,
        "saturation_occupancy": 0.5,
        "uncertainty_fraction": 0.25,
    }
    values.update(overrides)
    return CudaTimingCalibration(**values)


def _cost(**overrides: object) -> StaticCudaCost:
    cost = static_cuda_cost(
        GpuProfitability(
            semantic_traffic_bytes=100_000_000,
            arithmetic_operation_count=1_000_000_000,
            compiled_registers_per_thread=64,
            spill_store_bytes=0,
            spill_load_bytes=0,
            shared_bytes=0,
            launch_count=2,
        ),
        cuda_target_info("sm_120"),
        128,
        grid_blocks=680,
        sm_count=170,
    )
    return replace(cost, **overrides)


def test_roofline_terms_use_total_work_and_add_launch_latency_once() -> None:
    estimate = estimate_cuda_time(_cost(), _calibration())

    # 680 blocks * 128 threads / (170 SMs * 1536 threads/SM) = 1/3.
    # Scale = (1/3)/(1/2) = 2/3; each saturated body term is 1 ms.
    assert estimate.parallelism_fraction == pytest.approx(1.0 / 3.0)
    assert estimate.parallelism_basis == "whole-device"
    assert estimate.parallel_scale == pytest.approx(2.0 / 3.0)
    assert estimate.compute_seconds == pytest.approx(0.0015)
    assert estimate.memory_seconds == pytest.approx(0.0015)
    assert estimate.launch_seconds == pytest.approx(0.00002)
    assert estimate.estimated_seconds == pytest.approx(0.00152)
    assert estimate.lower_seconds == pytest.approx(0.00114)
    assert estimate.upper_seconds == pytest.approx(0.00190)
    assert estimate.bottleneck == "balanced"
    assert any("linearly reduced" in item for item in estimate.diagnostics)


@pytest.mark.parametrize(
    ("field", "component", "diagnostic"),
    [
        ("arithmetic_operation_count", "compute_seconds", "arithmetic operation count"),
        ("semantic_traffic_bytes", "memory_seconds", "semantic traffic bytes"),
        ("launch_count", "launch_seconds", "launch count"),
    ],
)
def test_missing_evidence_preserves_independent_components(
    field: str,
    component: str,
    diagnostic: str,
) -> None:
    estimate = estimate_cuda_time(_cost(**{field: None}), _calibration())

    assert estimate.estimated_seconds is None
    assert estimate.lower_seconds is estimate.upper_seconds is None
    assert estimate.bottleneck is None
    assert getattr(estimate, component) is None
    for other in {"compute_seconds", "memory_seconds", "launch_seconds"} - {component}:
        assert getattr(estimate, other) is not None
    assert any(f"{diagnostic} is unavailable" in d for d in estimate.diagnostics)


@pytest.mark.parametrize("missing", ["sm_count", "grid_blocks", "both"])
def test_global_parallelism_required_unless_fallback_explicit(missing: str) -> None:
    cost = _cost(
        device_occupancy_upper_bound=None,
        **(
            {"sm_count": None, "grid_blocks": None}
            if missing == "both"
            else {missing: None}
        ),
    )
    unknown = estimate_cuda_time(cost, _calibration())
    assert unknown.estimated_seconds is None
    assert unknown.parallelism_fraction is None
    assert unknown.launch_seconds == pytest.approx(0.00002)
    assert any(
        "whole-device parallelism is unavailable" in d for d in unknown.diagnostics
    )

    fallback = estimate_cuda_time(cost, _calibration(), allow_per_sm_fallback=True)
    assert fallback.estimated_seconds == pytest.approx(0.00102)
    assert fallback.parallelism_basis == "per-sm-fallback"
    assert fallback.parallelism_fraction == cost.occupancy_upper_bound
    assert any(
        "without a global underfill correction" in d for d in fallback.diagnostics
    )


@pytest.mark.parametrize("allow_fallback", [False, True])
@pytest.mark.parametrize("grid", [0, 680])
def test_impossible_or_empty_grid_never_becomes_executable(
    grid: int, allow_fallback: bool
) -> None:
    cost = _cost(
        grid_blocks=grid,
        sm_count=None,
        device_occupancy_upper_bound=None,
        resident_blocks_per_sm_upper_bound=0 if grid else 8,
    )
    estimate = estimate_cuda_time(
        cost, _calibration(), allow_per_sm_fallback=allow_fallback
    )
    assert estimate.estimated_seconds is None
    assert estimate.parallelism_fraction == 0
    assert any("no executable parallelism" in d for d in estimate.diagnostics)


@pytest.mark.parametrize("spills", [None, 200])
def test_static_spills_cannot_be_added_as_dynamic_traffic(spills: int | None) -> None:
    cost = _cost(
        semantic_traffic_bytes=100, arithmetic_operation_count=1, spill_bytes=spills
    )
    calibration = _calibration(effective_memory_bytes_per_second=100.0)
    unknown = estimate_cuda_time(cost, calibration)
    assert unknown.memory_seconds is unknown.estimated_seconds is None
    assert unknown.compute_seconds is not None
    assert any("PTXAS spill bytes cannot be used" in d for d in unknown.diagnostics)

    # Dynamic traffic already covers all threads, loop iterations, and launches.
    # (100 semantic + 2000 dynamic spill) / 100 B/s / (2/3) = 31.5 s.
    explicit = estimate_cuda_time(cost, calibration, spill_traffic_bytes=2000)
    assert explicit.memory_seconds == pytest.approx(31.5)
    assert explicit.spill_traffic_bytes == 2000
    assert explicit.bottleneck == "memory"


def test_explicit_zero_spill_traffic_is_not_missing_evidence() -> None:
    estimate = estimate_cuda_time(
        _cost(spill_bytes=None), _calibration(), spill_traffic_bytes=0
    )
    assert estimate.estimated_seconds == pytest.approx(0.00152)


def test_static_evidence_caveats_survive_in_self_contained_payload() -> None:
    cost = static_cuda_cost(
        GpuProfitability(
            semantic_traffic_bytes=1, arithmetic_operation_count=1, launch_count=1
        ),
        cuda_target_info("sm_120"),
        128,
        grid_blocks=1360,
        sm_count=170,
    )
    calibration = _calibration()
    estimate = estimate_cuda_time(cost, calibration, spill_traffic_bytes=0)
    assert set(cost.diagnostics).issubset(estimate.diagnostics)
    payload = json.loads(json.dumps(estimate.to_payload(), allow_nan=False))
    assert payload["calibration"] == calibration.to_payload()
    assert payload["cost"]["evidence_stage"] == "static"
    assert payload["cost"]["arithmetic_operation_count"] == 1
    assert payload["model"] == "roofline-linear-occupancy.v1"
    assert "not an endpoint prediction" in payload["scope"]
    assert payload["device"] == calibration.device


def test_work_counts_and_repeated_launches_scale_together() -> None:
    base = _cost()
    one = estimate_cuda_time(base, _calibration())
    repeated = estimate_cuda_time(
        replace(
            base,
            arithmetic_operation_count=3_000_000_000,
            semantic_traffic_bytes=300_000_000,
            launch_count=6,
        ),
        _calibration(),
    )
    assert repeated.estimated_seconds == pytest.approx(3 * one.estimated_seconds)
    assert repeated.upper_seconds == pytest.approx(3 * one.upper_seconds)


def test_underfill_is_monotone_until_saturation() -> None:
    times = []
    for grid in (1, 40, 680, 1020, 1360, 10000):
        cost = static_cuda_cost(
            GpuProfitability(
                semantic_traffic_bytes=100_000_000,
                arithmetic_operation_count=1_000_000_000,
                compiled_registers_per_thread=64,
                shared_bytes=0,
                spill_store_bytes=0,
                spill_load_bytes=0,
                launch_count=2,
            ),
            cuda_target_info("sm_120"),
            128,
            grid_blocks=grid,
            sm_count=170,
        )
        times.append(estimate_cuda_time(cost, _calibration()).estimated_seconds)
    assert times == sorted(times, reverse=True)
    assert times[-3:] == pytest.approx([0.00102] * 3)


@pytest.mark.parametrize(
    ("operations", "traffic", "bottleneck"),
    [
        (2_000_000_000, 1, "compute"),
        (1, 200_000_000, "memory"),
        (1, 1, "launch"),
        (0, 0, "launch"),
    ],
)
def test_reports_dominant_cost_including_launch(
    operations: int, traffic: int, bottleneck: str
) -> None:
    result = estimate_cuda_time(
        _cost(arithmetic_operation_count=operations, semantic_traffic_bytes=traffic),
        _calibration(),
    )
    assert result.bottleneck == bottleneck


@pytest.mark.parametrize("uncertainty", [0.0, 1.0])
def test_engineering_interval_endpoints(uncertainty: float) -> None:
    result = estimate_cuda_time(_cost(), _calibration(uncertainty_fraction=uncertainty))
    assert result.lower_seconds == pytest.approx(0.00152 * (1 - uncertainty))
    assert result.upper_seconds == pytest.approx(0.00152 * (1 + uncertainty))


def test_known_noop_needs_no_resource_evidence() -> None:
    result = estimate_cuda_time(
        _cost(
            launch_count=0,
            arithmetic_operation_count=0,
            semantic_traffic_bytes=0,
            grid_blocks=0,
            sm_count=None,
            device_occupancy_upper_bound=None,
            spill_bytes=None,
        ),
        _calibration(),
    )
    assert (
        result.estimated_seconds == result.lower_seconds == result.upper_seconds == 0.0
    )
    assert result.bottleneck == "none"


def test_zero_launches_with_unknown_work_is_not_a_known_noop() -> None:
    assert (
        estimate_cuda_time(
            _cost(
                launch_count=0,
                arithmetic_operation_count=None,
                semantic_traffic_bytes=None,
            ),
            _calibration(),
        ).estimated_seconds
        is None
    )


@pytest.mark.parametrize(
    "field", ["arithmetic_operation_count", "semantic_traffic_bytes"]
)
def test_noop_cannot_hide_nonzero_work(field: str) -> None:
    cost = _cost(launch_count=0, arithmetic_operation_count=0, semantic_traffic_bytes=0)
    with pytest.raises(ValueError, match="zero launch count"):
        estimate_cuda_time(replace(cost, **{field: 1}), _calibration())


@pytest.mark.parametrize(
    "overrides",
    [{"architecture": "sm_80"}, {"sm_count": 100}, {"grid_blocks": None}],
)
def test_contradictory_device_evidence_fails_closed(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        estimate_cuda_time(_cost(**overrides), _calibration())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("arithmetic_operation_count", -1),
        ("semantic_traffic_bytes", True),
        ("launch_count", 1.5),
        ("spill_bytes", -1),
        ("grid_blocks", False),
        ("occupancy_upper_bound", float("nan")),
        ("device_occupancy_upper_bound", float("inf")),
        ("device_occupancy_upper_bound", -0.1),
        ("device_occupancy_upper_bound", 1.1),
    ],
)
def test_malformed_static_records_are_rejected(field: str, value: object) -> None:
    with pytest.raises((ValueError, TypeError)):
        estimate_cuda_time(_cost(**{field: value}), _calibration())


@pytest.mark.parametrize(
    "field",
    [
        "effective_compute_ops_per_second",
        "effective_memory_bytes_per_second",
        "launch_seconds",
        "saturation_occupancy",
        "uncertainty_fraction",
    ],
)
@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), -1.0, True, "1.0", 10**400]
)
def test_invalid_numeric_calibration_fails_closed(field: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError), match=field):
        _calibration(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("effective_compute_ops_per_second", 0),
        ("effective_memory_bytes_per_second", 0),
        ("launch_seconds", 0),
        ("saturation_occupancy", 0),
        ("saturation_occupancy", 1.1),
        ("uncertainty_fraction", 1.1),
        ("device", ""),
        ("architecture", "120"),
        ("workload", " "),
        ("provenance", None),
        ("sm_count", 0),
        ("sm_count", True),
        ("sm_count", None),
    ],
)
def test_calibration_requires_valid_explicit_identity_and_assumptions(
    field: str, value: object
) -> None:
    with pytest.raises((TypeError, ValueError), match=field):
        _calibration(**{field: value})


@pytest.mark.parametrize(
    ("cost_overrides", "calibration_overrides"),
    [
        ({"arithmetic_operation_count": 10**400}, {}),
        ({}, {"effective_compute_ops_per_second": 5e-324}),
        ({}, {"effective_memory_bytes_per_second": 5e-324}),
        ({"launch_count": 10**400}, {}),
        ({}, {"launch_seconds": 1e308}),
        ({"launch_count": 1}, {"launch_seconds": 1e308, "uncertainty_fraction": 1.0}),
    ],
)
def test_overflow_stays_unknown_and_payload_is_strict_json(
    cost_overrides: dict[str, object], calibration_overrides: dict[str, object]
) -> None:
    result = estimate_cuda_time(
        _cost(**cost_overrides), _calibration(**calibration_overrides)
    )
    assert result.estimated_seconds is None
    assert any("finite timing range" in d for d in result.diagnostics)
    json.dumps(result.to_payload(), allow_nan=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spill_traffic_bytes": -1},
        {"spill_traffic_bytes": True},
        {"allow_per_sm_fallback": 1},
    ],
)
def test_invalid_estimation_options_fail_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        estimate_cuda_time(_cost(), _calibration(), **kwargs)


def test_estimator_requires_typed_inputs() -> None:
    with pytest.raises(TypeError, match="StaticCudaCost"):
        estimate_cuda_time({}, _calibration())
    with pytest.raises(TypeError, match="CudaTimingCalibration"):
        estimate_cuda_time(_cost(), {})


def test_refined_model_accounts_for_crossover_and_fixed_batch_time() -> None:
    cost = static_cuda_cost(
        GpuProfitability(
            arithmetic_operation_count=200,
            semantic_traffic_bytes=100,
            launch_count=1,
            compiled_registers_per_thread=32,
            shared_bytes=0,
            spill_store_bytes=0,
            spill_load_bytes=0,
        ),
        cuda_target_info("sm_120"),
        128,
        grid_blocks=2040,
        sm_count=170,
    )
    calibration = _calibration(
        model="roofline-calibrated-overlap.v2",
        effective_compute_ops_per_second=100,
        effective_memory_bytes_per_second=100,
        launch_seconds=0.02,
        batch_seconds=0.03,
        memory_throughput_curve=((0, 0), (1, 1)),
        crossover_penalty_curve=((0, 0.4), (1, 0.4)),
    )
    estimate = estimate_cuda_time(cost, calibration)
    # C=2, M=1, crossover=0.4*1^2/2=0.2, launch=0.02, batch=0.03.
    assert estimate.estimated_seconds == pytest.approx(2.25)
    assert estimate.overlap_seconds == pytest.approx(0.8)
    assert estimate.batch_seconds == 0.03
    payload = estimate.to_payload()
    assert payload["model"] == "roofline-calibrated-overlap.v2"
    assert payload["calibration"]["schema"].endswith(".v2")
    json.dumps(payload, allow_nan=False)


def test_refined_memory_curve_is_independent_of_compute_saturation() -> None:
    estimate = estimate_cuda_time(
        _cost(),
        _calibration(
            model="roofline-calibrated-overlap.v2",
            memory_throughput_curve=((0, 0), (0.25, 0.5), (1, 1)),
        ),
    )
    # At occupancy 1/3, interpolate memory scale to 5/9; compute remains 2/3.
    assert estimate.parallel_scale == pytest.approx(2 / 3)
    assert estimate.memory_parallel_scale == pytest.approx(5 / 9)
    assert estimate.compute_seconds == pytest.approx(0.0015)
    assert estimate.memory_seconds == pytest.approx(0.0018)


@pytest.mark.parametrize(
    ("grid", "body"), [(85, 1.0), (170, 1.0), (255, 2.0), (340, 2.0)]
)
def test_uniform_compute_blocks_pay_for_partial_sm_wave(grid: int, body: float) -> None:
    cost = static_cuda_cost(
        GpuProfitability(
            arithmetic_operation_count=grid * 100,
            semantic_traffic_bytes=0,
            launch_count=1,
            compiled_registers_per_thread=32,
            shared_bytes=0,
            spill_store_bytes=0,
            spill_load_bytes=0,
        ),
        cuda_target_info("sm_120"),
        128,
        grid_blocks=grid,
        sm_count=170,
    )
    calibration = _calibration(
        model="roofline-calibrated-overlap.v2",
        compute_wave_correction=True,
        saturation_occupancy=1 / 12,
        effective_compute_ops_per_second=17000,
    )
    estimate = estimate_cuda_time(cost, calibration)
    assert estimate.compute_seconds == pytest.approx(body)


def test_refined_noop_does_not_charge_batch_synchronization() -> None:
    result = estimate_cuda_time(
        _cost(launch_count=0, arithmetic_operation_count=0, semantic_traffic_bytes=0),
        _calibration(model="roofline-calibrated-overlap.v2", batch_seconds=1.0),
    )
    assert result.estimated_seconds == result.batch_seconds == 0.0


@pytest.mark.parametrize(
    "curve",
    [
        ((0, 0), (0.5, 0.8), (0.75, 0.7), (1, 1)),
        ((0, 0), (0.5, 0.5), (0.5, 0.7), (1, 1)),
        ((0, 0), (1, 0.9)),
        ((0, 0), (1, float("nan"))),
        ((0, 0), (1, True)),
        ((0, 0), (0.5, 0), (1, 1)),
    ],
)
def test_refined_memory_curve_rejects_invalid_or_nonmonotone_calibration(
    curve: tuple,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        _calibration(
            model="roofline-calibrated-overlap.v2", memory_throughput_curve=curve
        )


def test_refined_terms_cannot_silently_change_a_v1_calibration() -> None:
    with pytest.raises(ValueError, match="require the v2 model"):
        _calibration(batch_seconds=0.1)
    assert "model" not in _calibration().to_payload()
