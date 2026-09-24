"""Missing/invalid journal data must not turn into fabricated numeric evidence."""

from __future__ import annotations

import json
import math

import pytest

from benchmarks.scf_progress_evidence import analyze_progress_events


def _report(*events: dict) -> dict:
    return analyze_progress_events([{"event": "begin", "endpoint": "cold"}, *events])[0]


@pytest.mark.parametrize(
    "invalid", [None, float("nan"), float("inf"), "nan", -1.0, True, 10**400]
)
def test_invalid_cycle_time_does_not_bridge_two_intervals(invalid: object) -> None:
    report = _report(
        {"event": "scf_cycle", "elapsed_seconds": 1.0, "cycle": 0},
        {"event": "scf_cycle", "elapsed_seconds": invalid, "cycle": 1},
        {"event": "scf_cycle", "elapsed_seconds": 3.0, "cycle": 2},
    )
    assert report["scf_cycles_observed"] == 3
    assert "cycle_interval_mean_seconds" not in report
    assert report["first_cycle_elapsed_seconds"] == 1.0
    assert report["last_cycle_elapsed_seconds"] == 3.0
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("invalid", [-2.0, float("nan"), float("inf"), 10**400])
def test_invalid_norm_cannot_be_a_residual_minimum(invalid: object) -> None:
    report = _report(
        {"event": "scf_cycle", "norm_gorb": 2.0, "cycle": 0},
        {"event": "scf_cycle", "norm_gorb": invalid, "cycle": 1},
        {"event": "scf_cycle", "norm_gorb": 1.0, "cycle": 2},
    )
    assert report["residuals"]["norm_gorb"]["samples"] == 2
    assert report["residuals"]["norm_gorb"]["minimum"] == 1.0


def test_negative_elapsed_is_not_published() -> None:
    report = _report({"event": "scf_cycle", "elapsed_seconds": -1.0})
    assert "first_cycle_elapsed_seconds" not in report
    assert "last_cycle_elapsed_seconds" not in report


def test_finite_costs_with_overflowing_sum_still_have_finite_mean() -> None:
    report = _report(*({"event": "get_veff_end", "seconds": 1e308} for _ in range(3)))
    assert report["get_veff_mean_seconds"] == pytest.approx(1e308)
    assert report["get_veff_max_seconds"] == 1e308
    assert report["get_veff_total_overflow"] is True
    assert "get_veff_total_seconds" not in report
    json.dumps(report, allow_nan=False)


def test_normal_zero_and_positive_intervals_retain_statistics() -> None:
    report = _report(
        {"event": "get_veff_end", "seconds": 0},
        {"event": "get_veff_end", "seconds": 4},
        {"event": "scf_cycle", "elapsed_seconds": 1.0},
        {"event": "scf_cycle", "elapsed_seconds": 1.0},
        {"event": "scf_cycle", "elapsed_seconds": 5.0},
    )
    assert report["cycle_interval_mean_seconds"] == 2.0
    assert report["cycle_interval_min_seconds"] == 0.0
    assert report["cycle_interval_max_seconds"] == 4.0
    assert report["get_veff_total_seconds"] == 4.0
    assert report["get_veff_mean_seconds"] == 2.0
    assert "get_veff_total_overflow" not in report
    assert report["benchmark_qualification"] == "diagnostic-only"
    assert math.isfinite(report["get_veff_mean_seconds"])
