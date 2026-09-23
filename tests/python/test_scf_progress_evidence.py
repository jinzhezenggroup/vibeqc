"""SCF progress evidence stays objective, bounded, and invocation-local."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from benchmarks.scf_progress_evidence import (
    analyze_progress_events,
    read_progress_journal,
)

if TYPE_CHECKING:
    from pathlib import Path


def _event(event: str, elapsed: float, **values: object) -> dict[str, object]:
    return {
        "event": event,
        "endpoint": "reference/cold",
        "elapsed_seconds": elapsed,
        **values,
    }


def test_analysis_separates_setup_cost_oscillation_and_residual_trend() -> None:
    events = [
        _event("begin", 0.0),
        _event("get_veff_begin", 0.5),
        _event("get_veff_end", 4.5, seconds=4.0),
        _event("scf_cycle", 8.0, cycle=0, de=-0.5, norm_gorb=3.0, norm_ddm=0.9),
        _event("get_veff_begin", 8.1),
        _event("get_veff_end", 11.1, seconds=3.0),
        _event("scf_cycle", 11.4, cycle=1, de=0.2, norm_gorb=4.0, norm_ddm=0.8),
        _event("scf_cycle", 14.9, cycle=2, de=-0.1, norm_gorb=0.1, norm_ddm=0.8),
        _event("timeout", 60.0),
    ]

    report = analyze_progress_events(events)[0]
    assert report["benchmark_qualification"] == "diagnostic-only"
    assert report["terminal_event"] == "timeout"
    assert report["setup_before_first_get_veff_seconds"] == 0.5
    assert report["get_veff_completed"] == 2
    assert report["get_veff_mean_seconds"] == 3.5
    assert report["get_veff_max_seconds"] == 4.0
    assert report["cycle_interval_min_seconds"] == pytest.approx(3.4)
    assert report["cycle_interval_max_seconds"] == pytest.approx(3.5)
    assert report["de_sign_changes"] == 2
    assert report["residuals"]["norm_gorb"] == {
        "samples": 3,
        "first": 3.0,
        "latest": 0.1,
        "minimum": 0.1,
        "first_cycle": 0,
        "latest_cycle": 2,
        "minimum_cycle": 2,
        "improving_steps": 1,
        "non_improving_steps": 1,
        "longest_non_improving_run": 1,
    }
    assert report["residuals"]["norm_ddm"]["non_improving_steps"] == 1


def test_repeated_endpoint_labels_stay_in_separate_invocations() -> None:
    events = [
        _event("begin", 0.0),
        _event("scf_cycle", 2.0, cycle=0, de=-1.0, norm_gorb=2.0),
        _event("end", 3.0),
        _event("begin", 0.0),
        _event("scf_cycle", 1.0, cycle=0, de=0.1, norm_gorb=0.5),
        _event("failed", 1.5),
    ]

    reports = analyze_progress_events(events)
    assert len(reports) == 2
    assert reports[0]["terminal_event"] == "end"
    assert reports[0]["residuals"]["norm_gorb"]["first"] == 2.0
    assert reports[1]["terminal_event"] == "failed"
    assert reports[1]["residuals"]["norm_gorb"]["first"] == 0.5
    assert reports[0]["benchmark_qualification"] == "diagnostic-only"
    assert reports[1]["benchmark_qualification"] == "diagnostic-only"


def test_nonfinite_and_negative_costs_do_not_enter_numeric_evidence() -> None:
    events = [
        _event("begin", 0.0),
        _event("get_veff_begin", -1.0),
        _event("get_veff_end", 1.0, seconds=-2.0),
        _event("scf_cycle", 2.0, cycle=0, de="nan", norm_gorb="inf"),
        _event("scf_cycle", 1.0, cycle=1, de=0.0, norm_ddm=0.4),
    ]

    report = analyze_progress_events(events)[0]
    assert "setup_before_first_get_veff_seconds" not in report
    assert report["get_veff_completed"] == 0
    assert "get_veff_mean_seconds" not in report
    assert "cycle_interval_mean_seconds" not in report
    assert report["de_sign_changes"] == 0
    assert "norm_gorb" not in report["residuals"]
    assert report["residuals"]["norm_ddm"]["samples"] == 1


def test_reader_reports_exact_bad_json_line(tmp_path: Path) -> None:
    journal = tmp_path / "progress.jsonl"
    journal.write_text(
        json.dumps(_event("begin", 0.0)) + "\n" + "{bad\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"progress\.jsonl:2: invalid progress JSON"):
        read_progress_journal(journal)


def test_reader_rejects_non_object_event(tmp_path: Path) -> None:
    journal = tmp_path / "progress.jsonl"
    journal.write_text("[]\n", encoding="utf-8")

    with pytest.raises(
        TypeError, match=r"progress\.jsonl:1: progress event must be an object"
    ):
        read_progress_journal(journal)


def test_analysis_rejects_mixed_endpoint_segment() -> None:
    events = [
        _event("begin", 0.0),
        {
            "event": "scf_cycle",
            "endpoint": "reference/warm/0",
            "elapsed_seconds": 1.0,
            "cycle": 0,
        },
    ]

    with pytest.raises(ValueError, match="mixes endpoint labels"):
        analyze_progress_events(events)
