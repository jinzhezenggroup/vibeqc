"""Bounded reference traces should retain objective convergence evidence."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks._endpoint_progress import EndpointProgress


def test_diagnostic_summary_retains_cycle_cost_and_trajectory_signals(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result.json"
    progress = EndpointProgress(output, {"status": "running"}, 30, True)
    progress.endpoint = "reference/cold"

    progress.checkpoint("get_veff_begin")
    progress.checkpoint("get_veff_end", seconds=3.25)
    progress.checkpoint(
        "scf_cycle",
        cycle=0,
        e_tot=-75.0,
        de=-0.5,
        norm_gorb=4.0,
        norm_ddm=0.8,
    )
    progress.checkpoint("get_veff_begin")
    progress.checkpoint("get_veff_end", seconds=3.5)
    progress.checkpoint(
        "scf_cycle",
        cycle=1,
        e_tot=-74.8,
        de=0.2,
        norm_gorb=5.0,
        norm_ddm=0.6,
    )
    progress.checkpoint(
        "scf_cycle",
        cycle=2,
        e_tot=-75.1,
        de=-0.3,
        norm_gorb=0.2,
        norm_ddm=0.1,
    )

    summary = json.loads(output.read_text())["diagnostic_summary"]
    assert summary["endpoint"] == "reference/cold"
    assert summary["scf_cycles_observed"] == 3
    assert summary["first_cycle"] == 0
    assert summary["last_cycle"] == 2
    assert summary["de_sign_changes"] == 2
    assert summary["get_veff_completed"] == 2
    assert summary["get_veff_total_seconds"] == 6.75
    assert summary["get_veff_mean_seconds"] == 3.375
    assert summary["get_veff_max_seconds"] == 3.5
    assert summary["norm_gorb_latest"] == 0.2
    assert summary["norm_gorb_min"] == 0.2
    assert summary["norm_ddm_latest"] == 0.1
    assert summary["norm_ddm_min"] == 0.1
    assert summary["latest_cycle_values"] == {
        "e_tot": -75.1,
        "de": -0.3,
        "norm_gorb": 0.2,
        "norm_ddm": 0.1,
    }
    assert summary["first_get_veff_begin_elapsed_seconds"] >= 0.0
    assert (
        summary["last_cycle_elapsed_seconds"] >= summary["first_cycle_elapsed_seconds"]
    )
    assert summary["cycle_elapsed_span_seconds"] >= 0.0
    assert summary["mean_cycle_interval_seconds"] >= 0.0


def test_diagnostic_summary_resets_when_endpoint_changes(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    progress = EndpointProgress(output, {}, 30, True)

    progress.endpoint = "reference/cold"
    progress.checkpoint("scf_cycle", cycle=0, de=-1.0, norm_gorb=2.0)
    progress.endpoint = "reference/warm/0"
    progress.checkpoint("scf_cycle", cycle=0, de=0.1, norm_gorb=0.5)

    summary = json.loads(output.read_text())["diagnostic_summary"]
    assert summary["endpoint"] == "reference/warm/0"
    assert summary["scf_cycles_observed"] == 1
    assert summary["first_cycle"] == 0
    assert summary["last_cycle"] == 0
    assert summary["de_sign_changes"] == 0
    assert summary["norm_gorb_min"] == 0.5


def test_nonfinite_cycle_values_stay_out_of_numeric_summary(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    progress = EndpointProgress(output, {}, 30, True)
    progress.endpoint = "reference/cold"

    progress.checkpoint(
        "scf_cycle",
        cycle=0,
        e_tot=-75.0,
        de="nan",
        norm_gorb="inf",
        norm_ddm=0.25,
    )

    record = json.loads(output.read_text())
    assert record["diagnostic_progress"]["de"] == "nan"
    assert record["diagnostic_progress"]["norm_gorb"] == "inf"
    summary = record["diagnostic_summary"]
    assert summary["latest_cycle_values"] == {"e_tot": -75.0, "norm_ddm": 0.25}
    assert summary["de_sign_changes"] == 0
    assert "norm_gorb_min" not in summary
