"""Every measured invocation resets current diagnostics before its watchdog starts."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from benchmarks import _endpoint_progress as progress_module

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def fake_watchdog(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    spawned: list[dict] = []

    def process(**kwargs: object) -> SimpleNamespace:
        def start() -> None:
            args = kwargs["args"]
            spawned.append(json.loads(args[3].read_text()))

        return SimpleNamespace(
            start=start, join=lambda **kwargs: None, is_alive=lambda: False
        )

    connection = SimpleNamespace(close=lambda: None, poll=lambda seconds: False)
    context = SimpleNamespace(
        Pipe=lambda **kwargs: (connection, connection), Process=process
    )
    monkeypatch.setattr(progress_module.mp, "get_context", lambda name: context)
    return spawned


@pytest.mark.parametrize("endpoint", ["reference/cold", "reference/warm/0"])
@pytest.mark.parametrize("timeout", [False, True])
def test_reset_is_persisted_before_first_callback_or_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_watchdog: list[dict],
    endpoint: str,
    timeout: bool,
) -> None:
    output = tmp_path / "result.json"
    record = {"completed_samples": [{"energy": -1.0}]}
    progress = progress_module.EndpointProgress(output, record, 30, True)
    with progress.measure("reference/cold"):
        progress.checkpoint("get_veff_end", seconds=3.0)
        progress.checkpoint("scf_cycle", cycle=4, de=-0.5, norm_gorb=0.1)
    killed: list[int] = []
    monkeypatch.setattr(progress_module.os, "kill", lambda pid, sig: killed.append(pid))
    with progress.measure(endpoint):
        saved = json.loads(output.read_text())
        expected = {
            "endpoint": endpoint,
            "scf_cycles_observed": 0,
            "get_veff_completed": 0,
            "de_sign_changes": 0,
        }
        assert saved["diagnostic_summary"] == expected
        assert fake_watchdog[-1]["diagnostic_summary"] == expected
        assert "diagnostic_progress" not in saved
        assert saved["completed_samples"] == [{"energy": -1.0}]
        if timeout:
            connection = SimpleNamespace(poll=lambda seconds: False, close=lambda: None)
            progress_module._watch(connection, 123456, 30, output, record)
            stopped = json.loads(output.read_text())
            assert stopped["status"] == "stopped"
            assert stopped["stop"]["endpoint"] == endpoint
            assert stopped["diagnostic_summary"] == expected
            assert killed == [123456]


def test_same_label_replay_does_not_pool_cycle_counts_or_timing(
    tmp_path: Path, fake_watchdog: list[dict]
) -> None:
    output = tmp_path / "result.json"
    progress = progress_module.EndpointProgress(output, {}, 30, True)
    with progress.measure("reference/warm"):
        progress.checkpoint(
            "scf_cycle", cycle=5, de=-1.0, norm_gorb=0.01, elapsed_seconds=12.0
        )
    with progress.measure("reference/warm"):
        progress.checkpoint(
            "scf_cycle", cycle=0, de=0.5, norm_gorb=0.5, elapsed_seconds=0.2
        )
        summary = json.loads(output.read_text())["diagnostic_summary"]
        assert summary["scf_cycles_observed"] == 1
        assert summary["first_cycle"] == summary["last_cycle"] == 0
        assert summary["norm_gorb_min"] == 0.5
        assert summary["de_sign_changes"] == 0
        assert summary["cycle_elapsed_span_seconds"] == 0.0
        assert "mean_cycle_interval_seconds" not in summary


def test_multiple_callbacks_within_one_measure_still_accumulate(
    tmp_path: Path, fake_watchdog: list[dict]
) -> None:
    output = tmp_path / "result.json"
    progress = progress_module.EndpointProgress(output, {}, 30, True)
    with progress.measure("reference/cold"):
        progress.checkpoint("get_veff_end", seconds=2.0)
        progress.checkpoint("get_veff_end", seconds=3.0)
        progress.checkpoint("scf_cycle", cycle=0, de=-1.0, elapsed_seconds=2.0)
        progress.checkpoint("scf_cycle", cycle=1, de=0.5, elapsed_seconds=5.0)
    summary = json.loads(output.read_text())["diagnostic_summary"]
    assert summary["scf_cycles_observed"] == 2
    assert summary["get_veff_total_seconds"] == 5.0
    assert summary["get_veff_mean_seconds"] == 2.5
    assert summary["de_sign_changes"] == 1
    assert summary["mean_cycle_interval_seconds"] == 3.0
