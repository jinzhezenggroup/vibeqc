"""Failure checkpoints must retain completed endpoint evidence for review."""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from tools import benchmark_ccsdt_cpu_bundles as benchmark

if TYPE_CHECKING:
    from pathlib import Path


def _launch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    output = tmp_path / "report.json"
    monkeypatch.setattr(benchmark.platform, "system", lambda: "Linux")
    monkeypatch.setattr(benchmark.subprocess, "check_output", lambda *a, **kw: "head\n")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_ccsdt_cpu_bundles.py",
            "--cache-root",
            str(tmp_path / "cache"),
            "--output",
            str(output),
        ],
    )
    return output


@pytest.mark.parametrize("failure", ("child", "json", "interrupt"))
def test_later_child_failure_retains_first_completed_measurement(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    output = _launch(monkeypatch, tmp_path)
    calls = 0

    def run(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            return SimpleNamespace(
                stdout='{"wall_seconds": 1, "artifact_count": 1}', stderr=""
            )
        assert (
            json.loads(output.read_text())["measurements"]["separate"]["cold"][
                "wall_seconds"
            ]
            == 1
        )
        if failure == "child":
            raise subprocess.CalledProcessError(
                1, "child", output="partial", stderr="compiler failed"
            )
        if failure == "interrupt":
            raise KeyboardInterrupt
        return SimpleNamespace(stdout="NaN", stderr="")

    monkeypatch.setattr(benchmark.subprocess, "run", run)
    expected = KeyboardInterrupt if failure == "interrupt" else Exception
    with pytest.raises(expected):
        benchmark.main()

    report = json.loads(output.read_text())
    assert report["qualified"] is False
    assert set(report["measurements"]["separate"]) == {"cold"}
    assert report["failure"]["mode"] == "separate"
    assert report["failure"]["phase"] == "warm"
    if failure == "child":
        assert "compiler failed" in report["failure"]["stderr_tail"]
    if failure == "json":
        assert "NaN" in report["failure"]["stdout_tail"]
    if failure == "interrupt":
        assert report["failure"]["exception"] == "KeyboardInterrupt"


def test_failed_final_gate_preserves_all_four_measurements(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = _launch(monkeypatch, tmp_path)
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            stdout='{"wall_seconds": 1, "artifact_count": 1}', stderr=""
        ),
    )

    def reject(*args: object) -> None:
        raise ValueError("gradient gate failed")

    monkeypatch.setattr(benchmark, "_compare", reject)
    with pytest.raises(ValueError, match="gradient gate failed"):
        benchmark.main()

    report = json.loads(output.read_text())
    assert report["qualified"] is False
    assert all(
        len(report["measurements"][mode]) == 2 for mode in ("separate", "bundled")
    )
    assert report["failure"]["phase"] == "qualification"
    assert "gradient gate failed" in report["failure"]["message"]
