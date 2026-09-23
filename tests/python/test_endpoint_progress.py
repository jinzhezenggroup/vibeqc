"""The external watchdog must retain evidence even when a solve never returns."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks._endpoint_progress import EndpointProgress


def test_deadline_kills_only_its_benchmark_and_keeps_checkpoint(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    script = tmp_path / "hang.py"
    script.write_text(
        "from pathlib import Path\n"
        "import sys\n"
        "import time\n"
        # Pytest's import path is not inherited by a separate interpreter.
        # Bootstrap the checkout explicitly, including when spawned from /tmp.
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r})\n"
        "from benchmarks._endpoint_progress import EndpointProgress\n"
        "if __name__ == '__main__':\n"
        f"    progress = EndpointProgress(Path({str(output)!r}), "
        "{'prior_energy': -75.0}, 0.25, False)\n"
        "    with progress.measure('reference/cold'):\n"
        "        progress.checkpoint('scf_cycle', cycle=2, e_tot=-75.1)\n"
        "        time.sleep(30)\n"
    )
    result = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, timeout=10, check=False
    )
    assert result.returncode == -signal.SIGKILL
    record = json.loads(output.read_text())
    assert record["status"] == "stopped"
    assert record["prior_energy"] == -75.0
    assert record["stop"]["endpoint"] == "reference/cold"
    assert record["diagnostic_progress"]["event"] == "scf_cycle"
    assert record["diagnostic_progress"]["cycle"] == 2
    events = [
        json.loads(line)
        for line in output.with_suffix(".progress.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == ["begin", "scf_cycle", "timeout"]


def test_completed_and_failed_endpoints_disarm_watchdog(tmp_path: Path) -> None:
    progress = EndpointProgress(tmp_path / "result.json", {}, 2, False)
    with progress.measure("native/cold"):
        pass
    with (
        pytest.raises(ValueError, match="failed solve"),
        progress.measure("native/warm"),
    ):
        raise ValueError("failed solve")
    events = [json.loads(line) for line in progress.journal.read_text().splitlines()]
    assert [event["event"] for event in events] == ["begin", "end", "begin", "failed"]


def test_reference_trace_checkpoints_latest_stage_and_cycle(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    seen: list[dict] = []

    class Engine:
        def __init__(self) -> None:
            self.callback = seen.append

        def get_veff(self) -> float:
            return 1.0

        def kernel(self) -> float:
            self.get_veff()
            assert self.callback is not None
            self.callback(
                {
                    "cycle": 3,
                    "e_tot": -75.25,
                    "de": -1.0e-4,
                    "norm_gorb": 2.0e-3,
                    "norm_ddm": 4.0e-4,
                }
            )
            return -75.25

    synchronize_calls: list[None] = []
    cupy = SimpleNamespace(
        cuda=SimpleNamespace(
            Stream=SimpleNamespace(
                null=SimpleNamespace(synchronize=lambda: synchronize_calls.append(None))
            )
        )
    )
    engine = Engine()
    progress = EndpointProgress(output, {"status": "running"}, 2, True)
    progress.endpoint = "reference/cold"
    progress.instrument_reference(engine, cupy)

    assert engine.kernel() == -75.25
    record = json.loads(output.read_text())
    assert record["diagnostic_progress"]["event"] == "scf_cycle"
    assert record["diagnostic_progress"]["cycle"] == 3
    assert seen == [
        {
            "cycle": 3,
            "e_tot": -75.25,
            "de": -1.0e-4,
            "norm_gorb": 2.0e-3,
            "norm_ddm": 4.0e-4,
        }
    ]
    assert len(synchronize_calls) == 2
    events = [json.loads(line) for line in progress.journal.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "get_veff_begin",
        "get_veff_end",
        "scf_cycle",
    ]
