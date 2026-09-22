"""The external watchdog must retain evidence even when a solve never returns."""

from __future__ import annotations

import json
import signal
import subprocess
import sys
from pathlib import Path

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
    events = [
        json.loads(line)
        for line in output.with_suffix(".progress.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == ["begin", "timeout"]


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
