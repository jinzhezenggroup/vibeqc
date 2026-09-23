"""The external point deadline must also bound SIGTERM-ignoring children."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(sys.platform != "linux", reason="uses GNU timeout/process groups")
@pytest.mark.parametrize("mode", ["ignore-term", "success"])
def test_runner_deadline_and_outcome(tmp_path: Path, mode: str) -> None:
    bash, timeout = shutil.which("bash"), shutil.which("timeout")
    if bash is None or timeout is None:
        pytest.skip("bash and GNU timeout are required")
    helper = tmp_path / "python-probe"
    helper.write_text(
        f"#!{bash}\n"
        "trap '' TERM\n"
        "printf '%s\\n' \"$$\" > \"$PROBE_PID\"\n"
        "printf 'probe started\\n'\n"
        + ("while :; do sleep 60; done\n" if mode == "ignore-term" else "exit 0\n")
    )
    helper.chmod(0o700)
    output = tmp_path / "records"
    runner = Path(__file__).resolve().parents[2] / "benchmarks/run_readme_benchmarks.sh"
    env = {
        **os.environ,
        "SLURM_JOB_ID": "host-control-flow-fixture",
        "PROBE_PID": str(tmp_path / "probe.pid"),
        "VIBEQC_LIBRARY": "never-loaded",
        "README_BENCHMARK_PYTHON": str(helper),
        "README_BENCHMARK_OUTPUT": str(output),
        "README_BENCHMARK_POINT_TIMEOUT": "1",
    }
    child = subprocess.Popen(
        [bash, str(runner), "dft-paired"],
        env=env,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = child.communicate(timeout=10)
    finally:
        # Also contain the deliberately broken pre-fix implementation in red tests.
        if child.poll() is None:
            pid_file = tmp_path / "probe.pid"
            if pid_file.exists():
                try:
                    os.killpg(os.getpgid(int(pid_file.read_text())), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            os.killpg(child.pid, signal.SIGKILL)
            child.communicate(timeout=5)
    outcomes = sorted(output.rglob("*.outcome"))
    assert child.returncode == (1 if mode == "ignore-term" else 0), stdout + stderr
    assert len(outcomes) == (1 if mode == "ignore-term" else 5)
    for path in outcomes:
        record = json.loads(path.read_text())
        assert record["time_limit_seconds"] == 1
        assert record["exit_code"] == (137 if mode == "ignore-term" else 0)
    if mode == "ignore-term":
        assert "probe started" in (output / "dft/pbe-direct-3.log").read_text()
