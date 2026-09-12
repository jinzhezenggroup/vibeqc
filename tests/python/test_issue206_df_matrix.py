"""Hardware-free checks of failure retention and the Slurm launch contract."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks import issue206_df_matrix as matrix


@pytest.mark.parametrize("failure", ["exit", "launch", "missing_result", "gate"])
def test_matrix_retains_failures_and_finishes_remaining_cases(
    tmp_path, monkeypatch, failure
):
    # subprocess.run is replaced throughout: these tests never execute CUDA.
    monkeypatch.setenv("SLURM_JOB_ID", "protocol-test")
    monkeypatch.setattr(matrix, "_git", lambda *args: "")
    output = tmp_path / "endpoints"
    output.mkdir()
    (output / "96ao-b1.json").write_text('{"previous_attempt": true}')
    manifest = tmp_path / "manifest.json"
    payload = matrix.manifest_payload(
        cases=matrix.MATRIX[:2],
        repeats=1,
        python=sys.executable,
        library=tmp_path / "lib.so",
        output_dir=output,
    )
    calls = []

    def endpoint(command, **kwargs):
        calls.append(command)
        active = json.loads(manifest.read_text())["matrix"][len(calls) - 1]
        assert active["status"] == "running" and active["command"] == command
        assert kwargs["env"].get("CUDA_VISIBLE_DEVICES") == os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        )
        path = Path(command[-1])
        if len(calls) == 1:
            if failure == "launch":
                raise FileNotFoundError("missing interpreter")
            if failure in ("exit", "gate"):
                if failure == "gate":
                    path.write_text('{"gate": {"passed": false}}')
                return subprocess.CompletedProcess(command, 2, "", "endpoint failed")
            return subprocess.CompletedProcess(command, 0, "", "")
        path.write_text('{"converged": true}')
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(matrix.subprocess, "run", endpoint)
    # Exercise only the visibility preflight; no device state is changed and
    # the child launcher above is a pure Python test double.
    monkeypatch.setattr(
        matrix.os, "environ", {**os.environ, "CUDA_VISIBLE_DEVICES": "protocol-test"}
    )
    with pytest.raises(SystemExit, match="DF matrix failed"):
        matrix.run_matrix(
            payload,
            manifest_path=manifest,
            python=sys.executable,
            library=tmp_path / "lib.so",
            output_dir=output,
        )
    rows = json.loads(manifest.read_text())["matrix"]
    assert len(calls) == 2
    assert [row["status"] for row in rows] == ["failed", "passed"]
    if failure == "gate":
        assert (
            json.loads(Path(rows[0]["result"]).read_text())["gate"]["passed"] is False
        )
    else:
        assert rows[0]["result"] is None
    assert Path(rows[0]["log"]).is_file()
    assert Path(rows[1]["result"]).is_file()


def test_sbatch_spool_copy_uses_submission_checkout(tmp_path):
    root = Path(matrix.ROOT)
    spool = tmp_path / "slurm_script"
    spool.write_text((root / "run_issue206_df.slurm").read_text())
    # A harmless interpreter stub reports argv; even --run never reaches Python.
    interpreter = tmp_path / "python-stub"
    interpreter.write_text('#!/bin/bash\nprintf "%s\\n" "$PWD" "$@"\n')
    interpreter.chmod(0o755)
    environment = {
        **os.environ,
        "SLURM_SUBMIT_DIR": str(root),
        "ISSUE206_PYTHON": str(interpreter),
        "ISSUE206_OUTPUT_DIR": str(tmp_path / "results"),
    }
    environment.pop("ISSUE206_ROOT", None)
    completed = subprocess.run(
        ["bash", str(spool)],
        env=environment,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.splitlines()[:2] == [
        str(root),
        "benchmarks/issue206_df_matrix.py",
    ]


def test_published_archive_uses_verified_repository_format():
    from tools.unpack_evidence import unpack

    assert unpack(matrix.ROOT / "benchmarks/results/issue206-df-a") == 9
