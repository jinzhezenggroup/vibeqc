"""The CPU sweep must reject retained outputs before starting a measurement."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks import _retention
from benchmarks import cpu_linalg_sweep as sweep


@pytest.mark.parametrize("kind", ["direct", "alias", "traversal"])
def test_retained_output_is_rejected_before_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    monkeypatch.setattr(_retention, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    retained = tmp_path / "benchmarks/results"
    retained.mkdir(parents=True)
    old = retained / "old.json"
    old.write_text("reviewed evidence", encoding="utf-8")
    destination = old
    if kind == "alias":
        alias = tmp_path / "alias"
        alias.symlink_to(retained, target_is_directory=True)
        destination = alias / old.name
    elif kind == "traversal":
        destination = tmp_path / "scratch/../benchmarks/results/old.json"
    called = []
    monkeypatch.setattr(
        sweep, "run_sweep", lambda *args, **kwargs: called.append(1) or {}
    )
    monkeypatch.setattr(
        sys, "argv", ["sweep", "--probe", "unused", "--output", str(destination)]
    )
    with pytest.raises(SystemExit) as error:
        sweep.main()
    assert error.value.code == 2
    assert called == []
    assert old.read_text(encoding="utf-8") == "reviewed evidence"
    assert not (tmp_path / "scratch").exists()


@pytest.mark.parametrize("output", [False, True])
def test_valid_scratch_or_stdout_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    output: bool,
) -> None:
    monkeypatch.setattr(_retention, "_REPOSITORY_ROOT", tmp_path)
    payload = {"schema": sweep.SWEEP_SCHEMA, "records": [{"m": 16}]}
    monkeypatch.setattr(sweep, "run_sweep", lambda *args, **kwargs: payload)
    destination = tmp_path / ".artifacts/sweep.json"
    arguments = ["sweep", "--probe", "unused"]
    if output:
        arguments += ["--output", str(destination)]
    monkeypatch.setattr(sys, "argv", arguments)
    assert sweep.main() == 0
    rendered = (
        destination.read_text(encoding="utf-8") if output else capsys.readouterr().out
    )
    assert json.loads(rendered) == payload


def test_direct_script_applies_guard_before_probe_execution(tmp_path: Path) -> None:
    directory = tmp_path / "benchmarks"
    directory.mkdir()
    for filename in ("cpu_linalg_sweep.py", "_retention.py"):
        shutil.copyfile(Path(sweep.__file__).with_name(filename), directory / filename)
    destination = directory / "results/old.json"
    destination.parent.mkdir()
    destination.write_text("reviewed evidence", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(directory / "cpu_linalg_sweep.py"),
            "--probe",
            "does-not-exist",
            "--output",
            str(destination),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 2
    assert "raw_output_path" in completed.stderr
    assert destination.read_text(encoding="utf-8") == "reviewed evidence"
