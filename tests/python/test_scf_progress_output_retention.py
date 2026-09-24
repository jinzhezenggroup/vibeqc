"""The diagnostic analyzer must not overwrite its input or reviewed evidence."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest

from benchmarks import _retention, scf_progress_evidence

if TYPE_CHECKING:
    from pathlib import Path


def _journal(path: Path) -> bytes:
    content = b'{"event":"begin","endpoint":"cold","elapsed_seconds":0}\n'
    path.write_bytes(content)
    return content


@pytest.mark.parametrize("spelling", ("absolute", "relative", "parent", "symlink"))
def test_cli_rejects_reviewed_destination_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spelling: str
) -> None:
    retained = tmp_path / "benchmarks" / "results"
    retained.mkdir(parents=True)
    destination = retained / "reviewed.json"
    original = b"reviewed evidence\n"
    destination.write_bytes(original)
    monkeypatch.setattr(_retention, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    if spelling == "relative":
        destination = destination.relative_to(tmp_path)
    elif spelling == "parent":
        destination = retained.parent / "other" / ".." / "results" / "reviewed.json"
    elif spelling == "symlink":
        alias = tmp_path / "alias"
        try:
            alias.symlink_to(retained, target_is_directory=True)
        except OSError as error:
            pytest.skip(f"symlink creation unavailable: {error}")
        destination = alias / "reviewed.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["scf-progress", "missing-input.jsonl", "--output", str(destination)],
    )
    with pytest.raises(SystemExit) as error:
        scf_progress_evidence.main()
    assert error.value.code == 2
    assert (retained / "reviewed.json").read_bytes() == original


def test_cli_does_not_replace_its_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = tmp_path / "input.jsonl"
    original = _journal(journal)
    monkeypatch.setattr(
        sys, "argv", ["scf-progress", str(journal), "--output", str(journal)]
    )
    with pytest.raises(SystemExit) as error:
        scf_progress_evidence.main()
    assert error.value.code == 2
    assert journal.read_bytes() == original


def test_cli_writes_only_diagnostic_report_to_allowed_nested_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    journal = tmp_path / "input.jsonl"
    original = _journal(journal)
    destination = tmp_path / ".artifacts" / "diagnostic.json"
    monkeypatch.setattr(
        sys, "argv", ["scf-progress", str(journal), "--output", str(destination)]
    )
    scf_progress_evidence.main()
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["invocations"][0]["benchmark_qualification"] == "diagnostic-only"
    assert journal.read_bytes() == original


def test_cli_retains_stdout_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    journal = tmp_path / "input.jsonl"
    original = _journal(journal)
    monkeypatch.setattr(sys, "argv", ["scf-progress", str(journal)])
    scf_progress_evidence.main()
    report = json.loads(capsys.readouterr().out)
    assert report["invocations"][0]["terminal_event"] == "incomplete"
    assert journal.read_bytes() == original
