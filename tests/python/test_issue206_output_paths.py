"""Issue-206 runners reject raw writes into reviewed evidence before work."""

import sys
from pathlib import Path

import pytest

from benchmarks import _retention, issue206_rebuild, issue206_resident_sentinel


@pytest.mark.parametrize("runner", ["rebuild", "sentinel"])
@pytest.mark.parametrize("alias", [False, True])
def test_issue206_output_guard_precedes_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    runner: str,
    alias: bool,
) -> None:
    retained = tmp_path / "benchmarks/results"
    retained.mkdir(parents=True)
    evidence = retained / "old.json"
    evidence.write_text("reviewed evidence")
    destination = retained
    if alias:
        destination = tmp_path / "evidence-alias"
        destination.symlink_to(retained, target_is_directory=True)
    target = destination / "new/result.json"
    monkeypatch.setattr(_retention, "_REPOSITORY_ROOT", tmp_path)
    if runner == "rebuild":
        module = issue206_rebuild
        arguments = [
            "--case",
            "water-tetramer-def2-svp-spherical",
            "--library",
            str(tmp_path / "unbuilt.so"),
        ]
    else:
        module = issue206_resident_sentinel
        arguments = [str(tmp_path / "unread-trace.jsonl")]
    monkeypatch.setattr(sys, "argv", [runner, *arguments, "--output", str(target)])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert "invalid raw_output_path value" in capsys.readouterr().err
    assert evidence.read_text() == "reviewed evidence"
    assert not (retained / "new").exists()
