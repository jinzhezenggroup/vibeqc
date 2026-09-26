"""The fixed-density probe rejects retained destinations before starting work."""

import sys
from pathlib import Path

import pytest

from benchmarks import _retention, issue434_fixed_density


@pytest.mark.parametrize("alias", (False, True))
def test_fixed_density_cli_rejects_retained_output_before_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias: bool
) -> None:
    monkeypatch.setattr(_retention, "_REPOSITORY_ROOT", tmp_path)
    retained = tmp_path / "benchmarks/results"
    retained.mkdir(parents=True)
    output = retained / "new-run"
    if alias:
        link = tmp_path / "alias"
        link.symlink_to(retained, target_is_directory=True)
        output = link / "new-run"
    monkeypatch.setattr(
        sys,
        "argv",
        ["probe", "--probe", "unused", "--fixture", "unused", "--output", str(output)],
    )
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(SystemExit) as error:
        issue434_fixed_density.main()
    assert error.value.code == 2
    assert not output.exists()
    assert list(retained.iterdir()) == []


def test_fixed_density_cli_allows_scratch_until_allocation_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / ".artifacts/benchmarks/run"
    monkeypatch.setattr(_retention, "_REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["probe", "--probe", "unused", "--fixture", "unused", "--output", str(output)],
    )
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(SystemExit, match="Slurm GPU allocation"):
        issue434_fixed_density.main()
    assert not output.exists()
