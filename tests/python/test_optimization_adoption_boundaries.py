"""Adoption evidence must stay local and invalid budgets must fail predictably."""

from pathlib import Path
from typing import Any

import pytest

from tools import check_compiler_optimization_ledger as ledger


def _contract() -> dict[str, Any]:
    statuses = list(ledger._ALLOWED_ADOPTION_STATUSES)
    return {
        "adoption_contract": {
            "methods": ["HF", "DFT", "MP2", "CCSD(T)"],
            "backends": ["cpu", "cuda"],
            "statuses": statuses,
            "status_semantics": dict.fromkeys(statuses, "test-only declaration"),
            "benchmark_evidence_root": "benchmarks/results",
        }
    }


def _group(budget: Any, evidence: str) -> dict[str, Any]:
    rows = [
        {
            "methods": [method],
            "backends": [backend],
            "status": "unverified",
            "note": "synthetic validation fixture, not scientific evidence",
        }
        for method in ("HF", "DFT", "MP2", "CCSD(T)")
        for backend in ("cpu", "cuda")
    ]
    rows[0].update(
        status="benchmark-qualified",
        refs=["#682"],
        evidence=[evidence],
        max_regression_percent=budget,
    )
    return {"adoption": rows}


def _link(path: Path, target: Path, *, directory: bool = False) -> None:
    try:
        path.symlink_to(target, target_is_directory=directory)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlinks unavailable: {error}")


@pytest.mark.parametrize("component", ["benchmarks", "results"])
def test_evidence_root_cannot_escape_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, component: str
) -> None:
    root, outside = tmp_path / "repo", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    if component == "benchmarks":
        (outside / "results").mkdir()
        _link(root / "benchmarks", outside, directory=True)
    else:
        (root / "benchmarks").mkdir()
        _link(root / "benchmarks/results", outside, directory=True)
    monkeypatch.setattr(ledger, "ROOT", root)
    with pytest.raises(ledger.LedgerError, match="evidence root"):
        ledger._validate_contract(_contract())


@pytest.mark.parametrize("where", ["root", "receipt"])
def test_symlink_cycle_is_a_ledger_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, where: str
) -> None:
    monkeypatch.setattr(ledger, "ROOT", tmp_path)
    (tmp_path / "benchmarks").mkdir()
    if where == "root":
        _link(tmp_path / "benchmarks/results", Path("results"), directory=True)
        with pytest.raises(ledger.LedgerError, match="evidence root"):
            ledger._validate_contract(_contract())
    else:
        evidence = tmp_path / "benchmarks/results"
        evidence.mkdir()
        _link(evidence / "loop", Path("loop"))
        with pytest.raises(ledger.LedgerError, match="evidence"):
            ledger._validate_evidence_paths(
                ["benchmarks/results/loop"], "evidence", evidence
            )


@pytest.mark.parametrize(
    "budget",
    [
        10**1000,
        -(10**1000),
        float("nan"),
        float("inf"),
        -float("inf"),
        True,
        0,
        -1,
        101,
    ],
    ids=[
        "huge",
        "negative-huge",
        "nan",
        "inf",
        "negative-inf",
        "bool",
        "zero",
        "negative",
        "over-cap",
    ],
)
def test_invalid_budget_raises_ledger_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget: Any
) -> None:
    monkeypatch.setattr(ledger, "ROOT", tmp_path)
    evidence = tmp_path / "benchmarks/results"
    evidence.mkdir(parents=True)
    (evidence / "receipt.txt").write_text("fixture only\n")
    with pytest.raises(ledger.LedgerError, match="max_regression_percent"):
        ledger._validate_adoption(
            _group(budget, "benchmarks/results/receipt.txt"), "group", evidence
        )


@pytest.mark.parametrize("budget", [0.5, 1, 2.0, 100])
def test_valid_budget_and_internal_symlink_preserve_cell_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, budget: float
) -> None:
    monkeypatch.setattr(ledger, "ROOT", tmp_path)
    evidence = tmp_path / "benchmarks/results"
    evidence.mkdir(parents=True)
    (evidence / "receipt.txt").write_text("fixture only\n")
    _link(evidence / "alias.txt", Path("receipt.txt"))
    resolved = ledger._validate_contract(_contract())
    assert resolved == evidence
    assert ledger._validate_adoption(
        _group(budget, "benchmarks/results/alias.txt"), "group", resolved
    ) == (8, 1)


def test_leaf_symlink_cannot_escape_evidence_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ledger, "ROOT", tmp_path)
    evidence = tmp_path / "benchmarks/results"
    evidence.mkdir(parents=True)
    outside = tmp_path / "unretained.txt"
    outside.write_text("not retained evidence\n")
    _link(evidence / "alias.txt", outside)
    with pytest.raises(ledger.LedgerError, match="benchmarks/results"):
        ledger._validate_evidence_paths(
            ["benchmarks/results/alias.txt"], "evidence", evidence
        )
