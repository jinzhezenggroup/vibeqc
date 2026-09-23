"""Exercise the fixed ceiling and target-branch ratchet independently of GitHub."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tools.check_codegen_test_ownership import (
    OwnershipGuardError,
    check_legacy_codegen_test,
    line_count,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "budget", [True, False, 3.5, float("nan"), float("inf"), "3", None]
)
def test_invalid_budget_never_disables_the_guard(
    tmp_path: Path, budget: object
) -> None:
    path = tmp_path / "legacy.py"
    path.write_text("line\n" * 5, encoding="utf-8")
    with pytest.raises(
        ValueError, match="line budget must be non-negative and an integer"
    ):
        check_legacy_codegen_test(path, budget=budget)


@pytest.mark.parametrize(
    "source,expected",
    [
        ("", 0),
        ("a", 1),
        ("a\n", 1),
        ("a\r\nb\r\n", 2),
        ("a\rb", 2),
        ("# a\f b\n", 1),
        ("# a\v b\n", 1),
        ("# a\u2028 b\n", 1),
    ],
)
def test_only_physical_newlines_count(
    tmp_path: Path, source: str, expected: int
) -> None:
    path = tmp_path / "legacy.py"
    path.write_bytes(source.encode("utf-8"))
    assert line_count(path) == expected


@pytest.mark.parametrize("count", [0, 2, 3])
def test_ratchet_accepts_flat_or_shrinking_file(tmp_path: Path, count: int) -> None:
    path, baseline = tmp_path / "legacy.py", tmp_path / "baseline.py"
    path.write_text("x\n" * count, encoding="utf-8")
    baseline.write_text("x\n" * 3, encoding="utf-8")
    assert check_legacy_codegen_test(path, budget=6, baseline=baseline) == count


def test_ratchet_rejects_regrowth_below_original_ceiling(tmp_path: Path) -> None:
    path, baseline = tmp_path / "legacy.py", tmp_path / "baseline.py"
    path.write_text("x\n" * 4, encoding="utf-8")
    baseline.write_text("x\n" * 3, encoding="utf-8")
    with pytest.raises(OwnershipGuardError, match=r"grew to 4 lines \(budget 3\)"):
        check_legacy_codegen_test(path, budget=6, baseline=baseline)


def test_baseline_cannot_relax_absolute_ceiling(tmp_path: Path) -> None:
    path, baseline = tmp_path / "legacy.py", tmp_path / "baseline.py"
    path.write_text("x\n" * 4, encoding="utf-8")
    baseline.write_text("x\n" * 8, encoding="utf-8")
    with pytest.raises(OwnershipGuardError, match=r"grew to 4 lines \(budget 3\)"):
        check_legacy_codegen_test(path, budget=3, baseline=baseline)


def test_missing_requested_baseline_is_not_silently_ignored(tmp_path: Path) -> None:
    path = tmp_path / "legacy.py"
    path.write_text("x\n", encoding="utf-8")
    with pytest.raises(OwnershipGuardError, match="baseline.*missing"):
        check_legacy_codegen_test(path, budget=6, baseline=tmp_path / "absent.py")
