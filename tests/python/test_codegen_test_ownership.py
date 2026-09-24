from pathlib import Path

import pytest

from tools.check_codegen_test_ownership import (
    LEGACY_CODEGEN_TEST,
    LEGACY_LINE_BUDGET,
    OwnershipGuardError,
    check_legacy_codegen_test,
)


def _write_lines(path: Path, count: int) -> None:
    path.write_text("line\n" * count, encoding="utf-8")


def test_repository_legacy_codegen_file_stays_within_budget() -> None:
    assert check_legacy_codegen_test() <= LEGACY_LINE_BUDGET


def test_legacy_codegen_guard_accepts_exact_budget(tmp_path: Path) -> None:
    legacy = tmp_path / "test_codegen.py"
    _write_lines(legacy, 3)

    assert check_legacy_codegen_test(legacy, budget=3) == 3


def test_legacy_codegen_guard_rejects_regrowth(tmp_path: Path) -> None:
    legacy = tmp_path / "test_codegen.py"
    _write_lines(legacy, 4)

    with pytest.raises(OwnershipGuardError, match=r"grew to 4 lines \(budget 3\)"):
        check_legacy_codegen_test(legacy, budget=3)


def test_legacy_codegen_guard_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "test_codegen.py"

    with pytest.raises(
        OwnershipGuardError, match="legacy codegen test file is missing"
    ):
        check_legacy_codegen_test(missing, budget=3)


def test_legacy_codegen_guard_rejects_negative_budget(tmp_path: Path) -> None:
    legacy = tmp_path / "test_codegen.py"
    _write_lines(legacy, 1)

    with pytest.raises(ValueError, match="line budget must be non-negative"):
        check_legacy_codegen_test(legacy, budget=-1)


def test_guard_targets_the_legacy_compatibility_surface() -> None:
    assert LEGACY_CODEGEN_TEST.as_posix().endswith("tests/python/test_codegen.py")
