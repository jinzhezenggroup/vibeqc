"""Fail closed when the legacy codegen test catch-all grows again."""

from __future__ import annotations

import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LEGACY_CODEGEN_TEST = _REPOSITORY_ROOT / "tests/python/test_codegen.py"
LEGACY_LINE_BUDGET = 4518


class OwnershipGuardError(RuntimeError):
    """Raised when the legacy compatibility test file regresses in ownership."""


def line_count(path: Path) -> int:
    """Return physical text lines without normalizing the source first."""

    return len(path.read_text(encoding="utf-8").splitlines())


def check_legacy_codegen_test(
    path: Path = LEGACY_CODEGEN_TEST,
    *,
    budget: int = LEGACY_LINE_BUDGET,
) -> int:
    """Require the legacy catch-all to shrink monotonically from its baseline."""

    if budget < 0:
        raise ValueError("line budget must be non-negative")
    if not path.is_file():
        raise OwnershipGuardError(f"legacy codegen test file is missing: {path}")

    lines = line_count(path)
    if lines > budget:
        raise OwnershipGuardError(
            "legacy codegen catch-all grew to "
            f"{lines} lines (budget {budget}); move new responsibility-owned tests "
            "out of tests/python/test_codegen.py instead of expanding it"
        )
    return lines


def main() -> int:
    """Run the repository guard as a standalone CI-friendly command."""

    try:
        lines = check_legacy_codegen_test()
    except OwnershipGuardError as error:
        print(f"codegen test ownership guard: {error}", file=sys.stderr)
        return 1
    print(
        "codegen test ownership guard: "
        f"{lines}/{LEGACY_LINE_BUDGET} legacy lines retained"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
