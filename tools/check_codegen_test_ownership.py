"""Bound the legacy codegen catch-all and ratchet PRs against their tested base."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LEGACY_CODEGEN_TEST = _REPOSITORY_ROOT / "tests/python/test_codegen.py"
LEGACY_LINE_BUDGET = 4518


class OwnershipGuardError(RuntimeError):
    """Raised when the legacy compatibility test file regresses in ownership."""


def line_count(path: Path) -> int:
    """Count physical lines; form feeds and Unicode separators are not newlines."""

    with path.open(encoding="utf-8") as source:
        return sum(1 for _ in source)


def check_legacy_codegen_test(
    path: Path = LEGACY_CODEGEN_TEST,
    *,
    budget: int = LEGACY_LINE_BUDGET,
    baseline: Path | None = None,
) -> int:
    """Enforce the ceiling and, when supplied, the target-branch line count."""

    if type(budget) is not int or budget < 0:
        raise ValueError("line budget must be non-negative and an integer")
    if not path.is_file():
        raise OwnershipGuardError(f"legacy codegen test file is missing: {path}")
    if baseline is not None and not baseline.is_file():
        raise OwnershipGuardError(f"legacy codegen baseline file is missing: {baseline}")

    try:
        if baseline is not None:
            budget = min(budget, line_count(baseline))
        lines = line_count(path)
    except (OSError, UnicodeError) as error:
        raise OwnershipGuardError(f"cannot read legacy codegen source: {error}") from error
    if lines > budget:
        raise OwnershipGuardError(
            "legacy codegen catch-all grew to "
            f"{lines} lines (budget {budget}); move new responsibility-owned tests "
            "out of tests/python/test_codegen.py instead of expanding it"
        )
    return lines


def main() -> int:
    """Run the fixed ceiling locally or the target-branch ratchet in PR CI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, help="legacy source from the tested PR base")
    args = parser.parse_args()
    try:
        lines = check_legacy_codegen_test(baseline=args.baseline)
    except OwnershipGuardError as error:
        print(f"codegen test ownership guard: {error}", file=sys.stderr)
        return 1
    suffix = "; tested-base ratchet passed" if args.baseline is not None else ""
    print(
        "codegen test ownership guard: "
        f"{lines}/{LEGACY_LINE_BUDGET} legacy lines retained{suffix}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
