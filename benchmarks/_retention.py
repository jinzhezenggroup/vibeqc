"""Stdlib-only guard for raw benchmark output destinations."""

from __future__ import annotations

from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def raw_output_path(path: str | Path, *, repository_root: Path | None = None) -> Path:
    """Reject execution output inside the reviewed benchmark result tree.

    This helper intentionally imports no benchmark/runtime/GPU dependencies so
    argparse and dry-run paths can apply the storage boundary before scientific
    work starts.  ``repository_root`` exists for the shared writer's tests; live
    runners use this module's source-derived checkout root.
    """
    destination = Path(path)
    root = _REPOSITORY_ROOT if repository_root is None else Path(repository_root)
    retained = root / "benchmarks" / "results"
    if destination.absolute().is_relative_to(retained.absolute()) or (
        destination.resolve().is_relative_to(retained.resolve())
    ):
        raise ValueError(
            "raw benchmark output cannot target benchmarks/results/; use "
            ".artifacts/benchmarks/ and tools/evidence.py publish for reviewed evidence"
        )
    return destination
