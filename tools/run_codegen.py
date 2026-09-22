"""Run a source generator and emit its repository-local Python dependencies."""

from __future__ import annotations

import argparse
import importlib.util
import runpy
import sys
from pathlib import Path


def _source_path(module: object) -> Path | None:
    """Return a module's source path when it was loaded from a file."""

    spec = getattr(module, "__spec__", None)
    origin = getattr(spec, "origin", None) or getattr(module, "__file__", None)
    if not origin or origin in {"built-in", "frozen"}:
        return None
    path = Path(origin)
    if path.suffix in {".pyc", ".pyo"}:
        try:
            path = Path(importlib.util.source_from_cache(str(path)))
        except ValueError:
            return None
    return path


def _local_python_dependencies(source_root: Path) -> list[Path]:
    """Collect Python modules actually loaded from the source checkout."""

    root = source_root.resolve()
    dependencies: set[Path] = set()
    for module in tuple(sys.modules.values()):
        source = _source_path(module)
        if source is None:
            continue
        try:
            resolved = source.resolve()
            resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if resolved.suffix == ".py" and resolved.is_file():
            dependencies.add(resolved)
    return sorted(dependencies)


def _escape_depfile_path(path: Path) -> str:
    """Escape one Make/Ninja depfile token."""

    text_value = path.as_posix()
    return (
        text_value.replace("$", "$$")
        .replace("#", r"\#")
        .replace(" ", r"\ ")
        .replace(":", r"\:")
    )


def _write_depfile(path: Path, targets: list[Path], dependencies: list[Path]) -> None:
    """Atomically publish dependencies after successful generation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    target_text = " ".join(_escape_depfile_path(item) for item in targets)
    dependency_text = " ".join(_escape_depfile_path(item) for item in dependencies)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(f"{target_text}: {dependency_text}\n", encoding="utf-8")
    temporary.replace(path)


def _run_generator(generator: Path, arguments: list[str]) -> None:
    """Execute a generator with normal script argv/import-path semantics."""

    old_argv = sys.argv
    old_path = sys.path.copy()
    sys.argv = [str(generator), *arguments]
    sys.path[0] = str(generator.parent.resolve())
    try:
        try:
            runpy.run_path(str(generator), run_name="__main__")
        except SystemExit as error:
            if error.code not in (None, 0):
                raise
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depfile", type=Path, required=True)
    parser.add_argument("--target", action="append", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("generator", type=Path)
    parser.add_argument("generator_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    generator = args.generator.resolve()
    _run_generator(generator, args.generator_args)
    dependencies = _local_python_dependencies(args.source_root)
    _write_depfile(args.depfile, args.target, dependencies)


if __name__ == "__main__":
    main()
