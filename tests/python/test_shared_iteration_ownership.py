"""Regression gates for shared host iteration ownership."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tools.check_electronic_structure_boundaries import (
    ROOT,
    SUFFIXES,
    _cpp_tokens,
    _definition_locations,
)

if TYPE_CHECKING:
    from pathlib import Path


def _definition_sites(
    root: Path,
    *,
    area: str,
    kind: str,
    name: str,
) -> list[str]:
    source = root / "src"
    owner = source / area
    sites: list[str] = []
    if not owner.is_dir():
        return sites
    for path in sorted(owner.rglob("*")):
        if not path.is_file() or path.suffix not in SUFFIXES:
            continue
        tokens = _cpp_tokens(path.read_text(encoding="utf-8"))
        for line in _definition_locations(tokens, kind, name):
            sites.append(f"{path.relative_to(source).as_posix()}:{line}")
    return sites


def test_scf_cannot_redefine_shared_bounded_iteration_owner() -> None:
    """SCF must consume src/solver/iteration_control.hpp instead of forking it."""
    assert (
        _definition_sites(
            ROOT,
            area="scf",
            kind="function",
            name="run_bounded_iterations",
        )
        == []
    )


def test_scf_iteration_ownership_guard_ignores_nondefinitions(tmp_path: Path) -> None:
    scf = tmp_path / "src/scf"
    scf.mkdir(parents=True)
    (scf / "consumer.cpp").write_text(
        "void run_bounded_iterations();\n"
        "void consume() { run_bounded_iterations(); }\n"
        "// void run_bounded_iterations() {}\n",
        encoding="utf-8",
    )
    assert (
        _definition_sites(
            tmp_path,
            area="scf",
            kind="function",
            name="run_bounded_iterations",
        )
        == []
    )


def test_scf_iteration_ownership_guard_detects_method_local_fork(
    tmp_path: Path,
) -> None:
    scf = tmp_path / "src/scf"
    scf.mkdir(parents=True)
    (scf / "iteration.cpp").write_text(
        "void run_bounded_iterations() {}\n",
        encoding="utf-8",
    )
    assert _definition_sites(
        tmp_path,
        area="scf",
        kind="function",
        name="run_bounded_iterations",
    ) == ["scf/iteration.cpp:1"]
