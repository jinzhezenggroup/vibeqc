from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "src/xtb/gfn2_runtime"
FORBIDDEN_IDENTITY = re.compile(r"\b(?:xtbloom::|xtbloom_|XTBLOOM_)")
CODE_SUFFIXES = {".c", ".h", ".hpp", ".cpp", ".cu", ".cuh"}


def _runtime_code_files() -> list[Path]:
    return [
        path
        for path in RUNTIME.rglob("*")
        if path.is_file()
        and path.suffix in CODE_SUFFIXES
        and "LICENSES" not in path.parts
    ]


def test_runtime_code_has_vibeqc_owned_identity() -> None:
    paths = _runtime_code_files() + [
        ROOT / "src/methods/gfn2_runtime_bridge.cpp",
        ROOT / "cmake/VibeQCGfn2Runtime.cmake",
    ]
    offenders = [
        str(path.relative_to(ROOT))
        for path in paths
        if FORBIDDEN_IDENTITY.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_runtime_entry_headers_are_vibeqc_owned() -> None:
    include_root = RUNTIME / "include"
    assert (include_root / "vibeqc/xtb_runtime.h").is_file()
    assert (include_root / "vibeqc/xtb_runtime_version.h").is_file()
    assert not (include_root / "xtbloom/xtbloom.h").exists()
    assert not (include_root / "xtbloom/version.h").exists()
