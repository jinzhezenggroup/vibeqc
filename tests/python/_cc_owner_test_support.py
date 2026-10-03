"""Generated dependencies and cached compilation for standalone CC owner tests."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def write_df_cpu_headers(directory: Path) -> None:
    """Supply every solver dependency without importing tooling into pytest."""
    script = """
import sys
from pathlib import Path
root, target = map(Path, sys.argv[1:])
sys.path[:0] = [str(root), str(root / 'python')]
from tools import generate_df_ccsd_native as actions, generate_df_ccsd_core as core
from tools import generate_df_ccsd_hoisted as hoisted, generate_rccsd_native as conventional
if not (target / 'generated_rccsd_cpu.hpp').exists():
    (target / 'generated_rccsd_cpu.hpp').write_text(conventional.cpu_header())
for name, module in [('generated_df_ccsd', actions), ('generated_df_ccsd_core', core),
                     ('generated_df_ccsd_hoisted', hoisted)]:
    (target / (name + '_cpu.hpp')).write_text(module.cpu_header())
"""
    subprocess.run(
        [sys.executable, "-S", "-c", script, str(ROOT), str(directory)],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )


def compile_owner(
    compiler: str, directory: Path, sources: list[Path], output: Path
) -> None:
    """Cache individual objects; a multi-source compile cannot use ccache."""
    cache = shutil.which("ccache")
    if cache is None:
        pytest.skip("ccache is required for native owner probes")
    subprocess.run([cache, "--version"], check=True, capture_output=True)
    objects = []
    for i, source in enumerate(sources):
        obj = directory / f"owner-{i}.o"
        subprocess.run(
            [
                cache,
                compiler,
                "-std=c++20",
                "-O0",
                "-DGENERATIVEQC_HAS_CUDA=0",
                "-I" + str(ROOT / "src"),
                "-I" + str(ROOT / "include"),
                "-I" + str(directory),
                "-c",
                str(source),
                "-o",
                str(obj),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
        )
        objects.append(str(obj))
    subprocess.run(
        [compiler, *objects, "-o", str(output)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
