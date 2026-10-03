"""Execute generator declarations without running codegen or a CUDA compiler."""

from __future__ import annotations

import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

CMAKE_PROBE = r"""
cmake_minimum_required(VERSION 3.20)
project(GeneratorDeclarations LANGUAGES NONE)
set(CMAKE_CURRENT_SOURCE_DIR "${SOURCE_ROOT}")
set(CMAKE_CUDA_ARCHITECTURES 120)
set(GENERATIVEQC_ENABLE_CUDA ON)
set(GENERATIVEQC_ENABLE_STATIONARY_CPU_FORCE_AOT OFF)
file(WRITE "${CMAKE_BINARY_DIR}/producers.txt" "")
function(generativeqc_register_generated_sources)
  cmake_parse_arguments(P "ADD_TO_TARGET" "NAME;TARGET;GENERATOR;COMMENT"
                        "OUTPUTS;DEPENDS;ARGS;COMPILE_OPTIONS" ${ARGN})
  get_filename_component(generator "${P_GENERATOR}" NAME)
  foreach(output IN LISTS P_OUTPUTS)
    get_filename_component(name "${output}" NAME)
    file(APPEND "${CMAKE_BINARY_DIR}/producers.txt"
         "${name}|${generator}|${P_TARGET}|${P_ADD_TO_TARGET}\n")
  endforeach()
endfunction()
# Observe declarations only; this probe deliberately has no runtime target.
function(target_include_directories)
endfunction()
function(target_sources)
endfunction()
include("${SOURCE_ROOT}/cmake/GenerativeQCGeneratedSources.cmake")
generativeqc_register_host_generated_sources(generativeqc)
generativeqc_register_cuda_generated_sources(generativeqc)
"""

REQUIRED = {
    "generated_direct_cartesian.cuh": "generate_direct_cartesian_contraction.py",
    "generated_direct_contraction.cuh": "generate_direct_cartesian_contraction.py",
    "generated_direct_pair_cache.cuh": "generate_direct_pair_cache.py",
    "generated_libxc_semilocal_registry.hpp": "generate_libxc_semilocal_cpu_registry.py",
    "generated_libxc_semilocal_registry.cpp": "generate_libxc_semilocal_cpu_registry.py",
    **{
        f"generated_libxc_semilocal_{i}.cpp": "generate_libxc_semilocal_cpu_registry.py"
        for i in range(8)
    },
}


def _assert_producers(rows: list[str]) -> None:
    found = defaultdict(list)
    for row in rows:
        name, generator, target, attached = row.split("|")
        found[name].append((generator, target, attached))
    for name, generator in REQUIRED.items():
        assert found[name] == [(generator, "generativeqc", "TRUE")], (name, found[name])


def test_libxc_and_direct_headers_keep_one_attached_producer(tmp_path: Path) -> None:
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake unavailable")
    (tmp_path / "CMakeLists.txt").write_text(CMAKE_PROBE, encoding="utf-8")
    build = tmp_path / "build"
    result = subprocess.run(
        [cmake, "-S", str(tmp_path), "-B", str(build), f"-DSOURCE_ROOT={ROOT}"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    _assert_producers((build / "producers.txt").read_text().splitlines())


@pytest.mark.parametrize("duplicate", [False, True])
def test_producer_guard_rejects_missing_or_duplicate_registration(
    duplicate: bool,
) -> None:
    rows = [
        f"{name}|{generator}|generativeqc|TRUE" for name, generator in REQUIRED.items()
    ]
    missing = rows.pop(0)
    if duplicate:
        rows.extend([missing, missing])
    with pytest.raises(AssertionError, match="generated_direct_cartesian"):
        _assert_producers(rows)
