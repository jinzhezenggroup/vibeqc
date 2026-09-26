"""Build concurrency policy keeps generated AOT work independently tunable."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_generated_aot_shares_native_pool_until_explicitly_split() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    cuda = (ROOT / "cmake" / "VibeQCCuda.cmake").read_text(encoding="utf-8")

    assert 'set(VIBEQC_AOT_COMPILE_JOBS "" CACHE STRING' in cmake
    assert "set(_vibeqc_aot_compile_pool vibeqc_cuda_compile)" in cmake
    assert "vibeqc_cuda_compile=${VIBEQC_CUDA_COMPILE_JOBS}" in cmake
    assert "vibeqc_aot_compile=${VIBEQC_AOT_COMPILE_JOBS}" in cmake
    assert "set(_vibeqc_aot_compile_pool vibeqc_aot_compile)" in cmake

    native = cuda.split("if(VIBEQC_ENABLE_AOT_SHELLS)", 1)[0]
    aot = cuda.split("if(VIBEQC_ENABLE_AOT_SHELLS)", 1)[1]
    assert "JOB_POOL_COMPILE vibeqc_cuda_compile" in native
    assert "JOB_POOL_COMPILE ${_vibeqc_aot_compile_pool}" in aot


def test_fast_cuda_preset_uses_wider_aot_pool() -> None:
    presets = json.loads((ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
    fast = next(
        preset
        for preset in presets["configurePresets"]
        if preset["name"] == "cuda-dev-fast"
    )
    variables = fast["cacheVariables"]

    assert variables["VIBEQC_CUDA_COMPILE_JOBS"] == "2"
    assert variables["VIBEQC_AOT_COMPILE_JOBS"] == "4"


def test_host_pch_is_opt_in_and_cxx_only() -> None:
    presets = json.loads((ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
    fast = next(
        preset
        for preset in presets["configurePresets"]
        if preset["name"] == "cuda-dev-fast"
    )
    assert "VIBEQC_ENABLE_CXX_PCH" not in fast["cacheVariables"]

    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "option(VIBEQC_ENABLE_CXX_PCH" in cmake
    assert "target_precompile_headers(vibeqc PRIVATE" in cmake
    assert "$<$<COMPILE_LANGUAGE:CXX>:" in cmake

    pch = (ROOT / "src" / "pch.hpp").read_text(encoding="utf-8")
    assert '#include "' not in pch


def test_fast_cuda_preset_uses_bounded_aot_split_compile() -> None:
    presets = json.loads((ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
    fast = next(
        preset
        for preset in presets["configurePresets"]
        if preset["name"] == "cuda-dev-fast"
    )
    assert fast["cacheVariables"]["VIBEQC_AOT_SPLIT_COMPILE_THREADS"] == "2"

    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert 'set(VIBEQC_AOT_SPLIT_COMPILE_THREADS "1" CACHE STRING' in cmake

    cuda = (ROOT / "cmake" / "VibeQCCuda.cmake").read_text(encoding="utf-8")
    aot = cuda.split("if(VIBEQC_ENABLE_AOT_SHELLS)", 1)[1]
    assert "--split-compile=${VIBEQC_AOT_SPLIT_COMPILE_THREADS}" in aot
    assert "$<CUDA_COMPILER_ID:NVIDIA>" in aot


def test_release_preset_keeps_aot_split_compile_disabled() -> None:
    presets = json.loads((ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
    release = next(
        preset
        for preset in presets["configurePresets"]
        if preset["name"] == "cuda-release-sm120"
    )
    assert "VIBEQC_AOT_SPLIT_COMPILE_THREADS" not in release["cacheVariables"]


def test_fock_benchmark_records_aot_split_compile_identity() -> None:
    benchmark = (ROOT / "tools" / "benchmark_fock_strategies.py").read_text(
        encoding="utf-8"
    )
    assert '"VIBEQC_AOT_SPLIT_COMPILE_THREADS"' in benchmark


def test_dft_test_support_reuses_object_targets() -> None:
    tests = (ROOT / "cmake" / "VibeQCTests.cmake").read_text(encoding="utf-8")

    assert "add_library(vibeqc_dft_grid_test_objects OBJECT" in tests
    assert "add_library(vibeqc_dft_xc_test_objects OBJECT" in tests
    assert tests.count("$<TARGET_OBJECTS:vibeqc_dft_grid_test_objects>") == 3
    assert tests.count("$<TARGET_OBJECTS:vibeqc_dft_xc_test_objects>") == 2

    # Three grid/basis sources formerly compiled in three executables and two
    # XC/density sources in two executables. The object targets reduce those
    # 13 compile actions to five without changing test-only compile definitions.
    for source in (
        "src/dft/ao_grid.cpp",
        "src/dft/grid.cpp",
        "src/molecule/basis.cpp",
    ):
        assert tests.count(source) == 1
    for source in ("src/scf/density_factor.cpp", "src/dft/xc.cpp"):
        assert tests.count(source) == 1
