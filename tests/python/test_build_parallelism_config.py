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
