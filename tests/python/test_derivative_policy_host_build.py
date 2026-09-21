"""Shared host policy must generate without compiling derivative mathematics."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_policy_only_generation_compiles_actual_host_consumer(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    header = tmp_path / "generated_one_electron_derivative_policy.cuh"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools/generate_one_electron_kernels.py"),
            "--derivatives",
            "--derivative-policy-output",
            str(header),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    assert "schedule_code=3U" in header.read_text()
    assert list(tmp_path.iterdir()) == [header]
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            f"-I{ROOT / 'src'}",
            f"-I{ROOT / 'include'}",
            f"-I{tmp_path}",
            "-c",
            str(ROOT / "src/scf/cuda/rhf_policy.cpp"),
            "-o",
            str(tmp_path / "policy.o"),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )


def test_cpu_only_cmake_owns_derivative_policy_target(tmp_path: Path) -> None:
    if not all(shutil.which(x) for x in ("cmake", "ninja", "c++")):
        pytest.skip("CMake/Ninja/C++ unavailable")
    build = tmp_path / "build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(ROOT),
            "-B",
            str(build),
            "-G",
            "Ninja",
            "-DVIBEQC_ENABLE_CUDA=OFF",
            "-DVIBEQC_ENABLE_AOT_SHELLS=OFF",
            "-DVIBEQC_BUILD_TESTS=OFF",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ],
        check=True,
        capture_output=True,
        timeout=90,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build),
            "--target",
            "vibeqc_one_electron_derivative_policy_codegen",
            "-j2",
        ],
        check=True,
        capture_output=True,
        timeout=90,
    )
    header = build / "generated/generated_one_electron_derivative_policy.cuh"
    assert "schedule_code=3U" in header.read_text()
