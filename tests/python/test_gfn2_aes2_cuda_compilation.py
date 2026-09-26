"""CUDA constants must compile under the production NVCC language contract."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("consumer", [False, True])
def test_aes2_cuda_uses_device_callable_constants(
    tmp_path: Path, consumer: bool
) -> None:
    compiler = os.environ.get("VIBEQC_NVCC") or shutil.which("nvcc")
    if not compiler:
        pytest.skip("NVCC required for the AES2 device compilation gate")
    header = tmp_path / "generated_gfn2_aes2_native.cuh"
    subprocess.run(
        [
            sys.executable,
            "-S",
            str(ROOT / "tools/generate_gfn2_aes2_native.py"),
            "--cuda-output",
            str(header),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    runtime = ROOT / "src/xtb/native"
    source = runtime / "src/backends/cuda/gfn2_aes2.cu"
    if not consumer:
        source = tmp_path / "header.cu"
        source.write_text('#include "generated_gfn2_aes2_native.cuh"\n')
    result = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-arch=sm_80",
            "-c",
            str(source),
            "-I" + str(tmp_path),
            "-I" + str(ROOT / "src"),
            "-I" + str(runtime),
            "-I" + str(runtime / "include"),
            "-I" + str(runtime / "src"),
            "-o",
            str(tmp_path / "probe.o"),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
