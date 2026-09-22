"""The order-two generated-math adapter must own its header dependencies."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from vibeqc_compiler.integral.weighted_eri_cuda import emit_low_order_weighted_header

ROOT = Path(__file__).resolve().parents[2]


def test_order2_force_names_its_vector_accessor_owner() -> None:
    source = (ROOT / "src/scf/cuda/direct_force_order2.cuh").read_text()
    assert '#include "scf/cuda/cartesian_angular.cuh"' in source


def test_order2_force_header_compiles_without_prior_native_math(tmp_path: Path) -> None:
    compiler = os.environ.get("VIBEQC_NVCC") or shutil.which("nvcc")
    if not compiler:
        pytest.skip("NVCC unavailable for the standalone CUDA header gate")
    (tmp_path / "weighted_eri.cuh").write_text(emit_low_order_weighted_header())
    source = tmp_path / "probe.cu"
    source.write_text('#include "scf/cuda/direct_force_order2.cuh"\n')
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-c",
            "-I" + str(ROOT / "src"),
            "-I" + str(ROOT / "include"),
            "-I" + str(tmp_path),
            str(source),
            "-o",
            str(tmp_path / "probe.o"),
        ],
        check=True,
        capture_output=True,
        timeout=180,
    )
