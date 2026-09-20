"""Compile one generated stationary r2SCAN CUDA artifact without requiring a GPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.integral.first_derivative_native import emit_first_derivative_cuda
from vibeqc_compiler.method import resolve_method
from vibeqc_compiler.method.stationary_cuda import compile_stationary_cuda
from vibeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--architecture", default="sm_120")
    args = parser.parse_args()

    requests = (
        ("overlap", ("", "")),
        ("kinetic", ("", "")),
        ("nuclear_attraction", ("", "")),
        ("four_center_eri", ("", "", "", "")),
        ("nuclear", ()),
    )
    compiler = CudaCompilerAdapter(
        args.nvcc,
        cuda_target_info(args.architecture),
        compile_timeout=600,
    )
    artifact = compile_stationary_cuda(
        emit_first_derivative_cuda(requests),
        functional=2,
        plan=StationaryGradientPlan(
            resolve_method("R2SCAN", spin="unpolarized"),
            StationaryMeanField(SCF_POINT_MODEL),
        ),
        iterations=3,
        compiler=compiler,
        cache=args.cache,
    )
    if not artifact.library.is_file():
        raise RuntimeError("stationary r2SCAN CUDA compile produced no library")
    print(artifact.library)


if __name__ == "__main__":
    main()
