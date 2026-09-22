"""Boundary qualification for compiler-generated r2SCAN production code."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CPP = shutil.which("c++")

# The archived Libxc binary64 diagnostics are retained in the fixture. At the
# empty-spin floor they suffer cancellation themselves, so acceptance uses the
# original independent Libxc Maple C formulas evaluated with 113-bit arithmetic.
FIXTURE = json.loads((ROOT / "tests/data/xc/r2scan-tail-reference.json").read_text())
REFERENCE = tuple((point["inputs"], point["reference"]) for point in FIXTURE["points"])


def _generate(script: str, output: Path) -> None:
    """Exercise the actual AOT entry point without changing pytest imports."""
    subprocess.run(
        [sys.executable, "-I", str(ROOT / "tools" / script), "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.skipif(CPP is None, reason="C++ compiler unavailable")
def test_generated_r2scan_matches_high_precision_libxc_tail(tmp_path: Path) -> None:
    # Generator bootstrapping installs compiler-only package stubs. Keep it in
    # a child process so later runtime endpoint tests see the real packages.
    _generate("generate_xc_cpu.py", tmp_path / "r2scan.hpp")

    calls = []
    for values, _ in REFERENCE:
        arguments = ", ".join(repr(value) for value in values)
        calls.append(
            "{ auto v = vibeqc::dft::generated::r2scan_polarized("
            + arguments
            + '); std::printf("%.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\\n", '
            + "v.energy_density, v.feature_derivative[0], v.feature_derivative[1], "
            + "v.feature_derivative[2], v.feature_derivative[3], v.feature_derivative[4], "
            + "v.feature_derivative[5], v.feature_derivative[6]); }"
        )

    source = tmp_path / "probe.cpp"
    source.write_text(
        '#include <cstdio>\n#include "r2scan.hpp"\nint main() {\n'
        + "\n".join(calls)
        + "\n}\n"
    )
    executable = tmp_path / "probe"
    subprocess.run(
        [CPP, "-std=c++17", "-O2", str(source), "-o", str(executable)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=30
    )
    actual = np.asarray(
        [
            [float(value) for value in line.split()]
            for line in result.stdout.splitlines()
        ]
    )
    expected = np.asarray([values for _, values in REFERENCE])
    np.testing.assert_allclose(actual, expected, rtol=5e-12, atol=1e-12)


@pytest.mark.skipif(
    os.environ.get("VIBEQC_DFT_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated CUDA device",
)
def test_generated_cuda_r2scan_matches_high_precision_libxc_tail(
    tmp_path: Path,
) -> None:
    """Qualify CUDA against independent Libxc tail values and exchanged spins."""
    assert os.environ.get("SLURM_JOB_ID")
    compiler = shutil.which("nvcc")
    assert compiler is not None, "allocated CUDA qualification requires nvcc"
    _generate("generate_xc_r2scan_cuda.py", tmp_path / "r2scan.cuh")
    # Spin exchange permutes rho/sigma/tau channels without another oracle or
    # any dependence on the generated implementation's own mathematics.
    inputs = [values for values, _ in REFERENCE]
    outputs = [values for _, values in REFERENCE]
    inputs += [tuple(row[i] for i in (1, 0, 4, 3, 2, 6, 5)) for row in inputs[:]]
    outputs += [tuple(row[i] for i in (0, 2, 1, 5, 4, 3, 7, 6)) for row in outputs[:]]
    count = len(inputs)
    rows = ",\n".join(
        "{" + ",".join(float(x).hex() for x in row) + "}" for row in inputs
    )
    source = tmp_path / "probe.cu"
    source.write_text(
        '#include <cstdio>\n#include <cuda_runtime.h>\n#include "r2scan.cuh"\n'
        "__global__ void evaluate(const double* inputs, double* outputs) {\n"
        "  const auto i = blockIdx.x; const auto* p = inputs + 7 * i;\n"
        "  const auto v = vibeqc::dft::generated::r2scan_device(\n"
        "      p[0], p[1], p[2], p[3], p[4], p[5], p[6]);\n"
        "  outputs[8 * i] = v.energy_density;\n"
        "  for (unsigned k = 0; k < 7; ++k) outputs[8 * i + k + 1] = v.feature_derivative[k];\n"
        "}\n"
        f"int main() {{ const double inputs[{count}][7] = {{{rows}}};\n"
        f"  double outputs[{count}][8]; double *input_device, *output_device;\n"
        "  if (cudaMalloc(&input_device, sizeof(inputs)) != cudaSuccess ||\n"
        "      cudaMalloc(&output_device, sizeof(outputs)) != cudaSuccess) return 1;\n"
        "  if (cudaMemcpy(input_device, inputs, sizeof(inputs), cudaMemcpyHostToDevice)\n"
        "      != cudaSuccess) return 2;\n"
        f"  evaluate<<<{count}, 1>>>(input_device, output_device);\n"
        "  if (cudaGetLastError() != cudaSuccess ||\n"
        "      cudaMemcpy(outputs, output_device, sizeof(outputs), cudaMemcpyDeviceToHost)\n"
        "      != cudaSuccess) return 3;\n"
        "  cudaFree(input_device); cudaFree(output_device);\n"
        "  for (const auto& row : outputs) for (unsigned k = 0; k < 8; ++k)\n"
        "    std::printf(\"%.17g%c\", row[k], k == 7 ? '\\n' : ' ');\n}\n"
    )
    executable = tmp_path / "probe"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-arch=native",
            "--fmad=false",
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    result = subprocess.run(
        [str(executable)], check=True, capture_output=True, text=True, timeout=30
    )
    actual = np.asarray(
        [[float(x) for x in row.split()] for row in result.stdout.splitlines()]
    )
    np.testing.assert_allclose(actual, outputs, rtol=5e-12, atol=1e-12)
