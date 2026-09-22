"""Boundary qualification for compiler-generated r2SCAN production code."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CPP = shutil.which("c++")

# Independent compiled Libxc 7.0.0 values from the #1028 qualification probe.
REFERENCE = (
    (
        (0.073, 0.0, 0.021315999999999998, 0.0, 0.0, 0.05109999999999999, 0.0),
        (
            -0.03267288239243613,
            -0.5905186926667251,
            12.78273217828023,
            -0.07732912932848196,
            0.07560965137214307,
            0.03780482568607153,
            0.04700651202712578,
            -0.020311812271276885,
        ),
    ),
    (
        (0.0073, 0.0, 0.00021316, 0.0, 0.0, 0.00511, 0.0),
        (
            -0.0014132531993828632,
            -0.25095400954486075,
            -2.5087209282090828,
            -1.2375950557366102,
            0.979904289463213,
            0.4899521447316065,
            0.07855975356000833,
            -0.022931692425350425,
        ),
    ),
    (
        (0.00073, 0.0, 2.1315999999999996e-06, 0.0, 0.0, 0.000511, 0.0),
        (
            -5.974988984373865e-05,
            -0.11840278794271093,
            -3.6854566480993625,
            0.5523913350949331,
            3.003930663581292,
            1.501965331790646,
            0.005595854566402719,
            -0.004881234106355745,
        ),
    ),
    (
        (7.3e-05, 0.0, 2.1316e-08, 0.0, 0.0, 5.1099999999999995e-05, 0.0),
        (
            -2.5637583731920647e-06,
            -0.055116348672106845,
            -0.5631602714694769,
            11.109323744143314,
            8.69087344225384,
            4.34543672112692,
            0.00029873019968308147,
            -0.0005169812760035973,
        ),
    ),
    (
        (
            9.190155506097421e-06,
            0.0,
            3.37835832905011e-10,
            0.0,
            0.0,
            6.433108854268195e-06,
            0.0,
        ),
        (
            -1.4372142317377134e-07,
            -0.026008158606713943,
            0.04094251399167861,
            55.254777114683854,
            29.050547991206304,
            14.525273995603152,
            2.6615594943264455e-05,
            -7.477735560731988e-05,
        ),
    ),
)


# Independent Libxc 7.0.0 C API at an actual H2 default-grid point and its
# two adjacent FP64 occupied densities. The minority potential is sensitive to
# 1-zeta cancellation: endpoint comparisons must hold these inputs identical.
# Preserve the original fixtures and their 5e-12 / 1e-12 gates above.
ROUNDING_REFERENCE = (
    (
        (
            0.04603137593786226,
            0.0,
            0.0023299235150585004,
            0.0,
            0.0,
            0.029349040301134496,
            0.0,
        ),
        (
            -0.016665153367926778,
            -0.5205168262667701,
            60.04172865208638,
            -0.20806057887363344,
            0.30181603355670217,
            0.15090801677835108,
            0.06585768202669184,
            -0.0318473576993236,
        ),
    ),
    (
        (
            0.04603137593786225,
            0.0,
            0.0023299235150585004,
            0.0,
            0.0,
            0.029349040301134496,
            0.0,
        ),
        (
            -0.01666515336792677,
            -0.5205168262667699,
            60.0417286520867,
            -0.20806057887363363,
            0.301816033556702,
            0.150908016778351,
            0.0658576820266919,
            -0.0318473576993235,
        ),
    ),
    (
        (
            0.046031375937862266,
            0.0,
            0.0023299235150585004,
            0.0,
            0.0,
            0.029349040301134496,
            0.0,
        ),
        (
            -0.016665153367927083,
            -0.5205168262667764,
            60.14451563784888,
            -0.20806057887360477,
            0.3018160335567596,
            0.1509080167783798,
            0.06585768202667963,
            -0.031577007658775844,
        ),
    ),
)
REFERENCE += ROUNDING_REFERENCE


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
def test_generated_r2scan_matches_zero_minority_spin_libxc(tmp_path: Path) -> None:
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
def test_generated_cuda_r2scan_zero_minority_spin_libxc(tmp_path: Path) -> None:
    """Exercise device FP64 emission against Libxc, including exchanged spins."""
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
