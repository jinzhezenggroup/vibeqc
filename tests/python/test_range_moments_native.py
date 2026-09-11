"""Native modified moments against independent adaptive interval quadrature."""

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.integral.range_separation import CoulombKernel, reference_moments


@pytest.fixture(scope="module", params=("cpu", "cuda"))
def native_moments(request, tmp_path_factory):
    """Compile identical host/device arithmetic; CUDA execution is opt-in Slurm."""
    pytest.importorskip("scipy")
    cuda = request.param == "cuda"
    if cuda and os.environ.get("VIBEQC_TEST_RANGE_CUDA") != "1":
        pytest.skip("set VIBEQC_TEST_RANGE_CUDA=1 inside a Slurm GPU job")
    if cuda and not os.environ.get("SLURM_JOB_ID"):
        pytest.fail("native CUDA validation requires a Slurm allocation")
    compiler = shutil.which("nvcc" if cuda else "c++")
    if compiler is None:
        pytest.skip("native compiler unavailable")
    folder = tmp_path_factory.mktemp(f"range-moments-{request.param}")
    source = '#include "integrals/range_moments.hpp"\n'
    if cuda:
        source += r"""
#include <cuda_runtime.h>
__global__ void kernel(unsigned order, double t, double rho, unsigned range,
                       double omega, double* out) {
  out[14] = vibeqc::integrals::range_moments(
      order, t, rho, static_cast<vibeqc::integrals::CoulombRange>(range), omega, out);
}
extern "C" int evaluate(unsigned order, double t, double rho, unsigned range,
                        double omega, double* out) {
  double* device = nullptr;
  if (cudaMalloc(&device, 15 * sizeof(double)) != cudaSuccess) return -1;
  auto status = cudaMemcpy(device, out, 15 * sizeof(double), cudaMemcpyHostToDevice);
  if (status == cudaSuccess) {
    kernel<<<1,1>>>(order, t, rho, range, omega, device);
    status = cudaGetLastError();
  }
  if (status == cudaSuccess)
    status = cudaMemcpy(out, device, 15 * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(device);
  return status == cudaSuccess ? static_cast<int>(out[14]) : -1;
}
"""
    else:
        source += r"""
extern "C" int evaluate(unsigned order, double t, double rho, unsigned range,
                        double omega, double* out) {
  return vibeqc::integrals::range_moments(
      order, t, rho, static_cast<vibeqc::integrals::CoulombRange>(range), omega, out);
}
"""
    path = folder / ("probe.cu" if cuda else "probe.cpp")
    path.write_text(source)
    library = folder / "probe.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O3",
            "-shared",
            *(
                ["-arch=sm_120", "--fmad=false", "-Xcompiler=-fPIC"]
                if cuda
                else ["-fPIC", "-ffp-contract=off"]
            ),
            "-I" + str(Path(__file__).resolve().parents[2] / "src"),
            str(path),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    owner = ctypes.CDLL(str(library))
    owner.evaluate.argtypes = [
        ctypes.c_uint,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_uint,
        ctypes.c_double,
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
    ]
    owner.evaluate.restype = ctypes.c_int
    return owner.evaluate


@pytest.mark.parametrize("family", ["long_range", "short_range"])
@pytest.mark.parametrize("rho", [1e-12, 0.73, 1e12])
def test_native_moments_resolve_limits_and_separate_parts(native_moments, family, rho):
    """All orders retain relative accuracy, including tiny positive SR values."""
    tag = 1 if family == "long_range" else 2
    for omega in (0, 1e-12, 0.2, 1, 1e6, 1e150):
        for argument in (0, 1e-12, 0.1, 10, 100, 700, 1e6, 1e150, 1e300):
            actual = np.full(15, 123.0)
            assert native_moments(13, argument, rho, tag, omega, actual) == 1
            expected = reference_moments(
                13, argument, rho, CoulombKernel(family, omega)
            )
            np.testing.assert_allclose(
                actual[:14],
                expected,
                atol=1e-320,
                rtol=2e-12,
                err_msg=f"{family}, omega={omega}, T={argument}, rho={rho}",
            )
            assert np.all(actual[:14] >= 0)


def test_native_moment_controls_fail_without_touching_output(native_moments):
    cases = [
        (14, 1, 1, 1, 0.5),
        (0, float("nan"), 1, 1, 0.5),
        (0, float("inf"), 1, 1, 0.5),
        (0, -1, 1, 1, 0.5),
        (0, 1, 0, 1, 0.5),
        (0, 1, float("inf"), 1, 0.5),
        (0, 1, 1, 3, 0.5),
        (0, 1, 1, 0, 0.5),
        (0, 1, 1, 1, -1),
        (0, 1, 1, 1, float("nan")),
        (0, 1, 1, 1, float("inf")),
    ]
    for controls in cases:
        actual = np.full(15, 123.0)
        assert native_moments(*controls, actual) == 0
        np.testing.assert_array_equal(actual[:14], 123)


def test_native_moments_derivative_chain_and_complement(native_moments):
    values = []
    for tag in (0, 1, 2):
        omega = 0 if tag == 0 else 0.8
        output = np.zeros(15)
        assert native_moments(13, 1.3, 0.71, tag, omega, output) == 1
        values.append(output[:14].copy())
        errors = []
        for step in (0.01, 0.003, 0.001):
            plus, minus = np.zeros(15), np.zeros(15)
            assert native_moments(13, 1.3 + step, 0.71, tag, omega, plus) == 1
            assert native_moments(13, 1.3 - step, 0.71, tag, omega, minus) == 1
            errors.append(
                np.max(np.abs((plus[:13] - minus[:13]) / (2 * step) + output[1:14]))
            )
        assert errors[2] < errors[1] < errors[0]
        assert errors[2] < 2e-8
    np.testing.assert_allclose(values[1] + values[2], values[0], rtol=2e-13, atol=0)
