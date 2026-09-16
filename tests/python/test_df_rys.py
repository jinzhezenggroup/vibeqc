"""Independent numerical/provenance gates for low-order DF Rys quadrature."""

import ctypes
import math
import os
import random
import shutil
import subprocess
from pathlib import Path

import pytest
from vibeqc_compiler.integral.df_rys import (
    emit_df_rys_cuda,
    rys_roots,
)


def arguments():
    """Cover cancellation, branch boundaries, random interiors and asymptotics."""
    rng = random.Random(394)
    values = [0.0, 1e300]
    values += [10.0 ** (-30 + k / 10) for k in range(381)]
    values += [rng.uniform(0, 60) for _ in range(512)]
    for boundary in (0.5, *(float(2 * k) for k in range(1, 31))):
        values += [
            math.nextafter(boundary, 0),
            boundary,
            math.nextafter(boundary, math.inf),
        ]
    return values


def check_reference(evaluate, nroots):
    mp = pytest.importorskip("mpmath")
    with mp.workdps(75):
        for argument in arguments():
            actual = evaluate(argument, nroots)
            t = mp.mpf(argument)
            moments = [
                mp.mpf(1) / (2 * k + 1)
                if not t
                else mp.gammainc(k + mp.mpf("0.5"), 0, t)
                / (2 * t ** (k + mp.mpf("0.5")))
                for k in range(2)
            ]
            reference = ((moments[1] / moments[0],), (moments[0],))
            nodes, weights = actual
            assert len(nodes) == len(weights) == nroots
            assert all(0 < x < 1 for x in nodes)
            assert all(w > 0 for w in weights)
            assert tuple(sorted(nodes)) == tuple(nodes)
            for observed, expected in zip(
                (*nodes, *weights), (*reference[0], *reference[1]), strict=True
            ):
                assert abs(mp.mpf(observed) / expected - 1) < mp.mpf("5e-14"), (
                    argument,
                    actual,
                    reference,
                )
            # Check the defining integrals directly in addition to the
            # individual node/weight errors.
            t = mp.mpf(argument)
            for order in range(2 * nroots):
                moment = sum(
                    mp.mpf(w) * mp.mpf(x) ** order
                    for x, w in zip(nodes, weights, strict=True)
                )
                expected = (
                    mp.mpf(1) / (2 * order + 1)
                    if not t
                    else mp.gammainc(order + mp.mpf("0.5"), 0, t)
                    / (2 * t ** (order + mp.mpf("0.5")))
                )
                assert abs(moment / expected - 1) < mp.mpf("5e-14")


@pytest.mark.parametrize("nroots", (1,))
def test_python_evaluator_matches_independent_integrals(nroots):
    check_reference(rys_roots, nroots)


@pytest.mark.parametrize(
    "argument,nroots", ((-1, 1), (math.inf, 2), (math.nan, 1), (0, 3))
)
def test_invalid_host_evaluator_domain(argument, nroots):
    with pytest.raises(ValueError):
        rys_roots(argument, nroots)


@pytest.fixture(scope="module")
def cuda_evaluator(tmp_path_factory):
    """Evaluate the generated one-root arithmetic on the allocated GPU.

    Batch the full argument grid in each kernel, including different evaluator
    branches within a warp. No runtime library or handwritten root formula is
    used. Compile only during correctness validation, outside clean timing.
    """
    assert os.environ.get("SLURM_JOB_ID"), "allocate a GPU through Slurm"
    compiler = os.environ.get("CUDACXX") or shutil.which("nvcc")
    if compiler is None and os.environ.get("CUDA_HOME"):
        compiler = str(Path(os.environ["CUDA_HOME"]) / "bin" / "nvcc")
    if compiler:
        compiler = shutil.which(compiler) or compiler
    if not compiler or not Path(compiler).is_file():
        pytest.skip("CUDA compiler unavailable")
    directory = tmp_path_factory.mktemp("df_rys_cuda")
    (directory / "generated_df_rys.cuh").write_text(emit_df_rys_cuda())
    source = directory / "probe.cu"
    source.write_text(r"""
#include <cuda_runtime.h>
#include "generated_df_rys.cuh"
template<unsigned N>
__global__ void evaluate(const double* arguments, size_t count, double* output) {
  const auto index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  double nodes[2]{}, weights[2]{};
  vibeqc::scf::generated_df_rys::roots<N>(arguments[index], nodes, weights);
  for (unsigned root = 0; root < 2; ++root) {
    output[4 * index + root] = nodes[root];
    output[4 * index + 2 + root] = weights[root];
  }
}
extern "C" int probe(unsigned roots, const double* input, size_t count, double* output) {
  if (!count || roots != 1) return cudaErrorInvalidValue;
  double *arguments = nullptr, *values = nullptr;
  auto status = cudaMalloc(&arguments, count * sizeof(double));
  if (status == cudaSuccess) status = cudaMalloc(&values, 4 * count * sizeof(double));
  if (status == cudaSuccess)
    status = cudaMemcpy(arguments, input, count * sizeof(double), cudaMemcpyHostToDevice);
  if (status == cudaSuccess) {
    evaluate<1><<<(count + 127) / 128, 128>>>(arguments, count, values);
    status = cudaGetLastError();
  }
  if (status == cudaSuccess)
    status = cudaMemcpy(output, values, 4 * count * sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(values);
  cudaFree(arguments);
  return status;
}
""")
    output = directory / "probe.so"
    subprocess.run(
        [
            compiler,
            "-O3",
            "-std=c++17",
            "-arch=native",
            "-shared",
            "-Xcompiler=-fPIC",
            str(source),
            "-I",
            str(directory),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=240,
    )
    library = ctypes.CDLL(str(output))
    double = ctypes.POINTER(ctypes.c_double)
    library.probe.argtypes = [ctypes.c_uint, double, ctypes.c_size_t, double]
    library.probe.restype = ctypes.c_int
    grid = arguments()
    input_values = (ctypes.c_double * len(grid))(*grid)
    observed = {}
    for roots in (1,):
        values = (ctypes.c_double * (4 * len(grid)))()
        status = library.probe(roots, input_values, len(grid), values)
        assert status == 0, f"CUDA Rys evaluation failed with status {status}"
        observed[roots] = {
            argument: (
                tuple(values[4 * index + root] for root in range(roots)),
                tuple(values[4 * index + 2 + root] for root in range(roots)),
            )
            for index, argument in enumerate(grid)
        }
    return lambda argument, roots: observed[roots][argument]


@pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)
@pytest.mark.parametrize("nroots", (1,))
def test_cuda_evaluator_matches_independent_integrals(cuda_evaluator, nroots):
    check_reference(cuda_evaluator, nroots)


@pytest.fixture(scope="module")
def emitted_evaluator(tmp_path_factory):
    """Compile the exact CUDA arithmetic as ordinary C++, including its FMAs.

    This gate does not probe or load a GPU runtime. Real CUDA compilation and
    force-level validation are separate required promotion checks.
    """
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    directory = tmp_path_factory.mktemp("df_rys")
    source = directory / "probe.cpp"
    source.write_text(
        "#define __device__\n#define __forceinline__ inline\n"
        + emit_df_rys_cuda()
        + r"""
extern "C" void probe(unsigned n,double t,double* nodes,double* weights) {
  if(n==1) vibeqc::scf::generated_df_rys::roots<1>(t,nodes,weights);
}
"""
    )
    output = directory / "probe.so"
    subprocess.run(
        [
            compiler,
            "-O2",
            "-std=c++20",
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    library = ctypes.CDLL(str(output))
    double_pointer = ctypes.POINTER(ctypes.c_double)
    library.probe.argtypes = [
        ctypes.c_uint,
        ctypes.c_double,
        double_pointer,
        double_pointer,
    ]
    library.probe.restype = None

    def evaluate(argument, nroots):
        nodes = (ctypes.c_double * nroots)()
        weights = (ctypes.c_double * nroots)()
        library.probe(nroots, argument, nodes, weights)
        return tuple(nodes), tuple(weights)

    return evaluate


@pytest.mark.parametrize("nroots", (1,))
def test_emitted_evaluator_matches_independent_integrals(emitted_evaluator, nroots):
    check_reference(emitted_evaluator, nroots)
