"""Allocated FP32 AO jets against retained independent libcint fixtures.

This qualifies the generated arithmetic through third derivatives, including
diffuse/tight and Cartesian/spherical f shells. It does not admit mixed-precision
SCF or force endpoints; the native XC test separately gates energy and Vxc.
"""

from __future__ import annotations

import ctypes as ct
import os
import shutil
import subprocess

import numpy as np
import pytest
from vibeqc_compiler.dft import NativeAO
from vibeqc_compiler.dft.ao_cuda import (
    emit_grid_policy,
    emit_grid_scientific_kernels,
)
from vibeqc_compiler.dft.fixtures import NAMES, basis_arguments, load_fixture

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_GRID_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.fixture(scope="module")
def fp32_probe(tmp_path_factory: pytest.TempPathFactory) -> ct.CDLL:
    """Compile the actual emitted AO body with only a transport test wrapper."""
    assert os.environ.get("SLURM_JOB_ID")
    compiler = os.environ.get("VIBEQC_NVCC") or shutil.which("nvcc")
    assert compiler, "allocated qualification requires nvcc"
    directory = tmp_path_factory.mktemp("ao-fp32-cuda")
    policy = emit_grid_policy()
    end = policy.index(";", policy.index("__constant__ int derivatives")) + 1
    kernels = emit_grid_scientific_kernels()
    begin = kernels.index("__global__ void ao_kernel(")
    end_kernels = kernels.index("__global__ void feature_kernel(")
    source = directory / "probe.cu"
    source.write_text(
        "#include <cuda_runtime.h>\n#include <cmath>\n#include <cstddef>\n"
        "using I = std::size_t;\n"
        "__device__ double finite(double x, int* error, int) {\n"
        "  if (!isfinite(x)) atomicExch(error, 1); return x;\n}\n"
        + policy[:end]
        + "\n}\nusing namespace vibeqc_grid_policy;\n"
        + kernels[begin:end_kernels]
        + r"""
extern "C" int evaluate(const double* basis, size_t nbasis, size_t natom,
                        size_t nprimitive, size_t nao, const double* points,
                        size_t npoint, double* out) {
  double *b=nullptr, *p=nullptr, *o=nullptr;
  int *error=nullptr, result=0;
  const size_t count=20*npoint*nao;
  const auto release = [&] {
    cudaFree(b); cudaFree(p); cudaFree(o); cudaFree(error);
  };
#define CHECK(call) do { if ((call)!=cudaSuccess) { release(); return 2; } } while (0)
  CHECK(cudaMalloc(&b, nbasis*sizeof(double)));
  CHECK(cudaMalloc(&p, 3*npoint*sizeof(double)));
  CHECK(cudaMalloc(&o, count*sizeof(double)));
  CHECK(cudaMalloc(&error, sizeof(int)));
  CHECK(cudaMemcpy(b,basis,nbasis*sizeof(double),cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(p,points,3*npoint*sizeof(double),cudaMemcpyHostToDevice));
  CHECK(cudaMemset(error,0,sizeof(int)));
  // A nondivisible launch also covers the emitted grid-stride tail.
  ao_kernel_fp32<<<7, 128>>>(b,natom,nprimitive,nao,p,npoint,20,o,error,nullptr);
  CHECK(cudaGetLastError());
  CHECK(cudaMemcpy(out,o,count*sizeof(double),cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(&result,error,sizeof(int),cudaMemcpyDeviceToHost));
  release();
  return result;
}
"""
    )
    library = directory / "probe.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "--fmad=false",
            "-shared",
            "-Xcompiler=-fPIC",
            "-arch=" + os.environ.get("VIBEQC_GRID_CUDA_ARCH", "sm_120"),
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        timeout=180,
    )
    probe = ct.CDLL(str(library))
    pointer = ct.POINTER(ct.c_double)
    probe.evaluate.argtypes = [
        pointer,
        *([ct.c_size_t] * 4),
        pointer,
        ct.c_size_t,
        pointer,
    ]
    probe.evaluate.restype = ct.c_int
    return probe


@pytest.mark.parametrize("name", NAMES)
def test_fp32_ao_jets_against_independent_reference(
    fp32_probe: ct.CDLL, name: str
) -> None:
    meta, arrays = load_fixture(name)
    points = np.ascontiguousarray(arrays["points"])
    expected = arrays["ao_jets"]
    actual = np.empty_like(expected)
    pointer = ct.POINTER(ct.c_double)
    with NativeAO(**basis_arguments(meta)) as basis:
        status = fp32_probe.evaluate(
            basis.packed.ctypes.data_as(pointer),
            basis.packed.size,
            basis.natom,
            basis.nprimitive,
            basis.nao,
            points.ctypes.data_as(pointer),
            len(points),
            actual.ctypes.data_as(pointer),
        )
    assert status == 0
    assert np.isfinite(actual).all()
    # Cancellation zeros make pointwise relative errors ill-conditioned. Gate
    # each derivative component against its own independent maximum magnitude;
    # do not let large third derivatives hide errors in values or first jets.
    scale = np.maximum(np.max(np.abs(expected), axis=(1, 2)), 1e-30)
    error = np.max(np.abs(actual - expected), axis=(1, 2)) / scale
    assert np.max(error) <= 5e-6, (name, error)
