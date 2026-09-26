"""Execute the exact plan-derived reduction on host and opt-in CUDA."""

import ctypes as ct
import os
import shutil
import subprocess

import numpy as np
import pytest
from vibeqc_compiler.method import resolve_method
from vibeqc_compiler.method.stationary_cuda import emit_stationary_reduction_cuda
from vibeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)


@pytest.fixture(scope="module", params=("host", "cuda"))
def reduction(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> ct.CDLL:
    cuda = request.param == "cuda"
    if cuda and os.environ.get("VIBEQC_STATIONARY_REDUCTION_CUDA_TEST") != "1":
        pytest.skip(
            "set VIBEQC_STATIONARY_REDUCTION_CUDA_TEST=1 in an allocated GPU job"
        )
    if cuda and not os.environ.get("SLURM_JOB_ID"):
        pytest.fail("CUDA reduction qualification requires Slurm allocation")
    compiler = shutil.which("nvcc" if cuda else "c++")
    if compiler is None:
        pytest.fail("requested compiler is unavailable")
    plan = StationaryGradientPlan(
        resolve_method("PBE"), StationaryMeanField(SCF_POINT_MODEL)
    )
    emitted = emit_stationary_reduction_cuda(plan)
    prefix = "#include <cmath>\n#include <cstddef>\n#include <vector>\n"
    if cuda:
        prefix += "#include <cuda_runtime.h>\n"
    else:
        prefix += "#define __global__\n#define __device__\nstruct Dim {size_t x;};\nDim blockIdx{},blockDim{64},threadIdx{};\nint atomicExch(int* p,int v){int old=*p;*p=v;return old;}\n"
    prefix += "namespace vibeqc_stationary_cuda { __device__ double finite(double x,int* error,int){if(!isfinite(x)) atomicExch(error,1);return x;} }\n"
    if not cuda:
        prefix = prefix.replace("isfinite(x)", "std::isfinite(x)")
    wrapper = CUDA if cuda else HOST
    folder = tmp_path_factory.mktemp(f"stationary-reduction-{request.param}")
    source = folder / ("reduction.cu" if cuda else "reduction.cpp")
    source.write_text(prefix + emitted + wrapper)
    library = folder / "reduction.so"
    flags = (
        [
            "-arch=" + os.environ.get("VIBEQC_TEST_CUDA_ARCH", "sm_120"),
            "--fmad=false",
            "-Xcompiler=-fPIC",
        ]
        if cuda
        else ["-ffp-contract=off", "-fPIC"]
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-shared",
            *flags,
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    result = ct.CDLL(str(library))
    result.run.argtypes = [
        ct.POINTER(ct.c_double),
        ct.c_size_t,
        ct.POINTER(ct.c_double),
    ]
    result.run.restype = ct.c_int
    return result


HOST = r"""
extern "C" int run(const double* input,size_t na,double* output) {
  int error=0;std::vector<double> candidate(3*na,-99);
  for(size_t i=0;i<3*na+7;++i){blockIdx.x=i/64;threadIdx.x=i%64;
    vibeqc_stationary_cuda::source_reduce(input,na,candidate.data(),&error);}
  if(error) return error;
  for(size_t i=0;i<3*na;++i) output[i]=candidate[i];
  return 0;
}
"""
CUDA = r"""
extern "C" int run(const double* input,size_t na,double* output) {
  double *in=nullptr,*out=nullptr;int *error=nullptr,result=-1;
  do {
    if(cudaMalloc(&in,21*na*8)!=cudaSuccess || cudaMalloc(&out,3*na*8)!=cudaSuccess || cudaMalloc(&error,4)!=cudaSuccess) break;
    if(cudaMemcpy(in,input,21*na*8,cudaMemcpyHostToDevice)!=cudaSuccess || cudaMemset(error,0,4)!=cudaSuccess) break;
    vibeqc_stationary_cuda::source_reduce<<<(3*na+63)/64,64>>>(in,na,out,error);
    if(cudaGetLastError()!=cudaSuccess || cudaMemcpy(&result,error,4,cudaMemcpyDeviceToHost)!=cudaSuccess) {result=-1;break;}
    if(!result && cudaMemcpy(output,out,3*na*8,cudaMemcpyDeviceToHost)!=cudaSuccess) result=-1;
  }while(false);
  cudaFree(in);cudaFree(out);cudaFree(error);return result;
}
"""


@pytest.mark.parametrize("atoms", (1, 3, 32))
def test_generated_reduction_preserves_order_and_signed_values(
    reduction: ct.CDLL, atoms: int
) -> None:
    values = np.random.default_rng(777).normal(size=(7, atoms, 3))
    values[:3, 0, 0] = [1e16, 1.0, -1e16]
    values[:, 0, 1] = -0.0
    expected = np.zeros((atoms, 3))
    for source in values:
        expected += source
    actual = np.full_like(expected, -99.0)
    status = reduction.run(
        values.ctypes.data_as(ct.POINTER(ct.c_double)),
        atoms,
        actual.ctypes.data_as(ct.POINTER(ct.c_double)),
    )
    assert status == 0
    assert actual.tobytes() == expected.tobytes()


def test_nonfinite_reduction_does_not_publish(reduction: ct.CDLL) -> None:
    values = np.ones((7, 3, 3))
    values[-1, -1, -1] = np.nan
    output = np.full((3, 3), -99.0)
    assert (
        reduction.run(
            values.ctypes.data_as(ct.POINTER(ct.c_double)),
            3,
            output.ctypes.data_as(ct.POINTER(ct.c_double)),
        )
        == 1
    )
    np.testing.assert_array_equal(output, -99.0)
