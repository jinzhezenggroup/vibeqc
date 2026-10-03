"""Independent full-block gates for the generated auxiliary-g value domain."""

from __future__ import annotations

import ctypes as ct
import os
import shutil
import subprocess
import typing
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.integral.df_cuda import (
    emit_df_values_cpu,
    emit_df_values_cuda,
)
from generativeqc_compiler.integral.df_derivatives import axis_polynomial

from tools.generativeqc_validation.df_values import make_df_value_fixture

ROOT = Path(__file__).resolve().parents[2]
SIGNATURES = [a for a in product(range(5), repeat=2) if 4 in a] + [
    (a, b, 4) for a, b in product(range(4), repeat=2)
]


@pytest.fixture(scope="module", params=["cpu", "cuda"])
def value_probe(
    request: typing.Any, tmp_path_factory: pytest.TempPathFactory
) -> typing.Any:
    """Compile the identical generated value dispatch for host and allocated CUDA.

    Inputs use the validation fixture's 144-byte primitive ABI. CPU execution
    is an emission check; only the separate CUDA parameter exercises the GPU.
    """
    cuda = request.param == "cuda"
    if cuda and os.environ.get("GENERATIVEQC_DF_G_CUDA_TEST") != "1":
        pytest.skip("requires explicit finite Slurm allocation")
    if cuda:
        assert os.environ.get("SLURM_JOB_ID"), "allocate GPU through Slurm"
    compiler = shutil.which(os.environ.get("CUDACXX", "nvcc") if cuda else "c++")
    cache = shutil.which("ccache")
    if not compiler or not cache:
        pytest.skip("compiler and ccache required")
    directory = tmp_path_factory.mktemp("df-g-" + request.param)
    header = emit_df_values_cuda() if cuda else emit_df_values_cpu()
    (directory / "values.h").write_text(header)
    source = directory / ("probe.cu" if cuda else "probe.cpp")
    common = r"""
#include <cstddef>
#include <cstdint>
#include "values.h"
namespace df = generativeqc::scf::generated_df;
struct Input {
  std::uint32_t count; df::Angular angular[3]; df::Vec3 centers[3];
  double exponents[3], weight;
};
static_assert(sizeof(Input)==144 && offsetof(Input,weight)==136);
QUALIFIER double value(const Input& in) {
  const auto* e=in.exponents; const auto* r=in.centers; const auto* a=in.angular;
  return in.weight*(in.count==2 ? df::metric(e[0],r[0],a[0],e[2],r[2],a[2])
      : df::three_center(e[0],r[0],a[0],e[1],r[1],a[1],e[2],r[2],a[2]));
}
"""
    gpu = r"""
__global__ void values(const Input* inputs,std::size_t n,double* output) {
  const auto i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n) output[i]=value(inputs[i]);
}
extern "C" int probe(const Input* inputs,std::size_t n,double* output) {
  Input* d=nullptr; double* o=nullptr;
  auto status=cudaMalloc(&d,n*sizeof(Input));
  if(status==cudaSuccess) status=cudaMalloc(&o,n*sizeof(double));
  if(status==cudaSuccess) status=cudaMemcpy(d,inputs,n*sizeof(Input),cudaMemcpyHostToDevice);
  if(status==cudaSuccess) {
    values<<<(n+127)/128,128>>>(d,n,o);
    status=cudaGetLastError();
  }
  if(status==cudaSuccess) status=cudaMemcpy(output,o,n*sizeof(double),cudaMemcpyDeviceToHost);
  cudaFree(o);cudaFree(d);
  return status;
}
"""
    cpu = r"""
extern "C" int probe(const Input* inputs,std::size_t n,double* output) {
  for(std::size_t i=0;i<n;++i) output[i]=value(inputs[i]);
  return 0;
}
"""
    source.write_text(
        common.replace("QUALIFIER", "__device__" if cuda else "")
        + (gpu if cuda else cpu)
    )
    obj, library = directory / "probe.o", directory / "probe.so"
    flags = ["-arch=sm_120", "-Xcompiler=-fPIC"] if cuda else ["-fPIC"]
    subprocess.run(
        [
            cache,
            compiler,
            "-O2",
            "-std=c++17",
            *flags,
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        env={**os.environ, "CCACHE_BASEDIR": str(ROOT)},
        check=True,
        capture_output=True,
        timeout=300,
    )
    subprocess.run(
        [compiler, "-shared", str(obj), "-o", str(library)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    dll = ct.CDLL(str(library))
    dll.probe.argtypes = [ct.c_void_p, ct.c_size_t, ct.c_void_p]
    dll.probe.restype = ct.c_int

    def evaluate(records: np.ndarray) -> np.ndarray:
        output = np.full(len(records), np.nan)
        assert dll.probe(records.ctypes.data, len(records), output.ctypes.data) == 0
        return output

    return evaluate


def test_axis_value_domain_does_not_promote_derivatives() -> None:
    assert len(axis_polynomial(3, 3, 4, auxiliary_g=True)[1]) == 11
    assert len(axis_polynomial(4, 0, 4, auxiliary_g=True)[1]) == 9
    for powers in ((3, 3, 4), (4, 0, 4)):
        with pytest.raises(ValueError):
            axis_polynomial(*powers)
    for powers in ((4, 1, 0), (0, 4, 0), (0, 0, 5)):
        with pytest.raises(ValueError):
            axis_polynomial(*powers, auxiliary_g=True)


@pytest.mark.parametrize("angular", SIGNATURES)
@pytest.mark.parametrize("variant", ["coincident", "asymmetric"])
def test_all_added_cartesian_and_spherical_blocks(
    value_probe: typing.Any,
    angular: tuple[int, ...],
    variant: str,
) -> None:
    pytest.importorskip("pyscf")
    fixture = make_df_value_fixture(angular, variant=variant)
    actual = fixture.contract(value_probe(fixture.records))
    np.testing.assert_allclose(actual, fixture.reference, atol=3e-11, rtol=3e-11)
    np.testing.assert_allclose(
        fixture.spherical(actual), fixture.spherical_reference, atol=3e-11, rtol=3e-11
    )


@pytest.mark.parametrize("angular", [(4, 4), (3, 3, 4)])
@pytest.mark.parametrize(
    "exponent_scale,separation_scale", [(0.02, 5), (20, 1), (1, 10)]
)
def test_diffuse_tight_and_large_boys_argument(
    value_probe: typing.Any,
    angular: tuple[int, ...],
    exponent_scale: float,
    separation_scale: float,
) -> None:
    """Signed contractions and all components cover F10 and large-T cancellation."""
    pytest.importorskip("pyscf")
    fixture = make_df_value_fixture(
        angular, exponent_scale=exponent_scale, separation_scale=separation_scale
    )
    actual = fixture.contract(value_probe(fixture.records))
    np.testing.assert_allclose(actual, fixture.reference, atol=3e-11, rtol=3e-11)
    np.testing.assert_allclose(
        fixture.spherical(actual), fixture.spherical_reference, atol=3e-11, rtol=3e-11
    )


def test_unsupported_orbital_g_and_auxiliary_h_fail_closed(
    value_probe: typing.Any,
) -> None:
    pytest.importorskip("pyscf")
    records = make_df_value_fixture((0, 0, 0)).records.copy()
    records["angular"][:, 0, 0] = 4
    assert np.isnan(value_probe(records)).all()
    records["angular"][:, 0, 0] = 0
    records["angular"][:, 2, 0] = 5
    assert np.isnan(value_probe(records)).all()
