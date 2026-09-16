"""Independent libcint gates for class-specialized value lowerings."""

import ctypes
import subprocess

import numpy as np
import pytest
from vibeqc_compiler.integral.df_cuda import emit_df_values_cuda
from vibeqc_compiler.integral.df_derivatives_cuda import emit_df_derivatives_cuda
from vibeqc_compiler.integral.df_value_candidates import (
    VALUE_CLASSES,
    emit_df_value_candidates_cuda,
)

from tools.vibeqc_validation.df_values import make_df_value_fixture


@pytest.fixture(scope="module")
def evaluator(tmp_path_factory):
    """Execute emitted FP64 arithmetic on the host without a GPU dependency."""
    path = tmp_path_factory.mktemp("value_candidates")
    (path / "cuda_runtime.h").write_text("")
    for name, emit in (
        ("df_values.cuh", emit_df_values_cuda),
        ("generated_df_derivatives.cuh", emit_df_derivatives_cuda),
        ("candidates.cuh", emit_df_value_candidates_cuda),
    ):
        (path / name).write_text(emit())
    source = path / "probe.cpp"
    source.write_text(r"""
#define __device__
#define __forceinline__ inline
#define __noinline__ __attribute__((noinline))
#include "candidates.cuh"
using namespace vibeqc::scf;
struct Input {
  unsigned count; generated_df::Angular angular[3];
  generated_df::Vec3 centers[3]; double exponents[3],weight;
};
static_assert(sizeof(Input)==144);
template<unsigned Math> double value(const Input& x) {
  return x.weight*generated_df_value_candidates::three_center<Math>(
      x.exponents[0],x.centers[0],x.angular[0],x.exponents[1],x.centers[1],
      x.angular[1],x.exponents[2],x.centers[2],x.angular[2]);
}
extern "C" void probe(unsigned math,const Input* in,double* out,unsigned count) {
  for(unsigned i=0;i<count;++i)
    out[i]=math==1?value<1>(in[i]):math==2?value<2>(in[i]):value<3>(in[i]);
}
""")
    subprocess.run(
        [
            "c++",
            "-O2",
            "-std=c++20",
            "-shared",
            "-fPIC",
            str(source),
            "-I",
            str(path),
            "-o",
            str(path / "probe.so"),
        ],
        capture_output=True,
        text=True,
        timeout=240,
        check=True,
    )
    library = ctypes.CDLL(str(path / "probe.so"))
    library.probe.argtypes = [
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    library.probe.restype = None

    def evaluate(math, fixture):
        out = np.empty(len(fixture.records))
        library.probe(math, fixture.records.ctypes.data, out.ctypes.data, len(out))
        return fixture.contract(out)

    return evaluate


@pytest.mark.parametrize("angular", VALUE_CLASSES + ((1, 1, 1), (2, 2, 2)))
@pytest.mark.parametrize("variant", ("asymmetric", "coincident"))
def test_value_lowerings_against_libcint(evaluator, angular, variant):
    """Signed contractions, every component, spherical projection and fallback."""
    pytest.importorskip("pyscf")
    fixture = make_df_value_fixture(
        angular, variant=variant, primitive_lengths=(3, 2, 5)
    )
    for math in (1, 2, 3):
        actual = evaluator(math, fixture)
        np.testing.assert_allclose(actual, fixture.reference, atol=8e-11, rtol=8e-11)
        np.testing.assert_allclose(
            fixture.spherical(actual),
            fixture.spherical_reference,
            atol=8e-11,
            rtol=8e-11,
        )
