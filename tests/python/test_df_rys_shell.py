"""Independent scientific gates for the first generated Rys shell slice."""

import ctypes
import math
import shutil
import subprocess

import numpy as np
import pytest
from vibeqc_compiler.integral.df_derivatives_cuda import emit_df_derivatives_cuda
from vibeqc_compiler.integral.df_rys import emit_df_rys_cuda
from vibeqc_compiler.integral.df_rys_shell import (
    emit_df_rys_policy_cpp,
    emit_df_rys_shell_cuda,
)
from vibeqc_compiler.integral.df_shell_derivatives import emit_df_shell_derivatives_cuda

from tools.vibeqc_validation.df_derivatives import make_df_derivative_fixture


@pytest.fixture(scope="module")
def sss_evaluator(tmp_path_factory):
    """Compile the emitted primitive arithmetic; no native runtime or GPU probe."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    directory = tmp_path_factory.mktemp("df_rys_shell")
    (directory / "cuda_runtime.h").write_text("")
    for name, emit in (
        ("generated_df_derivatives.cuh", emit_df_derivatives_cuda),
        ("generated_df_shell_derivatives.cuh", emit_df_shell_derivatives_cuda),
        ("generated_df_rys.cuh", emit_df_rys_cuda),
        ("generated_df_rys_shell.cuh", emit_df_rys_shell_cuda),
        ("generated_df_rys_policy.hpp", emit_df_rys_policy_cpp),
    ):
        (directory / name).write_text(emit())
    source = directory / "probe.cpp"
    source.write_text(r"""
#define __device__
#define __forceinline__ inline
#define __noinline__ __attribute__((noinline))
#include "generated_df_rys_shell.cuh"
using namespace vibeqc::scf;
extern "C" void probe(bool rys,const double* e,const double* r,double* out) {
  generated_df_derivatives::Geometry g;
  const generated_df_derivatives::Vec3 a{r[0],r[1],r[2]}, b{r[3],r[4],r[5]}, c{r[6],r[7],r[8]};
  double independent[6]{};
  if(rys) {
    using Math=generated_df_shell::RysShell<0,0,0>;
    generated_df_derivatives::prepare_geometry_rys<1>(e[0],a,e[1],b,e[2],c,0,g);
    Math::accumulate(0,e[0],e[1],g,nullptr,1.0,independent);
  } else {
    using Math=generated_df_shell::Shell<0,0,0>;
    generated_df_derivatives::prepare_geometry(e[0],a,e[1],b,e[2],c,0,g);
    double cache[3*Math::axis_size];
    for(unsigned lane=0;lane<3;++lane) Math::prepare(g,cache,lane,3);
    Math::accumulate(0,e[0],e[1],g,cache,1.0,independent);
  }
  const auto value=generated_df_shell::Shell<0,0,0>::finish(independent);
  for(unsigned center=0;center<3;++center)
    for(unsigned axis=0;axis<3;++axis) out[3*center+axis]=value.gradient[center][axis];
}
""")
    output = directory / "probe.so"
    subprocess.run(
        [
            compiler,
            "-O2",
            "-std=c++20",
            "-shared",
            "-fPIC",
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
    library.probe.argtypes = [ctypes.c_bool, double, double, double]
    library.probe.restype = None

    def evaluate(exponents, centers, *, rys=True):
        exponents = np.ascontiguousarray(exponents, dtype=np.float64)
        centers = np.ascontiguousarray(centers, dtype=np.float64)
        out = np.empty((3, 3))
        library.probe(
            rys,
            exponents.ctypes.data_as(double),
            centers.ctypes.data_as(double),
            out.ctypes.data_as(double),
        )
        return out

    return evaluate


@pytest.mark.parametrize("variant", ("asymmetric", "coincident"))
@pytest.mark.parametrize("lengths", ((2, 1, 2), (3, 2, 5)))
def test_rys_sss_matches_independent_libcint_contractions(
    sss_evaluator, variant, lengths
):
    pytest.importorskip("pyscf")
    fixture = make_df_derivative_fixture(
        (0, 0, 0), variant=variant, primitive_lengths=lengths
    )
    output = np.array(
        [sss_evaluator(r["exponents"], r["centers"]) for r in fixture.records]
    )
    output *= fixture.records["weight"][:, None, None]
    actual = fixture.contract(output)
    np.testing.assert_allclose(actual, fixture.reference, atol=8e-11, rtol=8e-11)
    np.testing.assert_allclose(
        fixture.spherical(actual), fixture.spherical_reference, atol=8e-11, rtol=8e-11
    )


def test_rys_sss_retains_polynomial_force_contract_across_argument_branches(
    sss_evaluator,
):
    # Coincident orbital centers keep pair decay finite while C independently
    # sweeps small-argument cancellation and the large-argument asymptotic regime.
    exponents = (0.7, 1.3, 0.9)
    rho = sum(exponents[:2]) * exponents[2] / sum(exponents)
    arguments = [
        0.0,
        0.5,
        np.nextafter(0.5, 0),
        np.nextafter(0.5, 1),
        30.0,
        60.0,
        1e300,
    ]
    arguments.extend(np.geomspace(1e-16, 1e8, 120))
    for argument in arguments:
        centers = ((0, 0, 0), (0, 0, 0), (math.sqrt(argument / rho), 0, 0))
        actual = sss_evaluator(exponents, centers)
        expected = sss_evaluator(exponents, centers, rys=False)
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, atol=8e-11, rtol=8e-11)
        np.testing.assert_allclose(actual.sum(axis=0), 0, atol=8e-11, rtol=0)
