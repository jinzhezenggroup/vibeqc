"""Independent scientific checks of the emitted shell-shared moment consumer."""

import ctypes
import shutil
import subprocess
import typing

import numpy as np
import pytest
from vibeqc_compiler.integral.df_derivatives_cuda import emit_df_derivatives_cuda
from vibeqc_compiler.integral.df_shell_derivatives import (
    SHELL_CLASSES,
    emit_df_shell_derivatives_cuda,
    shell_work_model,
)

from tools.vibeqc_validation.df_derivatives import make_df_derivative_fixture


@pytest.fixture(scope="module")
def shell_library(tmp_path_factory: typing.Any) -> typing.Any:
    """Compile the exact generated arithmetic without requiring a GPU runtime."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    directory = tmp_path_factory.mktemp("df_shell_emission")
    (directory / "cuda_runtime.h").write_text("")
    # Count executed generated operations in the host fixture independently of
    # the compiler's analytical work model. This instrumentation is test-only.
    scalar_source = emit_df_derivatives_cuda().replace(
        "switch(a*20U+b*4U+c)", "++test_polynomial_calls; switch(a*20U+b*4U+c)"
    )
    shell_source = (
        emit_df_shell_derivatives_cuda()
        .replace(
            "other_moment+=v[j]*w[k]*g.f[i+j+k];",
            "other_moment+=(++test_convolution_iterations,v[j]*w[k]*g.f[i+j+k]);",
        )
        .replace(
            "if(lane<3) prepare_axis(g,lane,cache+lane*axis_size);",
            "if(lane<3) {++test_specialized_calls; prepare_axis(g,lane,cache+lane*axis_size);}",
        )
    )
    (directory / "generated_df_derivatives.cuh").write_text(scalar_source)
    (directory / "generated_df_shell_derivatives.cuh").write_text(shell_source)
    source = directory / "fixture.cpp"
    source.write_text(r"""
#define __device__
#define __forceinline__ inline
#define __noinline__ __attribute__((noinline))
static unsigned long long test_polynomial_calls=0,test_convolution_iterations=0,
                          test_specialized_calls=0;
#include "generated_df_shell_derivatives.cuh"
using namespace vibeqc::scf;
extern "C" void boys_probe(unsigned order,double argument,double* plain,double* observed,
                           unsigned* work) {
  generated_df_derivatives::BoysWork counts;
  generated_df_derivatives::boys_values(order,argument,plain);
  generated_df_derivatives::boys_values(order,argument,observed,&counts);
  work[0]=counts.series_iterations;work[1]=counts.series;
  work[2]=counts.small_argument;work[3]=counts.large_argument;
}
extern "C" void shell_work_counts(unsigned code,unsigned item,unsigned long long* work) {
  generated_df_shell::for_each_class([&]<unsigned A,unsigned B,unsigned C>() {
    if(code!=A*16+B*4+C) return;
    using Math=generated_df_shell::Shell<A,B,C>;
    generated_df_derivatives::Geometry g;
    generated_df_derivatives::prepare_geometry(1.0,{0.1,0.2,0.3},1.2,{0.5,-0.2,0.1},
                                               0.7,{-0.5,0.3,0.8},A+B+C,g);
    double cache[3*Math::axis_size];
    for(auto& value:cache) value=NAN;
    test_polynomial_calls=test_convolution_iterations=test_specialized_calls=0;
    for(unsigned lane=0;lane<32;++lane) Math::prepare(g,cache,lane,32);
    double independent[6]{};
    Math::accumulate(item,1.0,1.2,g,cache,1.0,independent);
    work[0]=test_polynomial_calls;work[1]=test_specialized_calls;
    work[2]=0;for(auto value:cache) work[2]+=std::isfinite(value);
    work[3]=test_convolution_iterations;work[4]=Math::convolution_work(item);
  });
}
extern "C" void shell_derivative(unsigned code,const double* e,const double* r,
                                const unsigned char* powers,double* out) {
  generated_df_shell::for_each_class([&]<unsigned A,unsigned B,unsigned C>() {
    if(code!=A*16+B*4+C) return;
    using Math=generated_df_shell::Shell<A,B,C>;
    generated_df_derivatives::Geometry g;
    generated_df_derivatives::prepare_geometry(e[0],{r[0],r[1],r[2]},e[1],{r[3],r[4],r[5]},
                                               e[2],{r[6],r[7],r[8]},A+B+C,g);
    double cache[3*Math::axis_size];
    for(unsigned lane=0;lane<32;++lane) Math::prepare(g,cache,lane,32);
    const unsigned i=generated_df_shell::cartesian_index(A,powers);
    const unsigned j=generated_df_shell::cartesian_index(B,powers+3);
    const unsigned p=generated_df_shell::cartesian_index(C,powers+6);
    double independent[6]{};
    Math::accumulate((i*Math::nb+j)*Math::nc+p,e[0],e[1],g,cache,1.0,independent);
    auto result=Math::finish(independent);
    for(unsigned center=0;center<3;++center)
      for(unsigned axis=0;axis<3;++axis) out[3*center+axis]=result.gradient[center][axis];
  });
}
""")
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
            str(directory / "fixture.so"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=240,
    )
    library = ctypes.CDLL(str(directory / "fixture.so"))
    double = ctypes.POINTER(ctypes.c_double)
    library.shell_derivative.argtypes = [
        ctypes.c_uint,
        double,
        double,
        ctypes.POINTER(ctypes.c_ubyte),
        double,
    ]
    library.shell_derivative.restype = None
    library.boys_probe.argtypes = [
        ctypes.c_uint,
        ctypes.c_double,
        double,
        double,
        ctypes.POINTER(ctypes.c_uint),
    ]
    library.boys_probe.restype = None
    library.shell_work_counts.argtypes = [
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_ulonglong),
    ]
    library.shell_work_counts.restype = None
    return library


@pytest.mark.parametrize("order", range(11))
def test_boys_diagnostics_preserve_values(
    shell_library: typing.Any, order: typing.Any
) -> None:
    """Observed branch/iteration counts leave the independently checked values intact."""
    special = pytest.importorskip("scipy.special")
    rng = np.random.default_rng(395)
    arguments = np.concatenate(
        (
            [0.0, 1e-12, 1e-8, np.nextafter(30.0, 0.0), 30.0],
            np.geomspace(1e-16, 1e5, 80),
            10 ** rng.uniform(-14, 5, 80),
        )
    )
    plain, observed = np.empty(11), np.empty(11)
    counts = np.empty(4, dtype=np.uint32)
    double = ctypes.POINTER(ctypes.c_double)
    for argument in arguments:
        shell_library.boys_probe(
            order,
            argument,
            plain.ctypes.data_as(double),
            observed.ctypes.data_as(double),
            counts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
        )
        np.testing.assert_array_equal(plain[: order + 1], observed[: order + 1])
        orders = np.arange(order + 1)
        reference = special.hyp1f1(orders + 0.5, orders + 1.5, -argument) / (
            2 * orders + 1
        )
        np.testing.assert_allclose(
            observed[: order + 1], reference, rtol=4e-13, atol=1e-300
        )
        assert tuple(counts[1:]) == (argument < 30, argument < 1e-8, argument >= 30)
        assert 1 <= counts[0] <= 179 if argument < 30 else counts[0] == 0
        if argument == 0:
            assert counts[0] == 1


@pytest.mark.parametrize("angular", SHELL_CLASSES)
def test_shell_work_model_matches_executed_generated_loops(
    shell_library: typing.Any, angular: typing.Any
) -> None:
    """Instrumented emitted C++ protects the ledger from stale analytical formulas."""
    model = shell_work_model(angular)
    loops = model["component_convolution_iterations"]
    observed = np.empty(5, dtype=np.uint64)
    for item in sorted({0, len(loops) // 2, len(loops) - 1}):
        shell_library.shell_work_counts(
            16 * angular[0] + 4 * angular[1] + angular[2],
            item,
            observed.ctypes.data_as(ctypes.POINTER(ctypes.c_ulonglong)),
        )
        assert tuple(observed) == (
            model["axis_polynomial_calls"],
            model["specialized_prepare_axis_calls"],
            model["cache_coefficient_values"],
            loops[item],
            loops[item],
        )


@pytest.mark.parametrize("angular", SHELL_CLASSES)
@pytest.mark.parametrize("variant", ["asymmetric", "coincident"])
def test_shell_moments_match_independent_contracted_blocks(
    shell_library: typing.Any, angular: typing.Any, variant: typing.Any
) -> None:
    """Libcint checks normalization, Gaussian decay and all three center channels."""
    pytest.importorskip("pyscf")
    fixture = make_df_derivative_fixture(angular, variant=variant)
    output = np.empty((len(fixture.records), 3, 3))
    double = ctypes.POINTER(ctypes.c_double)
    for i, record in enumerate(fixture.records):
        exponents = np.ascontiguousarray(record["exponents"])
        centers = np.ascontiguousarray(record["centers"])
        powers = np.ascontiguousarray(record["angular"], dtype=np.uint8)
        shell_library.shell_derivative(
            angular[0] * 16 + angular[1] * 4 + angular[2],
            exponents.ctypes.data_as(double),
            centers.ctypes.data_as(double),
            powers.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            output[i].ctypes.data_as(double),
        )
    output *= fixture.records["weight"][:, None, None]
    actual = fixture.contract(output)
    np.testing.assert_allclose(actual, fixture.reference, atol=8e-11, rtol=8e-11)
    np.testing.assert_allclose(
        fixture.spherical(actual), fixture.spherical_reference, atol=8e-11, rtol=8e-11
    )
