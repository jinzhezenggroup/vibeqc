"""Independent raw center/sign and emitted bounded-polynomial DF response gates."""

import ctypes
import shutil
import subprocess
from itertools import product

import numpy as np
import pytest
from vibeqc_compiler.integral.df_derivatives import (
    axis_polynomial,
    build_df_derivative_ir,
    build_df_derivative_kernel,
    evaluate_df_derivative,
)
from vibeqc_compiler.integral.df_values import (
    build_df_component_kernel,
    build_df_value_ir,
    evaluate_df_primitive,
)
from vibeqc_compiler.integral.shell_spec import cartesian_components

from tools.vibeqc_validation.df_derivatives import make_df_derivative_fixture

SIGNATURES = [
    angular for count in (2, 3) for angular in product(range(4), repeat=count)
]


@pytest.mark.parametrize("angular", SIGNATURES)
def test_physical_value_dag_derivatives_match_libcint(angular):
    pytest.importorskip("pyscf")
    count = len(angular)
    fixture = make_df_derivative_fixture(angular, primitive_lengths=(1,) * count)
    family = "coulomb_metric" if count == 2 else "three_center_eri"
    choices = list(product(*(cartesian_components(momentum) for momentum in angular)))
    middle = len(choices) // 2
    kernel = build_df_derivative_kernel(
        build_df_derivative_ir(family, angular), choices[middle]
    )
    record = fixture.records[middle]
    slots = (0, 2) if count == 2 else (0, 1, 2)
    actual = (
        np.array(
            evaluate_df_derivative(
                kernel, record["exponents"][list(slots)], record["centers"][list(slots)]
            )
        )
        * record["weight"]
    )
    expected = fixture.reference.reshape(count, 3, -1)[:, :, middle]
    np.testing.assert_allclose(actual, expected, atol=3e-11, rtol=3e-11)
    np.testing.assert_allclose(actual.sum(axis=0), 0, atol=2e-13)


@pytest.mark.parametrize("angular", [(3, 3), (1, 2, 3), (3, 3, 3)])
def test_independent_center_finite_differences_and_auxiliary_motion(angular):
    count = len(angular)
    family = "coulomb_metric" if count == 2 else "three_center_eri"
    components = tuple(cartesian_components(momentum)[-1] for momentum in angular)
    kernel = build_df_derivative_kernel(
        build_df_derivative_ir(family, angular), components
    )
    value = build_df_component_kernel(build_df_value_ir(family, angular), components)
    exponents = (0.5, 0.8, 1.2)[:count]
    centers = np.array([[0.2, -0.3, 0.7], [-0.1, 0.8, 0.3], [0.4, 0.2, -0.5]][:count])
    actual = np.array(evaluate_df_derivative(kernel, exponents, centers))
    assert np.max(np.abs(actual[-1])) > 1e-7
    for step in (2e-4, 5e-5):
        for center in range(count):
            for axis in range(3):
                plus, minus = centers.copy(), centers.copy()
                plus[center, axis] += step
                minus[center, axis] -= step
                fd = (
                    evaluate_df_primitive(value, exponents, plus)
                    - evaluate_df_primitive(value, exponents, minus)
                ) / (2 * step)
                assert actual[center, axis] == pytest.approx(fd, abs=5e-7, rel=5e-7)


def test_derivative_domain_layout_and_internal_moments():
    for angular in SIGNATURES:
        family = "coulomb_metric" if len(angular) == 2 else "three_center_eri"
        for weighted in (False, True):
            ir = build_df_derivative_ir(family, angular, weighted=weighted)
            assert ir.maximum_coulomb_order == sum(angular) + 1
            assert ir.derivative.independent_centers(ir.operator) == tuple(
                range(len(angular) - 1)
            )
    with pytest.raises(ValueError, match="s/p/d/f"):
        build_df_derivative_ir("three_center_eri", (3, 5, 0))
    for powers in ((4, 3, 3), (3, 4, 3)):
        _, coefficients = axis_polynomial(*powers)
        assert len(coefficients) == 11
    with pytest.raises(ValueError, match="s/p/d/f"):
        axis_polynomial(4, 4, 3)


@pytest.fixture(scope="module")
def emitted_library(tmp_path_factory):
    """Compile the actual bounded CUDA arithmetic as ordinary host C++."""
    from vibeqc_compiler.integral.df_derivatives_cuda import emit_df_derivatives_cuda

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    directory = tmp_path_factory.mktemp("df_derivative_emission")
    (directory / "cuda_runtime.h").write_text("")
    (directory / "df_derivatives.cuh").write_text(emit_df_derivatives_cuda())
    (directory / "fixture.cpp").write_text(r"""
#define __device__
#define __forceinline__ inline
#define __noinline__ __attribute__((noinline))
#include "df_derivatives.cuh"
using namespace vibeqc::scf::generated_df_derivatives;
extern "C" void derivative(unsigned count,const double* exponents,const double* centers,const unsigned* angular,double* out) {
  Vec3 A{centers[0],centers[1],centers[2]},B{centers[3],centers[4],centers[5]},C{centers[6],centers[7],centers[8]};
  Angular a{angular[0],angular[1],angular[2]},b{angular[3],angular[4],angular[5]},c{angular[6],angular[7],angular[8]};
  auto r=count==2?metric(exponents[0],A,a,exponents[2],C,c):three_center(exponents[0],A,a,exponents[1],B,b,exponents[2],C,c);
  out[0]=r.first.x;out[1]=r.first.y;out[2]=r.first.z;
  out[3]=r.second.x;out[4]=r.second.y;out[5]=r.second.z;
  out[6]=r.third.x;out[7]=r.third.y;out[8]=r.third.z;
}
""")
    subprocess.run(
        [
            compiler,
            "-O2",
            "-std=c++17",
            "-shared",
            "-fPIC",
            str(directory / "fixture.cpp"),
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
    library.derivative.argtypes = [
        ctypes.c_uint,
        double,
        double,
        ctypes.POINTER(ctypes.c_uint),
        double,
    ]
    library.derivative.restype = None
    return library


@pytest.mark.parametrize("variant", ["asymmetric", "coincident"])
def test_emitted_full_contracted_and_spherical_blocks(emitted_library, variant):
    pytest.importorskip("pyscf")
    double = ctypes.POINTER(ctypes.c_double)
    for angular in SIGNATURES:
        fixture = make_df_derivative_fixture(angular, variant=variant)
        output = np.empty((len(fixture.records), 3, 3))
        for i, record in enumerate(fixture.records):
            exponents = np.ascontiguousarray(record["exponents"])
            centers = np.ascontiguousarray(record["centers"])
            powers = np.ascontiguousarray(record["angular"])
            emitted_library.derivative(
                len(angular),
                exponents.ctypes.data_as(double),
                centers.ctypes.data_as(double),
                powers.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
                output[i].ctypes.data_as(double),
            )
        output *= fixture.records["weight"][:, None, None]
        if len(angular) == 2:
            output = output[:, [0, 2]]
        actual = fixture.contract(output)
        np.testing.assert_allclose(
            actual, fixture.reference, atol=8e-11, rtol=8e-11, err_msg=fixture.name
        )
        np.testing.assert_allclose(
            fixture.spherical(actual),
            fixture.spherical_reference,
            atol=8e-11,
            rtol=8e-11,
            err_msg=fixture.name,
        )
