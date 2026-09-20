"""Independent scientific gates for component and cooperative DF Rys shells."""

import ctypes
import math
import shutil
import subprocess
import typing

import numpy as np
import pytest
from vibeqc_compiler.integral.df_derivatives_cuda import emit_df_derivatives_cuda
from vibeqc_compiler.integral.df_rys import emit_df_rys_cuda
from vibeqc_compiler.integral.df_rys_shell import (
    AUXILIARY_F_RYS_SHELL_CLASSES,
    COMPONENT_RYS_SHELL_CLASSES,
    RYS_SHELL_CLASSES,
    emit_df_rys_policy_cpp,
    emit_df_rys_shell_cuda,
    shell_rys_roots,
)
from vibeqc_compiler.integral.df_shell_derivatives import emit_df_shell_derivatives_cuda

from tools.vibeqc_validation.df_derivatives import make_df_derivative_fixture


@pytest.fixture(scope="module")
def shell_evaluator(tmp_path_factory: typing.Any) -> typing.Any:
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
    source.write_text(
        r"""
#define __device__
#define __forceinline__ inline
#define __noinline__ __attribute__((noinline))
#include "generated_df_rys_shell.cuh"
using namespace vibeqc::scf;
template<unsigned A,unsigned B,unsigned C>
void evaluate(bool rys,unsigned item,const double* e,const double* r,double* out) {
  generated_df_derivatives::Geometry g;
  const generated_df_derivatives::Vec3 a{r[0],r[1],r[2]}, b{r[3],r[4],r[5]}, c{r[6],r[7],r[8]};
  double independent[6]{};
  if(rys) {
    using Math=generated_df_shell::RysShell<A,B,C>;
    generated_df_derivatives::prepare_geometry_rys<Math::nroots>(e[0],a,e[1],b,e[2],c,A+B+C,g);
    double cache[3*Math::axis_size];
    for(unsigned lane=0;lane<32;++lane) Math::prepare(g,cache,lane,32);
    Math::accumulate(item,e[0],e[1],g,cache,1.0,independent);
  } else {
    using Math=generated_df_shell::Shell<A,B,C>;
    generated_df_derivatives::prepare_geometry(e[0],a,e[1],b,e[2],c,A+B+C,g);
    double cache[3*Math::axis_size];
    for(unsigned lane=0;lane<3;++lane) Math::prepare(g,cache,lane,3);
    Math::accumulate(item,e[0],e[1],g,cache,1.0,independent);
  }
  const auto value=generated_df_shell::Shell<A,B,C>::finish(independent);
  for(unsigned center=0;center<3;++center)
    for(unsigned axis=0;axis<3;++axis) out[3*center+axis]=value.gradient[center][axis];
}
extern "C" void probe(unsigned cls,bool rys,unsigned item,const double* e,const double* r,double* out) {
  switch(cls) {
__SHELL_CASES__
  }
}
""".replace(
            "__SHELL_CASES__",
            "\n".join(
                f"    case {100 * a + 10 * b + c}: return evaluate<{a},{b},{c}>(rys,item,e,r,out);"
                for a, b, c in RYS_SHELL_CLASSES
            ),
        )
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
    library.probe.argtypes = [
        ctypes.c_uint,
        ctypes.c_bool,
        ctypes.c_uint,
        double,
        double,
        double,
    ]
    library.probe.restype = None

    def evaluate(
        exponents: typing.Any,
        centers: typing.Any,
        angular: typing.Any = ((0, 0, 0),) * 3,
        *,
        rys: typing.Any = True,
    ) -> typing.Any:
        exponents = np.ascontiguousarray(exponents, dtype=np.float64)
        centers = np.ascontiguousarray(centers, dtype=np.float64)
        out = np.empty((3, 3))
        degrees = [sum(powers) for powers in angular]
        item = 0
        for powers, degree in zip(angular, degrees, strict=True):
            row = int(powers[1] + powers[2])
            item = (
                item * ((degree + 1) * (degree + 2) // 2)
                + row * (row + 1) // 2
                + powers[2]
            )
        library.probe(
            int(100 * degrees[0] + 10 * degrees[1] + degrees[2]),
            rys,
            int(item),
            exponents.ctypes.data_as(double),
            centers.ctypes.data_as(double),
            out.ctypes.data_as(double),
        )
        return out

    return evaluate


@pytest.mark.parametrize("angular", RYS_SHELL_CLASSES)
@pytest.mark.parametrize("variant", ("asymmetric", "coincident"))
@pytest.mark.parametrize("lengths", ((2, 1, 2), (3, 2, 5)))
def test_rys_shell_matches_independent_libcint_contractions(
    shell_evaluator: typing.Any,
    angular: typing.Any,
    variant: typing.Any,
    lengths: typing.Any,
) -> None:
    pytest.importorskip("pyscf")
    fixture = make_df_derivative_fixture(
        angular, variant=variant, primitive_lengths=lengths
    )
    output = np.array(
        [
            shell_evaluator(r["exponents"], r["centers"], r["angular"])
            for r in fixture.records
        ]
    )
    output *= fixture.records["weight"][:, None, None]
    actual = fixture.contract(output)
    np.testing.assert_allclose(actual, fixture.reference, atol=8e-11, rtol=8e-11)
    np.testing.assert_allclose(
        fixture.spherical(actual), fixture.spherical_reference, atol=8e-11, rtol=8e-11
    )


@pytest.mark.parametrize("angular", RYS_SHELL_CLASSES)
@pytest.mark.parametrize(
    "exponents", ((0.7, 1.3, 0.9), (0.03, 7.1, 0.002), (40, 0.04, 15))
)
def test_rys_retains_polynomial_force_contract_across_argument_branches(
    shell_evaluator: typing.Any, angular: typing.Any, exponents: typing.Any
) -> None:
    # Coincident orbital centers keep pair decay finite while C independently
    # sweeps small-argument cancellation and the large-argument asymptotic regime.
    rho = sum(exponents[:2]) * exponents[2] / sum(exponents)
    arguments = [
        0.0,
        0.5,
        np.nextafter(0.5, 0),
        np.nextafter(0.5, 1),
        30.0,
        60.0,
    ]
    arguments.extend(np.geomspace(1e-16, 1e8, 40))
    for boundary in (3e-7, 48.0, 50.0, 55.0):
        arguments.extend(
            [np.nextafter(boundary, 0), boundary, np.nextafter(boundary, math.inf)]
        )
    powers = tuple((l, 0, 0) for l in angular)
    for argument in arguments:
        centers = ((0, 0, 0), (0, 0, 0), (math.sqrt(argument / rho), 0, 0))
        actual = shell_evaluator(exponents, centers, powers)
        expected = shell_evaluator(exponents, centers, powers, rys=False)
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(actual, expected, atol=8e-11, rtol=8e-11)
        np.testing.assert_allclose(actual.sum(axis=0), 0, atol=8e-11, rtol=0)


@pytest.mark.parametrize("angular", RYS_SHELL_CLASSES)
@pytest.mark.parametrize("argument", (0.5, 48.0, 1e300))
def test_independent_high_precision_center_differentiation(
    shell_evaluator: typing.Any, angular: typing.Any, argument: typing.Any
) -> None:
    """Differentiate the closed SSS integral; no generated DAG or Rys oracle.

    Cartesian p/d/f functions are center derivatives of an s Gaussian (d/f
    include their lower-order corrections). High precision resolves the tiny
    coordinate perturbations at T=1e300, where the polynomial FP64 reference
    itself overflows. Relative checks keep underflow-sized errors visible.
    """
    from itertools import product

    mp = pytest.importorskip("mpmath")
    exponents = (0.7, 1.3, 0.9)
    rho = sum(exponents[:2]) * exponents[2] / sum(exponents)
    positions = (0.0, 0.0, math.sqrt(argument / rho))
    powers = tuple((l, 0, 0) for l in angular)
    centers = tuple((x, 0, 0) for x in positions)
    actual = shell_evaluator(exponents, centers, powers)
    with mp.workdps(400 if argument > 1e100 else 90):
        alpha, beta, gamma = map(mp.mpf, exponents)
        p, q = alpha + beta, gamma
        rho_mp = p * q / (p + q)

        def sss(a: typing.Any, b: typing.Any, c: typing.Any) -> typing.Any:
            t = rho_mp * ((alpha * a + beta * b) / p - c) ** 2
            f0 = (
                mp.mpf(1)
                if not t
                else mp.sqrt(mp.pi) * mp.erf(mp.sqrt(t)) / (2 * mp.sqrt(t))
            )
            return (
                2
                * mp.pi ** mp.mpf("2.5")
                / (p * q * mp.sqrt(p + q))
                * mp.exp(-alpha * beta / p * (a - b) ** 2)
                * f0
            )

        operators = []
        for degree, exponent in zip(angular, (alpha, beta, gamma), strict=True):
            if degree == 0:
                operators.append(((0, mp.mpf(1)),))
            elif degree == 1:
                operators.append(((1, 1 / (2 * exponent)),))
            elif degree == 2:
                operators.append(((2, 1 / (4 * exponent**2)), (0, 1 / (2 * exponent))))
            elif degree == 3:
                operators.append(
                    ((3, 1 / (8 * exponent**3)), (1, 3 / (4 * exponent**2)))
                )
            else:
                raise AssertionError("independent Gaussian operator is not defined")
        expected = np.zeros((3, 3))
        point = tuple(map(mp.mpf, positions))
        for center in range(3):
            value = mp.mpf(0)
            for terms in product(*operators):
                orders = [term[0] for term in terms]
                orders[center] += 1
                value += mp.fprod(term[1] for term in terms) * mp.diff(
                    sss, point, tuple(orders)
                )
            expected[center, 0] = float(value)
    if angular in COMPONENT_RYS_SHELL_CLASSES:
        # Keep every previously qualified low-angular gate unchanged.
        np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=1e-323)
    else:
        np.testing.assert_allclose(actual[:2], expected[:2], rtol=5e-13, atol=1e-323)
        # A recovered C derivative can underflow while opposite A/B values
        # remain normal (211/212 at T=1e300). Its error must then be bounded
        # by the independently measured A/B errors plus one FP64 addition's
        # rounding, rather than a relative error against rounded zero. This
        # does not relax either independent derivative or any endpoint gate.
        propagated = np.abs(actual[:2] - expected[:2]).sum(axis=0)
        rounding = 2 * np.finfo(float).eps * np.abs(actual[:2]).sum(axis=0)
        assert np.all(np.abs(actual[2] - expected[2]) <= propagated + rounding + 1e-323)


def test_auxiliary_f_capability_does_not_admit_unqualified_five_root_math() -> None:
    assert AUXILIARY_F_RYS_SHELL_CLASSES == (
        (0, 0, 3),
        (1, 0, 3),
        (1, 1, 3),
        (2, 0, 3),
        (2, 1, 3),
    )
    assert [shell_rys_roots(c) for c in AUXILIARY_F_RYS_SHELL_CLASSES] == [
        3,
        3,
        4,
        4,
        4,
    ]
    for unsupported in ((2, 2, 3), (3, 0, 0), (3, 3, 3)):
        with pytest.raises(ValueError, match="not generated"):
            shell_rys_roots(unsupported)
