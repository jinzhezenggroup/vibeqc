"""The generated force bound must dominate independent weighted derivatives."""

import ctypes
import math
import subprocess
import typing

import numpy as np
import pytest
from vibeqc_compiler.integral.df_derivatives_cuda import emit_df_derivatives_cuda
from vibeqc_compiler.integral.df_screening import emit_sss_force_screening_cuda

from tools.generate_validation_references import pyscf_molecule
from tools.vibeqc_validation.f_shell_numerics import _normalized_primitives


@pytest.fixture(scope="module")
def bound(tmp_path_factory: typing.Any) -> typing.Any:
    path = tmp_path_factory.mktemp("screen_bound")
    (path / "cuda_runtime.h").write_text("")
    (path / "generated_df_derivatives.cuh").write_text(emit_df_derivatives_cuda())
    (path / "screen.cuh").write_text(emit_sss_force_screening_cuda())
    source = path / "probe.cpp"
    source.write_text(r"""
#define __device__
#define __forceinline__ inline
#define __noinline__ __attribute__((noinline))
#include "screen.cuh"
extern "C" double bound(const double* e,const double* r,double weight) {
  using namespace vibeqc::scf::generated_df_derivatives;
  return sss_force_bound(e[0],{r[0],r[1],r[2]},e[1],{r[3],r[4],r[5]},
                        e[2],{r[6],r[7],r[8]},weight);
}
""")
    subprocess.run(
        [
            "c++",
            "-O2",
            "-shared",
            "-fPIC",
            "-std=c++20",
            str(source),
            "-I",
            str(path),
            "-o",
            str(path / "probe.so"),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=True,
    )
    library = ctypes.CDLL(str(path / "probe.so"))
    double = ctypes.POINTER(ctypes.c_double)
    library.bound.argtypes = [double, double, ctypes.c_double]
    library.bound.restype = ctypes.c_double

    def evaluate(
        exponents: typing.Any, coordinates: typing.Any, weight: typing.Any
    ) -> typing.Any:
        e, r = np.ascontiguousarray(exponents), np.ascontiguousarray(coordinates)
        return library.bound(e.ctypes.data_as(double), r.ctypes.data_as(double), weight)

    return evaluate


def independent(
    exponents: typing.Any, coordinates: typing.Any, weight: typing.Any
) -> typing.Any:
    """Direct libcint derivatives and normalization, without the candidate IR."""
    inputs = {
        "name": "screening-oracle",
        "atomic_numbers": [1] * 3,
        "coordinates": coordinates.tolist(),
        "basis_representation": "cartesian",
        "charge": 0,
        "multiplicity": 2,
        "shells": [
            {"atom_index": i, "angular_momentum": 0, "primitives": [[float(e), 1.0]]}
            for i, e in enumerate(exponents)
        ],
    }
    mol, scales, _ = pyscf_molecule(inputs)
    normalize = math.prod(scales)
    derivatives = (
        np.array(
            [
                -mol.intor_by_shell("int3c2e_ip1_cart", (0, 1, 2), comp=3),
                -mol.intor_by_shell("int3c2e_ip1_cart", (1, 0, 2), comp=3),
                -mol.intor_by_shell("int3c2e_ip2_cart", (0, 1, 2), comp=3),
            ]
        ).reshape(3, 3)
        * normalize
        * weight
    )
    value = (
        float(mol.intor_by_shell("int3c2e_cart", (0, 1, 2)).item()) * normalize * weight
    )
    primitive_weight = weight * math.prod(
        _normalized_primitives(s)[0][1] for s in inputs["shells"]
    )
    return value, derivatives, primitive_weight


def test_bound_covers_signs_geometry_exponents_and_translation(
    bound: typing.Any,
) -> None:
    pytest.importorskip("pyscf")
    rng = np.random.default_rng(406)
    for _ in range(40):
        exponents = 10.0 ** rng.uniform(-2, 3, 3)
        coordinates = rng.normal(size=(3, 3)) * 10 ** rng.uniform(-2, 1)
        weight = rng.normal()
        _, derivatives, primitive_weight = independent(exponents, coordinates, weight)
        actual_bound = bound(exponents, coordinates, primitive_weight)
        assert np.max(np.abs(derivatives)) <= actual_bound + 1e-13
        np.testing.assert_allclose(derivatives.sum(axis=0), 0, atol=1e-10)


def test_small_value_large_force_is_not_screened(bound: typing.Any) -> None:
    pytest.importorskip("pyscf")
    exponents = np.array([1e10, 2e10, 1.5e10])
    coordinates = np.array([[0.0, 0.0, 0.0], [1e-6, 2e-6, 0.0], [2e-6, 0.0, -1e-6]])
    value, derivatives, primitive_weight = independent(exponents, coordinates, 1e-8)
    assert abs(value) < 1e-8
    assert np.max(np.abs(derivatives)) > 1e-6
    assert bound(exponents, coordinates, primitive_weight) > 1e-8
