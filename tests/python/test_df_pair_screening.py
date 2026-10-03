"""Spatial envelopes must dominate independent Coulomb values and derivatives."""

import ctypes as ct
import itertools
import math
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from generativeqc_compiler.integral.df_pair_screening import emit_df_pair_screening_cuda

Bound = Callable[[int, int, float, float, float, float, int], float]


@pytest.fixture(scope="module")
def bound(tmp_path_factory: pytest.TempPathFactory) -> Bound:
    """Compile the emitted production helper with ordinary host C++."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    folder = tmp_path_factory.mktemp("df-pair-bound")
    (folder / "bound.hpp").write_text(emit_df_pair_screening_cuda())
    source = folder / "bound.cpp"
    source.write_text(
        """#define GENERATIVEQC_DF_BOUND_DEVICE
#include "bound.hpp"
extern "C" double bound(int a,int b,double alpha,double beta,double distance,double weight,int derivative) {
  using namespace generativeqc::scf::generated_df_screening;
  return derivative ? radial_pair_derivative_norm(a,b,alpha,beta,distance,weight)
                    : radial_pair_norm(a,b,alpha,beta,distance,weight);
}
"""
    )
    command = [
        compiler,
        "-O2",
        "-std=c++20",
        "-shared",
        "-fPIC",
        str(source),
        "-o",
        str(folder / "bound.so"),
    ]
    if cache := shutil.which("ccache"):
        command.insert(0, cache)
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    library = ct.CDLL(str(folder / "bound.so"))
    library.bound.argtypes = [ct.c_int, ct.c_int] + [ct.c_double] * 4 + [ct.c_int]
    library.bound.restype = ct.c_double
    return library.bound


@pytest.mark.parametrize("angular", list(itertools.product(range(4), repeat=3)))
def test_contracted_values_and_shared_atom_forces(
    bound: Bound, angular: tuple[int, int, int]
) -> None:
    """All s/p/d/f components, signed contractions and independent libcint."""
    pytest.importorskip("pyscf")
    from tools.generate_validation_references import pyscf_molecule
    from tools.generativeqc_validation.f_shell_numerics import _normalized_primitives

    a, b, c = angular
    rng = np.random.default_rng(1784 + 16 * a + 4 * b + c)
    for _ in range(3):
        exponents = 10 ** rng.uniform(-2, 2, 3)
        positions = rng.normal(size=(3, 3)) * 10 ** rng.uniform(-1, 1)
        shells = [
            {
                "atom_index": i,
                "angular_momentum": l,
                "primitives": [
                    [float(exponents[i]), 0.83],
                    [float(exponents[i] * 2.7), -0.13],
                ],
            }
            for i, l in enumerate(angular)
        ]
        inputs = {
            "name": "pair-envelope",
            "atomic_numbers": [1] * 3,
            "coordinates": positions.tolist(),
            "basis_representation": "cartesian",
            "charge": 0,
            "multiplicity": 2,
            "shells": shells,
        }
        mol, scales, _ = pyscf_molecule(inputs)
        locations = mol.ao_loc_nr()
        norms = [scales[locations[i] : locations[i + 1]] for i in range(3)]
        normalization = np.einsum("i,j,k->ijk", *norms)
        with mol.with_integral_screen(1e-100):
            values = mol.intor_by_shell("int3c2e_cart", (0, 1, 2)) * normalization
            da = (
                -mol.intor_by_shell("int3c2e_ip1_cart", (0, 1, 2), comp=3)
                * normalization
            )
            db = (
                -mol.intor_by_shell("int3c2e_ip1_cart", (1, 0, 2), comp=3).transpose(
                    0, 2, 1, 3
                )
                * normalization
            )
            dc = (
                -mol.intor_by_shell("int3c2e_ip2_cart", (0, 1, 2), comp=3)
                * normalization
            )
            self_eri = mol.intor_by_shell("int2e_cart", (0, 1, 0, 1)) * np.einsum(
                "i,j,k,l->ijkl", norms[0], norms[1], norms[0], norms[1]
            )
        primitives = [_normalized_primitives(shell) for shell in shells]
        distance = float(np.linalg.norm(positions[0] - positions[1]))
        pair = sum(
            bound(a, b, ea, eb, distance, abs(ca * cb), 0)
            for ea, ca in primitives[0]
            for eb, cb in primitives[1]
        )
        derivative = sum(
            bound(a, b, ea, eb, distance, abs(ca * cb), 1)
            for ea, ca in primitives[0]
            for eb, cb in primitives[1]
        )
        auxiliary = sum(bound(c, 0, ec, 0, 0, abs(cc), 0) for ec, cc in primitives[2])
        weights = rng.normal(size=values.shape)
        allowance = derivative * auxiliary * np.abs(weights).sum()
        assert math.isfinite(allowance)
        # Every subset of coincident physical centers must remain bounded.
        for gradient in (da, db, dc, da + db, da + dc, db + dc):
            actual = np.max(np.abs(np.einsum("dijk,ijk->d", gradient, weights)))
            assert actual <= allowance * (1 + 1e-12) + 1e-250
        assert np.max(np.abs(values)) <= pair * auxiliary * (1 + 1e-12) + 1e-250
        size = len(norms[0]) * len(norms[1])
        reference_norm = np.sqrt(np.max(np.abs(np.diag(self_eri.reshape(size, size)))))
        assert reference_norm <= pair * (1 + 1e-12) + 1e-250


@pytest.mark.parametrize(
    "alpha,beta,distance,coefficient",
    [
        (0, 1, 1, 1),
        (-1, 1, 1, 1),
        (1, -1, 1, 1),
        (1, 1, -1, 1),
        (math.inf, 1, 1, 1),
        (1, math.nan, 1, 1),
        (1, 1, math.inf, 1),
        (1, 1, 1, math.nan),
        (1e-300, 1e-300, 1, 1),
    ],
)
def test_invalid_or_overflowing_envelopes_retain_work(
    bound: Bound, alpha: float, beta: float, distance: float, coefficient: float
) -> None:
    assert math.isinf(bound(3, 3, alpha, beta, distance, coefficient, 0))


def test_tiny_overlap_is_not_rounded_to_a_false_zero(bound: Bound) -> None:
    # Normalization is applied before exponentiation. More extreme attenuation
    # is deliberately floored upward so it cannot certify a spurious exact zero.
    actual = bound(0, 0, 1, 1, 40, 1e300, 0)
    assert 0 < actual < 1e-20
    assert bound(3, 3, 1, 1, 100, 1, 1) > 0
    assert math.isinf(bound(4, 3, 1, 1, 1, 1, 1))


def test_same_center_and_signed_coefficients(bound: Bound) -> None:
    for a, b in itertools.product(range(4), repeat=2):
        positive = bound(a, b, 0.3, 1.7, 0, 2, 0)
        assert math.isfinite(positive) and positive > 0
        assert bound(a, b, 0.3, 1.7, 0, -2, 0) == positive
        assert bound(a, b, 0.3, 1.7, 0, 0, 0) == 0


def test_device_math_names_and_negative_host_reference(
    bound: Bound, tmp_path: Path
) -> None:
    """Check exact device-branch name lookup without pretending to run CUDA."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    # CUDA provides device overloads in the global namespace. Preload the host
    # standard library, then forbid std references in the selected device arm.
    # Real target annotations/compilation are still checked by NVIDIA/CuMetal CI.
    prefix = """#include <cmath>
#include <cstddef>
using std::isfinite;
#pragma GCC poison std
#define __CUDA_ARCH__ 1200
#define GENERATIVEQC_DF_BOUND_DEVICE
"""
    emitted = emit_df_pair_screening_cuda()
    suffix = """
extern "C" double device_bound(int a,int b,double alpha,double beta,double distance,double weight,int derivative) {
  using namespace generativeqc::scf::generated_df_screening;
  return derivative ? radial_pair_derivative_norm(a,b,alpha,beta,distance,weight)
                    : radial_pair_norm(a,b,alpha,beta,distance,weight);
}
"""
    source = tmp_path / "device_lookup.cpp"
    source.write_text(prefix + emitted + suffix)
    library_path = tmp_path / "device_lookup.so"
    command = [
        compiler,
        "-O2",
        "-std=c++20",
        "-shared",
        "-fPIC",
        str(source),
        "-o",
        str(library_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    library = ct.CDLL(str(library_path))
    library.device_bound.argtypes = (
        [ct.c_int, ct.c_int] + [ct.c_double] * 4 + [ct.c_int]
    )
    library.device_bound.restype = ct.c_double
    for a, b, derivative in itertools.product(range(4), range(4), range(2)):
        for distance in (0.0, 1.7, 40.0):
            arguments = (a, b, 0.3, 1.7, distance, -2.0, derivative)
            assert library.device_bound(*arguments) == bound(*arguments)
    # The old host-only spelling must actually fail this device-name contract.
    assert "math::sqrt(pi)" in emitted
    source.write_text(
        prefix + emitted.replace("math::sqrt(pi)", "std::sqrt(pi)") + suffix
    )
    failed = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=60
    )
    assert failed.returncode != 0
    assert "poisoned" in failed.stderr and "std" in failed.stderr
