"""Host-compiled force-only CUDA expressions against an independent scalar formula."""

import ctypes as ct
import math
import shutil
import subprocess

import numpy as np
import pytest
from scipy.special import hyp1f1
from vibeqc_compiler.integral.weighted_eri_cuda import emit_low_order_weighted_header


class Geometry(ct.Structure):
    _fields_ = [
        ("inverse_two_p", ct.c_double),
        ("inverse_two_q", ct.c_double),
        ("rho", ct.c_double),
        ("difference", ct.c_double * 3),
        ("shifts", (ct.c_double * 3) * 4),
        ("product_scales", ct.c_double * 4),
        ("decay", (ct.c_double * 3) * 4),
        ("prefactor", ct.c_double),
        ("boys", ct.c_double * 14),
    ]


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> ct.CDLL:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler required")
    root = tmp_path_factory.mktemp("force-only")
    source = emit_low_order_weighted_header(inline_single_use=True)
    # Only qualifiers/includes are adapted: scalar generated expressions are exact.
    source = source.replace("#include <cuda_runtime.h>", "")
    source = source.replace("__device__ __forceinline__", "inline")
    source += r"""
extern "C" void force_only(int p, const vibeqc::scf::generated_weighted_eri::Geometry* g,
                          const double* weights, double* out) {
  using namespace vibeqc::scf::generated_weighted_eri;
  const auto result = p ? psss_force(*g, weights) : ssss_force(*g, weights);
  for (int c=0; c<3; ++c) for (int k=0; k<3; ++k) out[3*c+k]=result.center[c][k];
}
"""
    (root / "test.cpp").write_text(source)
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            "-fPIC",
            "-shared",
            str(root / "test.cpp"),
            "-o",
            str(root / "test.so"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    library = ct.CDLL(str(root / "test.so"))
    library.force_only.argtypes = [
        ct.c_int,
        ct.POINTER(Geometry),
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
    ]
    library.force_only.restype = None
    return library


def primitive(
    p_shell: bool, centers: np.ndarray, weights: np.ndarray
) -> tuple[float, Geometry]:
    # Independent unnormalized Gaussian psss/ssss value; no compiler DAG or AD.
    exponents = np.array([0.6, 0.8, 1.1, 0.9])
    a, b, c, d = exponents
    p, q = a + b, c + d
    mu, nu, rho = a * b / p, c * d / q, p * q / (p + q)
    A, B, C, D = centers
    P, Q = (a * A + b * B) / p, (c * C + d * D) / q
    delta = P - Q
    T = rho * np.dot(delta, delta)
    boys = np.array([hyp1f1(m + 0.5, m + 1.5, -T) / (2 * m + 1) for m in range(3)])
    prefactor = 2 * math.pi**2.5 / (p * q * math.sqrt(p + q))
    prefactor *= math.exp(-mu * np.dot(A - B, A - B) - nu * np.dot(C - D, C - D))
    scalar = prefactor * (
        np.dot(weights, (P - A) * boys[0] - rho / p * delta * boys[1])
        if p_shell
        else weights[0] * boys[0]
    )
    geometry = Geometry()
    storage = np.ctypeslib.as_array(
        (ct.c_double * (ct.sizeof(Geometry) // 8)).from_buffer(geometry)
    )
    storage.fill(np.nan)
    geometry.rho, geometry.prefactor = rho, prefactor
    geometry.product_scales[:3] = [a / p, b / p, c / q]
    geometry.difference[:] = delta
    for center, values in enumerate(
        [-2 * mu * (A - B), 2 * mu * (A - B), -2 * nu * (C - D)]
    ):
        geometry.decay[center][:] = values
    geometry.boys[: (3 if p_shell else 2)] = boys[: (3 if p_shell else 2)]
    if p_shell:
        geometry.inverse_two_p = 0.5 / p
        geometry.shifts[0][:] = P - A
    return scalar, geometry


@pytest.mark.parametrize("p_shell", [False, True])
@pytest.mark.parametrize("coincident", [False, True])
def test_poisoned_unused_geometry_matches_independent_finite_difference(
    generated: ct.CDLL, p_shell: bool, coincident: bool
) -> None:
    centers = np.array(
        [
            [0.13, -0.31, 0.24],
            [-0.43, 0.27, 0.51],
            [0.68, -0.14, -0.22],
            [-0.21, 0.48, -0.63],
        ]
    )
    if coincident:
        centers[1] = centers[0]
    weights = np.array([0.7, -1.3, 0.2] if p_shell else [-0.7], dtype=np.float64)
    _, geometry = primitive(p_shell, centers, weights)
    actual = np.full((3, 3), np.nan)
    generated.force_only(
        int(p_shell),
        ct.byref(geometry),
        weights.ctypes.data_as(ct.POINTER(ct.c_double)),
        actual.ctypes.data_as(ct.POINTER(ct.c_double)),
    )
    assert np.isfinite(actual).all()
    for h in [2e-5, 1e-5]:
        expected = np.empty((3, 3))
        for center in range(3):
            for axis in range(3):
                move = np.zeros((4, 3))
                move[center, axis] = h
                expected[center, axis] = (
                    primitive(p_shell, centers + move, weights)[0]
                    - primitive(p_shell, centers - move, weights)[0]
                ) / (2 * h)
        np.testing.assert_allclose(actual, expected, atol=3e-9, rtol=2e-8)
