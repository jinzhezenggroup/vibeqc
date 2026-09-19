"""Generated device H1/S1 plus CUDA J/K, independently checked on native states."""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.integral.first_derivatives_execute import FirstDerivativeEvaluator

from tools.vibeqc_hessian import (
    NativeRHFState,
    directional,
    directional_rhf_response,
    first_order,
)
from tools.vibeqc_posthf.export import conventional_fock
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_response import NativeJKBackend
from tools.vibeqc_validation.hessian_fixtures import fixture_inputs

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESPONSE_CUDA_TEST") != "1",
    reason="explicit real-GPU qualification",
)


@pytest.fixture(scope="module", params=("h2", "water"))
def case(request):
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    nvcc = shutil.which("nvcc")
    assert nvcc
    compiler = CudaCompilerAdapter(
        Path(nvcc), cuda_target_info(os.environ.get("VIBEQC_TEST_CUDA_ARCH", "sm_120"))
    )
    with NativeSource(**fixture_inputs(request.param)) as source:
        state = NativeRHFState.from_source(source)
        v = np.random.default_rng(180).normal(size=(state.nat, 3))
        v /= np.linalg.norm(v)
        oracle = source.integral_derivatives()
        eri = np.einsum("a,apqrs->pqrs", v.ravel(), oracle["eri"])
        h = (
            np.einsum("a,apq->pq", v.ravel(), oracle["hcore"])
            + np.einsum("pqrs,rs->pq", eri, state.P0)
            - 0.5 * np.einsum("prqs,rs->pq", eri, state.P0)
        )
        overlap = np.einsum("a,apq->pq", v.ravel(), oracle["overlap"])
        finite = []
        for step in (3e-3, 1e-3, 3e-4):
            pair = []
            for sign in (1, -1):
                atoms = [
                    (a.atomic_number, r)
                    for a, r in zip(
                        source.atoms, state.coords + sign * step * v, strict=True
                    )
                ]
                with NativeSource(
                    atoms, basis=source.shells, charge=source.charge
                ) as moved_source:
                    moved = NativeRHFState.from_source(moved_source, tolerance=1e-13)
                    c = moved.C[:, : moved.nocc]
                    w = (c * (2 * moved.eps[: moved.nocc])) @ c.T
                    pair.append(
                        (
                            conventional_fock(moved_source, state.P0),
                            moved.S0,
                            moved.P0,
                            w,
                        )
                    )
            finite.append(
                tuple((a - b) / (2 * step) for a, b in zip(*pair, strict=True))
            )

        def forbidden(*args, **kwargs):
            raise AssertionError(
                "CUDA first-source/response delegated scientific work to CPU/oracle"
            )

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(NativeRHFState, "first_order_inputs", property(forbidden))
            patch.setattr(directional, "generated_directional_first_order", forbidden)
            patch.setattr(first_order, "generated_directional_first_order", forbidden)
            patch.setattr(FirstDerivativeEvaluator, "contract", forbidden)
            patch.setattr(NativeJKBackend, "coulomb_exchange", forbidden)
            for method in ("integral_derivatives", "rhf_density", "tile", "requests"):
                patch.setattr(source, method, forbidden)
            actual = directional_rhf_response(
                state,
                v,
                first_backend="cuda",
                first_compiler=compiler,
                jk_backend="cuda",
            )
        yield state, v, actual, (h, overlap), finite, compiler


def test_device_first_sources_match_independent_analytic_integrals(case):
    _, _, actual, expected, _, _ = case
    np.testing.assert_allclose(
        actual.frozen_fock_derivative, expected[0], atol=2e-10, rtol=2e-10
    )
    np.testing.assert_allclose(
        actual.overlap_derivative, expected[1], atol=2e-11, rtol=2e-11
    )
    for m in (actual.frozen_fock_derivative, actual.overlap_derivative):
        np.testing.assert_allclose(m, m.T, atol=2e-10, rtol=0)


def test_three_step_native_differences_check_entire_directional_response(case):
    _, _, actual, _, finite, _ = case
    targets = (
        actual.frozen_fock_derivative,
        actual.overlap_derivative,
        actual.response.density_derivative,
        actual.response.energy_weighted_density_derivative,
    )
    errors = np.asarray(
        [
            [float(np.max(np.abs(a - b))) for a, b in zip(row, targets, strict=True)]
            for row in finite
        ]
    )
    assert np.all(errors[-1] < 2e-5), errors
    assert np.all(errors[-1] < np.maximum(0.2 * errors[0], 2e-7)), errors
    assert actual.response.solve_result.converged
    assert actual.response.solve_result.residual_norm < 1e-9


def test_no_raw_derivative_download_and_explicit_residency(case):
    s, _, actual, _, _, _ = case
    diag = actual.diagnostics
    provider = diag["first_derivative_provider"]
    assert diag["first_integral_derivatives"] == "cuda-generated"
    assert diag["direction_density_reduction"] == "cuda-generated"
    assert diag["jk_backend"] == "cuda"
    assert provider["raw_derivative_downloads"] == 0
    assert provider["matrix_downloads"] == 1
    assert provider["storage"]["output_bytes"] == 2 * s.nbf**2 * 8
    # These other stages have NOT been moved to CUDA by this PR.
    assert diag["ao_mo_transforms"] == diag["krylov_execution"] == "host"
    assert not diag["molecular_hvp"]


@pytest.mark.parametrize("case", ["h2"], indirect=True)
def test_zero_translation_and_signed_scaling(case):
    state, v, actual, _, _, compiler = case
    for scale in (0.0, -0.7):
        r = directional_rhf_response(
            state,
            scale * v,
            first_backend="cuda",
            first_compiler=compiler,
            jk_backend="cuda",
        )
        np.testing.assert_allclose(
            r.response.density_derivative,
            scale * actual.response.density_derivative,
            atol=2e-9,
            rtol=2e-9,
        )
        np.testing.assert_allclose(
            r.frozen_fock_derivative,
            scale * actual.frozen_fock_derivative,
            atol=2e-10,
            rtol=2e-10,
        )
    r = directional_rhf_response(
        state,
        np.tile([0.1, -0.2, 0.3], (state.nat, 1)),
        first_backend="cuda",
        first_compiler=compiler,
        jk_backend="cuda",
    )
    np.testing.assert_allclose(r.frozen_fock_derivative, 0, atol=2e-10, rtol=0)
    np.testing.assert_allclose(
        r.response.energy_weighted_density_derivative, 0, atol=2e-9, rtol=0
    )


@pytest.mark.parametrize("case", ["h2"], indirect=True)
def test_tiny_first_source_budget_fails_without_disabling_later_calls(case):
    state, v, expected, _, _, compiler = case
    with pytest.raises(MemoryError):
        directional_rhf_response(
            state,
            v,
            first_backend="cuda",
            first_compiler=compiler,
            jk_backend="cuda",
            first_budget_bytes=1,
        )
    r = directional_rhf_response(
        state, v, first_backend="cuda", first_compiler=compiler, jk_backend="cuda"
    )
    np.testing.assert_allclose(
        r.frozen_fock_derivative,
        expected.frozen_fock_derivative,
        atol=2e-10,
        rtol=2e-10,
    )
