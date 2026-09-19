"""Real-GPU qualification for generated RHF relaxation contraction."""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_target import cuda_target_info

from tools.vibeqc_hessian import NativeRHFState, rhf_hessian, rhf_hvp, rhf_hvp_many
from tools.vibeqc_hessian.directional import directional_rhf_response
from tools.vibeqc_hessian.first_order import generated_rhf_relaxation_contraction
from tools.vibeqc_hessian.first_order_cuda import (
    generated_rhf_relaxation_contraction_cuda,
)
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_validation.hessian_fixtures import fixture_inputs

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESPONSE_CUDA_TEST") != "1",
    reason="explicit real-GPU qualification",
)


@pytest.fixture(scope="module")
def compiler():
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    nvcc = shutil.which("nvcc")
    assert nvcc, "selected CUDA qualification needs nvcc on PATH"
    return CudaCompilerAdapter(
        Path(nvcc),
        cuda_target_info(os.environ.get("VIBEQC_TEST_CUDA_ARCH", "sm_120")),
    )


@pytest.fixture(scope="module")
def h2_case():
    with NativeSource(**fixture_inputs("h2")) as source:
        state = NativeRHFState.from_source(source)
        vector = np.random.default_rng(1810).normal(size=(state.nat, 3))
        vector /= np.linalg.norm(vector)
        response = directional_rhf_response(state, vector)
        yield state, vector, response


def test_cuda_relaxation_matches_independent_cpu_contraction(h2_case, compiler):
    state, _, response = h2_case
    density = response.response.density_derivative
    weighted = response.response.energy_weighted_density_derivative
    expected = generated_rhf_relaxation_contraction(state, density, weighted)
    actual, diagnostic = generated_rhf_relaxation_contraction_cuda(
        state, density, weighted, compiler
    )
    np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=3e-10)
    assert diagnostic["backend"] == "cuda-generated-weighted-gradient"
    assert diagnostic["weight_uploads"] == 1
    assert diagnostic["gradient_downloads"] == 1
    assert diagnostic["raw_derivative_downloads"] == 0
    assert diagnostic["intermediate_matrix_downloads"] == 0
    assert diagnostic["primitive_records"] > 0


def test_complete_hvp_can_select_cuda_relaxation_without_cpu_substitution(
    h2_case, compiler, monkeypatch
):
    state, vector, _ = h2_case
    expected = rhf_hvp(state, vector)

    def forbidden(*args, **kwargs):
        raise AssertionError("CUDA relaxation substituted the CPU contraction")

    monkeypatch.setattr(
        "tools.vibeqc_hessian.hvp.generated_rhf_relaxation_contraction",
        forbidden,
    )
    actual = rhf_hvp(
        state,
        vector,
        relaxation_backend="cuda",
        relaxation_compiler=compiler,
    )
    np.testing.assert_allclose(actual.value, expected.value, atol=1e-9, rtol=4e-10)
    np.testing.assert_allclose(
        actual.relaxation, expected.relaxation, atol=3e-10, rtol=3e-10
    )
    diagnostic = actual.diagnostics
    assert diagnostic["relaxation_first_integral_backend"] == (
        "cuda-generated-weighted-gradient"
    )
    assert diagnostic["execution_residency"] == "mixed-host-device"
    transfer = diagnostic["transfers"]["relaxation_first_integrals"]
    assert transfer["weight_uploads"] == 1
    assert transfer["gradient_downloads"] == 1
    assert transfer["raw_derivative_downloads"] == 0


def test_block_hvp_can_use_cuda_relaxation(h2_case, compiler):
    state, vector, _ = h2_case
    directions = np.stack((vector, -0.37 * vector))
    expected = rhf_hvp_many(state, directions, strategy="recycled")
    actual = rhf_hvp_many(
        state,
        directions,
        strategy="recycled",
        relaxation_backend="cuda",
        relaxation_compiler=compiler,
    )
    np.testing.assert_allclose(actual.values, expected.values, atol=1e-9, rtol=4e-10)
    np.testing.assert_allclose(
        actual.relaxation, expected.relaxation, atol=3e-10, rtol=3e-10
    )
    diagnostic = actual.diagnostics
    assert diagnostic["relaxation_backend"] == "cuda"
    assert len(diagnostic["relaxation_provider"]) == 2
    assert all(
        item["gradient_downloads"] == 1 and item["raw_derivative_downloads"] == 0
        for item in diagnostic["relaxation_provider"]
    )


def test_full_hessian_can_use_cuda_relaxation(h2_case, compiler):
    state, _, _ = h2_case
    expected = rhf_hessian(state, block_size=2, strategy="recycled")
    actual = rhf_hessian(
        state,
        block_size=2,
        strategy="recycled",
        relaxation_backend="cuda",
        relaxation_compiler=compiler,
    )
    np.testing.assert_allclose(actual.matrix, expected.matrix, atol=1e-9, rtol=4e-10)
    assert actual.diagnostics["raw_symmetry_error"] < 2e-9
    assert all(
        block["diagnostics"]["relaxation_backend"] == "cuda"
        for block in actual.diagnostics["blocks"]
    )
