"""Real-GPU qualification for #178 second-integral HVP consumers."""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_target import cuda_target_info

from tools.vibeqc_hessian import NativeRHFState, rhf_hessian, rhf_hvp, rhf_hvp_many
from tools.vibeqc_hessian.analytic import provider_hvp_components
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_validation.hessian_fixtures import fixture_inputs

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESPONSE_CUDA_TEST") != "1",
    reason="explicit real-GPU qualification",
)


@pytest.fixture(scope="module")
def compiler() -> CudaCompilerAdapter:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    nvcc = shutil.which("nvcc")
    assert nvcc, "selected CUDA qualification needs nvcc on PATH"
    return CudaCompilerAdapter(
        Path(nvcc),
        cuda_target_info(os.environ.get("VIBEQC_TEST_CUDA_ARCH", "sm_120")),
    )


@pytest.fixture(scope="module")
def h2_case() -> object:
    with NativeSource(**fixture_inputs("h2")) as source:
        state = NativeRHFState.from_source(source)
        vector = np.random.default_rng(1812).normal(size=(state.nat, 3))
        vector /= np.linalg.norm(vector)
        yield state, vector


def test_cuda_second_provider_matches_cpu_components(
    h2_case: tuple[NativeRHFState, np.ndarray], compiler: CudaCompilerAdapter
) -> None:
    state, vector = h2_case
    expected = provider_hvp_components(state, vector)
    actual, diagnostic = provider_hvp_components(
        state,
        vector,
        backend="cuda",
        compiler=compiler,
        return_diagnostics=True,
    )
    for name in ("core", "pulay", "two_electron"):
        np.testing.assert_allclose(actual[name], expected[name], atol=8e-10, rtol=4e-10)
    assert diagnostic["backend"] == "cuda-generated-weighted-hvp"
    assert diagnostic["provider_backend"] == "cuda"
    assert diagnostic["primitive_records"] > 0
    assert diagnostic["record_batch_uploads"] > 0
    assert diagnostic["result_tile_downloads"] > 0
    assert diagnostic["raw_hessian_downloads"] == 0
    assert diagnostic["intermediate_matrix_downloads"] == 0
    assert diagnostic["peak_device_bytes"] > 0
    assert diagnostic["program_identities"]


def test_complete_hvp_uses_cuda_second_provider_without_cpu_substitution(
    h2_case: tuple[NativeRHFState, np.ndarray],
    compiler: CudaCompilerAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.vibeqc_hessian import analytic

    state, vector = h2_case
    expected = rhf_hvp(state, vector)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("CUDA second-integral HVP substituted the CPU compiler")

    monkeypatch.setattr(analytic, "CppCompilerAdapter", forbidden)
    actual = rhf_hvp(
        state,
        vector,
        second_backend="cuda",
        second_compiler=compiler,
    )
    np.testing.assert_allclose(actual.value, expected.value, atol=1e-9, rtol=4e-10)
    for name in ("core", "pulay", "two_electron"):
        np.testing.assert_allclose(
            actual.components[name], expected.components[name], atol=8e-10, rtol=4e-10
        )
    diagnostic = actual.diagnostics
    assert diagnostic["second_integral_backend"] == "cuda-generated-weighted-hvp"
    assert diagnostic["execution_residency"] == "mixed-host-device"
    transfer = diagnostic["transfers"]["second_integral_hvp"]
    assert transfer["record_batch_uploads"] > 0
    assert transfer["result_tile_downloads"] > 0
    assert transfer["raw_hessian_downloads"] == 0


def test_block_hvp_can_use_cuda_second_provider(
    h2_case: tuple[NativeRHFState, np.ndarray], compiler: CudaCompilerAdapter
) -> None:
    state, vector = h2_case
    directions = np.stack((vector, -0.41 * vector))
    expected = rhf_hvp_many(state, directions, strategy="recycled")
    actual = rhf_hvp_many(
        state,
        directions,
        strategy="recycled",
        second_backend="cuda",
        second_compiler=compiler,
    )
    np.testing.assert_allclose(actual.values, expected.values, atol=1e-9, rtol=4e-10)
    diagnostic = actual.diagnostics
    assert diagnostic["second_backend"] == "cuda"
    assert diagnostic["second_integral_phase_numeric_bound_bytes"] > 0
    assert len(diagnostic["second_integral_provider"]) == 2
    assert all(
        item["backend"] == "cuda-generated-weighted-hvp"
        and item["result_tile_downloads"] > 0
        for item in diagnostic["second_integral_provider"]
    )


def test_full_hessian_can_use_cuda_second_provider(
    h2_case: tuple[NativeRHFState, np.ndarray], compiler: CudaCompilerAdapter
) -> None:
    state, _ = h2_case
    expected = rhf_hessian(state, block_size=3, strategy="recycled")
    actual = rhf_hessian(
        state,
        block_size=3,
        strategy="recycled",
        second_backend="cuda",
        second_compiler=compiler,
    )
    np.testing.assert_allclose(actual.matrix, expected.matrix, atol=1e-9, rtol=4e-10)
    assert actual.diagnostics["raw_symmetry_error"] < 2e-9
    assert all(
        block["diagnostics"]["second_backend"] == "cuda"
        for block in actual.diagnostics["blocks"]
    )
