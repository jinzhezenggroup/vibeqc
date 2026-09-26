"""Real-device qualification for final CUDA RHF HVP/Hessian assembly."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_target import cuda_target_info

from tools.vibeqc_hessian import NativeRHFState, rhf_hessian, rhf_hvp, rhf_hvp_many
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
        vector = np.random.default_rng(1802).normal(size=(state.nat, 3))
        vector /= np.linalg.norm(vector)
        yield state, vector


def _cuda_kwargs(compiler: CudaCompilerAdapter) -> dict[str, object]:
    return {
        "jk_backend": "cuda",
        "response_execution": "cuda-resident",
        "second_backend": "cuda",
        "second_compiler": compiler,
        "relaxation_backend": "cuda",
        "relaxation_compiler": compiler,
        "assembly_backend": "cuda",
    }


def test_scalar_hvp_final_assembly_publishes_only_final_vector(
    h2_case: tuple[NativeRHFState, np.ndarray],
    compiler: CudaCompilerAdapter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, vector = h2_case
    expected = rhf_hvp(state, vector)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("CUDA final assembly substituted a host HVP component")

    monkeypatch.setattr("tools.vibeqc_hessian.hvp.nuclear_hvp", forbidden)
    monkeypatch.setattr("tools.vibeqc_hessian.hvp.provider_hvp_components", forbidden)

    actual = rhf_hvp(state, vector, **_cuda_kwargs(compiler))
    np.testing.assert_allclose(actual.value, expected.value, atol=1e-9, rtol=4e-10)
    assert all(value is None for value in actual.components.values())
    diagnostic = actual.diagnostics
    assert diagnostic["final_assembly_backend"] == "cuda-device-resident"
    assert diagnostic["published_component_bytes"] == 0
    assert diagnostic["second_integral_provider"]["result_tile_downloads"] == 0
    assert diagnostic["second_integral_provider"]["device_result_consumptions"] > 0
    assert diagnostic["relaxation_provider"]["host_output_published"] is False
    assert diagnostic["final_assembly"]["d2h_bytes"] == actual.value.nbytes


@pytest.mark.parametrize("strategy", ("sequential", "blocked", "recycled"))
def test_block_hvp_final_assembly_keeps_components_on_device(
    h2_case: tuple[NativeRHFState, np.ndarray],
    compiler: CudaCompilerAdapter,
    strategy: str,
) -> None:
    state, vector = h2_case
    directions = np.stack((vector, -0.37 * vector))
    expected = rhf_hvp_many(state, directions, strategy=strategy)
    actual = rhf_hvp_many(
        state,
        directions,
        strategy=strategy,
        **_cuda_kwargs(compiler),
    )
    assert actual.values is not None
    np.testing.assert_allclose(actual.values, expected.values, atol=1e-9, rtol=4e-10)
    assert all(value is None for value in actual.components.values())
    diagnostic = actual.diagnostics
    assert diagnostic["assembly_backend"] == "cuda"
    assert diagnostic["published_component_bytes"] == 0
    assert diagnostic["published_hvp_bytes"] == actual.values.nbytes
    assert len(diagnostic["final_assembly"]) == len(directions)
    assert all(
        item["d2h_bytes"] == actual.values[0].nbytes
        for item in diagnostic["final_assembly"]
    )
    assert all(
        item["result_tile_downloads"] == 0 and item["device_result_consumptions"] > 0
        for item in diagnostic["second_integral_provider"]
    )
    assert all(
        item["host_output_published"] is False
        for item in diagnostic["relaxation_provider"]
    )


def test_full_hessian_downloads_matrix_once_without_host_hvp_columns(
    h2_case: tuple[NativeRHFState, np.ndarray],
    compiler: CudaCompilerAdapter,
) -> None:
    state, _ = h2_case
    expected = rhf_hessian(state, block_size=2, strategy="recycled")
    actual = rhf_hessian(
        state,
        block_size=2,
        strategy="recycled",
        **_cuda_kwargs(compiler),
    )
    np.testing.assert_allclose(actual.matrix, expected.matrix, atol=1e-9, rtol=4e-10)
    diagnostic = actual.diagnostics
    assert diagnostic["assembly_backend"] == "cuda"
    assert diagnostic["intermediate_hvp_host_publication"] is False
    assert diagnostic["final_matrix_downloads"] == 1
    assert diagnostic["matrix_device_assembly"]["d2h_bytes"] == actual.matrix.nbytes
    assert diagnostic["matrix_device_assembly"]["column_copies"] == 3 * state.nat
    assert diagnostic["raw_symmetry_error"] < 2e-9
    assert all(
        block["diagnostics"]["published_hvp_bytes"] == 0
        and block["diagnostics"]["published_component_bytes"] == 0
        for block in diagnostic["blocks"]
    )
