"""Real-GPU qualification for generated RHF relaxation contraction."""

from __future__ import annotations

import os
import shutil
import typing
from pathlib import Path
from typing import NoReturn

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
def compiler() -> typing.Any:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    nvcc = shutil.which("nvcc")
    assert nvcc, "selected CUDA qualification needs nvcc on PATH"
    return CudaCompilerAdapter(
        Path(nvcc),
        cuda_target_info(os.environ.get("VIBEQC_TEST_CUDA_ARCH", "sm_120")),
    )


@pytest.fixture(scope="module")
def h2_case() -> typing.Any:
    with NativeSource(**fixture_inputs("h2")) as source:
        state = NativeRHFState.from_source(source)
        vector = np.random.default_rng(1810).normal(size=(state.nat, 3))
        vector /= np.linalg.norm(vector)
        response = directional_rhf_response(state, vector)
        yield state, vector, response


def test_cuda_relaxation_matches_independent_cpu_contraction(
    h2_case: typing.Any, compiler: typing.Any
) -> None:
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
    h2_case: typing.Any, compiler: typing.Any, monkeypatch: typing.Any
) -> None:
    state, vector, _ = h2_case
    expected = rhf_hvp(state, vector)

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> NoReturn:
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


def test_resident_response_feeds_cuda_relaxation_without_host_d1_w1_reupload(
    h2_case: typing.Any,
    compiler: typing.Any,
    monkeypatch: typing.Any,
) -> None:
    state, vector, _ = h2_case
    expected = rhf_hvp(state, vector)

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> NoReturn:
        raise AssertionError("resident relaxation validated/reuploaded host D1/W1")

    monkeypatch.setattr(
        "tools.vibeqc_hessian.first_order_cuda._checked_ao_weight",
        forbidden,
    )
    actual = rhf_hvp(
        state,
        vector,
        jk_backend="cuda",
        response_execution="cuda-resident",
        relaxation_backend="cuda",
        relaxation_compiler=compiler,
    )
    np.testing.assert_allclose(actual.value, expected.value, atol=1e-9, rtol=4e-10)
    np.testing.assert_allclose(
        actual.relaxation, expected.relaxation, atol=3e-10, rtol=3e-10
    )
    provider = actual.diagnostics["relaxation_provider"]
    assert provider["response_weight_source"] == "resident-d2d"
    assert provider["resident_weight_imports"] == 1
    assert provider["gradient_downloads"] == 1
    assert actual.diagnostics["response_execution"] == "cuda-resident"


@pytest.mark.parametrize("strategy", ("sequential", "blocked", "recycled"))
def test_resident_block_response_feeds_cuda_relaxation_without_host_d1_w1_reupload(
    h2_case: typing.Any,
    compiler: typing.Any,
    monkeypatch: typing.Any,
    strategy: str,
) -> None:
    state, vector, _ = h2_case
    directions = np.stack((vector, -0.37 * vector))
    expected = rhf_hvp_many(state, directions, strategy=strategy)

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> NoReturn:
        raise AssertionError("resident block relaxation revalidated host D1/W1")

    monkeypatch.setattr(
        "tools.vibeqc_hessian.first_order_cuda._checked_ao_weight",
        forbidden,
    )
    actual = rhf_hvp_many(
        state,
        directions,
        strategy=strategy,
        jk_backend="cuda",
        response_execution="cuda-resident",
        relaxation_backend="cuda",
        relaxation_compiler=compiler,
    )
    np.testing.assert_allclose(actual.values, expected.values, atol=1e-9, rtol=4e-10)
    np.testing.assert_allclose(
        actual.relaxation, expected.relaxation, atol=3e-10, rtol=3e-10
    )
    diagnostic = actual.diagnostics
    assert diagnostic["execution_residency"] == (
        "mixed-host-device-resident-response-relaxation"
    )
    assert diagnostic["resident_response_relaxation_phase_numeric_bound_bytes"] > 0
    assert (
        diagnostic["complete_numeric_peak_bound_bytes"]
        <= diagnostic["total_budget_bytes"]
    )
    assert len(diagnostic["relaxation_provider"]) == len(directions)
    assert all(
        item["response_weight_source"] == "resident-d2d"
        and item["resident_weight_imports"] == 1
        and item["gradient_downloads"] == 1
        for item in diagnostic["relaxation_provider"]
    )


def test_resident_full_hessian_feeds_cuda_relaxation_without_host_d1_w1_reupload(
    h2_case: typing.Any,
    compiler: typing.Any,
    monkeypatch: typing.Any,
) -> None:
    state, _, _ = h2_case
    expected = rhf_hessian(state, block_size=2, strategy="recycled")

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> NoReturn:
        raise AssertionError("resident Hessian relaxation revalidated host D1/W1")

    monkeypatch.setattr(
        "tools.vibeqc_hessian.first_order_cuda._checked_ao_weight",
        forbidden,
    )
    actual = rhf_hessian(
        state,
        block_size=2,
        strategy="recycled",
        jk_backend="cuda",
        response_execution="cuda-resident",
        relaxation_backend="cuda",
        relaxation_compiler=compiler,
    )
    np.testing.assert_allclose(actual.matrix, expected.matrix, atol=1e-9, rtol=4e-10)
    assert actual.diagnostics["raw_symmetry_error"] < 2e-9
    blocks = actual.diagnostics["blocks"]
    assert blocks
    assert all(
        block["diagnostics"]["execution_residency"]
        == "mixed-host-device-resident-response-relaxation"
        and all(
            item["response_weight_source"] == "resident-d2d"
            for item in block["diagnostics"]["relaxation_provider"]
        )
        for block in blocks
    )


def test_block_hvp_can_use_cuda_relaxation(
    h2_case: typing.Any, compiler: typing.Any
) -> None:
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


def test_full_hessian_can_use_cuda_relaxation(
    h2_case: typing.Any, compiler: typing.Any
) -> None:
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
