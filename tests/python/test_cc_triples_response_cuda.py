"""Bounded CUDA standard-(T) response and corrected-Lambda tests."""

from __future__ import annotations

import os
import typing
from pathlib import Path

import numpy as np
import pytest
from test_cc_lambda_cuda import _cc_state, _fake_cuda_runtime
from vibeqc.profiles import find_nvcc
from vibeqc_compiler.common.resources import ResourceBudget
from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.integral.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import execute as cpu_execute

from tools.vibeqc_cc import PreparedCUDALambda
from tools.vibeqc_cc.lambda_solver import BoundCCSDLambda
from tools.vibeqc_cc.triples_cuda import TriplesTileConfig
from tools.vibeqc_cc.triples_lambda_response import solve_corrected_lambda
from tools.vibeqc_cc.triples_response import (
    TRIPLES_RESPONSE_INPUTS,
    accumulate_tile_triples_vjp,
)
from tools.vibeqc_cc.triples_response_cuda import (
    CudaTriplesResponseTiles,
    solve_corrected_lambda_cuda,
)
from tools.vibeqc_response.problem import ResponseCompatibilityError

if typing.TYPE_CHECKING:
    from typing_extensions import Self


class _Plan:
    def __init__(self, program: typing.Any, max_bytes: int) -> None:
        self.program = program
        self.peak_bytes = min(max_bytes, 4096)


class _Artifact:
    metadata: typing.ClassVar[dict[str, str]] = {"key": "fake-cuda-triples-response"}


class _Resident:
    def __init__(
        self,
        plan: _Plan,
        artifact: _Artifact,
        *,
        device: int = 0,
    ) -> None:
        del artifact
        self.plan = plan
        self.device = {"ordinal": device, "name": "fake-cuda"}
        self.feeds: dict[str, np.ndarray] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *unused: object) -> None:
        pass

    def upload(self, feeds: typing.Mapping[str, np.ndarray]) -> None:
        self.feeds.update(
            {name: np.array(value, copy=True) for name, value in feeds.items()}
        )

    def run(self, *, profile: bool = False) -> typing.Any:
        del profile
        result = cpu_execute(self.plan.program, self.feeds)
        return dict(result.outputs), {"backend": "fake-cuda"}

    def download(self, value: np.ndarray) -> np.ndarray:
        return np.array(value, copy=True)


def _fake_response_owner(
    nocc: int,
    nvir: int,
    compiler: CudaCompilerAdapter,
    tmp_path: Path,
    *,
    chunk: int = 1,
    max_bytes: int = 32 << 20,
) -> tuple[CudaTriplesResponseTiles, list[int]]:
    owner = CudaTriplesResponseTiles(
        TriplesTileConfig(nocc, nvir, chunk, max_bytes),
        compiler,
        tmp_path,
    )
    budgets: list[int] = []

    def plan(program: typing.Any, target: typing.Any, *, max_bytes: int) -> _Plan:
        del target
        budgets.append(max_bytes)
        return _Plan(program, max_bytes)

    owner._plan_cuda = plan
    owner._compile_resident = lambda *args, **kwargs: _Artifact()
    owner._PreparedResident = _Resident
    return owner, budgets


def _triples_arrays(bound: BoundCCSDLambda) -> dict[str, np.ndarray]:
    o = bound.reference.nocc
    eps = bound.reference.orbital_energies
    return {
        "ovvv": bound.feeds["ovvv"],
        "ovoo": bound.feeds["ovoo"],
        "ovov": bound.feeds["ovov"],
        "fov": bound.feeds["fov"],
        "t1": bound.feeds["t1"],
        "t2": bound.feeds["t2"],
        "eps_o": eps[:o],
        "eps_v": eps[o:],
    }


def test_cuda_reverse_tiles_match_generated_cpu_sources_and_honor_budget(
    tmp_path: Path,
) -> None:
    snapshot, cc = _cc_state("h2o")
    bound = BoundCCSDLambda(snapshot, cc)
    arrays = _triples_arrays(bound)
    compiler = CudaCompilerAdapter(Path("nvcc"), cuda_target_info("sm_120"))
    owner, budgets = _fake_response_owner(
        snapshot.nocc,
        snapshot.nmo - snapshot.nocc,
        compiler,
        tmp_path,
        chunk=1,
        max_bytes=17 << 20,
    )
    selected = TRIPLES_RESPONSE_INPUTS
    actual = owner.run_tiles(arrays, inputs=selected)
    expected = accumulate_tile_triples_vjp(
        snapshot.nocc,
        snapshot.nmo - snapshot.nocc,
        arrays["ovvv"],
        arrays["ovoo"],
        arrays["ovov"],
        arrays["fov"],
        arrays["t1"],
        arrays["t2"],
        arrays["eps_o"],
        arrays["eps_v"],
        vir_chunk_size=1,
        inputs=selected,
    )
    assert budgets and set(budgets) == {17 << 20}
    assert actual.provenance["cpu_fallback"] is False
    assert actual.provenance["full_t3_resident"] is False
    assert actual.peak_device_bytes <= 17 << 20
    assert actual.runtime_device["name"] == "fake-cuda"
    for name in selected:
        np.testing.assert_allclose(
            actual.sources[name], expected[name], atol=1e-12, rtol=1e-12
        )
        assert not actual.sources[name].flags.writeable
    assert np.linalg.norm(actual.sources["eps_o"]) > 0
    assert np.linalg.norm(actual.sources["eps_v"]) > 0


def test_cuda_corrected_lambda_matches_cpu_corrected_equations(
    monkeypatch: typing.Any, tmp_path: Path
) -> None:
    snapshot, cc = _cc_state("h2o")
    compiler = CudaCompilerAdapter(Path("nvcc"), cuda_target_info("sm_120"))
    source_bound = BoundCCSDLambda(snapshot, cc)
    owner, _budgets = _fake_response_owner(
        snapshot.nocc,
        snapshot.nmo - snapshot.nocc,
        compiler,
        tmp_path / "triples",
        chunk=1,
    )
    # Keep the bounded triples-reverse and resident-Lambda device peaks
    # sequential. The response result is host-owned after every tile closes.
    gpu_sources = owner.run_tiles(
        _triples_arrays(source_bound),
        inputs=("t1", "t2"),
    )

    _fake_cuda_runtime(monkeypatch)
    with PreparedCUDALambda(
        snapshot,
        cc,
        compiler,
        tmp_path / "lambda",
        budget=ResourceBudget(host_bytes=512 << 20, device_bytes=1 << 30),
    ) as prepared:
        baseline = prepared.solve(reference_identity=snapshot.identity)
        actual = solve_corrected_lambda_cuda(
            prepared,
            baseline,
            gpu_sources,
            reference_identity=snapshot.identity,
        )
        expected = solve_corrected_lambda(
            prepared.bound,
            baseline,
            vir_chunk_size=1,
        )
        np.testing.assert_allclose(
            actual.lambda1, expected.lambda1, atol=1e-12, rtol=1e-12
        )
        np.testing.assert_allclose(
            actual.lambda2, expected.lambda2, atol=1e-12, rtol=1e-12
        )
        np.testing.assert_allclose(
            actual.delta_lambda1,
            expected.delta_lambda1,
            atol=1e-12,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            actual.delta_lambda2,
            expected.delta_lambda2,
            atol=1e-12,
            rtol=1e-12,
        )
        assert actual.triples_source_identity == expected.triples_source_identity
        assert actual.provenance["cpu_fallback"] is False
        assert "resident-triples-vjp" in actual.provenance["tensor_backend"]
        assert actual.independent_lambda_residual_norm <= 1e-9
        assert actual.independent_lambda_residual_max <= 1e-9


def test_cuda_corrected_lambda_rejects_response_for_other_shape(
    monkeypatch: typing.Any, tmp_path: Path
) -> None:
    snapshot, cc = _cc_state("h2")
    compiler = CudaCompilerAdapter(Path("nvcc"), cuda_target_info("sm_120"))
    source_bound = BoundCCSDLambda(snapshot, cc)
    owner, _budgets = _fake_response_owner(
        snapshot.nocc,
        snapshot.nmo - snapshot.nocc,
        compiler,
        tmp_path / "triples",
    )
    result = owner.run_tiles(_triples_arrays(source_bound), inputs=("t1", "t2"))

    _fake_cuda_runtime(monkeypatch)
    with PreparedCUDALambda(
        snapshot,
        cc,
        compiler,
        tmp_path / "lambda",
        budget=ResourceBudget(host_bytes=512 << 20, device_bytes=1 << 30),
    ) as prepared:
        baseline = prepared.solve(reference_identity=snapshot.identity)
        forged = type(result)(
            sources=result.sources,
            inputs=result.inputs,
            nocc=result.nocc,
            nvir=result.nvir + 1,
            vir_chunk_size=result.vir_chunk_size,
            peak_device_bytes=result.peak_device_bytes,
            input_identity=result.input_identity,
            source_identity=result.source_identity,
            provenance=result.provenance,
        )
        with pytest.raises(ResponseCompatibilityError, match="another CC state"):
            solve_corrected_lambda_cuda(
                prepared,
                baseline,
                forged,
                reference_identity=snapshot.identity,
            )

        wrong_inputs = type(result)(
            sources=result.sources,
            inputs=result.inputs,
            nocc=result.nocc,
            nvir=result.nvir,
            vir_chunk_size=result.vir_chunk_size,
            peak_device_bytes=result.peak_device_bytes,
            input_identity="forged-input-state",
            source_identity=result.source_identity,
            provenance=result.provenance,
        )
        with pytest.raises(ResponseCompatibilityError, match="inputs belong"):
            solve_corrected_lambda_cuda(
                prepared,
                baseline,
                wrong_inputs,
                reference_identity=snapshot.identity,
            )


_REAL = os.environ.get("VIBEQC_CC_TRIPLES_RESPONSE_CUDA_TEST") == "1"


@pytest.mark.skipif(
    not _REAL, reason="requires explicitly allocated CUDA qualification"
)
def test_real_cuda_water_triples_response_and_corrected_lambda(
    tmp_path: Path,
) -> None:
    assert os.environ.get("SLURM_JOB_ID"), (
        "CUDA triples response qualification requires Slurm"
    )
    snapshot, cc = _cc_state("h2o")
    cpu_bound = BoundCCSDLambda(snapshot, cc)
    cpu_baseline = cpu_bound.solve(reference_identity=snapshot.identity)
    expected = solve_corrected_lambda(cpu_bound, cpu_baseline, vir_chunk_size=1)

    nvcc = find_nvcc()
    assert nvcc is not None
    compiler = CudaCompilerAdapter(
        nvcc, cuda_target_info(os.environ.get("VIBEQC_TENSOR_ARCH", "sm_120"))
    )
    cache = Path(
        os.environ.get("VIBEQC_TENSOR_CACHE", tmp_path / "triples-response-cuda")
    )
    response_owner = CudaTriplesResponseTiles(
        TriplesTileConfig(
            snapshot.nocc,
            snapshot.nmo - snapshot.nocc,
            1,
            1 << 30,
        ),
        compiler,
        cache,
    )
    response = response_owner.run_tiles(
        _triples_arrays(cpu_bound),
        inputs=TRIPLES_RESPONSE_INPUTS,
    )
    # Compare every actual device response block, not only the t1/t2 blocks
    # consumed by corrected Lambda or a nonzero-denominator sanity check.
    arrays = _triples_arrays(cpu_bound)
    expected_sources = accumulate_tile_triples_vjp(
        snapshot.nocc,
        snapshot.nmo - snapshot.nocc,
        arrays["ovvv"],
        arrays["ovoo"],
        arrays["ovov"],
        arrays["fov"],
        arrays["t1"],
        arrays["t2"],
        arrays["eps_o"],
        arrays["eps_v"],
        vir_chunk_size=1,
        inputs=TRIPLES_RESPONSE_INPUTS,
    )
    for name in TRIPLES_RESPONSE_INPUTS:
        np.testing.assert_allclose(
            response.sources[name], expected_sources[name], atol=2e-10, rtol=2e-10
        )
    # Do not overlap the reverse-tile arena with the persistent Lambda owner.
    with PreparedCUDALambda(
        snapshot,
        cc,
        compiler,
        cache,
        budget=ResourceBudget(host_bytes=1 << 30, device_bytes=2 << 30),
    ) as prepared:
        baseline = prepared.solve(reference_identity=snapshot.identity)
        corrected = solve_corrected_lambda_cuda(
            prepared,
            baseline,
            response,
            reference_identity=snapshot.identity,
        )
        np.testing.assert_allclose(
            corrected.lambda1, expected.lambda1, atol=2e-10, rtol=2e-10
        )
        np.testing.assert_allclose(
            corrected.lambda2, expected.lambda2, atol=2e-10, rtol=2e-10
        )
        assert corrected.independent_lambda_residual_norm <= 1.1e-9
        assert corrected.independent_lambda_residual_max <= 1.1e-9
        assert response.peak_device_bytes <= 1 << 30
        assert response.runtime_device is not None
        assert np.linalg.norm(response.sources["eps_o"]) > 0
        assert np.linalg.norm(response.sources["eps_v"]) > 0
