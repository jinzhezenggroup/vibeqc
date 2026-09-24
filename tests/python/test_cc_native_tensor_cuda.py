"""Shared CUDA CC TensorIR executor and RCCSD(T) parameter-response tests."""

from __future__ import annotations

import typing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_cc_lambda_cuda import _cc_state
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import execute as cpu_execute

from tools.vibeqc_cc import CudaCCTensorExecutor
from tools.vibeqc_cc.lambda_equations import build_parameter_vjp
from tools.vibeqc_cc.lambda_solver import BoundCCSDLambda
from tools.vibeqc_cc.triples_lambda_response import (
    BoundCCSDTResponse,
    solve_corrected_lambda,
)


class _Plan:
    def __init__(self, program: typing.Any, max_bytes: int) -> None:
        self.program = program
        self.peak_bytes = min(max_bytes, 8192)


class _Artifact:
    metadata: typing.ClassVar[dict[str, str]] = {"key": "fake-cuda-cc-response"}


class _Prepared:
    backend = "cuda-fp64-ordinary-stream"

    def __init__(
        self,
        plan: _Plan,
        artifact: _Artifact,
        *,
        device: int = 0,
    ) -> None:
        del artifact
        self.plan = plan
        self.device = device

    def __enter__(self) -> typing.Self:
        return self

    def __exit__(self, *unused: object) -> None:
        pass

    def execute(self, feeds: typing.Mapping[str, np.ndarray]) -> typing.Any:
        result = cpu_execute(self.plan.program, feeds)
        return SimpleNamespace(
            outputs=dict(result.outputs),
            metrics={"fake_cuda": True, "device": self.device},
            backend=self.backend,
        )


def _fake_executor(
    tmp_path: Path,
) -> tuple[CudaCCTensorExecutor, list[int], list[str]]:
    compiler = CudaCompilerAdapter(Path("nvcc"), cuda_target_info("sm_120"))
    executor = CudaCCTensorExecutor(
        max_bytes=64 << 20,
        compiler=compiler,
        cache=tmp_path,
    )
    budgets: list[int] = []
    compiled: list[str] = []

    def plan(program: typing.Any, target: typing.Any, *, max_bytes: int) -> _Plan:
        del target
        budgets.append(max_bytes)
        return _Plan(program, max_bytes)

    def compile_plan(
        plan: _Plan,
        compiler: CudaCompilerAdapter,
        cache: Path,
    ) -> _Artifact:
        del compiler, cache
        compiled.append(plan.program.logical_hash)
        return _Artifact()

    executor._plan_cuda = plan
    executor._compile_cuda = compile_plan
    executor._PreparedCuda = _Prepared
    return executor, budgets, compiled


def test_cuda_cc_tensor_executor_caches_program_and_releases_each_arena(
    tmp_path: Path,
) -> None:
    snapshot, cc = _cc_state("h2o")
    bound = BoundCCSDLambda(snapshot, cc)
    baseline = bound.solve(reference_identity=snapshot.identity)
    reverse = build_parameter_vjp(bound.programs.primal, "fov").program
    feeds = {
        **bound.feeds,
        "bar_correlation_energy": np.asarray(1.0),
        "bar_singles_residual": baseline.lambda1,
        "bar_doubles_residual": baseline.lambda2,
    }
    executor, budgets, compiled = _fake_executor(tmp_path)

    first = executor.execute(reverse, feeds)
    second = executor.execute(reverse, feeds)

    np.testing.assert_array_equal(first["bar_fov"], second["bar_fov"])
    assert budgets == [64 << 20]
    assert compiled == [reverse.logical_hash]
    assert executor.compiled_program_count == 1
    assert executor.metrics[reverse.logical_hash]["fake_cuda"] is True


def test_cuda_cc_tensor_executor_rejects_backend_substitution(tmp_path: Path) -> None:
    snapshot, cc = _cc_state("h2")
    bound = BoundCCSDLambda(snapshot, cc)
    baseline = bound.solve(reference_identity=snapshot.identity)
    reverse = build_parameter_vjp(bound.programs.primal, "fov").program
    feeds = {
        **bound.feeds,
        "bar_correlation_energy": np.asarray(1.0),
        "bar_singles_residual": baseline.lambda1,
        "bar_doubles_residual": baseline.lambda2,
    }
    executor, _budgets, _compiled = _fake_executor(tmp_path)

    class WrongBackend(_Prepared):
        def execute(self, feeds: typing.Mapping[str, np.ndarray]) -> typing.Any:
            result = super().execute(feeds)
            result.backend = "numpy-cpu-interpreter"
            return result

    executor._PreparedCuda = WrongBackend
    with pytest.raises(RuntimeError, match="no CPU or alternate fallback"):
        executor.execute(reverse, feeds)


def test_rccsdt_parameter_vjps_use_cuda_executor_without_cpu_replay(
    monkeypatch: typing.Any, tmp_path: Path
) -> None:
    snapshot, cc = _cc_state("h2o")
    bound = BoundCCSDLambda(snapshot, cc)
    baseline = bound.solve(reference_identity=snapshot.identity)
    corrected = solve_corrected_lambda(bound, baseline, vir_chunk_size=1)

    expected_response = BoundCCSDTResponse(
        bound,
        baseline,
        corrected,
        vir_chunk_size=1,
    )
    expected = expected_response.weight("fov", reference_identity=snapshot.identity)

    executor, budgets, compiled = _fake_executor(tmp_path)
    response = BoundCCSDTResponse(
        bound,
        baseline,
        corrected,
        vir_chunk_size=1,
        parameter_executor=executor,
    )

    def reject_bound_tensor_execution(
        *args: object, **kwargs: object
    ) -> typing.NoReturn:
        del args, kwargs
        raise AssertionError("bound CPU parameter TensorIR replayed")

    monkeypatch.setattr(
        BoundCCSDLambda, "_tensor_execute", reject_bound_tensor_execution
    )
    actual = response.weight("fov", reference_identity=snapshot.identity)

    assert response.response_identity == expected_response.response_identity
    np.testing.assert_allclose(actual.values, expected.values, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        actual.baseline_ccsd, expected.baseline_ccsd, atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(
        actual.delta_lambda, expected.delta_lambda, atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(
        actual.direct_triples, expected.direct_triples, atol=1e-12, rtol=1e-12
    )
    assert actual.provenance["parameter_response_backend"] == executor.backend
    assert budgets == [64 << 20, 64 << 20]
    assert len(compiled) == executor.compiled_program_count == 2
