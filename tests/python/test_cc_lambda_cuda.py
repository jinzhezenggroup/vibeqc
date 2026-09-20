"""Generated CUDA execution for the RCCSD Lambda response boundary."""

import os
import typing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from test_cc_api import fixture_problem
from vibeqc.profiles import find_nvcc
from vibeqc_compiler.common.resources import ResourceBudget
from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.integral.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import execute as cpu_execute

from tools.vibeqc_cc import (
    BoundCCSDLambda,
    BoundCCSDResponse,
    PreparedCUDALambda,
    solve,
)
from tools.vibeqc_cc.solver import SolverOptions
from tools.vibeqc_response.problem import ResponseCompatibilityError


class _Choice:
    def __init__(self, program: typing.Any, name: str) -> None:
        self.program = program
        self.request = SimpleNamespace(name=name)

    def factory(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        return self.program


class _Admission:
    identity = "fake-cuda-lambda-resource-plan"

    def __init__(self) -> None:
        self.peak_bytes = {"host": 4096, "device": 8192}

    def require_feasible(self) -> "_Admission":
        return self


class _Provider:
    backend = "cuda-fp64-ordinary-stream"

    def __init__(
        self, program: typing.Any, name: str, *, backend: str | None = None
    ) -> None:
        self.program = program
        self.identity = "fake-provider-" + name
        self._backend = backend or self.backend

    def execute(self, feeds: typing.Any) -> typing.Any:
        result = cpu_execute(self.program, feeds)
        return SimpleNamespace(
            outputs=result.outputs,
            backend=self._backend,
            metrics={
                "tracked_host_bytes": 128,
                "tracked_device_bytes": 256,
                "observed_host_to_device_bytes": 64,
                "observed_device_to_host_bytes": 32,
            },
        )


def _fake_cuda_runtime(
    monkeypatch: typing.Any, *, bad_stage: str | None = None
) -> None:
    import tools.vibeqc_cc.lambda_cuda as module

    programs = {}

    def choices(
        program: typing.Any, *args: typing.Any, name: str, **kwargs: typing.Any
    ) -> _Choice:
        stage = name.removeprefix("cc-lambda-")
        programs[stage] = program
        return _Choice(program, name)

    class Session:
        def __init__(self, admission: typing.Any, factories: typing.Any) -> None:
            self.plan = admission
            self.providers = {
                stage: _Provider(
                    program,
                    stage,
                    backend=("numpy-cpu-interpreter" if stage == bad_stage else None),
                )
                for stage, program in programs.items()
            }

        def advance(self, phase: int) -> None:
            assert phase == 0

        def provider(self, stage: str) -> _Provider:
            return self.providers[stage]

        def close(self) -> None:
            pass

    monkeypatch.setattr(module, "tensor_resource_choices", choices)
    monkeypatch.setattr(module, "plan_resources", lambda *a, **k: _Admission())
    monkeypatch.setattr(module, "ResourceSession", Session)


def _cc_state(name: str = "h2") -> typing.Any:
    snapshot, provider, _meta, _arrays = fixture_problem(name)
    cc = solve(
        snapshot,
        provider,
        options=SolverOptions(residual_tolerance=1e-11, energy_tolerance=1e-13),
    )
    assert cc.converged
    return snapshot, cc


def test_cuda_lambda_actions_match_cpu_and_feed_checked_response(
    monkeypatch: typing.Any, tmp_path: Path
) -> None:
    _fake_cuda_runtime(monkeypatch)
    snapshot, cc = _cc_state()
    cpu_bound = BoundCCSDLambda(snapshot, cc)
    expected = cpu_bound.solve(reference_identity=snapshot.identity)
    compiler = CudaCompilerAdapter(Path("nvcc"), cuda_target_info("sm_120"))
    with PreparedCUDALambda(
        snapshot,
        cc,
        compiler,
        tmp_path,
        budget=ResourceBudget(host_bytes=512 << 20, device_bytes=1 << 30),
    ) as prepared:
        actual = prepared.solve(reference_identity=snapshot.identity)
        np.testing.assert_allclose(
            actual.lambda1, expected.lambda1, atol=1e-12, rtol=1e-12
        )
        np.testing.assert_allclose(
            actual.lambda2, expected.lambda2, atol=1e-12, rtol=1e-12
        )
        assert actual.provenance["tensor_backend"] == prepared.backend
        assert actual.provenance["execution_owner_identity"] == prepared.identity
        assert actual.operator_actions > 0
        response = BoundCCSDResponse(prepared.bound, actual)
        weight = response.weight("foo", reference_identity=snapshot.identity)
        assert weight.parameter == "foo" and np.isfinite(weight.values).all()
    with pytest.raises(RuntimeError, match="closed"):
        prepared.solve(reference_identity=snapshot.identity)


def test_cuda_lambda_rejects_silent_backend_switch(
    monkeypatch: typing.Any, tmp_path: Path
) -> None:
    _fake_cuda_runtime(monkeypatch, bad_stage="shared_primal")
    snapshot, cc = _cc_state()
    compiler = CudaCompilerAdapter(Path("nvcc"), cuda_target_info("sm_120"))
    with pytest.raises(ResponseCompatibilityError, match="no CPU fallback"):
        PreparedCUDALambda(
            snapshot,
            cc,
            compiler,
            tmp_path,
            budget=ResourceBudget(host_bytes=512 << 20, device_bytes=1 << 30),
        )


@pytest.mark.parametrize(
    "budget",
    [None, ResourceBudget(host_bytes=1 << 20), ResourceBudget(device_bytes=1 << 20)],
)
def test_cuda_lambda_requires_explicit_two_sided_budget(
    monkeypatch: typing.Any, tmp_path: Path, budget: typing.Any
) -> None:
    _fake_cuda_runtime(monkeypatch)
    snapshot, cc = _cc_state()
    compiler = CudaCompilerAdapter(Path("nvcc"), cuda_target_info("sm_120"))
    with pytest.raises(ValueError, match="host and device budgets"):
        PreparedCUDALambda(
            snapshot,
            cc,
            compiler,
            tmp_path,
            budget=budget,
        )


_REAL = os.environ.get("VIBEQC_CC_LAMBDA_CUDA_TEST") == "1"


@pytest.mark.skipif(
    not _REAL, reason="requires explicitly allocated CUDA qualification"
)
def test_real_cuda_lambda_water_matches_cpu_and_reports_resources(
    tmp_path: Path,
) -> None:
    assert os.environ.get("SLURM_JOB_ID"), "CUDA Lambda qualification requires Slurm"
    snapshot, cc = _cc_state("h2o")
    cpu = BoundCCSDLambda(snapshot, cc).solve(reference_identity=snapshot.identity)
    nvcc = find_nvcc()
    assert nvcc is not None
    compiler = CudaCompilerAdapter(
        nvcc, cuda_target_info(os.environ.get("VIBEQC_TENSOR_ARCH", "sm_120"))
    )
    cache = Path(os.environ.get("VIBEQC_TENSOR_CACHE", tmp_path / "lambda-cuda"))
    with PreparedCUDALambda(
        snapshot,
        cc,
        compiler,
        cache,
        budget=ResourceBudget(host_bytes=1 << 30, device_bytes=2 << 30),
    ) as prepared:
        result = prepared.solve(reference_identity=snapshot.identity)
        np.testing.assert_allclose(result.lambda1, cpu.lambda1, atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(result.lambda2, cpu.lambda2, atol=2e-10, rtol=2e-10)
        assert result.independent_lambda_residual_norm <= 1.1e-9
        assert result.provenance["resource_peak_bytes"]["device"] > 0
        assert result.provenance["resource_peak_bytes"]["host"] > 0
        stages = result.provenance["transfer_metrics"]
        assert {"shared_primal", "independent_primal", "shared_transpose"} <= set(
            stages
        )
        assert (
            sum(row.get("observed_host_to_device_bytes", 0) for row in stages.values())
            > 0
        )
