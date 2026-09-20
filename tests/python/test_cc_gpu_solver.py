"""Experimental GPU solver: convergence/failure and replay state.

Preparation and ordinary-stream plan contracts run on CPU and are always
active. Real-device execution compiles the #148 physical equations through the
#146 planner and requires an explicitly allocated GPU validation window
(``VIBEQC_CC_CUDA_TEST=1`` plus ``VIBEQC_TENSOR_ARCH``/``VIBEQC_NVCC`` and a
tensor cache). Without that window the device tests are skipped, never faked.
"""

import os
import typing
from pathlib import Path

import numpy as np
import pytest
from test_cc_api import FixtureProvider, fixture_problem  # noqa: F401  (same fixtures)
from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.integral.cuda_target import cuda_target_info

from tools.vibeqc_cc.gpu_solver import PreparedGPUSolver, solve_gpu
from tools.vibeqc_cc.solver import SolverOptions


def test_gpu_solver_plans_compose_under_budget() -> None:
    from tools.vibeqc_cc.gpu_state import solver_plans

    s, _p, _meta, _ = fixture_problem("h2o")
    target = cuda_target_info("sm_90")
    primary, replay, diag = solver_plans(
        s.nocc, s.nmo - s.nocc, target, SolverOptions(), provider_peak_bytes=0
    )
    assert "next_t1" in primary.program.outputs
    assert "doubles_residual" in replay.program.outputs
    assert diag["combined_peak_bytes"] <= 256 << 20
    assert diag["combined_peak_bytes"] == primary.peak_bytes + replay.peak_bytes


def test_gpu_solver_requires_explicit_compiler_and_cache() -> None:
    s, p, _meta, _ = fixture_problem("h2")
    with pytest.raises(TypeError, match="CudaCompilerAdapter"):
        solve_gpu(s, p, compiler=None, cache=Path("."))
    with pytest.raises(TypeError, match="Path"):
        solve_gpu(
            s,
            p,
            compiler=CudaCompilerAdapter(Path("nvcc"), cuda_target_info("sm_90")),
            cache="not-a-path",
        )


def test_gpu_budget_rejection_precedes_integral_reads(
    monkeypatch: typing.Any, tmp_path: typing.Any
) -> None:
    """An impossible composed budget must not trigger AO-to-MO preparation."""
    s, p, _, _ = fixture_problem()
    monkeypatch.setattr(
        p, "get", lambda block: pytest.fail("integral read before budget gate")
    )
    with pytest.raises(ValueError, match="budget exhausted"):
        solve_gpu(
            s,
            p,
            compiler=CudaCompilerAdapter(Path("nvcc"), cuda_target_info("sm_120")),
            cache=tmp_path,
            provider_peak_bytes=SolverOptions().max_bytes,
        )


def test_replay_preparation_failure_releases_primary(
    monkeypatch: typing.Any, tmp_path: typing.Any
) -> None:
    """Retaining a failed constructor traceback must not retain device memory."""
    from tools.vibeqc_cc import gpu_solver

    closed = []

    class Executor:
        def __init__(
            self, plan: typing.Any, artifact: typing.Any, *, device: typing.Any
        ) -> None:
            if closed:
                raise RuntimeError("replay allocation failed")
            closed.append(False)

        def __enter__(self) -> typing.Any:
            return self

        def __exit__(self, *unused: object) -> None:
            closed[0] = True

    monkeypatch.setattr(gpu_solver, "compile_cuda", lambda *args: None)
    monkeypatch.setattr(gpu_solver, "PreparedCuda", Executor)
    s, p, _, _ = fixture_problem()
    with pytest.raises(RuntimeError, match="replay allocation") as failure:
        PreparedGPUSolver(
            s,
            p,
            CudaCompilerAdapter(Path("nvcc"), cuda_target_info("sm_120")),
            tmp_path,
        )
    assert failure.traceback is not None and closed == [True]


@pytest.mark.skipif(
    os.environ.get("VIBEQC_CC_CUDA_TEST") != "1",
    reason="requires explicitly allocated GPU validation window",
)
@pytest.mark.parametrize("name", ["h2", "h2o", "ch4"])
def test_real_device_gpu_solver_converges_and_replays(
    name: typing.Any, tmp_path: typing.Any
) -> None:
    from vibeqc.profiles import find_nvcc

    s, p, meta, _ = fixture_problem(name)
    compiler = CudaCompilerAdapter(
        find_nvcc(), cuda_target_info(os.environ["VIBEQC_TENSOR_ARCH"])
    )
    cache = Path(os.environ.get("VIBEQC_TENSOR_CACHE", tmp_path / "cache"))

    result = solve_gpu(
        s,
        p,
        compiler=compiler,
        cache=cache,
        options=SolverOptions(residual_tolerance=1e-10, energy_tolerance=1e-12),
    )
    assert result.converged, (name, result.reason, result.history[-1])
    assert result.provenance["backend"] == "cuda-fp64-ordinary-stream"
    assert result.provenance["residency"]
    transfers = result.provenance["transfer"]
    assert transfers["primary_evaluations"] == 2 * len(result.history) - 1
    assert transfers["independent_replay_evaluations"] >= 1
    assert abs(result.total_energy - meta["total_energy"]) <= 1e-8
    # Independent scaled physical residual gates (#138/#148).
    assert (
        max(
            result.history[-1]["independent_r1_max"],
            result.history[-1]["independent_r2_max"],
        )
        <= 1e-10
    )
    # Failure/replay state survives and re-reproduces on the CPU solver, within
    # cross-backend FP64 rounding (CPU BLAS vs cuBLAS, no exact bit guarantee).
    result.write(tmp_path / "gpu-state.json")
    from tools.replay_ccsd import replay

    reproduced = replay(tmp_path / "gpu-state.json")
    assert reproduced.status == result.status
    assert abs(reproduced.total_energy - result.total_energy) <= 1e-12
    np.testing.assert_allclose(reproduced.t1, result.t1, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(reproduced.t2, result.t2, atol=1e-10, rtol=1e-10)


@pytest.mark.skipif(
    os.environ.get("VIBEQC_CC_CUDA_TEST") != "1",
    reason="requires explicitly allocated GPU validation window",
)
def test_real_device_gpu_nonconvergence_is_an_explicit_failure_state(
    tmp_path: typing.Any,
) -> None:
    from vibeqc.profiles import find_nvcc

    s, p, _meta, _ = fixture_problem("ch4")
    compiler = CudaCompilerAdapter(
        find_nvcc(), cuda_target_info(os.environ["VIBEQC_TENSOR_ARCH"])
    )
    cache = Path(os.environ.get("VIBEQC_TENSOR_CACHE", tmp_path / "cache"))

    result = solve_gpu(
        s,
        p,
        compiler=compiler,
        cache=cache,
        options=SolverOptions(max_iterations=1),
    )
    assert result.status == "not_converged"
    assert not result.converged


@pytest.mark.skipif(
    os.environ.get("VIBEQC_CC_CUDA_TEST") != "1",
    reason="requires explicitly allocated GPU validation window",
)
def test_real_device_gpu_overflow_retains_serializable_failure(
    tmp_path: typing.Any,
) -> None:
    """Native arithmetic failure returns the last finite input for replay."""
    from vibeqc.profiles import find_nvcc

    s, p, _, a = fixture_problem("h2")
    result = solve_gpu(
        s,
        p,
        compiler=CudaCompilerAdapter(
            find_nvcc(), cuda_target_info(os.environ["VIBEQC_TENSOR_ARCH"])
        ),
        cache=Path(os.environ.get("VIBEQC_TENSOR_CACHE", tmp_path / "cache")),
        t1=np.full_like(a["t1"], 1e100),
        t2=np.full_like(a["t2"], 1e100),
    )
    assert result.status == "nonfinite" and not result.converged
    assert result.correlation_energy is None and not result.history
    assert np.isfinite(result.t1).all() and np.isfinite(result.t2).all()
    result.write(tmp_path / "overflow.json")
