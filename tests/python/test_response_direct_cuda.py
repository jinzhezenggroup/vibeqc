"""Independent numerical/device gates for direct CUDA RHF J/K response."""

import os
import typing
from dataclasses import replace

import numpy as np
import pytest

from tools.vibeqc_posthf.export import export_rhf
from tools.vibeqc_posthf.fixtures import (
    fixture_snapshot,
    load_fixture,
    source_arguments,
)
from tools.vibeqc_posthf.providers import ConventionalProvider
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_response import (
    CudaDirectJKBackend,
    DenseAOResponseBackend,
    GMRESOptions,
    NativeJKBackend,
    RHFResponseOperator,
    explicit_rhf_response_matrix,
    finite_rotation_jvp,
    solve_many,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESPONSE_CUDA_TEST") != "1",
    reason="requires an explicitly allocated real GPU",
)


def _no_cpu(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    raise AssertionError("CPU/ERI-tile fallback used by CUDA J/K response")


@pytest.mark.parametrize("name", ["h2", "lih", "water", "f_heh"])
def test_cuda_direct_signed_raw_jk_match_independent_ao_integrals(
    name: typing.Any, monkeypatch: typing.Any
) -> None:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    meta, arrays = load_fixture(name)
    with (
        NativeSource(**source_arguments(meta)) as source,
        CudaDirectJKBackend(source) as backend,
    ):
        n = source.nbf
        rng = np.random.default_rng(180)
        d = rng.normal(size=(n, n))
        d = (d + d.T) / 2
        d[0, 0] = -2.0
        expected_j = np.einsum("pqrs,rs->pq", arrays["ao"], d)
        expected_k = np.einsum("prqs,rs->pq", arrays["ao"], d)
        monkeypatch.setattr(source, "tile", _no_cpu)
        monkeypatch.setattr(source, "requests", _no_cpu)
        monkeypatch.setattr(NativeJKBackend, "coulomb_exchange", _no_cpu)
        j, k = backend.coulomb_exchange(d)
        np.testing.assert_allclose(j, expected_j, atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(k, expected_k, atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(j, j.T, atol=2e-11, rtol=2e-11)
        np.testing.assert_allclose(k, k.T, atol=2e-11, rtol=2e-11)
        for scale in (0.0, -0.7, 1.3):
            sj, sk = backend.coulomb_exchange(scale * d)
            np.testing.assert_allclose(sj, scale * j, atol=2e-10, rtol=2e-10)
            np.testing.assert_allclose(sk, scale * k, atol=2e-10, rtol=2e-10)
        with pytest.raises(ValueError):
            backend.coulomb_exchange(np.full_like(d, np.nan))
        np.testing.assert_array_equal(backend.coulomb_exchange(d)[0], j)
        assert backend.diagnostics["provider"]["screening_tolerance"] == 0
        assert backend.diagnostics["provider"]["auxiliary_rank"] == 0
        assert backend.device_resident_bytes > 0


@pytest.mark.parametrize("name", ["h2", "lih", "water"])
def test_cuda_direct_cphf_action_and_all_shared_multirhs_strategies(
    name: typing.Any, monkeypatch: typing.Any
) -> None:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    meta, arrays = load_fixture(name)
    with NativeSource(**source_arguments(meta)) as source:
        reference = fixture_snapshot(meta, arrays)
        with CudaDirectJKBackend(source) as backend:
            problem = RHFResponseOperator.build_problem(reference, backend)
            operator = RHFResponseOperator(problem, backend)
            with ConventionalProvider(reference, source) as provider:
                explicit = explicit_rhf_response_matrix(problem, provider)
            rng = np.random.default_rng(180)
            x = rng.normal(size=problem.dimension)
            y = rng.normal(size=problem.dimension)
            finite = finite_rotation_jvp(
                problem, DenseAOResponseBackend(arrays["ao"]), x, step=1e-6
            )
            monkeypatch.setattr(source, "tile", _no_cpu)
            monkeypatch.setattr(source, "requests", _no_cpu)
            monkeypatch.setattr(NativeJKBackend, "coulomb_exchange", _no_cpu)
            action = operator.apply(x)
            np.testing.assert_allclose(action, explicit @ x, atol=2e-10, rtol=2e-10)
            np.testing.assert_allclose(action, finite, atol=3e-8, rtol=3e-8)
            assert operator.dot_identity(x, y) < 1e-11
            np.testing.assert_allclose(
                operator.apply_transpose(x), explicit.T @ x, atol=2e-10, rtol=2e-10
            )
            rhs = rng.normal(size=(problem.dimension, 2))
            options = GMRESOptions(
                rtol=1e-11, atol=1e-12, restart=16, max_iterations=120
            )
            expected = np.linalg.solve(explicit, rhs)
            for strategy in ("sequential", "blocked", "recycled"):
                result = solve_many(operator, rhs, strategy=strategy, options=options)
                assert result.converged
                np.testing.assert_allclose(
                    result.solution, expected, atol=2e-8, rtol=2e-8
                )
                np.testing.assert_allclose(
                    explicit @ result.solution, rhs, atol=1e-9, rtol=1e-9
                )
                assert max(item.residual_norm for item in result.results) < 1e-9
            bad = replace(reference, geometry_hash="different-geometry")
            with pytest.raises(ValueError, match="geometry_hash mismatch"):
                RHFResponseOperator(replace(problem, reference=bad), backend)
            assert not backend.diagnostics["gpu_resident_response"]


def test_native_rhf_snapshot_connects_to_direct_cuda_response(
    monkeypatch: typing.Any,
) -> None:
    """A real VibeQC SCF state, not only an external/synthetic fixture snapshot."""
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    atoms = [(1, (0.0, 0.0, 0.0)), (1, (0.0, 0.0, 1.4))]
    with NativeSource(atoms) as source:
        reference, _ = export_rhf(source, backend="cpu", tolerance=1e-12)
        with CudaDirectJKBackend(source) as backend:
            problem = RHFResponseOperator.build_problem(reference, backend)
            operator = RHFResponseOperator(problem, backend)
            with ConventionalProvider(reference, source) as provider:
                explicit = explicit_rhf_response_matrix(problem, provider)
            monkeypatch.setattr(source, "tile", _no_cpu)
            monkeypatch.setattr(source, "requests", _no_cpu)
            np.testing.assert_allclose(
                operator.apply(np.ones(problem.dimension)),
                explicit.sum(axis=1),
                atol=2e-10,
                rtol=2e-10,
            )
            source.close()
            with pytest.raises(RuntimeError, match="closed"):
                operator.apply(np.ones(problem.dimension))


def test_cuda_direct_impossible_budget_fails_and_new_plan_replays() -> None:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    with NativeSource([(1, (0, 0, 0)), (1, (0, 0, 1.4))]) as source:
        with pytest.raises(MemoryError):
            CudaDirectJKBackend(source, device_budget_bytes=1)
        with CudaDirectJKBackend(source) as backend:
            assert np.isfinite(backend.coulomb_exchange(np.eye(2))[0]).all()
