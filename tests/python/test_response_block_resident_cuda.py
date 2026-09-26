"""Independent block-GMRES gates with actual resident vectors and raw counters."""

import os
import typing
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tools.vibeqc_posthf.fixtures import (
    fixture_snapshot,
    load_fixture,
    source_arguments,
)
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_response import (
    CudaDirectJKBackend,
    GMRESOptions,
    KrylovRecycleSpace,
    ResponseCompatibilityError,
    RHFResponseOperator,
    explicit_rhf_response_matrix,
    solve,
    solve_many,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESPONSE_CUDA_TEST") != "1",
    reason="requires an explicitly allocated real GPU",
)


def _independent_matrix(problem: typing.Any, ao: np.ndarray) -> np.ndarray:
    """Tiny dense MO oracle from the committed independent Libcint AO tensor."""
    c = problem.reference.coefficients
    mo = np.einsum("up,vq,wr,xs,uvwx->pqrs", c, c, c, c, ao, optimize=True)
    provider = SimpleNamespace(
        snapshot=problem.reference,
        get=lambda block: SimpleNamespace(to_host=lambda: mo[np.ix_(*block.slots)]),
    )
    return explicit_rhf_response_matrix(problem, provider)


def _forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    raise AssertionError("resident block solve crossed a host long-vector seam")


@pytest.mark.parametrize("name", ("h2", "lih", "water"))
@pytest.mark.parametrize("scale", (1.0, 1e-100))
@pytest.mark.parametrize("strategy", ("blocked", "sequential", "recycled"))
def test_resident_block_independent_solution_and_exact_transfers(
    name: str, scale: float, strategy: str, monkeypatch: typing.Any
) -> None:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    meta, arrays = load_fixture(name)
    with (
        NativeSource(**source_arguments(meta)) as source,
        CudaDirectJKBackend(source) as backend,
    ):
        reference = fixture_snapshot(meta, arrays)
        problem = RHFResponseOperator.build_problem(reference, backend)
        operator = RHFResponseOperator(problem, backend)
        matrix = _independent_matrix(problem, arrays["ao"])
        rng = np.random.default_rng(179)
        a, b = rng.normal(size=(2, problem.dimension))
        rhs = scale * np.column_stack((a, b, a + 2 * b, np.zeros_like(a)))
        expected = np.linalg.solve(matrix, rhs / scale)
        with backend.resident_response(
            problem, vector_slots=256, device_budget_bytes=16 << 20
        ) as resident:
            operator._krylov_engine = resident
            monkeypatch.setattr(operator, "apply", _forbidden)
            monkeypatch.setattr(backend, "coulomb_exchange", _forbidden)
            monkeypatch.setattr(source, "tile", _forbidden)
            monkeypatch.setattr(resident, "stack_host", _forbidden)
            calls: Counter = Counter()
            native_call = resident._call

            def count(name: str, *args: typing.Any) -> None:
                calls[name] += 1
                native_call(name, *args)

            monkeypatch.setattr(resident, "_call", count)
            before = resident.diagnostics
            answer = solve_many(
                operator,
                rhs,
                strategy=strategy,
                options=GMRESOptions(
                    rtol=1e-11, max_iterations=40, breakdown_tolerance=1e-14 * scale
                ),
                collect_basis=False,
                raise_on_failure=True,
            )
            after = resident.diagnostics
            np.testing.assert_allclose(
                answer.solution / scale, expected, atol=3e-9, rtol=3e-9
            )
            for column in range(rhs.shape[1]):
                residual = (
                    matrix @ (answer.solution[:, column] / scale)
                    - rhs[:, column] / scale
                )
                assert np.linalg.norm(residual) <= max(
                    1e-12, 1e-9 * np.linalg.norm(rhs[:, column] / scale)
                )
            assert answer.rank_deficient_rhs
            assert not resident._live
            assert not resident._retained
            assert calls["upload"] == rhs.shape[1]
            assert calls["download"] == rhs.shape[1]
            assert calls["apply"] == answer.operator_actions
            assert after["h2d_bytes"] - before["h2d_bytes"] == rhs.nbytes
            assert after["d2h_bytes"] - before["d2h_bytes"] == (
                answer.solution.nbytes
                + 8 * (calls["dot"] + calls["norm"])
                + 4 * calls["apply"]
            )
            assert after["owned_device_bytes"] == before["owned_device_bytes"]
            assert all(item.basis.shape[1] == 0 for item in answer.results)


def test_resident_recycle_identity_atomic_update_and_lifetime(
    monkeypatch: typing.Any,
) -> None:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    meta, arrays = load_fixture("water")
    with (
        NativeSource(**source_arguments(meta)) as source,
        CudaDirectJKBackend(source) as backend,
    ):
        reference = fixture_snapshot(meta, arrays)
        problem = RHFResponseOperator.build_problem(reference, backend)
        operator = RHFResponseOperator(problem, backend)
        matrix = _independent_matrix(problem, arrays["ao"])
        rhs = np.linspace(-0.3, 0.7, problem.dimension)
        with backend.resident_response(
            problem, vector_slots=128, device_budget_bytes=16 << 20
        ) as resident:
            operator._krylov_engine = resident
            with KrylovRecycleSpace(
                problem, max_vectors=3, vector_engine=resident
            ) as recycle:
                first = solve(
                    operator,
                    rhs,
                    recycle=recycle,
                    collect_basis=False,
                    raise_on_failure=True,
                )
                np.testing.assert_allclose(matrix @ first.solution, rhs, atol=1e-9)
                assert first.basis.shape[1] == 0
                assert recycle.storage_bytes > 0
                retained = set(resident._retained)
                assert resident._live == retained
                generation = recycle.generation
                retain = resident._retain

                def failure(vector: typing.Any) -> None:
                    retain(vector)
                    raise MemoryError("injected recycle publication failure")

                with monkeypatch.context() as patch:
                    patch.setattr(resident, "_retain", failure)
                    with pytest.raises(MemoryError, match="recycle publication"):
                        solve(operator, rhs, recycle=recycle, collect_basis=False)
                assert recycle.generation == generation
                assert resident._retained == resident._live == retained
                second = solve(
                    operator,
                    rhs,
                    recycle=recycle,
                    collect_basis=False,
                    raise_on_failure=True,
                )
                np.testing.assert_allclose(matrix @ second.solution, rhs, atol=1e-9)
                assert second.recycled_vectors == 3
                changed = replace(
                    problem, reference=replace(reference, generation_id="new")
                )
                with pytest.raises(ResponseCompatibilityError, match="stale"):
                    recycle.assert_compatible(changed)
                with backend.resident_response(
                    problem, vector_slots=128, device_budget_bytes=16 << 20
                ) as foreign:
                    operator._krylov_engine = foreign
                    before = foreign.diagnostics
                    with pytest.raises(
                        ResponseCompatibilityError, match="another resident"
                    ):
                        solve(operator, np.zeros(problem.dimension), recycle=recycle)
                    assert foreign.diagnostics == before
                operator._krylov_engine = resident
            assert not resident._live
            assert not resident._retained
        with pytest.raises(RuntimeError, match="closed"):
            solve(operator, np.zeros(problem.dimension), recycle=recycle)


def test_resident_block_capacity_and_exception_cleanup(monkeypatch: typing.Any) -> None:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    meta, arrays = load_fixture("water")
    with (
        NativeSource(**source_arguments(meta)) as source,
        CudaDirectJKBackend(source) as backend,
    ):
        reference = fixture_snapshot(meta, arrays)
        problem = RHFResponseOperator.build_problem(reference, backend)
        operator = RHFResponseOperator(problem, backend)
        rhs = np.ones((problem.dimension, 2))
        with backend.resident_response(
            problem, vector_slots=8, device_budget_bytes=16 << 20
        ) as small:
            operator._krylov_engine = small
            before = small.diagnostics
            rejected = solve_many(operator, rhs, strategy="blocked")
            assert not rejected.converged
            assert all(item.reason == "vector_slot_limit" for item in rejected.results)
            assert small.diagnostics == before
            assert not small._live
        with backend.resident_response(
            problem, vector_slots=128, device_budget_bytes=16 << 20
        ) as resident:
            operator._krylov_engine = resident
            apply = resident.apply
            retained = []

            def failing(*args: typing.Any) -> typing.Any:
                retained.append(apply(*args))
                raise RuntimeError("injected block action failure")

            with monkeypatch.context() as patch:
                patch.setattr(resident, "apply", failing)
                with pytest.raises(RuntimeError, match="injected block"):
                    solve_many(operator, rhs, strategy="blocked")
            assert retained and retained[0]._released
            assert not resident._live
            solve_many(operator, rhs, strategy="blocked", raise_on_failure=True)
            assert not resident._live
