"""Independent unrestricted CUDA response gates, run only under Slurm.

Committed Libcint/PySCF AO tensors are test-only oracles. Changing the molecular
charge/spin does not change these electron-repulsion integrals. Native SCF
supplies the actual open-shell reference; the explicit MO Hessian is assembled
independently of the response density/action implementation.
"""

import os
import typing
from dataclasses import replace

import numpy as np
import pytest

from tools.vibeqc_posthf.fixtures import load_fixture, source_arguments
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_response import (
    CudaSpinJKBackend,
    GMRESOptions,
    NativeJKBackend,
    UHFResponseOperator,
    solve,
    solve_many,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESPONSE_CUDA_TEST") != "1",
    reason="requires an explicitly allocated real GPU",
)


def _no_cpu(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
    raise AssertionError("CPU/ERI-tile fallback used by spin CUDA response")


def _inputs(name: str) -> typing.Any:
    """Remove one electron from the closed-shell integral fixture."""
    metadata, arrays = load_fixture(name)
    arguments = source_arguments(metadata)
    arguments["charge"] += 1
    arguments["multiplicity"] = 2
    return arguments, arrays


def _explicit_matrix(reference: typing.Any, eri: typing.Any) -> typing.Any:
    """Assemble spin MO Coulomb/exchange Hessian without response JVPs."""
    rotations = [
        (spin, i, a)
        for spin in ("alpha", "beta")
        for i in range(reference.nocc(spin))
        for a in range(reference.nocc(spin), reference.nbf)
    ]
    tensors = {}
    for spin in ("alpha", "beta"):
        for other in ("alpha", "beta"):
            left = getattr(reference, f"coefficients_{spin}")
            right = getattr(reference, f"coefficients_{other}")
            tensors[spin, other] = np.einsum(
                "pqrs,pa,qi,rj,sb->aijb",
                eri,
                left,
                left,
                right,
                right,
                optimize=True,
            )
    matrix = np.zeros((len(rotations), len(rotations)))
    for row, (spin, i, a) in enumerate(rotations):
        eps = getattr(reference, f"orbital_energies_{spin}")
        matrix[row, row] = eps[a] - eps[i]
        for col, (other, j, b) in enumerate(rotations):
            mo = tensors[spin, other]
            matrix[row, col] += 2 * mo[a, i, j, b]
            if spin == other:
                matrix[row, col] -= mo[a, j, i, b] + mo[a, b, i, j]
    return matrix


@pytest.mark.parametrize("approximation", ("exact", "density_fitted"))
@pytest.mark.parametrize("name", ("h2", "water", "f_heh"))
def test_spin_signed_raw_jk_against_independent_integrals(
    approximation: str,
    name: str,
    monkeypatch: typing.Any,
) -> None:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    arguments, arrays = _inputs(name)
    eri = arrays["ao" if approximation == "exact" else "df_ao"]
    with (
        NativeSource(**arguments) as source,
        CudaSpinJKBackend(
            source,
            approximation=approximation,
        ) as backend,
    ):
        rng = np.random.default_rng(179)
        densities = rng.normal(size=(2, source.nbf, source.nbf))
        densities = (densities + densities.swapaxes(1, 2)) / 2
        densities[0, 0, 0] = -2
        expected = (
            np.einsum("pqrs,rs->pq", eri, densities.sum(axis=0)),
            *np.einsum("prqs,xrs->xpq", eri, densities),
        )
        monkeypatch.setattr(source, "tile", _no_cpu)
        monkeypatch.setattr(source, "requests", _no_cpu)
        monkeypatch.setattr(NativeJKBackend, "coulomb_exchange", _no_cpu)
        actual = backend.spin_coulomb_exchange(*densities)
        np.testing.assert_allclose(actual, expected, atol=3e-9, rtol=3e-9)
        swapped = backend.spin_coulomb_exchange(*densities[::-1])
        np.testing.assert_allclose(
            swapped, (actual[0], actual[2], actual[1]), atol=3e-10
        )
        for scale in (0.0, -0.7, 1e-18):
            scaled = backend.spin_coulomb_exchange(*(scale * densities))
            # Tiny directions must survive: Fock-minus-hcore would fail here.
            np.testing.assert_allclose(
                scaled,
                scale * np.array(expected),
                atol=max(abs(scale), 1e-30) * 3e-9,
                rtol=3e-9,
            )
        before = backend.statistics["actions"]
        for bad in (
            np.full_like(densities[0], np.nan),
            densities[0].astype(complex) + 1j,
            densities[0][:1],
            np.triu(densities[0]),
        ):
            with pytest.raises(ValueError):
                backend.spin_coulomb_exchange(bad, densities[1])
        assert backend.statistics["actions"] == before
        np.testing.assert_allclose(
            backend.spin_coulomb_exchange(*densities), actual, atol=3e-10
        )
        diag = backend.diagnostics
        assert diag["provider"]["resolved"]["spin"] == "unrestricted"
        assert diag["provider"]["screening_tolerance"] == 0
        assert 0 < backend.device_resident_bytes <= 64 << 20
        assert not diag["gpu_resident_response"]
        diag["provider"]["backend"] = "tampered"
        assert backend.diagnostics["provider"]["backend"] == "cuda"


@pytest.mark.parametrize("approximation", ("exact", "density_fitted"))
@pytest.mark.parametrize("name", ("h2", "lih", "water"))
def test_native_spin_reference_actions_and_shared_multirhs(
    approximation: str,
    name: str,
    monkeypatch: typing.Any,
) -> None:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    arguments, arrays = _inputs(name)
    eri = arrays["ao" if approximation == "exact" else "df_ao"]
    with (
        NativeSource(**arguments) as source,
        CudaSpinJKBackend(
            source,
            approximation=approximation,
        ) as backend,
    ):
        reference, report = backend.export_reference(max_iterations=200)
        assert reference.scf_residual < 1e-8
        assert report["provider"]["backend"] == "cuda"
        # Check physical Focks with independent AO integrals, not only native
        # self-consistency. The hcore fixture is unchanged by ionization.
        densities = []
        for spin in ("alpha", "beta"):
            occupied = getattr(reference, f"coefficients_{spin}")[
                :, : reference.nocc(spin)
            ]
            densities.append(occupied @ occupied.T)
        total_j = np.einsum("pqrs,rs->pq", eri, sum(densities))
        for spin, density in zip(("alpha", "beta"), densities, strict=True):
            expected_fock = (
                arrays["conventional_h"]
                + total_j
                - np.einsum("prqs,rs->pq", eri, density)
            )
            np.testing.assert_allclose(
                getattr(reference, f"fock_{spin}"), expected_fock, atol=2e-8, rtol=2e-8
            )
        problem = UHFResponseOperator.build_problem(reference, backend)
        operator = UHFResponseOperator(problem, backend)
        explicit = _explicit_matrix(reference, eri)
        rng = np.random.default_rng(180)
        vector, other = rng.normal(size=(2, problem.dimension))
        monkeypatch.setattr(source, "tile", _no_cpu)
        monkeypatch.setattr(source, "requests", _no_cpu)
        monkeypatch.setattr(NativeJKBackend, "coulomb_exchange", _no_cpu)
        monkeypatch.setattr(backend._plan, "solve", _no_cpu)
        before = backend.statistics["actions"]
        np.testing.assert_allclose(
            operator.apply(vector), explicit @ vector, atol=3e-9, rtol=3e-9
        )
        assert backend.statistics["actions"] == before + 1
        np.testing.assert_allclose(
            operator.apply_transpose(vector), explicit.T @ vector, atol=3e-9, rtol=3e-9
        )
        assert operator.dot_identity(vector, other) < 3e-10
        if name == "h2":
            assert problem.layout.block_dimension("beta") == 0
        else:
            split = problem.layout.block_dimension("alpha")
            assert np.max(np.abs(explicit[:split, split:])) > 1e-5
        rhs = np.column_stack((vector, other, -2 * vector))
        expected = np.linalg.solve(explicit, rhs)
        options = GMRESOptions(rtol=1e-11, atol=1e-12, restart=32, max_iterations=180)
        for strategy in ("sequential", "blocked", "recycled"):
            result = solve_many(operator, rhs, strategy=strategy, options=options)
            assert result.converged
            np.testing.assert_allclose(result.solution, expected, atol=2e-7, rtol=2e-8)
            assert np.max(np.abs(explicit @ result.solution - rhs)) < 1e-9
            assert max(item.residual_norm for item in result.results) < 1e-9
        before = backend.statistics["actions"]
        limited = solve(operator, vector, options=GMRESOptions(max_workspace_bytes=1))
        assert not limited.converged and limited.reason == "workspace_limit"
        assert backend.statistics["actions"] == before
        bad = replace(reference, geometry_hash="changed")
        with pytest.raises(ValueError, match="geometry_hash mismatch"):
            UHFResponseOperator(replace(problem, reference=bad), backend)
        backend.close()
        with pytest.raises(RuntimeError, match="closed"):
            solve(operator, np.zeros(problem.dimension))
        for strategy in ("sequential", "blocked", "recycled"):
            with pytest.raises(RuntimeError, match="closed"):
                solve_many(
                    operator, np.zeros((problem.dimension, 2)), strategy=strategy
                )


@pytest.mark.parametrize("approximation", ("exact", "density_fitted"))
def test_spin_budget_failure_and_native_failure_leave_reusable_plan(
    approximation: str,
) -> None:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    arguments, _ = _inputs("lih")
    with NativeSource(**arguments) as source:
        # The prepared DF planner diagnoses an impossible requested envelope
        # as invalid input; the direct allocator reports out-of-memory.
        with pytest.raises((MemoryError, ValueError)):
            CudaSpinJKBackend(
                source, approximation=approximation, device_budget_bytes=1
            )
        with CudaSpinJKBackend(source, approximation=approximation) as backend:
            with pytest.raises(RuntimeError):
                backend.export_reference(max_iterations=1)
            reference, _ = backend.export_reference(max_iterations=200)
            assert reference.scf_residual < 1e-8
            source.close()
            with pytest.raises(RuntimeError, match="closed"):
                backend.spin_coulomb_exchange(
                    np.eye(backend.nbf), np.zeros((backend.nbf, backend.nbf))
                )


def test_spin_df_identity_binds_metric_policy_and_occupations() -> None:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    arguments, _ = _inputs("lih")
    with (
        NativeSource(**arguments) as source,
        CudaSpinJKBackend(
            source,
            approximation="density_fitted",
        ) as backend,
        CudaSpinJKBackend(
            source,
            approximation="density_fitted",
            metric_relative_threshold=1e-9,
        ) as changed,
    ):
        reference, _ = backend.export_reference(max_iterations=200)
        with pytest.raises(ValueError, match="hamiltonian_id mismatch"):
            changed.validate_reference(reference)
        bad = replace(reference, occupations_beta=np.zeros(source.nbf))
        with pytest.raises(ValueError, match="occupations mismatch"):
            backend.validate_reference(bad)
        problem = UHFResponseOperator.build_problem(reference, backend)
        with pytest.raises(ValueError, match="operator_identity"):
            UHFResponseOperator(problem, changed)
