"""Bounded GMRES, block solves, recycling and explicit failure semantics."""

from dataclasses import replace

import numpy as np
import pytest

from tools.vibeqc_posthf.fixtures import fixture_snapshot, load_fixture
from tools.vibeqc_response import (
    DenseAOResponseBackend,
    DiagonalPreconditioner,
    GMRESOptions,
    KrylovRecycleSpace,
    ResponseCompatibilityError,
    ResponseSolveError,
    ResponseUnsupported,
    RHFResponseOperator,
    solve,
    solve_many,
)
from tools.vibeqc_response.problem import ResponseProblem


class _MatrixOperator:
    """Small standalone operator with the public response-operator contract."""

    def __init__(self, matrix, dimension):
        self.matrix = np.asarray(matrix, dtype=float)
        self.dimension = dimension
        self.problem = type(
            "Problem",
            (),
            {
                "compatibility_identity": "synthetic",
                "validate_rhs": self._validate_rhs,
            },
        )()
        self.statistics = {"actions": 0}

    def _validate_rhs(self, values):
        value = np.asarray(values, dtype=float)
        if value.ndim == 1:
            value = value[:, None]
        if value.shape[0] != self.dimension or not np.isfinite(value).all():
            raise ValueError("invalid synthetic RHS")
        return value

    def apply(self, vector):
        self.statistics["actions"] += 1
        return self.matrix @ np.asarray(vector, dtype=float)

    def apply_transpose(self, vector):
        return self.matrix.T @ np.asarray(vector, dtype=float)


def _synthetic_operator(size=12, seed=179):
    diagonal = np.linspace(1.0, 4.0, size)
    matrix = np.diag(diagonal)
    matrix[0, 1] = 0.2
    matrix[1, 0] = -0.1
    return _MatrixOperator(matrix, size)


def test_gmres_true_residual_and_dot_identity():
    operator = _synthetic_operator()
    rhs = np.linspace(-1.0, 1.0, operator.dimension)
    result = solve(
        operator,
        rhs,
        options=GMRESOptions(rtol=1e-12, restart=8, max_iterations=100),
    )
    assert result.converged
    assert result.residual_norm <= 1e-11
    assert result.history[-1] == pytest.approx(result.residual_norm)
    np.testing.assert_allclose(
        operator.apply(result.solution), rhs, atol=2e-11, rtol=2e-11
    )
    left = np.arange(operator.dimension, dtype=float)
    right = np.ones(operator.dimension)
    lhs = np.dot(left, operator.apply(right))
    rhs_dot = np.dot(operator.apply_transpose(left), right)
    assert abs(lhs - rhs_dot) < 1e-12


def test_deliberate_nonconvergence_and_singular_operator_do_not_claim_success():
    operator = _synthetic_operator()
    rhs = np.ones(operator.dimension)
    result = solve(
        operator,
        rhs,
        options=GMRESOptions(
            rtol=1e-15, restart=2, max_iterations=1, stagnation_window=100
        ),
    )
    assert not result.converged
    assert result.reason in ("max_iterations", "stagnation", "breakdown")
    with pytest.raises(ResponseSolveError):
        result.require_converged()
    singular = _MatrixOperator(np.zeros((4, 4)), 4)
    failed = solve(
        singular,
        np.ones(4),
        options=GMRESOptions(rtol=1e-12, restart=4, max_iterations=10),
    )
    assert not failed.converged
    assert failed.reason in ("singular", "breakdown", "max_iterations")
    with pytest.raises(ResponseSolveError):
        failed.require_converged()


def test_workspace_limit_is_reported_before_operator_application():
    operator = _synthetic_operator()
    result = solve(
        operator,
        np.ones(operator.dimension),
        options=GMRESOptions(
            restart=8,
            max_iterations=10,
            max_workspace_bytes=8,
        ),
    )
    assert not result.converged
    assert result.reason == "workspace_limit"
    assert operator.statistics["actions"] == 0


def test_diagonal_preconditioner_is_separate_and_rejects_zero_denominator():
    operator = _synthetic_operator()
    diagonal = np.diag(operator.matrix)
    preconditioner = DiagonalPreconditioner(diagonal)
    rhs = np.linspace(-1.0, 1.0, operator.dimension)
    result = solve(
        operator,
        rhs,
        preconditioner=preconditioner,
        options=GMRESOptions(rtol=1e-12, restart=4, max_iterations=20),
    )
    assert result.converged
    with pytest.raises(ValueError, match="no denominator was clamped"):
        DiagonalPreconditioner([1.0, 0.0, 2.0])
    with pytest.raises(ValueError, match="blocked GMRES"):
        solve_many(
            operator,
            np.column_stack((rhs, rhs)),
            strategy="blocked",
            preconditioner=preconditioner,
        )


def test_failed_solve_does_not_poison_recycle_space():
    singular = _MatrixOperator(np.zeros((4, 4)), 4)
    space = KrylovRecycleSpace(singular.problem, max_vectors=2)
    result = solve(
        singular,
        np.ones(4),
        recycle=space,
        options=GMRESOptions(rtol=1e-12, restart=4, max_iterations=4),
    )
    assert not result.converged
    assert space._vectors == []


def test_block_solver_preserves_tiny_nonzero_rhs_and_indefinite_operator():
    tiny = _MatrixOperator(np.eye(2), 2)
    rhs = np.array([1e-13, 0.0])
    result = solve_many(
        tiny,
        rhs,
        strategy="blocked",
        options=GMRESOptions(rtol=1e-10, atol=0.0, restart=2, max_iterations=4),
    )
    assert result.converged
    np.testing.assert_allclose(result.solution[:, 0], rhs, atol=1e-25, rtol=1e-12)
    indefinite = _MatrixOperator(np.array([[0.0, 1.0], [1.0, 0.0]]), 2)
    result = solve_many(
        indefinite,
        np.array([1.0, 0.0]),
        strategy="blocked",
        options=GMRESOptions(rtol=1e-12, restart=2, max_iterations=4),
    )
    assert result.converged
    assert result.results[0].residual_norm < 1e-12


def test_block_workspace_preflight_accounts_for_initial_basis_and_retained_results():
    operator = _MatrixOperator(np.eye(100), 100)
    result = solve_many(
        operator,
        np.eye(100),
        strategy="blocked",
        options=GMRESOptions(
            rtol=1e-10,
            restart=1,
            max_iterations=1,
            max_workspace_bytes=2500,
        ),
    )
    assert not result.converged
    assert all(item.reason == "workspace_limit" for item in result.results)


def test_sequential_workspace_budget_charges_retained_results():
    operator = _MatrixOperator(np.eye(4), 4)
    rhs = np.column_stack((np.ones(4), np.arange(1.0, 5.0)))
    options = GMRESOptions(rtol=1e-12, restart=2, max_iterations=4)
    single = solve(operator, rhs[:, 0], options=options)
    retained = single.solution.nbytes + single.basis.nbytes
    budget = rhs.nbytes + single.workspace_bytes + retained - 1
    result = solve_many(
        operator,
        rhs,
        strategy="sequential",
        options=replace(options, max_workspace_bytes=budget),
    )
    assert result.results[0].converged
    assert result.results[1].reason == "workspace_limit"
    assert result.peak_workspace_bytes >= single.workspace_bytes


def test_recycled_workspace_budget_sums_results_recycle_and_projection():
    operator = _MatrixOperator(np.eye(4), 4)
    rhs = np.column_stack((np.ones(4), np.arange(1.0, 5.0)))
    options = GMRESOptions(rtol=1e-12, restart=2, max_iterations=4)
    probe_space = KrylovRecycleSpace(operator.problem, max_vectors=4)
    probe = solve_many(
        operator,
        rhs[:, :1],
        strategy="recycled",
        recycle=probe_space,
        options=options,
    )
    assert probe.converged
    probe_result = probe.results[0]
    retained = probe_result.solution.nbytes + probe_result.basis.nbytes
    recycle_bytes = probe_space.storage_bytes
    projection_bytes = probe_space.projection_bytes(operator.dimension)
    # One byte below the additive live-storage bound: the next recycled solve
    # must be rejected before it applies the operator.
    budget = (
        probe_result.workspace_bytes + retained + recycle_bytes + projection_bytes - 1
    )
    space = KrylovRecycleSpace(operator.problem, max_vectors=4)
    result = solve_many(
        operator,
        rhs,
        strategy="recycled",
        recycle=space,
        options=replace(options, max_workspace_bytes=budget),
    )
    assert result.results[0].converged
    assert result.results[1].reason == "workspace_limit"
    assert result.results[1].operator_actions == 0
    assert result.operator_actions == result.results[0].operator_actions


def test_relative_residual_is_relative_for_sub_unit_rhs_norms():
    singular = _MatrixOperator(np.zeros((2, 2)), 2)
    rhs = np.array([1e-13, 0.0])
    scalar = solve(
        singular,
        rhs,
        options=GMRESOptions(rtol=1e-10, atol=0.0, restart=2, max_iterations=4),
    )
    assert not scalar.converged
    assert scalar.relative_residual == pytest.approx(1.0)

    blocked = solve_many(
        singular,
        np.column_stack((rhs, np.zeros(2))),
        strategy="blocked",
        options=GMRESOptions(rtol=1e-10, atol=0.0, restart=2, max_iterations=4),
    )
    assert blocked.results[0].relative_residual == pytest.approx(1.0)
    assert blocked.results[1].relative_residual == 0.0


def test_zero_rhs_relative_residual_convention():
    identity = _MatrixOperator(np.eye(2), 2)
    exact = solve(identity, np.zeros(2))
    assert exact.converged
    assert exact.relative_residual == 0.0
    nonzero_residual = solve(
        identity,
        np.zeros(2),
        initial_guess=np.ones(2),
        options=GMRESOptions(rtol=1e-10, atol=0.0, restart=2, max_iterations=1),
    )
    assert not nonzero_residual.converged
    assert nonzero_residual.relative_residual == float("inf")


def test_true_residual_checkpoint_at_restart_and_iteration_limit():
    operator = _MatrixOperator(np.diag([1.0, 2.0]), 2)
    result = solve(
        operator,
        np.ones(2),
        options=GMRESOptions(
            rtol=1e-12,
            restart=2,
            max_iterations=2,
            true_residual_every=3,
        ),
    )
    assert result.converged
    assert result.residual_norm < 1e-11


def test_transport_validates_destination_dimension_and_storage_bound():
    source = _MatrixOperator(np.eye(1), 1)
    destination = _MatrixOperator(np.eye(2), 2)
    destination.problem.dimension = 2
    space = KrylovRecycleSpace(source.problem, max_vectors=1, max_bytes=32)
    space._vectors = [np.array([1.0])]
    transported = space.transport(destination.problem, lambda _: np.array([1.0, 0.0]))
    assert transported._vectors[0].shape == (2,)
    with pytest.raises(ValueError, match="invalid vector"):
        space.transport(destination.problem, lambda _: np.array([1.0]))


def test_near_degenerate_reference_is_diagnosed_without_clamping():
    meta, arrays = load_fixture("h2")
    reference = fixture_snapshot(meta, arrays)
    energies = reference.orbital_energies.copy()
    energies[1] = energies[0] + 1e-10
    fock = (
        reference.overlap
        @ reference.coefficients
        @ np.diag(energies)
        @ reference.coefficients.T
        @ reference.overlap
    )
    near = type(reference)(
        reference.overlap,
        reference.hcore,
        fock,
        reference.coefficients,
        energies,
        reference.occupations,
        reference.electron_count,
        reference.reference_energy,
        reference.scf_residual,
        reference.geometry_hash,
        reference.basis_hash,
        reference.generation_id,
        hamiltonian_id=reference.hamiltonian_id,
        hf_backend=reference.hf_backend,
    )
    problem = ResponseProblem.from_reference(
        near, method="rhf", operator_identity="synthetic"
    )
    assert problem.diagnostics["near_degenerate"]
    with pytest.raises(ResponseCompatibilityError, match="response problem"):
        # Deliberately compare the same numerical layout with a different
        # operator identity; the diagnostic gate itself is checked below.
        problem.assert_compatible(
            ResponseProblem.from_reference(
                near, method="rhf", operator_identity="different"
            )
        )
    with pytest.raises(ResponseUnsupported, match="near-degenerate"):
        problem.require_stable()


def test_multi_rhs_strategies_and_rank_deficient_rhs():
    operator = _synthetic_operator()
    rng = np.random.default_rng(179)
    rhs = rng.normal(size=(operator.dimension, 3))
    sequential = solve_many(
        operator,
        rhs,
        strategy="sequential",
        options=GMRESOptions(rtol=1e-12, restart=8, max_iterations=100),
    )
    blocked = solve_many(
        operator,
        rhs,
        strategy="blocked",
        options=GMRESOptions(rtol=1e-12, restart=8, max_iterations=100),
    )
    recycled = solve_many(
        operator,
        rhs,
        strategy="recycled",
        options=GMRESOptions(rtol=1e-12, restart=8, max_iterations=100),
    )
    assert sequential.converged and blocked.converged and recycled.converged
    np.testing.assert_allclose(sequential.solution, blocked.solution, atol=1e-10)
    np.testing.assert_allclose(sequential.solution, recycled.solution, atol=1e-10)
    assert all(result.recycled_vectors > 0 for result in recycled.results[1:])
    dependent = np.column_stack((rhs[:, 0], rhs[:, 0], rhs[:, 1]))
    dependent_result = solve_many(
        operator,
        dependent,
        strategy="blocked",
        options=GMRESOptions(rtol=1e-12, restart=8, max_iterations=100),
    )
    assert dependent_result.converged
    assert dependent_result.rank_deficient_rhs
    np.testing.assert_allclose(
        dependent_result.solution[:, 0], dependent_result.solution[:, 1], atol=1e-10
    )


def test_stale_recycle_space_fails_closed_and_explicit_transport_works():
    meta, arrays = load_fixture("h2")
    reference = fixture_snapshot(meta, arrays)
    backend = DenseAOResponseBackend(arrays["ao"])
    problem = RHFResponseOperator.build_problem(reference, backend)
    operator = RHFResponseOperator(problem, backend)
    space = KrylovRecycleSpace(problem, max_vectors=4)
    solve(
        operator,
        np.ones(problem.dimension),
        recycle=space,
        options=GMRESOptions(rtol=1e-12, restart=6, max_iterations=50),
    )
    changed = RHFResponseOperator.build_problem(
        reference, DenseAOResponseBackend(arrays["ao"] * 1.0000001)
    )
    # A different dense backend has a different operator identity even though
    # the numerical ERI happens to be equal.
    with pytest.raises(ResponseCompatibilityError, match="stale"):
        space.assert_compatible(changed)
    transported = space.transport(changed, lambda vector: vector)
    transported.assert_compatible(changed)


def test_direct_recycling_rejects_storage_before_projection(monkeypatch):
    """The scalar API must enforce the same reservation as solve_many."""
    operator = _MatrixOperator(np.eye(100), 100)
    space = KrylovRecycleSpace(operator.problem, max_vectors=100)
    space._vectors = [column.copy() for column in np.eye(100)]

    def unexpected_projection(*args):
        pytest.fail("projection ran before the recycle storage preflight")

    monkeypatch.setattr(space, "initial_guess", unexpected_projection)
    result = solve(
        operator,
        np.ones(100),
        recycle=space,
        options=GMRESOptions(restart=1, max_iterations=1, max_workspace_bytes=8000),
    )
    assert result.reason == "workspace_limit"
    assert result.workspace_bytes > space.storage_bytes > 8000
    assert result.operator_actions == 0
    assert operator.statistics["actions"] == 0


def test_recycle_replacement_is_reserved_before_solving(monkeypatch):
    """An empty recycle space still needs capacity to publish its replacement."""
    operator = _MatrixOperator(np.eye(100), 100)
    rhs = np.ones(100)
    options = GMRESOptions(restart=1, max_iterations=1)
    scalar = solve(operator, rhs, options=options)
    space = KrylovRecycleSpace(operator.problem, max_vectors=8)
    result = solve(
        operator,
        rhs,
        recycle=space,
        options=replace(options, max_workspace_bytes=scalar.workspace_bytes),
    )
    assert result.reason == "workspace_limit"
    assert result.operator_actions == 0
    assert space.generation == 0
    assert space.storage_bytes == 0


@pytest.mark.parametrize("strategy", ["sequential", "blocked", "recycled"])
def test_successful_peak_fits_exact_budget(strategy):
    operator = _MatrixOperator(np.diag([1.0, 2.0, 3.0, 4.0]), 4)
    rhs = np.column_stack((np.ones(4), np.arange(1.0, 5.0)))
    options = GMRESOptions(restart=4, max_iterations=8)
    probe = solve_many(operator, rhs, strategy=strategy, options=options)
    assert probe.converged
    exact = solve_many(
        operator,
        rhs,
        strategy=strategy,
        options=replace(options, max_workspace_bytes=probe.peak_workspace_bytes),
    )
    assert exact.converged
    assert exact.peak_workspace_bytes <= probe.peak_workspace_bytes
    np.testing.assert_allclose(
        exact.solution, np.linalg.solve(operator.matrix, rhs), atol=1e-12
    )
    below = solve_many(
        operator,
        rhs,
        strategy=strategy,
        options=replace(options, max_workspace_bytes=probe.peak_workspace_bytes - 1),
    )
    assert not below.converged
    assert any(result.reason == "workspace_limit" for result in below.results)


def test_sequential_peak_counts_current_result_once():
    operator = _MatrixOperator(np.eye(100), 100)
    rhs = np.ones((100, 1))
    options = GMRESOptions(restart=1, max_iterations=1)
    single = solve(operator, rhs[:, 0], options=options)
    budget = single.workspace_bytes + rhs.nbytes
    result = solve_many(
        operator, rhs, options=replace(options, max_workspace_bytes=budget)
    )
    assert result.converged
    assert result.peak_workspace_bytes == budget


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_true_residual_norm_does_not_underflow_or_overflow(scale):
    """A finite nonzero RHS cannot be accepted with the initial zero solution."""
    operator = _MatrixOperator(np.eye(2), 2)
    rhs = np.array([scale, 0.0])
    result = solve(operator, rhs)
    assert result.converged
    assert result.reason != "initial_residual"
    np.testing.assert_allclose(result.solution / scale, [1.0, 0.0], atol=1e-12)
    for matrix in (np.eye(2), np.zeros((2, 2))):
        blocked = solve_many(_MatrixOperator(matrix, 2), rhs, strategy="blocked")
        actual_relative = np.linalg.norm(
            matrix @ (blocked.solution[:, 0] / scale) - [1.0, 0.0]
        )
        if blocked.converged:
            assert actual_relative <= GMRESOptions().rtol
        else:
            assert blocked.results[0].relative_residual == pytest.approx(
                actual_relative
            )


def test_unrepresentable_rhs_norm_is_rejected():
    with pytest.raises(ValueError, match="norm overflows"):
        solve(_MatrixOperator(np.eye(2), 2), np.full(2, np.finfo(float).max))
