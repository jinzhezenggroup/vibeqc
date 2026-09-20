"""Bounded true-residual GMRES, block solves and safe Krylov recycling."""

from __future__ import annotations

import time
import typing
from contextlib import nullcontext
from dataclasses import dataclass, field, replace

import numpy as np
from vibeqc.profiles import canonical_hash

from tools.vibeqc_posthf.reference import immutable

from .problem import ResponseCompatibilityError, ResponseSolveError


def _vector_norm(value: typing.Any) -> typing.Any:
    """Scale before squaring so finite tiny/large vectors cannot look solved."""
    values = np.asarray(value, dtype=np.float64)
    scale = float(np.max(np.abs(values), initial=0.0))
    if not np.isfinite(scale):
        raise ValueError("response norm requires finite vector values")
    if scale == 0.0:
        return 0.0
    scaled = (values / scale).reshape(-1)
    result = scale * float(np.sqrt(np.dot(scaled, scaled)))
    if not np.isfinite(result):
        raise ValueError("response vector norm overflows FP64")
    return result


def _relative_residual(residual_norm: typing.Any, rhs_norm: typing.Any) -> typing.Any:
    """Return ``||r|| / ||b||`` with an explicit zero-RHS convention.

    A zero RHS has no scale, so its relative residual is zero for an exactly
    zero residual and ``inf`` otherwise.  This keeps the public diagnostic
    aligned with the convergence test instead of falling back to an absolute
    residual whenever ``||b|| < 1``.
    """
    if rhs_norm > 0.0:
        return float(residual_norm) / float(rhs_norm)
    return 0.0 if residual_norm == 0.0 else float("inf")


def _orthonormal_basis(
    matrix: typing.Any, *, tolerance: typing.Any = 1e-12, max_columns: typing.Any = None
) -> typing.Any:
    """Return an orthonormal basis for the finite column range of ``matrix``."""
    value = np.asarray(matrix, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("basis input must be a finite rank-two matrix")
    if value.shape[1] == 0:
        return np.empty((value.shape[0], 0)), 0
    left, singular, _ = np.linalg.svd(value, full_matrices=False)
    if not len(singular):
        return np.empty((value.shape[0], 0)), 0
    cutoff = max(tolerance, tolerance * float(singular[0]))
    rank = int(np.count_nonzero(singular > cutoff))
    if max_columns is not None:
        rank = min(rank, max_columns)
    return left[:, :rank], rank


def _orthogonalize_against(
    vectors: typing.Any,
    value: typing.Any,
    *,
    reorthogonalize: typing.Any = 2,
    tolerance: typing.Any = 1e-14,
) -> typing.Any:
    """Modified Gram-Schmidt with reorthogonalization and an explicit breakdown."""
    work = np.asarray(value, dtype=np.float64).copy()
    coefficients = np.zeros(vectors.shape[1])
    for _ in range(reorthogonalize):
        for column in range(vectors.shape[1]):
            projection = float(np.dot(vectors[:, column], work))
            coefficients[column] += projection
            work -= projection * vectors[:, column]
        norm = _vector_norm(work)
        if norm <= tolerance:
            break
    return work, coefficients, _vector_norm(work)


def _initial_block_basis(matrix: typing.Any, *, tolerance: typing.Any) -> typing.Any:
    """Orthonormalize RHS columns without an absolute rank cutoff.

    The scalar SVD helper intentionally treats tiny singular values as
    numerical rank zero for recycling.  Block solves cannot do that: a
    numerically small but nonzero RHS still has its own convergence target.
    This routine drops only directions whose *orthogonal residual* is at the
    explicit breakdown tolerance.
    """
    value = np.asarray(matrix, dtype=np.float64)
    columns = []
    for column in range(value.shape[1]):
        work, _, norm = _orthogonalize_against(
            np.column_stack(columns) if columns else np.empty((value.shape[0], 0)),
            value[:, column],
            reorthogonalize=2,
            tolerance=tolerance,
        )
        if norm > tolerance:
            columns.append(work / norm)
    if not columns:
        return np.empty((value.shape[0], 0)), 0
    return np.column_stack(columns), len(columns)


def _single_workspace_bytes(n: typing.Any, options: typing.Any) -> typing.Any:
    """Bound solver-owned numeric buffers, including publication and LAPACK work.

    Reserve Arnoldi storage, overlapping old/new/immutable basis copies,
    residual/iterate temporaries and conservative small least-squares scratch.
    Operator and preconditioner storage have their own independent budgets.
    """
    restart = min(n, options.restart, options.max_iterations)
    return 8 * (
        (restart + 1) * n
        + 4 * n * restart
        + 20 * n
        + 8 * (restart + 1) ** 2
        + 4 * (options.max_iterations + 2)
    )


def _block_workspace_bytes(
    n: typing.Any, nrhs: typing.Any, options: typing.Any, max_columns: typing.Any
) -> typing.Any:
    """Bound live Arnoldi/SVD buffers and every published per-RHS basis.

    SVD and least-squares input/output arrays can coexist with the old basis,
    new basis, operator-image columns and stacked arrays. Reserve their full
    dimensions before constructing the initial block.
    """
    return 8 * (
        12 * n * max_columns
        + 12 * max_columns**2
        + 8 * max_columns * nrhs
        + 12 * n * nrhs
        + nrhs * n * max_columns
        + 4 * (options.max_iterations + nrhs + 1)
    )


def _workspace_failure(
    n: typing.Any,
    nrhs: typing.Any,
    required: typing.Any,
    *,
    reason: typing.Any = "workspace_limit",
) -> typing.Any:
    """Explicit nonconverged results for a preflight workspace rejection."""
    zero = immutable(np.zeros(n))
    return tuple(
        SolveResult(
            zero,
            False,
            float("inf"),
            float("inf"),
            0,
            reason,
            (),
            0,
            0,
            0.0,
            0.0,
            required,
            basis=np.empty((n, 0)),
        )
        for _ in range(nrhs)
    )


@dataclass(frozen=True)
class GMRESOptions:
    """Bounded restarted-GMRES controls with true-residual termination."""

    rtol: float = 1e-10
    atol: float = 0.0
    restart: int = 30
    max_iterations: int = 200
    max_workspace_bytes: int = 64 << 20
    reorthogonalize: int = 2
    breakdown_tolerance: float = 1e-14
    true_residual_every: int = 1
    stagnation_window: int = 25
    stagnation_tolerance: float = 1e-14

    def __post_init__(self) -> None:
        if not np.isfinite(self.rtol) or not 0 <= self.rtol < 1:
            raise ValueError("rtol must be finite and in [0,1)")
        if not np.isfinite(self.atol) or self.atol < 0:
            raise ValueError("atol must be finite and nonnegative")
        for name in (
            "restart",
            "max_iterations",
            "max_workspace_bytes",
            "reorthogonalize",
            "true_residual_every",
            "stagnation_window",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.breakdown_tolerance < 0 or not np.isfinite(self.breakdown_tolerance):
            raise ValueError("breakdown_tolerance must be finite and nonnegative")

    @property
    def workspace_bytes(self) -> typing.Any:
        return (self.restart + 1) ** 2 * 8 + (self.restart + 1) * 8


class DiagonalPreconditioner:
    """Separate diagonal preconditioner with an explicit nonzero gate."""

    def __init__(
        self, diagonal: typing.Any, *, relative_threshold: typing.Any = 1e-14
    ) -> None:
        value = np.asarray(diagonal, dtype=np.float64)
        if value.ndim != 1 or not np.isfinite(value).all():
            raise ValueError("preconditioner diagonal must be a finite vector")
        if not np.isfinite(relative_threshold) or relative_threshold < 0:
            raise ValueError("relative_threshold must be finite and nonnegative")
        scale = max(float(np.max(np.abs(value))), 1.0)
        cutoff = relative_threshold * scale
        if np.any(np.abs(value) <= cutoff):
            raise ValueError(
                "diagonal preconditioner contains a zero/near-zero entry; "
                "no denominator was clamped"
            )
        self.diagonal = immutable(value)
        self.inverse = immutable(1.0 / value)

    def apply(self, vector: typing.Any) -> typing.Any:
        value = np.asarray(vector, dtype=np.float64)
        if value.shape != self.diagonal.shape:
            raise ValueError("preconditioner vector shape mismatch")
        return self.inverse * value


@dataclass(frozen=True)
class SolveResult:
    """One response solve with the actual residual and failure reason."""

    solution: np.ndarray
    converged: bool
    residual_norm: float
    relative_residual: float
    iterations: int
    reason: str
    history: tuple[float, ...]
    operator_actions: int
    preconditioner_actions: int
    orthogonalization_seconds: float
    operator_seconds: float
    workspace_bytes: int
    recycled_vectors: int = 0
    rank: int = 0
    basis: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))

    def require_converged(self) -> typing.Any:
        """Raise with the complete diagnostic if the solve did not converge."""
        if not self.converged:
            raise ResponseSolveError(self)
        return self


@dataclass(frozen=True)
class MultiRHSResult:
    """Results and aggregate costs for one multi-RHS strategy."""

    results: tuple[SolveResult, ...]
    strategy: str
    seconds: float
    operator_actions: int
    peak_workspace_bytes: int
    rhs_rank: int
    rank_deficient_rhs: bool

    @property
    def converged(self) -> typing.Any:
        return all(result.converged for result in self.results)

    @property
    def solution(self) -> typing.Any:
        if not self.results:
            return np.empty((0, 0))
        return np.column_stack([result.solution for result in self.results])

    def require_converged(self) -> typing.Any:
        """Raise the first nonconverged result with its actual residual."""
        for result in self.results:
            result.require_converged()
        return self


class _HostKrylovEngine:
    """NumPy vector engine preserving the original #179 solver semantics."""

    resident = False

    def __init__(self, dimension: typing.Any) -> None:
        self.dimension = dimension

    def reset(self) -> None:
        return None

    def from_host(self, values: typing.Any) -> typing.Any:
        return np.asarray(values, dtype=np.float64).copy()

    def zeros(self) -> typing.Any:
        return np.zeros(self.dimension)

    def copy(self, value: typing.Any) -> typing.Any:
        return np.asarray(value, dtype=np.float64).copy()

    def scale(self, value: typing.Any, alpha: typing.Any) -> typing.Any:
        return np.asarray(value, dtype=np.float64) * float(alpha)

    def subtract(self, left: typing.Any, right: typing.Any) -> typing.Any:
        return np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)

    def norm(self, value: typing.Any) -> typing.Any:
        return _vector_norm(value)

    def apply(self, operator: typing.Any, value: typing.Any) -> typing.Any:
        return np.asarray(operator.apply(value), dtype=np.float64)

    def precondition(self, preconditioner: typing.Any, value: typing.Any) -> typing.Any:
        if preconditioner is None:
            return self.copy(value)
        return np.asarray(preconditioner.apply(value), dtype=np.float64)

    def orthogonalize(
        self,
        basis: typing.Any,
        value: typing.Any,
        *,
        reorthogonalize: typing.Any,
        tolerance: typing.Any,
    ) -> typing.Any:
        matrix = (
            np.column_stack(basis)
            if basis
            else np.empty((self.dimension, 0), dtype=np.float64)
        )
        return _orthogonalize_against(
            matrix,
            value,
            reorthogonalize=reorthogonalize,
            tolerance=tolerance,
        )

    def combination(
        self,
        base: typing.Any,
        basis: typing.Any,
        coefficients: typing.Any,
        preconditioner: typing.Any,
    ) -> typing.Any:
        direction = np.zeros(self.dimension)
        for coefficient, vector in zip(coefficients, basis, strict=True):
            direction += float(coefficient) * np.asarray(vector, dtype=np.float64)
        direction = self.precondition(preconditioner, direction)
        return np.asarray(base, dtype=np.float64) + direction

    def to_host(self, value: typing.Any) -> typing.Any:
        return np.asarray(value, dtype=np.float64).copy()

    def stack_host(self, values: typing.Any) -> typing.Any:
        if not values:
            return np.empty((self.dimension, 0))
        return np.column_stack([self.to_host(value) for value in values])


def _solve_single(
    operator: typing.Any,
    rhs: typing.Any,
    options: typing.Any,
    *,
    initial_guess: typing.Any = None,
    preconditioner: typing.Any = None,
    collect_basis: typing.Any = True,
) -> typing.Any:
    """Restarted GMRES with one control flow and pluggable vector residency."""
    b_host = np.asarray(rhs, dtype=np.float64)
    if b_host.ndim != 1 or not np.isfinite(b_host).all():
        raise ValueError("GMRES RHS must be a finite vector")
    n = b_host.size
    if n != operator.dimension:
        raise ValueError("GMRES RHS dimension mismatch")
    guess_host = None
    if initial_guess is not None:
        guess_host = np.asarray(initial_guess, dtype=np.float64)
        if guess_host.shape != (n,) or not np.isfinite(guess_host).all():
            raise ValueError(
                "initial guess must be a finite vector of operator dimension"
            )

    engine = getattr(operator, "_krylov_engine", None)
    if engine is None:
        engine = _HostKrylovEngine(n)
    if getattr(engine, "dimension", None) != n:
        raise ValueError("Krylov vector engine dimension mismatch")
    engine.reset()

    required_workspace = _single_workspace_bytes(n, options)
    if required_workspace > options.max_workspace_bytes:
        x = np.zeros(n) if guess_host is None else guess_host.copy()
        return SolveResult(
            immutable(x),
            False,
            float("inf"),
            float("inf"),
            0,
            "workspace_limit",
            (),
            0,
            0,
            0.0,
            0.0,
            required_workspace,
            basis=np.empty((n, 0)),
        )

    b = engine.from_host(b_host)
    x = engine.zeros() if guess_host is None else engine.from_host(guess_host)
    operator_actions = 0
    preconditioner_actions = 0
    ortho_seconds = 0.0
    operator_seconds = 0.0
    history = []
    total_steps = 0

    def apply(value: typing.Any) -> typing.Any:
        nonlocal operator_actions, operator_seconds
        begin = time.perf_counter()
        result = engine.apply(operator, value)
        operator_seconds += time.perf_counter() - begin
        operator_actions += 1
        return result

    def true_residual_norm(candidate: typing.Any) -> typing.Any:
        image = apply(candidate)
        return engine.norm(engine.subtract(b, image))

    def publish(
        solution: typing.Any,
        converged: typing.Any,
        residual_norm: typing.Any,
        iterations: typing.Any,
        reason: typing.Any,
        basis: typing.Any,
        rhs_norm: typing.Any,
    ) -> typing.Any:
        return SolveResult(
            immutable(engine.to_host(solution)),
            converged,
            residual_norm,
            _relative_residual(residual_norm, rhs_norm),
            iterations,
            reason,
            tuple(history),
            operator_actions,
            preconditioner_actions,
            ortho_seconds,
            operator_seconds,
            required_workspace,
            basis=immutable(
                engine.stack_host(basis) if collect_basis else np.empty((n, 0))
            ),
        )

    residual = (
        engine.subtract(b, apply(x)) if guess_host is not None else engine.copy(b)
    )
    beta = engine.norm(residual)
    history.append(beta)
    rhs_norm = engine.norm(b)
    target = max(options.atol, options.rtol * rhs_norm)
    if beta <= target:
        return publish(x, True, beta, 0, "initial_residual", (), rhs_norm)

    best_x = engine.copy(x)
    best_residual = beta
    best_basis = ()
    restart = min(n, options.restart, options.max_iterations)
    while total_steps < options.max_iterations:
        basis = [engine.scale(residual, 1.0 / beta)]
        h = np.zeros((restart + 1, restart))
        base_x = engine.copy(best_x)
        steps_this_cycle = 0
        candidate_x = engine.copy(best_x)
        candidate_residual = best_residual
        stagnation = 0
        for column in range(restart):
            if total_steps >= options.max_iterations:
                break
            ortho_started = time.perf_counter()
            work = engine.precondition(preconditioner, basis[column])
            if preconditioner is not None:
                preconditioner_actions += 1
            work = apply(work)
            work, coefficients, norm = engine.orthogonalize(
                basis[: column + 1],
                work,
                reorthogonalize=options.reorthogonalize,
                tolerance=options.breakdown_tolerance,
            )
            h[: column + 1, column] = coefficients
            ortho_seconds += time.perf_counter() - ortho_started
            if norm <= options.breakdown_tolerance:
                steps_this_cycle = column + 1
                total_steps += 1
                y, *_ = np.linalg.lstsq(
                    h[: column + 1, : column + 1],
                    beta * np.eye(column + 1, 1)[:, 0],
                    rcond=None,
                )
                candidate_x = engine.combination(
                    base_x, basis[: column + 1], y, preconditioner
                )
                if preconditioner is not None:
                    preconditioner_actions += 1
                candidate_residual = true_residual_norm(candidate_x)
                history.append(candidate_residual)
                break
            basis.append(engine.scale(work, 1.0 / norm))
            h[column + 1, column] = norm
            steps_this_cycle = column + 1
            total_steps += 1
            if (
                steps_this_cycle % options.true_residual_every == 0
                or steps_this_cycle == restart
                or total_steps == options.max_iterations
            ):
                y, *_ = np.linalg.lstsq(
                    h[: column + 2, : column + 1],
                    beta * np.eye(column + 2, 1)[:, 0],
                    rcond=None,
                )
                candidate_x = engine.combination(
                    base_x, basis[: column + 1], y, preconditioner
                )
                if preconditioner is not None:
                    preconditioner_actions += 1
                candidate_residual = true_residual_norm(candidate_x)
                history.append(candidate_residual)
                if candidate_residual <= target:
                    best_x, best_residual = candidate_x, candidate_residual
                    best_basis = tuple(basis[: column + 1])
                    return publish(
                        best_x,
                        True,
                        best_residual,
                        total_steps,
                        "converged",
                        best_basis,
                        rhs_norm,
                    )
                if candidate_residual >= best_residual * (
                    1.0 - options.stagnation_tolerance
                ):
                    stagnation += 1
                else:
                    stagnation = 0
                if candidate_residual < best_residual:
                    best_x, best_residual = candidate_x, candidate_residual
                    best_basis = tuple(basis[: column + 1])
                if stagnation >= options.stagnation_window:
                    return publish(
                        best_x,
                        False,
                        best_residual,
                        total_steps,
                        "stagnation",
                        best_basis,
                        rhs_norm,
                    )
        if candidate_residual <= target:
            return publish(
                candidate_x,
                True,
                candidate_residual,
                total_steps,
                "converged",
                tuple(basis[:steps_this_cycle]),
                rhs_norm,
            )
        if steps_this_cycle == 0:
            break
        if best_residual >= beta:
            return publish(
                best_x,
                False,
                best_residual,
                total_steps,
                "breakdown" if best_residual > 0 else "singular",
                best_basis,
                rhs_norm,
            )
        residual = engine.subtract(b, apply(best_x))
        beta = engine.norm(residual)
        history.append(beta)

    reason = "max_iterations" if total_steps >= options.max_iterations else "breakdown"
    return publish(
        best_x, False, best_residual, total_steps, reason, best_basis, rhs_norm
    )


class KrylovRecycleSpace:
    """Reference-bound retained Krylov vectors with explicit reset/transport."""

    def __init__(
        self,
        problem: typing.Any,
        *,
        max_vectors: typing.Any = 8,
        max_bytes: typing.Any = 8 << 20,
    ) -> None:
        if type(max_vectors) is not int or max_vectors < 1:
            raise ValueError("max_vectors must be positive")
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.problem = problem
        self.max_vectors = max_vectors
        self.max_bytes = max_bytes
        self._vectors = []
        self.generation = 0

    @property
    def key(self) -> typing.Any:
        return self.problem.compatibility_identity

    @property
    def identity(self) -> typing.Any:
        return canonical_hash(
            {
                "problem": self.key,
                "generation": self.generation,
                "vectors": [hashlib_sha(v) for v in self._vectors],
            }
        )

    @property
    def storage_bytes(self) -> typing.Any:
        """Bytes held by the independently retained immutable vectors."""
        return sum(vector.nbytes for vector in self._vectors)

    def projection_bytes(self, dimension: typing.Any) -> typing.Any:
        """Reserve the output guess and one scaled retained-vector temporary."""
        if type(dimension) is not int or dimension < 0:
            raise ValueError("dimension must be a nonnegative integer")
        return 2 * dimension * 8

    def update_bytes(self, dimension: typing.Any) -> typing.Any:
        """Bound replacement vectors and Gram-Schmidt publication temporaries.

        Existing vectors are charged separately by ``storage_bytes``. The
        update builds at most the destination capacity, without stacking or
        taking an SVD of the complete old/result basis.
        """
        if dimension == 0:
            return 0
        capacity = min(self.max_vectors, dimension, self.max_bytes // (dimension * 8))
        return (capacity + 3) * dimension * 8

    def assert_compatible(self, problem: typing.Any) -> None:
        """Fail closed on a changed reference/model/operator/layout."""
        if problem.compatibility_identity != self.key:
            raise ResponseCompatibilityError(
                "stale Krylov subspace: reference/model/operator compatibility key changed"
            )

    def initial_guess(self, problem: typing.Any, rhs: typing.Any) -> typing.Any:
        """Project one RHS onto the already orthonormal retained vectors."""
        self.assert_compatible(problem)
        b = np.asarray(rhs, dtype=np.float64)
        if b.ndim != 1 or not np.isfinite(b).all():
            raise ValueError("recycled RHS must be a finite vector")
        guess = np.zeros(b.size)
        for vector in self._vectors:
            if vector.shape != b.shape:
                raise ValueError("recycled vector dimension mismatch")
            guess += vector * np.dot(vector, b)
        return guess

    def update(self, problem: typing.Any, result: typing.Any) -> typing.Any:
        """Publish a bounded orthonormal replacement after a successful solve."""
        self.assert_compatible(problem)
        if not result.converged:
            return self
        n = result.solution.size
        # A valid UHF reference can have no occupied-virtual rotations. Its
        # solved empty vector has no reusable directions or storage cost.
        capacity = min(self.max_vectors, n, self.max_bytes // (n * 8)) if n else 0
        replacement = []
        # Retain old directions first and stop as soon as capacity is reached;
        # no full candidate matrix is created.
        candidates = (*self._vectors, result.solution, *result.basis.T)
        for candidate in candidates:
            if len(replacement) == capacity:
                break
            work = np.asarray(candidate, dtype=np.float64).copy()
            if work.shape != (n,) or not np.isfinite(work).all():
                raise ValueError("invalid recycle candidate")
            original_norm = _vector_norm(work)
            for _ in range(2):
                for vector in replacement:
                    work -= np.dot(vector, work) * vector
            norm = _vector_norm(work)
            if norm > 1e-12 * max(1.0, original_norm):
                work /= norm
                replacement.append(immutable(work))
        self._vectors = replacement
        self.generation += 1
        return self

    def reset(self, problem: typing.Any = None) -> typing.Any:
        """Discard all vectors; optionally bind a fresh compatible problem."""
        if problem is not None:
            self.problem = problem
        self._vectors = []
        self.generation += 1
        return self

    def transport(self, problem: typing.Any, transform: typing.Any) -> typing.Any:
        """Explicitly transport vectors to a new problem under a caller map."""
        if not callable(transform):
            raise TypeError("transport requires an explicit callable")
        if not hasattr(problem, "dimension") or type(problem.dimension) is not int:
            raise TypeError("transport destination must expose an integer dimension")
        transported = []
        for vector in self._vectors:
            value = np.asarray(transform(vector), dtype=np.float64)
            if value.shape != (problem.dimension,) or not np.isfinite(value).all():
                raise ValueError("transport produced an invalid vector")
            transported.append(value)
        replacement = KrylovRecycleSpace(
            problem, max_vectors=self.max_vectors, max_bytes=self.max_bytes
        )
        if transported:
            basis, _ = _orthonormal_basis(
                np.column_stack(transported), max_columns=self.max_vectors
            )
            while basis.shape[1] > 1 and basis.nbytes > self.max_bytes:
                basis = basis[:, :-1]
            if basis.nbytes > self.max_bytes:
                basis = np.empty((problem.dimension, 0))
            replacement._vectors = [
                immutable(basis[:, column]) for column in range(basis.shape[1])
            ]
        replacement.generation = self.generation + 1
        return replacement


def hashlib_sha(value: typing.Any) -> typing.Any:
    """Stable hash for a retained vector without importing a public helper."""
    import hashlib

    return hashlib.sha256(
        np.ascontiguousarray(value, dtype="<f8").tobytes()
    ).hexdigest()


def solve(
    operator: typing.Any,
    rhs: typing.Any,
    *,
    options: typing.Any = None,
    initial_guess: typing.Any = None,
    recycle: typing.Any = None,
    preconditioner: typing.Any = None,
    raise_on_failure: typing.Any = False,
    collect_basis: typing.Any = True,
) -> typing.Any:
    """Solve one RHS with bounded true-residual GMRES."""
    # A live reference lease must hold even when a zero RHS skips all actions.
    validate_current = getattr(operator, "validate_current", lambda: None)
    validate_current()
    options = GMRESOptions() if options is None else options
    b = np.asarray(rhs)
    if (
        b.shape != (operator.dimension,)
        or np.iscomplexobj(b)
        or not np.isfinite(b).all()
    ):
        raise ValueError("GMRES RHS must be a finite real vector of operator dimension")
    if initial_guess is not None:
        guess = np.asarray(initial_guess)
        if (
            guess.shape != b.shape
            or np.iscomplexobj(guess)
            or not np.isfinite(guess).all()
        ):
            raise ValueError(
                "initial guess must be a finite real vector of operator dimension"
            )
    reservation = 0
    recycled_vectors = 0
    if recycle is not None:
        recycle.assert_compatible(operator.problem)
        recycled_vectors = len(recycle._vectors)
        reservation = (
            recycle.storage_bytes
            + recycle.projection_bytes(operator.dimension)
            + recycle.update_bytes(operator.dimension)
        )
    required = _single_workspace_bytes(operator.dimension, options) + reservation
    if required > options.max_workspace_bytes:
        # This preflight owns the whole public solve, including projection and
        # replacement. Direct callers receive the same bound as solve_many.
        result = _workspace_failure(operator.dimension, 1, required)[0]
    else:
        if recycle is not None and initial_guess is None:
            initial_guess = recycle.initial_guess(operator.problem, rhs)
        engine = getattr(operator, "_krylov_engine", None)
        workspace = getattr(engine, "solver_workspace", nullcontext)
        with workspace():
            result = _solve_single(
                operator,
                rhs,
                replace(
                    options,
                    max_workspace_bytes=options.max_workspace_bytes - reservation,
                ),
                initial_guess=initial_guess,
                preconditioner=preconditioner,
                collect_basis=collect_basis or recycle is not None,
            )
        validate_current()
        if recycle is not None and result.converged:
            recycle.update(operator.problem, result)
    result = replace(
        result, recycled_vectors=recycled_vectors, workspace_bytes=required
    )
    if raise_on_failure:
        result.require_converged()
    validate_current()
    return result


def _block_solve(
    operator: typing.Any, rhs: typing.Any, options: typing.Any
) -> typing.Any:
    """Block GMRES with block-Arnoldi expansion and true residuals.

    Resident vector execution qualifies scalar/recycled GMRES only. The block
    Arnoldi implementation remains host-only and fails rather than relabeling
    host vector work as device-resident.
    """
    if getattr(getattr(operator, "_krylov_engine", None), "resident", False):
        raise ValueError(
            "blocked GMRES is not qualified for resident vector execution; "
            "use sequential or recycled"
        )

    # Expansion uses orthogonalized operator images rather than the projected
    # Galerkin residual. This is required for indefinite/nonsymmetric operators
    # where the projected matrix can be singular even though the operator is
    # nonsingular.
    b = np.asarray(rhs, dtype=np.float64)
    n, nrhs = b.shape
    max_columns = min(n, nrhs + options.max_iterations)
    required_workspace = _block_workspace_bytes(n, nrhs, options, max_columns)
    if required_workspace > options.max_workspace_bytes:
        return (
            _workspace_failure(n, nrhs, required_workspace),
            0,
            required_workspace,
            False,
        )
    basis, rank = _initial_block_basis(b, tolerance=options.breakdown_tolerance)
    rank_deficient = rank < nrhs
    if rank == 0:
        results = []
        for column in range(nrhs):
            norm = _vector_norm(b[:, column])
            target = max(options.atol, options.rtol * norm)
            converged = norm <= target
            results.append(
                SolveResult(
                    immutable(np.zeros(n)),
                    converged,
                    norm,
                    _relative_residual(norm, norm),
                    0,
                    "zero_rhs" if converged else "rank_deficient_rhs",
                    (norm,),
                    0,
                    0,
                    0.0,
                    0.0,
                    required_workspace,
                    rank=0,
                    basis=np.empty((n, 0)),
                )
            )
        return tuple(results), 0, required_workspace, rank_deficient

    q_initial = basis.shape[1]
    initial_coefficients = basis.T @ b
    hbar = np.zeros((max_columns, max_columns))
    solution = np.zeros((n, nrhs))
    residual = b.copy()
    action_seconds = 0.0
    ortho_seconds = 0.0
    actions = 0
    iterations = 0
    history = [float(_vector_norm(b[:, column])) for column in range(nrhs)]
    last_start = 0
    breakdown = False
    while iterations < options.max_iterations:
        q = basis.shape[1]
        block = basis[:, last_start:q]
        apply_started = time.perf_counter()
        image = np.column_stack(
            [operator.apply(block[:, column]) for column in range(block.shape[1])]
        )
        actions += block.shape[1]
        action_seconds += time.perf_counter() - apply_started
        h_top = np.zeros((q, block.shape[1]))
        work = image.copy()
        ortho_started = time.perf_counter()
        for _ in range(options.reorthogonalize):
            for index in range(q):
                projection = basis[:, index] @ work
                h_top[index, :] += projection
                work -= np.outer(basis[:, index], projection)
        ortho_seconds += time.perf_counter() - ortho_started
        left, singular, right = np.linalg.svd(work, full_matrices=False)
        cutoff = max(
            options.breakdown_tolerance,
            np.finfo(float).eps
            * max(n, block.shape[1])
            * (singular[0] if len(singular) else 0.0),
        )
        keep = int(np.count_nonzero(singular > cutoff))
        keep = min(keep, max_columns - q)
        hbar[:q, last_start:q] = h_top
        if keep:
            new_basis = left[:, :keep]
            hbar[q : q + keep, last_start:q] = singular[:keep, None] * right[:keep, :]
            basis = np.column_stack((basis, new_basis))
        q_new = basis.shape[1]
        projected = hbar[:q_new, :q]
        for column in range(nrhs):
            rhs_projected = np.zeros(q_new)
            rhs_projected[:q_initial] = initial_coefficients[:, column]
            coefficients, *_ = np.linalg.lstsq(projected, rhs_projected, rcond=None)
            solution[:, column] = basis[:, :q] @ coefficients
        apply_started = time.perf_counter()
        residual = b - np.column_stack(
            [operator.apply(solution[:, column]) for column in range(nrhs)]
        )
        actions += nrhs
        action_seconds += time.perf_counter() - apply_started
        norms = np.array([_vector_norm(residual[:, column]) for column in range(nrhs)])
        history = [float(max(old, new)) for old, new in zip(history, norms)]
        targets = np.maximum(
            options.atol,
            options.rtol
            * np.array([_vector_norm(b[:, column]) for column in range(nrhs)]),
        )
        if np.all(norms <= targets):
            break
        if keep == 0:
            breakdown = True
            break
        iterations += keep
        last_start = q
    results = []
    for column in range(nrhs):
        norm = _vector_norm(b[:, column] - operator.apply(solution[:, column]))
        target = max(options.atol, options.rtol * _vector_norm(b[:, column]))
        converged = norm <= target
        if converged:
            reason = "converged"
        elif breakdown:
            reason = "breakdown"
        else:
            reason = "max_iterations"
        results.append(
            SolveResult(
                immutable(solution[:, column]),
                converged,
                norm,
                _relative_residual(norm, float(_vector_norm(b[:, column]))),
                iterations,
                reason,
                tuple(history),
                actions + nrhs,
                0,
                ortho_seconds,
                action_seconds,
                required_workspace,
                rank=rank,
                basis=immutable(basis),
            )
        )
    return tuple(results), actions + nrhs, required_workspace, rank_deficient


def solve_many(
    operator: typing.Any,
    rhs: typing.Any,
    *,
    strategy: typing.Any = "sequential",
    options: typing.Any = None,
    recycle: typing.Any = None,
    preconditioner: typing.Any = None,
    raise_on_failure: typing.Any = False,
) -> typing.Any:
    """Compare sequential, blocked and recycled multi-RHS response solves."""
    validate_current = getattr(operator, "validate_current", lambda: None)
    validate_current()
    if strategy not in ("sequential", "blocked", "recycled"):
        raise ValueError("strategy must be sequential, blocked or recycled")
    if strategy == "blocked" and preconditioner is not None:
        raise ValueError(
            "blocked GMRES does not accept a preconditioner; use sequential "
            "or recycled, or add an explicitly tested block preconditioner"
        )
    options = GMRESOptions() if options is None else options
    values = operator.problem.validate_rhs(rhs)
    started = time.perf_counter()
    # validate_rhs publishes an owned immutable array for real ResponseProblems.
    # Charge it even when a test/custom operator happens to return a view.
    input_bytes = values.nbytes
    available = options.max_workspace_bytes - input_bytes
    if available <= 0:
        answer = MultiRHSResult(
            _workspace_failure(values.shape[0], values.shape[1], input_bytes),
            strategy,
            time.perf_counter() - started,
            0,
            input_bytes,
            0,
            False,
        )
    elif strategy == "blocked":
        results, actions, peak, rank_deficient = _block_solve(
            operator, values, replace(options, max_workspace_bytes=available)
        )
        answer = MultiRHSResult(
            tuple(results),
            strategy,
            time.perf_counter() - started,
            actions,
            input_bytes + peak,
            results[0].rank if results else 0,
            rank_deficient,
        )
    else:
        use_recycle = strategy == "recycled"
        if use_recycle and recycle is None:
            recycle = KrylovRecycleSpace(operator.problem)
        # Rank diagnostics use a value-only SVD. Reserve its input/workspace
        # separately; it is released before the first Krylov solve starts.
        n, nrhs = values.shape
        rank_workspace = 8 * (4 * n * nrhs + 8 * min(n, nrhs) ** 2)
        if use_recycle:
            recycle.assert_compatible(operator.problem)
            rank_workspace += recycle.storage_bytes
        if rank_workspace > available:
            answer = MultiRHSResult(
                _workspace_failure(n, nrhs, input_bytes + rank_workspace),
                strategy,
                time.perf_counter() - started,
                0,
                input_bytes + rank_workspace,
                0,
                False,
            )
            if raise_on_failure:
                answer.require_converged()
            return answer
        rank = int(np.linalg.matrix_rank(values, tol=1e-12)) if nrhs else 0
        results = []
        actions = 0
        peak = input_bytes + rank_workspace
        retained_results = 0
        for column in range(values.shape[1]):
            # solve owns all recycle reservations. Only earlier results and the
            # shared RHS copy remain outside it; the current result is already
            # included in solve's bound and must not be counted a second time.
            retained = input_bytes + retained_results
            remaining = options.max_workspace_bytes - retained
            if remaining <= 0:
                results.extend(
                    _workspace_failure(
                        values.shape[0], values.shape[1] - column, retained
                    )
                )
                peak = max(peak, retained)
                break
            result = solve(
                operator,
                values[:, column],
                options=replace(options, max_workspace_bytes=remaining),
                recycle=recycle if use_recycle else None,
                preconditioner=preconditioner,
            )
            peak = max(peak, retained + result.workspace_bytes)
            results.append(result)
            actions += result.operator_actions
            if result.reason == "workspace_limit":
                # No later column can gain capacity. Share the immutable failure
                # result rather than allocating a zero solution for every RHS.
                results.extend([result] * (values.shape[1] - column - 1))
                break
            retained_results += result.solution.nbytes + result.basis.nbytes
        answer = MultiRHSResult(
            tuple(results),
            strategy,
            time.perf_counter() - started,
            actions,
            peak,
            rank,
            rank < values.shape[1],
        )
    if raise_on_failure:
        answer.require_converged()
    validate_current()
    return answer
