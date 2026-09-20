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
    coefficients = np.zeros(len(vectors))
    for _ in range(reorthogonalize):
        for column, vector in enumerate(vectors):
            projection = float(np.dot(vector, work))
            coefficients[column] += projection
            work -= projection * vector
        norm = _vector_norm(work)
        if norm <= tolerance:
            break
    return work, coefficients, _vector_norm(work)


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


def resident_vector_slots(
    dimension: int,
    options: typing.Any,
    *,
    rhs_count: int = 1,
    strategy: str = "sequential",
    recycle_capacity: int = 8,
) -> int:
    """Conservative vector-slot plan for the shared resident algorithms.

    This is a logical lease bound, separate from the native owner's device
    byte request. Consumers must reject an unsupported count instead of
    clamping it to the owner's maximum and discovering exhaustion mid-solve.
    """
    if (
        type(dimension) is not int
        or dimension < 0
        or type(rhs_count) is not int
        or rhs_count < 0
    ):
        raise ValueError("dimension and rhs_count must be nonnegative integers")
    if strategy not in ("sequential", "blocked", "recycled"):
        raise ValueError("unknown resident multi-RHS strategy")
    if not isinstance(options, GMRESOptions):
        raise TypeError("resident vector planning requires GMRESOptions")
    if type(recycle_capacity) is not int or recycle_capacity < 0:
        raise ValueError("recycle_capacity must be a nonnegative integer")
    if strategy == "blocked":
        return (
            4 * min(dimension, rhs_count + options.max_iterations) + 2 * rhs_count + 12
        )
    restart = min(dimension, options.restart, options.max_iterations)
    return (
        2 * restart
        + 16
        + (2 * min(recycle_capacity, dimension) if strategy == "recycled" else 0)
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
    """One response solve with the actual residual and failure reason.

    Operator time measures only engine actions. Orthogonalization includes
    basis construction/projection and block range factorization; recycling
    measures projection/replacement of the retained space. These disjoint
    components are not a complete wall-time decomposition: residual vector
    arithmetic, small least squares, validation and publication remain in
    the enclosing solve's wall time.
    """

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
    recycling_seconds: float = 0.0

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

    def _aggregate_seconds(self, field: str) -> float:
        # Each blocked result describes the same shared solve; summing those
        # repeated records would multiply one operation's cost by nrhs.
        records = self.results[:1] if self.strategy == "blocked" else self.results
        return sum(getattr(item, field) for item in records)

    @property
    def operator_seconds(self) -> float:
        return self._aggregate_seconds("operator_seconds")

    @property
    def orthogonalization_seconds(self) -> float:
        return self._aggregate_seconds("orthogonalization_seconds")

    @property
    def recycling_seconds(self) -> float:
        return self._aggregate_seconds("recycling_seconds")

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

    def dot(self, left: typing.Any, right: typing.Any) -> float:
        return float(np.dot(left, right))

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
        return _orthogonalize_against(
            basis,
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
    resident_recycle: typing.Any = None,
    solution_consumer: typing.Any = None,
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
    has_guess = guess_host is not None
    recycle_seconds = 0.0
    if resident_recycle is not None and guess_host is None:
        recycle_started = time.perf_counter()
        x = resident_recycle._initial_guess_vector(operator.problem, b, engine)
        recycle_seconds += time.perf_counter() - recycle_started
        has_guess = bool(resident_recycle._vectors)
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
        if converged and solution_consumer is not None:
            solution_consumer(engine, solution)
        result = SolveResult(
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
        if converged and resident_recycle is not None:
            # Recycle before the temporary-vector scope is drained. Published
            # host diagnostics are never uploaded again to update the space.
            getattr(operator, "validate_current", lambda: None)()
            recycle_started = time.perf_counter()
            resident_recycle._update_vectors(operator.problem, solution, basis, engine)
            return replace(
                result,
                recycling_seconds=recycle_seconds
                + time.perf_counter()
                - recycle_started,
            )
        return replace(result, recycling_seconds=recycle_seconds)

    residual = engine.subtract(b, apply(x)) if has_guess else engine.copy(b)
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
            work = engine.precondition(preconditioner, basis[column])
            if preconditioner is not None:
                preconditioner_actions += 1
            work = apply(work)
            # Keep operator execution out of the orthogonalization ledger.
            ortho_started = time.perf_counter()
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
    """Reference-bound retained vectors with optional resident ownership.

    ``vector_engine=resident`` keeps projection and replacement on that exact
    owner. Close/reset this space before closing the borrowed resident owner.
    Default host storage and explicit host-result updates remain supported.
    """

    def __init__(
        self,
        problem: typing.Any,
        *,
        max_vectors: typing.Any = 8,
        max_bytes: typing.Any = 8 << 20,
        vector_engine: typing.Any = None,
    ) -> None:
        if type(max_vectors) is not int or max_vectors < 1:
            raise ValueError("max_vectors must be positive")
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.problem = problem
        self.max_vectors = max_vectors
        self.max_bytes = max_bytes
        self._engine = vector_engine
        if vector_engine is not None:
            if not getattr(vector_engine, "resident", False):
                raise TypeError("recycle vector_engine must be a resident owner")
            if (
                vector_engine.problem.compatibility_identity
                != problem.compatibility_identity
            ):
                raise ResponseCompatibilityError(
                    "resident recycle owner/problem mismatch"
                )
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
                "vectors": (
                    [hashlib_sha(v) for v in self._vectors]
                    if self._engine is None
                    else [v.slot for v in self._vectors]
                ),
                "vector_owner": None if self._engine is None else self._engine.identity,
            }
        )

    @property
    def storage_bytes(self) -> typing.Any:
        """Bytes held by the independently retained immutable vectors."""
        if self._engine is not None:
            return len(self._vectors) * self._engine.dimension * 8
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
        if self._engine is not None:
            self._engine._backend._ensure_open()
            if self._engine._closed:
                raise RuntimeError("resident recycle owner is closed")
            for vector in self._vectors:
                self._engine._validate_vector(vector)

    def initial_guess(self, problem: typing.Any, rhs: typing.Any) -> typing.Any:
        """Project one RHS onto the already orthonormal retained vectors."""
        self.assert_compatible(problem)
        b = np.asarray(rhs, dtype=np.float64)
        if b.ndim != 1 or not np.isfinite(b).all():
            raise ValueError("recycled RHS must be a finite vector")
        if self._engine is not None:
            # Explicit diagnostic API: the production solver uses the resident
            # vector seam below and avoids both this upload and publication.
            with self._engine.solver_workspace():
                guess = self._initial_guess_vector(
                    problem, self._engine.from_host(b), self._engine
                )
                return self._engine.to_host(guess)
        guess = np.zeros(b.size)
        for vector in self._vectors:
            if vector.shape != b.shape:
                raise ValueError("recycled vector dimension mismatch")
            guess += vector * np.dot(vector, b)
        return guess

    def _initial_guess_vector(
        self, problem: typing.Any, rhs: typing.Any, engine: typing.Any
    ) -> typing.Any:
        self.assert_compatible(problem)
        if engine is not self._engine:
            raise ResponseCompatibilityError(
                "recycle vectors belong to another resident owner"
            )
        coefficients = [engine.dot(vector, rhs) for vector in self._vectors]
        return engine.combination(engine.zeros(), self._vectors, coefficients, None)

    def update(self, problem: typing.Any, result: typing.Any) -> typing.Any:
        """Publish a bounded orthonormal replacement after a successful solve."""
        self.assert_compatible(problem)
        if not result.converged:
            return self
        n = result.solution.size
        if self._engine is not None:
            # Caller-requested import of a detached result, distinct from the
            # production solve path's direct resident update.
            with self._engine.solver_workspace():
                solution = self._engine.from_host(result.solution)
                basis = [self._engine.from_host(value) for value in result.basis.T]
                return self._update_vectors(problem, solution, basis, self._engine)
        return self._update_vectors(
            problem, result.solution, result.basis.T, _HostKrylovEngine(n)
        )

    def _update_vectors(
        self,
        problem: typing.Any,
        solution: typing.Any,
        basis: typing.Any,
        engine: typing.Any,
    ) -> typing.Any:
        """One bounded replacement policy for host and device vector storage."""
        self.assert_compatible(problem)
        if self._engine is not None and engine is not self._engine:
            raise ResponseCompatibilityError(
                "recycle vectors belong to another resident owner"
            )
        n = engine.dimension
        # A valid UHF reference can have no occupied-virtual rotations. Its
        # solved empty vector has no reusable directions or storage cost.
        capacity = min(self.max_vectors, n, self.max_bytes // (n * 8)) if n else 0
        replacement = []
        # Retain old directions first and stop as soon as capacity is reached;
        # no full candidate matrix is created.
        candidates = (*self._vectors, solution, *basis)
        try:
            for candidate in candidates:
                if len(replacement) == capacity:
                    break
                work = engine.copy(candidate)
                if self._engine is None and (
                    work.shape != (n,) or not np.isfinite(work).all()
                ):
                    raise ValueError("invalid recycle candidate")
                original_norm = engine.norm(work)
                work, _, norm = engine.orthogonalize(
                    replacement, work, reorthogonalize=2, tolerance=0.0
                )
                if norm > 1e-12 * max(1.0, original_norm):
                    vector = _normalized_vector(engine, work, norm)
                    replacement.append(
                        immutable(vector) if self._engine is None else vector
                    )
            if self._engine is not None:
                for vector in replacement:
                    engine._retain(vector)
        except BaseException:
            if self._engine is not None:
                for vector in replacement:
                    vector.release()
            raise
        if self._engine is not None:
            for vector in self._vectors:
                vector.release()
        self._vectors = replacement
        self.generation += 1
        return self

    def reset(self, problem: typing.Any = None) -> typing.Any:
        """Discard all vectors; optionally bind a fresh compatible problem."""
        if problem is not None:
            if self._engine is not None:
                self.assert_compatible(problem)
            self.problem = problem
        if self._engine is not None:
            for vector in self._vectors:
                vector.release()
        self._vectors = []
        self.generation += 1
        return self

    def close(self) -> None:
        """Release retained vector leases without closing the borrowed owner."""
        self.reset()

    def __enter__(self) -> typing.Self:
        self.assert_compatible(self.problem)
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_vectors"):
            self.close()

    def transport(self, problem: typing.Any, transform: typing.Any) -> typing.Any:
        """Explicitly transport vectors to a new problem under a caller map."""
        if not callable(transform):
            raise TypeError("transport requires an explicit callable")
        if not hasattr(problem, "dimension") or type(problem.dimension) is not int:
            raise TypeError("transport destination must expose an integer dimension")
        transported = []
        for vector in self._vectors:
            host_vector = (
                self._engine.to_host(vector) if self._engine is not None else vector
            )
            value = np.asarray(transform(host_vector), dtype=np.float64)
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
    solution_consumer: typing.Any = None,
) -> typing.Any:
    """Solve one RHS with bounded true-residual GMRES.

    solution_consumer runs synchronously on the converged engine-native
    solution before host publication and must not retain engine-owned leases.
    """
    # A live reference lease must hold even when a zero RHS skips all actions.
    validate_current = getattr(operator, "validate_current", lambda: None)
    validate_current()
    options = GMRESOptions() if options is None else options
    if solution_consumer is not None and not callable(solution_consumer):
        raise TypeError("solution_consumer must be callable")
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
    recycle_seconds = 0.0
    recycled_vectors = 0
    engine = getattr(operator, "_krylov_engine", None)
    resident_recycle = recycle is not None and recycle._engine is not None
    if recycle is not None:
        recycle.assert_compatible(operator.problem)
        if resident_recycle and recycle._engine is not engine:
            raise ResponseCompatibilityError(
                "recycle vectors belong to another resident owner"
            )
        recycled_vectors = len(recycle._vectors)
        reservation = (
            recycle.storage_bytes
            + recycle.projection_bytes(operator.dimension)
            + recycle.update_bytes(operator.dimension)
        )
    required = _single_workspace_bytes(operator.dimension, options) + reservation
    # Old best-cycle and current Arnoldi vectors may overlap. Resident recycle
    # replacement additionally overlaps both old and new retained spaces.
    retained_capacity = (
        min(
            recycle.max_vectors,
            operator.dimension,
            recycle.max_bytes // (8 * operator.dimension),
        )
        if resident_recycle and operator.dimension
        else 0
    )
    required_slots = (
        resident_vector_slots(operator.dimension, options)
        + recycled_vectors
        + retained_capacity
    )
    if required > options.max_workspace_bytes:
        # This preflight owns the whole public solve, including projection and
        # replacement. Direct callers receive the same bound as solve_many.
        result = _workspace_failure(operator.dimension, 1, required)[0]
    elif getattr(engine, "vector_slots", required_slots) < required_slots:
        result = _workspace_failure(
            operator.dimension, 1, required, reason="vector_slot_limit"
        )[0]
    else:
        if recycle is not None and initial_guess is None and not resident_recycle:
            recycle_started = time.perf_counter()
            initial_guess = recycle.initial_guess(operator.problem, rhs)
            recycle_seconds += time.perf_counter() - recycle_started
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
                collect_basis=collect_basis
                or (recycle is not None and not resident_recycle),
                resident_recycle=recycle if resident_recycle else None,
                solution_consumer=solution_consumer,
            )
        validate_current()
        if recycle is not None and result.converged and not resident_recycle:
            recycle_started = time.perf_counter()
            recycle.update(operator.problem, result)
            recycle_seconds += time.perf_counter() - recycle_started
    result = replace(
        result,
        recycled_vectors=recycled_vectors,
        workspace_bytes=required,
        recycling_seconds=result.recycling_seconds + recycle_seconds,
    )
    if not collect_basis and result.basis.shape[1]:
        # A host recycle update temporarily needs the solved basis too, but
        # that internal need must not override the caller's output policy.
        result = replace(result, basis=np.empty((operator.dimension, 0)))
    if raise_on_failure:
        result.require_converged()
    validate_current()
    return result


def _normalized_vector(
    engine: typing.Any, value: typing.Any, norm: float
) -> typing.Any:
    """Normalize without overflowing the reciprocal of a subnormal norm."""
    if norm < 1e-200:
        return engine.scale(engine.scale(value, 1e150), 1.0 / (norm * 1e150))
    return engine.scale(value, 1.0 / norm)


def _block_range_factor(
    engine: typing.Any,
    columns: typing.Any,
    *,
    tolerance: float,
    capacity: int,
) -> typing.Any:
    """Thin QR followed by a small SVD, keeping all long vectors in the engine.

    Twice-reorthogonalized MGS gives W=Q R. SVD(R) then makes the same
    singular-value rank decision as SVD(W), without a host N-by-block panel
    or the loss of precision from forming W.T@W. Projected scalars stay on
    the host for both vector engines. Keep even tiny nonzero QR residuals;
    only the subsequent SVD applies the declared block breakdown gate.
    """
    count = len(columns)
    qr = []
    factor = np.zeros((min(engine.dimension, count), count))
    for column, value in enumerate(columns):
        work, coefficients, norm = engine.orthogonalize(
            qr, engine.copy(value), reorthogonalize=2, tolerance=0.0
        )
        factor[: len(qr), column] = coefficients
        if norm > 0.0 and len(qr) < engine.dimension:
            factor[len(qr), column] = norm
            qr.append(_normalized_vector(engine, work, norm))
    left, singular, right = np.linalg.svd(factor[: len(qr)], full_matrices=False)
    cutoff = max(
        tolerance,
        np.finfo(float).eps
        * max(engine.dimension, count)
        * (singular[0] if len(singular) else 0.0),
    )
    keep = min(int(np.count_nonzero(singular > cutoff)), capacity)
    zero = engine.zeros()
    basis = [
        engine.combination(zero, qr, left[:, column], None) for column in range(keep)
    ]
    return basis, singular[:keep, None] * right[:keep, :]


def _block_solve(
    operator: typing.Any,
    rhs: typing.Any,
    options: typing.Any,
    *,
    collect_basis: bool = True,
    solution_consumers: typing.Any = None,
) -> typing.Any:
    """One block-Arnoldi algorithm with host or resident vector storage.

    Only projected coefficients and small least-squares/SVD problems are host
    data. Input upload and explicit final solution/basis publication delimit
    the resident solve; no intermediate long vector is downloaded.
    """

    # Expansion uses orthogonalized operator images rather than the projected
    # Galerkin residual. This is required for indefinite/nonsymmetric operators
    # where the projected matrix can be singular even though the operator is
    # nonsingular.
    b_host = np.asarray(rhs, dtype=np.float64)
    n, nrhs = b_host.shape
    consumers = (
        (None,) * nrhs if solution_consumers is None else tuple(solution_consumers)
    )
    if len(consumers) != nrhs or any(
        consumer is not None and not callable(consumer) for consumer in consumers
    ):
        raise ValueError("block solution_consumers must match RHS columns")
    engine = getattr(operator, "_krylov_engine", None) or _HostKrylovEngine(n)
    if engine.dimension != n:
        raise ValueError("Krylov vector engine dimension mismatch")
    engine.reset()
    max_columns = min(n, nrhs + options.max_iterations)
    required_workspace = _block_workspace_bytes(n, nrhs, options, max_columns)
    if required_workspace > options.max_workspace_bytes:
        return (
            _workspace_failure(n, nrhs, required_workspace),
            0,
            required_workspace,
            False,
        )
    if nrhs == 0:
        return (), 0, required_workspace, False
    # Reserve the live basis, RHS, iterates, QR/range panels and overlapping
    # replacement temporaries before the first upload or operator action.
    # The owner's physical arena has its own device-byte budget.
    required_slots = resident_vector_slots(
        n, options, rhs_count=nrhs, strategy="blocked"
    )
    if getattr(engine, "vector_slots", required_slots) < required_slots:
        return (
            _workspace_failure(n, nrhs, required_workspace, reason="vector_slot_limit"),
            0,
            required_workspace,
            False,
        )
    b = [engine.from_host(b_host[:, column]) for column in range(nrhs)]
    rhs_norms = np.array([engine.norm(value) for value in b])
    basis = []
    ortho_started = time.perf_counter()
    for value in b:
        work, _, norm = engine.orthogonalize(
            basis,
            engine.copy(value),
            reorthogonalize=2,
            tolerance=options.breakdown_tolerance,
        )
        if norm > options.breakdown_tolerance and len(basis) < n:
            basis.append(_normalized_vector(engine, work, norm))
    ortho_seconds = time.perf_counter() - ortho_started
    rank = len(basis)
    rank_deficient = rank < nrhs
    if rank == 0:
        results = []
        for column in range(nrhs):
            norm = rhs_norms[column]
            target = max(options.atol, options.rtol * norm)
            converged = norm <= target
            solution = engine.zeros()
            if converged and consumers[column] is not None:
                consumers[column](engine, solution)
            results.append(
                SolveResult(
                    immutable(engine.to_host(solution)),
                    converged,
                    norm,
                    _relative_residual(norm, norm),
                    0,
                    "zero_rhs" if converged else "rank_deficient_rhs",
                    (norm,),
                    0,
                    0,
                    ortho_seconds,
                    0.0,
                    required_workspace,
                    rank=0,
                    basis=np.empty((n, 0)),
                )
            )
        return tuple(results), 0, required_workspace, rank_deficient

    q_initial = len(basis)
    ortho_started = time.perf_counter()
    initial_coefficients = np.array(
        [[engine.dot(vector, value) for value in b] for vector in basis]
    )
    ortho_seconds += time.perf_counter() - ortho_started
    hbar = np.zeros((max_columns, max_columns))
    solution = [engine.zeros() for _ in b]
    action_seconds = 0.0
    actions = 0
    iterations = 0
    history = [float(norm) for norm in rhs_norms]
    last_start = 0
    breakdown = False

    def apply(value: typing.Any) -> typing.Any:
        # Match the scalar solver's action-only scope. Residual subtraction
        # and norms can synchronize too, but belong to complete solve time.
        nonlocal action_seconds, actions
        begin = time.perf_counter()
        result = engine.apply(operator, value)
        action_seconds += time.perf_counter() - begin
        actions += 1
        return result

    while iterations < options.max_iterations:
        q = len(basis)
        block = basis[last_start:q]
        images = [apply(vector) for vector in block]
        h_top = np.zeros((q, len(block)))
        work = []
        ortho_started = time.perf_counter()
        for column, image in enumerate(images):
            residual, coefficients, _ = engine.orthogonalize(
                basis,
                image,
                reorthogonalize=options.reorthogonalize,
                tolerance=options.breakdown_tolerance,
            )
            h_top[:, column] = coefficients
            work.append(residual)
        # Release image aliases before building the QR and replacement panel.
        images.clear()
        new_basis, h_bottom = _block_range_factor(
            engine,
            work,
            tolerance=options.breakdown_tolerance,
            capacity=max_columns - q,
        )
        ortho_seconds += time.perf_counter() - ortho_started
        keep = len(new_basis)
        hbar[:q, last_start:q] = h_top
        if keep:
            hbar[q : q + keep, last_start:q] = h_bottom
            basis.extend(new_basis)
        q_new = len(basis)
        projected = hbar[:q_new, :q]
        for column in range(nrhs):
            rhs_projected = np.zeros(q_new)
            rhs_projected[:q_initial] = initial_coefficients[:, column]
            coefficients, *_ = np.linalg.lstsq(projected, rhs_projected, rcond=None)
            solution[column] = engine.combination(
                engine.zeros(), basis[:q], coefficients, None
            )
        norms = np.array(
            [
                engine.norm(engine.subtract(value, apply(candidate)))
                for value, candidate in zip(b, solution, strict=True)
            ]
        )
        history = [float(max(old, new)) for old, new in zip(history, norms)]
        targets = np.maximum(
            options.atol,
            options.rtol * rhs_norms,
        )
        if np.all(norms <= targets):
            break
        if keep == 0:
            breakdown = True
            break
        iterations += keep
        last_start = q
    results = []
    # Explicit publication occurs once after iteration. All per-RHS results
    # may share the same immutable basis; no solve/recycle step uses this copy.
    published_basis = immutable(
        engine.stack_host(basis) if collect_basis else np.empty((n, 0))
    )
    final_norms = [
        engine.norm(engine.subtract(value, apply(candidate)))
        for value, candidate in zip(b, solution, strict=True)
    ]
    for column, norm in enumerate(final_norms):
        target = max(options.atol, options.rtol * rhs_norms[column])
        converged = norm <= target
        if converged:
            reason = "converged"
        elif breakdown:
            reason = "breakdown"
        else:
            reason = "max_iterations"
        if converged and consumers[column] is not None:
            consumers[column](engine, solution[column])
        results.append(
            SolveResult(
                immutable(engine.to_host(solution[column])),
                converged,
                norm,
                _relative_residual(norm, float(rhs_norms[column])),
                iterations,
                reason,
                tuple(history),
                actions,
                0,
                ortho_seconds,
                action_seconds,
                required_workspace,
                rank=rank,
                basis=published_basis,
            )
        )
    return tuple(results), actions, required_workspace, rank_deficient


def _solve_many_impl(
    operator: typing.Any,
    rhs: typing.Any,
    *,
    strategy: typing.Any = "sequential",
    options: typing.Any = None,
    recycle: typing.Any = None,
    preconditioner: typing.Any = None,
    raise_on_failure: typing.Any = False,
    collect_basis: bool = True,
    solution_consumers: typing.Any = None,
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
    consumers = (
        (None,) * values.shape[1]
        if solution_consumers is None
        else tuple(solution_consumers)
    )
    if len(consumers) != values.shape[1] or any(
        consumer is not None and not callable(consumer) for consumer in consumers
    ):
        raise ValueError("solution_consumers must match the multi-RHS column count")
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
        engine = getattr(operator, "_krylov_engine", None)
        workspace = getattr(engine, "solver_workspace", nullcontext)
        with workspace():
            results, actions, peak, rank_deficient = _block_solve(
                operator,
                values,
                replace(options, max_workspace_bytes=available),
                collect_basis=collect_basis,
                solution_consumers=consumers,
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
                collect_basis=collect_basis,
                solution_consumer=consumers[column],
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


def solve_many(
    operator: typing.Any,
    rhs: typing.Any,
    *,
    strategy: typing.Any = "sequential",
    options: typing.Any = None,
    recycle: typing.Any = None,
    preconditioner: typing.Any = None,
    raise_on_failure: typing.Any = False,
    collect_basis: bool = True,
    solution_consumers: typing.Any = None,
) -> typing.Any:
    """Solve multiple RHS with one shared algorithm and explicit publication.

    An automatic recycled space follows the vector engine and is released on
    every exit, including exceptions retained by a traceback. Caller-supplied
    spaces retain their existing host/resident ownership. ``collect_basis``
    controls final diagnostics, independently of resident recycle updates.
    """
    owned_recycle = strategy == "recycled" and recycle is None
    if owned_recycle:
        engine = getattr(operator, "_krylov_engine", None)
        recycle = KrylovRecycleSpace(
            operator.problem,
            vector_engine=engine if getattr(engine, "resident", False) else None,
        )
    try:
        return _solve_many_impl(
            operator,
            rhs,
            strategy=strategy,
            options=options,
            recycle=recycle,
            preconditioner=preconditioner,
            raise_on_failure=raise_on_failure,
            collect_basis=collect_basis,
            solution_consumers=solution_consumers,
        )
    finally:
        if owned_recycle:
            recycle.close()
