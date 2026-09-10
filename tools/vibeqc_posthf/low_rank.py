"""Incrementally refinable Coulomb factors with explicit approximation identity.

The initial CPU implementation is a bounded pivoted-Cholesky baseline over a
column provider, not a dense molecular integral cache. A factorization is an
approximate Hamiltonian even when its diagnostic residual is very small.
Electronic convergence and observable accuracy must be checked separately.
"""

import json
import math
import threading
import time
from dataclasses import asdict, dataclass
from hashlib import sha256

import numpy as np
from vibeqc.profiles import canonical_hash
from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.common.resources import (
    ResourceBudget,
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourceRequest,
    byte_product,
    plan_resources,
)


@dataclass(frozen=True)
class PivotRecord:
    """One committed column; prior columns survive every subsequent extension."""

    rank: int
    pair_index: int
    pivot_diagonal: float
    maximum_residual_diagonal: float
    residual_trace: float
    roundoff_allowance: float
    negative_roundoff_count: int
    maximum_negative_roundoff: float
    column_hash: str


@dataclass(frozen=True)
class RefinementResult:
    """A factor-generation transition requiring fresh consumer residual/history."""

    previous_identity: str
    identity: str
    previous_rank: int
    rank: int
    requested_threshold: float
    status: str
    seconds: float

    @property
    def invalidates_solver_history(self):
        return self.previous_identity != self.identity


class IncrementalCholesky:
    """Retain a fixed-capacity factor prefix and append deterministic pivots.

    ``columns`` supplies a PairSpace, immutable identity, numeric_bytes,
    check(), diagonal(begin,count), and column(pivot,begin,count). It must
    represent the same real symmetric PSD Coulomb matrix throughout its
    lifetime. Native raw sources use CoulombColumns. Tiny arbitrary matrices
    belong in independent test oracles.

    Each successful pivot commits atomically. If a later source read or PSD
    check fails, the completed prefix remains valid and its generation is
    visible; callers must revalidate their solver history. Factor rank never
    silently grows beyond the reserved capacity. Detached exports are tiled
    and contain a single captured generation.

    Negative diagonal residuals larger than a scale/rank-dependent rounding
    allowance fail. Smaller negative residuals are set to zero with explicit
    diagnostics. Schur-column Cauchy checks detect additional PSD violations.
    These finite-precision checks do not certify that an arbitrary source is
    globally PSD or bound the final relaxed energy/force.
    """

    def __init__(
        self,
        columns,
        *,
        rank_capacity,
        pair_tile=256,
        export_rank_tile=1,
        budget=None,
        _execution_requests=(),
    ):
        started = time.perf_counter()
        self._lock = threading.RLock()
        columns.check()
        self._columns = columns
        self._source_identity = columns.identity
        self._space = columns.space
        for value, name in (
            (rank_capacity, "rank capacity"),
            (pair_tile, "pair tile"),
            (export_rank_tile, "export rank tile"),
        ):
            if type(value) is not int or value < (0 if name == "rank capacity" else 1):
                raise ValueError(f"invalid {name}")
        if rank_capacity > self._space.size:
            raise ValueError("rank capacity exceeds the pair-space dimension")
        self._rank_capacity = rank_capacity
        self._pair_tile = min(pair_tile, self._space.size)
        self._export_rank_tile = export_rank_tile
        budget = ResourceBudget() if budget is None else budget
        n = self._space.size
        estimates = tuple(
            ResourceEstimate(name, size, "pageable", 0, 1, kind)
            for name, size, kind in (
                ("raw_source", columns.numeric_bytes, "runtime"),
                ("factor_capacity", byte_product(8, n, rank_capacity), "persistent"),
                # Original/current diagonals, raw/Schur/new columns, candidate
                # residual, validation temporaries and BLAS result vectors.
                ("pair_workspace", byte_product(8, n, 14), "workspace"),
                ("source_tile", byte_product(8, self._pair_tile, 3), "workspace"),
                (
                    "detached_factor_tile",
                    byte_product(16, n, export_rank_tile),
                    "output",
                ),
            )
        )
        request = ResourceRequest(
            "incremental_coulomb_cholesky",
            ResourceIdentity(
                "posthf",
                "pivoted_cholesky",
                "cpu",
                "fp64",
                json.dumps({"source": columns.identity, "nbf": self.space.nbf}),
                (self._space.convention,),
                "incremental_columns_v1",
            ),
            (ResourceCandidate("retained_factor_prefix", "streamed", estimates),),
            (
                "Python object metadata",
                "BLAS runtime overhead",
                "caller-retained detached exports",
            ),
        )
        # A native execution policy contributes its owned allocations before
        # either source reads or host/device allocation can begin.
        self._resource_plan = plan_resources((request, *_execution_requests), budget)
        self._resource_plan.require_feasible()
        self._factors = np.zeros((rank_capacity, n))
        self._original = np.empty(n)
        for begin in range(0, n, self._pair_tile):
            count = min(self._pair_tile, n - begin)
            self._original[begin : begin + count] = self._values(
                columns.diagonal(begin, count), count
            )
        if np.any(self._original < 0):
            raise ValueError("Coulomb source has a negative diagonal")
        self._residual = self._original.copy()
        self._scale = float(np.max(self._original))
        self._history = []
        self._refinements = []
        self._rank = 0
        self._closed = False
        self._factor_digest = sha256()
        self._setup_seconds = time.perf_counter() - started

    @staticmethod
    def _values(value, count):
        value = np.asarray(value)
        if (
            value.dtype != np.float64
            or value.shape != (count,)
            or not np.isfinite(value).all()
        ):
            raise ValueError(
                "column source must return a finite FP64 vector of the requested size"
            )
        return value

    def _check(self):
        if self._closed:
            raise RuntimeError("incremental factorization is closed")
        self._columns.check()
        if self._columns.identity != self._source_identity:
            raise ValueError("Coulomb source identity changed; rebuild factorization")

    @property
    def space(self):
        return self._space

    @property
    def rank(self):
        return self._rank

    @property
    def rank_capacity(self):
        return self._rank_capacity

    @property
    def resource_plan(self):
        return self._resource_plan

    @property
    def identity(self):
        """Hamiltonian identity changes with actual factor values, not a label."""
        with self._lock:
            self._check()
            return canonical_hash(
                {
                    "family": "incremental-pivoted-cholesky-v1",
                    "target": self._source_identity,
                    "pairs": self.space.convention,
                    "precision": "float64",
                    "rank": self.rank,
                    "factor_hash": self._factor_digest.hexdigest(),
                }
            )

    @property
    def hamiltonian_id(self):
        return "pivoted-cholesky:" + self.identity

    @property
    def history(self):
        with self._lock:
            self._check()
            return tuple(self._history)

    def _roundoff(self):
        return 64 * np.finfo(np.float64).eps * self._scale * (self.rank + 1)

    def _project_column(self, column, pivot):
        """Subtract the retained prefix; native policies may keep it resident."""
        if self.rank:
            column -= self._factors[: self.rank].T @ self._factors[: self.rank, pivot]
        return column

    def _commit_native_column(self, column):
        """Execution-policy hook, called after validation and before host commit."""

    def _pivot(self, pivot, diagonal):
        """Validate an entire new Schur column before mutating the prefix."""
        n = self.space.size
        column = np.empty(n)
        for begin in range(0, n, self._pair_tile):
            count = min(self._pair_tile, n - begin)
            column[begin : begin + count] = self._values(
                self._columns.column(pivot, begin, count), count
            )
        column = self._project_column(column, pivot)
        allowance = self._roundoff()
        if abs(column[pivot] - diagonal) > allowance:
            raise ValueError("source diagonal and incremental Schur column disagree")
        bound = np.sqrt(self._residual) * math.sqrt(diagonal)
        if np.any(np.abs(column) > bound + allowance):
            raise ValueError("Coulomb Schur column violates the PSD Cauchy bound")
        column /= math.sqrt(diagonal)
        # A committed pivot coordinate has exactly zero residual by definition.
        # Tiny cancellation at previously selected pivots is checked above and
        # removed before publication so their factor rows remain triangular.
        for previous in self._history:
            column[previous.pair_index] = 0.0
        column[pivot] = math.sqrt(diagonal)
        remaining = self._residual - column * column
        if not np.isfinite(column).all() or not np.isfinite(remaining).all():
            raise FloatingPointError("nonfinite Cholesky update")
        if np.min(remaining) < -allowance:
            raise ValueError("Coulomb update lost positive semidefiniteness")
        negative = remaining < 0
        negative_count = int(np.count_nonzero(negative))
        maximum_negative = max(0.0, -float(np.min(remaining)))
        remaining[negative] = 0.0
        remaining[pivot] = 0.0
        column_bytes = column.astype("<f8", copy=False).tobytes()
        record = PivotRecord(
            self.rank + 1,
            pivot,
            diagonal,
            float(np.max(remaining)),
            float(np.sum(remaining)),
            allowance,
            negative_count,
            maximum_negative,
            sha256(column_bytes).hexdigest(),
        )
        # Everything which can reject scientific data precedes this commit.
        self._commit_native_column(column)
        self._factors[self.rank] = column
        self._residual[:] = remaining
        self._factor_digest.update(column_bytes)
        self._history.append(record)
        self._rank += 1

    def refine(self, threshold, *, maximum_rank=None):
        """Append pivots until the diagonal threshold, rank cap or roundoff floor.

        The threshold is absolute in the normalized Coulomb pair matrix.
        ``roundoff_limited`` explicitly means the requested tolerance was not
        reached. A lower rank limit never truncates or recomputes old factors.
        Equal maxima choose the first pair index deterministically.
        """
        with self._lock:
            self._check()
            if (
                isinstance(threshold, (bool, str))
                or not math.isfinite(threshold)
                or threshold < 0
            ):
                raise ValueError("finite nonnegative absolute threshold required")
            maximum_rank = self.rank_capacity if maximum_rank is None else maximum_rank
            if (
                type(maximum_rank) is not int
                or not self.rank <= maximum_rank <= self.rank_capacity
            ):
                raise ValueError(
                    "maximum rank must preserve the existing prefix and fit capacity"
                )
            started, old_identity, old_rank = (
                time.perf_counter(),
                self.identity,
                self.rank,
            )
            while True:
                pivot = int(np.argmax(self._residual))
                diagonal = float(self._residual[pivot])
                if diagonal <= threshold:
                    status = "threshold_met"
                    break
                if self.rank == maximum_rank:
                    status = "rank_limited"
                    break
                if diagonal <= self._roundoff():
                    status = "roundoff_limited"
                    break
                self._pivot(pivot, diagonal)
            result = RefinementResult(
                old_identity,
                self.identity,
                old_rank,
                self.rank,
                float(threshold),
                status,
                time.perf_counter() - started,
            )
            self._refinements.append(result)
            return result

    def factor_tile(self, begin, count, *, identity=None):
        """Detach at most the reserved rank tile; reject a stale captured generation."""
        with self._lock:
            self._check()
            if identity is not None and identity != self.identity:
                raise ValueError("factor generation changed; rebuild consumer state")
            if (
                type(begin) is not int
                or type(count) is not int
                or min(begin, count) < 0
                or begin > self.rank
                or count > self.rank - begin
                or count > self._export_rank_tile
            ):
                raise ValueError(
                    "factor export exceeds the active rank or reserved tile"
                )
            return immutable(self._factors[begin : begin + count])

    def diagnostics(self):
        """Report conditional tensor residual diagnostics, never observable bounds."""
        with self._lock:
            self._check()
            maximum, trace = (
                float(np.max(self._residual)),
                float(np.sum(self._residual)),
            )
            return {
                "schema": "vibeqc.low-rank.v1",
                "family": "pivoted_cholesky",
                "target_operator_identity": self._source_identity,
                "approximate_hamiltonian_identity": self.hamiltonian_id,
                "pair_convention": self.space.convention,
                "precision": "float64",
                "rank": self.rank,
                "rank_capacity": self.rank_capacity,
                "maximum_residual_diagonal": maximum,
                "residual_trace": trace,
                "conditional_entry_bound": maximum,
                "conditional_spectral_and_frobenius_bound": trace,
                "bound_assumptions": "PSD Schur complement in exact arithmetic; finite-precision residuals are diagnostics, not a certified bound",
                "observable_certification": "unverified",
                "derivatives": "unsupported: truncated pivot/rank policy is not differentiated",
                "history": [asdict(row) for row in self._history],
                "refinements": [asdict(row) for row in self._refinements],
                "setup_seconds": self._setup_seconds,
                "refinement_seconds": sum(row.seconds for row in self._refinements),
            }

    def close(self):
        with self._lock:
            self._closed = True
            self._factors = self._original = self._residual = None

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *_):
        self.close()
