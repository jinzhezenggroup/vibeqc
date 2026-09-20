"""Typed symmetric matrix-function rule and CPU reference execution (#466).

The spectral rule is custom; its linear response contractions are ordinary
TensorIR. This is not native factorization, an automatic TensorIR AD node, or a
complete molecular force implementation. Only real FP64 inverse square roots
and first-order full-Frobenius JVP/VJP are supported.
"""

from __future__ import annotations

import hashlib
import typing
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    einsum,
    execute,
    input_tensor,
    multiply,
    transpose,
)

from .matrix_function_contract import RULE_VERSIONS

VERSION = "symmetric-matrix-function-v1"


@dataclass(frozen=True)
class SymmetricMatrixFunctionSpec:
    """Backend-neutral custom-rule contract, separate from DFT energy components.

    ``relative_threshold=None`` means the full SPD inverse square root. Otherwise
    eigenvalues <= threshold * largest are discarded. The retained *membership*
    is locally fixed, not the spectral projector: cross-subspace response stays.
    ``branch_guard`` is relative to the largest eigenvalue, not an absolute unit.
    """

    size: int
    matrix_identity: str
    relative_threshold: float | None = None
    branch_guard: float = 1e-12
    function: str = "inverse_sqrt"
    version: str = VERSION
    kind: ClassVar[str] = "symmetric_matrix_function"

    def __post_init__(self) -> None:
        if type(self.size) is not int or self.size <= 0 or self.size > 2**31 - 1:
            raise ValueError("matrix size must be a positive bounded integer")
        if self.logical_workspace_bytes > 2**63 - 1:
            raise ValueError("matrix size exceeds signed-64-bit logical byte capacity")
        if (
            not isinstance(self.matrix_identity, str)
            or not self.matrix_identity.strip()
        ):
            raise ValueError("matrix identity must be a nonempty string")
        if self.function not in RULE_VERSIONS or self.version != VERSION:
            raise ValueError("unsupported matrix function or schema version")
        for name in ("relative_threshold", "branch_guard"):
            value = getattr(self, name)
            if name == "relative_threshold" and value is None:
                continue
            if type(value) is not float or not np.isfinite(value) or not 0 < value < 1:
                raise ValueError(
                    f"{name} must be a finite float strictly between 0 and 1"
                )
        if (
            self.relative_threshold is not None
            and self.branch_guard >= self.relative_threshold
        ):
            raise ValueError("branch guard must be smaller than the relative threshold")

    def to_payload(self) -> dict:
        return {
            "kind": self.kind,
            "version": self.version,
            "function": self.function,
            "size": self.size,
            "matrix_identity": self.matrix_identity,
            "relative_threshold": self.relative_threshold,
            "branch_guard": self.branch_guard,
            "dtype": "float64",
            "inner_product": "full-frobenius",
            "derivative_rule": RULE_VERSIONS[self.function],
            "derivative_orders": [1],
        }

    @classmethod
    def from_payload(cls, payload: dict) -> SymmetricMatrixFunctionSpec:
        if not isinstance(payload, dict):
            raise TypeError("matrix-function payload must be an object")
        required = {
            "kind",
            "version",
            "function",
            "size",
            "matrix_identity",
            "relative_threshold",
            "branch_guard",
            "dtype",
            "inner_product",
            "derivative_rule",
            "derivative_orders",
        }
        if set(payload) != required:
            raise ValueError("matrix-function payload has missing or unknown fields")
        spec = cls(
            **{
                key: payload[key]
                for key in (
                    "size",
                    "matrix_identity",
                    "relative_threshold",
                    "branch_guard",
                    "function",
                    "version",
                )
            }
        )
        # Compare canonical JSON, not Python equality (True == 1).
        if canonical_hash(payload) != canonical_hash(spec.to_payload()):
            raise ValueError(
                "unsupported matrix-function derivative/dtype/inner-product contract"
            )
        return spec

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    @property
    def logical_workspace_bytes(self) -> int:
        """Conservative CPU logical-array admission, NOT an RSS/LAPACK bound.

        Covers retained spectral state, copied validation/outputs, and the
        TensorIR interpreter's live arrays. Caller storage, JSON/Python objects,
        and opaque NumPy/BLAS/LAPACK workspaces are outside this reference scope.
        """
        return 8 * (32 * self.size**2 + 8 * self.size)

    def response_program(self) -> Program:
        """Emit the self-adjoint first derivative using existing TensorIR ops.

        Spectral feeds are *fixed coefficients at one primal state*. Applying
        TensorIR AD to this program differentiates only the seed, not the matrix
        function a second time. Native spectral-state preparation is separate.
        """
        physical = IndexSpace("matrix", "ao", self.size)
        spectral = IndexSpace("spectrum", "auxiliary", self.size)
        p, q = Index("p", physical), Index("q", physical)
        i, j = Index("i", spectral), Index("j", spectral)
        vectors = input_tensor(
            "vectors", TensorSpec((p, i), role="parameter", differentiable=False)
        )
        divided = input_tensor(
            "divided", TensorSpec((i, j), role="parameter", differentiable=False)
        )
        seed = input_tensor(
            "seed", TensorSpec((p, q), role="input", differentiable=True)
        )
        sym = add(seed, transpose(seed, (1, 0)), coefficients=("1/2", "1/2"))
        left = einsum("pi,pq->iq", vectors, sym)
        rotated = einsum("iq,qj->ij", left, vectors)
        weighted = multiply(divided, rotated)
        right = einsum("pi,ij->pj", vectors, weighted)
        result = einsum("pj,qj->pq", right, vectors)
        result = add(result, transpose(result, (1, 0)), coefficients=("1/2", "1/2"))
        return Program(
            {"response": result},
            provenance={
                "matrix_function": self.identity,
                "rule": RULE_VERSIONS[self.function],
                "spectral_state": "fixed-first-order-coefficients",
            },
        )

    def prepare(
        self,
        matrix: np.ndarray,
        *,
        max_bytes: int = 256 << 20,
        expected_rank: int | None = None,
    ) -> MatrixFunctionEvaluation:
        """Snapshot one checked CPU state; optionally enforce a prior branch rank."""
        if (
            type(max_bytes) is not int
            or not self.logical_workspace_bytes <= max_bytes <= 2**63 - 1
        ):
            raise ValueError(
                "matrix-function logical workspace budget exceeded or invalid"
            )
        if expected_rank is not None and (
            type(expected_rank) is not int or not 1 <= expected_rank <= self.size
        ):
            raise ValueError("expected rank must be an integer within the matrix size")
        matrix = _matrix(matrix, self.size, symmetric=True)
        # Canonical symmetrization only follows a relative symmetry check.
        matrix = 0.5 * matrix + 0.5 * matrix.T
        values, vectors = np.linalg.eigh(matrix)
        if not np.isfinite(values).all() or not np.isfinite(vectors).all():
            raise ValueError("nonfinite matrix-function eigensystem")
        largest = float(values[-1])
        if largest <= 0:
            raise ValueError("matrix function has no positive spectral subspace")
        threshold = self.relative_threshold
        if threshold is None:
            if values[0] <= 0:
                raise ValueError("full inverse square root requires an SPD matrix")
        elif values[0] < -self.branch_guard * largest:
            raise ValueError(
                "fixed-rank inverse square root rejects an indefinite matrix"
            )
        scaled = values / largest
        if threshold is None:
            retained = np.ones(self.size, dtype=bool)
            gap = float(scaled[0])
        else:
            retained = scaled > threshold
            gap = float(np.min(np.abs(scaled - threshold)))
        if gap <= self.branch_guard:
            raise ValueError("matrix-function rank branch is unresolved at the cutoff")
        rank = int(np.count_nonzero(retained))
        if expected_rank is not None and rank != expected_rank:
            raise ValueError("matrix-function rank changed from the expected branch")
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            try:
                function = np.zeros_like(values)
                retained_values = values[retained]
                if self.function == "inverse_sqrt":
                    roots = np.sqrt(retained_values)
                    function[retained] = 1.0 / roots
                    # Rationalized divided differences avoid cancellation,
                    # including exact/near degeneracy in the retained subspace.
                    retained_divided = (
                        -1.0
                        / roots[:, None]
                        / roots[None, :]
                        / (roots[:, None] + roots[None, :])
                    )
                else:
                    function[retained] = 1.0 / retained_values
                    retained_divided = (
                        -1.0 / retained_values[:, None] / retained_values[None, :]
                    )
                divided = np.zeros((self.size, self.size))
                divided[np.ix_(retained, retained)] = retained_divided
                discarded = ~retained
                cross = function[retained, None] / (
                    values[retained, None] - values[None, discarded]
                )
                divided[np.ix_(retained, discarded)] = cross
                divided[np.ix_(discarded, retained)] = cross.T
                value = (vectors * function) @ vectors.T
                value = 0.5 * value + 0.5 * value.T
            except FloatingPointError as error:
                raise ValueError(
                    "nonfinite matrix-function spectral arithmetic"
                ) from error
        if not all(np.isfinite(a).all() for a in (divided, value)):
            raise ValueError("nonfinite matrix-function value or derivative")
        source_hash = hashlib.sha256(matrix.astype("<f8").tobytes()).hexdigest()
        return MatrixFunctionEvaluation(
            self,
            _immutable(value),
            _immutable(vectors),
            _immutable(divided),
            rank,
            tuple(bool(v) for v in retained),
            gap,
            source_hash,
            max_bytes,
        )


def _matrix(
    value: typing.Any, size: typing.Any, *, symmetric: typing.Any
) -> typing.Any:
    value = np.asarray(value)
    if value.shape != (size, size) or value.dtype != np.dtype("float64"):
        raise ValueError(
            "matrix/seed requires a real float64 square array of the declared size"
        )
    if not np.isfinite(value).all():
        raise ValueError("matrix/seed must be finite")
    if symmetric:
        scale = float(np.max(np.abs(value)))
        # Normalize before subtracting, avoiding overflow on large finite inputs.
        if scale and np.max(np.abs(value / scale - value.T / scale)) > 1e-12:
            raise ValueError("matrix/tangent must be symmetric")
    return value


def _immutable(array: typing.Any) -> typing.Any:
    # A bytes-backed view cannot be made writable again by setflags().
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


@dataclass(frozen=True)
class MatrixFunctionEvaluation:
    """Detached CPU spectral snapshot; responses reuse one generated linear map.

    ``value``/spectral arrays are immutable. Rebinding is explicit, rejects rank
    changes, and recomputes the projectors instead of freezing their directions.
    """

    spec: SymmetricMatrixFunctionSpec
    value: np.ndarray = field(repr=False, compare=False)
    vectors: np.ndarray = field(repr=False, compare=False)
    divided: np.ndarray = field(repr=False, compare=False)
    rank: int
    retained: tuple[bool, ...]
    relative_gap: float
    source_hash: str
    max_bytes: int = field(repr=False, compare=False)

    @property
    def manifest(self) -> dict:
        return {
            "spec": self.spec.to_payload(),
            "source_hash": self.source_hash,
            "rank": self.rank,
            "retained": list(self.retained),
            "relative_gap": self.relative_gap,
            "response_program": self.spec.response_program().logical_hash,
            "execution": "numpy-cpu-reference",
        }

    @property
    def identity(self) -> str:
        return canonical_hash(self.manifest)

    def _response(self, seed: typing.Any, *, symmetric: typing.Any) -> typing.Any:
        seed = _matrix(seed, self.spec.size, symmetric=symmetric)
        return execute(
            self.spec.response_program(),
            {
                "vectors": self.vectors,
                "divided": self.divided,
                "seed": seed,
            },
            max_bytes=self.max_bytes - self.value.nbytes * 3,
        ).outputs["response"]

    def jvp(self, tangent: np.ndarray) -> np.ndarray:
        """First directional derivative; the tangent must be symmetric."""
        return self._response(tangent, symmetric=True)

    def vjp(self, cotangent: np.ndarray) -> np.ndarray:
        """Full-Frobenius adjoint; nonsymmetric cotangents are projected once."""
        return self._response(cotangent, symmetric=False)

    def rebind(self, matrix: np.ndarray) -> MatrixFunctionEvaluation:
        """Prepare a new state on a branch with the same retained rank."""
        return self.spec.prepare(
            matrix, max_bytes=self.max_bytes, expected_rank=self.rank
        )
