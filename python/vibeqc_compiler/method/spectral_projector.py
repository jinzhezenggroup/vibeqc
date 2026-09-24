"""Gauge-invariant fixed-rank spectral-projector response.

This reference adapter reuses the validated eigensystem, branch decision, and
retained/discarded divided differences prepared by ``matrix_function.py``.
It differentiates only the spectral projector for a locally fixed retained
membership. It does not differentiate rank selection, claim smoothness through
threshold crossings, or define a local-correlation force by itself.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field

import numpy as np

from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.method.matrix_function import MatrixFunctionEvaluation

VERSION = "fixed-rank-spectral-projector-v1"
_FLOAT64_BYTES = 8


def projector_logical_workspace_bytes(size: int) -> int:
    """Return the bounded logical FP64-array footprint used by this reference."""
    if type(size) is not int or size <= 0 or size > 2**31 - 1:
        raise ValueError("projector size must be a positive bounded integer")
    required = _FLOAT64_BYTES * (5 * size * size + 2 * size)
    if required > 2**63 - 1:
        raise ValueError("projector size exceeds signed-64-bit logical byte capacity")
    return required


def _immutable(array: typing.Any) -> np.ndarray:
    value = np.asarray(array, dtype=np.float64, order="C")
    return np.frombuffer(value.tobytes(), dtype=np.float64).reshape(value.shape)


def _checked_seed(value: typing.Any, size: int, *, symmetric: bool) -> np.ndarray:
    seed = np.asarray(value)
    if seed.shape != (size, size) or seed.dtype != np.dtype("float64"):
        raise ValueError(
            "projector seed requires a real float64 square array of the declared size"
        )
    if not np.isfinite(seed).all():
        raise ValueError("projector seed must be finite")
    if symmetric:
        scale = float(np.max(np.abs(seed)))
        if scale and np.max(np.abs(seed / scale - seed.T / scale)) > 1e-12:
            raise ValueError("projector tangent must be symmetric")
    return seed


@dataclass(frozen=True)
class FixedRankProjectorEvaluation:
    """One immutable projector-response state tied to a matrix-function snapshot."""

    parent_identity: str
    size: int
    rank: int
    retained: tuple[bool, ...]
    relative_gap: float
    vectors: np.ndarray = field(repr=False, compare=False)
    projector: np.ndarray = field(repr=False, compare=False)
    divided: np.ndarray = field(repr=False, compare=False)
    max_bytes: int = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.parent_identity, str) or not self.parent_identity:
            raise ValueError("projector response requires a parent spectral identity")
        if type(self.size) is not int or self.size <= 0:
            raise ValueError("projector response requires a positive matrix size")
        if type(self.rank) is not int or not 0 <= self.rank <= self.size:
            raise ValueError("projector response rank is inconsistent with its size")
        if len(self.retained) != self.size or sum(self.retained) != self.rank:
            raise ValueError("projector response retained membership is inconsistent")
        if (
            type(self.relative_gap) is not float
            or not np.isfinite(self.relative_gap)
            or self.relative_gap <= 0
        ):
            raise ValueError("projector response requires a resolved spectral branch")
        if (
            type(self.max_bytes) is not int
            or self.max_bytes < projector_logical_workspace_bytes(self.size)
            or self.max_bytes > 2**63 - 1
        ):
            raise ValueError("projector response logical workspace budget is invalid")
        arrays = (
            (self.vectors, (self.size, self.size), "vectors"),
            (self.projector, (self.size, self.size), "projector"),
            (self.divided, (self.size, self.size), "divided differences"),
        )
        for value, shape, label in arrays:
            if value.shape != shape or value.dtype != np.dtype("float64"):
                raise ValueError(f"projector response {label} has an invalid layout")
            if not np.isfinite(value).all() or value.flags.writeable:
                raise ValueError(
                    f"projector response {label} must be finite and immutable"
                )

    @property
    def manifest(self) -> dict[str, typing.Any]:
        return {
            "kind": "fixed_rank_spectral_projector",
            "version": VERSION,
            "parent_matrix_function": self.parent_identity,
            "size": self.size,
            "rank": self.rank,
            "retained": list(self.retained),
            "relative_gap": self.relative_gap,
            "dtype": "float64",
            "inner_product": "full-frobenius",
            "fixed_membership": True,
            "derivative_orders": [1],
        }

    @property
    def identity(self) -> str:
        return canonical_hash(self.manifest)

    def _response(self, seed: typing.Any, *, symmetric: bool) -> np.ndarray:
        value = _checked_seed(seed, self.size, symmetric=symmetric)
        value = 0.5 * value + 0.5 * value.T
        rotated = self.vectors.T @ value @ self.vectors
        weighted = self.divided * rotated
        response = self.vectors @ weighted @ self.vectors.T
        response = 0.5 * response + 0.5 * response.T
        if not np.isfinite(response).all():
            raise ValueError("nonfinite fixed-rank projector response")
        return _immutable(response)

    def jvp(self, tangent: typing.Any) -> np.ndarray:
        """Differentiate projector directions with retained membership fixed."""
        return self._response(tangent, symmetric=True)

    def vjp(self, cotangent: typing.Any) -> np.ndarray:
        """Apply the full-Frobenius adjoint on the symmetric matrix domain."""
        return self._response(cotangent, symmetric=False)


def prepare_fixed_rank_projector(
    state: MatrixFunctionEvaluation,
    *,
    max_bytes: int = 256 << 20,
) -> FixedRankProjectorEvaluation:
    """Reuse a validated truncated matrix-function state as projector response.

    The matrix-function state owns eigensystem validation and the cutoff/rank
    branch. Its retained/discarded divided differences contain

    ``f(lambda_r) / (lambda_r - lambda_d)``.

    Dividing those cross terms by the retained matrix-function value recovers
    the gauge-invariant projector divided difference
    ``1 / (lambda_r - lambda_d)`` without a second eigendecomposition or any
    division by gaps internal to the retained or discarded subspaces.
    """
    if not isinstance(state, MatrixFunctionEvaluation):
        raise TypeError("projector response requires MatrixFunctionEvaluation")
    if state.spec.relative_threshold is None:
        raise ValueError(
            "projector response requires an explicit fixed-rank threshold branch"
        )
    required = projector_logical_workspace_bytes(state.spec.size)
    if type(max_bytes) is not int or not required <= max_bytes <= 2**63 - 1:
        raise ValueError("projector response logical workspace budget exceeded")

    retained = np.asarray(state.retained, dtype=bool)
    discarded = ~retained
    vectors = np.asarray(state.vectors, dtype=np.float64)
    retained_vectors = vectors[:, retained]
    projector = retained_vectors @ retained_vectors.T

    divided = np.zeros((state.spec.size, state.spec.size), dtype=np.float64)
    if state.rank and np.any(discarded):
        function_values = np.einsum(
            "pi,pq,qi->i",
            retained_vectors,
            state.value,
            retained_vectors,
            optimize=True,
        )
        if not np.isfinite(function_values).all() or np.any(function_values <= 0.0):
            raise ValueError("invalid retained matrix-function spectral values")
        cross = state.divided[np.ix_(retained, discarded)] / function_values[:, None]
        if not np.isfinite(cross).all():
            raise ValueError("nonfinite retained/discarded projector response")
        divided[np.ix_(retained, discarded)] = cross
        divided[np.ix_(discarded, retained)] = cross.T

    return FixedRankProjectorEvaluation(
        parent_identity=state.identity,
        size=state.spec.size,
        rank=state.rank,
        retained=state.retained,
        relative_gap=float(state.relative_gap),
        vectors=_immutable(vectors),
        projector=_immutable(projector),
        divided=_immutable(divided),
        max_bytes=max_bytes,
    )
