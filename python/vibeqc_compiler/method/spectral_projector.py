"""Gauge-invariant fixed-rank spectral-projector response.

This reference adapter reuses the validated eigensystem, branch decision, and
eigenvalues prepared by ``matrix_function.py``.
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

VERSION = "fixed-rank-spectral-projector-v2"
_FLOAT64_BYTES = 8


def projector_logical_workspace_bytes(size: int) -> int:
    """Conservatively admit owned state and transient logical FP64 arrays.

    Parent/caller storage and opaque NumPy/BLAS workspaces are excluded. This
    bounds the reference array schedule, not process RSS or allocator peaks.
    """
    if type(size) is not int or size <= 0 or size > 2**31 - 1:
        raise ValueError("projector size must be a positive bounded integer")
    required = _FLOAT64_BYTES * (12 * size * size + 4 * size)
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
        if self.rank in (0, self.size):
            # P is constant on this branch. Do not rotate a huge finite seed
            # only to multiply an irrelevant overflowing result by zero.
            return _immutable(np.zeros((self.size, self.size), dtype=np.float64))
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
    branch. Use its retained eigenvalues directly for the cross-subspace
    coefficient ``1 / (lambda_r - lambda_d)``. Dividing parent function divided
    differences by values recovered from the dense matrix loses accuracy for
    ill-conditioned functions and can silently inherit spectral underflow.
    There is no second eigendecomposition or within-subspace gap division.
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

    values = state.eigenvalues
    if (
        values is None
        or values.shape != (state.spec.size,)
        or values.dtype != np.dtype("float64")
        or values.flags.writeable
        or not np.isfinite(values).all()
    ):
        raise ValueError("projector response requires prepared spectral eigenvalues")
    retained = np.asarray(state.retained, dtype=bool)
    discarded = ~retained
    vectors = np.asarray(state.vectors, dtype=np.float64)
    retained_vectors = vectors[:, retained]
    projector = retained_vectors @ retained_vectors.T

    divided = np.zeros((state.spec.size, state.spec.size), dtype=np.float64)
    if state.rank and np.any(discarded):
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            try:
                cross = 1.0 / (values[retained, None] - values[None, discarded])
            except FloatingPointError as error:
                raise ValueError(
                    "nonfinite retained/discarded projector response"
                ) from error
        if not np.isfinite(cross).all() or np.any(cross <= 0.0):
            raise ValueError("invalid retained/discarded projector response")
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
