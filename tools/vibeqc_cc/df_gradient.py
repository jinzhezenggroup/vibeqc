"""Factorized DF three-index pullback for issue #158 slice A.

This module owns only the algebraic reverse edge from whitened MO three-index
factors B back to raw AO three-center integrals A and the auxiliary Coulomb
metric M. CCSD/(T)/Lambda cotangents are upstream responsibilities.
"""

from __future__ import annotations

import hashlib
import typing
from dataclasses import dataclass, replace

import numpy as np
from vibeqc_compiler.method.matrix_function import SymmetricMatrixFunctionSpec

from tools.vibeqc_posthf.reference import immutable


@dataclass(frozen=True)
class DFThreeIndexCotangent:
    """Cotangent for one B[Q,p,q] MO block."""

    p: tuple[int, ...]
    q: tuple[int, ...]
    values: np.ndarray


@dataclass(frozen=True)
class DFThreeIndexPullback:
    """Raw DF weights and fixed-rank metric-response diagnostics."""

    bar_a: np.ndarray
    bar_m: np.ndarray
    metric_rank: int
    metric_relative_gap: float
    metric_rule_identity: str
    logical_required_bytes: int


def _fp64(value: typing.Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.dtype("float64") or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite FP64 array")
    return array


def _symmetric(value: np.ndarray, name: str) -> np.ndarray:
    scale = float(np.max(np.abs(value))) if value.size else 0.0
    if scale and np.max(np.abs(value / scale - value.T / scale)) > 1e-12:
        raise ValueError(f"{name} must be symmetric")
    return 0.5 * value + 0.5 * value.T


def pullback_df_three_index(
    raw_a: typing.Any,
    metric: typing.Any,
    coefficients: typing.Any,
    cotangents: typing.Iterable[DFThreeIndexCotangent],
    *,
    relative_threshold: float,
    expected_rank: int | None = None,
    max_bytes: int = 256 << 20,
) -> DFThreeIndexPullback:
    """Reverse B[Q,p,q] = C C A M**(-1/2) into physical raw A/M weights."""

    a = _fp64(raw_a, "raw three-center integrals")
    m = _fp64(metric, "DF metric")
    c = _fp64(coefficients, "MO coefficients")
    blocks = tuple(cotangents)
    if a.ndim != 3 or c.ndim != 2:
        raise ValueError("raw A must be rank-3 and MO coefficients rank-2")
    nao, nmo = c.shape
    if a.shape[0] != nao or a.shape[1] != nao:
        raise ValueError("raw A AO dimensions must match MO coefficients")
    naux = a.shape[2]
    if m.shape != (naux, naux):
        raise ValueError("DF metric dimension must match the A auxiliary axis")
    if not blocks:
        raise ValueError("at least one DF B cotangent is required")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    if type(relative_threshold) is not float or not 0 < relative_threshold < 1:
        raise ValueError(
            "relative_threshold must be a float strictly between zero and one"
        )

    max_block = max_intermediate = max_columns = 0
    checked: list[tuple[tuple[int, ...], tuple[int, ...], np.ndarray]] = []
    for block in blocks:
        if not isinstance(block, DFThreeIndexCotangent):
            raise TypeError("cotangents must contain DFThreeIndexCotangent values")
        p, q = tuple(block.p), tuple(block.q)
        if (
            any(type(i) is not int or not 0 <= i < nmo for i in (*p, *q))
            or len(set(p)) != len(p)
            or len(set(q)) != len(q)
        ):
            raise ValueError("invalid DF B MO columns")
        bar = _fp64(block.values, "DF B cotangent")
        if bar.shape != (naux, len(p), len(q)):
            raise ValueError("DF B cotangent shape does not match its MO columns")
        max_block = max(max_block, bar.size)
        max_intermediate = max(max_intermediate, naux * nao * max(len(p), len(q)))
        max_columns = max(max_columns, len(p) + len(q))
        checked.append((p, q, bar))

    rule = SymmetricMatrixFunctionSpec(
        naux,
        "df-ccsdt-metric:admission",
        relative_threshold=relative_threshold,
    )
    # Include simultaneous accumulators, bounded contraction intermediates,
    # symmetric projection and immutable publication; exclude caller-owned input
    # arrays and opaque BLAS workspace, as in the shared spectral rule.
    logical_required = rule.logical_workspace_bytes + 8 * (
        4 * a.size
        + 6 * naux * naux
        + 4 * max_block
        + 2 * max_intermediate
        + 2 * nao * max_columns
    )
    if logical_required > max_bytes:
        raise MemoryError(
            f"DF three-index pullback requires {logical_required} logical numeric bytes"
        )

    m = _symmetric(m, "DF metric")
    metric_hash = hashlib.sha256(
        np.ascontiguousarray(m, dtype="<f8").tobytes()
    ).hexdigest()
    rule = replace(rule, matrix_identity="df-ccsdt-metric:" + metric_hash)
    evaluation = rule.prepare(
        m,
        max_bytes=max_bytes,
        expected_rank=expected_rank,
    )
    inverse_root = evaluation.value
    bar_a = np.zeros_like(a)
    bar_inverse_root = np.zeros_like(m)

    for p, q, bar_b in checked:
        cp, cq = c[:, p], c[:, q]
        transformed = np.einsum(
            "mp,nq,mnP->Ppq", cp, cq, a, optimize=["einsum_path", (0, 2), (0, 1)]
        )
        bar_transformed = np.einsum("PQ,Qpq->Ppq", inverse_root, bar_b, optimize=True)
        bar_a += np.einsum(
            "mp,nq,Ppq->mnP",
            cp,
            cq,
            bar_transformed,
            optimize=["einsum_path", (0, 2), (0, 1)],
        )
        bar_inverse_root += np.einsum("Ppq,Qpq->PQ", transformed, bar_b, optimize=True)

    # A and M are physical symmetric sources. Project each cotangent exactly once
    # before it reaches the generated raw derivative consumer.
    bar_a = 0.5 * bar_a + 0.5 * bar_a.transpose(1, 0, 2)
    bar_m = evaluation.vjp(bar_inverse_root)
    return DFThreeIndexPullback(
        immutable(bar_a),
        immutable(bar_m),
        evaluation.rank,
        evaluation.relative_gap,
        evaluation.identity,
        logical_required,
    )
