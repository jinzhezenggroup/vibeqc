"""Bounded contiguous staging buffers for local-CC pair-state buckets.

The shape planner in :mod:`tools.vibeqc_local_cc.buckets` decides which pair
states may execute together.  This module owns the next host-side boundary:
copying already validated pair-local FP64 matrices into one deterministic
contiguous buffer without changing pair gauges or silently allocating an
unbounded conversion workspace.

The explicit byte budget covers the packed output buffer only.  Caller-owned
input matrices remain outside that budget, and this routine therefore rejects
inputs that would require dtype/layout conversion rather than hiding an
additional numerical copy.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .buckets import PairStateBucket
from .common import checked_budget
from .spaces import PairSpace


def _pair_mapping(value: object, *, name: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping keyed by occupied pair")
    return value


def _require_exact_pair_keys(
    mapping: Mapping[object, object], bucket: PairStateBucket, *, name: str
) -> None:
    expected = set(bucket.pairs)
    actual = set(mapping)
    if actual != expected:
        missing = tuple(sorted(expected - actual))
        extra = tuple(sorted(actual - expected, key=repr))
        raise ValueError(
            f"{name} keys must exactly match the bucket pairs; "
            f"missing={missing}, extra={extra}"
        )


def _validated_pair_spaces(
    bucket: PairStateBucket, spaces: Mapping[object, object]
) -> dict[tuple[int, int], PairSpace]:
    _require_exact_pair_keys(spaces, bucket, name="pair-space")
    result: dict[tuple[int, int], PairSpace] = {}
    for pair, expected_gauge in zip(bucket.pairs, bucket.gauge_identities, strict=True):
        space = spaces[pair]
        if not isinstance(space, PairSpace):
            raise TypeError("pair-state staging requires canonical PairSpace records")
        if (
            space.pair != pair
            or space.reference_id != bucket.reference_id
            or space.localization_id != bucket.localization_id
        ):
            raise ValueError("pair-space provenance does not match the bucket")
        if space.rank != bucket.key.rank or space.gauge_identity != expected_gauge:
            raise ValueError("pair-space gauge does not match the bucket")
        result[pair] = space
    return result


def _validated_matrix(matrix: object, *, rank: int) -> np.ndarray:
    if not isinstance(matrix, np.ndarray):
        raise TypeError("pair-state matrices must already be NumPy arrays")
    array = matrix
    if array.shape != (rank, rank):
        raise ValueError(f"pair-state matrix must have shape {(rank, rank)}")
    if array.dtype != np.dtype(np.float64):
        raise TypeError("pair-state matrices must already be float64")
    if not array.flags.c_contiguous:
        raise ValueError("pair-state matrices must already be C-contiguous")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError("pair-state matrices must be finite")
    return array


def gather_pair_state_bucket(
    bucket: object,
    spaces: object,
    states: object,
    *,
    budget_bytes: object,
) -> np.ndarray:
    """Pack one planned pair bucket into a deterministic native-ready buffer.

    ``states`` maps each occupied pair to exactly
    ``bucket.state_matrices_per_pair`` square matrices expressed in that pair's
    current PNO gauge.  The returned array has shape
    ``(npair, nstate, rank, rank)`` in ``bucket.pairs`` order and owns an
    immutable C-contiguous float64 copy.

    The output allocation is preflighted against ``budget_bytes`` before any
    state matrix is coerced or inspected.  Pair-space provenance and exact gauge
    identities are then checked before numerical state is copied.  Inputs that
    need dtype or layout conversion are rejected so this boundary cannot hide a
    second unaccounted dense numerical buffer.
    """
    if not isinstance(bucket, PairStateBucket):
        raise TypeError("pair-state staging requires a canonical PairStateBucket")
    budget = checked_budget(budget_bytes)
    if bucket.numeric_bytes > budget:
        raise MemoryError(
            "pair-state staging buffer exceeds its declared numeric budget: "
            f"required={bucket.numeric_bytes}, budget={budget}"
        )

    pair_spaces = _pair_mapping(spaces, name="pair-space")
    _validated_pair_spaces(bucket, pair_spaces)
    pair_states = _pair_mapping(states, name="pair-state")
    _require_exact_pair_keys(pair_states, bucket, name="pair-state")

    packed = np.empty(
        (
            len(bucket.pairs),
            bucket.state_matrices_per_pair,
            bucket.key.rank,
            bucket.key.rank,
        ),
        dtype=np.float64,
        order="C",
    )
    if packed.nbytes != bucket.numeric_bytes:
        raise RuntimeError("pair-state bucket byte accounting drifted from staging")

    for pair_index, pair in enumerate(bucket.pairs):
        matrices = pair_states[pair]
        if not isinstance(matrices, (list, tuple)):
            raise TypeError("each pair state must be a list or tuple of matrices")
        if len(matrices) != bucket.state_matrices_per_pair:
            raise ValueError("pair-state matrix count does not match the bucket plan")
        for state_index, matrix in enumerate(matrices):
            packed[pair_index, state_index] = _validated_matrix(
                matrix, rank=bucket.key.rank
            )

    packed.setflags(write=False)
    return packed
