"""Bounded deterministic shape buckets for local-CC pair-state staging."""

from __future__ import annotations

import numpy as np
import pytest

from tools.vibeqc_local_cc.buckets import PairBucketKey, plan_pair_state_buckets
from tools.vibeqc_local_cc.spaces import PairSpace


def _space(
    pair: tuple[int, int],
    rank: int,
    *,
    reference_id: str = "reference",
    localization_id: str = "localized",
    rank_crossing: bool = False,
) -> PairSpace:
    virtual_rank = rank + 1
    columns = np.eye(virtual_rank)[:, 1:]
    if rank_crossing:
        eigenvalues = np.array([1.0, *([2.0] * rank)])
        threshold = 1.0
    else:
        eigenvalues = np.array([0.0, *([1.0] * rank)])
        threshold = 0.5
    return PairSpace(
        reference_id=reference_id,
        localization_id=localization_id,
        virtual_domain_id=f"domain-{pair[0]}-{pair[1]}",
        pair=pair,
        columns=columns,
        occupation_eigenvalues=eigenvalues,
        retained_indices=tuple(range(1, virtual_rank)),
        occupation_threshold=threshold,
        cluster_tolerance=1e-12,
        rank_crossing=rank_crossing,
        keep_full_space=False,
    )


def test_pair_state_buckets_group_by_rank_and_pair_class_deterministically() -> None:
    spaces = [
        _space((1, 2), 2),
        _space((0, 0), 2),
        _space((0, 1), 2),
        _space((2, 2), 1),
    ]

    buckets = plan_pair_state_buckets(
        spaces,
        state_matrices_per_pair=3,
        max_pairs_per_bucket=8,
        budget_bytes=1 << 20,
    )

    assert tuple(bucket.key for bucket in buckets) == (
        PairBucketKey(1, True),
        PairBucketKey(2, False),
        PairBucketKey(2, True),
    )
    assert tuple(bucket.pairs for bucket in buckets) == (
        ((2, 2),),
        ((0, 1), (1, 2)),
        ((0, 0),),
    )
    assert tuple(bucket.numeric_bytes for bucket in buckets) == (24, 192, 96)
    assert all(bucket.reference_id == "reference" for bucket in buckets)
    assert all(bucket.localization_id == "localized" for bucket in buckets)
    for bucket in buckets:
        assert len(bucket.gauge_identities) == len(bucket.pairs)
        assert all(bucket.gauge_identities)


def test_pair_state_buckets_split_by_numeric_budget() -> None:
    spaces = [_space((0, 1), 2), _space((0, 2), 2), _space((1, 2), 2)]

    buckets = plan_pair_state_buckets(
        spaces,
        state_matrices_per_pair=2,
        max_pairs_per_bucket=8,
        budget_bytes=128,
    )

    assert tuple(bucket.pairs for bucket in buckets) == (
        ((0, 1), (0, 2)),
        ((1, 2),),
    )
    assert tuple(bucket.numeric_bytes for bucket in buckets) == (128, 64)


def test_pair_state_buckets_respect_explicit_pair_count_ceiling() -> None:
    spaces = [_space((0, 1), 2), _space((0, 2), 2), _space((1, 2), 2)]

    buckets = plan_pair_state_buckets(
        spaces,
        state_matrices_per_pair=1,
        max_pairs_per_bucket=1,
        budget_bytes=1 << 20,
    )

    assert tuple(bucket.pairs for bucket in buckets) == (
        ((0, 1),),
        ((0, 2),),
        ((1, 2),),
    )


def test_pair_state_buckets_preserve_rank_crossing_diagnostics() -> None:
    spaces = [_space((0, 1), 1, rank_crossing=True), _space((0, 2), 1)]

    (bucket,) = plan_pair_state_buckets(
        spaces,
        state_matrices_per_pair=1,
        max_pairs_per_bucket=4,
        budget_bytes=1024,
    )

    assert bucket.pairs == ((0, 1), (0, 2))
    assert bucket.rank_crossings == (True, False)


def test_pair_state_buckets_reject_budget_too_small_for_one_pair() -> None:
    with pytest.raises(MemoryError, match="cannot hold one declared pair state"):
        plan_pair_state_buckets(
            [_space((0, 1), 2)],
            state_matrices_per_pair=2,
            max_pairs_per_bucket=8,
            budget_bytes=63,
        )


def test_pair_state_buckets_reject_duplicate_occupied_pairs() -> None:
    with pytest.raises(ValueError, match="unique occupied pairs"):
        plan_pair_state_buckets(
            [_space((0, 1), 1), _space((0, 1), 2)],
            state_matrices_per_pair=1,
            max_pairs_per_bucket=8,
            budget_bytes=1024,
        )


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (_space((0, 1), 1), _space((0, 2), 1, reference_id="other")),
        (_space((0, 1), 1), _space((0, 2), 1, localization_id="other")),
    ],
)
def test_pair_state_buckets_reject_mixed_parent_frames(
    first: PairSpace, second: PairSpace
) -> None:
    with pytest.raises(ValueError, match="one reference/localized occupied frame"):
        plan_pair_state_buckets(
            [first, second],
            state_matrices_per_pair=1,
            max_pairs_per_bucket=8,
            budget_bytes=1024,
        )


def test_pair_state_buckets_require_canonical_pair_spaces() -> None:
    with pytest.raises(TypeError, match="canonical PairSpace"):
        plan_pair_state_buckets(
            [object()],
            state_matrices_per_pair=1,
            max_pairs_per_bucket=8,
            budget_bytes=1024,
        )


@pytest.mark.parametrize(
    ("spaces", "matrices", "maximum", "budget", "match"),
    [
        ([], 1, 1, 1024, "nonempty"),
        ([_space((0, 1), 1)], True, 1, 1024, "state_matrices_per_pair"),
        ([_space((0, 1), 1)], 0, 1, 1024, "state_matrices_per_pair"),
        ([_space((0, 1), 1)], 1, True, 1024, "max_pairs_per_bucket"),
        ([_space((0, 1), 1)], 1, 0, 1024, "max_pairs_per_bucket"),
        ([_space((0, 1), 1)], 1, 1, 0, "numeric budget"),
    ],
)
def test_pair_state_buckets_fail_closed_on_invalid_limits(
    spaces: list[PairSpace],
    matrices: object,
    maximum: object,
    budget: object,
    match: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        plan_pair_state_buckets(
            spaces,
            state_matrices_per_pair=matrices,
            max_pairs_per_bucket=maximum,
            budget_bytes=budget,
        )
