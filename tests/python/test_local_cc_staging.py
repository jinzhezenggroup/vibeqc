"""Bounded host staging for homogeneous local-CC pair-state buckets."""

from __future__ import annotations

import numpy as np
import pytest

from tools.vibeqc_local_cc.buckets import PairStateBucket, plan_pair_state_buckets
from tools.vibeqc_local_cc.spaces import PairSpace
from tools.vibeqc_local_cc.staging import gather_pair_state_bucket


def _space(
    pair: tuple[int, int],
    rank: int,
    *,
    reference_id: str = "reference",
    localization_id: str = "localized",
    flip_first_column: bool = False,
) -> PairSpace:
    virtual_rank = rank + 1
    columns = np.eye(virtual_rank, dtype=np.float64)[:, 1:]
    if flip_first_column:
        columns[:, 0] *= -1.0
    return PairSpace(
        reference_id=reference_id,
        localization_id=localization_id,
        virtual_domain_id=f"domain-{pair[0]}-{pair[1]}",
        pair=pair,
        columns=columns,
        occupation_eigenvalues=np.array([0.0, *([1.0] * rank)]),
        retained_indices=tuple(range(1, virtual_rank)),
        occupation_threshold=0.5,
        cluster_tolerance=1e-12,
        rank_crossing=False,
        keep_full_space=False,
    )


def _bucket(
    spaces: list[PairSpace], *, states_per_pair: int = 2
) -> PairStateBucket:
    (bucket,) = plan_pair_state_buckets(
        spaces,
        state_matrices_per_pair=states_per_pair,
        max_pairs_per_bucket=16,
        budget_bytes=1 << 20,
    )
    return bucket


def test_pair_state_gather_uses_bucket_order_and_owns_immutable_copy() -> None:
    spaces = [_space((1, 2), 2), _space((0, 1), 2)]
    bucket = _bucket(spaces)
    first = np.full((2, 2), 1.0, dtype=np.float64)
    second = np.full((2, 2), 2.0, dtype=np.float64)
    third = np.full((2, 2), 3.0, dtype=np.float64)
    fourth = np.full((2, 2), 4.0, dtype=np.float64)

    packed = gather_pair_state_bucket(
        bucket,
        {space.pair: space for space in spaces},
        {
            (1, 2): (third, fourth),
            (0, 1): (first, second),
        },
        budget_bytes=bucket.numeric_bytes,
    )

    assert bucket.pairs == ((0, 1), (1, 2))
    assert packed.shape == (2, 2, 2, 2)
    assert packed.dtype == np.float64
    assert packed.flags.c_contiguous
    assert not packed.flags.writeable
    assert packed.nbytes == bucket.numeric_bytes
    np.testing.assert_array_equal(packed[0, 0], np.full((2, 2), 1.0))
    np.testing.assert_array_equal(packed[1, 1], np.full((2, 2), 4.0))

    first[0, 0] = 99.0
    assert packed[0, 0, 0, 0] == 1.0


def test_pair_state_gather_preflights_budget_before_state_array_access() -> None:
    class ExplodingArray:
        def __array__(self) -> np.ndarray:
            raise AssertionError("state conversion must not run before budget admission")

    space = _space((0, 1), 2)
    bucket = _bucket([space], states_per_pair=1)

    with pytest.raises(MemoryError, match="exceeds its declared numeric budget"):
        gather_pair_state_bucket(
            bucket,
            {space.pair: space},
            {space.pair: (ExplodingArray(),)},
            budget_bytes=bucket.numeric_bytes - 1,
        )


def test_pair_state_gather_rejects_stale_pair_gauge_before_copy() -> None:
    planned = _space((0, 1), 2)
    stale_gauge = _space((0, 1), 2, flip_first_column=True)
    bucket = _bucket([planned], states_per_pair=1)

    with pytest.raises(ValueError, match="gauge does not match"):
        gather_pair_state_bucket(
            bucket,
            {stale_gauge.pair: stale_gauge},
            {planned.pair: (np.eye(2, dtype=np.float64),)},
            budget_bytes=bucket.numeric_bytes,
        )


@pytest.mark.parametrize(
    "space",
    [
        _space((0, 1), 2, reference_id="other-reference"),
        _space((0, 1), 2, localization_id="other-localized"),
    ],
)
def test_pair_state_gather_rejects_stale_parent_provenance(space: PairSpace) -> None:
    planned = _space((0, 1), 2)
    bucket = _bucket([planned], states_per_pair=1)

    with pytest.raises(ValueError, match="provenance does not match"):
        gather_pair_state_bucket(
            bucket,
            {space.pair: space},
            {planned.pair: (np.eye(2, dtype=np.float64),)},
            budget_bytes=bucket.numeric_bytes,
        )


@pytest.mark.parametrize(
    ("matrix", "error", "match"),
    [
        ([[1.0, 0.0], [0.0, 1.0]], TypeError, "already be NumPy arrays"),
        (np.eye(3, dtype=np.float64), ValueError, "must have shape"),
        (np.eye(2, dtype=np.float32), TypeError, "already be float64"),
        (np.eye(2, dtype=np.complex128), TypeError, "already be float64"),
        (
            np.ascontiguousarray(np.arange(8, dtype=np.float64).reshape(2, 4))[:, ::2],
            ValueError,
            "C-contiguous",
        ),
        (np.array([[1.0, np.nan], [0.0, 1.0]]), ValueError, "must be finite"),
    ],
)
def test_pair_state_gather_rejects_state_requiring_hidden_conversion(
    matrix: object, error: type[Exception], match: str
) -> None:
    space = _space((0, 1), 2)
    bucket = _bucket([space], states_per_pair=1)

    with pytest.raises(error, match=match):
        gather_pair_state_bucket(
            bucket,
            {space.pair: space},
            {space.pair: (matrix,)},
            budget_bytes=bucket.numeric_bytes,
        )


def test_pair_state_gather_requires_exact_pair_keys_and_matrix_count() -> None:
    first, second = _space((0, 1), 2), _space((1, 2), 2)
    bucket = _bucket([first, second], states_per_pair=2)
    matrix = np.eye(2, dtype=np.float64)
    pair_spaces = {first.pair: first, second.pair: second}

    with pytest.raises(ValueError, match="keys must exactly match"):
        gather_pair_state_bucket(
            bucket,
            pair_spaces,
            {first.pair: (matrix, matrix)},
            budget_bytes=bucket.numeric_bytes,
        )

    with pytest.raises(ValueError, match="matrix count"):
        gather_pair_state_bucket(
            bucket,
            pair_spaces,
            {
                first.pair: (matrix,),
                second.pair: (matrix, matrix),
            },
            budget_bytes=bucket.numeric_bytes,
        )
