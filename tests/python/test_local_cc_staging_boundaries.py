"""Staging must retain finite values, rank-branch provenance and read-only views."""

from __future__ import annotations

import typing
from dataclasses import replace

import numpy as np
import pytest

from tools.vibeqc_local_cc import staging
from tools.vibeqc_local_cc.buckets import PairBucketKey, PairStateBucket
from tools.vibeqc_local_cc.spaces import PairSpace


def _space(*, crossing: bool = False) -> PairSpace:
    return PairSpace(
        reference_id="reference",
        localization_id="localized",
        virtual_domain_id="domain",
        pair=(0, 1),
        columns=np.eye(3, dtype=np.float64)[:, 1:],
        occupation_eigenvalues=np.array([0.0, 0.5 + 5e-13 if crossing else 1.0, 1.0]),
        retained_indices=(1, 2),
        occupation_threshold=0.5,
        cluster_tolerance=1e-12,
        rank_crossing=crossing,
        keep_full_space=False,
    )


def _bucket(space: PairSpace) -> PairStateBucket:
    return PairStateBucket(
        reference_id=space.reference_id,
        localization_id=space.localization_id,
        key=PairBucketKey(2, False),
        pairs=(space.pair,),
        gauge_identities=(space.gauge_identity,),
        rank_crossings=(space.rank_crossing,),
        state_matrices_per_pair=1,
        numeric_bytes=32,
    )


def _gather(space: PairSpace, matrix: np.ndarray) -> np.ndarray:
    bucket = _bucket(space)
    return staging.gather_pair_state_bucket(
        bucket, {space.pair: space}, {space.pair: (matrix,)}, budget_bytes=32
    )


@pytest.mark.parametrize("hidden", [np.nan, np.inf, -np.inf, 5.0])
def test_masked_state_is_rejected_without_discarding_mask(hidden: float) -> None:
    matrix = np.ma.array(
        [[hidden, 1.0], [2.0, 3.0]], mask=[[True, False], [False, False]]
    )
    with pytest.raises(TypeError, match="NumPy arrays"):
        _gather(_space(), matrix)


def test_array_subclass_cannot_override_finiteness_validation() -> None:
    class FiniteOverride(np.ndarray):
        def __array_ufunc__(
            self, ufunc: typing.Any, method: str, *args: typing.Any, **kwargs: typing.Any
        ) -> np.ndarray:
            return np.ones(self.shape, dtype=bool)

    matrix = np.array([[np.nan, 1.0], [2.0, 3.0]]).view(FiniteOverride)
    with pytest.raises(TypeError, match="NumPy arrays"):
        _gather(_space(), matrix)


@pytest.mark.parametrize("crossing", [False, True])
def test_rank_branch_must_match_before_allocating(
    crossing: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    planned = _space(crossing=crossing)
    changed = _space(crossing=not crossing)
    bucket = _bucket(planned)
    # The column gauge is unchanged. The separate rank-branch record is material.
    assert planned.gauge_identity == changed.gauge_identity
    allocations = []
    empty = np.empty

    def observed_empty(*args: typing.Any, **kwargs: typing.Any) -> np.ndarray:
        allocations.append(args)
        return empty(*args, **kwargs)

    monkeypatch.setattr(staging.np, "empty", observed_empty)
    with pytest.raises(ValueError, match="rank branch"):
        staging.gather_pair_state_bucket(
            bucket,
            {changed.pair: changed},
            {changed.pair: (np.eye(2),)},
            budget_bytes=32,
        )
    assert not allocations


@pytest.mark.parametrize("crossing", [False, True])
def test_matching_rank_branch_remains_admitted(crossing: bool) -> None:
    source = np.array([[1.0, 2.0], [3.0, 4.0]])
    result = _gather(_space(crossing=crossing), source)
    np.testing.assert_array_equal(result[0, 0], source)
    source.fill(99.0)
    assert result[0, 0, 0, 0] == 1.0


@pytest.mark.parametrize("view", ["direct", "flat", "view"])
def test_published_array_cannot_reenable_writes(view: str) -> None:
    result = _gather(_space(), np.eye(2))
    candidate = {
        "direct": result, "flat": result.reshape(-1), "view": result.view()
    }[view]
    with pytest.raises(ValueError):
        candidate.setflags(write=True)


def test_readonly_publication_keeps_one_owned_numeric_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocations = []
    empty = np.empty

    def observed_empty(*args: typing.Any, **kwargs: typing.Any) -> np.ndarray:
        result = empty(*args, **kwargs)
        allocations.append(result)
        return result

    monkeypatch.setattr(staging.np, "empty", observed_empty)
    result = _gather(_space(), np.eye(2))
    assert len(allocations) == 1
    assert result.nbytes == allocations[0].nbytes == 32
    assert np.shares_memory(result, allocations[0])
    assert result.flags.c_contiguous


def test_insufficient_budget_precedes_input_inspection() -> None:
    space = _space()
    with pytest.raises(MemoryError, match="numeric budget"):
        staging.gather_pair_state_bucket(
            _bucket(space), object(), object(), budget_bytes=31
        )


def test_gauge_mismatch_remains_rejected() -> None:
    space = _space()
    changed = replace(space, columns=-space.columns)
    with pytest.raises(ValueError, match="gauge"):
        staging.gather_pair_state_bucket(
            _bucket(space),
            {space.pair: changed},
            {space.pair: (np.eye(2),)},
            budget_bytes=32,
        )
