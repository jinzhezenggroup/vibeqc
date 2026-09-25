"""Fail-closed small-denominator diagnostics for local-CC pair equations."""

from __future__ import annotations

import numpy as np
import pytest

from tools.vibeqc_local_cc.denominators import (
    LocalCCDenominatorError,
    inspect_pair_denominators,
    require_safe_denominators,
)
from tools.vibeqc_local_cc.spaces import PairSpace


def _space(
    pair: tuple[int, int],
    rank: int,
    *,
    reference_id: str = "reference",
    localization_id: str = "localized",
) -> PairSpace:
    virtual_rank = max(1, rank + 1)
    columns = np.eye(virtual_rank)[:, 1 : rank + 1]
    eigenvalues = np.array([0.0, *([1.0] * rank)])
    return PairSpace(
        reference_id=reference_id,
        localization_id=localization_id,
        virtual_domain_id=f"domain-{pair[0]}-{pair[1]}",
        pair=pair,
        columns=columns,
        occupation_eigenvalues=eigenvalues,
        retained_indices=tuple(range(1, rank + 1)),
        occupation_threshold=0.5,
        cluster_tolerance=1e-12,
        rank_crossing=False,
        keep_full_space=False,
    )


def test_denominator_report_is_pair_local_and_deterministic() -> None:
    first = _space((1, 2), 2)
    second = _space((0, 0), 1)

    report = inspect_pair_denominators(
        [first, second],
        [np.array([[-1.0, -0.5], [-0.25, -2.0]]), np.array([[-3.0]])],
        minimum_absolute=0.1,
        budget_bytes=1024,
    )

    assert report.reference_id == "reference"
    assert report.localization_id == "localized"
    assert report.numeric_bytes == 40
    assert tuple(metric.pair for metric in report.metrics) == ((0, 0), (1, 2))
    assert report.metrics[0].minimum_absolute == pytest.approx(3.0)
    assert report.metrics[1].minimum_absolute == pytest.approx(0.25)
    assert report.metrics[1].gauge_identity == first.gauge_identity
    assert report.safe
    assert report.offending_pairs == ()


def test_small_and_zero_denominators_are_reported_without_clipping() -> None:
    matrix = np.array([[0.0, -1e-10], [1e-4, -2.0]], dtype=np.float64)
    original = matrix.copy()

    report = inspect_pair_denominators(
        [_space((0, 1), 2)],
        [matrix],
        minimum_absolute=1e-8,
    )

    metric = report.metrics[0]
    assert metric.small_count == 2
    assert metric.zero_count == 1
    assert metric.minimum_absolute == 0.0
    assert not metric.admissible
    assert not report.safe
    assert report.offending_pairs == ((0, 1),)
    np.testing.assert_array_equal(matrix, original)
    with pytest.raises(LocalCCDenominatorError, match=r"\(0, 1\)"):
        require_safe_denominators(report)
    np.testing.assert_array_equal(matrix, original)


def test_denominator_exactly_at_threshold_is_admitted() -> None:
    report = inspect_pair_denominators(
        [_space((0, 1), 1)],
        [np.array([[-1e-8]])],
        minimum_absolute=1e-8,
    )

    assert report.metrics[0].small_count == 0
    assert report.safe
    assert require_safe_denominators(report) is None


def test_empty_pair_space_is_explicitly_inadmissible() -> None:
    report = inspect_pair_denominators(
        [_space((0, 0), 0)],
        [np.empty((0, 0))],
        minimum_absolute=1e-8,
        budget_bytes=1,
    )

    metric = report.metrics[0]
    assert report.numeric_bytes == 0
    assert metric.rank == 0
    assert metric.minimum_absolute is None
    assert metric.small_count == 0
    assert metric.zero_count == 0
    assert metric.empty_virtual_space
    assert not report.safe


def test_denominator_budget_fails_before_array_coercion() -> None:
    class BombDenominator:
        def __array__(self, dtype: object = None) -> np.ndarray:
            raise AssertionError(f"denominator was coerced with dtype={dtype}")

    with pytest.raises(MemoryError, match="exceeding budget_bytes"):
        inspect_pair_denominators(
            [_space((0, 1), 2)],
            [BombDenominator()],
            minimum_absolute=1e-8,
            budget_bytes=31,
        )


@pytest.mark.parametrize(
    ("denominator", "match"),
    [
        (np.zeros((2, 1)), "shape"),
        (np.array([[1.0, np.nan], [1.0, 1.0]]), "finite"),
        (np.ones((2, 2), dtype=np.complex128), "real"),
    ],
)
def test_denominators_reject_invalid_physical_arrays(
    denominator: np.ndarray, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        inspect_pair_denominators(
            [_space((0, 1), 2)],
            [denominator],
            minimum_absolute=1e-8,
        )


@pytest.mark.parametrize(
    ("spaces", "denominators", "match"),
    [
        ([], [], "nonempty"),
        ([_space((0, 1), 1)], [], "same length"),
        (
            [_space((0, 1), 1), _space((0, 1), 1)],
            [np.ones((1, 1)), np.ones((1, 1))],
            "unique occupied pairs",
        ),
        (
            [_space((0, 1), 1), _space((0, 2), 1, reference_id="other")],
            [np.ones((1, 1)), np.ones((1, 1))],
            "one reference/localized occupied frame",
        ),
        (
            [_space((0, 1), 1), _space((0, 2), 1, localization_id="other")],
            [np.ones((1, 1)), np.ones((1, 1))],
            "one reference/localized occupied frame",
        ),
    ],
)
def test_denominators_fail_closed_on_incompatible_state(
    spaces: list[PairSpace], denominators: list[np.ndarray], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        inspect_pair_denominators(spaces, denominators, minimum_absolute=1e-8)


def test_denominators_require_canonical_pair_spaces() -> None:
    with pytest.raises(TypeError, match="canonical PairSpace"):
        inspect_pair_denominators([object()], [np.ones((1, 1))], minimum_absolute=1e-8)


@pytest.mark.parametrize("threshold", [0.0, -1.0, np.inf, True])
def test_denominators_reject_invalid_threshold(threshold: object) -> None:
    with pytest.raises((TypeError, ValueError), match="minimum absolute denominator"):
        inspect_pair_denominators(
            [_space((0, 1), 1)],
            [np.ones((1, 1))],
            minimum_absolute=threshold,
        )


def test_denominator_gate_requires_typed_report() -> None:
    with pytest.raises(TypeError, match="LocalCCDenominatorReport"):
        require_safe_denominators(object())
