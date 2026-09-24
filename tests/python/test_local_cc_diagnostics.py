"""Pair-local physical residual diagnostics for the local-CC baseline."""

from __future__ import annotations

import numpy as np
import pytest

from tools.vibeqc_local_cc.diagnostics import evaluate_pair_residuals
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


def test_pair_residual_report_is_pair_local_and_deterministic() -> None:
    first = _space((1, 2), 2)
    second = _space((0, 0), 1)

    report = evaluate_pair_residuals(
        [first, second],
        [np.array([[3.0, 4.0], [0.0, 0.0]]), np.array([[0.5]])],
        tolerance=2.0,
        budget_bytes=1024,
    )

    assert report.reference_id == "reference"
    assert report.localization_id == "localized"
    assert report.numeric_bytes == 40
    assert tuple(metric.pair for metric in report.metrics) == ((0, 0), (1, 2))
    assert report.metrics[0].rms == pytest.approx(0.5)
    assert report.metrics[0].maximum_absolute == pytest.approx(0.5)
    assert report.metrics[1].rms == pytest.approx(2.5)
    assert report.metrics[1].maximum_absolute == pytest.approx(4.0)
    assert report.metrics[1].gauge_identity == first.gauge_identity
    assert report.worst_metric.pair == (1, 2)
    assert not report.converged


def test_pair_residual_convergence_requires_every_pair_rms() -> None:
    spaces = [_space((0, 1), 1), _space((0, 2), 1)]
    residuals = [np.array([[1e-10]]), np.array([[2e-4]])]

    loose = evaluate_pair_residuals(spaces, residuals, tolerance=2e-4)
    strict = evaluate_pair_residuals(spaces, residuals, tolerance=1e-4)

    assert loose.converged
    assert not strict.converged
    assert strict.worst_metric.pair == (0, 2)


def test_empty_pair_space_has_zero_physical_residual_norm() -> None:
    report = evaluate_pair_residuals(
        [_space((0, 0), 0)],
        [np.empty((0, 0))],
        tolerance=1e-8,
        budget_bytes=1,
    )

    assert report.numeric_bytes == 0
    assert report.metrics[0].rank == 0
    assert report.metrics[0].rms == 0.0
    assert report.metrics[0].maximum_absolute == 0.0
    assert report.converged


def test_pair_residual_budget_fails_before_array_coercion() -> None:
    class BombResidual:
        def __array__(self, dtype: object = None) -> np.ndarray:
            raise AssertionError(f"residual was coerced with dtype={dtype}")

    with pytest.raises(MemoryError, match="exceeding budget_bytes"):
        evaluate_pair_residuals(
            [_space((0, 1), 2)],
            [BombResidual()],
            tolerance=1e-7,
            budget_bytes=31,
        )


@pytest.mark.parametrize(
    ("residual", "match"),
    [
        (np.zeros((2, 1)), "shape"),
        (np.array([[0.0, np.nan], [0.0, 0.0]]), "finite"),
        (np.ones((2, 2), dtype=np.complex128), "real"),
    ],
)
def test_pair_residuals_reject_invalid_physical_arrays(
    residual: np.ndarray, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        evaluate_pair_residuals(
            [_space((0, 1), 2)],
            [residual],
            tolerance=1e-7,
        )


def test_pair_residual_norm_is_stable_for_large_finite_values() -> None:
    maximum = np.finfo(np.float64).max
    report = evaluate_pair_residuals(
        [_space((0, 1), 2)],
        [np.full((2, 2), maximum)],
        tolerance=maximum,
    )

    assert report.metrics[0].rms == maximum
    assert report.metrics[0].maximum_absolute == maximum
    assert report.converged


@pytest.mark.parametrize(
    ("spaces", "residuals", "match"),
    [
        ([], [], "nonempty"),
        ([_space((0, 1), 1)], [], "same length"),
        (
            [_space((0, 1), 1), _space((0, 1), 1)],
            [np.zeros((1, 1)), np.zeros((1, 1))],
            "unique occupied pairs",
        ),
        (
            [_space((0, 1), 1), _space((0, 2), 1, reference_id="other")],
            [np.zeros((1, 1)), np.zeros((1, 1))],
            "one reference/localized occupied frame",
        ),
        (
            [_space((0, 1), 1), _space((0, 2), 1, localization_id="other")],
            [np.zeros((1, 1)), np.zeros((1, 1))],
            "one reference/localized occupied frame",
        ),
    ],
)
def test_pair_residuals_fail_closed_on_incompatible_state(
    spaces: list[PairSpace], residuals: list[np.ndarray], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        evaluate_pair_residuals(spaces, residuals, tolerance=1e-7)


def test_pair_residuals_require_canonical_pair_spaces() -> None:
    with pytest.raises(TypeError, match="canonical PairSpace"):
        evaluate_pair_residuals([object()], [np.zeros((1, 1))], tolerance=1e-7)  # type: ignore[list-item]


@pytest.mark.parametrize("tolerance", [0.0, -1.0, np.inf, True])
def test_pair_residuals_reject_invalid_tolerance(tolerance: object) -> None:
    with pytest.raises((TypeError, ValueError), match="pair residual tolerance"):
        evaluate_pair_residuals(
            [_space((0, 1), 1)],
            [np.zeros((1, 1))],
            tolerance=tolerance,
        )
