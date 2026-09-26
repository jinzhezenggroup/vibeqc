"""Admission and representable-result regressions for DF source cotangents."""

import numpy as np
import pytest

from tools.vibeqc_cc.df_gradient import DFThreeIndexCotangent, pullback_df_three_index


def test_finite_large_metric_is_not_overflowed_by_symmetrization() -> None:
    result = pullback_df_three_index(
        np.zeros((1, 1, 1)),
        np.array([[1e308]]),
        np.ones((1, 1)),
        [DFThreeIndexCotangent((0,), (0,), np.ones((1, 1, 1)))],
        relative_threshold=1e-8,
    )
    np.testing.assert_allclose(result.bar_a, 1e-154, rtol=1e-13, atol=0)


def test_finite_large_source_cotangent_is_not_overflowed_by_projection() -> None:
    result = pullback_df_three_index(
        np.zeros((1, 1, 1)),
        np.ones((1, 1)),
        np.ones((1, 1)),
        [DFThreeIndexCotangent((0,), (0,), np.full((1, 1, 1), 1e308))],
        relative_threshold=1e-8,
    )
    np.testing.assert_allclose(result.bar_a, 1e308, rtol=1e-13, atol=0)


def test_budget_includes_simultaneous_raw_source_buffers() -> None:
    raw = np.zeros((16, 16, 1))
    with pytest.raises(MemoryError, match="logical numeric bytes"):
        pullback_df_three_index(
            raw,
            np.ones((1, 1)),
            np.ones((16, 1)),
            [DFThreeIndexCotangent((0,), (0,), np.ones((1, 1, 1)))],
            relative_threshold=1e-8,
            max_bytes=2 * raw.nbytes - 1,
        )


def test_genuinely_nonfinite_pullback_remains_rejected() -> None:
    with np.errstate(over="ignore", invalid="ignore"), pytest.raises(ValueError):
        pullback_df_three_index(
            np.zeros((1, 1, 1)),
            np.ones((1, 1)),
            np.full((1, 1), 2.0),
            [DFThreeIndexCotangent((0,), (0,), np.full((1, 1, 1), 1e308))],
            relative_threshold=1e-8,
        )
