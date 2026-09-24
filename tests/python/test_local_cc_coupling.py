"""Pair-space coupling regressions for issue #183."""

import numpy as np
import pytest

from tools.vibeqc_local_cc.coupling import pair_transfer, project_pair_matrix
from tools.vibeqc_local_cc.spaces import PairSpace


def _pair_space(
    columns: np.ndarray,
    *,
    pair: tuple[int, int] = (0, 0),
    reference: str = "reference",
    localization: str = "localized",
    threshold: float = 0.0,
    keep_full: bool = True,
) -> PairSpace:
    rank = columns.shape[1]
    if keep_full:
        spectrum = np.linspace(0.1, 0.1 * columns.shape[0], columns.shape[0])
        retained = tuple(range(columns.shape[0]))
    else:
        spectrum = np.array([0.001, 0.2, 0.3])
        retained = (1, 2)
        assert rank == 2
    return PairSpace(
        reference,
        localization,
        "domain",
        pair,
        columns,
        spectrum,
        retained,
        threshold,
        1e-12,
        False,
        keep_full,
    )


def test_full_space_pair_transfer_preserves_physical_tensor() -> None:
    theta = 0.37
    rotation = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    source = _pair_space(np.eye(3))
    target = _pair_space(rotation, pair=(0, 1))
    values = np.array([[1.0, 0.2, -0.1], [0.4, 2.0, 0.3], [0.7, -0.2, 0.5]])

    projected = project_pair_matrix(values, source, target)

    expected = rotation.T @ values @ rotation
    np.testing.assert_allclose(projected, expected, atol=1e-14, rtol=0.0)
    np.testing.assert_allclose(
        target.columns @ projected @ target.columns.T,
        source.columns @ values @ source.columns.T,
        atol=2e-14,
        rtol=0.0,
    )


def test_truncated_pair_transfer_is_explicit_projection() -> None:
    source = _pair_space(np.eye(3))
    target = _pair_space(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        pair=(1, 1),
        threshold=0.1,
        keep_full=False,
    )
    values = np.arange(9.0).reshape(3, 3)

    projected = project_pair_matrix(values, source, target)

    expected = target.columns.T @ values @ target.columns
    np.testing.assert_array_equal(projected, expected)


def test_transfer_identity_tracks_both_pair_gauges() -> None:
    source = _pair_space(np.eye(3))
    target = _pair_space(np.eye(3), pair=(0, 1))

    transfer = pair_transfer(source, target)

    np.testing.assert_array_equal(transfer.overlap, np.eye(3))
    assert transfer.source_pair == (0, 0)
    assert transfer.target_pair == (0, 1)
    assert transfer.source_gauge_id == source.gauge_identity
    assert transfer.target_gauge_id == target.gauge_identity
    assert transfer.identity


def test_pair_transfer_rejects_stale_reference_or_localization() -> None:
    source = _pair_space(np.eye(3))
    stale_reference = _pair_space(np.eye(3), reference="other")
    stale_localization = _pair_space(np.eye(3), localization="other")

    with pytest.raises(ValueError, match="electronic references"):
        pair_transfer(source, stale_reference)
    with pytest.raises(ValueError, match="localized occupied frames"):
        pair_transfer(source, stale_localization)


def test_pair_projection_checks_shape_and_budget_before_contraction() -> None:
    source = _pair_space(np.eye(3))
    target = _pair_space(np.eye(3))

    with pytest.raises(ValueError):
        project_pair_matrix(np.ones((2, 2)), source, target)
    with pytest.raises(MemoryError, match="numeric bytes"):
        project_pair_matrix(np.eye(3), source, target, budget_bytes=1)


def test_pair_projection_detaches_inputs_and_outputs() -> None:
    source = _pair_space(np.eye(3))
    target = _pair_space(np.eye(3))
    values = np.eye(3)

    projected = project_pair_matrix(values, source, target)
    values[0, 0] = 9.0

    assert projected[0, 0] == 1.0
    assert not projected.flags.writeable
