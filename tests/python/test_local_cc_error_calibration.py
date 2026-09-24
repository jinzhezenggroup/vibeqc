"""Discarded-space calibration must stay separate from production error claims."""

from dataclasses import replace

import numpy as np
import pytest

from tools.vibeqc_local_cc.error_calibration import calibrate_discarded_space
from tools.vibeqc_local_cc.mp2 import LocalMP2Result, LocalSpacePlan, PairMP2Result
from tools.vibeqc_local_cc.spaces import PairSpace


def _pair(
    pair: tuple[int, int],
    *,
    discarded_weight: float,
    full_pair_energy: float,
    crossing: bool = False,
) -> PairMP2Result:
    threshold = discarded_weight if crossing else 0.1
    values = np.array([discarded_weight, 0.5])
    space = PairSpace(
        reference_id="reference",
        localization_id="localized",
        virtual_domain_id="domain",
        pair=pair,
        columns=np.array([[0.0], [1.0]]),
        occupation_eigenvalues=values,
        retained_indices=(1,),
        occupation_threshold=threshold,
        cluster_tolerance=1e-12,
        rank_crossing=crossing,
        keep_full_space=False,
    )
    zeros = np.zeros((1, 1))
    return PairMP2Result(space, zeros, zeros, 0.0, full_pair_energy)


def _result(*pairs: PairMP2Result, canonical: float) -> LocalMP2Result:
    return LocalMP2Result(
        reference_id="reference",
        hamiltonian_id="hamiltonian",
        localization_id="localized",
        pairs=tuple(pairs),
        correlation_energy=0.0,
        canonical_correlation_energy=canonical,
        minimum_absolute_denominator=1.0,
        plan=LocalSpacePlan(4096, 0, 4096, len(pairs), 1),
        provider_peak_bytes=0,
        provider_seconds=0.0,
        pair_transform_seconds=0.0,
        total_seconds=0.0,
        full_space_recovery=False,
    )


def test_calibration_keeps_pair_cancellation_visible() -> None:
    result = _result(
        _pair((1, 1), discarded_weight=0.02, full_pair_energy=-0.2),
        _pair((0, 1), discarded_weight=0.01, full_pair_energy=0.2),
        canonical=0.0,
    )

    report = calibrate_discarded_space(result)

    assert tuple(row.pair for row in report.pairs) == ((0, 1), (1, 1))
    assert report.full_virtual_correlation_energy == pytest.approx(0.0)
    assert report.observed_global_absolute_error == pytest.approx(0.0)
    assert report.pairwise_absolute_error_sum == pytest.approx(0.4)
    assert report.pairs[0].discarded_occupation_weight == pytest.approx(0.01)
    assert report.pairs[0].discarded_occupation_fraction == pytest.approx(0.01 / 0.51)
    assert report.pairs[0].observed_absolute_error == pytest.approx(0.2)
    assert report.stable_branch


def test_rank_crossing_is_retained_as_unstable_calibration_metadata() -> None:
    result = _result(
        _pair((0, 0), discarded_weight=0.1, full_pair_energy=-0.3, crossing=True),
        canonical=-0.3,
    )

    report = calibrate_discarded_space(result)

    assert report.pairs[0].rank_crossing
    assert not report.pairs[0].stable_branch
    assert not report.stable_branch


def test_full_virtual_oracle_must_reproduce_canonical_mp2() -> None:
    result = _result(
        _pair((0, 0), discarded_weight=0.01, full_pair_energy=-0.2),
        canonical=-0.1,
    )

    with pytest.raises(ValueError, match="does not reproduce canonical MP2"):
        calibrate_discarded_space(result)


def test_admitted_negative_roundoff_is_not_reported_as_discarded_weight() -> None:
    pair = _pair((0, 0), discarded_weight=0.01, full_pair_energy=-0.2)
    noisy_space = replace(
        pair.space,
        occupation_eigenvalues=np.array([-1e-13, 0.5]),
    )
    noisy = replace(pair, space=noisy_space)
    result = _result(noisy, canonical=-0.2)

    report = calibrate_discarded_space(result)

    assert report.pairs[0].discarded_occupation_weight == 0.0
    assert report.pairs[0].discarded_occupation_fraction == 0.0


@pytest.mark.parametrize("tolerance", [0.0, -1.0, np.inf, True, "1e-10"])
def test_invalid_oracle_tolerance_fails_closed(tolerance: object) -> None:
    result = _result(
        _pair((0, 0), discarded_weight=0.01, full_pair_energy=-0.2),
        canonical=-0.2,
    )

    with pytest.raises((TypeError, ValueError), match="oracle_tolerance"):
        calibrate_discarded_space(result, oracle_tolerance=tolerance)  # type: ignore[arg-type]


def test_calibration_requires_typed_local_mp2_evidence() -> None:
    with pytest.raises(TypeError, match="LocalMP2Result"):
        calibrate_discarded_space(object())
