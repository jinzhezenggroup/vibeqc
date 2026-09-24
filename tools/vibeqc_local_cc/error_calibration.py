"""Calibration-only discarded-space diagnostics for local-correlation research.

The records in this module compare PNO occupation loss with an *observed* MP2
full-space recovery error.  They deliberately do not predict local-CC error and
do not turn discarded occupation weight into an energy bound.  The full-space
pair energies are an independent calibration oracle supplied by ``LocalMP2Result``;
production refinement must not require that oracle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .mp2 import LocalMP2Result, PairMP2Result


@dataclass(frozen=True)
class PairDiscardedSpaceCalibration:
    """One pair's gauge-invariant truncation descriptors and observed MP2 error."""

    pair: tuple[int, int]
    space_identity: str
    retained_rank: int
    full_rank: int
    retained_occupation_weight: float
    discarded_occupation_weight: float
    discarded_occupation_fraction: float
    full_space_correction: float
    observed_absolute_error: float
    rank_crossing: bool

    @property
    def stable_branch(self) -> bool:
        """Whether this point is away from the declared PNO threshold crossing."""
        return not self.rank_crossing


@dataclass(frozen=True)
class LocalMP2DiscardedSpaceCalibration:
    """Calibration table for one exact reference/Hamiltonian/localized frame.

    ``observed_global_absolute_error`` includes cancellation between occupied
    pairs.  ``pairwise_absolute_error_sum`` intentionally retains the uncancelled
    diagnostic.  Neither quantity is a certified local-CC error bound.
    """

    reference_id: str
    hamiltonian_id: str
    localization_id: str
    pairs: tuple[PairDiscardedSpaceCalibration, ...]
    local_correlation_energy: float
    full_virtual_correlation_energy: float
    canonical_correlation_energy: float
    full_space_correction: float
    observed_global_absolute_error: float
    pairwise_absolute_error_sum: float

    @property
    def stable_branch(self) -> bool:
        """Require every calibrated PNO selection to be off its rank threshold."""
        return all(pair.stable_branch for pair in self.pairs)


def _occupation_weights(pair: PairMP2Result) -> tuple[float, float, float]:
    values = pair.space.occupation_eigenvalues
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError(
            f"pair {pair.space.pair} has a nonfinite PNO occupation spectrum"
        )

    # Pair-density spectra are positive semidefinite mathematically.  PairSpace
    # permits tiny negative roundoff down to -1e-12; clamp only that admitted
    # numerical noise before forming a dimensionless discarded-weight heuristic.
    positive = [max(0.0, float(value)) for value in values]
    # Sum disjoint spectral subsets directly: total - retained can erase a
    # small but nonzero discarded population beside the retained one.
    retained_indices = set(pair.space.retained_indices)
    try:
        retained = math.fsum(positive[index] for index in retained_indices)
        discarded = math.fsum(
            value
            for index, value in enumerate(positive)
            if index not in retained_indices
        )
        total = math.fsum((retained, discarded))
    except OverflowError as error:
        raise ValueError(
            f"pair {pair.space.pair} has nonfinite summed occupation weights"
        ) from error
    fraction = discarded / total if total > 0.0 else 0.0
    return retained, discarded, fraction


def _pair_calibration(pair: PairMP2Result) -> PairDiscardedSpaceCalibration:
    retained, discarded, fraction = _occupation_weights(pair)
    correction = pair.full_virtual_pair_energy - pair.energy
    absolute = abs(correction)
    if not math.isfinite(correction) or not math.isfinite(absolute):
        raise ValueError(f"pair {pair.space.pair} has a nonfinite calibration error")
    return PairDiscardedSpaceCalibration(
        pair=pair.space.pair,
        space_identity=pair.space.identity,
        retained_rank=pair.space.rank,
        full_rank=len(pair.space.occupation_eigenvalues),
        retained_occupation_weight=retained,
        discarded_occupation_weight=discarded,
        discarded_occupation_fraction=fraction,
        full_space_correction=correction,
        observed_absolute_error=absolute,
        rank_crossing=pair.space.rank_crossing,
    )


def calibrate_discarded_space(
    result: object,
    *,
    oracle_tolerance: float = 1e-10,
) -> LocalMP2DiscardedSpaceCalibration:
    """Compare discarded PNO weights with independently observed MP2 errors.

    The full-virtual pair-energy decomposition must reproduce the canonical MP2
    comparator before any calibration row is published.  This prevents a stale
    or mismatched per-pair oracle from being mistaken for truncation evidence.
    The function performs no fit and returns no predicted error or acceptance
    decision; later estimators must be calibrated and validated separately.
    """
    if not isinstance(result, LocalMP2Result):
        raise TypeError("discarded-space calibration requires a LocalMP2Result")
    if isinstance(oracle_tolerance, bool) or not isinstance(
        oracle_tolerance, (int, float)
    ):
        raise TypeError("oracle_tolerance must be a positive finite real number")
    tolerance = float(oracle_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("oracle_tolerance must be a positive finite real number")

    ordered = tuple(sorted(result.pairs, key=lambda pair: pair.space.pair))
    full_virtual = math.fsum(pair.full_virtual_pair_energy for pair in ordered)
    scale = max(1.0, abs(full_virtual), abs(result.canonical_correlation_energy))
    if abs(full_virtual - result.canonical_correlation_energy) > tolerance * scale:
        raise ValueError(
            "full-virtual pair calibration oracle does not reproduce canonical MP2"
        )

    pairs = tuple(_pair_calibration(pair) for pair in ordered)
    correction = full_virtual - result.correlation_energy
    observed = abs(correction)
    pairwise = math.fsum(pair.observed_absolute_error for pair in pairs)
    if not all(
        math.isfinite(value) for value in (full_virtual, correction, observed, pairwise)
    ):
        raise ValueError("discarded-space calibration produced nonfinite totals")

    return LocalMP2DiscardedSpaceCalibration(
        reference_id=result.reference_id,
        hamiltonian_id=result.hamiltonian_id,
        localization_id=result.localization_id,
        pairs=pairs,
        local_correlation_energy=result.correlation_energy,
        full_virtual_correlation_energy=full_virtual,
        canonical_correlation_energy=result.canonical_correlation_energy,
        full_space_correction=correction,
        observed_global_absolute_error=observed,
        pairwise_absolute_error_sum=pairwise,
    )
