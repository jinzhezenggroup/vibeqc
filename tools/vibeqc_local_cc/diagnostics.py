"""Fail-closed convergence diagnostics for projected local-CC pair residuals.

This module deliberately evaluates only physical residual arrays expressed in the
current :class:`PairSpace` gauges. It does not accept an energy change as a
convergence substitute and it does not transport stale residual history across
pair-space changes.
"""

from __future__ import annotations

import math
import typing
from dataclasses import dataclass

import numpy as np

from .common import checked_budget, number
from .spaces import PairSpace


@dataclass(frozen=True)
class PairResidualMetric:
    """One pair's residual norm in its exact current PNO gauge."""

    pair: tuple[int, int]
    gauge_identity: str
    rank: int
    rms: float
    maximum_absolute: float


@dataclass(frozen=True)
class LocalCCResidualReport:
    """Deterministic pair-local convergence report for one local occupied frame."""

    reference_id: str
    localization_id: str
    tolerance: float
    metrics: tuple[PairResidualMetric, ...]
    numeric_bytes: int

    @property
    def worst_metric(self) -> PairResidualMetric:
        """Return the largest pair RMS, breaking exact ties by pair label."""
        return max(self.metrics, key=lambda metric: (metric.rms, metric.pair))

    @property
    def converged(self) -> bool:
        """Require every projected pair residual RMS to satisfy the tolerance."""
        return self.worst_metric.rms <= self.tolerance


def _residual_array(
    value: typing.Any, *, rank: int, pair: tuple[int, int]
) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"pair {pair} residual must be real")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"pair {pair} residual must be numeric") from error
    if array.shape != (rank, rank):
        raise ValueError(
            f"pair {pair} residual must have shape {(rank, rank)}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"pair {pair} residual must be finite")
    return array


def _stable_rms_and_maximum(array: np.ndarray) -> tuple[float, float]:
    if array.size == 0:
        return 0.0, 0.0
    maximum = float(np.max(np.abs(array)))
    if maximum == 0.0:
        return 0.0, 0.0
    scaled = array / maximum
    rms = maximum * math.sqrt(float(np.mean(scaled * scaled)))
    if not math.isfinite(rms):
        raise ValueError("pair residual norm overflowed despite finite input")
    return rms, maximum


def evaluate_pair_residuals(
    spaces: typing.Iterable[typing.Any],
    residuals: typing.Iterable[typing.Any],
    *,
    tolerance: typing.Any = 1e-7,
    budget_bytes: typing.Any = 128 << 20,
) -> LocalCCResidualReport:
    """Evaluate bounded pair-local projected residual convergence.

    ``residuals`` must already be the physical projected residuals in the exact
    gauges of ``spaces``. The function intentionally accepts no energy-change
    argument: pair convergence cannot be inferred from global energy stability.

    The byte budget covers only the supplied dense float64 residual matrices.
    Amplitudes, DIIS/history, pair-overlap caches, integral providers, and backend
    workspace remain independent resource owners. The complete declared residual
    footprint is checked from pair ranks before any residual is coerced to NumPy.
    """

    pair_spaces = tuple(spaces)
    values = tuple(residuals)
    if not pair_spaces:
        raise ValueError("pair residual diagnostics require a nonempty pair-space set")
    if len(pair_spaces) != len(values):
        raise ValueError("pair spaces and residuals must have the same length")
    if any(not isinstance(space, PairSpace) for space in pair_spaces):
        raise TypeError("pair residual diagnostics require canonical PairSpace records")

    reference_id = pair_spaces[0].reference_id
    localization_id = pair_spaces[0].localization_id
    if any(
        space.reference_id != reference_id or space.localization_id != localization_id
        for space in pair_spaces
    ):
        raise ValueError("pair residuals must share one reference/localized occupied frame")

    pairs = tuple(space.pair for space in pair_spaces)
    if len(set(pairs)) != len(pairs):
        raise ValueError("pair residual diagnostics require unique occupied pairs")

    checked_tolerance = number(tolerance, "pair residual tolerance", positive=True)
    budget = checked_budget(budget_bytes)
    required_bytes = sum(8 * space.rank * space.rank for space in pair_spaces)
    if required_bytes > budget:
        raise MemoryError(
            "declared pair residuals need "
            f"{required_bytes} numeric bytes, exceeding budget_bytes={budget}"
        )

    metrics: list[PairResidualMetric] = []
    paired = zip(pair_spaces, values, strict=True)
    for space, value in sorted(paired, key=lambda item: item[0].pair):
        array = _residual_array(value, rank=space.rank, pair=space.pair)
        rms, maximum = _stable_rms_and_maximum(array)
        metrics.append(
            PairResidualMetric(
                pair=space.pair,
                gauge_identity=space.gauge_identity,
                rank=space.rank,
                rms=rms,
                maximum_absolute=maximum,
            )
        )

    return LocalCCResidualReport(
        reference_id=reference_id,
        localization_id=localization_id,
        tolerance=checked_tolerance,
        metrics=tuple(metrics),
        numeric_bytes=required_bytes,
    )
