"""Fail-closed denominator diagnostics for projected local-CC pair equations.

Local pair equations may encounter small orbital-energy denominators or empty
virtual spaces.  Those conditions are scientific diagnostics, not invitations
to clip values or silently drop pairs.  This module records them against the
exact :class:`PairSpace` gauges while leaving the supplied denominator matrices
untouched.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy as np

from .common import checked_budget, number
from .spaces import PairSpace


class LocalCCDenominatorError(ValueError):
    """One or more pair equations have inadmissible denominator diagnostics."""


@dataclass(frozen=True)
class PairDenominatorMetric:
    """Small-denominator diagnostics for one occupied-pair PNO gauge."""

    pair: tuple[int, int]
    gauge_identity: str
    rank: int
    minimum_absolute: float | None
    small_count: int
    zero_count: int
    empty_virtual_space: bool

    @property
    def admissible(self) -> bool:
        """Require a nonempty space and no denominator below the threshold."""
        return not self.empty_virtual_space and self.small_count == 0


@dataclass(frozen=True)
class LocalCCDenominatorReport:
    """Deterministic denominator report for one local occupied frame."""

    reference_id: str
    localization_id: str
    threshold: float
    metrics: tuple[PairDenominatorMetric, ...]
    numeric_bytes: int

    @property
    def safe(self) -> bool:
        return all(metric.admissible for metric in self.metrics)

    @property
    def offending_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(metric.pair for metric in self.metrics if not metric.admissible)


def _denominator_array(
    value: typing.Any, *, rank: int, pair: tuple[int, int]
) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"pair {pair} denominators must be real")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"pair {pair} denominators must be numeric") from error
    if array.shape != (rank, rank):
        raise ValueError(
            f"pair {pair} denominators must have shape {(rank, rank)}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"pair {pair} denominators must be finite")
    return array


def inspect_pair_denominators(
    spaces: typing.Iterable[typing.Any],
    denominators: typing.Iterable[typing.Any],
    *,
    minimum_absolute: typing.Any = 1e-8,
    budget_bytes: typing.Any = 128 << 20,
) -> LocalCCDenominatorReport:
    """Inspect projected pair denominators without clipping or pair deletion.

    ``denominators`` must already represent the physical denominator matrices
    for the exact pair gauges in ``spaces``.  Entries with absolute value below
    ``minimum_absolute`` are diagnosed as small; an exactly equal denominator is
    admitted.  Rank-zero pair spaces are reported explicitly as empty and are
    therefore not solver-admissible.

    The byte budget covers only the declared dense float64 denominator matrices.
    It is checked from pair ranks before any denominator object is coerced to a
    NumPy array.  No modified denominator matrix is produced by this API.
    """

    pair_spaces = tuple(spaces)
    values = tuple(denominators)
    if not pair_spaces:
        raise ValueError("denominator diagnostics require a nonempty pair-space set")
    if len(pair_spaces) != len(values):
        raise ValueError("pair spaces and denominators must have the same length")
    if any(not isinstance(space, PairSpace) for space in pair_spaces):
        raise TypeError("denominator diagnostics require canonical PairSpace records")

    reference_id = pair_spaces[0].reference_id
    localization_id = pair_spaces[0].localization_id
    if any(
        space.reference_id != reference_id or space.localization_id != localization_id
        for space in pair_spaces
    ):
        raise ValueError(
            "pair denominators must share one reference/localized occupied frame"
        )

    pairs = tuple(space.pair for space in pair_spaces)
    if len(set(pairs)) != len(pairs):
        raise ValueError("denominator diagnostics require unique occupied pairs")

    threshold = number(minimum_absolute, "minimum absolute denominator", positive=True)
    budget = checked_budget(budget_bytes)
    required_bytes = sum(8 * space.rank * space.rank for space in pair_spaces)
    if required_bytes > budget:
        raise MemoryError(
            "declared pair denominators need "
            f"{required_bytes} numeric bytes, exceeding budget_bytes={budget}"
        )

    metrics: list[PairDenominatorMetric] = []
    paired = zip(pair_spaces, values, strict=True)
    for space, value in sorted(paired, key=lambda item: item[0].pair):
        array = _denominator_array(value, rank=space.rank, pair=space.pair)
        if array.size == 0:
            minimum = None
            small_count = 0
            zero_count = 0
        else:
            absolute = np.abs(array)
            minimum = float(np.min(absolute))
            small_count = int(np.count_nonzero(absolute < threshold))
            zero_count = int(np.count_nonzero(array == 0.0))
        metrics.append(
            PairDenominatorMetric(
                pair=space.pair,
                gauge_identity=space.gauge_identity,
                rank=space.rank,
                minimum_absolute=minimum,
                small_count=small_count,
                zero_count=zero_count,
                empty_virtual_space=space.rank == 0,
            )
        )

    return LocalCCDenominatorReport(
        reference_id=reference_id,
        localization_id=localization_id,
        threshold=threshold,
        metrics=tuple(metrics),
        numeric_bytes=required_bytes,
    )


def require_safe_denominators(report: typing.Any) -> None:
    """Reject unsafe pair equations without changing their denominators."""

    if not isinstance(report, LocalCCDenominatorReport):
        raise TypeError("denominator gate requires LocalCCDenominatorReport")
    if report.safe:
        return
    pairs = ", ".join(f"({i}, {j})" for i, j in report.offending_pairs)
    raise LocalCCDenominatorError(
        "local-CC pair equations contain small denominators or empty virtual spaces: "
        f"{pairs}"
    )
