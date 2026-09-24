"""Exact pair-space coordinate transfers for bounded local-correlation prototypes."""

import typing
from dataclasses import dataclass

import numpy as np

from tools.vibeqc_posthf.reference import immutable

from .common import checked_budget, fingerprint
from .spaces import PairSpace


@dataclass(frozen=True, eq=False)
class PairTransfer:
    """A provenance-bound overlap map from one PNO gauge to another.

    ``overlap`` is ``Q_target.T @ Q_source``. It changes coordinates only; it
    does not assert that two truncated pair spaces span the same subspace or
    that a projected local-CC residual is converged.
    """

    reference_id: str
    localization_id: str
    source_pair: tuple[int, int]
    target_pair: tuple[int, int]
    source_gauge_id: str
    target_gauge_id: str
    overlap: np.ndarray

    def __post_init__(self) -> None:
        overlap = immutable(self.overlap)
        if overlap.ndim != 2:
            raise ValueError("pair transfer overlap must be a matrix")
        for value in (
            self.reference_id,
            self.localization_id,
            self.source_gauge_id,
            self.target_gauge_id,
        ):
            if not isinstance(value, str) or not value:
                raise ValueError("pair transfer requires complete parent identities")
        for pair in (self.source_pair, self.target_pair):
            if (
                len(pair) != 2
                or any(type(index) is not int or index < 0 for index in pair)
                or pair[0] > pair[1]
            ):
                raise ValueError("pair transfer requires ordered occupied-pair labels")
        object.__setattr__(self, "overlap", overlap)

    @property
    def identity(self) -> str:
        return fingerprint(
            {
                "reference": self.reference_id,
                "localized": self.localization_id,
                "source_pair": self.source_pair,
                "target_pair": self.target_pair,
                "source_gauge": self.source_gauge_id,
                "target_gauge": self.target_gauge_id,
            },
            overlap=self.overlap,
        )


def pair_transfer(source: PairSpace, target: PairSpace) -> PairTransfer:
    """Build the exact coordinate map between compatible local pair gauges."""
    if not isinstance(source, PairSpace) or not isinstance(target, PairSpace):
        raise TypeError("pair transfer requires PairSpace inputs")
    if source.reference_id != target.reference_id:
        raise ValueError("pair transfer cannot cross electronic references")
    if source.localization_id != target.localization_id:
        raise ValueError("pair transfer cannot cross localized occupied frames")
    overlap = target.overlap(source)
    return PairTransfer(
        source.reference_id,
        source.localization_id,
        source.pair,
        target.pair,
        source.gauge_identity,
        target.gauge_identity,
        overlap,
    )


def project_pair_matrix(
    values: typing.Any,
    source: PairSpace,
    target: PairSpace,
    *,
    budget_bytes: int = 128 << 20,
) -> np.ndarray:
    """Project a rank-two pair tensor into the target PNO coordinates.

    If ``O = Q_target.T @ Q_source``, the returned tensor is ``O T O.T``.
    This is an exact coordinate transformation when both spaces span the same
    virtual subspace and an explicitly projected initial/intermediate tensor
    otherwise. It is not a local-CC equation or convergence certificate.
    """
    transfer = pair_transfer(source, target)
    source_rank = source.rank
    target_rank = target.rank
    array = immutable(values, shape=(source_rank, source_rank))
    peak_elements = (
        source_rank * source_rank
        + target_rank * source_rank
        + target_rank * target_rank
    )
    peak_bytes = 8 * peak_elements
    if peak_bytes > checked_budget(budget_bytes):
        raise MemoryError(f"pair projection needs {peak_bytes} numeric bytes")
    return immutable(transfer.overlap @ array @ transfer.overlap.T)
