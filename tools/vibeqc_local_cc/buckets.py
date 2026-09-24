"""Deterministic bounded shape buckets for local-CC pair-state staging.

This module does not own local spaces or contraction equations.  It consumes the
canonical :class:`PairSpace` records from ``spaces.py`` and groups fixed pair
states into homogeneous square-rank batches that a later native local-CC solver
can lower onto existing tensor/GEMM schedules.

The byte accounting here is intentionally narrow: it covers only an explicitly
declared number of dense rank-by-rank pair-state matrices.  Integral providers,
pair-overlap caches, DIIS history, and backend workspace remain separate budget
owners and must be accounted for by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

from .common import checked_budget
from .spaces import PairSpace

_FLOAT64_BYTES = 8


@dataclass(frozen=True, order=True)
class PairBucketKey:
    """Native-shape identity for one class of square pair-state matrices."""

    rank: int
    diagonal_pair: bool

    def __post_init__(self) -> None:
        if type(self.rank) is not int or self.rank <= 0:
            raise ValueError("pair bucket rank must be a positive integer")
        if type(self.diagonal_pair) is not bool:
            raise TypeError("pair bucket diagonal flag must be boolean")


@dataclass(frozen=True)
class PairStateBucket:
    """One sequentially executable homogeneous pair-state bucket.

    ``numeric_bytes`` is the exact float64 footprint of
    ``state_matrices_per_pair`` dense ``rank x rank`` matrices for the listed
    pairs.  It is deliberately not a whole-solver peak-memory claim.
    """

    reference_id: str
    localization_id: str
    key: PairBucketKey
    pairs: tuple[tuple[int, int], ...]
    gauge_identities: tuple[str, ...]
    rank_crossings: tuple[bool, ...]
    state_matrices_per_pair: int
    numeric_bytes: int

    def __post_init__(self) -> None:
        if not self.reference_id or not self.localization_id:
            raise ValueError("pair buckets require complete parent identities")
        if not self.pairs or len(self.pairs) != len(self.gauge_identities):
            raise ValueError("pair bucket metadata lengths are inconsistent")
        if len(self.pairs) != len(self.rank_crossings):
            raise ValueError("pair bucket rank-branch metadata is incomplete")
        if (
            type(self.state_matrices_per_pair) is not int
            or self.state_matrices_per_pair <= 0
        ):
            raise ValueError("state matrix count must be a positive integer")
        if any(
            len(pair) != 2
            or any(type(index) is not int or index < 0 for index in pair)
            or pair[0] > pair[1]
            or (pair[0] == pair[1]) != self.key.diagonal_pair
            for pair in self.pairs
        ):
            raise ValueError("pair bucket contains an invalid pair label")
        if len(set(self.pairs)) != len(self.pairs):
            raise ValueError("pair bucket contains duplicate pair labels")
        if any(not identity for identity in self.gauge_identities):
            raise ValueError("pair bucket requires gauge identities for every pair")
        if any(type(crossing) is not bool for crossing in self.rank_crossings):
            raise TypeError("pair bucket rank-crossing flags must be boolean")
        expected = (
            len(self.pairs)
            * self.state_matrices_per_pair
            * self.key.rank
            * self.key.rank
            * _FLOAT64_BYTES
        )
        if type(self.numeric_bytes) is not int or self.numeric_bytes != expected:
            raise ValueError("pair bucket numeric-byte accounting is inconsistent")


@dataclass(frozen=True)
class _PairRecord:
    key: PairBucketKey
    pair: tuple[int, int]
    gauge_identity: str
    rank_crossing: bool


def _pair_record(space: PairSpace) -> _PairRecord:
    if not isinstance(space, PairSpace):
        raise TypeError("shape bucketing requires canonical PairSpace records")
    rank = space.rank
    if type(rank) is not int or rank <= 0:
        raise ValueError("local-CC execution cannot bucket an empty pair space")
    gauge_identity = space.gauge_identity
    if not isinstance(gauge_identity, str) or not gauge_identity:
        raise ValueError("pair space has no stable gauge identity")
    pair = tuple(space.pair)
    return _PairRecord(
        PairBucketKey(rank, pair[0] == pair[1]),
        pair,
        gauge_identity,
        space.rank_crossing,
    )


def plan_pair_state_buckets(
    spaces: object,
    *,
    state_matrices_per_pair: object,
    max_pairs_per_bucket: object,
    budget_bytes: object,
) -> tuple[PairStateBucket, ...]:
    """Group fixed pair spaces into deterministic, independently bounded buckets.

    Buckets are ordered by ``(rank, diagonal_pair, pair)`` and split by both the
    explicit pair-count ceiling and the per-bucket numeric budget.  The same
    electronic reference and localized occupied frame are required throughout;
    pair-specific virtual domains and PNO gauges are retained as metadata rather
    than treated as interchangeable.

    The budget applies to one bucket at a time.  This makes the plan suitable for
    sequential native execution without implying that other solver allocations
    fit inside the same byte count.
    """
    if type(state_matrices_per_pair) is not int or state_matrices_per_pair <= 0:
        raise ValueError("state_matrices_per_pair must be a positive integer")
    if type(max_pairs_per_bucket) is not int or max_pairs_per_bucket <= 0:
        raise ValueError("max_pairs_per_bucket must be a positive integer")
    budget = checked_budget(budget_bytes)
    if not isinstance(spaces, (list, tuple)) or not spaces:
        raise ValueError("pair-state bucketing requires a nonempty list or tuple")

    reference_id: str | None = None
    localization_id: str | None = None
    records: list[_PairRecord] = []
    seen_pairs: set[tuple[int, int]] = set()
    for space in spaces:
        if not isinstance(space, PairSpace):
            raise TypeError("shape bucketing requires canonical PairSpace records")
        if reference_id is None:
            reference_id = space.reference_id
            localization_id = space.localization_id
        elif (
            space.reference_id != reference_id
            or space.localization_id != localization_id
        ):
            raise ValueError(
                "all bucketed pairs must share one reference/localized occupied frame"
            )
        record = _pair_record(space)
        if record.pair in seen_pairs:
            raise ValueError("pair-state bucketing requires unique occupied pairs")
        seen_pairs.add(record.pair)
        records.append(record)

    assert reference_id is not None and localization_id is not None
    records.sort(key=lambda record: (record.key, record.pair))

    buckets: list[PairStateBucket] = []
    begin = 0
    while begin < len(records):
        key = records[begin].key
        end = begin + 1
        while end < len(records) and records[end].key == key:
            end += 1

        bytes_per_pair = state_matrices_per_pair * key.rank * key.rank * _FLOAT64_BYTES
        budget_capacity = budget // bytes_per_pair
        if budget_capacity == 0:
            raise MemoryError(
                "pair-state bucket budget cannot hold one declared pair state: "
                f"rank={key.rank}, matrices={state_matrices_per_pair}, "
                f"required={bytes_per_pair}, budget={budget}"
            )
        chunk_size = min(max_pairs_per_bucket, budget_capacity)
        for chunk_begin in range(begin, end, chunk_size):
            chunk = records[chunk_begin : min(chunk_begin + chunk_size, end)]
            buckets.append(
                PairStateBucket(
                    reference_id=reference_id,
                    localization_id=localization_id,
                    key=key,
                    pairs=tuple(record.pair for record in chunk),
                    gauge_identities=tuple(record.gauge_identity for record in chunk),
                    rank_crossings=tuple(record.rank_crossing for record in chunk),
                    state_matrices_per_pair=state_matrices_per_pair,
                    numeric_bytes=len(chunk) * bytes_per_pair,
                )
            )
        begin = end

    return tuple(buckets)
