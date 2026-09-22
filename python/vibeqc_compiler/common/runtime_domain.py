"""Bounded runtime index-domain planning shared by compiler consumers.

The domain is pure immutable metadata plus a deterministic reference iterator.
It owns no scientific equation, device allocation, NumPy array, or backend
execution. Consumers may lower the same domain to CUDA, TensorIR control maps,
or bounded native task pages without inventing parallel batching semantics.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from itertools import islice, pairwise
from math import comb, prod

from .provenance import canonical_hash

_SCHEMA = "vibeqc.runtime_task_domain.v1"
_KINDS = frozenset(("rectangular", "nonincreasing"))


def _positive_int(value: typing.Any, label: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class RuntimeTaskPage:
    """One bounded reference page from a deterministic runtime domain."""

    domain_identity: str
    rank: int
    ordinal: int
    offset: int
    capacity: int
    coordinates: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if (
            type(self.domain_identity) is not str
            or len(self.domain_identity) != 64
            or any(c not in "0123456789abcdef" for c in self.domain_identity)
        ):
            raise ValueError("runtime page requires a domain SHA-256 identity")
        _positive_int(self.rank, "runtime page rank")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("runtime page ordinal must be nonnegative")
        if type(self.offset) is not int or self.offset < 0:
            raise ValueError("runtime page offset must be nonnegative")
        _positive_int(self.capacity, "runtime page capacity")
        coordinates = tuple(tuple(row) for row in self.coordinates)
        if not coordinates or len(coordinates) > self.capacity:
            raise ValueError("runtime page must contain one bounded nonempty batch")
        if any(
            len(row) != self.rank
            or any(type(value) is not int or value < 0 for value in row)
            for row in coordinates
        ):
            raise ValueError(
                "runtime page coordinates must be nonnegative integer tuples"
            )
        object.__setattr__(self, "coordinates", coordinates)

    @property
    def count(self) -> int:
        return len(self.coordinates)

    @property
    def padding(self) -> int:
        return self.capacity - self.count

    @property
    def identity(self) -> str:
        return canonical_hash(
            {
                "schema": _SCHEMA,
                "domain": self.domain_identity,
                "rank": self.rank,
                "ordinal": self.ordinal,
                "offset": self.offset,
                "capacity": self.capacity,
                "count": self.count,
                "coordinates": self.coordinates,
            }
        )


@dataclass(frozen=True, slots=True)
class RuntimeTaskDomain:
    """Finite runtime index space with deterministic bounded page semantics.

    rectangular is a Cartesian product over zero-based extents.
    nonincreasing is the monotone simplex x0 >= x1 >= ... >= 0 with an
    optional half-open outer-axis window. The latter covers triangular triples
    without embedding any coupled-cluster policy in the common compiler layer.
    """

    kind: str
    extents: tuple[int, ...]
    outer: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        extents = tuple(self.extents)
        if not extents or any(type(value) is not int or value < 1 for value in extents):
            raise ValueError("runtime domain extents must be positive integers")
        if self.kind not in _KINDS:
            raise ValueError(f"unsupported runtime domain kind: {self.kind}")
        if self.kind == "rectangular":
            if self.outer is not None:
                raise ValueError(
                    "rectangular runtime domains do not take an outer window"
                )
        else:
            if len(set(extents)) != 1:
                raise ValueError(
                    "nonincreasing runtime domains require equal axis extents"
                )
            if (
                not isinstance(self.outer, tuple)
                or len(self.outer) != 2
                or any(type(value) is not int for value in self.outer)
                or not 0 <= self.outer[0] < self.outer[1] <= extents[0]
            ):
                raise ValueError("invalid nonincreasing runtime-domain outer window")
        object.__setattr__(self, "extents", extents)

    @classmethod
    def rectangular(cls, extents: typing.Iterable[int]) -> RuntimeTaskDomain:
        return cls("rectangular", tuple(extents))

    @classmethod
    def nonincreasing(
        cls,
        extent: int,
        rank: int,
        *,
        outer_start: int = 0,
        outer_stop: int | None = None,
    ) -> RuntimeTaskDomain:
        _positive_int(extent, "runtime domain extent")
        _positive_int(rank, "runtime domain rank")
        stop = extent if outer_stop is None else outer_stop
        return cls("nonincreasing", (extent,) * rank, (outer_start, stop))

    @property
    def rank(self) -> int:
        return len(self.extents)

    @property
    def logical_size(self) -> int:
        if self.kind == "rectangular":
            return prod(self.extents)
        start, stop = typing.cast("tuple[int, int]", self.outer)
        # Hockey-stick identity for monotone tuples with fixed outer coordinate.
        return comb(stop + self.rank - 1, self.rank) - comb(
            start + self.rank - 1, self.rank
        )

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "schema": _SCHEMA,
            "kind": self.kind,
            "extents": self.extents,
            "outer": self.outer,
            "order": "lexicographic-row-major",
        }

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    def contains(self, coordinate: typing.Any) -> bool:
        try:
            coordinate = tuple(coordinate)
        except TypeError:
            return False
        if (
            len(coordinate) != self.rank
            or any(type(value) is not int for value in coordinate)
            or any(
                not 0 <= value < extent
                for value, extent in zip(coordinate, self.extents)
            )
        ):
            return False
        if self.kind == "rectangular":
            return True
        start, stop = typing.cast("tuple[int, int]", self.outer)
        return start <= coordinate[0] < stop and all(
            left >= right for left, right in pairwise(coordinate)
        )

    def __iter__(self) -> typing.Iterator[tuple[int, ...]]:
        if self.kind == "rectangular":
            # itertools.product pools every input range before its first yield.
            # Advance mixed-radix coordinates with only O(rank) retained state.
            coordinate = [0] * self.rank
            while True:
                yield tuple(coordinate)
                for axis in range(self.rank - 1, -1, -1):
                    coordinate[axis] += 1
                    if coordinate[axis] < self.extents[axis]:
                        break
                    coordinate[axis] = 0
                else:
                    return
        start, stop = typing.cast("tuple[int, int]", self.outer)

        def descend(prefix: tuple[int, ...]) -> typing.Iterator[tuple[int, ...]]:
            if len(prefix) == self.rank:
                yield prefix
                return
            for value in range(prefix[-1] + 1):
                yield from descend((*prefix, value))

        for first in range(start, stop):
            yield from descend((first,))

    def page_count(self, capacity: int) -> int:
        capacity = _positive_int(capacity, "runtime page capacity")
        return (self.logical_size + capacity - 1) // capacity

    def pages(self, capacity: int) -> typing.Iterator[RuntimeTaskPage]:
        """Yield bounded coordinate pages without materializing the full domain."""
        capacity = _positive_int(capacity, "runtime page capacity")
        iterator = iter(self)
        offset = 0
        ordinal = 0
        while coordinates := tuple(islice(iterator, capacity)):
            yield RuntimeTaskPage(
                self.identity,
                self.rank,
                ordinal,
                offset,
                capacity,
                coordinates,
            )
            offset += len(coordinates)
            ordinal += 1
