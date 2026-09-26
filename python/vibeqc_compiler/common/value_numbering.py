"""Backend-neutral, effect-aware value numbering for compiler IR adapters.

Callers own semantic canonicalization and alias proofs.  This module only reuses
values that are explicitly PURE under the shared liveness effect contract.
Unknown effects therefore fail closed by default.
"""

from __future__ import annotations

import typing
from collections.abc import Callable, Hashable
from dataclasses import dataclass

from .liveness import EffectKind

T = typing.TypeVar("T")


@dataclass(frozen=True)
class ValueNumberResult(typing.Generic[T]):
    """One value-number decision and its canonical representative."""

    number: int
    representative: T
    reused: bool


@dataclass(frozen=True)
class ValueNumberingDiagnostics:
    """Compact deterministic evidence from one value-numbering table."""

    candidates: int
    unique_values: int
    reused_values: int
    copy_propagations: int
    effect_barriers: int
    rejected_key_collisions: int

    @property
    def eliminated_values(self) -> int:
        """Values represented by an already available semantic value."""

        return self.reused_values + self.copy_propagations

    def to_payload(self) -> dict[str, int]:
        """Return stable JSON-compatible counters for optimizer provenance."""

        return {
            "candidates": self.candidates,
            "unique_values": self.unique_values,
            "reused_values": self.reused_values,
            "copy_propagations": self.copy_propagations,
            "effect_barriers": self.effect_barriers,
            "rejected_key_collisions": self.rejected_key_collisions,
            "eliminated_values": self.eliminated_values,
        }


@dataclass(frozen=True)
class _Entry(typing.Generic[T]):
    number: int
    representative: T


class ValueNumberTable(typing.Generic[T]):
    """Deterministically number equivalent pure values.

    Without an equivalence callback the key is the complete semantic identity
    and dictionary equality is the proof.  With a callback the key is only a
    lookup bucket and every hit must pass the semantic guard before reuse.
    """

    def __init__(self, *, equivalent: Callable[[T, T], bool] | None = None) -> None:
        if equivalent is not None and not callable(equivalent):
            raise TypeError("value-number equivalence must be callable")
        self._equivalent = equivalent
        self._next_number = 0
        self._buckets: dict[Hashable, _Entry[T] | list[_Entry[T]]] = {}
        self._exact_numbers: dict[Hashable, int] = {}
        self._candidates = 0
        self._unique_values = 0
        self._reused_values = 0
        self._copy_propagations = 0
        self._effect_barriers = 0
        self._rejected_key_collisions = 0

    def _fresh(self, value: T) -> ValueNumberResult[T]:
        number = self._next_number
        self._next_number += 1
        self._unique_values += 1
        return ValueNumberResult(number, value, False)

    def _reuse(self, entry: _Entry[T]) -> ValueNumberResult[T]:
        self._reused_values += 1
        return ValueNumberResult(entry.number, entry.representative, True)

    def number_exact_pure(self, key: Hashable) -> int:
        """Return a dense number for a proven-pure complete semantic key."""

        if self._equivalent is not None:
            raise RuntimeError("exact-pure numbering requires a complete key table")
        self._candidates += 1
        number = self._exact_numbers.get(key)
        if number is not None:
            self._reused_values += 1
            return number
        number = self._next_number
        self._next_number += 1
        self._unique_values += 1
        self._exact_numbers[key] = number
        return number

    def number(
        self,
        key: Hashable,
        value: T,
        *,
        effect: EffectKind = EffectKind.OPAQUE,
    ) -> ValueNumberResult[T]:
        """Return a canonical representative when reuse is proven legal."""

        if not isinstance(key, Hashable):
            raise TypeError("value-number key must be hashable")
        if not isinstance(effect, EffectKind):
            raise TypeError("value-number effect must be EffectKind")
        equivalent = self._equivalent
        if equivalent is None:
            raise RuntimeError(
                "general numbering requires semantic verification; "
                "use number_exact_pure for complete pure keys"
            )
        self._candidates += 1
        if effect is not EffectKind.PURE:
            self._effect_barriers += 1
            return self._fresh(value)

        bucket = self._buckets.get(key)
        if bucket is None:
            result = self._fresh(value)
            self._buckets[key] = _Entry(result.number, result.representative)
            return result

        if isinstance(bucket, _Entry):
            entries = (bucket,)
        else:
            entries = tuple(bucket)
        for candidate in entries:
            if equivalent(value, candidate.representative):
                return self._reuse(candidate)
            self._rejected_key_collisions += 1

        result = self._fresh(value)
        entry = _Entry(result.number, result.representative)
        if isinstance(bucket, _Entry):
            self._buckets[key] = [bucket, entry]
        else:
            bucket.append(entry)
        return result

    def copy(self, source_number: int, representative: T) -> ValueNumberResult[T]:
        """Propagate a proven identity/copy to an existing value number."""

        if (
            type(source_number) is not int
            or source_number < 0
            or source_number >= self._next_number
        ):
            raise ValueError("copy source must be a known value number")
        self._candidates += 1
        self._copy_propagations += 1
        return ValueNumberResult(source_number, representative, True)

    @property
    def diagnostics(self) -> ValueNumberingDiagnostics:
        """Snapshot deterministic counters without exposing mutable buckets."""

        return ValueNumberingDiagnostics(
            candidates=self._candidates,
            unique_values=self._unique_values,
            reused_values=self._reused_values,
            copy_propagations=self._copy_propagations,
            effect_barriers=self._effect_barriers,
            rejected_key_collisions=self._rejected_key_collisions,
        )
