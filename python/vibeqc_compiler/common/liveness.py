"""Backend-neutral backward liveness for immutable compiler IRs.

The analysis is deliberately small: callers describe value dependencies and
whether an operation is proven pure. Unknown/opaque operations are retained.
No scientific subsystem policy belongs here.
"""

from __future__ import annotations

import typing
from collections.abc import Hashable
from dataclasses import dataclass
from enum import Enum


class EffectKind(str, Enum):
    """Observable-effect contract used by dead-code elimination."""

    PURE = "pure"
    EFFECTFUL = "effectful"
    OPAQUE = "opaque"


def _values(items: typing.Any, name: str) -> tuple[Hashable, ...]:
    if not isinstance(items, (tuple, list)):
        raise TypeError(f"{name} must be a sequence")
    result = tuple(items)
    for item in result:
        if not isinstance(item, Hashable):
            raise TypeError(f"{name} values must be hashable")
    return result


@dataclass(frozen=True)
class LivenessNode:
    """One ordered SSA-like operation for backward liveness analysis."""

    key: Hashable
    reads: tuple[Hashable, ...]
    writes: tuple[Hashable, ...]
    effect: EffectKind = EffectKind.OPAQUE

    def __post_init__(self) -> None:
        if not isinstance(self.key, Hashable):
            raise TypeError("liveness node key must be hashable")
        object.__setattr__(self, "reads", _values(self.reads, "reads"))
        object.__setattr__(self, "writes", _values(self.writes, "writes"))
        if not self.writes and self.effect is EffectKind.PURE:
            raise ValueError("pure liveness node must produce at least one value")
        if len(set(self.writes)) != len(self.writes):
            raise ValueError("liveness node contains duplicate writes")
        if not isinstance(self.effect, EffectKind):
            raise TypeError("liveness effect must be EffectKind")


@dataclass(frozen=True)
class LivenessResult:
    """Deterministic retained/removed node and value inventory."""

    live_node_keys: tuple[Hashable, ...]
    removed_node_keys: tuple[Hashable, ...]
    live_values: tuple[Hashable, ...]
    live_external_values: tuple[Hashable, ...]


def analyze_liveness(
    nodes: typing.Any,
    *,
    roots: typing.Any,
    external_values: typing.Any = (),
) -> LivenessResult:
    """Return the minimal ordered node set required by roots and effects.

    Nodes must already be in dependency order. PURE nodes are removable when
    none of their writes are demanded. EFFECTFUL and OPAQUE nodes are
    conservative roots, so their reads remain live even when their writes are
    otherwise unused.
    """

    if not isinstance(nodes, (tuple, list)):
        raise TypeError("liveness nodes must be a sequence")
    nodes = tuple(nodes)
    if any(not isinstance(node, LivenessNode) for node in nodes):
        raise TypeError("liveness analysis requires LivenessNode records")

    roots = _values(roots, "roots")
    external_values = _values(external_values, "external values")
    if len(set(external_values)) != len(external_values):
        raise ValueError("duplicate external value")

    keys: set[Hashable] = set()
    available = set(external_values)
    value_order = list(external_values)
    for node in nodes:
        if node.key in keys:
            raise ValueError("duplicate liveness node key")
        keys.add(node.key)
        missing = tuple(value for value in node.reads if value not in available)
        if missing:
            raise ValueError(
                f"{node.key!r}: missing, forward, or cyclic liveness dependency"
            )
        overlap = tuple(value for value in node.writes if value in available)
        if overlap:
            raise ValueError(f"{node.key!r}: duplicate value producer")
        available.update(node.writes)
        value_order.extend(node.writes)

    unknown_roots = tuple(value for value in roots if value not in available)
    if unknown_roots:
        raise ValueError("liveness root has no producer or external owner")

    needed = set(roots)
    live_keys: set[Hashable] = set()
    retained_values = set(roots)
    for node in reversed(nodes):
        demanded = any(value in needed for value in node.writes)
        if demanded or node.effect is not EffectKind.PURE:
            live_keys.add(node.key)
            needed.update(node.reads)
            retained_values.update(node.reads)
            retained_values.update(node.writes)

    live_node_keys = tuple(node.key for node in nodes if node.key in live_keys)
    removed_node_keys = tuple(node.key for node in nodes if node.key not in live_keys)
    live_values = tuple(value for value in value_order if value in retained_values)
    live_external_values = tuple(
        value for value in external_values if value in retained_values
    )
    return LivenessResult(
        live_node_keys=live_node_keys,
        removed_node_keys=removed_node_keys,
        live_values=live_values,
        live_external_values=live_external_values,
    )
