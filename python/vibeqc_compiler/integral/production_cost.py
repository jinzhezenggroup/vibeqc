"""Stable compile-cost and shard-partition policy for production kernels.

This leaf module intentionally depends only on shell-class identity.  Production
manifest parsing and CUDA/source emission may consume the policy, but the policy
must not import those orchestration layers back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterable, Sized

from .shell_spec import ShellClassSpec, shell_pair_class


class _ScheduleCostView(Protocol):
    """Schedule fields used by the structural cold-compile estimate."""

    @property
    def block_threads(self) -> int: ...


class ProductionSelectionCostView(Protocol):
    """Read-only selection surface required by production cost policy."""

    @property
    def spec(self) -> ShellClassSpec: ...

    @property
    def consumers(self) -> Sized: ...

    @property
    def schedule(self) -> _ScheduleCostView: ...

    @property
    def recurrence(self) -> str: ...

    @property
    def compile_seconds(self) -> float | None: ...

    @property
    def source_bytes(self) -> int | None: ...

    @property
    def object_bytes(self) -> int | None: ...


_SelectionT = TypeVar("_SelectionT", bound=ProductionSelectionCostView)

_STABLE_AOT_SHARD_MAP_VERSION = 1

# The map is intentionally keyed by shell name rather than manifest position.
# Its slots were chosen from the measured component/recurrence/consumer cost
# of the sm_120 production profile. Keeping this small, versioned table in
# source means adding or removing a manifest row cannot move an unrelated class
# to another translation unit (the failure mode of both manifest-order and
# whole-profile greedy partitioners). A future measurement refresh should
# increment the version and deliberately invalidate the affected cache keys.
_STABLE_AOT_SHARD_SLOTS: dict[str, int] = {
    "ssss": 0,
    "psss": 1,
    "psps": 2,
    "ppss": 2,
    "ppps": 4,
    "pppp": 7,
    "dsss": 6,
    "dsps": 5,
    "dspp": 7,
    "dpss": 2,
    "dpps": 5,
    "dppp": 4,
    "dpds": 5,
    "dpdp": 3,
    "dsds": 3,
    "ddss": 6,
    "ddps": 6,
    "ddpp": 2,
    "ddds": 7,
    "dddp": 1,
    "dddd": 0,
    "fpps": 7,
}


def shell_class_index(spec: ShellClassSpec) -> int:
    """Return the production triangular quartet-class index."""

    first = shell_pair_class(*spec.angular[:2])
    second = shell_pair_class(*spec.angular[2:])
    high = max(first, second)
    low = min(first, second)
    return high * (high + 1) // 2 + low


def production_compile_cost(selection: ProductionSelectionCostView) -> float:
    """Estimate the relative cold-compile cost of one production selection.

    Real compiler timings, when attached to a manifest row, are preferred.
    Older manifests have no such measurements, so this deterministic estimate
    uses the generated component count, recurrence root count, and number of
    emitted consumers. It is a partitioning signal only; it must never be
    used as a runtime or scientific quality metric.
    """

    if selection.compile_seconds is not None:
        return float(selection.compile_seconds)
    recurrence_factor = {
        "subset_wick": 1.0,
        "rys2": 1.15,
        "rys3": 1.35,
        "rys4": 1.7,
        "rys5": 2.0,
    }[selection.recurrence]
    consumer_factor = 1.0 + 0.65 * max(len(selection.consumers) - 1, 0)
    schedule_factor = max(selection.schedule.block_threads / 32.0, 1.0)
    structural = (
        selection.spec.component_count
        * (selection.spec.maximum_force_coulomb_order + 1)
        * recurrence_factor
        * consumer_factor
    )
    # Keep measured source/object sizes useful without allowing a stale size
    # estimate to dominate the mathematical structure of a new class.
    artifact_factor = 1.0
    if selection.source_bytes is not None:
        artifact_factor += min(selection.source_bytes / 1.0e6, 4.0) * 0.05
    if selection.object_bytes is not None:
        artifact_factor += min(selection.object_bytes / 1.0e6, 4.0) * 0.05
    return float(structural * artifact_factor * schedule_factor)


def stable_aot_shard_slot(selection: ProductionSelectionCostView) -> int:
    """Return the versioned, manifest-order-independent virtual shard slot."""

    slot = _STABLE_AOT_SHARD_SLOTS.get(selection.spec.name)
    if slot is not None:
        return slot
    # Unknown classes remain stable across manifest edits. The triangular
    # index is part of the canonical shell ABI, unlike list ordering.
    return shell_class_index(selection.spec)


def _partition_production_selections(
    selections: Iterable[_SelectionT], shard_count: int
) -> tuple[tuple[_SelectionT, ...], ...]:
    """Assign classes to versioned, compile-cost-aware stable translation units.

    Slots are deliberately not recomputed from the current manifest. A
    weighted greedy pass would balance a clean build but would also move every
    class after an insertion/removal, destroying the cache identity this layer
    is meant to preserve. The checked-in slot map is therefore the stable
    result of that pass over the current measured profile; unknown classes use
    their canonical shell-class index as a deterministic fallback.
    """

    if isinstance(shard_count, bool) or not isinstance(shard_count, int):
        raise TypeError("shard_count must be an integer")
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    shards: list[list[_SelectionT]] = [[] for _ in range(shard_count)]
    for selection in sorted(
        selections,
        key=lambda item: (stable_aot_shard_slot(item) % shard_count, item.spec.name),
    ):
        shards[stable_aot_shard_slot(selection) % shard_count].append(selection)
    return tuple(tuple(shard) for shard in shards)
