"""Explicit owner for PreparedBatch warm-start and restart lifecycle state."""

from __future__ import annotations

import typing
from dataclasses import dataclass, field


@dataclass
class WarmStartState:
    """Track retained seeds and their provenance independently of execution code.

    The native batch handle still owns device-resident warm densities.  This
    object owns only Python metadata and restart/projection bookkeeping, which
    keeps checkpoint and progressive initialization from mutating unrelated
    facade fields directly.
    """

    item_count: int
    enabled: bool = True
    updates: bool = True
    metadata: list[dict[str, typing.Any] | None] = field(default_factory=list)
    restart_indices: set[int] = field(default_factory=set)
    projection_indices: set[int] = field(default_factory=set)
    projection_diagnostics: dict[str, typing.Any] | None = None
    checkpoint_diagnostics: dict[str, typing.Any] | None = None

    def __post_init__(self) -> None:
        if not self.metadata:
            self.metadata = [None] * self.item_count
        elif len(self.metadata) != self.item_count:
            raise ValueError("warm-start metadata must match the batch size")

    def clear(self) -> None:
        """Forget all Python-side restart metadata after a native reset."""

        self.restart_indices.clear()
        self.projection_indices.clear()
        self.projection_diagnostics = None
        self.metadata = [None] * self.item_count

    def record_success(
        self, index: int, *, controls: typing.Any, backend: str
    ) -> None:
        """Record a successful execution when update policy is enabled."""

        if not self.enabled or not self.updates:
            return
        self.metadata[index] = {"controls": controls, "backend": backend}
        self.restart_indices.discard(index)
        self.projection_indices.discard(index)

    def origin(
        self, index: int, *, warm_start_used: bool, warm_start_fallback: bool
    ) -> str:
        """Return the stable public restart-origin label for one item."""

        if warm_start_fallback:
            return "cold_fallback"
        if warm_start_used and index in self.projection_indices:
            return "basis_projection"
        if warm_start_used and index in self.restart_indices:
            return "persistent_restart"
        if warm_start_used:
            return "in_process_warm"
        return "cold"

    def target_report(self, items: typing.Iterable[typing.Any]) -> list[dict[str, typing.Any]]:
        """Build the common checkpoint/projection verification report."""

        return [
            {
                "index": item.index,
                "converged": item.converged,
                "status": item.status,
                "energy": item.energy if item.succeeded else None,
                "density_rms": item.density_rms if item.succeeded else None,
                "iterations": item.iterations,
                "restart_origin": item.restart_origin,
            }
            for item in items
        ]


__all__ = ["WarmStartState"]
