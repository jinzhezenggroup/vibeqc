"""Bound explicit AO maps to immutable geometry, grid and derivative domains.

This private owner supplies storage/lifetime policy only. The caller chooses an
explicit sampled-jet cutoff and qualifies complete energy/force errors. No default
screening policy is registered. The producer and every consumer remain resident;
only compact AO index arrays are retained on the host.
"""

from __future__ import annotations

import math
import typing
from dataclasses import dataclass
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class ResidentAoMapDomain:
    """Exact caller-validated geometry/grid domain; excludes density generation.

    The molecular-grid lease must be token-checked before entering this owner.
    Pointer equality is an additional lifetime check, never a scientific identity.
    A changed basis, geometry, point order, derivative order or tile shape requires
    a new owner. An order-1 map cannot be relabelled for an order-2 force consumer.
    """

    basis_identity: str
    geometry_identity: str
    grid_identity: str
    device: int
    point_pointer: int
    point_count: int
    tile_points: int
    derivative_order: int


_DENSE = object()


class ResidentAoMapCache:
    """Retain immutable maps under an explicit additional numeric host budget.

    The full AO scratch is already charged to CudaGrid. This budget includes
    retained indices and a conservative 20 bytes per global AO for simultaneous
    flag/index/immutable-output staging in the AO-only producer. It also covers
    the smaller host map copy in a subsequent feature lease. Python object and
    list headers are excluded, consistently with the existing grid resource
    contract. Budget/capability misses select the dense map before any omission;
    invalid identity, numerical and device failures propagate.
    """

    def __init__(
        self,
        grid: typing.Any,
        domain: ResidentAoMapDomain,
        *,
        cutoff: float,
        budget_bytes: int,
    ) -> None:
        if type(domain) is not ResidentAoMapDomain:
            raise TypeError("resident AO cache requires an explicit domain")
        if any(
            not isinstance(value, str) or not value
            for value in (
                domain.basis_identity,
                domain.geometry_identity,
                domain.grid_identity,
            )
        ):
            raise ValueError("resident AO cache requires scientific identities")
        if any(
            type(value) is not int or value <= 0
            for value in (domain.point_pointer, domain.point_count, domain.tile_points)
        ):
            raise ValueError("resident AO cache requires positive point dimensions")
        if (
            type(domain.device) is not int
            or domain.device < 0
            or type(domain.derivative_order) is not int
            or not 0 <= domain.derivative_order <= 3
        ):
            raise ValueError(
                "resident AO cache requires a device and supported jet order"
            )
        if type(cutoff) not in (int, float) or not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("resident AO cutoff must be finite and positive")
        if type(budget_bytes) is not int or budget_bytes < 0:
            raise ValueError("resident AO cache budget must be a nonnegative integer")
        self._grid = grid
        self._domain = domain
        self._cutoff = float(cutoff)
        self._budget_bytes = budget_bytes
        self._geometry_generation = grid.geometry_generation
        self._basis_generation = grid.basis_generation
        self._validate_binding(grid, domain)
        self._maps: dict[int, typing.Any] = {}
        self._retained_bytes = 0
        self._transient_bytes = 20 * grid.plan.nao
        self._capability_missing = False
        self.reset_work()

    @property
    def domain(self) -> ResidentAoMapDomain:
        return self._domain

    @property
    def cutoff(self) -> float:
        return self._cutoff

    @property
    def budget_bytes(self) -> int:
        return self._budget_bytes

    def _validate_binding(self, grid: typing.Any, domain: ResidentAoMapDomain) -> None:
        if (
            grid is not self._grid
            or domain != self.domain
            or grid.basis_identity != domain.basis_identity
            or grid.device_id != domain.device
            or grid.plan.order != domain.derivative_order
            or grid.plan.tile_points != domain.tile_points
            or grid.plan.active_ao_capacity != grid.plan.nao
            or grid.geometry_generation != self._geometry_generation
            or grid.basis_generation != self._basis_generation
        ):
            raise ValueError(
                "resident AO cache geometry/grid/derivative binding mismatch"
            )

    def reset_work(self) -> None:
        """Begin one endpoint's counters without evicting geometry-bound maps."""
        self._work = {
            "lookups": 0,
            "cache_hits": 0,
            "discoveries": 0,
            "discovery_seconds": 0.0,
            "discovery_ao_jet_values": 0,
            "discovery_density_contractions": 0,
            "dense_budget_tiles": 0,
            "dense_capability_tiles": 0,
            "point_ao_visits": 0,
            "point_ao_square_sum": 0,
            "dense_point_ao_square_sum": 0,
            "active_aos_min": None,
            "active_aos_max": 0,
            "active_aos_sum": 0,
            "tile_count": 0,
            "empty_tile_count": 0,
        }

    @property
    def work(self) -> dict[str, typing.Any]:
        """Detached per-endpoint work and current numeric capacity accounting."""
        return dict(
            self._work,
            retained_map_bytes=self._retained_bytes,
            transient_reserve_bytes=self._transient_bytes,
            budget_bytes=self.budget_bytes,
            numeric_peak_bound_bytes=(
                self._retained_bytes + self._transient_bytes
                if self.budget_bytes >= self._transient_bytes
                else 0
            ),
        )

    def select(
        self, grid: typing.Any, domain: ResidentAoMapDomain, begin: int, count: int
    ) -> np.ndarray | None:
        """Return a retained selected map, or ``None`` for the full AO domain.

        Every call rechecks scientific/lifetime binding, even on a cache hit.
        Discovery is synchronous and must occur before borrowing the AO task.
        Failures never publish a partially discovered map. Geometry rebinding
        invalidates this owner even if the CUDA allocator reuses an address.
        """
        self._validate_binding(grid, domain)
        if (
            type(begin) is not int
            or type(count) is not int
            or begin < 0
            or begin >= domain.point_count
            or begin % domain.tile_points
            or count != min(domain.tile_points, domain.point_count - begin)
        ):
            raise ValueError("resident AO lookup differs from its point tile domain")
        # Share the grid lock with native lease admission and center rebinding.
        with grid._lock:
            grid._check_open()
            self._validate_binding(grid, domain)
            self._work["lookups"] += 1
            if begin in self._maps:
                self._work["cache_hits"] += 1
                selected = self._maps[begin]
            elif self._capability_missing:
                self._work["dense_capability_tiles"] += 1
                selected = _DENSE
            elif (
                self._retained_bytes
                + self._transient_bytes
                + grid.plan.nao * np.dtype(np.uintp).itemsize
                > self.budget_bytes
            ):
                self._work["dense_budget_tiles"] += 1
                selected = _DENSE
            else:
                started = perf_counter()
                try:
                    selected = grid.select_ao_device_points(
                        domain.point_pointer + begin * 3 * 8, count, cutoff=self.cutoff
                    )
                except NotImplementedError:
                    self._capability_missing = True
                    self._work["dense_capability_tiles"] += 1
                    selected = _DENSE
                else:
                    if (
                        selected.dtype != np.dtype(np.uintp)
                        or selected.ndim != 1
                        or selected.flags.writeable
                        or selected.size > grid.plan.nao
                        or (selected.size and selected[-1] >= grid.plan.nao)
                        or np.any(selected[1:] <= selected[:-1])
                    ):
                        raise ValueError("resident AO producer returned an invalid map")
                    self._work["discoveries"] += 1
                    order = domain.derivative_order
                    jets = (order + 1) * (order + 2) * (order + 3) // 6
                    self._work["discovery_ao_jet_values"] += (
                        count * grid.plan.nao * jets
                    )
                    # A full identity map needs no retained numerical array.
                    if selected.size == grid.plan.nao:
                        selected = _DENSE
                    else:
                        self._retained_bytes += selected.nbytes
                    self._maps[begin] = selected
                finally:
                    self._work["discovery_seconds"] += perf_counter() - started
            active = grid.plan.nao if selected is _DENSE else selected.size
            self._work["tile_count"] += 1
            self._work["empty_tile_count"] += int(active == 0)
            prior_min = self._work["active_aos_min"]
            self._work["active_aos_min"] = (
                active if prior_min is None else min(prior_min, active)
            )
            self._work["active_aos_max"] = max(self._work["active_aos_max"], active)
            self._work["active_aos_sum"] += active
            self._work["point_ao_visits"] += count * active
            self._work["point_ao_square_sum"] += count * active**2
            self._work["dense_point_ao_square_sum"] += count * grid.plan.nao**2
            return None if selected is _DENSE else selected
