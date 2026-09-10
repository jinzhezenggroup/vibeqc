"""Prepared local-dense spatial execution through the shared resource planner.

This is an explicit candidate; dense PreparedGrid remains available. Geometry
and fixed AO masks are immutable, while density values are replaced on each
execution. No cached AO-by-molecular-grid table is retained.
"""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.common.resources import (
    MAX_BYTES,
    ResourceBudget,
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourceRequest,
    plan_resources,
)

from .ao import NativeAO, jet_indices
from .cuda import CudaGrid
from .features import density_features, spin_densities
from .grid import ExplicitGrid, MolecularGrid, checked_int
from .plan import plan_tiles
from .spatial import (
    SpatialPolicy,
    SpatialTasks,
    build_spatial_tasks,
    spatial_resource_request,
)


@dataclass(frozen=True)
class SpatialFeatureTile:
    """Detached bounded tile in source-grid point order with its fixed AO map."""

    point_ids: np.ndarray
    ao_ids: np.ndarray
    points: np.ndarray
    weights: np.ndarray
    owners: tuple[int, ...]
    features: dict
    generation_id: str
    ao_jets: np.ndarray | None = None


def _request(name, estimates, topology, backend="cpu"):
    return ResourceRequest(
        name,
        ResourceIdentity(
            "dft",
            name,
            backend,
            "fp64",
            json.dumps(topology),
            ("ao_jets", "spin_features", "matrix_scatter"),
            "stream_recompute",
        ),
        (ResourceCandidate("stream_recompute", "streamed", tuple(estimates)),),
        (
            "Python object headers and allocator rounding",
            "caller-retained detached output tiles",
            "CUDA driver/context/modules/stacks beyond the declared library allowance",
        ),
    )


def _previous_request(plan):
    # Charge the previous complete capacities, including its scratch, over the
    # replacement's construction and execution phases. This deliberately
    # overestimates overlap instead of pretending old allocations disappeared.
    estimates = [
        ResourceEstimate(
            "previous_host",
            plan.peak_bytes.get("host", 0),
            "pageable",
            0,
            2,
            kind="persistent",
        ),
    ]
    estimates.extend(
        ResourceEstimate("previous_" + key, value, key, 0, 2, kind="persistent")
        for key, value in plan.peak_bytes.items()
        if key.startswith("device:")
    )
    return _request("previous_spatial_owner", estimates, {"identity": plan.identity})


def _materialize_grid(basis, grid, budget, previous):
    if isinstance(grid, ExplicitGrid):
        return grid
    if not isinstance(grid, MolecularGrid):
        raise TypeError("expected ExplicitGrid or MolecularGrid")
    if (grid.atoms, grid.charge, grid.multiplicity) != (
        basis.atoms,
        basis.charge,
        basis.multiplicity,
    ):
        raise ValueError("stale molecular grid geometry or spin policy")
    # Retain O(points) quadrature only. ExplicitGrid owns immutable copies and
    # its identity serialization temporarily owns another numeric-sized view.
    # The existing point generator's partition scratch is tiled independently.
    estimate = basis.numeric_bytes + grid.numeric_bytes + grid.setup_scratch_bytes
    estimate += 256 * grid.npoint + 256 * 8 * basis.natom
    request = _request(
        "spatial_quadrature",
        [ResourceEstimate("quadrature_construction", estimate, "pageable", 0, 0)],
        {"points": grid.npoint, "grid": grid.identity},
    )
    plan_resources((*previous, request), budget).require_feasible()
    return grid.explicit(max_points=max(1, grid.npoint))


class PreparedSpatialGrid:
    """Prepare an explicit local-dense candidate with bounded host/device owners.

    Basis and quadrature are borrowed immutable scientific inputs. This owner
    keeps their references and charges their numeric storage; it never closes
    the caller's NativeAO. Total ResourceBudget replaces any new user-level
    per-subsystem cap. Only a bounded tile is recomputed on every execution.
    """

    def __init__(
        self,
        basis,
        grid,
        *,
        policy=None,
        tasks=None,
        backend="cpu",
        tile_points=64,
        resource_budget=None,
        artifact=None,
        device_id=0,
        _previous_plan=None,
    ):
        self._lock = threading.RLock()
        self._cuda = None
        self._closed = False
        self._execution = 0
        self._leased = False
        if not isinstance(basis, NativeAO):
            raise TypeError("expected NativeAO")
        checked_int(tile_points, "spatial execution tile points")
        checked_int(device_id, "visible device ordinal", low=0)
        if backend not in ("cpu", "cuda"):
            raise ValueError("unsupported spatial backend")
        policy = SpatialPolicy() if policy is None else policy
        if not isinstance(policy, SpatialPolicy):
            raise TypeError("expected SpatialPolicy")
        # Features consume all first jets. Higher requested domains may be
        # retained for downstream consumers, but never silently omit x/y/z.
        if not set(jet_indices(1)).issubset(policy.derivatives):
            raise ValueError(
                "spatial features require values and all first derivatives"
            )
        order = max(map(sum, policy.derivatives))
        if policy.derivatives != jet_indices(order):
            raise ValueError(
                "this execution candidate requires a complete through-order jet domain"
            )
        self.basis, self.source_grid = basis, grid
        self.budget = resource_budget or ResourceBudget()
        previous = (
            () if _previous_plan is None else (_previous_request(_previous_plan),)
        )
        started = time.perf_counter()
        self.grid = grid = _materialize_grid(basis, grid, self.budget, previous)
        metadata = spatial_resource_request(basis, grid, policy)
        # Construction preflight includes the still-live old owner. This check
        # precedes envelopes/maps, then actual mask sizes refine execution.
        plan_resources((*previous, metadata), self.budget).require_feasible()
        if tasks is None:
            tasks = build_spatial_tasks(basis, grid, policy=policy, budget=self.budget)
        if not isinstance(tasks, SpatialTasks) or tasks.policy != policy:
            raise ValueError("spatial tasks require the same requested policy")
        tasks.validate(basis, grid)
        active = max((len(t.ao_ids) for t in tasks.tasks), default=0)
        self.tasks = tasks
        self.tile_plan = plan_tiles(
            basis,
            backend=backend,
            order=order,
            tile_points=tile_points,
            budget_bytes=MAX_BYTES,
            active_ao_capacity=max(1, active),
        )
        estimates = [
            # Basis/grid/maps already belong to metadata. The legacy tile
            # capacity includes basis storage once; subtract that duplicate.
            ResourceEstimate(
                "local_host_execution",
                self.tile_plan.host_bytes - basis.numeric_bytes,
                "pageable",
                1,
                2,
                kind="persistent",
                recomputable=True,
            ),
        ]
        if isinstance(self.source_grid, MolecularGrid):
            estimates.append(
                ResourceEstimate(
                    "molecular_quadrature_topology",
                    self.source_grid.numeric_bytes,
                    "pageable",
                    0,
                    2,
                    kind="persistent",
                )
            )
        if backend == "cuda":
            estimates.extend(
                (
                    ResourceEstimate(
                        "local_device_arena",
                        self.tile_plan.allocation_bytes,
                        f"device:{device_id}",
                        1,
                        2,
                        kind="persistent",
                        recomputable=True,
                    ),
                    ResourceEstimate(
                        "cublas_allowance",
                        self.tile_plan.provider_bytes,
                        f"device:{device_id}",
                        1,
                        2,
                        kind="library",
                        accounting="runtime_allowance",
                    ),
                )
            )
        execution = _request(
            "spatial_execution",
            estimates,
            {
                "points": tile_points,
                "ao": basis.nao,
                "active": active,
                "jets": len(policy.derivatives),
            },
            backend,
        )
        self.resource_plan = plan_resources((metadata, execution), self.budget)
        self.resource_plan.require_feasible()
        self.replacement_plan = plan_resources(
            (*previous, metadata, execution), self.budget
        )
        self.replacement_plan.require_feasible()
        self.backend = backend
        self._settings = {
            "policy": policy,
            "backend": backend,
            "tile_points": tile_points,
            "resource_budget": self.budget,
            "artifact": artifact,
            "device_id": device_id,
        }
        try:
            if backend == "cuda":
                if artifact is None:
                    raise ValueError(
                        "CUDA spatial execution requires a compiled grid artifact"
                    )
                self._cuda = CudaGrid(
                    basis,
                    artifact,
                    order=order,
                    tile_points=tile_points,
                    active_ao_capacity=max(1, active),
                    budget_bytes=MAX_BYTES,
                    device_id=device_id,
                )
        except Exception:
            self.close()
            raise
        self.timings = {
            "construction_seconds": time.perf_counter() - started,
            "ao_seconds": 0.0,
            "density_seconds": 0.0,
            "tiles": 0,
        }

    def _check(self):
        if self._closed:
            raise RuntimeError("prepared spatial grid is closed")
        if self._leased:
            raise RuntimeError("prepared spatial grid has an active task lease")

    def _tiles(self):
        for task in self.tasks.tasks:
            for begin in range(0, len(task.point_ids), self.tile_plan.tile_points):
                yield task, task.point_ids[begin : begin + self.tile_plan.tile_points]

    def iter_features(self, density, *, include_jets=False):
        """Yield detached diagnostic feature tiles; a new execution invalidates old iterators."""
        with self._lock:
            self._check()
            if type(include_jets) is not bool:
                raise ValueError("include_jets must be boolean")
            d = spin_densities(density, self.basis.nao)
            self._execution += 1
            execution = self._execution
            if self._cuda:
                self._cuda.set_density(d)
        for task, ids in self._tiles():
            with self._lock:
                self._check()
                if execution != self._execution:
                    raise RuntimeError("stale spatial execution iterator")
                points = immutable(self.grid.points[ids])
                if self._cuda:
                    values = self._cuda.evaluate(
                        points, ao_ids=task.ao_ids, download_jets=include_jets
                    )
                    jets = values.pop("ao_jets", None)
                else:
                    started = time.perf_counter()
                    jets = self.basis.evaluate(
                        points,
                        self.tile_plan.order,
                        ao_ids=task.ao_ids,
                        budget_bytes=MAX_BYTES,
                    )
                    self.timings["ao_seconds"] += time.perf_counter() - started
                    started = time.perf_counter()
                    local_d = d[:, task.ao_ids[:, None], task.ao_ids[None, :]]
                    values = density_features(jets, local_d)
                    self.timings["density_seconds"] += time.perf_counter() - started
                self.timings["tiles"] += 1
                result = SpatialFeatureTile(
                    ids,
                    task.ao_ids,
                    points,
                    immutable(self.grid.weights[ids]),
                    tuple(self.grid.owners[i] for i in ids),
                    values,
                    self.tasks.generation_id,
                    jets if include_jets else None,
                )
            yield result

    @contextmanager
    def device_tasks(self, density):
        """Lend a serial iterator of native task leases without feature/jet D2H.

        Each yielded tuple is (SpatialTask, point_ids, DeviceGridTask). Finish
        its consumer and scatter before requesting the next task. The outer
        context closes even a partially consumed iterator deterministically.
        """
        with self._lock:
            self._check()
            if self._cuda is None:
                raise ValueError("device task consumption requires CUDA")
            self._cuda.set_density(spin_densities(density, self.basis.nao))
            self._execution += 1
            self._leased = True

            def iterator():
                for task, ids in self._tiles():
                    with self._cuda.task(self.grid.points[ids], task.ao_ids) as lease:
                        yield task, ids, lease

            result = iterator()
        # The lease flag rejects competing operations promptly. Keeping this
        # lock across user code would instead block other threads indefinitely.
        try:
            yield result
        finally:
            with self._lock:
                result.close()
                self._leased = False

    def reconfigure(self, basis, grid, **changes):
        """Replace immutable scientific state transactionally, charging both owners."""
        with self._lock:
            self._check()
            settings = {**self._settings, **changes}
            replacement = PreparedSpatialGrid(
                basis, grid, _previous_plan=self.resource_plan, **settings
            )
            if self._cuda:
                self._cuda.close()
            execution = self._execution + 1
            for name, value in replacement.__dict__.items():
                if name != "_lock":
                    setattr(self, name, value)
            self._execution = execution
            replacement._cuda = None
            replacement._closed = True

    def close(self):
        """Close owned native state while leaving caller-owned basis data alive."""
        with self._lock:
            if self._leased:
                raise RuntimeError("prepared spatial grid has an active task lease")
            if self._cuda:
                self._cuda.close()
                self._cuda = None
            self._closed = True
            self._execution += 1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if hasattr(self, "_lock"):
            self.close()
