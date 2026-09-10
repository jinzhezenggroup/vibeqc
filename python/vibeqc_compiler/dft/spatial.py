"""Backend-neutral spatial partitions and explicit fixed AO masks.

Region size is a scientific partition parameter. Device tile sizes, padding,
warp ownership and matrix-library choices belong to later execution schedules.
No full point-by-AO table is constructed to discover a task's active columns.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from hashlib import sha256

import numpy as np

from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.common.resources import (
    ResourceBudget,
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourceRequest,
    byte_product,
    plan_resources,
)

from .ao import NativeAO
from .envelopes import ao_region_envelopes, derivative_domain
from .grid import ExplicitGrid, checked_int


def _indices(values):
    """Own irreversibly immutable native int64 maps, without float conversion."""
    raw = np.asarray(values)
    if raw.ndim != 1 or (
        raw.size
        and (
            raw.dtype.kind not in "iu"
            or np.any(raw < 0)
            or np.any(raw > np.iinfo(np.int64).max)
        )
    ):
        raise ValueError("index maps require nonnegative int64-compatible integers")
    array = np.asarray(raw, dtype=np.int64)
    return np.frombuffer(array.tobytes(), dtype=np.int64).reshape(array.shape)


def _ao_shell_map(basis):
    """Derive topology from the existing public shell representation only."""
    sizes = [
        2 * s.angular_momentum + 1
        if basis.representation == "real_spherical"
        else (s.angular_momentum + 1) * (s.angular_momentum + 2) // 2
        for s in basis.shells
    ]
    result = np.repeat(np.arange(len(sizes), dtype=np.int64), sizes)
    if len(result) != basis.nao:
        raise ValueError("native AO/shell map dimensions differ")
    return result


def _generation(basis, grid, policy):
    return canonical_hash(
        {
            "schema": "vibeqc.spatial-tasks.v1",
            "basis": basis.identity,
            "grid": grid.identity,
            "policy": asdict(policy),
        }
    )


@dataclass(frozen=True)
class SpatialPolicy:
    """Deterministic median partition with an optional absolute AO-jet cutoff.

    A fixed mask defines an approximated collocation on this exact geometry.
    It does not guarantee a smooth branch under nuclear motion. A new geometry
    requires new bounds/membership and cannot reuse the old task identity.
    """

    region_points: int = 128
    derivatives: tuple[tuple[int, int, int], ...] = (
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    )
    screening: str = "off"
    cutoff: float = 0.0
    version: int = 1

    def __post_init__(self):
        checked_int(self.region_points, "spatial region points")
        checked_int(self.version, "spatial policy version", high=1)
        object.__setattr__(self, "derivatives", derivative_domain(self.derivatives))
        if self.screening not in ("off", "absolute_ao_jet"):
            raise ValueError("unsupported spatial screening policy")
        if type(self.cutoff) not in (int, float) or not math.isfinite(self.cutoff):
            raise ValueError("AO screening cutoff must be finite")
        if (self.screening == "off" and self.cutoff != 0) or (
            self.screening != "off" and self.cutoff <= 0
        ):
            raise ValueError(
                "screening off requires zero cutoff; screening on requires a positive cutoff"
            )


@dataclass(frozen=True, eq=False)
class SpatialTask:
    """One exact point permutation and one local-to-global active AO map.

    Point IDs refer to the immutable source grid, preserving weights and owner
    atoms. AO IDs are unique and sorted; consumers must gather the complete
    D[I,I] submatrix, including cross-shell terms. ``discarded_max`` bounds each
    omitted AO derivative individually and is not an energy/force error bound.
    """

    point_ids: np.ndarray
    bounds: np.ndarray
    active_shell_ids: np.ndarray
    ao_ids: np.ndarray
    derivatives: tuple[tuple[int, int, int], ...]
    discarded_count: int
    discarded_max: np.ndarray
    generation_id: str
    identity: str = field(init=False)

    def __post_init__(self):
        for name in ("point_ids", "active_shell_ids", "ao_ids"):
            indices = _indices(getattr(self, name))
            if len(np.unique(indices)) != len(indices):
                raise ValueError("duplicate point/shell/AO index in spatial task")
            if name != "point_ids" and np.any(np.diff(indices) <= 0):
                raise ValueError("local shell/AO maps must be strictly sorted")
            object.__setattr__(self, name, indices)
        bounds = immutable(self.bounds, shape=(2, 3))
        if np.any(bounds[0] > bounds[1]) or not len(self.point_ids):
            raise ValueError("a spatial task needs points and ordered bounds")
        object.__setattr__(self, "bounds", bounds)
        derivatives = derivative_domain(self.derivatives)
        if tuple(self.derivatives) != derivatives:
            raise ValueError("task derivative diagnostics require canonical jet order")
        object.__setattr__(self, "derivatives", derivatives)
        checked_int(self.discarded_count, "discarded AO count", low=0)
        discarded = immutable(self.discarded_max, shape=(len(derivatives),))
        if np.any(discarded < 0):
            raise ValueError("discarded AO envelopes must be nonnegative")
        object.__setattr__(self, "discarded_max", discarded)
        if not isinstance(self.generation_id, str) or len(self.generation_id) != 64:
            raise ValueError("spatial task requires a complete generation identity")
        object.__setattr__(
            self,
            "identity",
            canonical_hash(
                {
                    "generation": self.generation_id,
                    "points": sha256(self.point_ids.tobytes()).hexdigest(),
                    "active_ao": sha256(self.ao_ids.tobytes()).hexdigest(),
                }
            ),
        )

    @property
    def numeric_bytes(self):
        return sum(
            a.nbytes
            for a in (
                self.point_ids,
                self.bounds,
                self.active_shell_ids,
                self.ao_ids,
                self.discarded_max,
            )
        )


@dataclass(frozen=True, eq=False)
class SpatialTasks:
    """Immutable task inventory and shared-budget construction/retention plan."""

    tasks: tuple[SpatialTask, ...]
    policy: SpatialPolicy
    basis_identity: str
    grid_identity: str
    generation_id: str
    resource_plan: object

    def __post_init__(self):
        tasks = tuple(self.tasks)
        if not isinstance(self.policy, SpatialPolicy) or any(
            not isinstance(t, SpatialTask) for t in tasks
        ):
            raise TypeError("spatial inventory requires typed policy and tasks")
        object.__setattr__(self, "tasks", tasks)

    def validate(self, basis, grid):
        """Validate a task inventory before preparing an execution owner.

        This includes O(points) map validation; a prepared owner may retain the
        validated immutable inventory and check only its generation on replay.
        """
        if not isinstance(basis, NativeAO) or not isinstance(grid, ExplicitGrid):
            raise TypeError("spatial tasks require NativeAO and ExplicitGrid")
        if basis.identity != self.basis_identity or grid.identity != self.grid_identity:
            raise ValueError("stale spatial geometry/basis/quadrature identity")
        if self.generation_id != _generation(basis, grid, self.policy):
            raise ValueError("stale spatial policy/generation identity")
        ao_shells = _ao_shell_map(basis)
        ids = []
        for task in self.tasks:
            if (
                task.generation_id != self.generation_id
                or task.derivatives != self.policy.derivatives
            ):
                raise ValueError(
                    "spatial task generation/domain differs from inventory"
                )
            if (
                np.any(task.point_ids >= len(grid.points))
                or np.any(task.ao_ids >= basis.nao)
                or np.any(task.active_shell_ids >= len(basis.shells))
            ):
                raise ValueError("spatial task map is outside its source domain")
            if task.discarded_count != basis.nao - len(task.ao_ids) or np.any(
                task.discarded_max > self.policy.cutoff
            ):
                raise ValueError(
                    "spatial task discard diagnostics differ from its mask"
                )
            if self.policy.screening == "off" and len(task.ao_ids) != basis.nao:
                raise ValueError("screening-off task omitted AO columns")
            if not np.array_equal(
                task.active_shell_ids, np.unique(ao_shells[task.ao_ids])
            ):
                raise ValueError("active shell map differs from selected AO columns")
            points = grid.points[task.point_ids]
            if np.any(points < task.bounds[0]) or np.any(points > task.bounds[1]):
                raise ValueError("spatial task bounds do not enclose its points")
            if len(task.point_ids) > self.policy.region_points:
                raise ValueError("spatial task exceeds its scientific region capacity")
            if self.policy.screening != "off" and task.discarded_count:
                # Public immutable descriptors can still be constructed or
                # replaced by a caller. Revalidate the bound certificate once
                # at preparation; trusting supplied diagnostics would permit
                # a forged mask to silently drop a large AO column.
                envelopes = ao_region_envelopes(basis, task.bounds, task.derivatives)
                omitted = np.ones(basis.nao, dtype=bool)
                omitted[task.ao_ids] = False
                certified = np.max(envelopes[:, omitted], axis=1)
                if np.any(certified > task.discarded_max):
                    raise ValueError(
                        "spatial mask is not certified by its AO envelopes"
                    )
            ids.append(task.point_ids)
        actual = np.concatenate(ids) if ids else np.empty(0, dtype=np.int64)
        if not np.array_equal(np.sort(actual), np.arange(len(grid.points))):
            raise ValueError(
                "spatial inventory must preserve every grid point exactly once"
            )

    @property
    def numeric_bytes(self):
        return sum(task.numeric_bytes for task in self.tasks)

    @property
    def identity(self):
        """Bind actual point/mask membership as well as the construction policy.

        A caller-supplied conservative task inventory may retain extra AOs.
        Its approximated collocation must not share the factory mask identity.
        """
        return canonical_hash(
            {
                "generation": self.generation_id,
                "tasks": [task.identity for task in self.tasks],
            }
        )


def spatial_resource_request(basis, grid, policy):
    """Preflight O(points + regions*AO) metadata through the existing planner.

    Retained maps are conservatively sized for all AOs in every region.
    Construction additionally owns sort indices/coordinate panels and one
    region's derivative envelopes. Caller-owned basis/grid numeric storage is
    included; execution D/V, AO jets and device resources are separate requests.
    """
    npoint, nao, jets = len(grid.points), basis.nao, len(policy.derivatives)
    regions = min(
        npoint, 2 * ((npoint + policy.region_points - 1) // policy.region_points)
    )
    retained = byte_product(8, npoint + regions * (2 * nao + 6 + jets))
    inputs = (
        basis.numeric_bytes
        + grid.points.nbytes
        + grid.weights.nbytes
        + byte_product(8, npoint)
    )
    scratch = byte_product(8, 10 * npoint + 4 * nao * jets + 4 * nao)
    return ResourceRequest(
        "spatial_tasks",
        ResourceIdentity(
            "dft",
            "spatial_partition",
            "cpu",
            "fp64",
            json.dumps({"points": npoint, "ao": nao, "policy": asdict(policy)}),
            ("task_maps", "ao_envelopes"),
            "median_regions_v1",
        ),
        (
            ResourceCandidate(
                "retained_maps",
                "resident",
                (
                    ResourceEstimate(
                        "spatial_inputs", inputs, "pageable", 0, 2, kind="persistent"
                    ),
                    ResourceEstimate(
                        "spatial_maps", retained, "pageable", 0, 2, kind="persistent"
                    ),
                    ResourceEstimate("spatial_construction", scratch, "pageable", 0, 0),
                ),
            ),
        ),
        (
            "Python object headers and allocator rounding",
            "caller-retained older task inventories",
            "execution density/output matrices and AO/feature workspaces",
        ),
    )


def build_spatial_tasks(basis, grid, *, policy=None, budget=None):
    """Build deterministic local-dense candidates from an explicit quadrature.

    Screening-off changes only point order. Screened tasks omit an AO only
    when every requested jet has an absolute region bound <= cutoff. This
    initial descriptor builder accepts ExplicitGrid; streamed molecular-grid
    preparation must supply the same point/weight/owner identity explicitly.
    """
    if not isinstance(basis, NativeAO) or not isinstance(grid, ExplicitGrid):
        raise TypeError("spatial tasks require NativeAO and ExplicitGrid")
    policy = SpatialPolicy() if policy is None else policy
    if not isinstance(policy, SpatialPolicy):
        raise TypeError("expected SpatialPolicy")
    if any(owner >= basis.natom for owner in grid.owners):
        raise ValueError("point owner is outside the AO atom domain")
    budget = ResourceBudget() if budget is None else budget
    resource_plan = plan_resources(
        (spatial_resource_request(basis, grid, policy),), budget
    )
    resource_plan.require_feasible()
    generation = _generation(basis, grid, policy)
    ao_shells = _ao_shell_map(basis)
    tasks, pending = [], [np.arange(len(grid.points), dtype=np.int64)]
    while pending:
        ids = pending.pop()
        if not len(ids):
            continue
        points = grid.points[ids]
        bounds = np.stack((points.min(axis=0), points.max(axis=0)))
        if len(ids) > policy.region_points:
            with np.errstate(over="ignore"):
                axis = int(np.argmax(bounds[1] - bounds[0]))
            # Coordinate tie breaks are stable under ordinary input reorderings.
            # Identical points finally use source IDs and have identical masks.
            order = np.lexsort(
                (ids, points[:, 2], points[:, 1], points[:, 0], points[:, axis])
            )
            ordered = ids[order]
            middle = len(ordered) // 2
            pending.extend((ordered[middle:], ordered[:middle]))
            continue
        if policy.screening == "off":
            active = np.ones(basis.nao, dtype=bool)
            discarded_max = np.zeros(len(policy.derivatives))
        else:
            envelopes = ao_region_envelopes(basis, bounds, policy.derivatives)
            active = np.any(envelopes > policy.cutoff, axis=0)
            discarded_max = (
                np.max(envelopes[:, ~active], axis=1)
                if np.any(~active)
                else np.zeros(len(policy.derivatives))
            )
        ao_ids = _indices(np.flatnonzero(active))
        point_ids = _indices(ids)
        tasks.append(
            SpatialTask(
                point_ids,
                immutable(bounds),
                _indices(np.unique(ao_shells[active])),
                ao_ids,
                policy.derivatives,
                int(np.count_nonzero(~active)),
                immutable(discarded_max),
                generation,
            )
        )
    return SpatialTasks(
        tuple(tasks), policy, basis.identity, grid.identity, generation, resource_plan
    )
