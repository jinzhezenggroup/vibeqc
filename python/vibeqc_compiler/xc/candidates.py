"""Executable D/C registrations for the existing DFT tuning/evidence workflow.

This module binds candidates to one current source and prepared XC contract.
It does not search schedules, install profiles or select a performance winner.
"""

from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from hashlib import sha256

import numpy as np

from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.dft.density_source import DensitySource, DensityStamp
from vibeqc_compiler.dft.features import spin_densities

from .prepared import PreparedXCContractions


@dataclass(frozen=True)
class DensityWorkload:
    """Exact fixed-input identity plus explanatory costs, not a selector model.

    Each active-AO row is (AO count, task count, total task points). Orbital
    counts include every supplied column, even zero-occupation columns; local
    maps never truncate delocalized occupied orbitals. Capacity fields retain
    the existing resource planner's explicit numeric-buffer scope/exclusions.
    """

    prepared: str
    resources: str
    basis: str
    grid: str
    mask: str | None
    source: str
    factor: str | None
    density_direction: str | None
    functional: str
    spin: str
    observable: str
    ingredients: tuple[str, ...]
    collocation_backend: str
    npoint: int
    nao: int
    active_ao_distribution: tuple[tuple[int, int, int], ...]
    occupied_counts: tuple[int, int]
    ao_order: int
    point_tile: int
    orbital_tile: int
    host_bytes: int
    device_bytes: int

    @property
    def identity(self):
        return canonical_hash(
            {"schema": "vibeqc.dft-density-workload.v1", **asdict(self)}
        )

    def describe(self):
        """Return detached metadata suitable for #138 evidence records."""
        return {
            **asdict(self),
            "identity": self.identity,
            "feature_cost_inputs": {
                "D_g_m_squared": sum(
                    m * m * points for m, _, points in self.active_ao_distribution
                ),
                "C_g_m_nocc": sum(
                    m * points for m, _, points in self.active_ao_distribution
                )
                * sum(self.occupied_counts),
                "scope": "feature contractions only; excludes XC, AO potential assembly, packing and transfers",
            },
        }


@dataclass(frozen=True)
class DensityCandidate:
    """One executable route, bound to an immutable current-density snapshot.

    Supply the CURRENT consumer stamp on each replay. An unavailable C route
    raises; its registered D sibling remains executable on the original D.
    GPU/upload failures propagate and never become successful fallback timings.
    Instances borrow their prepared owner, which must remain open.
    """

    route: str
    workload: DensityWorkload
    available: bool
    reason: str | None
    _prepared: PreparedXCContractions = field(repr=False, compare=False)
    _source: DensitySource = field(repr=False, compare=False)
    _delta_density: np.ndarray | None = field(repr=False, compare=False)

    @property
    def identity(self):
        return canonical_hash(
            {
                "schema": "vibeqc.dft-density-candidate.v1",
                "workload": self.workload.identity,
                "route": self.route,
            }
        )

    def describe(self):
        """Registration is separate from numerical acceptance and promotion."""
        return {
            "identity": self.identity,
            "route": self.route,
            "available": self.available,
            "reason": self.reason,
            "fallback": "density_matrix" if not self.available else None,
            "workload": self.workload.describe(),
            "promotion": {
                "eligible": False,
                "reason": "fixed-density XC registration; complete energy/force promotion remains owned by #168",
            },
        }

    def execute(self, *, stamp: DensityStamp):
        """Return (XC outputs, detached execution record), with no timing filter.

        The owner's reentrant lock covers execution and statistics capture;
        another candidate cannot replace the measured source between them.
        PreparedXCContractions retains its existing spatial/CUDA lease order.
        """
        if stamp != self._source.stamp:
            raise ValueError("stale registered density source for the current stamp")
        if not self.available:
            raise ValueError(f"density candidate unavailable: {self.reason}")
        owner = self._prepared
        with owner._lock:
            if (
                owner.identity != self.workload.prepared
                or owner.resource_plan.identity != self.workload.resources
            ):
                raise ValueError("stale registered XC/resource identity")
            options = {"delta_density": self._delta_density}
            if owner.density_grid is not None:
                options.update(stamp=stamp, route=self.route)
                value = owner.execute(self._source, **options)
            else:
                value = owner.execute(self._source.density, **options)
            statistics = deepcopy(owner.statistics)
            source_statistics = statistics.get("source")
            executed = (
                source_statistics.get("source_kind", "density_matrix")
                if isinstance(source_statistics, dict)
                else "density_matrix"
            )
            if executed != self.route:
                raise RuntimeError(
                    "registered candidate did not execute its requested density route"
                )
            return value, {
                "candidate": self.identity,
                "workload": self.workload.identity,
                "requested_route": self.route,
                "executed_route": executed,
                "statistics": statistics,
            }


def density_candidates(prepared, source, *, stamp, delta_density=None):
    """Register D and C against identical scientific inputs and resource owners.

    Geometry/response C derivatives remain unavailable; their D candidate
    executes the existing native derivative consumer. The generated CUDA
    consumer currently supports E/V. Capability checks do not assert numerical
    validity of arbitrary signed D in an XC functional's physical domain.
    A response request requires its direction at registration, so different
    perturbations cannot share a fixed-input identity or mutate on replay.
    """
    if not isinstance(prepared, PreparedXCContractions) or not isinstance(
        source, DensitySource
    ):
        raise TypeError(
            "density candidates require PreparedXCContractions and DensitySource"
        )
    if stamp != source.stamp:
        raise ValueError("stale density source for candidate registration")
    with prepared._lock, ExitStack() as leases:
        if prepared.spatial is not None:
            leases.enter_context(prepared.spatial._lock)
        prepared._check()  # Reject active spatial task leases before waiting on CUDA.
        cuda = prepared.density_grid
        if cuda is not None:
            leases.enter_context(cuda._lock)
            prepared._check()
        if (
            source.stamp.basis_identity != prepared.basis.identity
            or source.density.shape != (2, prepared.basis.nao, prepared.basis.nao)
        ):
            raise ValueError("density source differs from the prepared AO basis")
        if cuda is not None and stamp.basis_generation != cuda.basis_generation:
            raise ValueError("stale CUDA basis generation for candidate registration")
        if prepared.program.spec.spin == "unpolarized" and not np.array_equal(
            source.density[0], source.density[1]
        ):
            raise ValueError("unpolarized XC candidate requires equal spin densities")
        counts = (
            (0, 0)
            if source.occupations is None
            else (len(source.occupations[0]), len(source.occupations[1]))
        )
        observable = prepared.program.contract.request.observable
        if observable == "response":
            if delta_density is None:
                raise ValueError("response candidate requires a density direction")
            direction = spin_densities(delta_density, prepared.basis.nao)
            if prepared.program.spec.spin == "unpolarized" and not np.array_equal(
                direction[0], direction[1]
            ):
                raise ValueError(
                    "unpolarized XC candidate requires equal spin directions"
                )
        else:
            if delta_density is not None:
                raise ValueError("density direction requires a response request")
            direction = None
        reason = None
        if observable in ("response", "geometry"):
            reason = "unvalidated_orbital_derivative"
        elif source.stamp.role == "response":
            reason = "response_density"
        elif source.coefficients is None:
            reason = source.fallback_reason
        elif cuda is None:
            reason = "prepared_cpu_orbital_consumer_unavailable"
        elif cuda.plan.orbital_capacity is None or any(
            n > cap for n, cap in zip(counts, cuda.plan.orbital_capacity, strict=True)
        ):
            reason = "orbital_capacity_exceeded"
        rows = {}
        if prepared.spatial is None:
            rows[prepared.basis.nao] = [
                (prepared.npoint + prepared.tile_points - 1) // prepared.tile_points,
                prepared.npoint,
            ]
        else:
            for task in prepared.spatial.tasks.tasks:
                row = rows.setdefault(len(task.ao_ids), [0, 0])
                row[0] += 1
                row[1] += len(task.point_ids)
        ingredients = (
            ("rho",)
            if prepared.program.contract.ingredients.family == "lda"
            else ("rho", "gradient", "sigma")
        )
        workload = DensityWorkload(
            prepared=prepared.identity,
            resources=prepared.resource_plan.identity,
            basis=prepared.basis.identity,
            grid=prepared.grid.identity,
            mask=prepared._mask,
            source=source.stamp.identity,
            factor=source.factor_identity,
            density_direction=None
            if direction is None
            else sha256(direction.tobytes()).hexdigest(),
            functional=prepared.program.contract.identity,
            spin=prepared.program.spec.spin,
            observable=observable,
            ingredients=cuda.ingredients if cuda is not None else ingredients,
            collocation_backend="cuda" if cuda is not None else "cpu",
            npoint=prepared.npoint,
            nao=prepared.basis.nao,
            active_ao_distribution=tuple((m, *row) for m, row in sorted(rows.items())),
            occupied_counts=counts,
            ao_order=cuda.plan.order
            if cuda is not None
            else prepared.program.contract.ao_order,
            point_tile=prepared.tile_points,
            orbital_tile=0 if cuda is None else cuda.plan.orbital_tile,
            host_bytes=prepared.resource_plan.peak_bytes["host"],
            device_bytes=prepared.resource_plan.peak_bytes["device"],
        )
        return (
            DensityCandidate(
                "density_matrix", workload, True, None, prepared, source, direction
            ),
            DensityCandidate(
                "orbitals",
                workload,
                reason is None,
                reason,
                prepared,
                source,
                direction,
            ),
        )
