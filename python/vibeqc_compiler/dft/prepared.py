"""Prepared grid topology, transactional invalidation and bounded ragged batches."""

from __future__ import annotations

import threading
import time
import typing
from dataclasses import asdict, dataclass

import numpy as np

from vibeqc_compiler.common.provenance import canonical_hash

from .ao import NativeAO
from .cuda import CudaGrid
from .features import density_features, spin_densities
from .grid import GridSpec, MolecularGrid, checked_int
from .plan import plan_tiles


@dataclass(frozen=True)
class FeatureTile:
    """Global point offset plus owned point/weight/spin-feature arrays."""

    begin: int
    points: np.ndarray
    weights: np.ndarray
    features: dict[str, np.ndarray]
    generation: int


class PreparedGrid:
    """Reuse native basis/grid/device state and explicitly invalidate on changes.

    Coordinate, atom-order, basis, grid-rule or charge/spin updates build a new
    state before releasing the previous one. A failed update leaves the old
    state usable. Density changes always upload new spin matrices. No DFT
    energy, complete nuclear derivative or SCF capability is advertised.
    """

    def __init__(
        self,
        atoms: typing.Any,
        basis: typing.Any = "sto-3g",
        *,
        spec: typing.Any = None,
        representation: typing.Any = "cartesian",
        charge: typing.Any = 0,
        multiplicity: typing.Any = 1,
        backend: typing.Any = "cpu",
        order: typing.Any = 1,
        tile_points: typing.Any = 256,
        budget_bytes: typing.Any = 256 << 20,
        artifact: typing.Any = None,
        device_id: typing.Any = 0,
    ) -> None:
        spec = GridSpec() if spec is None else spec
        self._lock = threading.RLock()
        self._basis = None
        self._cuda = None
        self._generation = 0
        self._execution = 0
        self._replacement_peak_bytes = 0
        self._settings = {
            "atoms": atoms,
            "basis": basis,
            "spec": spec,
            "representation": representation,
            "charge": charge,
            "multiplicity": multiplicity,
            "backend": backend,
            "order": order,
            "tile_points": tile_points,
            "budget_bytes": budget_bytes,
            "artifact": artifact,
            "device_id": device_id,
        }
        started = time.perf_counter()
        if order < 1:
            raise ValueError("prepared density features require first derivatives")
        try:
            self._basis = NativeAO(
                atoms,
                basis,
                representation=representation,
                charge=charge,
                multiplicity=multiplicity,
            )
            self._grid = MolecularGrid(self._basis.atoms, spec, charge, multiplicity)
            self._plan = plan_tiles(
                self._basis,
                backend=backend,
                order=order,
                tile_points=tile_points,
                budget_bytes=budget_bytes,
                grid=self._grid,
            )
            if backend == "cuda":
                if artifact is None:
                    raise ValueError(
                        "CUDA requires an explicitly compiled grid artifact"
                    )
                self._cuda = CudaGrid(
                    self._basis,
                    artifact,
                    order=order,
                    tile_points=tile_points,
                    budget_bytes=budget_bytes,
                    device_id=device_id,
                    grid=self._grid,
                )
            # Keep only immutable canonical input topology; a mutable user list
            # cannot affect a later geometry-only reconfiguration.
            self._settings.update(atoms=self._basis.atoms, basis=self._basis.shells)
            self._setup_seconds = time.perf_counter() - started
            self._timings = {
                "grid_seconds": 0.0,
                "ao_seconds": 0.0,
                "feature_seconds": 0.0,
                "endpoint_seconds": 0.0,
                "tiles": 0,
            }
            self._refresh_identity()
        except Exception:
            self.close()
            raise

    @property
    def plan(self) -> typing.Any:
        return self._plan

    @property
    def grid(self) -> typing.Any:
        return self._grid

    @property
    def basis_identity(self) -> typing.Any:
        basis = self._basis
        if basis is None:
            raise RuntimeError("prepared grid is closed")
        return basis.identity

    @property
    def nao(self) -> typing.Any:
        return self._plan.nao

    def _refresh_identity(self) -> None:
        basis = self._basis
        if basis is None:
            raise RuntimeError("prepared grid is closed")
        self._identity = canonical_hash(
            {
                "basis": basis.identity,
                "grid": self._grid.identity,
                "backend": self._plan.backend,
                "order": self._plan.order,
                "generation": self.generation,
            }
        )

    @property
    def identity(self) -> typing.Any:
        return self._identity

    @property
    def generation(self) -> typing.Any:
        return self._generation

    def reconfigure(self, **changes: typing.Any) -> None:
        """Transactionally replace changed numerical state and advance generation.

        ``coordinates`` moves the existing ordered atoms. To reorder/change
        atoms and shells, supply both ``atoms`` and ``basis`` explicitly.
        """
        with self._lock:
            if self._basis is None:
                raise RuntimeError("prepared grid is closed")
            options = dict(self._settings)
            if "coordinates" in changes:
                coordinates = changes.pop("coordinates")
                if "atoms" in changes or len(coordinates) != len(self._basis.atoms):
                    raise ValueError("coordinate update must preserve atom topology")
                changes["atoms"] = [
                    (a.atomic_number, xyz)
                    for a, xyz in zip(self._basis.atoms, coordinates, strict=True)
                ]
            if set(changes) - set(options):
                raise ValueError("unknown prepared-grid setting")
            options.update(changes)
            # Preserve the old state on failure without concealing the overlap
            # between old and replacement arenas. Reject before allocating the
            # replacement GPU plan if both states cannot fit simultaneously.
            total_budget = options["budget_bytes"]
            checked_int(total_budget, "reconfiguration budget", high=2**63 - 1)
            old_peak = self.plan.peak_bytes
            options["budget_bytes"] = total_budget - old_peak
            if options["budget_bytes"] <= 0:
                raise ValueError("reconfiguration budget cannot retain the old plan")
            replacement = PreparedGrid(**options)
            replacement._settings["budget_bytes"] = total_budget
            overlap = old_peak + replacement.plan.peak_bytes
            self.close()
            self._basis, replacement._basis = replacement._basis, None
            self._cuda, replacement._cuda = replacement._cuda, None
            for name in ("_settings", "_grid", "_plan", "_setup_seconds", "_timings"):
                setattr(self, name, getattr(replacement, name))
            self._replacement_peak_bytes = max(self._replacement_peak_bytes, overlap)
            self._generation += 1
            self._execution += 1
            self._refresh_identity()

    def iter_features(self, density: typing.Any) -> typing.Any:
        """Yield bounded detached tiles; stale iterators fail after reconfiguration.

        Starting a new iterator invalidates any previous iterator on this plan,
        preventing interleaved GPU requests from accidentally sharing stale D.
        A caller retaining several yielded tiles owns the extra memory. Batch
        consumers integrate/discard each tile before the next one is generated.
        """
        with self._lock:
            if self._basis is None:
                raise RuntimeError("prepared grid is closed")
            d = spin_densities(density, self.nao)
            generation = self.generation
            grid = self._grid
            self._execution += 1
            execution = self._execution
            if self._cuda:
                self._cuda.set_density(d)
        iterator = grid.tiles(self.plan.tile_points)
        while True:
            started = time.perf_counter()
            try:
                tile = next(iterator)
            except StopIteration:
                return
            grid_seconds = time.perf_counter() - started
            with self._lock:
                if (
                    self._basis is None
                    or generation != self.generation
                    or execution != self._execution
                ):
                    raise RuntimeError("stale or closed prepared-grid iterator")
                self._timings["grid_seconds"] += grid_seconds
                if self._cuda:
                    features = self._cuda.evaluate(tile.points)
                else:
                    before = time.perf_counter()
                    jets = self._basis.evaluate(
                        tile.points,
                        self.plan.order,
                        budget_bytes=self._settings["budget_bytes"],
                    )
                    self._timings["ao_seconds"] += time.perf_counter() - before
                    before = time.perf_counter()
                    features = density_features(jets, d)
                    self._timings["feature_seconds"] += time.perf_counter() - before
                self._timings["endpoint_seconds"] += time.perf_counter() - started
                self._timings["tiles"] += 1
            yield FeatureTile(
                tile.begin, tile.points, tile.weights, features, generation
            )

    def integrate(self, density: typing.Any) -> typing.Any:
        """Stream spin electron counts and kinetic densities without renormalizing."""
        # Serialize complete integrations; a yielded low-level iterator remains
        # explicitly invalidatable at its next tile boundary.
        with self._lock:
            started = time.perf_counter()
            electrons = np.zeros(2)
            tau = np.zeros(2)
            for tile in self.iter_features(density):
                electrons += tile.features["rho"] @ tile.weights
                tau += tile.features["tau"] @ tile.weights
            return {
                "electrons": electrons.tolist(),
                "integrated_tau": tau.tolist(),
                "seconds": time.perf_counter() - started,
                "identity": self.identity,
                "generation": self.generation,
                "points": self.grid.npoint,
                "diagnostics": self.diagnostics(),
            }

    def diagnostics(self) -> typing.Any:
        """Report host construction/staging and synchronized native section costs."""
        with self._lock:
            result = {
                "backend": self.plan.backend,
                "screening": "none",
                "grid_placement": "host-streamed",
                "ao_placement": "device" if self._cuda else "host",
                "feature_placement": "device then explicit D2H"
                if self._cuda
                else "host",
                "setup_seconds": self._setup_seconds,
                **self._timings,
                "plan": asdict(self.plan),
                "peak_bytes": self.plan.peak_bytes,
                "replacement_peak_bytes": self._replacement_peak_bytes,
                "basis_identity": self.basis_identity,
                "grid_identity": self.grid.identity,
            }
            if self._cuda:
                result["cuda"] = self._cuda.metrics()
            return result

    def close(self) -> None:
        with self._lock:
            if self._cuda:
                self._cuda.close()
                self._cuda = None
            if self._basis:
                self._basis.close()
                self._basis = None

    def __enter__(self) -> typing.Any:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_lock"):
            self.close()


class PreparedGridBatch:
    """Ordered independent grid plans with summed capacity and failure isolation.

    Plans execute sequentially, each on its private stream. This initial ragged
    wrapper does not claim one fused GPU batch launch. Point/AO offsets remain
    explicit even when a peer's density or geometry update fails.
    """

    def __init__(
        self, items: typing.Any, *, budget_bytes: typing.Any = 1 << 30
    ) -> None:
        checked_int(budget_bytes, "batch grid budget", high=2**63 - 1)
        self._items = []
        self._budget_bytes = budget_bytes
        self._lock = threading.RLock()
        self.point_offsets = [0]
        self.ao_offsets = [0]
        try:
            remaining = budget_bytes
            for item in items:
                options = dict(item)
                options["budget_bytes"] = min(
                    options.get("budget_bytes", remaining), remaining
                )
                plan = PreparedGrid(**options)
                self._items.append(plan)
                remaining -= plan.plan.peak_bytes
                self.point_offsets.append(self.point_offsets[-1] + plan.grid.npoint)
                self.ao_offsets.append(self.ao_offsets[-1] + plan.nao)
            self.peak_bytes = budget_bytes - remaining
        except Exception:
            self.close()
            raise

    def execute(self, densities: typing.Any) -> typing.Any:
        """Isolate invalid/nonfinite density items while preserving original order."""
        with self._lock:
            if len(densities) != len(self._items):
                raise ValueError("ragged density count mismatch")
            result = []
            for index, (plan, density) in enumerate(
                zip(self._items, densities, strict=True)
            ):
                record = {
                    "index": index,
                    "point_begin": self.point_offsets[index],
                    "ao_begin": self.ao_offsets[index],
                }
                try:
                    record.update(status="pass", result=plan.integrate(density))
                except (TypeError, ValueError, RuntimeError) as error:
                    record.update(status="fail", reason=str(error))
                result.append(record)
            return result

    def reconfigure(self, index: typing.Any, **changes: typing.Any) -> None:
        """Update one item under the fleet budget, rebuilding ragged offsets."""
        with self._lock:
            checked_int(index, "batch index", low=0, high=len(self._items) - 1)
            plan = self._items[index]
            peers = self.peak_bytes - plan.plan.peak_bytes
            available = self._budget_bytes - peers
            changes["budget_bytes"] = min(
                changes.get("budget_bytes", available), available
            )
            plan.reconfigure(**changes)
            self.peak_bytes = sum(p.plan.peak_bytes for p in self._items)
            self.point_offsets = [0]
            self.ao_offsets = [0]
            for item in self._items:
                self.point_offsets.append(self.point_offsets[-1] + item.grid.npoint)
                self.ao_offsets.append(self.ao_offsets[-1] + item.nao)

    def close(self) -> None:
        with self._lock:
            for item in self._items:
                item.close()
            self._items = []

    def __enter__(self) -> typing.Any:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_items"):
            self.close()
