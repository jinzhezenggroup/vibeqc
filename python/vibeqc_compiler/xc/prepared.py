"""Bounded native CPU XC endpoints on fixed dense or spatial collocation.

The shared ResourceBudget owns the cap. NativeAO, generated point code and
CPU matrix products stream tiles; neither AO^4 data nor a molecular AO table
is retained. This explicit candidate does not register a complete DFT method.
"""

import json
import threading
from contextlib import nullcontext
from time import perf_counter

import numpy as np

from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.common.resources import (
    MAX_BYTES,
    ResourceBudget,
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourceRequest,
    byte_product,
    plan_resources,
)
from vibeqc_compiler.dft import ExplicitGrid, MolecularGrid, NativeAO
from vibeqc_compiler.dft.features import spin_densities
from vibeqc_compiler.dft.grid import checked_int
from vibeqc_compiler.dft.spatial_prepared import PreparedSpatialGrid

from .contractions import GeometryPartials
from .integration import _tiles
from .native import NativeContractionProgram
from .spec import UnsupportedXC


class PreparedXCContractions:
    """Compose a compiled contraction program with immutable basis/quadrature.

    Output matrices retain functional-spin layout; the existing method adapter
    owns averaging for total-density input. Geometry returns independent point,
    AO-center and weight partials. A spatial adapter keeps its fixed mask and
    must have a CPU jet domain covering this request.
    """

    def __init__(
        self,
        program,
        basis,
        grid,
        *,
        tile_points=64,
        resource_budget=None,
        spatial=None,
    ):
        if not isinstance(program, NativeContractionProgram) or not isinstance(
            basis, NativeAO
        ):
            raise TypeError("expected native contraction program and NativeAO")
        if not isinstance(grid, (ExplicitGrid, MolecularGrid)):
            raise TypeError("expected fixed explicit or molecular quadrature")
        if isinstance(grid, MolecularGrid) and (
            grid.atoms,
            grid.charge,
            grid.multiplicity,
        ) != (basis.atoms, basis.charge, basis.multiplicity):
            raise ValueError("stale molecular grid for XC contraction")
        checked_int(tile_points, "XC contraction tile points")
        if spatial is not None:
            if not isinstance(spatial, PreparedSpatialGrid) or spatial.backend != "cpu":
                raise TypeError("native CPU XC requires a CPU spatial adapter")
            if (
                spatial.basis.identity != basis.identity
                or spatial.source_grid.identity != grid.identity
            ):
                raise ValueError("stale spatial XC basis/quadrature")
            if program.contract.ao_order > spatial.tile_plan.order:
                raise ValueError(
                    "spatial certificate does not cover the requested AO jet domain"
                )
            tile_points = spatial.tile_plan.tile_points
        npoint = grid.npoint if isinstance(grid, MolecularGrid) else len(grid.points)
        grid_bytes = (
            grid.numeric_bytes
            if isinstance(grid, MolecularGrid)
            else byte_product(npoint, 5, 8)
        )
        setup = grid.setup_scratch_bytes if isinstance(grid, MolecularGrid) else 0
        self.program, self.basis, self.grid, self.spatial = (
            program,
            basis,
            grid,
            spatial,
        )
        self.tile_points, self.npoint = tile_points, npoint
        self.budget = resource_budget or ResourceBudget()
        self._lock, self._closed = threading.RLock(), False
        self._mask = None if spatial is None else spatial.tasks.identity
        self._signature = (
            program.contract.identity,
            canonical_hash(program.metadata),
            basis.identity,
            grid.identity,
            self._mask,
        )
        self.identity = canonical_hash(
            {
                "schema": "vibeqc.prepared-xc-contractions.v1",
                "scientific": self._signature,
                "native": program.artifact.metadata["key"],
                "tile_points": tile_points,
            }
        )
        # Conservative numeric capacities cover immutable input/output copies,
        # full D/delta-D and spin matrices, AO/pullback/directional panels, and
        # point features/coefficient outputs. Native scalar SSA is per point,
        # so it never retains one array per DAG node over the whole tile.
        fixed = (
            basis.numeric_bytes
            + grid_bytes
            + byte_product(8, 16 * npoint + 12 * basis.natom)
        )
        workspace = byte_product(
            8,
            20 * basis.nao * basis.nao
            + 128 * tile_points * basis.nao
            + 256 * tile_points,
        )
        request = ResourceRequest(
            "xc_contractions",
            ResourceIdentity(
                "dft",
                "xc_contractions",
                "cpu",
                "fp64",
                json.dumps(
                    {
                        "contract": program.contract.identity,
                        "basis": basis.identity,
                        "grid": grid.identity,
                        "mask": self._mask,
                        "tile_points": tile_points,
                    }
                ),
                (program.contract.request.observable,),
                "streamed_compact_native",
            ),
            (
                ResourceCandidate(
                    "streamed",
                    "streamed",
                    (
                        ResourceEstimate(
                            "xc_inputs_and_outputs",
                            fixed,
                            "pageable",
                            0,
                            1,
                            kind="persistent",
                        ),
                        ResourceEstimate(
                            "xc_numeric_workspace", workspace + setup, "pageable", 0, 1
                        ),
                    ),
                ),
            ),
            (
                "Python object headers and allocator rounding",
                "CPU BLAS workspace/thread stacks and native scalar stack/code pages",
                "caller-retained results from previous executions",
                "borrowed spatial-owner capacities may be conservatively charged again",
            ),
        )
        requests = () if spatial is None else spatial.resource_plan.requests
        self.resource_plan = plan_resources(
            (*requests, request), self.budget
        ).require_feasible()
        self.statistics = {}

    def _check(self):
        if self._closed:
            raise RuntimeError("prepared XC contractions are closed")
        mask = None if self.spatial is None else self.spatial.tasks.identity
        if (
            self.program.contract.identity,
            canonical_hash(self.program.metadata),
            self.basis.identity,
            self.grid.identity,
            mask,
        ) != self._signature:
            raise ValueError("stale XC contraction or spatial mask identity")
        if self.spatial is not None:
            self.spatial._check()

    def _collocation(self, density):
        order = self.program.contract.ao_order
        if self.spatial is None:
            for tile in _tiles(self.grid, self.tile_points):
                jets = self.basis.evaluate(tile.points, order, budget_bytes=MAX_BYTES)
                yield (
                    np.arange(tile.begin, tile.begin + len(tile.weights)),
                    None,
                    tile.weights,
                    jets,
                    None,
                )
        elif self.program.contract.request.observable != "potential":
            # Response/geometry need their own generated contractions. Borrow
            # validated immutable maps without computing an unused feature
            # sweep first; selected NativeAO evaluates only these columns.
            for task in self.spatial.tasks.tasks:
                for begin in range(0, len(task.point_ids), self.tile_points):
                    ids = task.point_ids[begin : begin + self.tile_points]
                    jets = self.basis.evaluate(
                        self.spatial.grid.points[ids],
                        order,
                        ao_ids=task.ao_ids,
                        budget_bytes=MAX_BYTES,
                    )
                    yield ids, task.ao_ids, self.spatial.grid.weights[ids], jets, None
        else:
            requested = (
                ("rho",)
                if self.program.contract.ingredients.family == "lda"
                else ("rho", "gradient", "sigma")
            )
            for tile in self.spatial.iter_features(
                density, include_jets=True, ingredients=requested, order=order
            ):
                yield (
                    tile.point_ids,
                    tile.ao_ids,
                    tile.weights,
                    tile.ao_jets,
                    tile.features,
                )

    def execute(self, density, *, delta_density=None):
        """Execute one fixed-density request without caching numerical tile data."""
        # There is no yield to user code during execution. Hold the borrowed
        # CPU map lock so reconfiguration cannot mix old AO maps with new
        # points between collocation and contraction.
        with self._lock, nullcontext() if self.spatial is None else self.spatial._lock:
            self._check()
            started = perf_counter()
            d = spin_densities(density, self.basis.nao)
            observable = self.program.contract.request.observable
            if observable != "response" and delta_density is not None:
                raise ValueError("density direction requires a response request")
            dd = (
                spin_densities(delta_density, self.basis.nao)
                if observable == "response"
                else None
            )
            if self.program.spec.spin == "unpolarized" and (
                not np.array_equal(d[0], d[1])
                or (dd is not None and not np.array_equal(dd[0], dd[1]))
            ):
                raise UnsupportedXC(
                    "unpolarized native XC requires equal spin matrices and directions"
                )
            nspin = 2 if self.program.spec.spin == "polarized" else 1
            result = {"energy": 0.0, "electrons": np.zeros(2)}
            if observable in ("potential", "response"):
                result[observable] = np.zeros((nspin, self.basis.nao, self.basis.nao))
            if observable == "geometry":
                centers, points, weights = (
                    np.zeros((self.basis.natom, 3)),
                    np.zeros((self.npoint, 3)),
                    np.zeros(self.npoint),
                )
                ao_atoms = np.repeat(
                    [s.atom_index for s in self.basis.shells],
                    [
                        2 * s.angular_momentum + 1
                        if self.basis.representation == "real_spherical"
                        else (s.angular_momentum + 1) * (s.angular_momentum + 2) // 2
                        for s in self.basis.shells
                    ],
                )
            tiles = evaluated_tiles = 0
            for ids, active, quadrature, jets, features in self._collocation(d):
                self._check()
                if active is not None and len(active) == 0:
                    # The fixed empty map contributes constant zero for every
                    # D and motion on that branch, without vacuum derivatives.
                    tiles += 1
                    continue
                local = d if active is None else d[:, active[:, None], active[None, :]]
                options = {}
                if observable == "response":
                    options["delta_density"] = (
                        dd
                        if active is None
                        else dd[:, active[:, None], active[None, :]]
                    )
                elif observable == "geometry":
                    options.update(
                        ao_atoms=ao_atoms if active is None else ao_atoms[active],
                        natom=self.basis.natom,
                    )
                values = (
                    self.program.potential_tile(jets, features, quadrature)
                    if observable == "potential" and features is not None
                    else self.program.evaluate(jets, local, quadrature, **options)
                )
                evaluated_tiles += 1
                result["energy"] += values["energy"]
                result["electrons"] += values["electrons"]
                if observable in ("potential", "response"):
                    if active is None:
                        result[observable] += values[observable]
                    else:
                        result[observable][:, active[:, None], active[None, :]] += (
                            values[observable]
                        )
                if observable == "geometry":
                    partials = values["geometry"]
                    centers += partials.centers
                    points[ids] = partials.points
                    weights[ids] = partials.weights
                tiles += 1
            self._check()
            if not np.isfinite(result["energy"]):
                raise ArithmeticError("nonfinite accumulated XC energy")
            for name in ("electrons", "potential", "response"):
                if name in result:
                    result[name] = immutable(result[name])
            if observable == "geometry":
                result["geometry"] = GeometryPartials(
                    immutable(centers), immutable(points), immutable(weights)
                )
            self.statistics = {
                "tiles": tiles,
                "seconds": perf_counter() - started,
                "scalar_calls": evaluated_tiles,
                "point_coefficient_calls": 0
                if observable == "energy"
                else evaluated_tiles,
                "ao_pullback_calls": nspin * evaluated_tiles
                if observable == "geometry"
                else 0,
                "matrix_assembly_products": nspin
                * evaluated_tiles
                * (1 if self.program.contract.ingredients.family == "lda" else 2)
                if observable in ("potential", "response")
                else 0,
                "planned_host_peak_bytes": self.resource_plan.peak_bytes["host"],
                "memory_scope": "numeric capacity bound with explicit resource-plan exclusions; not a measured process peak",
            }
            # Count logical matrix products in the actual nonempty tile
            # schedule, including feature reductions and geometric D*AO jets.
            # These are separate from generated point-function calls; BLAS
            # may internally split a matrix product into several kernels.
            self.statistics["density_matrix_products"] = evaluated_tiles * (
                4 if observable == "response" else 2
            )
            self.statistics["geometry_matrix_products"] = (
                nspin
                * evaluated_tiles
                * (1 if self.program.contract.ingredients.family == "lda" else 4)
                if observable == "geometry"
                else 0
            )
            self.statistics["total_matrix_products"] = sum(
                self.statistics[key]
                for key in (
                    "density_matrix_products",
                    "geometry_matrix_products",
                    "matrix_assembly_products",
                )
            )
            return result

    def close(self):
        """End this execution lease without closing borrowed basis/spatial owners."""
        with self._lock:
            self._closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, *_):
        self.close()
