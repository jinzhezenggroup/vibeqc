"""Bounded native CPU XC with CPU or explicit CUDA density collocation.

The shared ResourceBudget owns the cap. NativeAO, generated point code and
CPU matrix products stream tiles; neither AO^4 data nor a molecular AO table
is retained. This explicit candidate does not register a complete DFT method.
"""

import json
import threading
import typing
from contextlib import ExitStack
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
from vibeqc_compiler.dft import DensitySource, ExplicitGrid, MolecularGrid, NativeAO
from vibeqc_compiler.dft.cuda import CudaGrid
from vibeqc_compiler.dft.features import density_feature_block, spin_densities
from vibeqc_compiler.dft.grid import checked_int
from vibeqc_compiler.dft.spatial_prepared import PreparedSpatialGrid

from .contractions import GeometryPartials
from .integration import _tiles
from .native import NativeContractionProgram
from .program_ir import fixed_density_tile_program
from .spec import UnsupportedXC, functional


def _native_device_xc(
    program: typing.Any, spatial: typing.Any, density_grid: typing.Any
) -> typing.Any:
    """Keep feature validation and execution on the same canonical CUDA route."""
    return (
        spatial is not None
        and density_grid is not None
        and program.contract.request.observable == "potential"
        and program.spec.identifier in ("LDA_XC_PW", "PBE")
        and program.spec == functional(program.spec.identifier, spin=program.spec.spin)
    )


class PreparedXCContractions:
    """Compose a compiled contraction program with immutable basis/quadrature.

    Output matrices retain functional-spin layout; the existing method adapter
    owns averaging for total-density input. Geometry returns independent point,
    AO-center and weight partials. A spatial adapter keeps its fixed mask and
    must cover the requested jet domain. Canonical LDA/PBE potentials on CUDA
    spatial owners use native device XC; other compositions retain CPU XC.
    """

    def __init__(
        self,
        program: typing.Any,
        basis: typing.Any,
        grid: typing.Any,
        *,
        tile_points: typing.Any = 64,
        resource_budget: typing.Any = None,
        spatial: typing.Any = None,
        density_grid: typing.Any = None,
    ) -> None:
        # Reconfiguration publishes several related fields under this lock.
        # Keep validation, identity capture and resource composition in one
        # snapshot so a consumer cannot bind old quadrature to a new owner.
        with ExitStack() as leases:
            if spatial is not None:
                if not isinstance(spatial, PreparedSpatialGrid):
                    raise TypeError("expected PreparedSpatialGrid")
                leases.enter_context(spatial._lock)
            self._initialize(
                program,
                basis,
                grid,
                tile_points=tile_points,
                resource_budget=resource_budget,
                spatial=spatial,
                density_grid=density_grid,
            )

    def _initialize(
        self,
        program: typing.Any,
        basis: typing.Any,
        grid: typing.Any,
        *,
        tile_points: typing.Any,
        resource_budget: typing.Any,
        spatial: typing.Any,
        density_grid: typing.Any,
    ) -> None:
        """Capture the complete borrowed configuration under its spatial lock."""
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
            spatial._check()
            if density_grid is not None:
                raise ValueError("spatial owns its collocation; omit density_grid")
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
            density_grid = spatial._cuda
        if density_grid is not None:
            if not isinstance(density_grid, CudaGrid):
                raise TypeError("density_grid must be a CudaGrid owner")
            if spatial is None and density_grid.plan.active_ao_capacity is not None:
                raise ValueError(
                    "CUDA XC density adapter currently requires dense AO tiles"
                )
            if program.contract.request.observable != "potential":
                raise ValueError("CUDA density adapter supports fixed-density E/V only")
            required = {"rho"}
            if program.contract.ingredients.family != "lda":
                required.add("gradient")
                # Native CUDA XC forms sigma from gradients. The CPU fallback
                # consumes an explicit sigma feature in its collocation tiles.
                if not _native_device_xc(program, spatial, density_grid):
                    required.add("sigma")
            if program.contract.ingredients.family == "mgga":
                required.add("tau")
            if (
                density_grid.basis_identity != basis.identity
                or density_grid.plan.nao != basis.nao
                or density_grid.plan.order < program.contract.ao_order
                or not required.issubset(density_grid.ingredients)
            ):
                raise ValueError(
                    "CUDA density basis or requested ingredients do not match XC"
                )
            tile_points = density_grid.plan.tile_points
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
        self.density_grid = density_grid
        self._density_signature = self._density_contract()
        self.budget = resource_budget or ResourceBudget()
        self._lock, self._closed = threading.RLock(), False
        self._mask = None if spatial is None else spatial.tasks.identity
        self._spatial_resources = (
            None if spatial is None else spatial.resource_plan.identity
        )
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
                **(
                    {"density_collocation": self._density_signature}
                    if density_grid is not None
                    else {}
                ),
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
        if density_grid is not None and spatial is None:
            # The spatial plan already charges its complete CUDA owner.
            requests += density_grid.resource_plan.requests
        self.resource_plan = plan_resources(
            (*requests, request), self.budget
        ).require_feasible()
        # Phase A ProgramIR describes only the synchronous dense CPU E/V
        # boundary. Its estimates are NOT added to the conservative total
        # resource plan above: those allocations are already accounted for.
        self._packed_feature_layout = (
            spatial is None
            and density_grid is None
            and program.contract.request.observable == "potential"
            and program.spec.spin == "polarized"
        )
        self._tile_program = (
            fixed_density_tile_program(
                program.contract,
                nao=basis.nao,
                tile_points=tile_points,
                basis_bytes=basis.numeric_bytes,
                grid_bytes=grid_bytes,
                basis_identity=basis.identity,
                native_identity=canonical_hash(program.metadata),
                packed_features=self._packed_feature_layout,
            )
            if spatial is None
            and density_grid is None
            and program.contract.request.observable == "potential"
            else None
        )
        self._tile_program_identity = (
            None if self.tile_program is None else self.tile_program.identity
        )
        terminal = "vxc" if self._packed_feature_layout else "xc"
        self._tile_releases = (
            ()
            if self.tile_program is None
            else self.tile_program.release_after(terminal)
        )
        # This CPU template implements only these checked synchronous releases.
        # Async/aliasing providers still require a different explicit contract.
        self._release_tile_boundaries = bool(self._tile_releases)
        self.statistics = {}

    @property
    def tile_program(self) -> typing.Any:
        """Immutable boundary-only ProgramIR, or None for unqualified routes."""
        return self._tile_program

    def _density_contract(self) -> typing.Any:
        """Borrow only fixed CUDA topology/code; each call uploads its current D/B."""
        cuda = self.density_grid
        return (
            None
            if cuda is None
            else (
                cuda.basis_identity,
                cuda.basis_generation,
                cuda.ingredients,
                cuda.resource_plan.identity,
                cuda.artifact.metadata["key"],
            )
        )

    def _check(self) -> None:
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
            if self.spatial.resource_plan.identity != self._spatial_resources:
                raise ValueError("stale spatial resource contract")
            if self.spatial._cuda is not self.density_grid:
                raise ValueError("stale spatial CUDA owner; prepare a new XC consumer")
        if self._density_contract() != self._density_signature:
            raise ValueError("stale CUDA density collocation contract")
        if self.density_grid is not None:
            self.density_grid._check_open()

    def _collocation(self, density: typing.Any) -> typing.Any:
        order = self.program.contract.ao_order
        if self.density_grid is not None:
            if self.spatial is None:
                tiles = (
                    (
                        np.arange(tile.begin, tile.begin + len(tile.weights)),
                        None,
                        tile.points,
                        tile.weights,
                    )
                    for tile in _tiles(self.grid, self.tile_points)
                )
            else:
                # Consume the same immutable task descriptors as diagnostic
                # and device-task execution, retaining every local cross term.
                tiles = (
                    (
                        ids,
                        task.ao_ids,
                        self.spatial.grid.points[ids],
                        self.spatial.grid.weights[ids],
                    )
                    for task, ids in self.spatial._tiles()
                )
            for ids, active, points, weights in tiles:
                # Transfers are explicit: AO/features run on GPU, generated
                # XC point code and AO potential assembly remain on CPU.
                values = self.density_grid.evaluate(
                    points,
                    ao_ids=active,
                    stamp=self._execution_stamp,
                    download_jets=True,
                )
                jets = values.pop("ao_jets")
                yield (
                    ids,
                    active,
                    weights,
                    jets,
                    values,
                )
        elif self.spatial is None:
            for tile in _tiles(self.grid, self.tile_points):
                jets = self.basis.evaluate(tile.points, order, budget_bytes=MAX_BYTES)
                yield (
                    np.arange(tile.begin, tile.begin + len(tile.weights)),
                    None,
                    tile.weights,
                    jets,
                    None,
                )
                # A suspended generator is an owner too. The consumer has
                # finished this tile before requesting the next allocation.
                del jets
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
                else ("rho", "gradient", "sigma", "tau")
                if self.program.contract.ingredients.family == "mgga"
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

    def _device_xc(
        self,
        density: typing.Any,
        stamp: typing.Any,
        route: typing.Any,
        nspin: typing.Any,
    ) -> typing.Any:
        """Run the audited LDA/PBE potential contract entirely on CUDA tiles."""
        result = {
            "energy": 0.0,
            "electrons": np.zeros(2),
            "potential": np.zeros((nspin, self.basis.nao, self.basis.nao)),
        }
        tiles = evaluated = 0
        last_nonempty = max(
            (
                index
                for index, (task, _) in enumerate(self.spatial._tiles())
                if len(task.ao_ids)
            ),
            default=None,
        )
        with self.spatial.device_xc_tasks(
            density, self.program.spec.identifier, stamp=stamp, route=route
        ) as tasks:
            for index, (_, ids, lease) in enumerate(tasks):
                active = lease.view.nactive
                if active:
                    try:
                        integrals, potential = lease.xc(
                            self.spatial.grid.weights[ids],
                            self.program.spec.identifier,
                            restricted=nspin == 1,
                            reset=evaluated == 0,
                            download=index == last_nonempty,
                        )
                    except RuntimeError as error:
                        if str(error) == "invalid or nonfinite CUDA XC output":
                            raise UnsupportedXC(
                                "CUDA XC output is outside the audited interior-v1 domain"
                            ) from error
                        raise
                    result["energy"] += integrals[0]
                    result["electrons"] += integrals[1:]
                    evaluated += 1
                    if potential is not None:
                        result["potential"] = potential if nspin == 2 else potential[:1]
                tiles += 1
        return result, tiles, evaluated

    def execute(
        self,
        density: typing.Any,
        *,
        delta_density: typing.Any = None,
        stamp: typing.Any = None,
        route: typing.Any = "auto",
    ) -> typing.Any:
        """Execute fixed-density XC; the optional CUDA owner requires DensitySource.

        Supply the current consumer stamp for CUDA D/C collocation. Entire
        executions hold the borrowed owner's lock, isolating concurrent source
        uploads. GPU failures propagate; no CPU collocation retry is implicit.
        """
        # There is no yield to user code during execution. Hold the borrowed
        # CPU map lock so reconfiguration cannot mix old AO maps with new
        # points between collocation and contraction.
        with self._lock, ExitStack() as leases:
            if self.spatial is not None:
                leases.enter_context(self.spatial._lock)
                # A device task holds the CUDA lock while its outer context
                # needs the spatial lock to close. Reject its lease before
                # waiting on CUDA, otherwise those two threads can deadlock.
                self.spatial._check()
            if self.density_grid is not None:
                leases.enter_context(self.density_grid._lock)
            self._check()
            started = perf_counter()
            if self.density_grid is not None:
                if not isinstance(density, DensitySource):
                    raise TypeError("CUDA XC collocation requires DensitySource")
                d = density.density
            else:
                if stamp is not None or route != "auto":
                    raise ValueError("density source options require CUDA collocation")
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
            device_xc = _native_device_xc(self.program, self.spatial, self.density_grid)
            if self.density_grid is not None:
                before_metrics = self.density_grid.metrics()
                if not device_xc:
                    if self.spatial is None:
                        self.density_grid.set_source(density, stamp=stamp, route=route)
                    else:
                        self.spatial._start_execution(density, stamp=stamp, route=route)
                self._execution_stamp = stamp
            nspin = 2 if self.program.spec.spin == "polarized" else 1
            if device_xc:
                result, tiles, evaluated_tiles = self._device_xc(
                    density, stamp, route, nspin
                )
                cpu_contraction_seconds = 0.0
            else:
                result = {"energy": 0.0, "electrons": np.zeros(2)}
                if observable in ("potential", "response"):
                    result[observable] = np.zeros(
                        (nspin, self.basis.nao, self.basis.nao)
                    )
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
            if not device_xc:
                tiles = evaluated_tiles = 0
                cpu_contraction_seconds = 0.0
            for ids, active, quadrature, jets, features in (
                () if device_xc else self._collocation(d)
            ):
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
                contraction_started = perf_counter()
                packed = rows = packed_features = None
                if (
                    self._packed_feature_layout
                    and observable == "potential"
                    and features is None
                ):
                    requested = (
                        ("rho",)
                        if self.program.contract.ingredients.family == "lda"
                        else ("rho", "gradient", "sigma", "tau")
                        if self.program.contract.ingredients.family == "mgga"
                        else ("rho", "gradient", "sigma")
                    )
                    packed = density_feature_block(jets, local, ingredients=requested)
                    packed_features = packed.features()
                    rows = self.program.scalar_values_packed(packed.scalar)
                    values = self.program.potential_from_rows(
                        jets, packed_features, quadrature, rows
                    )
                else:
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
                cpu_contraction_seconds += perf_counter() - contraction_started
                tiles += 1
                if self._release_tile_boundaries:
                    # ProgramIR's final use is complete. No callback or
                    # asynchronous lease retains these dense CPU boundaries.
                    if packed is not None:
                        del packed_features, rows, packed
                    del jets, features, values
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
            if self.tile_program is not None:
                self.statistics["tile_program_identity"] = self._tile_program_identity
                self.statistics["tile_boundary_releases"] = self._tile_releases
                self.statistics["tile_layouts"] = {
                    buffer.name: buffer.layout.to_payload()
                    for buffer in self.tile_program.buffers
                    if buffer.layout is not None
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
            if self.density_grid is not None:
                cuda = self.density_grid
                after_metrics = cuda.metrics()
                count = (
                    2
                    if cuda.source_kind == "density_matrix"
                    else sum(
                        (n + cuda.plan.orbital_tile - 1) // cuda.plan.orbital_tile
                        for n in cuda.source_statistics["occupied_counts"]
                    )
                )
                # GGA's orbital route needs value plus three derivative Psi
                # products. Its D route needs only Phi*D when tau is omitted.
                panels = (
                    (4 if any(k != "rho" for k in cuda.ingredients) else 1)
                    if cuda.source_kind == "orbitals"
                    else (4 if "tau" in cuda.ingredients else 1)
                )
                self.statistics["density_matrix_products"] = (
                    0
                    if cuda.source_kind == "orbitals"
                    else evaluated_tiles * count * panels
                )
                self.statistics["orbital_matrix_products"] = (
                    evaluated_tiles * count * panels
                    if cuda.source_kind == "orbitals"
                    else 0
                )
                self.statistics["total_matrix_products"] = (
                    self.statistics["density_matrix_products"]
                    + self.statistics["orbital_matrix_products"]
                    + self.statistics["matrix_assembly_products"]
                )
                self.statistics.update(
                    collocation_backend="cuda",
                    xc_backend="native_cuda" if device_xc else "native_cpu",
                    cpu_contraction_seconds=cpu_contraction_seconds,
                    source=dict(cuda.source_statistics),
                    native_metrics=after_metrics,
                    device_seconds={
                        k: (after_metrics[k] - before_metrics[k]) / 1000
                        for k in (
                            "kernel_ms",
                            "library_ms",
                            "packing_ms",
                            "input_ms",
                            "output_ms",
                        )
                    },
                    planned_device_peak_bytes=self.resource_plan.peak_bytes["device"],
                )
                if self.spatial is not None:
                    self.statistics["spatial"] = {
                        "generation_id": self.spatial.tasks.generation_id,
                        "mask_identity": self._mask,
                        "tasks": len(self.spatial.tasks.tasks),
                        "grid_points": self.npoint,
                        "active_ao_counts": tuple(
                            len(task.ao_ids) for task in self.spatial.tasks.tasks
                        ),
                        "orbital_tile": cuda.plan.orbital_tile,
                        "ingredients": cuda.ingredients,
                        "potential_scatter_backend": (
                            "native_cuda" if device_xc else "native_cpu"
                        ),
                    }
            self.statistics["seconds"] = perf_counter() - started
            return result

    def close(self) -> None:
        """End this execution lease without closing borrowed basis/spatial owners."""
        with self._lock:
            self._closed = True

    def __enter__(self) -> typing.Any:
        self._check()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
