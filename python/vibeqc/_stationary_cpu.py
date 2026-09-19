"""Complete bounded CPU RKS/UKS gradient diagnostic for issue #163 B2/B3.

SCF state, AO jets, XC point coefficients and generated integral derivatives
execute natively. The explicit native selector also compiles TensorIR weights/
reduction, local AO pullbacks and Becke adjoints from their existing graphs.
Python orchestration and NumPy BLAS/map reductions remain host boundaries.
The reference selector retains interpreter execution for A/B diagnostics; neither
selector enables public Calculator forces or global resource qualification.
"""

import ctypes as ct
import os
import tempfile
from dataclasses import dataclass
from itertools import islice, product
from pathlib import Path
from types import MappingProxyType

import numpy as np
from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.native_runtime import compile_runtime
from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.integral.first_derivative_native import emit_first_derivative_cpu
from vibeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)
from vibeqc_compiler.tensor import execute
from vibeqc_compiler.tensor.cpu import NativeTensorProgram
from vibeqc_compiler.xc.contractions import ContractionProgram
from vibeqc_compiler.xc.grid_native import NativeGridContraction
from vibeqc_compiler.xc.grid_response import partition_response
from vibeqc_compiler.xc.native import NativeContractionProgram

from ._dft_gradient import (
    StationaryDerivativeContract,
    _native_ao_atoms,
    native_ao_geometry_identity,
)
from .ks import resolve_ks_method


@dataclass(frozen=True)
class DiagnosticStationaryGradient:
    """Transactional detached evidence; units are Eh/bohr, sign is gradient."""

    gradient: np.ndarray
    components: object
    plan_identity: str
    state_identity: object
    work: object
    execution: str = "native-cpu-primitives/compiler-interpreter-diagnostic-v1"


def _publish_source(path, source):
    """Publish complete immutable compiler input before hashing or compilation.

    Concurrent calls may reuse a cache. Never truncate a hash-named source
    while another call is hashing/compiling it; the shared artifact cache
    assumes that its input is stable for the entire compilation.
    """
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=".stationary-source-",
            suffix=".cpp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(source)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class _PrimitiveExecutor:
    """Compile finite requested component coverage and stream fixed-size records.

    The common compiler cache verifies source, transitive headers and binary.
    The first integration deliberately admits s/p bases only to bound generated
    code size. Wider component/subset scheduling is separate qualification work.
    """

    def __init__(self, basis, cache, primitive_tile, compiler):
        if any(shell.angular_momentum > 1 for shell in basis.shells):
            raise NotImplementedError(
                "complete CPU gradient diagnostic supports s/p bases only"
            )
        start = 3 * basis.natom
        self.primitives = basis.packed[start : start + 2 * basis.nprimitive].reshape(
            -1, 2
        )
        self.aos = basis.packed[start + 2 * basis.nprimitive :].reshape(-1, 16)
        self.centers = basis.packed[:start].reshape(-1, 3)
        self.components = tuple(
            "".join(axis * int(power) for axis, power in zip("xyz", r[4:7]))
            for r in self.aos
        )
        if any(int(r[3]) != 1 for r in self.aos):
            raise NotImplementedError(
                "this diagnostic requires single-component public AOs"
            )
        domain = sorted(set(self.components))
        requests = [
            (op, c)
            for op in ("overlap", "kinetic", "nuclear_attraction")
            for c in product(domain, repeat=2)
        ]
        requests += [("four_center_eri", c) for c in product(domain, repeat=4)]
        requests += [("nuclear", ())]
        self.kinds = {key: i for i, key in enumerate(requests)}
        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        source = emit_first_derivative_cpu(tuple(requests))
        path = cache / (canonical_hash(source) + ".cpp")
        _publish_source(path, source)
        headers = tuple(
            asset_path("src/integrals/" + name)
            for name in (
                "first_derivative_runtime.hpp",
                "eri_geometry.hpp",
                "range_moments.hpp",
            )
        )
        artifact = compile_runtime(
            compiler,
            cache,
            path,
            headers=headers,
            options=("-ffp-contract=off", f"-I{headers[0].parents[1]}"),
        )
        self.library = ct.CDLL(str(artifact.library))
        self.call = self.library.vibeqc_first_derivative_cpu
        self.call.argtypes = [
            ct.c_uint,
            ct.POINTER(ct.c_double),
            ct.c_size_t,
            ct.POINTER(ct.c_double),
        ]
        self.call.restype = ct.c_int
        self.buffer = np.zeros((primitive_tile, 17))
        self.records = 0

    def _run(self, kind, count):
        out = np.empty((4, 3))
        if self.call(
            kind,
            self.buffer.ctypes.data_as(ct.POINTER(ct.c_double)),
            count,
            out.ctypes.data_as(ct.POINTER(ct.c_double)),
        ):
            raise ArithmeticError("generated CPU integral derivative failed")
        self.records += count
        return out

    def integral(self, operator, indices, weight, nucleus=None):
        """Contract primitive normalization before publishing center derivatives."""
        rows = self.aos[list(indices)]
        rank = len(indices)
        owners = [int(r[0]) for r in rows]
        if nucleus is not None:
            owners.append(nucleus)
        self.buffer.fill(0)
        self.buffer[:, 4 : 4 + 3 * len(owners)] = self.centers[owners].reshape(-1)
        components = tuple(self.components[i] for i in indices)
        kind = self.kinds[operator, components]
        primitive_ranges = [range(int(r[1]), int(r[1] + r[2])) for r in rows]
        norm = weight * np.prod(rows[:, 7])
        result = np.zeros((4, 3))
        count = 0
        for ids in product(*primitive_ranges):
            primitives = self.primitives[list(ids)]
            self.buffer[count, :rank] = primitives[:, 0]
            self.buffer[count, 16] = norm * np.prod(primitives[:, 1])
            count += 1
            if count == len(self.buffer):
                result += self._run(kind, count)
                count = 0
        if count:
            result += self._run(kind, count)
        return owners, result[: len(owners)]

    def nuclear(self, a, b, charges):
        self.buffer.fill(0)
        self.buffer[0, :2] = charges[[a, b]]
        self.buffer[0, 4:10] = self.centers[[a, b]].reshape(-1)
        self.buffer[0, 16] = 1
        return self._run(self.kinds["nuclear", ()], 1)[:2]


def complete_rks_gradient_diagnostic(
    state,
    basis,
    *,
    cache,
    tile_points=256,
    integral_terms=32,
    primitive_tile=128,
    compiler=None,
    execution="reference",
):
    """Consume one live native CPU RKS/UKS state with complete plan-owned sources.

    Admitted domain: direct real FP64 integer RKS/UKS, canonical
    LDA or PBE, s/p AOs, native unpruned version-one grid, distinct nuclei and no
    point/center collisions. CPU is explicit; CUDA snapshots are rejected.
    Caller chooses an ignored/temporary compilation cache and may supply a
    CppCompilerAdapter; otherwise CXX (or c++) selects the executable. Scientific work is
    full ordered AO pairs/quartets, without screening or symmetry shortcuts.
    Working arrays scale with a point tile times (AO + atom), one primitive
    record tile, D/W, and seven atom gradients, never coordinate-grid-AO pairs.
    The native state already retains its full discrete grid and dense SCF data.
    execution="native" selects compiled consumers of the same mathematical
    graphs. execution="reference" retains the validated interpreter route.
    Both retain Python primitive enumeration/scatter and NumPy XC BLAS/maps;
    neither establishes an overall endpoint/SCF memory budget.
    Scalar-ECP CPU snapshots additionally bind effective ionic charges and two
    residual derivative sources to the actual energy owner. Their existing
    independent CPU provider materializes 2*3*natom*nao**2 derivative elements;
    this is an explicit diagnostic, not generated native ECP production or a
    public/budget-qualified DFT force capability.
    """
    if execution not in ("reference", "native"):
        raise ValueError("execution must be reference or native")
    contract = StationaryDerivativeContract(state.identity)
    contract.validate(state)
    if state._source.backend != "cpu":
        raise NotImplementedError("complete diagnostic requires a native CPU KS state")
    if (
        basis.identity != state.identity.basis_identity
        or native_ao_geometry_identity(basis) != state.identity.geometry_identity
    ):
        raise ValueError("stationary diagnostic basis/geometry mismatch")
    for value, name, cap in (
        (tile_points, "tile_points", 4096),
        (integral_terms, "integral_terms", 128),
        (primitive_tile, "primitive_tile", 4096),
    ):
        if type(value) is not int or not 1 <= value <= cap:
            raise ValueError(f"{name} must be an integer in [1,{cap}]")
    method, functional = resolve_ks_method(state.identity.method)
    plan = StationaryGradientPlan(
        method,
        StationaryMeanField(
            SCF_POINT_MODEL,
            hamiltonian=state._source.hamiltonian,
        ),
    )
    density = state.density if contract.spin == "polarized" else state.density[0]
    if compiler is None:
        compiler = CppCompilerAdapter(Path(os.environ.get("CXX", "c++")))
    if not isinstance(compiler, CppCompilerAdapter):
        raise TypeError("the CPU diagnostic requires an explicit C++ compiler adapter")
    native = _PrimitiveExecutor(basis, cache, primitive_tile, compiler)
    natom, n = basis.natom, basis.nao
    components = {name: np.zeros((natom, 3)) for name in plan.source_names}
    charges = np.asarray([atom.atomic_number for atom in basis.atoms]) - np.asarray(
        state._source.ecp_cores
    )
    work = {
        "ordered_pairs": n * n,
        "ordered_quartets": n**4,
        "xc_points": len(state.grid.points),
        "grid_directional_points": (
            3 * natom * len(state.grid.points) if execution == "reference" else 0
        ),
        "grid_adjoint_points": len(state.grid.points) if execution == "native" else 0,
        "grid_pair_visits": (2 if execution == "native" else 3 * natom)
        * (natom * (natom - 1) // 2)
        * len(state.grid.points),
        "grid_center_pair_validations": (
            ((len(state.grid.points) + tile_points - 1) // tile_points)
            * (natom * (natom - 1) // 2)
            if execution == "native"
            else 0
        ),
        "point_tile_capacity": tile_points,
        "primitive_tile_capacity": primitive_tile,
        "integral_term_capacity": integral_terms,
    }
    tensor_consumers = {}
    # TensorIR AD supplies D, D*D/2, and -W. The runtime never rebuilds these
    # scientific coefficients from a method-name-specific gradient formula.
    for source, rank in (("one_electron", 2), ("overlap_pulay", 2), ("coulomb", 4)):
        iterator = product(range(n), repeat=rank)
        while tuples := tuple(islice(iterator, integral_terms)):
            ids = np.asarray(tuples)
            key = (source, len(tuples))
            if key not in tensor_consumers:
                block = plan.integral_block(source, terms=len(tuples))
                tensor_consumers[key] = (
                    NativeTensorProgram(block.weights, compiler=compiler, cache=cache)
                    if execution == "native"
                    else block.weights
                )
            if source == "overlap_pulay":
                feeds = {
                    "weighted_density": state.weighted_density[:, ids[:, 0], ids[:, 1]]
                }
            else:
                feeds = {"density_left": state.density[:, ids[:, 0], ids[:, 1]]}
                if rank == 4:
                    feeds["density_right"] = state.density[:, ids[:, 2], ids[:, 3]]
            consumer = tensor_consumers[key]
            weights = (
                consumer.execute(feeds)["weights"]
                if execution == "native"
                else execute(consumer, feeds).outputs["weights"]
            )
            for indices, weight in zip(tuples, weights, strict=True):
                operator = {
                    "one_electron": "kinetic",
                    "overlap_pulay": "overlap",
                    "coulomb": "four_center_eri",
                }[source]
                owners, values = native.integral(operator, indices, weight)
                np.add.at(components[source], owners, values)
                if source == "one_electron":
                    for atom in range(natom):
                        owners, values = native.integral(
                            "nuclear_attraction", indices, weight * charges[atom], atom
                        )
                        np.add.at(components[source], owners, values)
    for a in range(natom):
        for b in range(a):
            np.add.at(components["nuclear"], [a, b], native.nuclear(a, b, charges))

    if state._source.hamiltonian == "scalar-semilocal-ecp":
        derivatives = state._source.ecp_derivatives()
        work["ecp_derivative_bytes"] = derivatives.nbytes
        # The existing independent CPU ECP provider includes both AO-center
        # and ECP-center motion. TensorIR generates spin-summed weights and
        # contracts bounded AO-pair tiles; no separate force formula lives here.
        for k, source in enumerate(("ecp_local", "ecp_nonlocal")):
            iterator = product(range(n), repeat=2)
            while tuples := tuple(islice(iterator, integral_terms)):
                ids = np.asarray(tuples)
                block = plan.integral_block(
                    source, terms=len(tuples), coordinates=3 * natom
                )
                feeds = {
                    "density_left": state.density[:, ids[:, 0], ids[:, 1]],
                    "integral_derivatives": np.ascontiguousarray(
                        derivatives[k, :, :, ids[:, 0], ids[:, 1]].reshape(
                            len(tuples), 3 * natom
                        )
                    ),
                }
                value = (
                    NativeTensorProgram(
                        block.contraction, compiler=compiler, cache=cache
                    ).execute(feeds)["gradient"]
                    if execution == "native"
                    else execute(block.contraction, feeds).outputs["gradient"]
                )
                components[source] += value.reshape(natom, 3)

    program = (
        NativeContractionProgram(functional, "geometry", compiler=compiler, cache=cache)
        if execution == "native"
        else ContractionProgram(functional, "geometry")
    )
    grid, spec = state.grid, state._source.grid_spec
    grid_consumer = (
        NativeGridContraction(
            compiler=compiler, cache=cache, iterations=spec.partition_iterations
        )
        if execution == "native"
        else None
    )
    ao_atoms = _native_ao_atoms(basis)
    pbe = contract.family == "gga"
    for begin in range(0, len(grid.points), tile_points):
        end = min(begin + tile_points, len(grid.points))
        points, weights, atoms = (
            grid.points[begin:end],
            grid.weights[begin:end],
            np.asarray(grid.owners[begin:end]),
        )
        jets = basis.evaluate(points, program.contract.ao_order)
        features = program.features(jets, density)
        coefficients = state._source.evaluate_xc_points(
            pbe,
            features["rho"],
            features.get("gradient", np.zeros((2, end - begin, 3))),
        )
        partials = program.geometry_from_cartesian_coefficients(
            jets,
            density,
            weights,
            coefficients["energy"],
            coefficients["rho"],
            coefficients["gradient"] if pbe else None,
            ao_atoms=ao_atoms,
            natom=natom,
        )
        components["xc_ao"] += partials.centers
        np.add.at(components["xc_grid"], atoms, partials.points)
        if grid_consumer is not None:
            with np.errstate(over="raise", invalid="raise"):
                seeds = partials.weights * state._source.atomic_weights[begin:end]
            components["xc_weight"] += grid_consumer.contract(
                points,
                native.centers,
                atoms.astype(np.int64),
                seeds,
                coincident_tolerance=spec.coincident_tolerance,
            )
            continue
        # One coordinate at a time bounds storage; this interpreter boundary
        # explicitly costs 3*natom partition traversals per point tile.
        for a in range(natom):
            for axis in range(3):
                motion = np.zeros((natom, 3))
                motion[a, axis] = 1
                response = partition_response(
                    points,
                    native.centers,
                    point_motion=motion[atoms],
                    center_motion=motion,
                    iterations=spec.partition_iterations,
                    coincident_tolerance=spec.coincident_tolerance,
                )
                selected = (np.arange(end - begin), atoms)
                derivative = response.directional[selected]
                # Native atomic measures avoid division by tiny partition
                # weights and preserve the exact radial/angular prescription.
                components["xc_weight"][a, axis] += np.dot(
                    partials.weights,
                    state._source.atomic_weights[begin:end] * derivative,
                )
    gradient = (
        NativeTensorProgram(
            plan.reduction_program(atoms=natom, sources=components.keys()),
            compiler=compiler,
            cache=cache,
        ).execute(components)["gradient"]
        if execution == "native"
        else plan.reduce_diagnostic(components, atoms=natom)
    )
    contract.validate(state)  # No partial publication after replay/failure/replacement.
    work["primitive_records"] = native.records
    return DiagnosticStationaryGradient(
        immutable(gradient),
        MappingProxyType({key: immutable(value) for key, value in components.items()}),
        plan.identity,
        state.identity,
        MappingProxyType(work),
        execution=(
            "compiled-cpu-consumers/python-numpy-orchestration-v1"
            if execution == "native"
            else "native-cpu-primitives/compiler-interpreter-diagnostic-v1"
        ),
    )
