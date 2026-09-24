"""Complete bounded CPU RKS/UKS gradient diagnostic for issue #163 B2/B3.

SCF state, AO jets, XC point coefficients and generated integral derivatives
execute natively. The explicit native selector also compiles TensorIR weights/
reduction, local AO pullbacks and Becke adjoints from their existing graphs.
Python orchestration and NumPy BLAS/map reductions remain host boundaries.
The reference selector retains interpreter execution for A/B diagnostics. The
public CPU ECP wrapper selects native execution with additional byte admission.
"""

import ctypes as ct
import os
import tempfile
import typing
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
from vibeqc_compiler.dft.nonlocal_integration import FixedDensityNonlocalCorrelation
from vibeqc_compiler.integral.ecp_policy import (
    COARSE_POLAR_POINTS,
    COARSE_RADIAL_POINTS,
    REFINED_POLAR_POINTS,
    REFINED_RADIAL_POINTS,
)
from vibeqc_compiler.integral.first_derivative_native import emit_first_derivative_cpu
from vibeqc_compiler.method.nonlocal_correlation import NonlocalCorrelationPrimitive
from vibeqc_compiler.method.spec import RangeSeparatedExchangePrimitive
from vibeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)
from vibeqc_compiler.tensor import execute
from vibeqc_compiler.tensor.cpu import NativeTensorProgram
from vibeqc_compiler.xc.contractions import (
    ExternalPointContraction,
)
from vibeqc_compiler.xc.grid_native import NativeGridContraction
from vibeqc_compiler.xc.grid_response import partition_response
from vibeqc_compiler.xc.native import NativeContractionProgram

from ._dft_gradient import (
    StationaryDerivativeContract,
    _native_ao_atoms,
    native_ao_geometry_identity,
)
from ._stationary_rsh_cpu import RangeExchangeExecutor
from .nonlocal_runtime import NativeNonlocalPairProvider


@dataclass(frozen=True)
class DiagnosticStationaryGradient:
    """Transactional detached evidence; units are Eh/bohr, sign is gradient."""

    gradient: np.ndarray
    components: object
    plan_identity: str
    state_identity: object
    work: object
    execution: str = "native-cpu-primitives/compiler-interpreter-diagnostic-v1"


def _publish_source(path: typing.Any, source: typing.Any) -> None:
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


def _compile_primitive_library(
    source: str, cache: str | Path, compiler: CppCompilerAdapter
) -> tuple[ct.CDLL, typing.Any]:
    """Revalidate source, transitive headers, compiler and binary on every load."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
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
    library = ct.CDLL(str(artifact.library))
    call = library.vibeqc_first_derivative_cpu
    call.argtypes = [
        ct.c_uint,
        ct.POINTER(ct.c_double),
        ct.c_size_t,
        ct.POINTER(ct.c_double),
    ]
    call.restype = ct.c_int
    return library, call


class _PrimitiveExecutor:
    """Compile finite requested component coverage and stream fixed-size records.

    The common compiler cache verifies source, transitive headers and binary.
    The first integration deliberately admits s/p bases only to bound generated
    code size. Wider component/subset scheduling is separate qualification work.
    """

    def __init__(
        self,
        basis: typing.Any,
        cache: typing.Any,
        primitive_tile: typing.Any,
        compiler: typing.Any,
    ) -> None:
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
        source = emit_first_derivative_cpu(tuple(requests))
        self.library, self.call = _compile_primitive_library(source, cache, compiler)
        self.buffer = np.zeros((primitive_tile, 17))
        self.records = 0

    def _run(self, kind: typing.Any, count: typing.Any) -> typing.Any:
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

    def integral(
        self,
        operator: typing.Any,
        indices: typing.Any,
        weight: typing.Any,
        nucleus: typing.Any = None,
    ) -> typing.Any:
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

    def nuclear(self, a: typing.Any, b: typing.Any, charges: typing.Any) -> typing.Any:
        self.buffer.fill(0)
        self.buffer[0, :2] = charges[[a, b]]
        self.buffer[0, 4:10] = self.centers[[a, b]].reshape(-1)
        self.buffer[0, 16] = 1
        return self._run(self.kinds["nuclear", ()], 1)[:2]


def _admit_work(
    state: typing.Any,
    basis: typing.Any,
    execution: str,
    tile_points: int,
    max_primitive_records: int,
    max_grid_points: int,
    max_grid_pair_visits: int,
    max_ecp_pair_samples: int,
) -> dict[str, int]:
    """Metadata-only admission; no derivative compiler, provider or allocations.

    Counts describe semantic loops, not FLOPs or timing. CPU and CUDA now consume
    the same compiler-owned ECP grid policy; count both complete provider grids.
    """
    if any(shell.angular_momentum > 2 for shell in basis.shells):
        raise NotImplementedError(
            "complete CPU gradient diagnostic supports s/p/d bases only"
        )
    natom, n = basis.natom, basis.nao
    aos = basis.packed[3 * natom + 2 * basis.nprimitive :].reshape(-1, 16)
    if any(int(row[3]) not in (1, 2, 3) for row in aos):
        raise NotImplementedError(
            "this diagnostic supports at most three Cartesian components per AO"
        )
    primitive_sum = sum(int(row[2]) * int(row[3]) for row in aos)
    pairs = natom * (natom - 1) // 2
    method_ir = getattr(state._source, "method_ir", None)
    exact_exchange = bool(getattr(method_ir, "full_range_exact_exchange", 0))
    nonlocal_correlation = any(
        isinstance(primitive, NonlocalCorrelationPrimitive)
        for primitive in getattr(method_ir, "primitives", ())
    )
    range_exchange = sum(
        type(primitive) is RangeSeparatedExchangePrimitive
        for primitive in getattr(method_ir, "primitives", ())
    )
    if range_exchange and any(shell.angular_momentum > 1 for shell in basis.shells):
        raise NotImplementedError(
            "CPU RSH stationary gradients currently support s/p bases only"
        )
    # Coulomb always traverses every ordered primitive quartet. Each full- or
    # range-separated exact-exchange source is an independently weighted ERI
    # derivative traversal over that same ordered quartet domain.
    quartet_passes = 1 + int(exact_exchange) + range_exchange
    records = quartet_passes * primitive_sum**4 + (natom + 2) * primitive_sum**2 + pairs
    points = len(state.grid.points)
    visits = (2 if execution == "native" else 3 * natom) * pairs * points
    validations = ((points + tile_points - 1) // tile_points) * pairs
    # Native adjoint validates once per tile; the reference directional route
    # validates on every coordinate traversal. Include both in admission.
    if execution == "reference":
        validations *= 3 * natom
    nonlocal_pair_visits = points * points if nonlocal_correlation else 0
    nonlocal_partition_visits = visits + validations if nonlocal_correlation else 0
    grid_pair_work = (
        visits + validations + nonlocal_pair_visits + nonlocal_partition_visits
    )
    for actual, budget, label in (
        (records, max_primitive_records, "primitive"),
        (points, max_grid_points, "grid point"),
        (grid_pair_work, max_grid_pair_visits, "grid pair"),
    ):
        if actual > budget:
            raise ValueError(f"{label} work budget exceeded")
    ecp_samples = 0
    if state._source.hamiltonian == "scalar-semilocal-ecp":
        # Explicit dense-provider domain bounds radial term/AO preparation as
        # well as pair sampling. This does not claim a host memory budget.
        if (
            n > 16
            or natom > 8
            or basis.nprimitive > 128
            or len(state._source.ecp_terms) > 128
        ):
            raise ValueError("ECP diagnostic dense-export domain exceeded")
        ecp_samples = (
            sum(core > 0 for core in state._source.ecp_cores)
            * (n * (n + 1) // 2)
            * 2
            * (
                COARSE_RADIAL_POINTS * COARSE_POLAR_POINTS**2
                + REFINED_RADIAL_POINTS * REFINED_POLAR_POINTS**2
            )
        )
        if ecp_samples > max_ecp_pair_samples:
            raise ValueError("ECP quadrature pair-sample work budget exceeded")
    return {
        "ordered_pairs": n * n,
        "ordered_quartets": n**4,
        "primitive_record_bound": records,
        "primitive_record_budget": max_primitive_records,
        "xc_points": points,
        "grid_point_budget": max_grid_points,
        "grid_directional_points": 3 * natom * points
        if execution == "reference"
        else 0,
        "grid_adjoint_points": points if execution == "native" else 0,
        "grid_pair_visits": visits,
        "grid_center_pair_validations": validations,
        "nonlocal_pair_evaluations": nonlocal_pair_visits,
        "nonlocal_partition_pair_work": nonlocal_partition_visits,
        "grid_pair_work_bound": grid_pair_work,
        "grid_pair_work_budget": max_grid_pair_visits,
        "ecp_quadrature_pair_samples": ecp_samples,
        "ecp_pair_sample_budget": max_ecp_pair_samples,
    }


def complete_rks_gradient_diagnostic(
    state: typing.Any,
    basis: typing.Any,
    *,
    cache: typing.Any,
    tile_points: typing.Any = 256,
    integral_terms: typing.Any = 32,
    primitive_tile: typing.Any = 128,
    compiler: typing.Any = None,
    execution: typing.Any = "reference",
    component_execution: str = "native",
    max_primitive_records: int = 2_000_000,
    max_grid_points: int = 1_000_000,
    max_grid_pair_visits: int = 100_000_000,
    max_ecp_pair_samples: int = 200_000_000,
    max_host_bytes: int | None = None,
) -> typing.Any:
    """Consume one live native CPU RKS/UKS state with complete plan-owned sources.

    Admitted public-state domain is direct real FP64 integer RKS/UKS with
    validated LDA/PBE/r2SCAN/global-hybrid MethodIR. The consumer also executes
    range-separated exchange sources for validated stationary plans with s/p AOs;
    public RSH snapshot/capability admission remains a separate boundary. Uses a
    native unpruned version-one grid, distinct nuclei and no
    point/center collisions. CPU is explicit; CUDA snapshots are rejected.
    Caller chooses an ignored/temporary compilation cache and may supply a
    CppCompilerAdapter; otherwise CXX (or c++) selects the executable. Scientific work is
    full ordered AO pairs/quartets, without screening or symmetry shortcuts.
    Working arrays scale with a point tile times (AO + atom), one primitive
    record tile, D/W, and bounded per-source atom gradients, never coordinate-grid-AO pairs.
    The native state already retains its full discrete grid and dense SCF data.
    execution="native" selects compiled consumers of the same mathematical
    graphs. execution="reference" retains the validated interpreter route.
    s/p/d component enumeration uses a bounded native consumer; the private
    component_execution="python" selector retains the ordered baseline.
    Both retain Python AO enumeration/scatter and NumPy XC BLAS/maps;
    neither alone establishes an overall endpoint/SCF memory budget. Semantic work
    budgets reject before derivative compilation or provider execution, after
    the caller's SCF and snapshot export. ECP pair-samples are a conservative
    two-grid bound, including radial shells the provider may skip.
    Scalar-ECP CPU snapshots additionally bind effective ionic charges and two
    residual derivative sources to the actual energy owner. Their existing
    generated CPU provider materializes 2*3*natom*nao**2 derivative elements;
    this provider shares compiler-owned mathematics with CUDA. The public
    CPU wrapper explicitly selects it and reserves the extra numeric capacity;
    no PySCF callback is involved. max_host_bytes requires compiled execution
    and covers snapshot/export plus bounded numeric staging, excluding Python,
    compiler, loaded-code and opaque BLAS/runtime storage.
    """
    if execution not in ("reference", "native"):
        raise ValueError("execution must be reference or native")
    if component_execution not in ("native", "python"):
        raise ValueError("component_execution must be native or python")
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
        (max_primitive_records, "max_primitive_records", 1 << 40),
        (max_grid_points, "max_grid_points", 1 << 40),
        (max_grid_pair_visits, "max_grid_pair_visits", 1 << 40),
        (max_ecp_pair_samples, "max_ecp_pair_samples", 1 << 40),
    ):
        if type(value) is not int or not 1 <= value <= cap:
            raise ValueError(f"{name} must be an integer in [1,{cap}]")
    work = _admit_work(
        state,
        basis,
        execution,
        tile_points,
        max_primitive_records,
        max_grid_points,
        max_grid_pair_visits,
        max_ecp_pair_samples,
    )
    if max_host_bytes is not None:
        from ._cpu_force_resources import cpu_force_inventory

        if execution != "native":
            raise ValueError("CPU host budget requires the compiled native consumer")
        if type(max_host_bytes) is not int or not 1 <= max_host_bytes <= 1 << 40:
            raise ValueError("max_host_bytes must be an integer in [1,1099511627776]")
        inventory = cpu_force_inventory(
            basis,
            grid_points=len(state.grid.points),
            ecp_terms=len(state._source.ecp_terms),
            nonlocal_correlation=any(
                isinstance(primitive, NonlocalCorrelationPrimitive)
                for primitive in state._source.method_ir.primitives
            ),
            tile_points=tile_points,
            primitive_tile=primitive_tile,
            integral_terms=integral_terms,
            range_exchange_sources=sum(
                type(p) is RangeSeparatedExchangePrimitive
                for p in state._source.method_ir.primitives
            ),
        )
        host_bound = sum(inventory.values())
        if host_bound > max_host_bytes:
            raise ValueError("CPU force additional-host byte budget exceeded")
        work.update(
            additional_host_numeric_bound=host_bound,
            additional_host_budget=max_host_bytes,
        )
    # Consume the exact graph proven by the live snapshot. Re-resolving the
    # descriptive method alias here would discard custom/global-hybrid
    # coefficients and split energy/Fock semantics from the derivative.
    method = state._source.method_ir
    functional = state._source.functional
    plan = StationaryGradientPlan(
        method,
        StationaryMeanField(
            state._source._batch._calculator._ks_options.scf_domain
            if state.identity.method.startswith("wb97m-v")
            else SCF_POINT_MODEL,
            hamiltonian=state._source.hamiltonian,
        ),
    )
    density = state.density if contract.spin == "polarized" else state.density[0]
    if compiler is None:
        compiler = CppCompilerAdapter(Path(os.environ.get("CXX", "c++")))
    if not isinstance(compiler, CppCompilerAdapter):
        raise TypeError("the CPU diagnostic requires an explicit C++ compiler adapter")
    if any(shell.angular_momentum == 2 for shell in basis.shells):
        from ._stationary_cpu_components import ComponentPrimitiveExecutor
        from ._stationary_cpu_streaming import CompiledComponentExecutor

        executor = (
            CompiledComponentExecutor
            if component_execution == "native"
            else ComponentPrimitiveExecutor
        )
        native = executor(basis, cache, primitive_tile, compiler)
        work.update(native.compilation_work)
        work["component_execution"] = component_execution
    else:
        native = _PrimitiveExecutor(basis, cache, primitive_tile, compiler)
    natom, n = basis.natom, basis.nao
    components = {name: np.zeros((natom, 3)) for name in plan.source_names}
    charges = np.asarray([atom.atomic_number for atom in basis.atoms]) - np.asarray(
        state._source.ecp_cores
    )
    work.update(
        point_tile_capacity=tile_points,
        primitive_tile_capacity=primitive_tile,
        integral_term_capacity=integral_terms,
    )
    tensor_consumers = {}
    # TensorIR AD supplies D, Coulomb D*D/2, exact-exchange same-spin
    # D[a,c]*D[b,d]*cK/2, and -W. Runtime only binds tuple-indexed state;
    # it never rebuilds method coefficients from a named-functional formula.
    integral_sources = [
        ("one_electron", 2),
        ("overlap_pulay", 2),
        ("coulomb", 4),
    ]
    if plan.exchange is not None:
        integral_sources.append(("exact_exchange", 4))
    for source, rank in integral_sources:
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
            elif source == "exact_exchange":
                # For each ordered ERI (ab|cd), K contracts same-spin
                # D[a,c] D[b,d]. Cross-spin exchange is deliberately absent.
                feeds = {
                    "density_left": state.density[:, ids[:, 0], ids[:, 2]],
                    "density_right": state.density[:, ids[:, 1], ids[:, 3]],
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
                    "exact_exchange": "four_center_eri",
                }[source]
                owners, values = native.integral(operator, indices, weight)
                np.add.at(components[source], owners, values)
                if source == "one_electron":
                    for atom in range(natom):
                        owners, values = native.integral(
                            "nuclear_attraction", indices, weight * charges[atom], atom
                        )
                        np.add.at(components[source], owners, values)

    range_native = None
    if plan.range_exchange_primitives:
        range_native = RangeExchangeExecutor(basis, cache, primitive_tile, compiler)
        try:
            for range_source in plan.range_exchange_sources:
                primitive = plan.range_exchange_primitive(range_source.name)
                iterator = product(range(n), repeat=4)
                while tuples := tuple(islice(iterator, integral_terms)):
                    ids = np.asarray(tuples)
                    block = plan.integral_block(range_source.name, terms=len(tuples))
                    feeds = {
                        "density_left": state.density[:, ids[:, 0], ids[:, 2]],
                        "density_right": state.density[:, ids[:, 1], ids[:, 3]],
                    }
                    key = (range_source.name, len(tuples))
                    if key not in tensor_consumers:
                        tensor_consumers[key] = (
                            NativeTensorProgram(
                                block.weights, compiler=compiler, cache=cache
                            )
                            if execution == "native"
                            else block.weights
                        )
                    consumer = tensor_consumers[key]
                    weights = (
                        consumer.execute(feeds)["weights"]
                        if execution == "native"
                        else execute(consumer, feeds).outputs["weights"]
                    )
                    for indices, weight in zip(tuples, weights, strict=True):
                        owners, values = range_native.integral(
                            primitive, indices, weight
                        )
                        np.add.at(components[range_source.name], owners, values)
        finally:
            range_native.close()
    for a in range(natom):
        for b in range(a):
            np.add.at(components["nuclear"], [a, b], native.nuclear(a, b, charges))

    if state._source.hamiltonian == "scalar-semilocal-ecp":
        derivatives = state._source.ecp_derivatives()
        work["ecp_derivative_bytes"] = derivatives.nbytes
        # The generated CPU ECP provider includes both AO-center
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
        else ExternalPointContraction(functional, "geometry")
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
            functional,
            features["rho"],
            features.get("gradient", np.zeros((2, end - begin, 3))),
            features.get("tau"),
        )
        partials = program.geometry_from_cartesian_coefficients(
            jets,
            density,
            weights,
            coefficients["energy"],
            coefficients["rho"],
            coefficients["gradient"] if contract.family != "lda" else None,
            coefficients["kinetic"] if contract.family == "mgga" else None,
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
    nonlocal_primitive = next(
        (
            primitive
            for primitive in method.primitives
            if isinstance(primitive, NonlocalCorrelationPrimitive)
        ),
        None,
    )
    if nonlocal_primitive is not None:
        calculator = state._source._batch._calculator
        provider = NativeNonlocalPairProvider(
            device="cpu",
            device_id=0,
            memory_budget_bytes=calculator._ks_options.nonlocal_memory_budget_bytes,
            library=state._source._library,
        )
        geometry = FixedDensityNonlocalCorrelation(
            nonlocal_primitive.spec,
            coefficient=nonlocal_primitive.coefficient,
            pair_provider=provider,
            density_policy=state._source.nonlocal_density_policy,
        ).geometry(basis, grid, density, tile_points=tile_points)
        components["nonlocal_ao"] += np.asarray(geometry.centers)
        owners = np.asarray(grid.owners, dtype=np.int64)
        np.add.at(components["nonlocal_grid"], owners, np.asarray(geometry.points))
        if grid_consumer is not None:
            with np.errstate(over="raise", invalid="raise"):
                seeds = np.asarray(geometry.weights) * np.asarray(
                    state._source.atomic_weights
                )
            components["nonlocal_weight"] += grid_consumer.contract(
                grid.points,
                native.centers,
                owners,
                seeds,
                coincident_tolerance=spec.coincident_tolerance,
            )
        else:
            for a in range(natom):
                for axis in range(3):
                    motion = np.zeros((natom, 3))
                    motion[a, axis] = 1
                    for begin in range(0, len(grid.points), tile_points):
                        end = min(begin + tile_points, len(grid.points))
                        atoms = owners[begin:end]
                        response = partition_response(
                            grid.points[begin:end],
                            native.centers,
                            point_motion=motion[atoms],
                            center_motion=motion,
                            iterations=spec.partition_iterations,
                            coincident_tolerance=spec.coincident_tolerance,
                        )
                        selected = (np.arange(end - begin), atoms)
                        components["nonlocal_weight"][a, axis] += np.dot(
                            geometry.weights[begin:end],
                            state._source.atomic_weights[begin:end]
                            * response.directional[selected],
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
    primitive_records = native.records + (
        0 if range_native is None else range_native.records
    )
    if primitive_records != work["primitive_record_bound"]:
        raise RuntimeError("CPU derivative primitive work differs from admission")
    work["primitive_records"] = primitive_records
    if range_native is not None:
        work["range_exchange_primitive_records"] = range_native.records
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
