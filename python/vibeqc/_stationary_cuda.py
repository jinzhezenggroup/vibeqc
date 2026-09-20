"""Complete CUDA RKS/UKS gradient diagnostic with explicit host export.

This bounded consumer also supplies qualified public Calculator CUDA forces. Native
CUDA SCF exports its verified D/W frame to the host. Python enumerates primitive
records and gathers TensorIR inputs; all derivative/normalization/contraction,
AO/features/XC work, atom scatter and final source reduction execute on CUDA.
No CPU derivative or interpreter fallback is available.
"""

from __future__ import annotations

import ctypes as ct
import typing
from contextlib import ExitStack
from itertools import islice, product
from pathlib import Path
from time import perf_counter
from types import MappingProxyType

import numpy as np
from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.provenance import file_hash
from vibeqc_compiler.dft.cuda import (
    CudaGrid,
    GridTaskView,
)
from vibeqc_compiler.dft.cuda import (
    compile_cuda as compile_grid,
)
from vibeqc_compiler.dft.plan import plan_tiles
from vibeqc_compiler.integral.first_derivative_native import emit_first_derivative_cuda
from vibeqc_compiler.method.stationary_cuda import compile_stationary_cuda
from vibeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)
from vibeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
from vibeqc_compiler.tensor.cuda_plan import plan_cuda

from ._dft_gradient import (
    StationaryDerivativeContract,
    _native_ao_atoms,
    native_ao_geometry_identity,
)
from ._stationary_cpu import DiagnosticStationaryGradient
from .ks import resolve_ks_method

_DOUBLE = ct.POINTER(ct.c_double)
_INT = ct.POINTER(ct.c_int64)
_SOURCE_NAMES = (
    "one_electron",
    "coulomb",
    "xc_ao",
    "xc_grid",
    "xc_weight",
    "overlap_pulay",
    "nuclear",
)


def _ptr(array: typing.Any) -> typing.Any:
    return array.ctypes.data_as(_INT if array.dtype == np.int64 else _DOUBLE)


def _checked(
    array: typing.Any, shape: typing.Any, dtype: typing.Any = np.float64
) -> typing.Any:
    """Reject lossy/coercive admission before ctypes or device access."""
    value = np.asarray(array)
    if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
        raise ValueError(f"CUDA source requires finite {dtype} with shape {shape}")
    return np.ascontiguousarray(value)


def _layout(basis: typing.Any) -> typing.Any:
    """Read the native normalized basis records without evaluating integrals."""
    if any(s.angular_momentum > 1 for s in basis.shells):
        raise NotImplementedError("CUDA gradient diagnostic admits s/p bases only")
    start = 3 * basis.natom
    primitives = basis.packed[start : start + 2 * basis.nprimitive].reshape(-1, 2)
    aos = basis.packed[start + 2 * basis.nprimitive :].reshape(-1, 16)
    if any(int(r[3]) != 1 for r in aos):
        raise NotImplementedError("CUDA diagnostic requires single-component AOs")
    components = tuple("".join(a * int(l) for a, l in zip("xyz", r[4:7])) for r in aos)
    domain = sorted(set(components))
    requests = tuple(
        [
            (op, c)
            for op in ("overlap", "kinetic", "nuclear_attraction")
            for c in product(domain, repeat=2)
        ]
        + [("four_center_eri", c) for c in product(domain, repeat=4)]
        + [("nuclear", ())]
    )
    return primitives, aos, components, requests


class _CudaSources:
    """Serialized finite owner; device accumulators publish only after success."""

    def __init__(
        self,
        basis: typing.Any,
        artifact: typing.Any,
        compiler: typing.Any,
        device: typing.Any,
        points: typing.Any,
        records: typing.Any,
        budget: typing.Any,
    ) -> None:
        if file_hash(artifact.library) != artifact.metadata["binary_sha256"]:
            raise ValueError("stationary CUDA binary hash mismatch")
        self.artifact = artifact
        self.handle = ct.c_void_p()
        self.library = lib = ct.CDLL(str(artifact.library))
        self.natom, self.nao, self.point_capacity = basis.natom, basis.nao, points
        self.buffer = np.ones((records, 26))
        self.maps = np.full((records, 4), -1, dtype=np.int64)
        self.used = 0
        self.pending = None
        self.device = device
        self.borrowed_streams = set()
        self.centers = np.ascontiguousarray(
            basis.packed[: 3 * basis.natom].reshape(-1, 3)
        )
        self.ao_atoms = np.ascontiguousarray(_native_ao_atoms(basis), dtype=np.int64)
        self.primitives, self.aos, self.components, requests = _layout(basis)
        self.kinds = {key: i for i, key in enumerate(requests)}
        tail = [ct.c_char_p, ct.c_size_t]
        lib.stationary_create.argtypes = (
            [ct.c_int] * 3 + [ct.c_size_t] * 5 + [ct.POINTER(ct.c_void_p), *tail]
        )
        lib.stationary_reset.argtypes = [ct.c_void_p, _DOUBLE, _INT, ct.c_double, *tail]
        lib.stationary_records.argtypes = [
            ct.c_void_p,
            ct.c_uint,
            ct.c_uint,
            _DOUBLE,
            _INT,
            ct.c_size_t,
            *tail,
        ]
        lib.stationary_geometry.argtypes = [
            ct.c_void_p,
            ct.POINTER(GridTaskView),
            _DOUBLE,
            _INT,
            _DOUBLE,
            _DOUBLE,
            *tail,
        ]
        lib.stationary_finish.argtypes = [ct.c_void_p, _DOUBLE, ct.c_size_t, *tail]
        lib.stationary_metrics.argtypes = [
            ct.c_void_p,
            ct.POINTER(ct.c_uint64),
            ct.c_size_t,
        ]
        lib.stationary_destroy.argtypes = [ct.c_void_p]
        lib.stationary_destroy.restype = None
        self._call(
            "stationary_create",
            device,
            *compiler.target.compute_capability,
            basis.natom,
            basis.nao,
            points,
            records,
            budget,
            ct.byref(self.handle),
        )

    def _call(self, name: typing.Any, *args: typing.Any) -> None:
        error = ct.create_string_buffer(2048)
        if getattr(self.library, name)(*args, error, len(error)):
            raise RuntimeError(error.value.decode())

    def reset(self, tolerance: typing.Any) -> None:
        self.used, self.pending = 0, None
        self._call(
            "stationary_reset",
            self.handle,
            _ptr(self.centers),
            _ptr(self.ao_atoms),
            tolerance,
        )

    def flush(self) -> None:
        if self.used:
            self._call(
                "stationary_records",
                self.handle,
                *self.pending,
                _ptr(self.buffer),
                _ptr(self.maps),
                self.used,
            )
            self.used = 0

    def integral(
        self,
        source: typing.Any,
        operator: typing.Any,
        indices: typing.Any,
        weight: typing.Any,
        nucleus: typing.Any = None,
        charge: typing.Any = 1.0,
    ) -> None:
        """Pack exponents, raw normalization factors and plan weights separately.

        Host work is discrete record enumeration. CUDA multiplies every
        normalization factor and performs the weighted derivative/scatter.
        """
        key = self.kinds[operator, tuple(self.components[i] for i in indices)], source
        if key != self.pending:
            self.flush()
            self.pending = key
        rows = self.aos[list(indices)]
        owners = [int(r[0]) for r in rows]
        if nucleus is not None:
            owners.append(nucleus)
        ranges = [range(int(r[1]), int(r[1] + r[2])) for r in rows]
        for ids in product(*ranges):
            r, m = self.buffer[self.used], self.maps[self.used]
            r.fill(1)
            m.fill(-1)
            primitives = self.primitives[list(ids)]
            r[: len(ids)] = primitives[:, 0]
            r[4 : 4 + 3 * len(owners)] = self.centers[owners].reshape(-1)
            r[16 : 16 + len(ids)] = primitives[:, 1]
            r[20 : 20 + len(ids)] = rows[:, 7]
            r[24], r[25] = weight, charge
            m[: len(owners)] = owners
            self.used += 1
            if self.used == len(self.buffer):
                self.flush()

    def nuclear(self, a: typing.Any, b: typing.Any, charges: typing.Any) -> None:
        self.flush()
        self.pending = self.kinds["nuclear", ()], 6
        r, m = self.buffer[0], self.maps[0]
        r.fill(1)
        m.fill(-1)
        r[:2] = charges[[a, b]]
        r[4:10] = self.centers[[a, b]].reshape(-1)
        m[:2] = a, b
        self.used = 1

    def geometry(
        self,
        task: typing.Any,
        owners: typing.Any,
        weights: typing.Any,
        raw: typing.Any,
        *,
        pbe: typing.Any,
    ) -> None:
        view = task.view
        if task._owner.device_id != self.device:
            raise ValueError("stationary/grid current owner device mismatch")
        self.borrowed_streams.add(view.stream)
        owners = _checked(owners, (view.npoint,), np.int64)
        weights = _checked(weights, (view.npoint,))
        raw = _checked(raw, (view.npoint,))
        work = task.density_jets(4 if pbe else 1)
        self._call(
            "stationary_geometry",
            self.handle,
            ct.byref(view),
            work,
            _ptr(owners),
            _ptr(weights),
            _ptr(raw),
        )

    def finish(self) -> typing.Any:
        self.flush()
        out = np.empty((7, self.natom, 3))
        self._call("stationary_finish", self.handle, _ptr(out), out.size)
        return {name: out[i] for i, name in enumerate(_SOURCE_NAMES)}

    def metrics(self) -> typing.Any:
        values = (ct.c_uint64 * 8)()
        if self.library.stationary_metrics(self.handle, values, 8):
            raise RuntimeError("stationary metrics unavailable")
        return dict(
            zip(
                (
                    "owned_device_bytes",
                    "h2d_bytes",
                    "d2h_bytes",
                    "launches",
                    "primitive_records",
                    "xc_points",
                    "grid_pair_visits",
                    "stream",
                ),
                values,
            )
        )

    def close(self) -> None:
        if self.handle:
            self.library.stationary_destroy(self.handle)
            self.handle = ct.c_void_p()

    def __enter__(self) -> typing.Any:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "handle"):
            self.close()


def complete_rks_cuda_gradient_diagnostic(
    state: typing.Any,
    basis: typing.Any,
    *,
    compiler: typing.Any,
    cache: typing.Any,
    tile_points: typing.Any = 256,
    integral_terms: typing.Any = 32,
    primitive_tile: typing.Any = 128,
    max_device_bytes: typing.Any = 512 << 20,
    max_host_bytes: typing.Any = 256 << 20,
    max_grid_points: typing.Any = 1_000_000,
    max_primitive_records: typing.Any = 2_000_000,
    max_grid_pair_visits: typing.Any = 100_000_000,
    max_ecp_pair_samples: int = 100_000_000,
) -> typing.Any:
    """Consume a current native CUDA RKS/UKS snapshot with every plan source.

    Domain: real FP64 direct all-electron s/p LDA/PBE RKS/UKS, native version-three
    unpruned grid and distinct noncolliding centers. No CPKS is required.
    Device ordinal comes only from the opaque native snapshot. CUDA source
    accumulators, grid owner and one TensorIR consumer coexist under the stated
    additional-device budget; the pre-existing SCF owner/export and Python
    objects are reported separately. No claim of full device residency is made.
    Compilation and the selected GPU allocation are explicit.
    Scalar-ECP v5 adds generated CUDA local/nonlocal derivatives and effective
    charges (nine sources). Its small dense export is separately budgeted and
    preserves the checked native two-grid gate. The public wrapper restricts ECP
    force capability to Cartesian/real-spherical s/p records.
    """
    started = perf_counter()
    contract = StationaryDerivativeContract(state.identity)
    contract.validate(state)
    if state._source.backend != "cuda":
        raise NotImplementedError("CUDA diagnostic requires a native CUDA KS state")
    if state._source.metadata[0] not in (3, 5) or state._source.grid_spec is None:
        raise NotImplementedError("CUDA diagnostic requires snapshot v3 raw measures")
    if (
        basis.identity != state.identity.basis_identity
        or native_ao_geometry_identity(basis) != state.identity.geometry_identity
    ):
        raise ValueError("stationary CUDA basis/geometry mismatch")
    ecp = state._source.hamiltonian == "scalar-semilocal-ecp"
    # Legacy v3 has no Hamiltonian records. Core-adjusted v3 remains rejected.
    if (
        float(np.sum(state.occupations))
        != sum(a.atomic_number for a in basis.atoms)
        - sum(state._source.ecp_cores)
        - basis.charge
    ):
        raise NotImplementedError("CUDA gradient diagnostic requires bound ECP states")
    if not isinstance(compiler, CudaCompilerAdapter):
        raise TypeError("an explicit CUDA compiler adapter is required")
    for value, name, cap in (
        (tile_points, "tile_points", 4096),
        (primitive_tile, "primitive_tile", 4096),
        (integral_terms, "integral_terms", 128),
        (max_device_bytes, "max_device_bytes", 1 << 40),
        (max_host_bytes, "max_host_bytes", 1 << 40),
        (max_grid_points, "max_grid_points", 1 << 40),
        (max_primitive_records, "max_primitive_records", 1 << 40),
        (max_grid_pair_visits, "max_grid_pair_visits", 1 << 40),
        (max_ecp_pair_samples, "max_ecp_pair_samples", 1 << 40),
    ):
        if type(value) is not int or not 1 <= value <= cap:
            raise ValueError(f"{name} must be an integer in [1,{cap}]")
    na, n = basis.natom, basis.nao
    if not 1 <= na <= 32 or not 1 <= n <= 128:
        raise ValueError("CUDA diagnostic small-domain atom/AO cap exceeded")
    _, aos, _, requests = _layout(basis)
    primitive_sum = sum(int(r[2]) for r in aos)
    records = primitive_sum**4 + (na + 2) * primitive_sum**2 + na * (na - 1) // 2
    pair_visits = (1 + 2 * len(state.grid.points)) * na * (na - 1) // 2
    if records > max_primitive_records:
        raise ValueError("primitive work budget exceeded")
    if len(state.grid.points) > max_grid_points:
        raise ValueError("grid point work budget exceeded")
    if pair_visits > max_grid_pair_visits:
        raise ValueError("grid work budget exceeded")
    method, _ = resolve_ks_method(state.identity.method)
    plan = StationaryGradientPlan(
        method,
        StationaryMeanField(
            SCF_POINT_MODEL,
            hamiltonian="scalar-semilocal-ecp" if ecp else "all-electron",
        ),
    )
    density = state.density if contract.spin == "polarized" else state.density[0]
    if (
        tuple(s for s in plan.source_names if s not in ("ecp_local", "ecp_nonlocal"))
        != _SOURCE_NAMES
    ):
        raise ValueError(
            "CUDA runtime source coverage differs from StationaryGradientPlan"
        )
    pbe = contract.family == "gga"
    device = int(state._source.metadata[12])
    grid_plan = plan_tiles(
        basis,
        backend="cuda",
        order=2 if pbe else 1,
        tile_points=tile_points,
        active_ao_capacity=n,
        budget_bytes=max_device_bytes,
    )
    source_bytes = (
        8 * (42 * primitive_tile + 24 * na + 3 * tile_points + 576 * na + n) + 256
    )
    available = max_device_bytes - grid_plan.peak_bytes - source_bytes
    if available <= 0:
        raise ValueError("stationary additional-device budget exceeded")
    tensor_plans = {
        name: plan_cuda(
            plan.integral_block(name, terms=integral_terms).weights,
            compiler.target,
            max_bytes=available,
        )
        for name in ("one_electron", "overlap_pulay", "coulomb")
    }
    tensor_plans["reduction"] = plan_cuda(
        plan.reduction_program(atoms=na), compiler.target, max_bytes=available
    )
    # Conservative numeric-array bound: record/maps, adapter staging, D spin
    # conversion, gathered feeds, candidate/publication copies, and tile owners.
    # Compiler objects, Python headers and the caller's existing SCF snapshot
    # are explicit exclusions, as in the reused grid/TensorIR resource contracts.
    host_bound = (
        grid_plan.host_bytes
        + 8
        * (
            30 * primitive_tile
            + 4 * plan.spin_blocks * n * n
            + 120 * na
            + 26 * integral_terms
            + 3 * tile_points
            + n
            + 80
        )
        + max(tp.host_bytes for tp in tensor_plans.values())
    )
    if host_bound > max_host_bytes:
        raise ValueError("stationary additional-host byte budget exceeded")
    ecp_workspace = ecp_pair_samples = 0
    if ecp:
        from vibeqc_compiler.integral.ecp_policy import (
            COARSE_POLAR_POINTS,
            COARSE_RADIAL_POINTS,
            REFINED_POLAR_POINTS,
            REFINED_RADIAL_POINTS,
        )

        from .resources_hf import _ecp_workspace

        # Dense export/contraction is a deliberately small diagnostic domain.
        if (
            n > 16
            or na > 8
            or basis.nprimitive > 128
            or len(state._source.ecp_terms) > 128
        ):
            raise ValueError("ECP diagnostic dense-export domain exceeded")
        ecp_pair_samples = (
            sum(core > 0 for core in state._source.ecp_cores)
            * (n * (n + 1) // 2)
            * 2
            * (
                COARSE_RADIAL_POINTS * COARSE_POLAR_POINTS**2
                + REFINED_RADIAL_POINTS * REFINED_POLAR_POINTS**2
            )
        )
        if ecp_pair_samples > max_ecp_pair_samples:
            raise ValueError("ECP quadrature pair-sample work budget exceeded")
        ecp_workspace = _ecp_workspace(
            {
                "atoms": na,
                "orbital": {
                    "nbf": n,
                    "cartesian_nbf": n,
                    "primitives": basis.nprimitive,
                    "ecp_terms": len(state._source.ecp_terms),
                },
            },
            cuda=True,
        )
        for name in ("ecp_local", "ecp_nonlocal"):
            tensor_plans[name] = plan_cuda(
                plan.integral_block(name, terms=n * n, coordinates=3 * na).contraction,
                compiler.target,
                max_bytes=available,
            )
        # Provider export occurs before the grid/source/TensorIR owners exist.
        # Account both native snapshots, dense derivatives and immutable copies;
        # include the existing conservative two-grid provider workspace.
        host_bound += ecp_workspace + 4 * state._source.values.nbytes + 144 * na * n * n
        host_bound += max(t.host_bytes for t in tensor_plans.values())
        if host_bound > max_host_bytes:
            raise ValueError("ECP additional-host byte budget exceeded")
        if ecp_workspace > max_device_bytes:
            raise ValueError("ECP additional-device budget exceeded")
    cache = Path(cache)
    spec = state._source.grid_spec
    artifact = compile_stationary_cuda(
        emit_first_derivative_cuda(requests),
        pbe=pbe,
        iterations=spec.partition_iterations,
        compiler=compiler,
        cache=cache,
    )
    grid_artifact = compile_grid(compiler, cache)
    artifacts = [artifact, grid_artifact]
    tensor_work = {
        "executions": 0,
        "h2d_numeric_bytes": 0,
        "d2h_bytes": 0,
        "owned_device_peak_bytes": 0,
        "endpoint_ms": 0.0,
        "device_ms": 0.0,
    }

    def record_tensor(result: typing.Any, feeds: typing.Any) -> None:
        # Aggregate in constant storage; retaining one metrics dictionary per
        # AO quartet block would defeat the bounded diagnostic orchestration.
        tensor_work["executions"] += 1
        tensor_work["h2d_numeric_bytes"] += sum(a.nbytes for a in feeds.values())
        tensor_work["d2h_bytes"] += sum(a.nbytes for a in result.outputs.values()) + 4
        tensor_work["owned_device_peak_bytes"] = max(
            tensor_work["owned_device_peak_bytes"], result.metrics["owned_device_bytes"]
        )
        for name in ("endpoint_ms", "device_ms"):
            tensor_work[name] += result.metrics[name]

    # Run the checked CUDA provider only after all admission checks pass.
    derivatives = state._source.ecp_derivatives() if ecp else None
    peak = max(ecp_workspace, grid_plan.peak_bytes + source_bytes)
    charges = np.asarray([a.atomic_number for a in basis.atoms]) - np.asarray(
        state._source.ecp_cores
    )
    with ExitStack() as stack:
        sources = stack.enter_context(
            _CudaSources(
                basis,
                artifact,
                compiler,
                device,
                tile_points,
                primitive_tile,
                source_bytes,
            )
        )
        sources.reset(spec.coincident_tolerance)
        ao = stack.enter_context(
            CudaGrid(
                basis,
                grid_artifact,
                order=2 if pbe else 1,
                tile_points=tile_points,
                budget_bytes=grid_plan.peak_bytes,
                device_id=device,
                active_ao_capacity=n,
                # tau requests all four D*jet panels for the PBE AO pullback.
                ingredients=("rho", "gradient", "tau") if pbe else ("rho",),
            )
        )
        ao.set_density(density)
        for source, rank, operator in (
            ("one_electron", 2, "kinetic"),
            ("overlap_pulay", 2, "overlap"),
            ("coulomb", 4, "four_center_eri"),
        ):
            tp = tensor_plans[source]
            ta = compile_cuda(tp, compiler, cache)
            artifacts.append(ta)
            peak = max(peak, grid_plan.peak_bytes + source_bytes + tp.peak_bytes)
            with PreparedCuda(tp, ta, device=device) as weights:
                iterator = product(range(n), repeat=rank)
                while tuples := tuple(islice(iterator, integral_terms)):
                    ids = np.zeros((integral_terms, rank), dtype=np.int64)
                    ids[: len(tuples)] = tuples
                    if source == "overlap_pulay":
                        feeds = {
                            "weighted_density": state.weighted_density[
                                :, ids[:, 0], ids[:, 1]
                            ]
                        }
                    else:
                        feeds = {"density_left": state.density[:, ids[:, 0], ids[:, 1]]}
                        if rank == 4:
                            feeds["density_right"] = state.density[
                                :, ids[:, 2], ids[:, 3]
                            ]
                    result = weights.execute(feeds)
                    record_tensor(result, feeds)
                    for indices, weight in zip(tuples, result.outputs["weights"]):
                        sources.integral(
                            _SOURCE_NAMES.index(source), operator, indices, weight
                        )
                        if source == "one_electron":
                            for a in range(na):
                                sources.integral(
                                    0,
                                    "nuclear_attraction",
                                    indices,
                                    weight,
                                    a,
                                    charges[a],
                                )
            sources.flush()
        for a in range(na):
            for b in range(a):
                sources.nuclear(a, b, charges)
        sources.flush()
        grid = state.grid
        for begin in range(0, len(grid.points), tile_points):
            end = min(begin + tile_points, len(grid.points))
            with ao.xc_task(
                grid.points[begin:end],
                np.arange(n, dtype=np.uintp),
                "PBE" if pbe else "LDA_XC_PW",
            ) as task:
                sources.geometry(
                    task,
                    np.asarray(grid.owners[begin:end], dtype=np.int64),
                    grid.weights[begin:end],
                    state._source.atomic_weights[begin:end],
                    pbe=pbe,
                )
        components = sources.finish()
        if ecp:
            # Full ordered AO-pair contraction; the existing TensorIR supplies
            # spin summation and every scientific weight/reduction on CUDA.
            for k, name in enumerate(("ecp_local", "ecp_nonlocal")):
                tp = tensor_plans[name]
                ta = compile_cuda(tp, compiler, cache)
                artifacts.append(ta)
                peak = max(peak, grid_plan.peak_bytes + source_bytes + tp.peak_bytes)
                feeds = {
                    "density_left": np.ascontiguousarray(
                        state.density.reshape(plan.spin_blocks, n * n)
                    ),
                    "integral_derivatives": np.ascontiguousarray(
                        derivatives[k].reshape(3 * na, n * n).T
                    ),
                }
                with PreparedCuda(tp, ta, device=device) as contraction:
                    result = contraction.execute(feeds)
                    record_tensor(result, feeds)
                    components[name] = result.outputs["gradient"].reshape(na, 3)
        # Validate actual coverage before the pre-admitted complete reduction.
        plan.reduction_program(atoms=na, sources=components)
        tp = tensor_plans["reduction"]
        ta = compile_cuda(tp, compiler, cache)
        artifacts.append(ta)
        peak = max(peak, grid_plan.peak_bytes + source_bytes + tp.peak_bytes)
        with PreparedCuda(tp, ta, device=device) as reduction:
            reduced = reduction.execute(components)
            record_tensor(reduced, components)
            gradient = reduced.outputs["gradient"]
        work = sources.metrics()
        work["grid_metrics"] = ao.metrics()
        work["borrowed_grid_streams"] = tuple(sorted(sources.borrowed_streams))
        if work["owned_device_bytes"] != source_bytes:
            raise RuntimeError("stationary allocation disagrees with admitted bytes")
    contract.validate(state)  # Replay/replacement/closure revokes publication.
    if work["primitive_records"] != records or work["grid_pair_visits"] != pair_visits:
        raise RuntimeError("CUDA executed work disagrees with admitted source coverage")
    work.update(
        ecp_provider="generated-cuda/two-grid/dense-host-export" if ecp else None,
        ecp_provider_workspace_bound=ecp_workspace,
        ecp_state_export=(
            "one additional live final-state read/validation before CUDA ECP export"
            if ecp
            else None
        ),
        ecp_derivative_export_bytes=derivatives.nbytes if ecp else 0,
        ecp_ordered_pairs=2 * n * n if ecp else 0,
        ecp_quadrature_pair_samples=ecp_pair_samples,
        ecp_pair_sample_budget=max_ecp_pair_samples,
        ordered_pairs=n * n,
        ordered_quartets=n**4,
        additional_device_peak_bound=peak,
        additional_device_budget=max_device_bytes,
        device_ordinal=device,
        tensor_executions=tensor_work["executions"],
        tensor_work=tensor_work,
        additional_host_numeric_bound=host_bound,
        additional_host_budget=max_host_bytes,
        snapshot_host_bytes=state._source.values.nbytes,
        snapshot_export_work=dict(state._source.export_work),
        snapshot_export="explicit native CUDA final-state export; W/frame validation is host work",
        host_scope="snapshot validation; primitive enumeration and record packing; density gathers; TensorIR H2D/D2H; immutable result copies",
        endpoint_seconds=perf_counter() - started,
        artifacts=tuple(
            {
                "library": str(a.library),
                "binary_sha256": a.metadata["binary_sha256"],
                "key": a.metadata["key"],
            }
            for a in artifacts
        ),
    )
    return DiagnosticStationaryGradient(
        immutable(gradient),
        MappingProxyType({k: immutable(v) for k, v in components.items()}),
        plan.identity,
        state.identity,
        MappingProxyType(work),
        execution=("cuda-nine-source" if ecp else "cuda-seven-source")
        + "/explicit-host-snapshot-and-orchestration-v1",
    )
