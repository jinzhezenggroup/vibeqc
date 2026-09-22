"""Complete CUDA RKS/UKS gradient diagnostic with explicit host export.

This bounded consumer also supplies qualified public Calculator CUDA forces. Native
CUDA SCF exports its verified D/W frame to the host. Python submits compact AO-tuple
tasks; immutable basis topology and D/W are resident while primitive enumeration,
generated TensorIR source-weight evaluation, derivative contraction, AO/features/XC
work, atom scatter and final source reduction execute on CUDA. No CPU derivative or
interpreter fallback is available.
"""

from __future__ import annotations

import ctypes as ct
import threading
import typing
from contextlib import ExitStack, contextmanager, nullcontext
from hashlib import sha256
from itertools import product
from pathlib import Path
from time import perf_counter
from types import MappingProxyType

import numpy as np
from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_runtime import CudaArtifact
from vibeqc_compiler.common.cuda_target import CudaTargetInfo
from vibeqc_compiler.common.prepared_execution import (
    PreparedArtifactBinding,
    PreparedExecutionLease,
    PreparedExecutionMismatch,
    PreparedExecutionRequest,
)
from vibeqc_compiler.common.provenance import canonical_hash, file_hash
from vibeqc_compiler.common.runtime_domain import RuntimeTaskDomain
from vibeqc_compiler.dft.cuda import (
    CudaGrid,
    GridTaskView,
)
from vibeqc_compiler.dft.cuda import (
    compile_cuda as compile_grid,
)
from vibeqc_compiler.dft.plan import plan_tiles
from vibeqc_compiler.integral.first_derivative_native import emit_first_derivative_cuda
from vibeqc_compiler.integral.first_derivative_schedule import (
    derivative_dispatch_id,
    derivative_dispatch_entries,
    derivative_requests,
)
from vibeqc_compiler.method.stationary_cuda import (
    STATIONARY_RUNTIME_SOURCE_NAMES,
    compile_stationary_cuda,
    load_stationary_aot_artifact,
    qualified_sp_requests,
)
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
_SOURCE_NAMES = STATIONARY_RUNTIME_SOURCE_NAMES


class _ExclusiveWallTimeline:
    """Additive host-wall phases that do not introduce CUDA synchronization."""

    def __init__(self, clock: typing.Callable[[], float] | None = None) -> None:
        self._clock = perf_counter if clock is None else clock
        self._started = self._last = self._clock()
        self._active = "preparation"
        self._seconds: dict[str, float] = {}
        self._closed = False

    def _charge(self, now: float) -> None:
        elapsed = now - self._last
        if elapsed < 0:
            raise RuntimeError("stationary timeline clock moved backwards")
        self._seconds[self._active] = self._seconds.get(self._active, 0.0) + elapsed
        self._last = now

    def switch(self, name: str) -> None:
        if self._closed or not name:
            raise RuntimeError("invalid stationary timeline phase transition")
        now = self._clock()
        self._charge(now)
        self._active = name

    @contextmanager
    def phase(self, name: str) -> typing.Iterator[None]:
        previous = self._active
        self.switch(name)
        try:
            yield
        finally:
            self.switch(previous)

    def finish(self) -> dict[str, typing.Any]:
        if self._closed:
            raise RuntimeError("stationary timeline is already closed")
        now = self._clock()
        self._charge(now)
        self._closed = True
        endpoint = now - self._started
        reconciled = sum(self._seconds.values())
        return {
            "schema": "vibeqc.stationary-cuda-exclusive-wall.v1",
            "clock": "time.perf_counter",
            "exclusive_wall_seconds": dict(sorted(self._seconds.items())),
            "reconciled_seconds": reconciled,
            "endpoint_seconds": endpoint,
            "reconciliation_error_seconds": endpoint - reconciled,
            "measurement_policy": (
                "exclusive host-wall phases; CUDA transfer/kernel attribution is reported "
                "separately and is not added to wall time"
            ),
        }


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


def _basis_topology_identity(basis: typing.Any) -> str:
    """Hash immutable AO topology while deliberately excluding Cartesian centers."""
    digest = sha256()
    digest.update(
        repr(
            (
                basis.natom,
                basis.nprimitive,
                basis.nao,
                basis.representation,
                basis.charge,
                basis.multiplicity,
                tuple(atom.atomic_number for atom in basis.atoms),
            )
        ).encode()
    )
    digest.update(np.ascontiguousarray(basis.packed[3 * basis.natom :]).tobytes())
    return digest.hexdigest()


def _layout(basis: typing.Any) -> typing.Any:
    """Read the native normalized basis records without evaluating integrals."""
    if any(s.angular_momentum > 2 for s in basis.shells):
        raise NotImplementedError("CUDA gradient diagnostic admits s/p/d bases only")
    start = 3 * basis.natom
    primitives = basis.packed[start : start + 2 * basis.nprimitive].reshape(-1, 2)
    aos = basis.packed[start + 2 * basis.nprimitive :].reshape(-1, 16)
    if any(int(r[3]) not in (1, 2, 3) for r in aos):
        raise NotImplementedError("CUDA diagnostic requires one to three Cartesian AO components")
    expansions = tuple(
        tuple(
            ("".join(axis * int(power) for axis, power in zip("xyz", row[4 + 4 * t : 7 + 4 * t])), float(row[7 + 4 * t]))
            for t in range(int(row[3]))
        )
        for row in aos
    )
    domain = tuple(sorted({label for expansion in expansions for label, _ in expansion}))
    extended = any(int(r[3]) > 1 for r in aos)
    requests = derivative_requests(domain) if extended else qualified_sp_requests()
    return primitives, aos, expansions, requests, extended, (domain if extended else None)


class _CudaSources:
    """Serialized finite owner; bounded AO tasks expand primitives only on CUDA."""

    def __init__(
        self,
        basis: typing.Any,
        artifact: typing.Any,
        compiler: typing.Any,
        device: typing.Any,
        points: typing.Any,
        records: typing.Any,
        budget: typing.Any,
        spin_blocks: typing.Any = 1,
        target: typing.Any = None,
        work_budget: typing.Any = 2_000_000,
        timeline: _ExclusiveWallTimeline | None = None,
        profile_device: bool = False,
    ) -> None:
        if file_hash(artifact.library) != artifact.metadata["binary_sha256"]:
            raise ValueError("stationary CUDA binary hash mismatch")
        self.artifact = artifact
        self.timeline = timeline
        self.profile_device = False
        self.handle = ct.c_void_p()
        self.library = lib = ct.CDLL(str(artifact.library))
        self.natom, self.nao, self.point_capacity = basis.natom, basis.nao, points
        if spin_blocks not in (1, 2):
            raise ValueError("stationary CUDA requires one or two density spin blocks")
        self.spin_blocks = spin_blocks
        self.tasks = np.full((records, 9), -1, dtype=np.int64)
        self.charges = np.ones(records)
        self.used = 0
        self.device = device
        self.borrowed_streams = set()
        self.centers = np.ascontiguousarray(
            basis.packed[: 3 * basis.natom].reshape(-1, 3)
        )
        self.ao_atoms = np.ascontiguousarray(_native_ao_atoms(basis), dtype=np.int64)
        self.primitives, self.aos, self.expansions, requests, self.extended, self.dispatch_domain = _layout(basis)
        self.primitive_table = np.ascontiguousarray(self.primitives, dtype=np.float64)
        self.ao_ranges = np.ascontiguousarray(self.aos[:, 1:3], dtype=np.int64)
        self.ao_norms = np.ascontiguousarray(
            np.ones(self.aos.shape[0]) if self.extended else self.aos[:, 7], dtype=np.float64
        )
        self.topology_identity = _basis_topology_identity(basis)
        self.bound_basis_identity = basis.identity
        self.kinds = {key: i for i, key in enumerate(requests)}
        tail = [ct.c_char_p, ct.c_size_t]
        lib.stationary_create.argtypes = (
            [ct.c_int] * 3 + [ct.c_size_t] * 8 + [ct.POINTER(ct.c_void_p), *tail]
        )
        lib.stationary_topology.argtypes = [
            ct.c_void_p,
            _DOUBLE,
            _INT,
            _DOUBLE,
            _INT,
            *tail,
        ]
        lib.stationary_reset.argtypes = [
            ct.c_void_p,
            _DOUBLE,
            _DOUBLE,
            _DOUBLE,
            ct.c_double,
            *tail,
        ]
        lib.stationary_tasks.argtypes = [
            ct.c_void_p,
            _INT,
            _DOUBLE,
            ct.c_size_t,
            *tail,
        ]
        lib.stationary_nuclear.argtypes = [
            ct.c_void_p,
            ct.c_uint,
            ct.c_int64,
            ct.c_int64,
            ct.c_double,
            ct.c_double,
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
        lib.stationary_finish_reduced.argtypes = [
            ct.c_void_p,
            _DOUBLE,
            ct.c_size_t,
            *tail,
        ]
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
            *(compiler.target if target is None else target).compute_capability,
            basis.natom,
            basis.nao,
            basis.nprimitive,
            points,
            records,
            spin_blocks,
            work_budget,
            budget,
            ct.byref(self.handle),
        )
        if profile_device:
            self.enable_profile()
        self._call(
            "stationary_topology",
            self.handle,
            _ptr(self.primitive_table),
            _ptr(self.ao_ranges),
            _ptr(self.ao_norms),
            _ptr(self.ao_atoms),
        )

    def _call(self, name: typing.Any, *args: typing.Any) -> None:
        error = ct.create_string_buffer(2048)
        if getattr(self.library, name)(*args, error, len(error)):
            raise RuntimeError(error.value.decode())

    def enable_profile(self) -> None:
        if self.profile_device:
            return
        tail = [ct.c_char_p, ct.c_size_t]
        self.library.stationary_profile.argtypes = [ct.c_void_p, *tail]
        self._call("stationary_profile", self.handle)
        self.profile_device = True

    def rebind_geometry(self, basis: typing.Any) -> None:
        """Refresh centers only for a basis with the prepared scientific topology."""
        if (
            basis.natom != self.natom
            or basis.nao != self.nao
            or _basis_topology_identity(basis) != self.topology_identity
        ):
            raise ValueError("stationary CUDA prepared basis topology changed")
        self.centers = np.ascontiguousarray(
            basis.packed[: 3 * basis.natom].reshape(-1, 3)
        )
        self.ao_atoms = np.ascontiguousarray(_native_ao_atoms(basis), dtype=np.int64)
        self.bound_basis_identity = basis.identity

    def reset(
        self,
        tolerance: typing.Any,
        density: typing.Any,
        weighted_density: typing.Any,
    ) -> None:
        self.used = 0
        self.borrowed_streams.clear()
        shape = (self.spin_blocks, self.nao, self.nao)
        density = _checked(density, shape)
        weighted_density = _checked(weighted_density, shape)
        self._call(
            "stationary_reset",
            self.handle,
            _ptr(self.centers),
            _ptr(density),
            _ptr(weighted_density),
            tolerance,
        )

    def flush(self) -> None:
        if self.used:
            pending = self.tasks[: self.used]
            order = np.lexsort((pending[:, 1], pending[:, 0]))
            tasks = np.ascontiguousarray(pending[order])
            charges = np.ascontiguousarray(self.charges[: self.used][order])
            phase = (
                self.timeline.phase("primitive_derivative_reduction_sync")
                if self.timeline is not None
                else nullcontext()
            )
            with phase:
                self._call(
                    "stationary_tasks",
                    self.handle,
                    _ptr(tasks),
                    _ptr(charges),
                    self.used,
                )
            self.used = 0

    def integral(
        self,
        source: typing.Any,
        operator: typing.Any,
        indices: typing.Any,
        nucleus: typing.Any = None,
        charge: typing.Any = 1.0,
    ) -> None:
        """Append one AO task; primitive Cartesian products are traversed natively."""
        indices = tuple(int(i) for i in indices)
        rank = len(indices)
        if rank not in (2, 4):
            raise ValueError("stationary CUDA task rank must be two or four")
        rows = self.aos[list(indices)]
        terms = (
            product(*(self.expansions[i] for i in indices))
            if self.extended
            else ((tuple(self.expansions[i][0][0] for i in indices), 1.0),)
        )
        for selected in terms:
            if self.extended:
                labels = tuple(label for label, _ in selected)
                coefficient = float(np.prod([factor for _, factor in selected]))
                kind = derivative_dispatch_id(operator, labels)
            else:
                labels, coefficient = selected
                kind = self.kinds[operator, labels]
            primitive_work = 1
            for row in rows:
                primitive_work *= int(row[2])
            if self.used == len(self.tasks):
                self.flush()
            task = self.tasks[self.used]
            task.fill(-1)
            task[:4] = (kind, source, rank, -1 if nucleus is None else int(nucleus))
            task[4 : 4 + rank] = indices
            task[8] = primitive_work
            self.charges[self.used] = charge * coefficient
            self.used += 1

    def nuclear(self, a: typing.Any, b: typing.Any, charges: typing.Any) -> None:
        self.flush()
        self._call(
            "stationary_nuclear",
            self.handle,
            self.kinds["nuclear", ()],
            int(a),
            int(b),
            float(charges[a]),
            float(charges[b]),
        )

    def geometry(
        self,
        task: typing.Any,
        owners: typing.Any,
        weights: typing.Any,
        raw: typing.Any,
        *,
        functional: typing.Any = None,
        pbe: typing.Any = None,
    ) -> None:
        view = task.view
        if task._owner.device_id != self.device:
            raise ValueError("stationary/grid current owner device mismatch")
        self.borrowed_streams.add(view.stream)
        owners = _checked(owners, (view.npoint,), np.int64)
        weights = _checked(weights, (view.npoint,))
        raw = _checked(raw, (view.npoint,))
        if functional is None:
            if type(pbe) is not bool:
                raise TypeError(
                    "stationary geometry requires functional=0/1/2 or pbe bool"
                )
            functional = int(pbe)
        elif pbe is not None:
            raise ValueError("specify functional or pbe, not both")
        if type(functional) is not int or functional not in (0, 1, 2):
            raise ValueError("unsupported stationary semilocal functional")
        work = task.density_jets(4 if functional else 1)
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

    def reduced(self) -> typing.Any:
        """Return the fixed seven-source all-electron sum reduced on CUDA."""
        out = np.empty((self.natom, 3))
        self._call("stationary_finish_reduced", self.handle, _ptr(out), out.size)
        return out

    def metrics(self) -> typing.Any:
        values = (ct.c_uint64 * 14)()
        if self.library.stationary_metrics(self.handle, values, 14):
            raise RuntimeError("stationary metrics unavailable")
        metrics = dict(
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
                    "task_descriptors",
                    "task_batches",
                    "h2d_calls",
                    "d2h_calls",
                    "synchronizations",
                    "geometry_batches",
                ),
                values,
            )
        )
        metrics["primitive_batches"] = metrics["task_batches"]
        profile = (ct.c_double * 10)()
        if self.profile_device:
            self.library.stationary_profile_metrics.argtypes = [
                ct.c_void_p,
                ct.POINTER(ct.c_double),
                ct.c_size_t,
            ]
            if self.library.stationary_profile_metrics(self.handle, profile, 10):
                raise RuntimeError("stationary profile metrics unavailable")
        metrics["device_profile_enabled"] = self.profile_device
        metrics["device_phase_ms"] = dict(
            zip(
                (
                    "synchronization_wait_wall",
                    "setup_transfer_and_clear",
                    "setup_validation_kernel",
                    "primitive_h2d",
                    "primitive_derivative_kernel",
                    "primitive_reduction",
                    "geometry_h2d",
                    "geometry_kernel",
                    "geometry_reduction",
                    "final_d2h_wall",
                ),
                profile,
            )
        )
        return metrics

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


def _native_grid_artifact(library: typing.Any, architecture: str) -> CudaArtifact:
    """Bind the already built native CUDA grid code without recompilation."""
    path = Path(library).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"native CUDA library not found: {path}")
    digest = file_hash(path)
    identity = {
        "schema": "vibeqc.native-grid-aot.v1",
        "target": {"architecture": architecture},
    }
    return CudaArtifact(
        path,
        {
            "identity": identity,
            "binary_sha256": digest,
            "key": canonical_hash({"identity": identity, "binary_sha256": digest}),
            "artifact_kind": "native-build-aot",
        },
    )


class PreparedStationaryCudaTopologyMismatch(ValueError):
    """Retained execution is incompatible with the requested scientific topology."""


class PreparedStationaryCudaExecution:
    """Retain method-owned CUDA resources behind the shared prepared-region lease."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stack: ExitStack | None = None
        self._lease = PreparedExecutionLease()
        self.preparation_seconds = 0.0

    @property
    def identity(self) -> str | None:
        return self._lease.identity

    @property
    def host_bound(self) -> int:
        contract = self._lease.contract
        return 0 if contract is None else contract.host_bytes

    @property
    def device_peak_bound(self) -> int:
        contract = self._lease.contract
        return 0 if contract is None else contract.device_bytes

    def _request(
        self,
        *,
        state: typing.Any,
        basis: typing.Any,
        contract: typing.Any,
        plan: typing.Any,
        tensor_plans: typing.Any,
        target: typing.Any,
        aot_directory: typing.Any,
        native_grid_library: typing.Any,
        ecp: bool,
        device: int,
        spec: typing.Any,
        grid_plan: typing.Any,
        source_bytes: int,
        tile_points: int,
        primitive_tile: int,
        integral_terms: int,
        work_budget: int,
    ) -> PreparedExecutionRequest:
        topology = _basis_topology_identity(basis)
        scientific_identity = canonical_hash(
            {
                "plan": plan.identity,
                "method": state.identity.method,
                "family": contract.family,
                "spin": contract.spin,
                "ecp": ecp,
                "topology": topology,
                "backend": state._source.backend,
                "functional": state.identity.functional_identity,
                "regularization": state.identity.regularization_identity,
                "ecp_model": repr(
                    (state._source.ecp_cores, state._source.ecp_terms)
                    if ecp
                    else ("all-electron",)
                ),
                "grid_spec": repr(spec),
            }
        )
        schedule_identity = canonical_hash(
            {
                "partition_iterations": spec.partition_iterations,
                "tile_points": tile_points,
                "primitive_tile": primitive_tile,
                "integral_terms": integral_terms,
                "work_budget": work_budget,
                "grid_allocation_bytes": grid_plan.allocation_bytes,
                "tensor_plans": [
                    (name, value.identity)
                    for name, value in sorted(tensor_plans.items())
                ],
                "aot_directory": (
                    None
                    if aot_directory is None
                    else str(Path(aot_directory).resolve())
                ),
                "native_grid_library": (
                    None
                    if native_grid_library is None
                    else str(Path(native_grid_library).resolve())
                ),
            }
        )
        workspace_identity = canonical_hash(
            {
                "grid_peak_bytes": grid_plan.peak_bytes,
                "source_bytes": source_bytes,
                "tensor": [
                    (name, value.peak_bytes, value.host_bytes)
                    for name, value in sorted(tensor_plans.items())
                ],
            }
        )
        return PreparedExecutionRequest(
            "stationary-dft-cuda",
            scientific_identity,
            canonical_hash(target.to_payload()),
            schedule_identity,
            workspace_identity,
            device=device,
        )

    def ensure(
        self,
        *,
        state: typing.Any,
        basis: typing.Any,
        contract: typing.Any,
        plan: typing.Any,
        tensor_plans: typing.Any,
        compiler: typing.Any,
        cache: typing.Any,
        aot_directory: typing.Any = None,
        native_grid_library: typing.Any = None,
        target: typing.Any = None,
        requests: typing.Any,
        dispatch_domain: tuple[str, ...] | None = None,
        functional: int,
        ecp: bool,
        device: int,
        spec: typing.Any,
        grid_plan: typing.Any,
        source_bytes: int,
        tile_points: int,
        primitive_tile: int,
        integral_terms: int,
        work_budget: int,
        max_device_bytes: int,
        max_host_bytes: int,
        host_bound: int,
        profile_device: bool = False,
    ) -> None:
        target = compiler.target if target is None else target
        request = self._request(
            state=state,
            basis=basis,
            contract=contract,
            plan=plan,
            tensor_plans=tensor_plans,
            target=target,
            aot_directory=aot_directory,
            native_grid_library=native_grid_library,
            ecp=ecp,
            device=device,
            spec=spec,
            grid_plan=grid_plan,
            source_bytes=source_bytes,
            tile_points=tile_points,
            primitive_tile=primitive_tile,
            integral_terms=integral_terms,
            work_budget=work_budget,
        )
        if self._lease.contract is not None:
            try:
                self._lease.require(
                    request,
                    max_host_bytes=max_host_bytes,
                    max_device_bytes=max_device_bytes,
                )
            except PreparedExecutionMismatch as error:
                raise PreparedStationaryCudaTopologyMismatch(
                    "stationary CUDA prepared execution topology changed"
                ) from error
            if getattr(self.sources, "profile_device", False) != profile_device:
                raise PreparedStationaryCudaTopologyMismatch(
                    "stationary CUDA prepared profiling mode changed"
                )
            if (
                basis.identity != self._bound_basis_identity
                or self._lease.needs_refresh
            ):
                if any(
                    file_hash(artifact.library) != artifact.metadata["binary_sha256"]
                    for artifact in self.artifacts
                ):
                    raise ValueError("stationary CUDA prepared artifact hash mismatch")
                self.sources.rebind_geometry(basis)
                self.grid._rebind_centers(
                    np.ascontiguousarray(
                        basis.packed[: 3 * basis.natom].reshape(basis.natom, 3)
                    )
                )
                self._bound_basis_identity = basis.identity
                self._lease.mark_refresh()
            return

        tensor_peak = sum(value.peak_bytes for value in tensor_plans.values())
        device_peak_bound = grid_plan.peak_bytes + source_bytes + tensor_peak
        if device_peak_bound > max_device_bytes:
            raise ValueError("prepared stationary CUDA device budget exceeded")
        retained_host = host_bound + sum(
            value.host_bytes for value in tensor_plans.values()
        )
        if retained_host > max_host_bytes:
            raise ValueError("prepared stationary CUDA host budget exceeded")

        started = perf_counter()
        cache = Path(cache)
        stationary_artifact = (
            compile_stationary_cuda(
                emit_first_derivative_cuda(requests),
                functional=functional,
                plan=plan,
                iterations=spec.partition_iterations,
                compiler=compiler,
                cache=cache,
                dispatch_entries=(derivative_dispatch_entries(dispatch_domain, requests)
                                  if dispatch_domain is not None else None),
            )
            if aot_directory is None or ecp
            else load_stationary_aot_artifact(
                aot_directory,
                functional=functional,
                spin=contract.spin,
                plan=plan,
                architecture=target.architecture,
                iterations=spec.partition_iterations,
            )
        )
        grid_artifact = (
            compile_grid(compiler, cache)
            if native_grid_library is None
            else _native_grid_artifact(native_grid_library, target.architecture)
        )
        tensor_artifacts = {
            name: compile_cuda(value, compiler, cache)
            for name, value in tensor_plans.items()
        }
        stack = ExitStack()
        try:
            sources = stack.enter_context(
                _CudaSources(
                    basis,
                    stationary_artifact,
                    compiler,
                    device,
                    tile_points,
                    primitive_tile,
                    source_bytes,
                    spin_blocks=plan.spin_blocks,
                    target=target,
                    work_budget=work_budget,
                    profile_device=profile_device,
                )
            )
            needs_first = functional != 0
            grid = stack.enter_context(
                CudaGrid(
                    basis,
                    grid_artifact,
                    order=2 if needs_first else 1,
                    tile_points=tile_points,
                    budget_bytes=grid_plan.peak_bytes,
                    device_id=device,
                    active_ao_capacity=basis.nao,
                    ingredients=(
                        ("rho", "gradient", "tau") if needs_first else ("rho",)
                    ),
                )
            )
            tensors = {
                name: stack.enter_context(
                    PreparedCuda(value, tensor_artifacts[name], device=device)
                )
                for name, value in tensor_plans.items()
            }
        except Exception:
            stack.close()
            raise
        artifacts = (
            stationary_artifact,
            grid_artifact,
            *(tensor_artifacts[name] for name in sorted(tensor_artifacts)),
        )
        try:
            self._lease.install(
                request,
                tuple(PreparedArtifactBinding.from_artifact(a) for a in artifacts),
                host_bytes=retained_host,
                device_bytes=device_peak_bound,
            )
        except Exception:
            stack.close()
            raise
        self._stack = stack
        self.sources, self.grid, self.tensors = sources, grid, tensors
        self.stationary_plan = plan
        self.tensor_plans = dict(tensor_plans)
        self.grid_plan = grid_plan
        self.stationary_artifact = stationary_artifact
        self.grid_artifact = grid_artifact
        self.tensor_artifacts = tensor_artifacts
        self.artifacts = artifacts
        self._bound_basis_identity = basis.identity
        self.preparation_seconds = perf_counter() - started

    def close(self) -> None:
        with self._lock:
            if self._stack is not None:
                self._stack.close()
                self._stack = None
            self._lease.invalidate()

    def __enter__(self) -> typing.Any:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        if hasattr(self, "_lock"):
            self.close()


@contextmanager
def _tensor_execution(
    prepared: PreparedStationaryCudaExecution | None,
    name: str,
    plan: typing.Any,
    compiler: typing.Any,
    cache: typing.Any,
    device: int,
    artifacts: list[typing.Any],
    timeline: _ExclusiveWallTimeline,
) -> typing.Iterator[PreparedCuda]:
    if prepared is not None:
        yield prepared.tensors[name]
        return
    with timeline.phase("artifact_lookup_compile"):
        artifact = compile_cuda(plan, compiler, cache)
    artifacts.append(artifact)
    with timeline.phase("owner_construction"):
        owner = PreparedCuda(plan, artifact, device=device)
    with owner:
        yield owner


def _metric_delta(after: typing.Any, before: typing.Any) -> typing.Any:
    result = dict(after)
    for name in (
        "h2d_bytes",
        "d2h_bytes",
        "launches",
        "primitive_records",
        "xc_points",
        "grid_pair_visits",
        "task_descriptors",
        "task_batches",
    ):
        result[name] = after[name] - before[name]
    for name in (
        "h2d_calls",
        "d2h_calls",
        "synchronizations",
        "primitive_batches",
        "geometry_batches",
    ):
        if name in after and name in before:
            result[name] = after[name] - before[name]
    if "device_phase_ms" in after and "device_phase_ms" in before:
        result["device_phase_ms"] = {
            name: value - before["device_phase_ms"][name]
            for name, value in after["device_phase_ms"].items()
        }
    return result


def _grid_metric_delta(after: typing.Any, before: typing.Any) -> typing.Any:
    result = dict(after)
    for name in (
        "device_ms",
        "input_ms",
        "output_ms",
        "packing_ms",
        "library_ms",
        "kernel_ms",
    ):
        result[name] = after[name] - before[name]
    return result


def _complete_rks_cuda_gradient_diagnostic(
    state: typing.Any,
    basis: typing.Any,
    *,
    compiler: typing.Any,
    cache: typing.Any,
    aot_directory: typing.Any = None,
    native_grid_library: typing.Any = None,
    target: CudaTargetInfo | None = None,
    tile_points: typing.Any = 256,
    integral_terms: typing.Any = 32,
    primitive_tile: typing.Any = 4096,
    max_device_bytes: typing.Any = 512 << 20,
    max_host_bytes: typing.Any = 256 << 20,
    max_grid_points: typing.Any = 1_000_000,
    max_primitive_records: typing.Any = 2_000_000,
    max_grid_pair_visits: typing.Any = 100_000_000,
    max_ecp_pair_samples: int = 100_000_000,
    prepared: PreparedStationaryCudaExecution | None = None,
    profile_device: bool = False,
) -> typing.Any:
    """Consume a current native CUDA RKS/UKS snapshot with every plan source.

    Domain: real FP64 direct all-electron s/p LDA/PBE/r2SCAN RKS/UKS, native version-three
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
    timeline = _ExclusiveWallTimeline()
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
    if compiler is None:
        if ecp or aot_directory is None or native_grid_library is None:
            raise TypeError(
                "runtime compilation requires an explicit CUDA compiler adapter"
            )
        if not isinstance(target, CudaTargetInfo):
            raise TypeError(
                "packaged stationary CUDA requires an explicit execution target"
            )
    elif not isinstance(compiler, CudaCompilerAdapter):
        raise TypeError("an explicit CUDA compiler adapter is required")
    elif target is not None and target != compiler.target:
        raise ValueError("stationary CUDA compiler/execution target mismatch")
    else:
        target = compiler.target
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
    if not 1 <= basis.nprimitive <= 4096:
        raise ValueError("CUDA diagnostic primitive-topology cap exceeded")
    _, aos, _, requests, extended, dispatch_domain = _layout(basis)
    if extended and not ecp:
        raise NotImplementedError("all-electron CUDA gradient diagnostic admits s/p bases only")
    primitive_sum = sum(int(r[2]) * (int(r[3]) if extended else 1) for r in aos)
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
    functional = {"lda": 0, "gga": 1, "mgga": 2}[contract.family]
    if functional == 2 and ecp:
        raise NotImplementedError(
            "r2SCAN CUDA stationary gradients do not inherit ECP support"
        )
    needs_first = functional != 0
    functional_name = ("LDA_XC_PW", "PBE", "R2SCAN")[functional]
    device = int(state._source.metadata[12])
    grid_plan = plan_tiles(
        basis,
        backend="cuda",
        order=2 if needs_first else 1,
        tile_points=tile_points,
        active_ao_capacity=n,
        budget_bytes=max_device_bytes,
    )
    source_bytes = (
        8
        * (
            22 * primitive_tile
            + 2 * basis.nprimitive
            + 4 * n
            + 600 * na
            + 3 * tile_points
            + 2 * plan.spin_blocks * n * n
        )
        + 256
    )
    available = max_device_bytes - grid_plan.peak_bytes - source_bytes
    if available <= 0:
        raise ValueError("stationary additional-device budget exceeded")
    tensor_plans = {}
    if ecp:
        tensor_plans["reduction"] = plan_cuda(
            plan.reduction_program(atoms=na), target, max_bytes=available
        )
    # Conservative numeric-array bound: compact task pages/sort staging, resident
    # topology mirrors, D/W admission copies, adapter staging,
    # candidate/publication copies, and tile owners.
    # Compiler objects, Python headers and the caller's existing SCF snapshot
    # are explicit exclusions, as in the reused grid/TensorIR resource contracts.
    host_bound = (
        grid_plan.host_bytes
        + 8
        * (
            34 * primitive_tile
            + 4 * plan.spin_blocks * n * n
            + 120 * na
            + 26 * integral_terms
            + 3 * tile_points
            + 2 * basis.nprimitive
            + 4 * n
            + 80
        )
        + max((tp.host_bytes for tp in tensor_plans.values()), default=0)
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
                target,
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
    if prepared is None:
        with timeline.phase("artifact_lookup_compile"):
            artifact = (
                compile_stationary_cuda(
                    emit_first_derivative_cuda(requests),
                    functional=functional,
                    plan=plan,
                    iterations=spec.partition_iterations,
                    compiler=compiler,
                    cache=cache,
                    dispatch_entries=(derivative_dispatch_entries(dispatch_domain, requests)
                                      if dispatch_domain is not None else None),
                )
                if aot_directory is None or ecp
                else load_stationary_aot_artifact(
                    aot_directory,
                    functional=functional,
                    spin=contract.spin,
                    plan=plan,
                    architecture=target.architecture,
                    iterations=spec.partition_iterations,
                )
            )
            grid_artifact = (
                compile_grid(compiler, cache)
                if native_grid_library is None
                else _native_grid_artifact(native_grid_library, target.architecture)
            )
        artifacts = [artifact, grid_artifact]
    else:
        with timeline.phase("prepared_owner_lookup_or_construction"):
            prepared.ensure(
                state=state,
                basis=basis,
                contract=contract,
                plan=plan,
                tensor_plans=tensor_plans,
                compiler=compiler,
                cache=cache,
                aot_directory=aot_directory,
                native_grid_library=native_grid_library,
                target=target,
                requests=requests,
                dispatch_domain=dispatch_domain,
                functional=functional,
                ecp=ecp,
                device=device,
                spec=spec,
                grid_plan=grid_plan,
                source_bytes=source_bytes,
                tile_points=tile_points,
                primitive_tile=primitive_tile,
                integral_terms=integral_terms,
                work_budget=records,
                max_device_bytes=max_device_bytes,
                max_host_bytes=max_host_bytes,
                host_bound=host_bound,
                profile_device=profile_device,
            )
        artifact = prepared.stationary_artifact
        grid_artifact = prepared.grid_artifact
        artifacts = list(prepared.artifacts)
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
    with timeline.phase("ecp_provider") if ecp else nullcontext():
        derivatives = state._source.ecp_derivatives() if ecp else None
    peak = (
        max(ecp_workspace, grid_plan.peak_bytes + source_bytes)
        if prepared is None
        else max(ecp_workspace, prepared.device_peak_bound)
    )
    if peak > max_device_bytes:
        raise ValueError("stationary additional-device budget exceeded")
    charges = np.asarray([a.atomic_number for a in basis.atoms]) - np.asarray(
        state._source.ecp_cores
    )
    with ExitStack() as stack:
        if prepared is None:
            with timeline.phase("owner_construction"):
                sources = stack.enter_context(
                    _CudaSources(
                        basis,
                        artifact,
                        compiler,
                        device,
                        tile_points,
                        primitive_tile,
                        source_bytes,
                        spin_blocks=plan.spin_blocks,
                        target=target,
                        work_budget=records,
                        timeline=timeline,
                        profile_device=profile_device,
                    )
                )
                ao = stack.enter_context(
                    CudaGrid(
                        basis,
                        grid_artifact,
                        order=2 if needs_first else 1,
                        tile_points=tile_points,
                        budget_bytes=grid_plan.peak_bytes,
                        device_id=device,
                        active_ao_capacity=n,
                        # GGA/meta-GGA geometry needs all four D*jet panels; r2SCAN
                        # additionally consumes tau from the same current density.
                        ingredients=("rho", "gradient", "tau")
                        if needs_first
                        else ("rho",),
                    )
                )
            source_before = grid_before = None
        else:
            sources, ao = prepared.sources, prepared.grid
            sources.timeline = timeline
            with timeline.phase("metrics_collection"):
                source_before, grid_before = sources.metrics(), ao.metrics()
        with timeline.phase("owner_state_reset"):
            sources.reset(
                spec.coincident_tolerance, state.density, state.weighted_density
            )
            ao.set_density(density)
        timeline.switch("python_packing")
        for source, rank, operator in (
            ("one_electron", 2, "kinetic"),
            ("overlap_pulay", 2, "overlap"),
            ("coulomb", 4, "four_center_eri"),
        ):
            domain = RuntimeTaskDomain.rectangular((n,) * rank)
            for page in domain.pages(integral_terms):
                for indices in page.coordinates:
                    sources.integral(_SOURCE_NAMES.index(source), operator, indices)
                    if source == "one_electron":
                        for atom in range(na):
                            sources.integral(
                                0,
                                "nuclear_attraction",
                                indices,
                                atom,
                                charges[atom],
                            )
            sources.flush()
        for atom in range(na):
            for other in range(atom):
                sources.nuclear(atom, other, charges)
        sources.flush()
        grid = state.grid
        with timeline.phase("xc_geometry_and_sync"):
            for begin in range(0, len(grid.points), tile_points):
                end = min(begin + tile_points, len(grid.points))
                with ao.xc_task(
                    grid.points[begin:end],
                    np.arange(n, dtype=np.uintp),
                    functional_name,
                ) as task:
                    sources.geometry(
                        task,
                        np.asarray(grid.owners[begin:end], dtype=np.int64),
                        grid.weights[begin:end],
                        state._source.atomic_weights[begin:end],
                        functional=functional,
                    )
        with timeline.phase("source_d2h_publication"):
            components = sources.finish()
        if ecp:
            # Full ordered AO-pair contraction; the existing TensorIR supplies
            # spin summation and every scientific weight/reduction on CUDA.
            for k, name in enumerate(("ecp_local", "ecp_nonlocal")):
                tp = tensor_plans[name]
                if prepared is None:
                    peak = max(
                        peak, grid_plan.peak_bytes + source_bytes + tp.peak_bytes
                    )
                feeds = {
                    "density_left": np.ascontiguousarray(
                        state.density.reshape(plan.spin_blocks, n * n)
                    ),
                    "integral_derivatives": np.ascontiguousarray(
                        derivatives[k].reshape(3 * na, n * n).T
                    ),
                }
                with _tensor_execution(
                    prepared, name, tp, compiler, cache, device, artifacts, timeline
                ) as contraction:
                    with timeline.phase("tensorir_weight_execution"):
                        result = contraction.execute(feeds)
                    record_tensor(result, feeds)
                    components[name] = result.outputs["gradient"].reshape(na, 3)
        # Validate actual coverage before the complete reduction. All-electron
        # seven-source work reduces inside the stationary owner; ECP retains the
        # generated TensorIR sum because its two extra sources are separate owners.
        plan.reduction_program(atoms=na, sources=components)
        if ecp:
            tp = tensor_plans["reduction"]
            if prepared is None:
                peak = max(peak, grid_plan.peak_bytes + source_bytes + tp.peak_bytes)
            with _tensor_execution(
                prepared, "reduction", tp, compiler, cache, device, artifacts, timeline
            ) as reduction:
                with timeline.phase("final_reduction"):
                    reduced = reduction.execute(components)
                record_tensor(reduced, components)
                gradient = reduced.outputs["gradient"]
        else:
            with timeline.phase("final_reduction"):
                gradient = sources.reduced()
        with timeline.phase("metrics_collection"):
            source_after = sources.metrics()
            grid_after = ao.metrics()
            work = (
                source_after
                if source_before is None
                else _metric_delta(source_after, source_before)
            )
            work["grid_metrics"] = (
                grid_after
                if grid_before is None
                else _grid_metric_delta(grid_after, grid_before)
            )
            work["borrowed_grid_streams"] = tuple(sorted(sources.borrowed_streams))
        if work["owned_device_bytes"] != source_bytes:
            raise RuntimeError("stationary allocation disagrees with admitted bytes")
        timeline.switch("owner_cleanup")
    timeline.switch("publication_validation")
    contract.validate(state)  # Replay/replacement/closure revokes publication.
    if work["primitive_records"] != records or work["grid_pair_visits"] != pair_visits:
        raise RuntimeError("CUDA executed work disagrees with admitted source coverage")
    published_gradient = immutable(gradient)
    published_components = MappingProxyType(
        {key: immutable(value) for key, value in components.items()}
    )
    artifacts_record = tuple(
        {
            "library": str(item.library),
            "binary_sha256": item.metadata["binary_sha256"],
            "key": item.metadata["key"],
        }
        for item in artifacts
    )
    timeline.switch("publication_metadata")
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
        stationary_weight_lowering="generated-tensorir/device-pointwise-v1",
        stationary_weight_plan_identity=plan.identity,
        stationary_weight_programs={
            name: plan.integral_block(name, terms=1).weights.logical_hash
            for name in ("one_electron", "overlap_pulay", "coulomb")
        },
        stationary_weight_tensor_executions=0,
        stationary_weight_roundtrip_bytes=0,
        stationary_final_reduction=(
            "generated-tensorir-v1" if ecp else "native-seven-source-device-sum-v1"
        ),
        stationary_state_dw_upload_bytes=(
            state.density.nbytes + state.weighted_density.nbytes
        ),
        additional_host_numeric_bound=(
            host_bound if prepared is None else prepared.host_bound
        ),
        additional_host_budget=max_host_bytes,
        prepared_execution=prepared is not None,
        prepared_execution_identity=None if prepared is None else prepared.identity,
        prepared_execution_reused=(
            False if prepared is None else prepared._lease.executions > 0
        ),
        prepared_execution_index=(
            None if prepared is None else prepared._lease.executions + 1
        ),
        prepared_owner_preparation_seconds=(
            0.0 if prepared is None else prepared.preparation_seconds
        ),
        prepared_geometry_rebinds=(
            0 if prepared is None else prepared._lease.refreshes
        ),
        snapshot_host_bytes=state._source.values.nbytes,
        snapshot_export_work=dict(state._source.export_work),
        snapshot_export="explicit native CUDA final-state export; W/frame validation is host work",
        host_scope="snapshot validation; AO task descriptor packing/sorting; one D/W owner upload; final TensorIR reduction; immutable result copies",
        measurement_profile_enabled=bool(work.get("device_profile_enabled", False)),
        transfer_work={
            "source_h2d_bytes": work["h2d_bytes"],
            "source_d2h_bytes": work["d2h_bytes"],
            "source_h2d_calls": work.get("h2d_calls", 0),
            "source_d2h_calls": work.get("d2h_calls", 0),
            "source_synchronizations": work.get("synchronizations", 0),
            "tensor_h2d_numeric_bytes": tensor_work["h2d_numeric_bytes"],
            "tensor_d2h_bytes": tensor_work["d2h_bytes"],
        },
        stationary_artifact_kind=artifact.metadata.get("artifact_kind", "runtime-jit"),
        stationary_runtime_module_load=True,
        stationary_driver_ptx_jit_possible=artifact.metadata.get(
            "driver_ptx_jit_possible", True
        ),
        stationary_driver_ptx_jit_required=artifact.metadata.get(
            "driver_ptx_jit_required", False
        ),
        grid_artifact_kind=grid_artifact.metadata.get("artifact_kind", "runtime-jit"),
        artifacts=artifacts_record,
    )
    timeline_record = timeline.finish()
    work.update(
        endpoint_seconds=timeline_record["endpoint_seconds"],
        timeline=timeline_record,
    )
    return DiagnosticStationaryGradient(
        published_gradient,
        published_components,
        plan.identity,
        state.identity,
        MappingProxyType(work),
        execution=("cuda-nine-source" if ecp else "cuda-seven-source")
        + "/generated-device-stationary-weights-v1",
    )


def complete_rks_cuda_gradient_diagnostic(
    state: typing.Any,
    basis: typing.Any,
    *,
    compiler: typing.Any,
    cache: typing.Any,
    aot_directory: typing.Any = None,
    native_grid_library: typing.Any = None,
    target: CudaTargetInfo | None = None,
    tile_points: typing.Any = 256,
    integral_terms: typing.Any = 32,
    primitive_tile: typing.Any = 4096,
    max_device_bytes: typing.Any = 512 << 20,
    max_host_bytes: typing.Any = 256 << 20,
    max_grid_points: typing.Any = 1_000_000,
    max_primitive_records: typing.Any = 2_000_000,
    max_grid_pair_visits: typing.Any = 100_000_000,
    max_ecp_pair_samples: int = 100_000_000,
    prepared: PreparedStationaryCudaExecution | None = None,
    profile_device: bool = False,
) -> typing.Any:
    """Execute once, optionally retaining validated CUDA owners for later replay."""
    kwargs = {
        "compiler": compiler,
        "cache": cache,
        "aot_directory": aot_directory,
        "native_grid_library": native_grid_library,
        "target": target,
        "tile_points": tile_points,
        "integral_terms": integral_terms,
        "primitive_tile": primitive_tile,
        "max_device_bytes": max_device_bytes,
        "max_host_bytes": max_host_bytes,
        "max_grid_points": max_grid_points,
        "max_primitive_records": max_primitive_records,
        "max_grid_pair_visits": max_grid_pair_visits,
        "max_ecp_pair_samples": max_ecp_pair_samples,
        "prepared": prepared,
        "profile_device": profile_device,
    }
    if prepared is None:
        return _complete_rks_cuda_gradient_diagnostic(state, basis, **kwargs)
    if not isinstance(prepared, PreparedStationaryCudaExecution):
        raise TypeError("prepared CUDA execution has the wrong owner type")
    with prepared._lock:
        try:
            result = _complete_rks_cuda_gradient_diagnostic(state, basis, **kwargs)
        except Exception:
            prepared._lease.mark_failure()
            raise
        prepared._lease.mark_success()
        return result
