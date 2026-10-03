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
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
from pathlib import Path
from time import perf_counter
from types import MappingProxyType

import numpy as np
from generativeqc_compiler.common.arrays import immutable
from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from generativeqc_compiler.common.cuda_runtime import CudaArtifact
from generativeqc_compiler.common.cuda_target import CudaTargetInfo
from generativeqc_compiler.common.execution import CompiledExecutionIdentity
from generativeqc_compiler.common.native_call import checked_native_call
from generativeqc_compiler.common.prepared_execution import (
    PreparedArtifactBinding,
    PreparedExecutionLease,
    PreparedExecutionMismatch,
    PreparedExecutionRequest,
)
from generativeqc_compiler.common.provenance import canonical_hash, file_hash
from generativeqc_compiler.common.runtime_domain import (
    RuntimeTaskDomain,
    RuntimeTaskPage,
)
from generativeqc_compiler.dft.cuda import (
    CudaGrid,
    GridTaskView,
)
from generativeqc_compiler.dft.cuda import (
    compile_cuda as compile_grid,
)
from generativeqc_compiler.dft.plan import plan_tiles
from generativeqc_compiler.integral.first_derivative_native import (
    emit_first_derivative_cuda,
)
from generativeqc_compiler.integral.first_derivative_schedule import (
    COMPONENT_LABELS,
    CUDA_REQUESTS_PER_UNIT,
    derivative_binding,
    derivative_cuda_sources,
    derivative_requests,
)
from generativeqc_compiler.method.stationary_cuda import (
    QUALIFIED_SPD_COMPONENTS,
    STATIONARY_RUNTIME_SOURCE_NAMES,
    compile_stationary_cuda,
    encode_stationary_derivative_kind,
    load_stationary_aot_artifact,
    qualified_sp_requests,
    stationary_runtime_sources,
)
from generativeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)
from generativeqc_compiler.method.stationary_resources import (
    plan_stationary_cuda_grid_work,
    plan_stationary_cuda_resources,
    stationary_cuda_allocation_bytes,
    stationary_cuda_requires_native_integrals,
    stationary_native_pair_reserve,
)
from generativeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
from generativeqc_compiler.tensor.cuda_plan import plan_cuda
from generativeqc_compiler.xc._generated_native_semilocal import (
    SEMILOCAL_FAMILY_BY_CODE,
    SEMILOCAL_FAMILY_CODES,
)
from generativeqc_compiler.xc._generated_split_hybrids import SPLIT_HYBRIDS

from ._dft_gradient import (
    StationaryDerivativeContract,
    _native_ao_atoms,
    native_ao_geometry_identity,
)
from ._stationary_cpu import DiagnosticStationaryGradient

_REGISTERED_STATIONARY_CODES = SEMILOCAL_FAMILY_CODES | frozenset(
    record["functional_code"] for record in SPLIT_HYBRIDS.values()
)


def _stationary_density_jet_count(functional: int) -> int:
    record = SEMILOCAL_FAMILY_BY_CODE.get(functional)
    return 4 if record is None or record["requires_gradient"] else 1


_DOUBLE = ct.POINTER(ct.c_double)
_INT = ct.POINTER(ct.c_int64)
_SOURCE_NAMES = STATIONARY_RUNTIME_SOURCE_NAMES
_DEFAULT_MAX_PRIMITIVE_RECORDS = 16_000_000


class _StationaryTaskSource(typing.Protocol):
    """Versioned/identity-bearing bounded derivative task producer."""

    @property
    def identity(self) -> str: ...

    @property
    def logical_size(self) -> int: ...

    def pages(self, capacity: int) -> typing.Iterator[RuntimeTaskPage]: ...

    def to_payload(self) -> dict[str, typing.Any]: ...


@dataclass(frozen=True, slots=True)
class _StationaryTaskExecution:
    """Bounded producer evidence independent of the derivative task source."""

    mode: str
    source_schema: str
    domain_identity: str
    logical_tasks: int
    fixed_capacity: int
    resident_capacity: int
    page_capacity: int
    producer_pages: int


class _BoundedStationaryTaskExecutor:
    """Run a finite derivative-task producer without retaining its full domain.

    The current AO producer is only one client of this boundary. #1477 can
    replace it with compact shell tasks without changing the execution policy.
    resident_capacity classifies when one logical producer fits the current
    descriptor reservoir; native task-batch metrics remain authoritative when
    Cartesian component expansion causes an earlier flush.
    """

    def __init__(
        self,
        *,
        fixed_capacity: int,
        resident_capacity: int,
        page_capacity: int,
    ) -> None:
        for value, label in (
            (fixed_capacity, "stationary fixed task capacity"),
            (resident_capacity, "stationary resident task capacity"),
            (page_capacity, "stationary task page capacity"),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        if fixed_capacity > resident_capacity:
            raise ValueError(
                "stationary fixed task capacity exceeds resident task capacity"
            )
        if page_capacity > resident_capacity:
            raise ValueError(
                "stationary task page capacity exceeds resident task capacity"
            )
        self.fixed_capacity = fixed_capacity
        self.resident_capacity = resident_capacity
        self.page_capacity = page_capacity

    def execute_pages(
        self,
        source: _StationaryTaskSource,
        submit_page: typing.Callable[[RuntimeTaskPage], None],
        *,
        finish_page: typing.Callable[[], None] | None = None,
    ) -> _StationaryTaskExecution:
        try:
            identity = source.identity
            logical_tasks = source.logical_size
            page_source = source.pages
            payload_source = source.to_payload
        except AttributeError as error:
            raise TypeError(
                "stationary derivative producer requires a versioned identity-bearing task source"
            ) from error
        if (
            type(identity) is not str
            or len(identity) != 64
            or any(char not in "0123456789abcdef" for char in identity)
        ):
            raise ValueError(
                "stationary derivative task source requires a SHA-256 identity"
            )
        if type(logical_tasks) is not int or logical_tasks < 0:
            raise ValueError(
                "stationary derivative task source requires nonnegative logical size"
            )
        if not callable(page_source) or not callable(payload_source):
            raise TypeError(
                "stationary derivative task source requires bounded pages/payload"
            )
        payload = payload_source()
        if not isinstance(payload, dict) or type(payload.get("schema")) is not str:
            raise ValueError(
                "stationary derivative task source requires a versioned schema"
            )
        source_schema = payload["schema"]
        if not source_schema.startswith("generativeqc.") or not source_schema.endswith(
            ".v1"
        ):
            raise ValueError("unsupported stationary derivative task-source schema")
        if canonical_hash(payload) != identity:
            raise ValueError(
                "stationary derivative task-source identity/payload mismatch"
            )
        if not callable(submit_page):
            raise TypeError("stationary derivative producer requires a page callback")
        if finish_page is not None and not callable(finish_page):
            raise TypeError("stationary derivative producer requires a finish callback")

        mode = (
            "empty"
            if logical_tasks == 0
            else (
                "fixed"
                if logical_tasks <= self.fixed_capacity
                else (
                    "resident" if logical_tasks <= self.resident_capacity else "paged"
                )
            )
        )
        submitted = pages = 0
        page_rank: int | None = None
        for page in page_source(self.page_capacity):
            if not isinstance(page, RuntimeTaskPage):
                raise TypeError(
                    "stationary derivative task source yielded an invalid page"
                )
            if (
                page.domain_identity != identity
                or page.ordinal != pages
                or page.offset != submitted
                or page.capacity != self.page_capacity
            ):
                raise RuntimeError(
                    "stationary derivative task page identity/order mismatch"
                )
            if page.count > self.page_capacity:
                raise RuntimeError("stationary task producer exceeded page capacity")
            if page_rank is None:
                page_rank = page.rank
            elif page.rank != page_rank:
                raise RuntimeError(
                    "stationary derivative task source changed page rank"
                )
            submit_page(page)
            if finish_page is not None:
                finish_page()
            submitted += page.count
            pages += 1
        if submitted != logical_tasks:
            raise RuntimeError("stationary task producer coverage mismatch")
        return _StationaryTaskExecution(
            mode,
            source_schema,
            identity,
            logical_tasks,
            self.fixed_capacity,
            self.resident_capacity,
            self.page_capacity,
            pages,
        )

    def execute(
        self,
        source: _StationaryTaskSource,
        submit: typing.Callable[[tuple[int, ...]], None],
        *,
        finish_page: typing.Callable[[], None] | None = None,
    ) -> _StationaryTaskExecution:
        if not callable(submit):
            raise TypeError("stationary derivative producer requires a submit callback")

        def submit_page(page: RuntimeTaskPage) -> None:
            for coordinate in page.coordinates:
                submit(coordinate)

        return self.execute_pages(source, submit_page, finish_page=finish_page)


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
            "schema": "generativeqc.stationary-cuda-exclusive-wall.v1",
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


def _component_mode(expansions: typing.Any) -> bool:
    """Return whether public AOs need Cartesian component expansion."""

    return any(
        len(expansion) != 1 or len(expansion[0][0]) > 1 for expansion in expansions
    )


def _component_domain(expansions: typing.Any) -> tuple[str, ...]:
    labels = {component for expansion in expansions for component, _ in expansion}
    domain = tuple(component for component in COMPONENT_LABELS if component in labels)
    if len(domain) != len(labels):
        raise ValueError("stationary CUDA component domain exceeds s/p/d labels")
    return domain


def _artifact_derivative_requests(
    requests: typing.Any, artifact: typing.Any
) -> typing.Any:
    """Use the loaded component artifact's inventory as the task-kind ABI.

    JIT kernels use the compact basis inventory. Packaged kernels contain the
    complete SPD inventory, whose request indices differ for a basis subset.
    Selecting the package must therefore also select its request numbering.
    """
    domain = artifact.metadata.get("component_domain")
    if domain is None:
        return requests
    inventory = derivative_requests(tuple(domain))
    if not set(requests).issubset(inventory):
        raise ValueError("stationary CUDA artifact omits required derivative requests")
    return inventory


def _layout(basis: typing.Any, *, integral_derivatives: bool = True) -> typing.Any:
    """Read normalized public-AO records without evaluating integrals.

    Generic stationary integral descriptors remain qualified through d shells.
    Geometry-only consumers can reuse the same packed AO topology through f
    without constructing the combinatorial Cartesian derivative inventory.
    """
    if type(integral_derivatives) is not bool:
        raise TypeError("integral_derivatives must be boolean")
    if integral_derivatives and any(s.angular_momentum > 2 for s in basis.shells):
        raise NotImplementedError(
            "CUDA gradient integral descriptors admit s/p/d bases only"
        )
    if not integral_derivatives and any(s.angular_momentum > 3 for s in basis.shells):
        raise NotImplementedError(
            "CUDA stationary geometry admits through f bases only"
        )
    start = 3 * basis.natom
    primitives = basis.packed[start : start + 2 * basis.nprimitive].reshape(-1, 2)
    aos = basis.packed[start + 2 * basis.nprimitive :].reshape(-1, 16)
    if any(int(row[3]) not in (1, 2, 3) for row in aos):
        raise ValueError("public AOs require one to three Cartesian components")
    expansions = tuple(
        tuple(
            (
                "".join(
                    axis * int(power)
                    for axis, power in zip("xyz", row[4 + 4 * term : 7 + 4 * term])
                ),
                float(row[7 + 4 * term]),
            )
            for term in range(int(row[3]))
        )
        for row in aos
    )
    requests = (
        (
            derivative_requests(_component_domain(expansions))
            if _component_mode(expansions)
            else qualified_sp_requests()
        )
        if integral_derivatives
        else (("nuclear", ()),)
    )
    return primitives, aos, expansions, requests


class _CudaSources:
    """Serialized finite owner; bounded AO tasks expand primitives only on CUDA.

    Geometry-only owners use plain nuclear dispatch kinds, independently of
    their AO angular momentum. Encoded Cartesian bindings belong exclusively
    to the sharded integral-derivative dispatcher.
    """

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
        page_work_budget: typing.Any = _DEFAULT_MAX_PRIMITIVE_RECORDS,
        timeline: _ExclusiveWallTimeline | None = None,
        profile_device: bool = False,
        source_names: tuple[str, ...] = _SOURCE_NAMES,
        integral_derivatives: bool = True,
        cooperative_becke: bool | None = None,
    ) -> None:
        if type(integral_derivatives) is not bool:
            raise TypeError("integral_derivatives must be boolean")
        self.source_names = source_names
        self.integral_derivatives = integral_derivatives
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
        self.page_work_budget = int(page_work_budget)
        self.pending_primitive_records = 0
        self.primitive_pages = 0
        self.primitive_page_peak_records = 0
        self.bulk_pack_chunks = 0
        self.bulk_packed_descriptors = 0
        self.scalar_packed_descriptors = 0
        self.device = device
        self.borrowed_streams = set()
        self.centers = np.ascontiguousarray(
            basis.packed[: 3 * basis.natom].reshape(-1, 3)
        )
        self.ao_atoms = np.ascontiguousarray(_native_ao_atoms(basis), dtype=np.int64)
        self.primitives, self.aos, expansions, requests = _layout(
            basis, integral_derivatives=integral_derivatives
        )
        self.expansions = tuple(expansions)
        self.component_mode = integral_derivatives and _component_mode(self.expansions)
        self.components = tuple(expansion[0][0] for expansion in self.expansions)
        self.primitive_table = np.ascontiguousarray(self.primitives, dtype=np.float64)
        self.ao_ranges = np.ascontiguousarray(self.aos[:, 1:3], dtype=np.int64)
        self.ao_norms = np.ascontiguousarray(
            np.ones(len(self.aos)) if self.component_mode else self.aos[:, 7],
            dtype=np.float64,
        )
        self.topology_identity = _basis_topology_identity(basis)
        self.bound_basis_identity = basis.identity
        self.kinds = {
            key: i
            for i, key in enumerate(_artifact_derivative_requests(requests, artifact))
        }
        if integral_derivatives:
            component_index = {label: i for i, label in enumerate(COMPONENT_LABELS)}
            self.component_ids = np.asarray(
                [component_index[label] for label in self.components], dtype=np.int64
            )
        else:
            self.component_ids = None
        self.kind_tables: dict[tuple[str, int], np.ndarray] = {}
        tail = [ct.c_char_p, ct.c_size_t]
        lib.stationary_create.argtypes = (
            [ct.c_int] * 3 + [ct.c_size_t] * 10 + [ct.POINTER(ct.c_void_p), *tail]
        )
        lib.stationary_configure_becke.argtypes = [
            ct.c_void_p,
            ct.c_size_t,
            ct.c_size_t,
            *tail,
        ]
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
        lib.stationary_geometry_reset.argtypes = [
            ct.c_void_p,
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
        geometry_args = [
            ct.c_void_p,
            ct.POINTER(GridTaskView),
            _DOUBLE,
            _INT,
            _DOUBLE,
            _DOUBLE,
            *tail,
        ]
        lib.stationary_geometry.argtypes = geometry_args
        lib.stationary_geometry_enqueue.argtypes = geometry_args
        resident_external_args = [
            ct.c_void_p,
            ct.POINTER(GridTaskView),
            _DOUBLE,
            _INT,
            _DOUBLE,
            _DOUBLE,
            ct.c_void_p,
            ct.c_size_t,
            ct.c_size_t,
            *tail,
        ]
        lib.stationary_geometry_external_device.argtypes = resident_external_args
        lib.stationary_geometry_external_device_enqueue.argtypes = (
            resident_external_args
        )
        molecular_args = [
            ct.c_void_p,
            ct.POINTER(GridTaskView),
            _DOUBLE,
            ct.c_size_t,
            ct.c_size_t,
            _DOUBLE,
            _DOUBLE,
            *tail,
        ]
        lib.stationary_geometry_molecular_enqueue.argtypes = molecular_args
        resident_molecular_args = [
            ct.c_void_p,
            ct.POINTER(GridTaskView),
            _DOUBLE,
            ct.c_size_t,
            ct.c_size_t,
            _DOUBLE,
            _DOUBLE,
            ct.c_void_p,
            ct.c_size_t,
            ct.c_size_t,
            *tail,
        ]
        lib.stationary_geometry_external_device_molecular_enqueue.argtypes = (
            resident_molecular_args
        )
        molecular_resident_weight_args = [
            ct.c_void_p,
            ct.POINTER(GridTaskView),
            _DOUBLE,
            ct.c_size_t,
            ct.c_size_t,
            ct.c_void_p,
            _DOUBLE,
            *tail,
        ]
        lib.stationary_geometry_molecular_resident_weights_enqueue.argtypes = (
            molecular_resident_weight_args
        )
        resident_molecular_resident_weight_args = [
            ct.c_void_p,
            ct.POINTER(GridTaskView),
            _DOUBLE,
            ct.c_size_t,
            ct.c_size_t,
            ct.c_void_p,
            _DOUBLE,
            ct.c_void_p,
            ct.c_size_t,
            ct.c_size_t,
            *tail,
        ]
        lib.stationary_geometry_external_device_molecular_resident_weights_enqueue.argtypes = resident_molecular_resident_weight_args
        lib.stationary_geometry_drain.argtypes = [ct.c_void_p, *tail]
        lib.stationary_finish.argtypes = [ct.c_void_p, _DOUBLE, ct.c_size_t, *tail]
        lib.stationary_finish_span.argtypes = [
            ct.c_void_p,
            ct.c_size_t,
            ct.c_size_t,
            _DOUBLE,
            ct.c_size_t,
            *tail,
        ]
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
        # This is the admitted upper bound. Native metrics report actual retained
        # geometry bytes if a typed device-allocation failure selects the fallback.
        self.resources = plan_stationary_cuda_resources(
            atoms=basis.natom,
            aos=basis.nao,
            primitives=basis.nprimitive,
            points=points,
            tasks=records,
            spins=spin_blocks,
            sources=len(source_names),
            target=compiler.target if target is None else target,
            budget_bytes=budget,
            cooperative_becke=cooperative_becke,
        )
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
            page_work_budget,
            budget,
            self.resources.geometry_lanes,
            self.resources.geometry_threads,
            ct.byref(self.handle),
        )
        self._call(
            "stationary_configure_becke",
            self.handle,
            self.resources.becke_threads_per_point,
            self.resources.becke_shared_bytes,
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
        checked_native_call(getattr(self.library, name), *args)

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
        self.pending_primitive_records = 0
        self.primitive_pages = 0
        self.primitive_page_peak_records = 0
        self.bulk_pack_chunks = 0
        self.bulk_packed_descriptors = 0
        self.scalar_packed_descriptors = 0
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

    def reset_geometry(self, tolerance: typing.Any) -> None:
        """Reset only geometry accumulators without uploading unused D/W matrices."""
        self.used = 0
        self.pending_primitive_records = 0
        self.primitive_pages = 0
        self.primitive_page_peak_records = 0
        self.bulk_pack_chunks = 0
        self.bulk_packed_descriptors = 0
        self.scalar_packed_descriptors = 0
        self.borrowed_streams.clear()
        self._call(
            "stationary_geometry_reset",
            self.handle,
            _ptr(self.centers),
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
            page_primitive_records = int(np.sum(tasks[:, 8], dtype=np.int64))
            with phase:
                self._call(
                    "stationary_tasks",
                    self.handle,
                    _ptr(tasks),
                    _ptr(charges),
                    self.used,
                )
            self.primitive_pages += 1
            self.primitive_page_peak_records = max(
                self.primitive_page_peak_records, page_primitive_records
            )
            self.used = 0
            self.pending_primitive_records = 0

    def integral(
        self,
        source: typing.Any,
        operator: typing.Any,
        indices: typing.Any,
        nucleus: typing.Any = None,
        charge: typing.Any = 1.0,
    ) -> None:
        """Append public-AO tasks while Cartesian primitive products stay native."""
        if not self.integral_derivatives:
            raise NotImplementedError(
                "geometry-only stationary CUDA source does not admit integral tasks"
            )
        indices = tuple(int(i) for i in indices)
        rank = len(indices)
        if rank not in (2, 4):
            raise ValueError("stationary CUDA task rank must be two or four")
        rows = self.aos[list(indices)]
        primitive_work = 1
        for row in rows:
            primitive_work *= int(row[2])
        if self.component_mode:
            for terms in product(*(self.expansions[i] for i in indices)):
                binding = derivative_binding(
                    operator, tuple(component for component, _ in terms)
                )
                kind = encode_stationary_derivative_kind(
                    self.kinds[binding.request],
                    binding,
                    rank=rank,
                    has_nucleus=nucleus is not None,
                )
                coefficient = float(charge) * float(
                    np.prod([value for _, value in terms])
                )
                self._append_task(
                    kind,
                    source,
                    rank,
                    indices,
                    primitive_work,
                    nucleus,
                    coefficient,
                )
            return
        kind = self.kinds[operator, tuple(self.components[i] for i in indices)]
        self._append_task(
            kind, source, rank, indices, primitive_work, nucleus, float(charge)
        )

    def _kind_table(self, operator: str, rank: int) -> np.ndarray:
        key = (operator, rank)
        cached = self.kind_tables.get(key)
        if cached is not None:
            return cached
        shape = (len(COMPONENT_LABELS),) * rank
        table = np.full(shape, -1, dtype=np.int64)
        component_index = {label: i for i, label in enumerate(COMPONENT_LABELS)}
        for request, kind in self.kinds.items():
            request_operator, components = request
            if request_operator != operator or len(components) != rank:
                continue
            table[tuple(component_index[label] for label in components)] = kind
        self.kind_tables[key] = table
        return table

    def integral_page(
        self,
        source: int,
        operator: str,
        coordinates: typing.Iterable[tuple[int, ...]],
        nucleus: int | None = None,
        charge: float = 1.0,
    ) -> None:
        """Append one bounded logical page, vectorizing the scalar AO producer."""
        if not self.integral_derivatives:
            raise NotImplementedError(
                "geometry-only stationary CUDA source does not admit integral tasks"
            )
        coordinates = tuple(coordinates)
        if not coordinates:
            return
        rank = len(coordinates[0])
        if rank not in (2, 4) or any(len(row) != rank for row in coordinates):
            raise ValueError("stationary CUDA page requires uniform rank two or four")
        if self.component_mode:
            for indices in coordinates:
                self.integral(
                    source,
                    operator,
                    indices,
                    nucleus=nucleus,
                    charge=charge,
                )
            return
        indices = np.asarray(coordinates, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= self.nao):
            raise ValueError("stationary CUDA page contains invalid AO indices")
        table = self._kind_table(operator, rank)
        selectors = tuple(self.component_ids[indices[:, axis]] for axis in range(rank))
        kinds = table[selectors]
        if np.any(kinds < 0):
            raise ValueError(
                "stationary CUDA page requests an unavailable derivative kind"
            )
        primitive_work = np.prod(
            self.aos[indices, 2].astype(np.int64),
            axis=1,
            dtype=np.int64,
        )
        if np.any(primitive_work > self.page_work_budget):
            raise ValueError(
                "stationary CUDA descriptor exceeds primitive page work budget"
            )

        offset = 0
        while offset < len(coordinates):
            if self.used == len(self.tasks):
                self.flush()
            descriptor_room = len(self.tasks) - self.used
            work_room = self.page_work_budget - self.pending_primitive_records
            if work_room <= 0:
                self.flush()
                continue
            candidate = primitive_work[offset : offset + descriptor_room]
            cumulative = np.cumsum(candidate, dtype=np.int64)
            count = int(np.searchsorted(cumulative, work_room, side="right"))
            if count == 0:
                self.flush()
                continue

            begin, end = self.used, self.used + count
            source_begin, source_end = offset, offset + count
            tasks = self.tasks[begin:end]
            tasks.fill(-1)
            tasks[:, 0] = kinds[source_begin:source_end]
            tasks[:, 1] = int(source)
            tasks[:, 2] = rank
            tasks[:, 3] = -1 if nucleus is None else int(nucleus)
            tasks[:, 4 : 4 + rank] = indices[source_begin:source_end]
            selected_work = primitive_work[source_begin:source_end]
            tasks[:, 8] = selected_work
            self.charges[begin:end] = float(charge)
            self.used = end
            self.pending_primitive_records += int(np.sum(selected_work, dtype=np.int64))
            self.bulk_pack_chunks += 1
            self.bulk_packed_descriptors += count
            offset = source_end

    def _append_task(
        self,
        kind: int,
        source: int,
        rank: int,
        indices: tuple[int, ...],
        primitive_work: int,
        nucleus: typing.Any,
        charge: float,
    ) -> None:
        if primitive_work > self.page_work_budget:
            raise ValueError(
                "stationary CUDA descriptor exceeds primitive page work budget"
            )
        if (
            self.used == len(self.tasks)
            or self.pending_primitive_records + primitive_work > self.page_work_budget
        ):
            self.flush()
        task = self.tasks[self.used]
        task.fill(-1)
        task[:4] = (
            kind,
            source,
            rank,
            -1 if nucleus is None else int(nucleus),
        )
        task[4 : 4 + rank] = indices
        task[8] = primitive_work
        self.charges[self.used] = charge
        self.used += 1
        self.pending_primitive_records += primitive_work
        self.scalar_packed_descriptors += 1

    def nuclear(self, a: typing.Any, b: typing.Any, charges: typing.Any) -> None:
        self.flush()
        kind = self.kinds["nuclear", ()]
        if self.component_mode:
            kind = encode_stationary_derivative_kind(
                kind,
                derivative_binding("nuclear", ()),
                rank=2,
                has_nucleus=False,
            )
        self._call(
            "stationary_nuclear",
            self.handle,
            kind,
            int(a),
            int(b),
            float(charges[a]),
            float(charges[b]),
        )

    def nuclear_all(self, charges: typing.Any) -> None:
        """Submit every nuclear pair in deterministic atom-major order with one gate."""
        self.flush()
        charges = _checked(charges, (self.natom,))
        kind = self.kinds["nuclear", ()]
        if self.component_mode:
            kind = encode_stationary_derivative_kind(
                kind,
                derivative_binding("nuclear", ()),
                rank=2,
                has_nucleus=False,
            )
        tail = [ct.c_char_p, ct.c_size_t]
        self.library.stationary_nuclear_all.argtypes = [
            ct.c_void_p,
            ct.c_uint,
            _DOUBLE,
            ct.c_size_t,
            *tail,
        ]
        self._call(
            "stationary_nuclear_all",
            self.handle,
            kind,
            _ptr(charges),
            self.natom,
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
        """Enqueue one semilocal geometry tile on the borrowed grid stream.

        Production execution defers the host error/synchronization gate until
        drain_geometry(). Detailed device profiling keeps the legacy synchronous
        call so its per-phase event timings remain attributable.
        """
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
                    "stationary geometry requires a registered functional code or pbe bool"
                )
            functional = int(pbe)
        elif pbe is not None:
            raise ValueError("specify functional or pbe, not both")
        if (
            type(functional) is not int
            or functional not in _REGISTERED_STATIONARY_CODES
        ):
            raise ValueError("unsupported stationary semilocal functional")
        work = task.density_jets(_stationary_density_jet_count(functional))
        self._call(
            "stationary_geometry"
            if self.profile_device
            else "stationary_geometry_enqueue",
            self.handle,
            ct.byref(view),
            work,
            _ptr(owners),
            _ptr(weights),
            _ptr(raw),
        )

    def geometry_molecular(
        self,
        task: typing.Any,
        owner_offset: typing.Any,
        points_per_atom: typing.Any,
        weights: typing.Any,
        raw: typing.Any,
        *,
        functional: typing.Any = None,
        pbe: typing.Any = None,
    ) -> None:
        """Consume atom-major molecular-grid ownership without an owner upload."""
        view = task.view
        if task._owner.device_id != self.device:
            raise ValueError("stationary/grid current owner device mismatch")
        if (
            type(owner_offset) is not int
            or owner_offset < 0
            or type(points_per_atom) is not int
            or points_per_atom <= 0
        ):
            raise ValueError("invalid molecular-grid owner interval")
        self.borrowed_streams.add(view.stream)
        weights = _checked(weights, (view.npoint,))
        raw = _checked(raw, (view.npoint,))
        if functional is None:
            if type(pbe) is not bool:
                raise TypeError(
                    "stationary geometry requires a registered functional code or pbe bool"
                )
            functional = int(pbe)
        elif pbe is not None:
            raise ValueError("specify functional or pbe, not both")
        if (
            type(functional) is not int
            or functional not in _REGISTERED_STATIONARY_CODES
        ):
            raise ValueError("unsupported stationary semilocal functional")
        if self.profile_device:
            owners = (
                np.arange(owner_offset, owner_offset + view.npoint, dtype=np.int64)
                // points_per_atom
            )
            self.geometry(task, owners, weights, raw, functional=functional)
            return
        work = task.density_jets(_stationary_density_jet_count(functional))
        self._call(
            "stationary_geometry_molecular_enqueue",
            self.handle,
            ct.byref(view),
            work,
            owner_offset,
            points_per_atom,
            _ptr(weights),
            _ptr(raw),
        )

    @staticmethod
    def _device_pointer(value: typing.Any, label: str) -> ct.c_void_p:
        if isinstance(value, int):
            pointer = ct.c_void_p(value)
        elif isinstance(value, ct.c_void_p):
            pointer = value
        else:
            try:
                pointer = ct.cast(value, ct.c_void_p)
            except (TypeError, ValueError) as error:
                raise TypeError(f"{label} requires a device pointer") from error
        if not pointer.value:
            raise ValueError(f"{label} device pointer is null")
        return pointer

    def geometry_molecular_resident_weights(
        self,
        task: typing.Any,
        owner_offset: typing.Any,
        points_per_atom: typing.Any,
        device_weights: typing.Any,
        host_weights: typing.Any,
        device_raw: typing.Any,
        host_raw: typing.Any,
        *,
        functional: typing.Any = None,
        pbe: typing.Any = None,
    ) -> None:
        """Consume resident partition weights and implicit molecular-grid owners."""
        view = task.view
        if task._owner.device_id != self.device:
            raise ValueError("stationary/grid current owner device mismatch")
        if (
            type(owner_offset) is not int
            or owner_offset < 0
            or type(points_per_atom) is not int
            or points_per_atom <= 0
        ):
            raise ValueError("invalid molecular-grid owner interval")
        self.borrowed_streams.add(view.stream)
        device_weights = self._device_pointer(
            device_weights, "resident molecular-grid weights"
        )
        device_raw = self._device_pointer(
            device_raw, "resident molecular-grid atomic measures"
        )
        if functional is None:
            if type(pbe) is not bool:
                raise TypeError(
                    "stationary geometry requires a registered functional code or pbe bool"
                )
            functional = int(pbe)
        elif pbe is not None:
            raise ValueError("specify functional or pbe, not both")
        if (
            type(functional) is not int
            or functional not in _REGISTERED_STATIONARY_CODES
        ):
            raise ValueError("unsupported stationary semilocal functional")
        if self.profile_device:
            self.geometry_molecular(
                task,
                owner_offset,
                points_per_atom,
                _checked(host_weights, (view.npoint,)),
                _checked(host_raw, (view.npoint,)),
                functional=functional,
            )
            return
        work = task.density_jets(_stationary_density_jet_count(functional))
        self._call(
            "stationary_geometry_molecular_resident_weights_enqueue",
            self.handle,
            ct.byref(view),
            work,
            owner_offset,
            points_per_atom,
            device_weights,
            ct.cast(device_raw, _DOUBLE),
        )

    def geometry_external_device_molecular_resident_weights(
        self,
        task: typing.Any,
        owner_offset: typing.Any,
        points_per_atom: typing.Any,
        device_weights: typing.Any,
        host_weights: typing.Any,
        device_raw: typing.Any,
        host_raw: typing.Any,
        external_device: typing.Any,
        external_stride: typing.Any,
        external_offset: typing.Any = 0,
    ) -> None:
        """Borrow resident partition weights and resident nonlocal force seeds."""
        view = task.view
        if task._owner.device_id != self.device:
            raise ValueError("stationary/grid current owner device mismatch")
        if (
            type(owner_offset) is not int
            or owner_offset < 0
            or type(points_per_atom) is not int
            or points_per_atom <= 0
        ):
            raise ValueError("invalid molecular-grid owner interval")
        self.borrowed_streams.add(view.stream)
        device_weights = self._device_pointer(
            device_weights, "resident molecular-grid weights"
        )
        device_raw = self._device_pointer(
            device_raw, "resident molecular-grid atomic measures"
        )
        external_device = self._device_pointer(
            external_device, "resident nonlocal seeds"
        )
        if type(external_stride) is not int or type(external_offset) is not int:
            raise TypeError("resident nonlocal seed stride/offset must be integers")
        if (
            external_stride <= 0
            or external_offset < 0
            or external_offset > external_stride
            or view.npoint > external_stride - external_offset
        ):
            raise ValueError("resident nonlocal seed tile exceeds its strided owner")
        if self.profile_device:
            self.geometry_external_device_molecular(
                task,
                owner_offset,
                points_per_atom,
                _checked(host_weights, (view.npoint,)),
                _checked(host_raw, (view.npoint,)),
                external_device,
                external_stride,
                external_offset,
            )
            return
        work = task.density_jets(4)
        self._call(
            "stationary_geometry_external_device_molecular_resident_weights_enqueue",
            self.handle,
            ct.byref(view),
            work,
            owner_offset,
            points_per_atom,
            device_weights,
            ct.cast(device_raw, _DOUBLE),
            external_device,
            external_stride,
            external_offset,
        )

    def geometry_external_device_molecular(
        self,
        task: typing.Any,
        owner_offset: typing.Any,
        points_per_atom: typing.Any,
        weights: typing.Any,
        raw: typing.Any,
        external_device: typing.Any,
        external_stride: typing.Any,
        external_offset: typing.Any = 0,
    ) -> None:
        """Borrow resident nonlocal seeds with implicit atom-major grid owners."""
        view = task.view
        if task._owner.device_id != self.device:
            raise ValueError("stationary/grid current owner device mismatch")
        if (
            type(owner_offset) is not int
            or owner_offset < 0
            or type(points_per_atom) is not int
            or points_per_atom <= 0
        ):
            raise ValueError("invalid molecular-grid owner interval")
        self.borrowed_streams.add(view.stream)
        weights = _checked(weights, (view.npoint,))
        raw = _checked(raw, (view.npoint,))
        if type(external_stride) is not int or type(external_offset) is not int:
            raise TypeError("resident nonlocal seed stride/offset must be integers")
        if (
            external_stride <= 0
            or external_offset < 0
            or external_offset > external_stride
            or view.npoint > external_stride - external_offset
        ):
            raise ValueError("resident nonlocal seed tile exceeds its strided owner")
        if isinstance(external_device, int):
            external_device = ct.c_void_p(external_device)
        elif not isinstance(external_device, ct.c_void_p):
            try:
                external_device = ct.cast(external_device, ct.c_void_p)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "resident nonlocal seeds require a device pointer"
                ) from error
        if not external_device.value:
            raise ValueError("resident nonlocal seed device pointer is null")
        if self.profile_device:
            owners = (
                np.arange(owner_offset, owner_offset + view.npoint, dtype=np.int64)
                // points_per_atom
            )
            self.geometry_external_device(
                task,
                owners,
                weights,
                raw,
                external_device,
                external_stride,
                external_offset,
            )
            return
        work = task.density_jets(4)
        self._call(
            "stationary_geometry_external_device_molecular_enqueue",
            self.handle,
            ct.byref(view),
            work,
            owner_offset,
            points_per_atom,
            _ptr(weights),
            _ptr(raw),
            external_device,
            external_stride,
            external_offset,
        )

    def geometry_external_device(
        self,
        task: typing.Any,
        owners: typing.Any,
        weights: typing.Any,
        raw: typing.Any,
        external_device: typing.Any,
        external_stride: typing.Any,
        external_offset: typing.Any = 0,
    ) -> None:
        """Borrow one strided tile from a resident [6, stride] nonlocal seed owner.

        The seed allocation must belong to the same CUDA device and remain alive
        until drain_geometry() when production uses the deferred entry point.
        No seed values cross the host boundary here; the device kernel validates
        all six seed fields before consuming them.
        """
        view = task.view
        if task._owner.device_id != self.device:
            raise ValueError("stationary/grid current owner device mismatch")
        self.borrowed_streams.add(view.stream)
        owners = _checked(owners, (view.npoint,), np.int64)
        weights = _checked(weights, (view.npoint,))
        raw = _checked(raw, (view.npoint,))
        if type(external_stride) is not int or type(external_offset) is not int:
            raise TypeError("resident nonlocal seed stride/offset must be integers")
        if (
            external_stride <= 0
            or external_offset < 0
            or external_offset > external_stride
            or view.npoint > external_stride - external_offset
        ):
            raise ValueError("resident nonlocal seed tile exceeds its strided owner")
        if isinstance(external_device, int):
            external_device = ct.c_void_p(external_device)
        elif not isinstance(external_device, ct.c_void_p):
            try:
                external_device = ct.cast(external_device, ct.c_void_p)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    "resident nonlocal seeds require a device pointer"
                ) from error
        if not external_device.value:
            raise ValueError("resident nonlocal seed device pointer is null")
        work = task.density_jets(4)
        self._call(
            (
                "stationary_geometry_external_device"
                if self.profile_device
                else "stationary_geometry_external_device_enqueue"
            ),
            self.handle,
            ct.byref(view),
            work,
            _ptr(owners),
            _ptr(weights),
            _ptr(raw),
            external_device,
            external_stride,
            external_offset,
        )

    def drain_geometry(self) -> None:
        """Complete all queued semilocal geometry tiles with one error/sync gate."""
        self._call("stationary_geometry_drain", self.handle)

    def finish(self) -> typing.Any:
        self.flush()
        out = np.empty((len(self.source_names), self.natom, 3))
        self._call("stationary_finish", self.handle, _ptr(out), out.size)
        return {name: out[i] for i, name in enumerate(self.source_names)}

    def finish_span(self, names: typing.Iterable[str]) -> typing.Any:
        """Publish one contiguous source interval without downloading unused sources."""
        self.flush()
        names = tuple(names)
        if not names:
            raise ValueError("stationary source span must not be empty")
        if len(names) != len(set(names)):
            raise ValueError("stationary source span contains duplicate names")
        try:
            indices = tuple(self.source_names.index(name) for name in names)
        except ValueError as error:
            raise ValueError(
                "stationary source span contains an unknown source"
            ) from error
        start = indices[0]
        if indices != tuple(range(start, start + len(names))):
            raise ValueError(
                "stationary source span must be contiguous and in runtime order"
            )
        out = np.empty((len(names), self.natom, 3))
        self._call(
            "stationary_finish_span",
            self.handle,
            start,
            len(names),
            _ptr(out),
            out.size,
        )
        return dict(zip(names, out, strict=True))

    def reduced(self) -> typing.Any:
        """Return the complete plan-ordered all-electron sum reduced on CUDA."""
        out = np.empty((self.natom, 3))
        self._call("stationary_finish_reduced", self.handle, _ptr(out), out.size)
        return out

    def metrics(self) -> typing.Any:
        values = (ct.c_uint64 * 24)()
        if self.library.stationary_metrics(self.handle, values, 24):
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
                    "geometry_lane_capacity",
                    "geometry_threads",
                    "geometry_scratch_bytes",
                    "geometry_peak_lanes",
                    "center_geometry_bytes",
                    "center_distance_evaluations",
                    "center_geometry_preparations",
                    "becke_threads_per_point",
                    "becke_shared_bytes",
                    "becke_pair_state_evaluations",
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
        "schema": "generativeqc.native-grid-aot.v1",
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


def _unique_prepared_artifacts(
    artifacts: tuple[typing.Any, ...],
) -> tuple[typing.Any, ...]:
    """Bind shared compiled kernels once while keeping each tensor slot separate."""
    unique: dict[str, typing.Any] = {}
    for artifact in artifacts:
        key = artifact.metadata["key"]
        previous = unique.get(key)
        if previous is not None:
            if previous.metadata["binary_sha256"] != artifact.metadata["binary_sha256"]:
                raise ValueError(
                    "stationary CUDA artifact key has conflicting binaries"
                )
            continue
        unique[key] = artifact
    return tuple(unique.values())


class PreparedStationaryCudaTopologyMismatch(ValueError):
    """Retained execution is incompatible with the requested scientific topology."""


class PreparedStationaryCudaExecution:
    """Retain method-owned CUDA resources behind the shared prepared-region lease."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stack: ExitStack | None = None
        self._lease = PreparedExecutionLease()
        self.preparation_seconds = 0.0
        self._resident_ao_maps = None
        self._resident_ao_map_key = None

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
        page_work_budget: int,
        resident_ao_cutoff: float | None = None,
        resident_ao_cache_bytes: int = 0,
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
                "primitive_page_work_budget": page_work_budget,
                "grid_allocation_bytes": grid_plan.allocation_bytes,
                "resident_ao_cutoff": resident_ao_cutoff,
                "resident_ao_cache_bytes": resident_ao_cache_bytes,
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
        functional: int,
        ecp: bool,
        device: int,
        spec: typing.Any,
        grid_plan: typing.Any,
        source_bytes: int,
        tile_points: int,
        primitive_tile: int,
        integral_terms: int,
        page_work_budget: int,
        max_device_bytes: int,
        max_host_bytes: int,
        host_bound: int,
        profile_device: bool = False,
        resident_ao_cutoff: float | None = None,
        resident_ao_cache_bytes: int = 0,
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
            page_work_budget=page_work_budget,
            resident_ao_cutoff=resident_ao_cutoff,
            resident_ao_cache_bytes=resident_ao_cache_bytes,
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
                self._resident_ao_maps = None
                self._resident_ao_map_key = None
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
        _, _, expansions, _ = _layout(basis)
        component_mode = _component_mode(expansions)
        stationary_artifact = (
            compile_stationary_cuda(
                (
                    tuple(
                        source
                        for _, source in derivative_cuda_sources(
                            _component_domain(expansions)
                        )
                    )
                    if component_mode
                    else emit_first_derivative_cuda(requests)
                ),
                functional=functional,
                plan=plan,
                iterations=spec.partition_iterations,
                compiler=compiler,
                cache=cache,
                primitive_shard_width=(
                    CUDA_REQUESTS_PER_UNIT if component_mode else None
                ),
            )
            if aot_directory is None or ecp
            else load_stationary_aot_artifact(
                aot_directory,
                functional=functional,
                spin=contract.spin,
                plan=plan,
                architecture=target.architecture,
                iterations=spec.partition_iterations,
                component_domain=(QUALIFIED_SPD_COMPONENTS if component_mode else None),
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
            source_names = stationary_runtime_sources(plan)
            needs_first = functional != 0
            # Register the borrowed grid owner first so ExitStack closes the
            # stationary consumer before destroying the CUDA stream it borrows.
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
                    source_names=source_names,
                    target=target,
                    page_work_budget=page_work_budget,
                    profile_device=profile_device,
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
        try:
            artifacts = _unique_prepared_artifacts(
                (
                    stationary_artifact,
                    grid_artifact,
                    *(tensor_artifacts[name] for name in sorted(tensor_artifacts)),
                )
            )
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
        self.compiled_execution_identity = CompiledExecutionIdentity.from_payloads(
            owner="stationary-cuda",
            request={
                "plan": plan.identity,
                "method": state.identity.method,
                "contract": {
                    "family": contract.family,
                    "spin": contract.spin,
                    "ecp": ecp,
                },
                "basis_topology": _basis_topology_identity(basis),
                "source_backend": state._source.backend,
                "functional": state.identity.functional_identity,
                "regularization": state.identity.regularization_identity,
                "grid": asdict(spec),
                "schedule": {
                    "tile_points": tile_points,
                    "primitive_tile": primitive_tile,
                    "integral_terms": integral_terms,
                    "primitive_page_work_budget": page_work_budget,
                    "grid_allocation_bytes": grid_plan.allocation_bytes,
                    "resident_ao_cutoff": resident_ao_cutoff,
                    "resident_ao_cache_bytes": resident_ao_cache_bytes,
                    "geometry_resources": asdict(sources.resources),
                },
                "tensor_plans": tuple(
                    (name, tensor_plans[name].identity) for name in sorted(tensor_plans)
                ),
            },
            artifacts=tuple(
                {
                    "key": artifact.metadata["key"],
                    "binary_sha256": artifact.metadata["binary_sha256"],
                }
                for artifact in self.artifacts
            ),
            runtime={
                "device": device,
                "target": target.to_payload(),
            },
        )

    def close(self) -> None:
        with self._lock:
            self._resident_ao_maps = None
            self._resident_ao_map_key = None
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
        "center_distance_evaluations",
        "center_geometry_preparations",
        "becke_pair_state_evaluations",
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


def _stationary_ao_map_reserve(
    cutoff: float | None, requested_bytes: int, host_bound: int, max_host_bytes: int
) -> int:
    """Admit optional mask storage without stealing the dense fallback's budget.

    The cache includes its producer's transient numeric staging. Exhausting this
    additional allowance selects full AO tiles, not an unbudgeted discovery or a
    different scientific cutoff. An explicit zero allowance is a useful control.
    """
    if cutoff is not None and (
        type(cutoff) not in (int, float) or not np.isfinite(cutoff) or cutoff <= 0
    ):
        raise ValueError("resident AO cutoff must be finite and positive")
    if type(requested_bytes) is not int or not 0 <= requested_bytes <= 1 << 40:
        raise ValueError("resident AO cache budget must be an integer in [0,2**40]")
    if host_bound > max_host_bytes:
        raise ValueError("stationary additional-host byte budget exceeded")
    return 0 if cutoff is None else min(requested_bytes, max_host_bytes - host_bound)


def _stationary_resident_ao_cache(
    prepared: PreparedStationaryCudaExecution | None,
    grid: typing.Any,
    state: typing.Any,
    resident: typing.Any,
    *,
    cutoff: float | None,
    budget_bytes: int,
) -> typing.Any:
    """Bind optional force masks to the current token-checked resident grid.

    Density generations are deliberately not part of this key: AO jets depend
    only on basis/geometry/points. Native snapshots still validate their current
    owner before every endpoint. The grid's center/basis generations prevent
    address reuse or a failed prepared execution from reviving stale masks.
    """
    if cutoff is None or resident is None:
        if prepared is not None:
            prepared._resident_ao_maps = None
            prepared._resident_ao_map_key = None
        return None
    from ._resident_ao_maps import ResidentAoMapCache, ResidentAoMapDomain

    state._source.check_current()
    domain = ResidentAoMapDomain(
        basis_identity=grid.basis_identity,
        geometry_identity=state.identity.geometry_identity,
        grid_identity=state.identity.grid_identity,
        device=resident.device,
        point_pointer=resident.points,
        point_count=resident.point_count,
        tile_points=grid.plan.tile_points,
        derivative_order=grid.plan.order,
    )
    key = (
        domain,
        id(grid),
        grid.geometry_generation,
        grid.basis_generation,
        float(cutoff),
        budget_bytes,
    )
    owner = None if prepared is None else prepared._resident_ao_maps
    if owner is None or prepared._resident_ao_map_key != key:
        # Release the old maps before constructing new ones: both geometries
        # must never coexist under an allowance sized for just one cache.
        if prepared is not None:
            prepared._resident_ao_maps = None
            prepared._resident_ao_map_key = None
        owner = None
        owner = ResidentAoMapCache(
            grid, domain, cutoff=cutoff, budget_bytes=budget_bytes
        )
        if prepared is not None:
            prepared._resident_ao_maps = owner
            prepared._resident_ao_map_key = key
    owner.reset_work()
    return owner


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
    max_primitive_records: typing.Any = _DEFAULT_MAX_PRIMITIVE_RECORDS,
    max_grid_pair_visits: typing.Any = 100_000_000,
    max_pending_grid_tiles: typing.Any = 64,
    max_pending_grid_pair_visits: typing.Any = 100_000_000,
    max_ecp_pair_samples: int = 100_000_000,
    prepared: PreparedStationaryCudaExecution | None = None,
    profile_device: bool = False,
    resident_ao_cutoff: float | None = None,
    resident_ao_cache_bytes: int = 16 << 20,
) -> typing.Any:
    """Consume a current native CUDA RKS/UKS snapshot with every plan source.

    Domain: qualified real FP64 RKS/UKS compositions, native unpruned grid and
    distinct noncolliding centers. Enlarged domains require the shared native
    stationary integral owner. No CPKS is required. None for either whole-grid
    work guard admits the finite complete grid, retaining bounded submission
    windows and the unchanged host/device byte budgets.
    Device ordinal comes only from the opaque native snapshot. CUDA source
    accumulators, grid owner and one TensorIR consumer coexist under the stated
    additional-device budget; the pre-existing SCF owner/export and Python
    objects are reported separately. No claim of full device residency is made.
    Compilation and the selected GPU allocation are explicit.
    Scalar-ECP v5 adds generated CUDA local/nonlocal derivatives and effective
    charges (nine sources). Its small dense export is separately budgeted and
    preserves the checked native two-grid gate. The public wrapper restricts ECP
    force capability to Cartesian/real-spherical s/p records.
    Resident AO selection is experimental and disabled unless a sampled-jet
    cutoff is explicitly supplied. It changes only force-grid AO membership;
    it does not screen SCF, shrink full-capacity arenas, or prune grid points.
    """
    timeline = _ExclusiveWallTimeline()
    contract = StationaryDerivativeContract(state.identity)
    contract.validate(state)
    if state._source.backend != "cuda":
        raise NotImplementedError("CUDA diagnostic requires a native CUDA KS state")
    if state._source.metadata[0] not in (3, 5, 8) or state._source.grid_spec is None:
        raise NotImplementedError(
            "CUDA diagnostic requires supported raw-measure/composition snapshots"
        )
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
        (max_primitive_records, "max_primitive_records", 1 << 40),
        (max_ecp_pair_samples, "max_ecp_pair_samples", 1 << 40),
    ):
        if type(value) is not int or not 1 <= value <= cap:
            raise ValueError(f"{name} must be an integer in [1,{cap}]")
    na, n = basis.natom, basis.nao
    requires_native_integrals = stationary_cuda_requires_native_integrals(
        atoms=na, aos=n, primitives=basis.nprimitive
    )
    grid_work = plan_stationary_cuda_grid_work(
        atoms=na,
        grid_points=len(state.grid.points),
        tile_points=tile_points,
        max_grid_points=max_grid_points,
        max_grid_pair_visits=max_grid_pair_visits,
        max_pending_tiles=max_pending_grid_tiles,
        max_pending_pair_visits=max_pending_grid_pair_visits,
    )
    if requires_native_integrals and (
        ecp
        or not callable(
            getattr(
                state._source,
                "density_fitted_integral_derivatives"
                if bool(getattr(state._source, "density_fitted", False))
                else "cuda_integral_derivatives",
                None,
            )
        )
    ):
        raise NotImplementedError(
            "enlarged stationary CUDA domains require prepared native integral derivatives"
        )
    _, aos, expansions, requests = _layout(basis)
    component_mode = _component_mode(expansions)
    primitive_sum = sum(
        int(row[2]) * len(expansion)
        for row, expansion in zip(aos, expansions, strict=True)
    )
    has_exchange = bool(state._source.method_ir.full_range_exact_exchange)
    records = (
        (1 + int(has_exchange)) * primitive_sum**4
        + (na + 2) * primitive_sum**2
        + na * (na - 1) // 2
    )
    # Keep the frozen whole-force capacity expression above intact for
    # admission/evidence tooling. Production removes this AO^4 contribution
    # from executed primitive work only after the prepared shell source succeeds.
    ao_quartet_primitive_records = (1 + int(has_exchange)) * primitive_sum**4
    ao_pair_primitive_records = (na + 2) * primitive_sum**2
    ao_integral_primitive_records = (
        ao_quartet_primitive_records + ao_pair_primitive_records
    )
    if records > np.iinfo(np.uint64).max:
        raise ValueError("primitive work count exceeds uint64 metric range")
    pair_visits = grid_work.grid_pair_visits
    # Preserve the actual composition; the public selector is only a label.
    method = state._source.method_ir
    plan = StationaryGradientPlan(
        method,
        StationaryMeanField(
            SCF_POINT_MODEL,
            hamiltonian="scalar-semilocal-ecp" if ecp else "all-electron",
        ),
    )
    source_names = stationary_runtime_sources(plan)
    density = state.density if contract.spin == "polarized" else state.density[0]
    if (
        tuple(s for s in plan.source_names if s not in ("ecp_local", "ecp_nonlocal"))
        != source_names
    ):
        raise ValueError(
            "CUDA runtime source coverage differs from StationaryGradientPlan"
        )
    functional = int(state._source.metadata[6])
    if functional not in _REGISTERED_STATIONARY_CODES:
        raise ValueError("native snapshot reported an unknown semilocal functional")
    record = SEMILOCAL_FAMILY_BY_CODE.get(functional)
    if ecp and record is not None and not record["stationary_ecp_gradient"]:
        raise NotImplementedError(
            f"{record['name']} CUDA stationary gradients do not inherit ECP support"
        )
    ingredients = state._source.functional.ingredients
    needs_first = "sigma" in ingredients
    device = int(state._source.metadata[12])
    grid_plan = plan_tiles(
        basis,
        backend="cuda",
        order=2 if needs_first else 1,
        tile_points=tile_points,
        active_ao_capacity=n,
        budget_bytes=max_device_bytes,
    )
    # Admit the bounded fallback first. Optional lane expansion uses only the
    # space left after every other retained force owner has been planned.
    minimum_source_bytes = stationary_cuda_allocation_bytes(
        atoms=na,
        aos=n,
        primitives=basis.nprimitive,
        points=tile_points,
        tasks=primitive_tile,
        spins=plan.spin_blocks,
        sources=len(source_names),
        geometry_lanes=min(32, tile_points),
    )
    available = max_device_bytes - grid_plan.peak_bytes - minimum_source_bytes
    if available <= 0:
        raise ValueError("stationary additional-device budget exceeded")
    tensor_plans = {}
    if ecp:
        tensor_plans["reduction"] = plan_cuda(
            plan.reduction_program(atoms=na), target, max_bytes=available
        )
    # Conservative numeric-array bound: compact task pages/sort staging, resident
    # topology mirrors, D/W admission copies, adapter staging, cached Cartesian
    # derivative-kind lookup tables, candidate/publication copies, and tile owners.
    # Compiler objects, Python headers and the caller's existing SCF snapshot
    # are explicit exclusions, as in the reused grid/TensorIR resource contracts.
    host_bound = (
        grid_plan.host_bytes
        + 8
        * (
            34 * primitive_tile
            + 4 * plan.spin_blocks * n * n
            + 120 * na
            + 12 * (len(source_names) - len(_SOURCE_NAMES)) * na
            + 26 * integral_terms
            + len(COMPONENT_LABELS) ** 4
            + 3 * len(COMPONENT_LABELS) ** 2
            + 3 * tile_points
            + 2 * basis.nprimitive
            + 4 * n
            + 80
        )
        + max((tp.host_bytes for tp in tensor_plans.values()), default=0)
    )
    # The shared Direct provider can use its explicit paired one-electron
    # fallback while these force owners coexist. Charge that host staging as
    # well as reserving its device allowance below; incidental grid-plan slack
    # is not a concurrent-owner resource contract. Retained SCF/snapshot storage
    # remains outside this additional-consumer bound.
    native_integral_host_reserve = (
        stationary_native_pair_reserve(atoms=na, aos=n, primitives=basis.nprimitive)
        if not ecp
        and not bool(getattr(state._source, "density_fitted", False))
        and callable(getattr(state._source, "cuda_integral_derivatives", None))
        else 0
    )
    host_bound += native_integral_host_reserve
    if host_bound > max_host_bytes:
        raise ValueError("stationary additional-host byte budget exceeded")
    ecp_workspace = ecp_pair_samples = 0
    if ecp:
        from generativeqc_compiler.integral.ecp_policy import (
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
    # Lane expansion is optional: retain the old native provider allowance so
    # a tighter geometry budget cannot disable or OOM an already-admitted
    # prepared integral path. If the old remainder was itself too small, keep
    # all of it and leave that existing provider decision unchanged. The fitted
    # provider has a separate resource contract, so preserve its full allowance
    # rather than borrowing the Direct provider's one-electron estimate.
    native_geometry_reserve = (
        max(0, available - sum(value.peak_bytes for value in tensor_plans.values()))
        if not ecp and bool(getattr(state._source, "density_fitted", False))
        else min(
            max(
                0, available - sum(value.peak_bytes for value in tensor_plans.values())
            ),
            stationary_native_pair_reserve(
                atoms=na, aos=n, primitives=basis.nprimitive
            ),
        )
        if not ecp
        and callable(getattr(state._source, "cuda_integral_derivatives", None))
        else 0
    )
    source_resources = plan_stationary_cuda_resources(
        atoms=na,
        aos=n,
        primitives=basis.nprimitive,
        points=tile_points,
        tasks=primitive_tile,
        spins=plan.spin_blocks,
        sources=len(source_names),
        target=target,
        budget_bytes=max_device_bytes
        - grid_plan.peak_bytes
        - sum(value.peak_bytes for value in tensor_plans.values())
        - native_geometry_reserve,
    )
    source_bytes = source_resources.allocation_bytes
    ao_map_reserve = _stationary_ao_map_reserve(
        resident_ao_cutoff,
        resident_ao_cache_bytes,
        host_bound
        + (
            sum(value.host_bytes for value in tensor_plans.values())
            if prepared is not None
            else 0
        ),
        max_host_bytes,
    )
    host_bound += ao_map_reserve
    cache = Path(cache)
    spec = state._source.grid_spec
    if prepared is None:
        with timeline.phase("artifact_lookup_compile"):
            artifact = (
                compile_stationary_cuda(
                    (
                        tuple(
                            source
                            for _, source in derivative_cuda_sources(
                                _component_domain(expansions)
                            )
                        )
                        if component_mode
                        else emit_first_derivative_cuda(requests)
                    ),
                    functional=functional,
                    plan=plan,
                    iterations=spec.partition_iterations,
                    compiler=compiler,
                    cache=cache,
                    primitive_shard_width=(
                        CUDA_REQUESTS_PER_UNIT if component_mode else None
                    ),
                )
                if aot_directory is None or ecp
                else load_stationary_aot_artifact(
                    aot_directory,
                    functional=functional,
                    spin=contract.spin,
                    plan=plan,
                    architecture=target.architecture,
                    iterations=spec.partition_iterations,
                    component_domain=(
                        QUALIFIED_SPD_COMPONENTS if component_mode else None
                    ),
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
                functional=functional,
                ecp=ecp,
                device=device,
                spec=spec,
                grid_plan=grid_plan,
                source_bytes=source_bytes,
                tile_points=tile_points,
                primitive_tile=primitive_tile,
                integral_terms=integral_terms,
                page_work_budget=max_primitive_records,
                max_device_bytes=max_device_bytes,
                max_host_bytes=max_host_bytes,
                host_bound=host_bound,
                profile_device=profile_device,
                resident_ao_cutoff=resident_ao_cutoff,
                resident_ao_cache_bytes=ao_map_reserve,
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
                # ExitStack unwinds in reverse: keep the borrowed grid stream
                # alive until the stationary consumer has drained and closed.
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
                        source_names=source_names,
                        target=target,
                        page_work_budget=max_primitive_records,
                        timeline=timeline,
                        profile_device=profile_device,
                    )
                )
            source_before = grid_before = None
        else:
            sources, ao = prepared.sources, prepared.grid
            sources.timeline = timeline
            with timeline.phase("metrics_collection"):
                source_before, grid_before = sources.metrics(), ao.metrics()
        native_integral_components = None
        native_integral_resources: typing.Mapping[str, int] = MappingProxyType({})
        fitted_integral_provider = getattr(
            state._source, "density_fitted_integral_derivatives", None
        )
        direct_integral_provider = getattr(
            state._source, "cuda_integral_derivatives", None
        )
        use_fitted_integrals = bool(getattr(state._source, "density_fitted", False))
        integral_provider = (
            fitted_integral_provider
            if use_fitted_integrals
            else direct_integral_provider
        )
        native_integral_budget = max_device_bytes - peak
        if not ecp and native_integral_budget > 0 and callable(integral_provider):
            with timeline.phase("prepared_stationary_integral_derivatives"):
                native_integral = (
                    integral_provider(na, native_integral_budget)
                    if use_fitted_integrals
                    else integral_provider(
                        na,
                        native_integral_budget,
                        range_exchange=False,
                    )
                )
            if native_integral is not None:
                native_integral_components, native_integral_resources = native_integral
                native_integral_components = np.asarray(native_integral_components)
                if (
                    native_integral_components.shape != (4, na, 3)
                    or not np.isfinite(native_integral_components).all()
                ):
                    raise RuntimeError(
                        "prepared stationary integral source returned invalid output"
                    )
        if use_fitted_integrals and native_integral_components is None:
            raise NotImplementedError(
                "density-fitted stationary derivative provider is unavailable; "
                "Direct derivative fallback would change the Hamiltonian"
            )
        native_complete_integrals = native_integral_components is not None
        if (
            not use_fitted_integrals
            and int(native_integral_resources.get("one_electron_host_peak_bytes", 0))
            > native_integral_host_reserve
        ):
            raise RuntimeError(
                "native stationary host staging exceeds admitted reserve"
            )
        if requires_native_integrals and not native_complete_integrals:
            raise NotImplementedError(
                "prepared native integral derivatives are unavailable within the admitted "
                "budget; enlarged stationary CUDA domains cannot use AO-task fallback"
            )
        resident_grid_density = None
        if native_complete_integrals:
            resident_provider = getattr(state._source, "cuda_resident_density", None)
            if callable(resident_provider):
                resident_grid_density = resident_provider()
        with timeline.phase("owner_state_reset"):
            if native_complete_integrals:
                sources.reset_geometry(spec.coincident_tolerance)
                if resident_grid_density is None:
                    ao.set_density(density)
                else:
                    ao.set_density_device(
                        device_id=resident_grid_density.device,
                        alpha=resident_grid_density.alpha,
                        beta=resident_grid_density.beta,
                        matrix_elements=resident_grid_density.matrix_elements,
                        spins=resident_grid_density.spins,
                        source_stream=resident_grid_density.source_stream,
                    )
                    state._source.check_current()
            else:
                sources.reset(
                    spec.coincident_tolerance, state.density, state.weighted_density
                )
                ao.set_density(density)
        shell_full_range = None
        shell_provider = getattr(state._source, "cuda_full_range_derivatives", None)
        if not native_complete_integrals and not ecp and callable(shell_provider):
            with timeline.phase("direct_shell_integral_derivatives"):
                shell_full_range = shell_provider(na)
        native_shell_full_range = shell_full_range is not None
        if native_complete_integrals:
            records -= ao_integral_primitive_records
        elif native_shell_full_range:
            shell_full_range = np.asarray(shell_full_range)
            if (
                shell_full_range.shape != (2, na, 3)
                or not np.isfinite(shell_full_range).all()
            ):
                raise RuntimeError(
                    "prepared Direct shell derivative source returned invalid output"
                )
            records -= ao_quartet_primitive_records
        timeline.switch("python_packing")
        # integral_terms and primitive_tile are admitted independently. A fixed
        # producer must fit both the logical fixed threshold and the resident
        # native descriptor reservoir.
        fixed_task_capacity = min(integral_terms, primitive_tile)
        task_executor = _BoundedStationaryTaskExecutor(
            fixed_capacity=fixed_task_capacity,
            resident_capacity=primitive_tile,
            page_capacity=primitive_tile,
        )
        task_executions: list[dict[str, typing.Any]] = []
        task_sources = (
            ()
            if native_complete_integrals
            else (
                ("one_electron", 2, "kinetic"),
                ("overlap_pulay", 2, "overlap"),
                *(
                    ()
                    if native_shell_full_range
                    else (
                        ("coulomb", 4, "four_center_eri"),
                        *(
                            (("exact_exchange", 4, "four_center_eri"),)
                            if has_exchange
                            else ()
                        ),
                    )
                ),
            )
        )
        for source, rank, operator in task_sources:
            domain = RuntimeTaskDomain.rectangular((n,) * rank)
            source_index = source_names.index(source)

            def submit_page(
                page: RuntimeTaskPage,
                *,
                _source_index: int = source_index,
                _operator: str = operator,
                _source: str = source,
            ) -> None:
                sources.integral_page(
                    _source_index,
                    _operator,
                    page.coordinates,
                )
                if _source == "one_electron":
                    for atom in range(na):
                        sources.integral_page(
                            0,
                            "nuclear_attraction",
                            page.coordinates,
                            atom,
                            charges[atom],
                        )

            execution = task_executor.execute_pages(domain, submit_page)
            task_executions.append(
                {
                    "source": source,
                    "rank": rank,
                    **asdict(execution),
                }
            )
            sources.flush()
        for atom in range(na):
            for other in range(atom):
                sources.nuclear(atom, other, charges)
        sources.flush()
        grid = state.grid
        grid_points = len(grid.points)
        resident_grid_provider = getattr(state._source, "cuda_resident_grid", None)
        resident_grid = (
            resident_grid_provider() if callable(resident_grid_provider) else None
        )
        if resident_grid is not None:
            if (
                resident_grid.device != device
                or resident_grid.point_count != grid_points
            ):
                raise ValueError(
                    "resident molecular-grid lease differs from stationary state"
                )
            points_per_atom = (
                spec.radial_points * spec.angular_polar * spec.angular_azimuth
            )
            if points_per_atom <= 0 or grid_points != na * points_per_atom:
                raise ValueError(
                    "resident molecular grid is not the expected atom-major GridSpec"
                )
            if profile_device:
                grid_residency = {
                    "grid_owner_source": "profile-host-materialized-atom-major-index",
                    "grid_owner_h2d_bytes": grid_points * 8,
                    "grid_point_source": "exact-native-resident-grid",
                    "grid_point_h2d_bytes": 0,
                    "grid_weight_source": "profile-host-snapshot",
                    "grid_weight_h2d_bytes": grid_points * 8,
                    "grid_atomic_measure_source": "profile-host-snapshot",
                    "grid_atomic_measure_h2d_bytes": grid_points * 8,
                }
            else:
                grid_residency = {
                    "grid_owner_source": "implicit-atom-major-index",
                    "grid_owner_h2d_bytes": 0,
                    "grid_point_source": "exact-native-resident-grid",
                    "grid_point_h2d_bytes": 0,
                    "grid_weight_source": "exact-native-resident-grid",
                    "grid_weight_h2d_bytes": 0,
                    "grid_atomic_measure_source": "exact-native-resident-grid",
                    "grid_atomic_measure_h2d_bytes": 0,
                }
        else:
            points_per_atom = 0
            grid_residency = {
                "grid_owner_source": "host-grid-owners",
                "grid_owner_h2d_bytes": grid_points * 8,
                "grid_point_source": "host-grid-points",
                "grid_point_h2d_bytes": 3 * grid_points * 8,
                "grid_weight_source": "host-grid-weights",
                "grid_weight_h2d_bytes": grid_points * 8,
                "grid_atomic_measure_source": "host-grid-atomic-measures",
                "grid_atomic_measure_h2d_bytes": grid_points * 8,
            }
        ao_maps = _stationary_resident_ao_cache(
            prepared,
            ao,
            state,
            resident_grid,
            cutoff=resident_ao_cutoff,
            budget_bytes=ao_map_reserve,
        )
        for chunk_begin, chunk_end in grid_work.chunks():
            with timeline.phase("xc_geometry_enqueue"):
                for begin in range(chunk_begin, chunk_end, tile_points):
                    end = min(begin + tile_points, chunk_end)
                    if resident_grid is not None:
                        point_pointer = resident_grid.points + 3 * begin * 8
                        selected_ao_ids = (
                            None
                            if ao_maps is None
                            else ao_maps.select(ao, ao_maps.domain, begin, end - begin)
                        )
                        with ao.feature_task_device_points(
                            point_pointer,
                            end - begin,
                            selected_ao_ids,
                            ingredients,
                        ) as task:
                            sources.geometry_molecular_resident_weights(
                                task,
                                begin,
                                points_per_atom,
                                resident_grid.weights + begin * 8,
                                grid.weights[begin:end] if profile_device else None,
                                resident_grid.atomic_weights + begin * 8,
                                (
                                    state._source.atomic_weights[begin:end]
                                    if profile_device
                                    else None
                                ),
                                functional=functional,
                            )
                    else:
                        with ao.feature_task(
                            grid.points[begin:end],
                            None,
                            ingredients,
                            defer_error_to_consumer=True,
                        ) as task:
                            sources.geometry(
                                task,
                                np.asarray(grid.owners[begin:end], dtype=np.int64),
                                grid.weights[begin:end],
                                state._source.atomic_weights[begin:end],
                                functional=functional,
                            )
            with timeline.phase("xc_geometry_drain"):
                sources.drain_geometry()
        with timeline.phase("source_d2h_publication"):
            components = sources.finish()
        if native_complete_integrals:
            components["one_electron"] = np.ascontiguousarray(
                native_integral_components[0]
            )
            components["overlap_pulay"] = np.ascontiguousarray(
                native_integral_components[1]
            )
            components["coulomb"] = np.ascontiguousarray(native_integral_components[2])
            if has_exchange:
                components["exact_exchange"] = np.ascontiguousarray(
                    native_integral_components[3]
                )
            elif np.any(native_integral_components[3] != 0):
                raise RuntimeError(
                    "semilocal prepared stationary source published unexpected K"
                )
        elif native_shell_full_range:
            components["coulomb"] = np.ascontiguousarray(shell_full_range[0])
            if has_exchange:
                components["exact_exchange"] = np.ascontiguousarray(shell_full_range[1])
            elif np.any(shell_full_range[1] != 0):
                raise RuntimeError(
                    "semilocal Direct shell derivative published unexpected K"
                )
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
        # plan-owned work reduces inside the stationary owner; ECP retains the
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
                if native_complete_integrals:
                    gradient = (
                        gradient
                        + components["one_electron"]
                        + components["overlap_pulay"]
                        + components["coulomb"]
                    )
                    if has_exchange:
                        gradient = gradient + components["exact_exchange"]
                elif native_shell_full_range:
                    gradient = gradient + components["coulomb"]
                    if has_exchange:
                        gradient = gradient + components["exact_exchange"]
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
            work.update(grid_residency)
            work["primitive_pages"] = sources.primitive_pages
            work["primitive_page_peak_records"] = sources.primitive_page_peak_records
            work["primitive_record_page_budget"] = max_primitive_records
            work["bulk_pack_chunks"] = sources.bulk_pack_chunks
            work["bulk_packed_descriptors"] = sources.bulk_packed_descriptors
            work["scalar_packed_descriptors"] = sources.scalar_packed_descriptors
        # A typed cache-allocation failure may retain the exact direct arena.
        # Keep admission/peak accounting conservative, but verify actual storage.
        actual_center_bytes = work["center_geometry_bytes"]
        planned_center_bytes = source_resources.center_geometry_bytes
        if (
            actual_center_bytes not in (0, planned_center_bytes)
            or work["owned_device_bytes"]
            != source_bytes - planned_center_bytes + actual_center_bytes
        ):
            raise RuntimeError("stationary allocation disagrees with admitted bytes")
        timeline.switch("owner_cleanup")
    timeline.switch("publication_validation")
    contract.validate(state)  # Replay/replacement/closure revokes publication.
    if (
        work["primitive_records"] != records
        or work["grid_pair_visits"] != pair_visits
        or work["xc_points"] != grid_work.grid_points
    ):
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
        grid_work_plan={
            "schema": "generativeqc.stationary-grid-work.v1",
            **asdict(grid_work),
        },
        resident_ao_selection={
            "schema": "generativeqc.stationary-resident-ao-selection.v1",
            "mode": (
                "disabled"
                if resident_ao_cutoff is None
                else "dense-no-resident-grid"
                if ao_maps is None
                else "explicit-sampled-jet-cutoff"
            ),
            "cutoff": resident_ao_cutoff,
            "cache_budget_requested_bytes": resident_ao_cache_bytes,
            "cache_host_reserve_bytes": ao_map_reserve,
            "full_ao_capacity": n,
            "derivative_order": grid_plan.order,
            "work": None if ao_maps is None else ao_maps.work,
        },
        native_integrals_required=requires_native_integrals,
        ordered_pairs=n * n,
        ordered_quartets=(1 + int(has_exchange)) * n**4,
        exchange_ordered_quartets=n**4 if has_exchange else 0,
        full_range_derivative_route=(
            "prepared-native-stationary"
            if native_complete_integrals
            else (
                "prepared-direct-shell"
                if native_shell_full_range
                else "bounded-ao-task"
            )
        ),
        full_range_ao_task_domain_elided=bool(
            native_complete_integrals or native_shell_full_range
        ),
        full_range_shell_sources=(
            ("coulomb", "exact_exchange")
            if (native_complete_integrals or native_shell_full_range) and has_exchange
            else (
                ("coulomb",)
                if native_complete_integrals or native_shell_full_range
                else ()
            )
        ),
        stationary_integral_derivative_route=(
            "prepared-native-complete"
            if native_complete_integrals
            else (
                "prepared-shell-two-electron"
                if native_shell_full_range
                else "bounded-ao-task"
            )
        ),
        stationary_native_integral_sources=(
            (
                "one_electron",
                "overlap_pulay",
                "coulomb",
                *(("exact_exchange",) if has_exchange else ()),
            )
            if native_complete_integrals
            else ()
        ),
        native_integral_resources=dict(native_integral_resources),
        native_integral_host_reserve=native_integral_host_reserve,
        additional_device_peak_bound=peak
        + int(native_integral_resources.get("one_electron_device_peak_bytes", 0)),
        additional_device_budget=max_device_bytes,
        device_ordinal=device,
        tensor_executions=tensor_work["executions"],
        tensor_work=tensor_work,
        stationary_weight_lowering="generated-tensorir/device-pointwise-v1",
        stationary_weight_plan_identity=plan.identity,
        stationary_weight_programs={
            name: plan.integral_block(name, terms=1).weights.logical_hash
            for name in (
                "one_electron",
                "overlap_pulay",
                "coulomb",
                *(("exact_exchange",) if has_exchange else ()),
            )
        },
        stationary_weight_tensor_executions=0,
        stationary_weight_roundtrip_bytes=0,
        stationary_final_reduction=(
            "generated-tensorir-v1"
            if ecp
            else (
                "native-plan-source-device-sum-plus-prepared-integrals-v1"
                if native_complete_integrals
                else (
                    "native-plan-source-device-sum-plus-direct-shell-compose-v1"
                    if native_shell_full_range
                    else (
                        "native-plan-source-device-sum-v1"
                        if has_exchange
                        else "native-seven-source-device-sum-v1"
                    )
                )
            )
        ),
        stationary_state_dw_upload_bytes=(
            0
            if native_complete_integrals
            else state.density.nbytes + state.weighted_density.nbytes
        ),
        grid_density_source=(
            "exact-final-scf-device-binding"
            if resident_grid_density is not None
            else "host-snapshot-density"
        ),
        grid_density_h2d_bytes=(
            0 if resident_grid_density is not None else np.asarray(density).nbytes
        ),
        stationary_task_executor={
            "schema": "generativeqc.stationary-bounded-task-executor.v2",
            "fixed_capacity": fixed_task_capacity,
            "resident_capacity": primitive_tile,
            "page_capacity": primitive_tile,
            "primitive_record_page_budget": max_primitive_records,
            "logical_primitive_records": records,
            "sources": tuple(task_executions),
        },
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
        snapshot_grid_cache_work=dict(state._source.grid_cache_work),
        snapshot_export_work=dict(state._source.export_work),
        snapshot_export="explicit native CUDA final-state export; W/frame validation is host work",
        host_scope=(
            "snapshot validation; bounded geometry scheduling; immutable result copies"
            if native_complete_integrals
            else "snapshot validation; AO task descriptor packing/sorting; one D/W owner upload; final TensorIR reduction; immutable result copies"
        ),
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
    if use_fitted_integrals:
        work["density_fitted_response_resources_included"] = False
        work["native_integral_resource_scope"] = (
            "compact-publication-and-host-one-electron-only"
        )
        work["additional_device_peak_bound_scope"] = (
            "stationary-consumer-only; excludes DF-provider response scratch"
        )
        work["transfer_work"]["density_fitted_response_included"] = False
        work["host_scope"] += "; retained H'/S' source contraction"
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
        execution=(
            "cuda-nine-source"
            if ecp
            else ("cuda-eight-source" if has_exchange else "cuda-seven-source")
        )
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
    max_primitive_records: typing.Any = _DEFAULT_MAX_PRIMITIVE_RECORDS,
    max_grid_pair_visits: typing.Any = 100_000_000,
    max_pending_grid_tiles: typing.Any = 64,
    max_pending_grid_pair_visits: typing.Any = 100_000_000,
    max_ecp_pair_samples: int = 100_000_000,
    prepared: PreparedStationaryCudaExecution | None = None,
    profile_device: bool = False,
    resident_ao_cutoff: float | None = None,
    resident_ao_cache_bytes: int = 16 << 20,
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
        "max_pending_grid_tiles": max_pending_grid_tiles,
        "max_pending_grid_pair_visits": max_pending_grid_pair_visits,
        "max_ecp_pair_samples": max_ecp_pair_samples,
        "prepared": prepared,
        "profile_device": profile_device,
        "resident_ao_cutoff": resident_ao_cutoff,
        "resident_ao_cache_bytes": resident_ao_cache_bytes,
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
