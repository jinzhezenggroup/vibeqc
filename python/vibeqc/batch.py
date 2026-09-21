"""Persistent ragged-batch interface backed by the native fleet plan."""

from __future__ import annotations

import ctypes
import os
import typing
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np

from . import _native
from .calculator import Atom, Calculator, _read_correlation_result
from .ks_diagnostics import (
    KsDiagnostic,
    KsTransportDiagnostic,
    read_ks_diagnostic,
    read_ks_transport_diagnostic,
)

if typing.TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from .accuracy import AccuracyAssessment
    from .calculator import CorrelationResult


@dataclass(frozen=True)
class BatchItemResult:
    index: int
    status: int
    status_message: str
    energy: float
    forces: np.ndarray | None
    converged: bool
    iterations: int
    energy_change: float
    density_rms: float
    executed_backend: str
    bucket_id: int
    warm_start_used: bool
    warm_start_fallback: bool
    basis_metadata: dict = field(default_factory=dict)
    accuracy: AccuracyAssessment | None = None
    restart_origin: str = "cold"
    fock_builds: int | None = None
    # None means this item did not complete a solve, or the library predates the query.
    precision: dict | None = None
    # Physical commutator at the returned density; absent for unsupported methods.
    physical_residual_rms: float | None = None
    ks_diagnostic: KsDiagnostic | None = None
    correlation: CorrelationResult | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == _native.STATUS_SUCCESS


@dataclass(frozen=True)
class BatchResult:
    """Input-ordered results for a ragged batch; forces are never padded."""

    items: tuple[BatchItemResult, ...]

    @property
    def energies(self) -> np.ndarray:
        return np.asarray(
            [item.energy if item.succeeded else np.nan for item in self.items],
            dtype=np.float64,
        )

    @property
    def failure_indices(self) -> tuple[int, ...]:
        return tuple(item.index for item in self.items if not item.succeeded)

    @property
    def succeeded(self) -> bool:
        return not self.failure_indices

    def raise_for_failures(self) -> None:
        failures = [
            f"{item.index}: {item.status_message}"
            for item in self.items
            if not item.succeeded
        ]
        if failures:
            raise RuntimeError("batched item failures: " + "; ".join(failures))


def _decode_triangular_class(index: int) -> tuple[int, int]:
    """Decode the scheduler's triangular high/low canonical class."""

    high = 0
    while (high + 1) * (high + 2) // 2 <= index:
        high += 1
    return high, index - high * (high + 1) // 2


@dataclass(frozen=True)
class ShellClassProfileEntry:
    """Final-density direct work retained for one canonical shell class."""

    shell_class: int
    shell_angular: tuple[int, int, int, int]
    shell_quartets: int
    tiles: int
    ao_quartets: int
    primitive_quartets: int

    @property
    def label(self) -> str:
        """Return the conventional canonical label, for example ``dppp``."""

        angular_labels = "spdf"
        return "".join(angular_labels[value] for value in self.shell_angular)


@dataclass(frozen=True)
class DensityFittingMetricDiagnostic:
    """CUDA DF value/J/K plan evidence; peaks exclude generated-force staging."""

    bucket_id: int
    system_index: int
    effective_rank: int
    absolute_threshold: float
    condition_number: float
    solver_device_workspace_bytes: int
    solver_host_workspace_bytes: int
    device_resident_bytes: int
    peak_device_bytes: int
    host_resident_bytes: int
    peak_host_bytes: int
    auxiliary_tile: int
    streamed: bool

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready diagnostics for benchmark and telemetry clients."""

        return {
            "bucket_id": self.bucket_id,
            "system_index": self.system_index,
            "effective_rank": self.effective_rank,
            "absolute_threshold": self.absolute_threshold,
            "condition_number": self.condition_number,
            "solver_device_workspace_bytes": self.solver_device_workspace_bytes,
            "solver_host_workspace_bytes": self.solver_host_workspace_bytes,
            "device_resident_bytes": self.device_resident_bytes,
            "peak_device_bytes": self.peak_device_bytes,
            "host_resident_bytes": self.host_resident_bytes,
            "peak_host_bytes": self.peak_host_bytes,
            "auxiliary_tile": self.auxiliary_tile,
            "streamed": self.streamed,
        }


@dataclass(frozen=True)
class PppsQueueProfile:
    """Final-density statistics for the exact resident PPPS force queue.

    Block-indexed tuples use 32, 64, 128, then 256 threads. Orientation
    tuples use ``1110`` then ``1011``. Primitive histograms use exact buckets
    0..63 and an overflow bucket at index 64.
    """

    descriptor_slots: int
    non_empty_descriptors: int
    empty_descriptors: int
    tasks: int
    primitive_work: int
    ket_count_min: int
    ket_count_median: int
    ket_count_p90: int
    ket_count_p99: int
    ket_count_max: int
    lane_efficiency: tuple[float, ...]
    primitive_warp_efficiency: float
    task_tail_imbalance: tuple[float, ...]
    primitive_tail_imbalance: tuple[float, ...]
    orientation_tasks: tuple[int, int]
    orientation_primitive_work: tuple[int, int]
    bra_primitive_tasks: tuple[int, ...]
    bra_primitive_work: tuple[int, ...]
    ket_primitive_tasks: tuple[int, ...]
    ket_primitive_work: tuple[int, ...]

    @property
    def hole_rate(self) -> float:
        """Return the fraction of descriptor slots that launch as no-ops."""

        if self.descriptor_slots == 0:
            return 0.0
        return self.empty_descriptors / self.descriptor_slots


@dataclass(frozen=True)
class EigensolverDiagnostic:
    """Setup-time eigensolver selection and exact Graph probe evidence."""

    bucket_id: int
    ordinary_family: str
    graph_family: str
    selection_source: str
    matrix_dimension: int
    physical_system_count: int
    solver_batch_count: int
    api_eligible: bool
    api_reason: str
    matrix_batch_product: int
    probe_failure_stage: str
    device_workspace_bytes: int
    host_workspace_bytes: int
    available_device_bytes: int
    device_id: int
    device_uuid: str
    device_name: str
    compute_capability: tuple[int, int]
    cuda_runtime_version: int
    cuda_driver_version: int
    cusolver_version: int
    cuda_error: int
    cusolver_error: int
    ordinary_execution_passed: bool
    graph_capture_passed: bool
    host_graph_replay_passed: bool
    device_tail_replay_passed: bool
    graph_eligible: bool
    maximum_eigenvalue_error: float
    maximum_residual: float
    maximum_orthogonality_error: float

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready evidence without losing exact status codes."""

        return {
            "bucket_id": self.bucket_id,
            "ordinary_family": self.ordinary_family,
            "graph_family": self.graph_family,
            "selection_source": self.selection_source,
            "matrix_dimension": self.matrix_dimension,
            "physical_system_count": self.physical_system_count,
            "solver_batch_count": self.solver_batch_count,
            "api_eligible": self.api_eligible,
            "api_reason": self.api_reason,
            "matrix_batch_product": self.matrix_batch_product,
            "probe_failure_stage": self.probe_failure_stage,
            "device_workspace_bytes": self.device_workspace_bytes,
            "host_workspace_bytes": self.host_workspace_bytes,
            "available_device_bytes": self.available_device_bytes,
            "device_id": self.device_id,
            "device_uuid": self.device_uuid,
            "device_name": self.device_name,
            "compute_capability": list(self.compute_capability),
            "cuda_runtime_version": self.cuda_runtime_version,
            "cuda_driver_version": self.cuda_driver_version,
            "cusolver_version": self.cusolver_version,
            "cuda_error": self.cuda_error,
            "cusolver_error": self.cusolver_error,
            "ordinary_execution_passed": self.ordinary_execution_passed,
            "graph_capture_passed": self.graph_capture_passed,
            "host_graph_replay_passed": self.host_graph_replay_passed,
            "device_tail_replay_passed": self.device_tail_replay_passed,
            "graph_eligible": self.graph_eligible,
            "maximum_eigenvalue_error": self.maximum_eigenvalue_error,
            "maximum_residual": self.maximum_residual,
            "maximum_orthogonality_error": self.maximum_orthogonality_error,
        }


@dataclass(frozen=True)
class InactiveEigensolverProfileEntry:
    """One device-timed eigensolve from the device-tail SCF loop."""

    bucket_id: int
    iteration: int
    family: str
    physical_system_count: int
    solver_batch_count: int
    active_physical_count: int
    active_solver_count: int
    solver_elapsed_nanoseconds: int
    inactive_input_nonfinite_count: int
    inactive_submission_nonfinite_count: int
    inactive_info_nonzero_count: int
    inactive_touch_flags: int
    provider_invoked: bool

    @property
    def inactive_solver_count(self) -> int:
        return self.solver_batch_count - self.active_solver_count

    @property
    def inactive_fraction(self) -> float:
        if self.solver_batch_count == 0:
            return 0.0
        return self.inactive_solver_count / self.solver_batch_count

    @property
    def inactive_touches(self) -> tuple[str, ...]:
        names = []
        if self.inactive_touch_flags & _native.EIGENSOLVER_INACTIVE_TOUCH_COPY:
            names.append("copy")
        if (
            self.inactive_touch_flags
            & _native.EIGENSOLVER_INACTIVE_TOUCH_CUBLAS_TRANSFORM
        ):
            names.append("cublas_transform")
        if (
            self.inactive_touch_flags
            & _native.EIGENSOLVER_INACTIVE_TOUCH_IDENTITY_SANITIZE
        ):
            names.append("identity_sanitize")
        return tuple(names)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready record with derived inactive work."""

        return {
            "bucket_id": self.bucket_id,
            "iteration": self.iteration,
            "family": self.family,
            "physical_system_count": self.physical_system_count,
            "solver_batch_count": self.solver_batch_count,
            "active_physical_count": self.active_physical_count,
            "active_solver_count": self.active_solver_count,
            "inactive_solver_count": self.inactive_solver_count,
            "inactive_fraction": self.inactive_fraction,
            "solver_elapsed_nanoseconds": self.solver_elapsed_nanoseconds,
            "inactive_input_nonfinite_count": (self.inactive_input_nonfinite_count),
            "inactive_submission_nonfinite_count": (
                self.inactive_submission_nonfinite_count
            ),
            "inactive_info_nonzero_count": self.inactive_info_nonzero_count,
            "inactive_touches": list(self.inactive_touches),
            "provider_invoked": self.provider_invoked,
        }


class PreparedBatch:
    """Persistent topology-aware native fleet plan.

    The object is not concurrently re-entrant because successful executions
    may update per-system warm-start densities. Use separate plans for
    concurrent callers. Warm-start updates can be frozen after an initial
    execution when reproducible replays from one fixed dm0 are required.
    """

    def __init__(
        self,
        calculator: Calculator,
        systems: Sequence[Iterable[Atom | tuple[str | int, Sequence[float]]]],
        *,
        charges: Sequence[int] | None = None,
        multiplicities: Sequence[int] | None = None,
        warm_start: bool = True,
        shell_class_profiling: bool = False,
        inactive_eigensolver_profiling: bool = False,
        resource_plan: typing.Any = None,
    ) -> None:
        if not systems:
            raise ValueError("a batch requires at least one system")
        self._last_statuses = None
        # The KS ResourcePlan reserves one serialized generated-force staging cap.
        # Keep one retained execution per PreparedBatch and reprepare on topology drift.
        self._stationary_cuda_execution: typing.Any = None
        self._restart_indices = set()
        self._projection_indices = set()
        self.projection_diagnostics = None
        self.checkpoint_diagnostics = None
        self._calculator = calculator
        self._library = calculator._library
        self._systems = tuple(
            tuple(Atom.from_value(atom) for atom in system) for system in systems
        )
        if any(not system for system in self._systems):
            raise ValueError("every batch item requires at least one atom")
        count = len(self._systems)
        self._warm_enabled = warm_start
        self._warm_updates = True
        self._warm_metadata = [None] * count
        self._charges = (
            tuple(0 for _ in range(count)) if charges is None else tuple(charges)
        )
        self._multiplicities = (
            tuple(1 for _ in range(count))
            if multiplicities is None
            else tuple(multiplicities)
        )
        if len(self._charges) != count or len(self._multiplicities) != count:
            raise ValueError("charges and multiplicities must match the batch size")
        self._ks_profile_selection = calculator._effective_ks_selection(
            self._systems,
            charges=self._charges,
            multiplicities=self._multiplicities,
        )
        self._effective_ks_options = self._ks_profile_selection.options
        for atoms in self._systems:
            calculator._preflight_hf_basis(
                atoms,
                compute_forces="forces"
                in calculator._capabilities.supported_properties,
            )
        self.resource_plan = resource_plan
        self.resource_diagnostics = None
        self._resource_ledger = None
        if resource_plan is not None or calculator._resource_budget is not None:
            request = calculator._resource_request(
                self._systems,
                charges=self._charges,
                multiplicities=self._multiplicities,
                ks_options=self._effective_ks_options,
            )
            if resource_plan is None:
                from .resources import plan_resources

                self.resource_plan = plan_resources(
                    (request,), calculator._resource_budget
                )
            else:
                owned = {r.name: r for r in resource_plan.requests}
                if owned.get(request.name) != request:
                    raise ValueError(
                        f"prepared {request.name.upper()} inputs differ from the global resource plan"
                    )
                if (
                    calculator._resource_budget is not None
                    and resource_plan.budget != calculator._resource_budget
                ):
                    raise ValueError(
                        "global resource plan differs from the calculator budget"
                    )
            self.resource_plan.require_feasible()
            if request.identity.backend == "cuda":
                from .resources_native import NativeDeviceLedger

                self._resource_ledger = NativeDeviceLedger(
                    self._library, self.resource_plan, owner=request.name
                )
            if request.identity.backend == "cuda" and (
                shell_class_profiling or inactive_eigensolver_profiling
            ):
                raise NotImplementedError(
                    "CUDA resource plans exclude optional profiling allocations"
                )
        self._model_signature = calculator._model_signature()
        self._basis_metadata = tuple(
            calculator.basis_metadata(atoms, charge=charge, multiplicity=multiplicity)
            for atoms, charge, multiplicity in zip(
                self._systems, self._charges, self._multiplicities, strict=True
            )
        )
        self._atom_counts = tuple(len(system) for system in self._systems)
        self._atomic_numbers = tuple(
            tuple(atom.atomic_number for atom in system) for system in self._systems
        )
        self._context = ctypes.c_void_p()
        self._batch = ctypes.c_void_p()
        auxiliary_handle = ctypes.c_void_p()
        self._shell_class_profiling = shell_class_profiling
        self._inactive_eigensolver_profiling = inactive_eigensolver_profiling

        _native.check(
            self._library,
            self._library.vibeqc_context_create(
                ctypes.byref(calculator._context_descriptor()),
                ctypes.byref(self._context),
            ),
        )
        system_handles: list[ctypes.c_void_p] = []
        try:
            for atoms, charge, multiplicity in zip(
                self._systems, self._charges, self._multiplicities, strict=True
            ):
                system_handles.append(
                    calculator._create_native_system(
                        self._context, atoms, charge, multiplicity
                    )
                )
            handle_array = (ctypes.c_void_p * count)(
                *(handle.value for handle in system_handles)
            )
            if calculator._auxiliary_basis is not None:
                auxiliary_handle = calculator._create_native_system(
                    self._context,
                    self._systems[0],
                    self._charges[0],
                    self._multiplicities[0],
                    calculator._auxiliary_basis,
                )
            method = calculator._method_descriptor(
                auxiliary_handle if auxiliary_handle.value else None,
                resource_plan=self.resource_plan,
                ks_options=self._effective_ks_options,
            )
            flags = _native.BATCH_ENABLE_WARM_STARTS if warm_start else 0
            if shell_class_profiling:
                flags |= _native.BATCH_ENABLE_SHELL_CLASS_PROFILING
            if inactive_eigensolver_profiling:
                flags |= _native.BATCH_ENABLE_INACTIVE_EIGENSOLVER_PROFILING

            def prepare() -> typing.Any:
                return self._library.vibeqc_batch_prepare(
                    self._context,
                    handle_array,
                    count,
                    ctypes.byref(method),
                    flags,
                    ctypes.byref(self._batch),
                )

            if self.resource_plan is None:
                _native.check(
                    self._library,
                    prepare(),
                    context=(
                        self._context
                        if calculator._method
                        in (_native.METHOD_MP2, _native.METHOD_RCCSD)
                        else None
                    ),
                )
            else:
                from .resources_native import check_resource_status, observe_method_call

                status, self.resource_diagnostics = observe_method_call(
                    self._library,
                    self.resource_plan,
                    self._resource_ledger,
                    prepare,
                    owner=request.name,
                    phase="preparation",
                )
                check_resource_status(self._library, status, self.resource_diagnostics)
        except Exception:
            # Construction owns native handles before resource-status conversion,
            # which can raise MemoryError as well as ordinary validation errors.
            self.close()
            raise
        finally:
            if auxiliary_handle.value:
                self._library.vibeqc_system_destroy(auxiliary_handle)
            for handle in system_handles:
                self._library.vibeqc_system_destroy(handle)

    @property
    def system_count(self) -> int:
        self._ensure_open()
        return int(self._library.vibeqc_batch_get_system_count(self._batch))

    @property
    def atomic_numbers(self) -> tuple[tuple[int, ...], ...]:
        return self._atomic_numbers

    @property
    def charges(self) -> tuple[int, ...]:
        """Charges retained by this fixed-topology native plan."""

        return self._charges

    @property
    def multiplicities(self) -> tuple[int, ...]:
        """Spin multiplicities retained by this fixed-topology native plan."""

        return self._multiplicities

    @property
    def basis_metadata(self) -> typing.Any:
        """Detached resolved provenance/identities for benchmark and result records."""
        return deepcopy(self._basis_metadata)

    @property
    def ks_transport_diagnostics(self) -> tuple[KsTransportDiagnostic | None, ...]:
        """Input-ordered cumulative CUDA KS movement snapshots."""
        self._ensure_open()
        return tuple(
            read_ks_transport_diagnostic(self._library, self._batch, index)
            for index in range(self.system_count)
        )

    def _ensure_open(self) -> None:
        if not self._batch.value:
            raise RuntimeError("prepared batch is closed")

    def _stationary_cuda_compiler(self) -> typing.Any:
        """Lazily bind the generated-force compiler to this native device."""
        compiler = getattr(self, "_c2_stationary_compiler", None)
        if compiler is not None:
            return compiler
        from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
        from vibeqc_compiler.common.cuda_target import cuda_target_info

        from .profiles import find_nvcc, probe_device

        nvcc = find_nvcc()
        if nvcc is None:
            raise NotImplementedError(
                "public CUDA DFT forces require NVCC; set CUDACXX or CUDA_PATH"
            )
        device = probe_device(self._library, self._calculator._device_id)["device"]
        target = cuda_target_info(f"sm_{device['major']}{device['minor']}")
        compiler = CudaCompilerAdapter(Path(nvcc), target, compile_timeout=600)
        self._c2_stationary_compiler = compiler
        return compiler

    def _public_dft_cuda_force(
        self, index: typing.Any, atoms: typing.Any
    ) -> typing.Any:
        """Execute the qualified seven/nine-source plan against one live item."""
        from vibeqc_compiler.dft import NativeAO

        from ._dft_gradient import StationaryKsState
        from ._stationary_cuda import (
            PreparedStationaryCudaExecution,
            PreparedStationaryCudaTopologyMismatch,
            complete_rks_cuda_gradient_diagnostic,
        )

        calculator = self._calculator
        prepared = self._stationary_cuda_execution
        if prepared is None:
            prepared = PreparedStationaryCudaExecution()
            self._stationary_cuda_execution = prepared
        with NativeAO(
            atoms,
            basis=calculator._basis,
            representation=calculator._representation_name,
            charge=self._charges[index],
            multiplicity=self._multiplicities[index],
        ) as basis:
            state = StationaryKsState.from_native(self, basis, index=index)
            try:
                if state._source.backend != "cuda" or (
                    "forces" not in calculator._capabilities.supported_properties
                ):
                    raise NotImplementedError(
                        "public CUDA DFT forces require a qualified CUDA owner"
                    )
                kwargs = {
                    "compiler": self._stationary_cuda_compiler(),
                    "cache": Path(
                        os.environ.get(
                            "VIBEQC_STATIONARY_CACHE", ".cache/stationary-cuda"
                        )
                    ),
                }
                try:
                    result = complete_rks_cuda_gradient_diagnostic(
                        state, basis, prepared=prepared, **kwargs
                    )
                except PreparedStationaryCudaTopologyMismatch:
                    prepared.close()
                    prepared = PreparedStationaryCudaExecution()
                    self._stationary_cuda_execution = prepared
                    result = complete_rks_cuda_gradient_diagnostic(
                        state, basis, prepared=prepared, **kwargs
                    )
                # StationaryGradientPlan publishes +dE/dR. Public API is force.
                return -np.asarray(result.gradient).copy(), dict(result.work)
            finally:
                state._source.close()

    def _public_dft_cpu_force(self, index: typing.Any, atoms: typing.Any) -> typing.Any:
        """Bounded CPU ECP force; checked native CPU ECP is an explicit provider."""
        from vibeqc_compiler.dft import NativeAO

        from ._cpu_force_resources import (
            CPU_FORCE_HOST_CAP,
            cpu_force_inventory,
            qualified_basis,
        )
        from ._dft_gradient import StationaryKsState
        from ._stationary_cpu import complete_rks_gradient_diagnostic
        from .ecp import resolve_ecp

        calculator = self._calculator
        if calculator._device_name != "cpu" or not qualified_basis(calculator._basis):
            raise NotImplementedError(
                "public CPU ECP forces require a qualified CPU owner"
            )
        if len(atoms) > 8:
            raise ValueError("CPU public force dense-export domain exceeded")
        with NativeAO(
            atoms,
            basis=calculator._basis,
            representation=calculator._representation_name,
            charge=self._charges[index],
            multiplicity=self._multiplicities[index],
        ) as basis:
            grid = calculator._ks_options.grid
            _, terms = resolve_ecp(calculator._basis, atoms)
            inventory = cpu_force_inventory(
                basis,
                grid_points=len(atoms)
                * grid.radial_points
                * grid.angular_polar
                * grid.angular_azimuth,
                ecp_terms=len(terms),
            )
            if sum(inventory.values()) > CPU_FORCE_HOST_CAP:
                raise ValueError("CPU force additional-host byte budget exceeded")
            # Reject before exporting the live SCF/grid snapshot. The consumer
            # repeats admission using the actual exported shape and term count.
            state = StationaryKsState.from_native(self, basis, index=index)
            try:
                if state._source.backend != "cpu":
                    raise NotImplementedError(
                        "public CPU ECP forces require a qualified CPU owner"
                    )
                result = complete_rks_gradient_diagnostic(
                    state,
                    basis,
                    execution="native",
                    max_host_bytes=CPU_FORCE_HOST_CAP,
                    cache=Path(
                        os.environ.get(
                            "VIBEQC_STATIONARY_CACHE", ".cache/stationary-cpu"
                        )
                    ),
                )
                work = dict(result.work)
                work["ecp_provider"] = "checked-native-cpu-two-grid-v1"
                work["host_inventory"] = inventory
                return -np.asarray(result.gradient).copy(), work
            finally:
                state._source.close()

    def execute(
        self,
        coordinates: Sequence[Sequence[Sequence[float]] | np.ndarray | None]
        | None = None,
        *,
        strict: bool = False,
        properties: Iterable[str] | None = None,
    ) -> BatchResult:
        """Replay the fleet, optionally omitting analytic forces.

        Each supplied coordinate array must be real and have shape
        ``(natoms, 3)``. Atom-count and nonfinite-coordinate errors retain the
        native per-item failure contract; malformed layouts fail before replay.

        The default requests the method's supported properties. Energy-only
        methods return ``forces=None``; HF can omit forces explicitly with
        ``properties=("energy",)``.
        Output selection does not change the prepared model or warm snapshot;
        a later force replay rebuilds response caches when necessary. Resource
        plans retain their conservative energy-plus-force capacity allowance.
        """
        self._ensure_open()
        if properties is None:
            properties = self._calculator._capabilities.supported_properties
        if isinstance(properties, (str, bytes)):
            raise TypeError("properties must be an iterable of property names")
        try:
            requested = frozenset(properties)
        except TypeError as error:
            raise TypeError(
                "properties must contain hashable property names"
            ) from error
        if "energy" not in requested:
            raise ValueError("properties must include 'energy'")
        unknown = requested - {"energy", "forces"}
        if unknown:
            names = ", ".join(sorted(repr(name) for name in unknown))
            raise ValueError(f"unsupported properties: {names}")
        unsupported = requested - self._calculator._capabilities.supported_properties
        if unsupported:
            raise ValueError(
                f"method {self._calculator._method_name!r} does not support properties: "
                + ", ".join(sorted(unsupported))
            )
        compute_forces = "forces" in requested
        public_dft_forces = (
            compute_forces
            and self._calculator._capabilities.family == "density_functional"
        )
        native_compute_forces = compute_forces and not public_dft_forces
        if self._calculator._model_signature() != self._model_signature:
            raise RuntimeError(
                "prepared basis/model identity changed; prepare a new batch before reusing densities or Fock/DIIS state"
            )
        from .checkpoint import _controls

        controls = _controls(self._calculator)
        count = len(self._systems)
        coordinate_storage: list[np.ndarray] = []
        replay_systems = list(self._systems)
        inputs_pointer = None
        input_count = 0
        if coordinates is not None:
            if len(coordinates) != count:
                raise ValueError("coordinate list must match the prepared batch size")
            input_descriptors: list[_native.BatchInputDescriptor] = []
            for index, item in enumerate(coordinates):
                if item is None:
                    input_descriptors.append(
                        _native.BatchInputDescriptor(
                            ctypes.sizeof(_native.BatchInputDescriptor),
                            _native.ABI_VERSION,
                            None,
                            0,
                        )
                    )
                    continue
                raw = np.asarray(item)
                if np.iscomplexobj(raw):
                    raise ValueError("coordinates must be real")
                # Wrong element counts already have a native per-item failure
                # contract, including flat invalid sentinels. Reject a layout
                # here only when its count could otherwise pass native admission.
                if raw.size == 3 * self._atom_counts[index] and (
                    raw.ndim != 2 or raw.shape[1] != 3
                ):
                    raise ValueError("coordinates must have shape (natoms, 3)")
                array = np.ascontiguousarray(raw, dtype=np.float64).reshape(-1)
                coordinate_storage.append(array)
                if raw.shape == (self._atom_counts[index], 3) and np.all(
                    np.isfinite(raw)
                ):
                    replay_systems[index] = tuple(
                        Atom(
                            atom.atomic_number,
                            tuple(float(value) for value in position),
                        )
                        for atom, position in zip(
                            self._systems[index], raw, strict=True
                        )
                    )
                else:
                    # Preserve native per-item failure semantics for malformed or
                    # nonfinite payloads. Keep the prepared geometry only as the
                    # profile-selection placeholder for this invalid row; every
                    # other executable row must still be requalified.
                    replay_systems[index] = self._systems[index]
                input_descriptors.append(
                    _native.BatchInputDescriptor(
                        ctypes.sizeof(_native.BatchInputDescriptor),
                        _native.ABI_VERSION,
                        array.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                        array.size,
                    )
                )
            input_array = (_native.BatchInputDescriptor * count)(*input_descriptors)
            inputs_pointer = input_array
            input_count = count
            replay_selection = self._calculator._effective_ks_selection(
                tuple(replay_systems),
                charges=self._charges,
                multiplicities=self._multiplicities,
            )
            if replay_selection.options != self._effective_ks_options:
                raise RuntimeError(
                    "profile-selected KS execution schedule changed for replay coordinates; prepare a new batch"
                )
            if (
                self._ks_profile_selection.exact_profile_match
                and not replay_selection.exact_profile_match
            ):
                raise RuntimeError(
                    "profile-selected KS execution schedule is not qualified for replay coordinates; prepare a new batch"
                )

        force_storage = [
            (ctypes.c_double * (3 * atom_count))() if native_compute_forces else None
            for atom_count in self._atom_counts
        ]
        output_array = (_native.BatchItemResultDescriptor * count)(
            *(
                _native.BatchItemResultDescriptor(
                    ctypes.sizeof(_native.BatchItemResultDescriptor),
                    _native.ABI_VERSION,
                    _native.STATUS_INVALID_ARGUMENT,
                    0.0,
                    force_storage[index],
                    len(force_storage[index]) if native_compute_forces else 0,
                    0,
                    0.0,
                    0.0,
                    0,
                    _native.BACKEND_CPU_REFERENCE,
                    0,
                    0,
                    0,
                )
                for index in range(count)
            )
        )
        if self.resource_plan is None:
            status = self._library.vibeqc_batch_execute(
                self._batch,
                inputs_pointer,
                input_count,
                output_array,
                count,
            )
        else:
            from .resources_native import observe_method_call

            current = self._calculator._resource_request(
                self._systems,
                charges=self._charges,
                multiplicities=self._multiplicities,
                ks_options=self._effective_ks_options,
            )
            if current != next(
                r for r in self.resource_plan.requests if r.name == current.name
            ):
                raise ValueError(
                    f"{current.name.upper()} resource inputs or execution schedule changed after preparation"
                )
            status, self.resource_diagnostics = observe_method_call(
                self._library,
                self.resource_plan,
                self._resource_ledger,
                lambda: self._library.vibeqc_batch_execute(
                    self._batch, inputs_pointer, input_count, output_array, count
                ),
                owner=current.name,
                previous=self.resource_diagnostics,
            )
        if self.resource_diagnostics is None:
            _native.check(self._library, status)
        else:
            from .resources_native import check_resource_status

            check_resource_status(self._library, status, self.resource_diagnostics)

        if self.resource_diagnostics is not None:
            # Separate from the native SCF ledger: generated libraries own
            # their own bounded allocations and export/work observations.
            self.resource_diagnostics["generated_force"] = []
        items: list[BatchItemResult] = []
        for index, output in enumerate(output_array):
            builds = ctypes.c_uint64()
            count_status = self._library.vibeqc_batch_get_last_fock_builds(
                self._batch, index, ctypes.byref(builds)
            )
            if count_status not in (
                _native.STATUS_SUCCESS,
                _native.STATUS_NOT_IMPLEMENTED,
            ):
                _native.check(self._library, count_status)
            succeeded = output.status == _native.STATUS_SUCCESS
            public_force = None
            if succeeded and public_dft_forces:
                atoms = self._systems[index]
                if coordinates is not None and coordinates[index] is not None:
                    xyz = np.asarray(coordinates[index], dtype=np.float64).reshape(
                        -1, 3
                    )
                    atoms = tuple(
                        Atom(atom.atomic_number, tuple(position))
                        for atom, position in zip(atoms, xyz, strict=True)
                    )
                try:
                    consumer = (
                        self._public_dft_cuda_force
                        if self._calculator._device_name == "cuda"
                        else self._public_dft_cpu_force
                    )
                    public_force, force_work = consumer(index, atoms)
                    if self.resource_diagnostics is not None:
                        self.resource_diagnostics["generated_force"].append(
                            {"index": index, "work": force_work}
                        )
                except NotImplementedError:
                    output.status = _native.STATUS_NOT_IMPLEMENTED
                    succeeded = False
                except MemoryError:
                    output.status = _native.STATUS_OUT_OF_MEMORY
                    succeeded = False
                except (TypeError, ValueError):
                    output.status = _native.STATUS_INVALID_ARGUMENT
                    succeeded = False
                except (RuntimeError, OSError, ArithmeticError):
                    output.status = _native.STATUS_NUMERICAL_FAILURE
                    succeeded = False
            physical_residual_rms = None
            scf_getter = getattr(self._library, "vibeqc_batch_get_scf_diagnostic", None)
            if scf_getter is not None:
                diagnostic = _native.ScfDiagnostic(
                    ctypes.sizeof(_native.ScfDiagnostic), _native.ABI_VERSION
                )
                status = scf_getter(self._batch, index, ctypes.byref(diagnostic))
                if status != _native.STATUS_NOT_IMPLEMENTED:
                    _native.check(self._library, status, context=self._context)
                    physical_residual_rms = diagnostic.physical_residual_rms
            forces = (
                public_force
                if succeeded and public_dft_forces
                else np.ctypeslib.as_array(force_storage[index]).copy().reshape(-1, 3)
                if succeeded and native_compute_forces
                else None
            )
            message = self._library.vibeqc_status_message(output.status).decode("utf-8")
            correlation = None
            if self._calculator._method in (_native.METHOD_MP2, _native.METHOD_RCCSD):
                correlation = _read_correlation_result(
                    self._library,
                    self._batch,
                    index=index,
                    context=self._context,
                )
            accuracy = None
            if succeeded and self._calculator._target_accuracy is not None:
                atoms = self._systems[index]
                if coordinates is not None and coordinates[index] is not None:
                    xyz = np.asarray(coordinates[index], dtype=np.float64).reshape(
                        -1, 3
                    )
                    atoms = tuple(
                        Atom(atom.atomic_number, tuple(position))
                        for atom, position in zip(atoms, xyz, strict=True)
                    )
                accuracy = self._calculator._accuracy_assessment(
                    atoms,
                    self._charges[index],
                    self._multiplicities[index],
                    bool(output.converged),
                )
            items.append(
                BatchItemResult(
                    index=index,
                    status=output.status,
                    fock_builds=builds.value
                    if count_status == _native.STATUS_SUCCESS
                    else None,
                    status_message=message,
                    energy=output.energy,
                    forces=forces,
                    converged=bool(output.converged),
                    iterations=output.iterations,
                    energy_change=output.energy_change,
                    density_rms=output.density_rms,
                    physical_residual_rms=physical_residual_rms,
                    ks_diagnostic=read_ks_diagnostic(self._library, self._batch, index)
                    if self._calculator._ks_options is not None
                    else None,
                    correlation=correlation,
                    executed_backend={
                        _native.BACKEND_CPU_REFERENCE: "cpu_reference",
                        _native.BACKEND_CUDA: "cuda",
                        _native.BACKEND_HYBRID_CUDA: "hybrid_cuda",
                    }.get(output.executed_backend, "unknown"),
                    bucket_id=output.bucket_id,
                    warm_start_used=bool(output.warm_start_used),
                    warm_start_fallback=bool(output.warm_start_fallback),
                    basis_metadata=deepcopy(self._basis_metadata[index]),
                    precision=self._calculator._precision_provenance(
                        self._batch, index
                    ),
                    accuracy=accuracy,
                    restart_origin=(
                        "cold_fallback"
                        if output.warm_start_fallback
                        else "basis_projection"
                        if output.warm_start_used and index in self._projection_indices
                        else "persistent_restart"
                        if output.warm_start_used and index in self._restart_indices
                        else "in_process_warm"
                        if output.warm_start_used
                        else "cold"
                    ),
                )
            )
        result = BatchResult(tuple(items))
        self._last_statuses = tuple(item.status for item in result.items)
        for index, item in enumerate(result.items):
            if item.succeeded and self._warm_enabled and self._warm_updates:
                self._warm_metadata[index] = {
                    "controls": deepcopy(controls),
                    "backend": "cuda" if item.executed_backend == "cuda" else "cpu",
                }
                self._restart_indices.discard(index)
                self._projection_indices.discard(index)
        if self.projection_diagnostics:
            self.projection_diagnostics["target_verification"] = "executed"
            self.projection_diagnostics["target_results"] = [
                {
                    "index": i.index,
                    "converged": i.converged,
                    "status": i.status,
                    "energy": i.energy if i.succeeded else None,
                    "density_rms": i.density_rms if i.succeeded else None,
                    "iterations": i.iterations,
                    "fock_builds": i.fock_builds,
                    "restart_origin": i.restart_origin,
                }
                for i in result.items
            ]
        if (
            self.checkpoint_diagnostics
            and "target_verification" in self.checkpoint_diagnostics
        ):
            self.checkpoint_diagnostics["target_verification"] = "executed"
            self.checkpoint_diagnostics["target_results"] = [
                {
                    "index": i.index,
                    "converged": i.converged,
                    "status": i.status,
                    "energy": i.energy if i.succeeded else None,
                    "density_rms": i.density_rms if i.succeeded else None,
                    "restart_origin": i.restart_origin,
                }
                for i in result.items
            ]
        if strict:
            try:
                result.raise_for_failures()
            except RuntimeError as error:
                if self.resource_diagnostics is not None:
                    error.resource_diagnostics = self.resource_diagnostics
                raise
        return result

    def save_checkpoint(
        self, path: typing.Any, *, max_bytes: typing.Any = 256 << 20
    ) -> typing.Any:
        """Atomically persist retained HF seeds, identities and source diagnostics.

        Failed/no-state items keep their input slots. A one-item prepared batch
        provides single-system checkpoint/restart with the same contract.
        """
        from .checkpoint import save_checkpoint

        return save_checkpoint(self, path, max_bytes=max_bytes)

    def load_checkpoint(
        self,
        path: typing.Any,
        *,
        allow_warm: typing.Any = False,
        strict: typing.Any = True,
        max_bytes: typing.Any = 256 << 20,
    ) -> typing.Any:
        """Restore compatible seeds as proposals for the next normal execution.

        Exact restart is the default. ``allow_warm`` permits changed geometry or
        numerical controls with the same scientific model. ``strict=False``
        preserves incompatible neighbors; corruption always rejects the file
        before any seed is applied. Runtime resources follow this target plan.
        """
        from .checkpoint import load_checkpoint

        return load_checkpoint(
            self, path, allow_warm=allow_warm, strict=strict, max_bytes=max_bytes
        )

    def clear_warm_starts(self) -> None:
        self._ensure_open()
        _native.check(
            self._library,
            self._library.vibeqc_batch_clear_warm_starts(self._batch),
        )
        self._restart_indices.clear()
        self._projection_indices.clear()
        self.projection_diagnostics = None
        self._warm_metadata = [None] * len(self._systems)

    def initialize_from(
        self,
        source: typing.Any,
        *,
        policy: typing.Any = None,
        strict: typing.Any = True,
        maximum_host_bytes: typing.Any = 256 << 20,
    ) -> typing.Any:
        """Project a converged source batch into this fresh target's AO metric.

        The next execute rebuilds and converges the target equations. See
        ``vibeqc.progressive.initialize_from`` for compatibility and fallback.
        """
        from .progressive import initialize_from

        return initialize_from(
            self,
            source,
            policy=policy,
            strict=strict,
            maximum_host_bytes=maximum_host_bytes,
        )

    def set_warm_start_updates(self, enabled: bool) -> None:
        """Control whether successful executions replace retained densities.

        Disabling updates freezes the current per-system snapshots without
        disabling warm starts. It is intended for controlled A/B benchmarks
        where every replay must begin from exactly the same post-cold dm0.
        """

        self._ensure_open()
        _native.check(
            self._library,
            self._library.vibeqc_batch_set_warm_start_updates(
                self._batch, int(bool(enabled))
            ),
        )
        self._warm_updates = bool(enabled)

    def last_shell_class_profile(self) -> tuple[ShellClassProfileEntry, ...]:
        """Return work surviving the most recent final-density CUDA screening.

        Profiling is intentionally opt-in because collecting it adds a CUDA
        reduction and device-to-host copy outside the normal hot path.
        """

        self._ensure_open()
        if not self._shell_class_profiling:
            raise RuntimeError(
                "the batch was not prepared with shell_class_profiling=True"
            )
        native_entries = (
            _native.ShellClassProfileEntry * _native.DIRECT_SHELL_CLASS_COUNT
        )()
        _native.check(
            self._library,
            self._library.vibeqc_batch_get_last_shell_class_profile(
                self._batch,
                native_entries,
                len(native_entries),
            ),
        )
        result = []
        for shell_class, native in enumerate(native_entries):
            first_pair, second_pair = _decode_triangular_class(shell_class)
            first_high, first_low = _decode_triangular_class(first_pair)
            second_high, second_low = _decode_triangular_class(second_pair)
            result.append(
                ShellClassProfileEntry(
                    shell_class=shell_class,
                    shell_angular=(
                        first_high,
                        first_low,
                        second_high,
                        second_low,
                    ),
                    shell_quartets=int(native.shell_quartets),
                    tiles=int(native.tiles),
                    ao_quartets=int(native.ao_quartets),
                    primitive_quartets=int(native.primitive_quartets),
                )
            )
        return tuple(result)

    def last_ppps_queue_profile(self) -> PppsQueueProfile:
        """Return production PPPS occupancy and primitive-divergence data.

        The batch must opt into ``shell_class_profiling``. Collection copies a
        compact signature for every screened PPPS ket task and is therefore a
        benchmark/debug operation, not part of normal endpoint timing.
        """

        self._ensure_open()
        if not self._shell_class_profiling:
            raise RuntimeError(
                "the batch was not prepared with shell_class_profiling=True"
            )
        native = _native.PppsQueueProfile()
        _native.check(
            self._library,
            self._library.vibeqc_batch_get_last_ppps_queue_profile(
                self._batch, ctypes.byref(native)
            ),
        )
        return PppsQueueProfile(
            descriptor_slots=int(native.descriptor_slots),
            non_empty_descriptors=int(native.non_empty_descriptors),
            empty_descriptors=int(native.empty_descriptors),
            tasks=int(native.tasks),
            primitive_work=int(native.primitive_work),
            ket_count_min=int(native.ket_count_min),
            ket_count_median=int(native.ket_count_median),
            ket_count_p90=int(native.ket_count_p90),
            ket_count_p99=int(native.ket_count_p99),
            ket_count_max=int(native.ket_count_max),
            lane_efficiency=tuple(float(value) for value in native.lane_efficiency),
            primitive_warp_efficiency=float(native.primitive_warp_efficiency),
            task_tail_imbalance=tuple(
                float(value) for value in native.task_tail_imbalance
            ),
            primitive_tail_imbalance=tuple(
                float(value) for value in native.primitive_tail_imbalance
            ),
            orientation_tasks=tuple(int(value) for value in native.orientation_tasks),
            orientation_primitive_work=tuple(
                int(value) for value in native.orientation_primitive_work
            ),
            bra_primitive_tasks=tuple(
                int(value) for value in native.bra_primitive_tasks
            ),
            bra_primitive_work=tuple(int(value) for value in native.bra_primitive_work),
            ket_primitive_tasks=tuple(
                int(value) for value in native.ket_primitive_tasks
            ),
            ket_primitive_work=tuple(int(value) for value in native.ket_primitive_work),
        )

    def last_eigensolver_diagnostics(
        self,
    ) -> tuple[EigensolverDiagnostic, ...]:
        """Return one cached setup decision for every CUDA workload bucket."""

        self._ensure_open()
        count = ctypes.c_uint32()
        _native.check(
            self._library,
            self._library.vibeqc_batch_get_last_eigensolver_diagnostics(
                self._batch, None, 0, ctypes.byref(count)
            ),
        )
        native_entries = (_native.EigensolverDiagnostic * count.value)()
        written = ctypes.c_uint32()
        _native.check(
            self._library,
            self._library.vibeqc_batch_get_last_eigensolver_diagnostics(
                self._batch,
                native_entries,
                len(native_entries),
                ctypes.byref(written),
            ),
        )
        if written.value != count.value:
            raise RuntimeError("eigensolver diagnostic count changed during copy")
        diagnostics = []
        for native in native_entries:
            diagnostics.append(
                EigensolverDiagnostic(
                    bucket_id=int(native.bucket_id),
                    ordinary_family=_native.EIGENSOLVER_FAMILY_NAMES[
                        native.ordinary_family
                    ],
                    graph_family=_native.EIGENSOLVER_FAMILY_NAMES[native.graph_family],
                    selection_source=(
                        _native.EIGENSOLVER_SELECTION_SOURCE_NAMES[
                            native.selection_source
                        ]
                    ),
                    matrix_dimension=int(native.matrix_dimension),
                    physical_system_count=int(native.physical_system_count),
                    solver_batch_count=int(native.solver_batch_count),
                    api_eligible=bool(native.api_eligible),
                    api_reason=_native.XSYEV_ELIGIBILITY_REASON_NAMES[
                        native.api_reason
                    ],
                    matrix_batch_product=int(native.matrix_batch_product),
                    probe_failure_stage=_native.XSYEV_GRAPH_PROBE_STAGE_NAMES[
                        native.probe_failure_stage
                    ],
                    device_workspace_bytes=int(native.device_workspace_bytes),
                    host_workspace_bytes=int(native.host_workspace_bytes),
                    available_device_bytes=int(native.available_device_bytes),
                    device_id=int(native.device_id),
                    device_uuid=bytes(native.device_uuid).hex(),
                    device_name=bytes(native.device_name)
                    .split(b"\0", 1)[0]
                    .decode("utf-8", errors="replace"),
                    compute_capability=(
                        int(native.compute_capability_major),
                        int(native.compute_capability_minor),
                    ),
                    cuda_runtime_version=int(native.cuda_runtime_version),
                    cuda_driver_version=int(native.cuda_driver_version),
                    cusolver_version=int(native.cusolver_version),
                    cuda_error=int(native.cuda_error),
                    cusolver_error=int(native.cusolver_error),
                    ordinary_execution_passed=bool(native.ordinary_execution_passed),
                    graph_capture_passed=bool(native.graph_capture_passed),
                    host_graph_replay_passed=bool(native.host_graph_replay_passed),
                    device_tail_replay_passed=bool(native.device_tail_replay_passed),
                    graph_eligible=bool(native.graph_eligible),
                    maximum_eigenvalue_error=float(native.maximum_eigenvalue_error),
                    maximum_residual=float(native.maximum_residual),
                    maximum_orthogonality_error=float(
                        native.maximum_orthogonality_error
                    ),
                )
            )
        return tuple(diagnostics)

    def last_density_fitting_metric_diagnostics(
        self,
    ) -> tuple[DensityFittingMetricDiagnostic, ...]:
        """Return CUDA DF metric conditioning/allocation records from the last run."""

        self._ensure_open()
        count = ctypes.c_uint32()
        _native.check(
            self._library,
            self._library.vibeqc_batch_get_last_density_fitting_metric_diagnostics(
                self._batch, None, 0, ctypes.byref(count)
            ),
        )
        native_entries = (_native.DensityFittingMetricDiagnostic * count.value)()
        written = ctypes.c_uint32()
        _native.check(
            self._library,
            self._library.vibeqc_batch_get_last_density_fitting_metric_diagnostics(
                self._batch,
                native_entries,
                len(native_entries),
                ctypes.byref(written),
            ),
        )
        if written.value != count.value:
            raise RuntimeError("DF metric diagnostic count changed during copy")
        return tuple(
            DensityFittingMetricDiagnostic(
                bucket_id=int(native.bucket_id),
                system_index=int(native.system_index),
                effective_rank=int(native.effective_rank),
                absolute_threshold=float(native.absolute_threshold),
                condition_number=float(native.condition_number),
                solver_device_workspace_bytes=int(native.solver_device_workspace_bytes),
                solver_host_workspace_bytes=int(native.solver_host_workspace_bytes),
                device_resident_bytes=int(native.device_resident_bytes),
                peak_device_bytes=int(native.peak_device_bytes),
                host_resident_bytes=int(native.host_resident_bytes),
                peak_host_bytes=int(native.peak_host_bytes),
                auxiliary_tile=int(native.auxiliary_tile),
                streamed=bool(native.streamed),
            )
            for native in native_entries
        )

    def last_inactive_eigensolver_profile(
        self,
    ) -> tuple[InactiveEigensolverProfileEntry, ...]:
        """Return device-timed iteration records from the last CUDA run."""

        self._ensure_open()
        if not self._inactive_eigensolver_profiling:
            raise RuntimeError(
                "the batch was not prepared with inactive_eigensolver_profiling=True"
            )
        count = ctypes.c_uint32()
        _native.check(
            self._library,
            self._library.vibeqc_batch_get_last_inactive_eigensolver_profile(
                self._batch, None, 0, ctypes.byref(count)
            ),
        )
        native_entries = (_native.InactiveEigensolverProfileEntry * count.value)()
        written = ctypes.c_uint32()
        _native.check(
            self._library,
            self._library.vibeqc_batch_get_last_inactive_eigensolver_profile(
                self._batch,
                native_entries,
                len(native_entries),
                ctypes.byref(written),
            ),
        )
        if written.value != count.value:
            raise RuntimeError("inactive eigensolver profile count changed during copy")
        return tuple(
            InactiveEigensolverProfileEntry(
                bucket_id=int(native.bucket_id),
                iteration=int(native.iteration),
                family=_native.EIGENSOLVER_FAMILY_NAMES[native.family],
                physical_system_count=int(native.physical_system_count),
                solver_batch_count=int(native.solver_batch_count),
                active_physical_count=int(native.active_physical_count),
                active_solver_count=int(native.active_solver_count),
                solver_elapsed_nanoseconds=int(native.solver_elapsed_nanoseconds),
                inactive_input_nonfinite_count=int(
                    native.inactive_input_nonfinite_count
                ),
                inactive_submission_nonfinite_count=int(
                    native.inactive_submission_nonfinite_count
                ),
                inactive_info_nonzero_count=int(native.inactive_info_nonzero_count),
                inactive_touch_flags=int(native.inactive_touch_flags),
                provider_invoked=bool(native.provider_invoked),
            )
            for native in native_entries
        )

    def close(self) -> None:
        if self._stationary_cuda_execution is not None:
            with suppress(Exception):
                self._stationary_cuda_execution.close()
            self._stationary_cuda_execution = None
        if self._batch.value:
            self._library.vibeqc_batch_destroy(self._batch)
            self._batch = ctypes.c_void_p()
        if self._context.value:
            self._library.vibeqc_context_destroy(self._context)
            self._context = ctypes.c_void_p()
        if self._resource_ledger is not None:
            self._resource_ledger.close()

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()
