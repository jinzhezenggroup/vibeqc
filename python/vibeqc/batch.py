"""Persistent ragged-batch interface backed by the native fleet plan."""

from __future__ import annotations

import ctypes
from collections.abc import Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Self

import numpy as np

from . import _native
from .calculator import Atom, Calculator


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
            raise RuntimeError("batched HF item failures: " + "; ".join(failures))


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
    """CUDA DF metric conditioning and allocation evidence for one plan slot."""

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
    ) -> None:
        if not systems:
            raise ValueError("a batch requires at least one system")
        self._calculator = calculator
        self._library = calculator._library
        self._systems = tuple(
            tuple(Atom.from_value(atom) for atom in system) for system in systems
        )
        if any(not system for system in self._systems):
            raise ValueError("every batch item requires at least one atom")
        count = len(self._systems)
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
                auxiliary_handle if auxiliary_handle.value else None
            )
            flags = _native.BATCH_ENABLE_WARM_STARTS if warm_start else 0
            if shell_class_profiling:
                flags |= _native.BATCH_ENABLE_SHELL_CLASS_PROFILING
            if inactive_eigensolver_profiling:
                flags |= _native.BATCH_ENABLE_INACTIVE_EIGENSOLVER_PROFILING
            _native.check(
                self._library,
                self._library.vibeqc_batch_prepare(
                    self._context,
                    handle_array,
                    count,
                    ctypes.byref(method),
                    flags,
                    ctypes.byref(self._batch),
                ),
            )
        except Exception:
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

    def _ensure_open(self) -> None:
        if not self._batch.value:
            raise RuntimeError("prepared batch is closed")

    def execute(
        self,
        coordinates: Sequence[Sequence[Sequence[float]] | np.ndarray | None]
        | None = None,
        *,
        strict: bool = False,
    ) -> BatchResult:
        self._ensure_open()
        count = len(self._systems)
        coordinate_storage: list[np.ndarray] = []
        inputs_pointer = None
        input_count = 0
        if coordinates is not None:
            if len(coordinates) != count:
                raise ValueError("coordinate list must match the prepared batch size")
            input_descriptors: list[_native.BatchInputDescriptor] = []
            for item in coordinates:
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
                array = np.ascontiguousarray(item, dtype=np.float64).reshape(-1)
                coordinate_storage.append(array)
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

        force_storage = [
            (ctypes.c_double * (3 * atom_count))() for atom_count in self._atom_counts
        ]
        output_array = (_native.BatchItemResultDescriptor * count)(
            *(
                _native.BatchItemResultDescriptor(
                    ctypes.sizeof(_native.BatchItemResultDescriptor),
                    _native.ABI_VERSION,
                    _native.STATUS_INVALID_ARGUMENT,
                    0.0,
                    force_storage[index],
                    len(force_storage[index]),
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
        _native.check(
            self._library,
            self._library.vibeqc_batch_execute(
                self._batch,
                inputs_pointer,
                input_count,
                output_array,
                count,
            ),
        )

        items: list[BatchItemResult] = []
        for index, output in enumerate(output_array):
            succeeded = output.status == _native.STATUS_SUCCESS
            forces = (
                np.ctypeslib.as_array(force_storage[index]).copy().reshape(-1, 3)
                if succeeded
                else None
            )
            message = self._library.vibeqc_status_message(output.status).decode("utf-8")
            items.append(
                BatchItemResult(
                    index=index,
                    status=output.status,
                    status_message=message,
                    energy=output.energy,
                    forces=forces,
                    converged=bool(output.converged),
                    iterations=output.iterations,
                    energy_change=output.energy_change,
                    density_rms=output.density_rms,
                    executed_backend={
                        _native.BACKEND_CPU_REFERENCE: "cpu_reference",
                        _native.BACKEND_CUDA: "cuda",
                        _native.BACKEND_HYBRID_CUDA: "hybrid_cuda",
                    }.get(output.executed_backend, "unknown"),
                    bucket_id=output.bucket_id,
                    warm_start_used=bool(output.warm_start_used),
                    warm_start_fallback=bool(output.warm_start_fallback),
                )
            )
        result = BatchResult(tuple(items))
        if strict:
            result.raise_for_failures()
        return result

    def clear_warm_starts(self) -> None:
        self._ensure_open()
        _native.check(
            self._library,
            self._library.vibeqc_batch_clear_warm_starts(self._batch),
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
        if self._batch.value:
            self._library.vibeqc_batch_destroy(self._batch)
            self._batch = ctypes.c_void_p()
        if self._context.value:
            self._library.vibeqc_context_destroy(self._context)
            self._context = ctypes.c_void_p()

    def __enter__(self) -> Self:
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()
