"""Batch diagnostic records and native-to-Python translation.

This internal owner may depend on the narrow native ABI module, but it must not
import the public batch facade.  ``batch`` re-exports the record classes and
delegates diagnostic reads here so execution ownership remains one-way.
"""

from __future__ import annotations

import ctypes
import typing
from dataclasses import dataclass

from . import _native


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


def _decode_triangular_class(index: int) -> tuple[int, int]:
    """Decode the scheduler's triangular high/low canonical class."""

    high = 0
    while (high + 1) * (high + 2) // 2 <= index:
        high += 1
    return high, index - high * (high + 1) // 2


def decode_shell_class_profile(
    native_entries: typing.Iterable[typing.Any],
) -> tuple[ShellClassProfileEntry, ...]:
    result = []
    for shell_class, native in enumerate(native_entries):
        first_pair, second_pair = _decode_triangular_class(shell_class)
        first_high, first_low = _decode_triangular_class(first_pair)
        second_high, second_low = _decode_triangular_class(second_pair)
        result.append(
            ShellClassProfileEntry(
                shell_class=shell_class,
                shell_angular=(first_high, first_low, second_high, second_low),
                shell_quartets=int(native.shell_quartets),
                tiles=int(native.tiles),
                ao_quartets=int(native.ao_quartets),
                primitive_quartets=int(native.primitive_quartets),
            )
        )
    return tuple(result)


def decode_ppps_queue_profile(native: typing.Any) -> PppsQueueProfile:
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
        task_tail_imbalance=tuple(float(value) for value in native.task_tail_imbalance),
        primitive_tail_imbalance=tuple(
            float(value) for value in native.primitive_tail_imbalance
        ),
        orientation_tasks=(
            int(native.orientation_tasks[0]),
            int(native.orientation_tasks[1]),
        ),
        orientation_primitive_work=(
            int(native.orientation_primitive_work[0]),
            int(native.orientation_primitive_work[1]),
        ),
        bra_primitive_tasks=tuple(int(value) for value in native.bra_primitive_tasks),
        bra_primitive_work=tuple(int(value) for value in native.bra_primitive_work),
        ket_primitive_tasks=tuple(int(value) for value in native.ket_primitive_tasks),
        ket_primitive_work=tuple(int(value) for value in native.ket_primitive_work),
    )


def decode_eigensolver_diagnostics(
    native_entries: typing.Iterable[typing.Any],
) -> tuple[EigensolverDiagnostic, ...]:
    return tuple(
        EigensolverDiagnostic(
            bucket_id=int(native.bucket_id),
            ordinary_family=_native.EIGENSOLVER_FAMILY_NAMES[native.ordinary_family],
            graph_family=_native.EIGENSOLVER_FAMILY_NAMES[native.graph_family],
            selection_source=_native.EIGENSOLVER_SELECTION_SOURCE_NAMES[
                native.selection_source
            ],
            matrix_dimension=int(native.matrix_dimension),
            physical_system_count=int(native.physical_system_count),
            solver_batch_count=int(native.solver_batch_count),
            api_eligible=bool(native.api_eligible),
            api_reason=_native.XSYEV_ELIGIBILITY_REASON_NAMES[native.api_reason],
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
            maximum_orthogonality_error=float(native.maximum_orthogonality_error),
        )
        for native in native_entries
    )


def decode_density_fitting_metric_diagnostics(
    native_entries: typing.Iterable[typing.Any],
) -> tuple[DensityFittingMetricDiagnostic, ...]:
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


def decode_inactive_eigensolver_profile(
    native_entries: typing.Iterable[typing.Any],
) -> tuple[InactiveEigensolverProfileEntry, ...]:
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
            inactive_input_nonfinite_count=int(native.inactive_input_nonfinite_count),
            inactive_submission_nonfinite_count=int(
                native.inactive_submission_nonfinite_count
            ),
            inactive_info_nonzero_count=int(native.inactive_info_nonzero_count),
            inactive_touch_flags=int(native.inactive_touch_flags),
            provider_invoked=bool(native.provider_invoked),
        )
        for native in native_entries
    )


def read_shell_class_profile(
    library: typing.Any, handle: typing.Any
) -> tuple[ShellClassProfileEntry, ...]:
    native_entries = (
        _native.ShellClassProfileEntry * _native.DIRECT_SHELL_CLASS_COUNT
    )()
    _native.check(
        library,
        library.vibeqc_batch_get_last_shell_class_profile(
            handle, native_entries, len(native_entries)
        ),
    )
    return decode_shell_class_profile(native_entries)


def read_ppps_queue_profile(
    library: typing.Any, handle: typing.Any
) -> PppsQueueProfile:
    native = _native.PppsQueueProfile()
    _native.check(
        library,
        library.vibeqc_batch_get_last_ppps_queue_profile(handle, ctypes.byref(native)),
    )
    return decode_ppps_queue_profile(native)


def read_eigensolver_diagnostics(
    library: typing.Any, handle: typing.Any
) -> tuple[EigensolverDiagnostic, ...]:
    count = ctypes.c_uint32()
    _native.check(
        library,
        library.vibeqc_batch_get_last_eigensolver_diagnostics(
            handle, None, 0, ctypes.byref(count)
        ),
    )
    native_entries = (_native.EigensolverDiagnostic * count.value)()
    written = ctypes.c_uint32()
    _native.check(
        library,
        library.vibeqc_batch_get_last_eigensolver_diagnostics(
            handle, native_entries, len(native_entries), ctypes.byref(written)
        ),
    )
    if written.value != count.value:
        raise RuntimeError("eigensolver diagnostic count changed during copy")
    return decode_eigensolver_diagnostics(native_entries)


def read_density_fitting_metric_diagnostics(
    library: typing.Any, handle: typing.Any
) -> tuple[DensityFittingMetricDiagnostic, ...]:
    count = ctypes.c_uint32()
    _native.check(
        library,
        library.vibeqc_batch_get_last_density_fitting_metric_diagnostics(
            handle, None, 0, ctypes.byref(count)
        ),
    )
    native_entries = (_native.DensityFittingMetricDiagnostic * count.value)()
    written = ctypes.c_uint32()
    _native.check(
        library,
        library.vibeqc_batch_get_last_density_fitting_metric_diagnostics(
            handle, native_entries, len(native_entries), ctypes.byref(written)
        ),
    )
    if written.value != count.value:
        raise RuntimeError("DF metric diagnostic count changed during copy")
    return decode_density_fitting_metric_diagnostics(native_entries)


def read_inactive_eigensolver_profile(
    library: typing.Any, handle: typing.Any
) -> tuple[InactiveEigensolverProfileEntry, ...]:
    count = ctypes.c_uint32()
    _native.check(
        library,
        library.vibeqc_batch_get_last_inactive_eigensolver_profile(
            handle, None, 0, ctypes.byref(count)
        ),
    )
    native_entries = (_native.InactiveEigensolverProfileEntry * count.value)()
    written = ctypes.c_uint32()
    _native.check(
        library,
        library.vibeqc_batch_get_last_inactive_eigensolver_profile(
            handle, native_entries, len(native_entries), ctypes.byref(written)
        ),
    )
    if written.value != count.value:
        raise RuntimeError("inactive eigensolver profile count changed during copy")
    return decode_inactive_eigensolver_profile(native_entries)


# Preserve the established pickle/global path while ``batch`` remains the
# compatibility facade.  The aliases are installed during the same import.
for _public_record in (
    ShellClassProfileEntry,
    DensityFittingMetricDiagnostic,
    PppsQueueProfile,
    EigensolverDiagnostic,
    InactiveEigensolverProfileEntry,
):
    _public_record.__module__ = "vibeqc.batch"
del _public_record
