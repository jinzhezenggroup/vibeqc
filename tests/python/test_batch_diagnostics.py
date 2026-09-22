"""Behavior-neutral ownership tests for public batch diagnostics."""

import ast
import ctypes
import json
import pickle
import typing
from pathlib import Path

import pytest
import vibeqc
import vibeqc._batch_diagnostics as diagnostics
import vibeqc.batch as batch_facade
from vibeqc import _native

PUBLIC_RECORDS = (
    "ShellClassProfileEntry",
    "DensityFittingMetricDiagnostic",
    "PppsQueueProfile",
    "EigensolverDiagnostic",
    "InactiveEigensolverProfileEntry",
)


def test_public_record_identity_and_pickle_path_remain_compatible() -> None:
    for name in PUBLIC_RECORDS:
        public = getattr(vibeqc, name)
        assert public is getattr(batch_facade, name)
        assert public is getattr(diagnostics, name)
        assert public.__module__ == "vibeqc.batch"

    record = vibeqc.ShellClassProfileEntry(54, (3, 3, 3, 3), 1, 2, 3, 4)
    assert pickle.loads(pickle.dumps(record)) == record  # noqa: S301
    assert json.loads(json.dumps(record.__dict__)) == {
        "shell_class": 54,
        "shell_angular": [3, 3, 3, 3],
        "shell_quartets": 1,
        "tiles": 2,
        "ao_quartets": 3,
        "primitive_quartets": 4,
    }


def test_internal_owner_has_one_way_dependency_on_native_abi() -> None:
    source = Path(diagnostics.__file__).read_text(encoding="utf-8")
    imports = [
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    ]
    assert "batch" not in imports
    assert "calculator" not in imports


def test_pure_shell_and_ppps_decoders_cover_empty_and_derived_fields() -> None:
    native_shells = (_native.ShellClassProfileEntry * 55)()
    native_shells[0].shell_quartets = 11
    native_shells[54].primitive_quartets = 29
    shell_profile = diagnostics.decode_shell_class_profile(native_shells)
    assert len(shell_profile) == 55
    assert shell_profile[0].label == "ssss"
    assert shell_profile[0].shell_quartets == 11
    assert shell_profile[-1].label == "ffff"
    assert shell_profile[-1].primitive_quartets == 29
    assert diagnostics.decode_shell_class_profile(()) == ()

    native_ppps = _native.PppsQueueProfile()
    native_ppps.descriptor_slots = 8
    native_ppps.non_empty_descriptors = 5
    native_ppps.empty_descriptors = 3
    native_ppps.tasks = 13
    native_ppps.primitive_work = 21
    native_ppps.ket_count_p90 = 7
    native_ppps.lane_efficiency[:] = (0.25, 0.5, 0.75, 1.0)
    native_ppps.orientation_tasks[:] = (2, 11)
    native_ppps.ket_primitive_work[64] = 34
    ppps = diagnostics.decode_ppps_queue_profile(native_ppps)
    assert ppps.hole_rate == pytest.approx(3 / 8)
    assert ppps.lane_efficiency == (0.25, 0.5, 0.75, 1.0)
    assert ppps.orientation_tasks == (2, 11)
    assert ppps.ket_primitive_work[-1] == 34


def test_pure_eigensolver_df_and_inactive_decoders_preserve_native_meaning() -> None:
    native_eigensolver = _native.EigensolverDiagnostic()
    native_eigensolver.bucket_id = 4
    native_eigensolver.ordinary_family = 2
    native_eigensolver.graph_family = 3
    native_eigensolver.selection_source = 1
    native_eigensolver.matrix_dimension = 48
    native_eigensolver.api_eligible = 1
    native_eigensolver.api_reason = 0
    native_eigensolver.probe_failure_stage = 0
    native_eigensolver.device_uuid[:] = range(16)
    native_eigensolver.device_name = b"fixture-gpu"
    native_eigensolver.compute_capability_major = 9
    native_eigensolver.compute_capability_minor = 0
    native_eigensolver.cuda_error = 17
    native_eigensolver.graph_eligible = 1
    (eigensolver,) = diagnostics.decode_eigensolver_diagnostics((native_eigensolver,))
    assert eigensolver.ordinary_family == "xsyev_batched"
    assert eigensolver.graph_family == "graph_native"
    assert eigensolver.selection_source == "exact_probe"
    assert eigensolver.api_reason == "eligible"
    assert eigensolver.probe_failure_stage == "none"
    assert eigensolver.device_uuid == "000102030405060708090a0b0c0d0e0f"
    assert eigensolver.device_name == "fixture-gpu"
    assert eigensolver.compute_capability == (9, 0)
    assert eigensolver.to_dict()["compute_capability"] == [9, 0]

    native_df = _native.DensityFittingMetricDiagnostic()
    native_df.bucket_id = 2
    native_df.system_index = 3
    native_df.effective_rank = 19
    native_df.absolute_threshold = 1.25e-12
    native_df.condition_number = 8.5
    native_df.peak_device_bytes = 4096
    native_df.auxiliary_tile = 64
    native_df.streamed = 1
    (df,) = diagnostics.decode_density_fitting_metric_diagnostics((native_df,))
    assert df.to_dict() == {
        "bucket_id": 2,
        "system_index": 3,
        "effective_rank": 19,
        "absolute_threshold": 1.25e-12,
        "condition_number": 8.5,
        "solver_device_workspace_bytes": 0,
        "solver_host_workspace_bytes": 0,
        "device_resident_bytes": 0,
        "peak_device_bytes": 4096,
        "host_resident_bytes": 0,
        "peak_host_bytes": 0,
        "auxiliary_tile": 64,
        "streamed": True,
    }

    native_inactive = _native.InactiveEigensolverProfileEntry()
    native_inactive.family = 1
    native_inactive.solver_batch_count = 8
    native_inactive.active_solver_count = 3
    native_inactive.inactive_touch_flags = (
        _native.EIGENSOLVER_INACTIVE_TOUCH_COPY
        | _native.EIGENSOLVER_INACTIVE_TOUCH_IDENTITY_SANITIZE
    )
    native_inactive.provider_invoked = 1
    (inactive,) = diagnostics.decode_inactive_eigensolver_profile((native_inactive,))
    assert inactive.family == "jacobi_batched"
    assert inactive.inactive_solver_count == 5
    assert inactive.inactive_fraction == pytest.approx(5 / 8)
    assert inactive.inactive_touches == ("copy", "identity_sanitize")
    assert inactive.to_dict()["inactive_touches"] == [
        "copy",
        "identity_sanitize",
    ]

    assert diagnostics.decode_eigensolver_diagnostics(()) == ()
    assert diagnostics.decode_density_fitting_metric_diagnostics(()) == ()
    assert diagnostics.decode_inactive_eigensolver_profile(()) == ()


class _VariableDiagnosticLibrary:
    def __init__(self, *, count: int = 0, written: int | None = None) -> None:
        self.count = count
        self.written = count if written is None else written
        self.calls = 0

    def vibeqc_batch_get_last_eigensolver_diagnostics(
        self,
        handle: typing.Any,
        entries: typing.Any,
        capacity: typing.Any,
        output_count: typing.Any,
    ) -> int:
        del handle, capacity
        self.calls += 1
        output_count._obj.value = self.count if entries is None else self.written
        return _native.STATUS_SUCCESS


class _EmptyDiagnosticLibrary:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def vibeqc_batch_get_last_shell_class_profile(
        self,
        handle: typing.Any,
        entries: typing.Any,
        capacity: int,
    ) -> int:
        del handle, entries
        assert capacity == _native.DIRECT_SHELL_CLASS_COUNT
        self.calls.append("shell")
        return _native.STATUS_SUCCESS

    def vibeqc_batch_get_last_ppps_queue_profile(
        self, handle: typing.Any, output: typing.Any
    ) -> int:
        del handle, output
        self.calls.append("ppps")
        return _native.STATUS_SUCCESS

    def _empty_variable(
        self,
        name: str,
        handle: typing.Any,
        entries: typing.Any,
        capacity: int,
        output_count: typing.Any,
    ) -> int:
        del handle
        assert capacity == 0
        self.calls.append(f"{name}:{'query' if entries is None else 'copy'}")
        output_count._obj.value = 0
        return _native.STATUS_SUCCESS

    def vibeqc_batch_get_last_eigensolver_diagnostics(
        self,
        handle: typing.Any,
        entries: typing.Any,
        capacity: int,
        output_count: typing.Any,
    ) -> int:
        return self._empty_variable(
            "eigensolver", handle, entries, capacity, output_count
        )

    def vibeqc_batch_get_last_density_fitting_metric_diagnostics(
        self,
        handle: typing.Any,
        entries: typing.Any,
        capacity: int,
        output_count: typing.Any,
    ) -> int:
        return self._empty_variable("df", handle, entries, capacity, output_count)

    def vibeqc_batch_get_last_inactive_eigensolver_profile(
        self,
        handle: typing.Any,
        entries: typing.Any,
        capacity: int,
        output_count: typing.Any,
    ) -> int:
        return self._empty_variable("inactive", handle, entries, capacity, output_count)


def test_all_readers_preserve_native_call_counts_for_empty_records() -> None:
    library = _EmptyDiagnosticLibrary()
    handle = object()
    shell = diagnostics.read_shell_class_profile(library, handle)
    assert len(shell) == _native.DIRECT_SHELL_CLASS_COUNT
    assert shell[0].shell_quartets == 0
    assert diagnostics.read_ppps_queue_profile(library, handle).descriptor_slots == 0
    assert diagnostics.read_eigensolver_diagnostics(library, handle) == ()
    assert diagnostics.read_density_fitting_metric_diagnostics(library, handle) == ()
    assert diagnostics.read_inactive_eigensolver_profile(library, handle) == ()
    assert library.calls == [
        "shell",
        "ppps",
        "eigensolver:query",
        "eigensolver:copy",
        "df:query",
        "df:copy",
        "inactive:query",
        "inactive:copy",
    ]


def test_reader_preserves_two_call_empty_and_changed_count_semantics() -> None:
    empty = _VariableDiagnosticLibrary()
    assert diagnostics.read_eigensolver_diagnostics(empty, object()) == ()
    assert empty.calls == 2

    changed = _VariableDiagnosticLibrary(count=1, written=0)
    with pytest.raises(RuntimeError, match="count changed during copy"):
        diagnostics.read_eigensolver_diagnostics(changed, object())
    assert changed.calls == 2


def test_reader_preserves_native_status_error_and_call_count() -> None:
    class Unsupported:
        calls = 0

        @staticmethod
        def vibeqc_status_message(status: int) -> bytes:
            assert status == _native.STATUS_NOT_IMPLEMENTED
            return b"not implemented"

        def vibeqc_batch_get_last_eigensolver_diagnostics(
            self,
            handle: typing.Any,
            entries: typing.Any,
            capacity: typing.Any,
            output_count: typing.Any,
        ) -> int:
            del handle, entries, capacity, output_count
            self.calls += 1
            return _native.STATUS_NOT_IMPLEMENTED

    library = Unsupported()
    with pytest.raises(NotImplementedError, match="VIBEQC error 3: not implemented"):
        diagnostics.read_eigensolver_diagnostics(library, object())
    assert library.calls == 1


@pytest.mark.parametrize(
    ("method", "reader", "flag"),
    (
        ("last_shell_class_profile", "read_shell_class_profile", "shell"),
        ("last_ppps_queue_profile", "read_ppps_queue_profile", "shell"),
        ("last_eigensolver_diagnostics", "read_eigensolver_diagnostics", None),
        (
            "last_density_fitting_metric_diagnostics",
            "read_density_fitting_metric_diagnostics",
            None,
        ),
        (
            "last_inactive_eigensolver_profile",
            "read_inactive_eigensolver_profile",
            "inactive",
        ),
    ),
)
def test_prepared_batch_diagnostic_methods_are_thin_delegates(
    monkeypatch: pytest.MonkeyPatch, method: str, reader: str, flag: str | None
) -> None:
    prepared = object.__new__(batch_facade.PreparedBatch)
    prepared._batch = ctypes.c_void_p(123)
    monkeypatch.setattr(prepared, "_library", object(), raising=False)
    prepared._shell_class_profiling = flag == "shell"
    prepared._inactive_eigensolver_profiling = flag == "inactive"
    sentinel = object()
    calls: list[tuple[object, int | None]] = []

    def fake_reader(library: object, handle: ctypes.c_void_p) -> object:
        calls.append((library, handle.value))
        return sentinel

    monkeypatch.setattr(batch_facade, reader, fake_reader)
    assert getattr(prepared, method)() is sentinel
    assert calls == [(prepared._library, 123)]


def test_prepared_batch_opt_in_guards_run_before_native_readers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = object.__new__(batch_facade.PreparedBatch)
    prepared._batch = ctypes.c_void_p(123)
    monkeypatch.setattr(prepared, "_library", object(), raising=False)
    prepared._shell_class_profiling = False
    prepared._inactive_eigensolver_profiling = False

    def unexpected_reader(library: object, handle: ctypes.c_void_p) -> object:
        del library, handle
        raise AssertionError("reader must not run before its opt-in guard")

    monkeypatch.setattr(batch_facade, "read_shell_class_profile", unexpected_reader)
    monkeypatch.setattr(batch_facade, "read_ppps_queue_profile", unexpected_reader)
    monkeypatch.setattr(
        batch_facade, "read_inactive_eigensolver_profile", unexpected_reader
    )
    with pytest.raises(RuntimeError, match="shell_class_profiling=True"):
        prepared.last_shell_class_profile()
    with pytest.raises(RuntimeError, match="shell_class_profiling=True"):
        prepared.last_ppps_queue_profile()
    with pytest.raises(RuntimeError, match="inactive_eigensolver_profiling=True"):
        prepared.last_inactive_eigensolver_profile()
