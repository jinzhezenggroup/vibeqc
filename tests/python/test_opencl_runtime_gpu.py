"""Optional native OpenCL ownership/error gates; enabled failures are never skips."""

import ctypes as c
import os

import numpy as np
import pytest

from tools.vibeqc_codegen.opencl_runtime import OpenCLError, OpenCLRuntime, Resource
from tools.vibeqc_codegen.runtime_backend import (
    ExecutionShape,
    LibraryRequest,
    UnsupportedBackendFeature,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_OPENCL_TEST") != "1", reason="explicit Slurm OpenCL GPU tier"
)

SOURCE = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void scale(__global const double* input, __global double* output, ulong count) {
  size_t i = get_global_id(0);
  if (i < count) output[i] = input[i] * 2.0;
}
"""


@pytest.fixture
def runtime():
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    with OpenCLRuntime(library=os.environ.get("VIBEQC_OPENCL_LIBRARY")) as context:
        yield context


def test_transfers_bounds_and_context_ownership(runtime):
    data = np.arange(17, dtype=np.float64)
    buffer = runtime.allocate(data.nbytes)
    runtime.write(buffer, data.tobytes())
    assert runtime.read(buffer, data.nbytes) == data.tobytes()
    for offset, size in ((-1, 8), (0, data.nbytes + 1), (data.nbytes, 1)):
        with pytest.raises(ValueError, match="exceeds"):
            runtime.read(buffer, size, offset=offset)
    forged = Resource("buffer", buffer.handle, runtime, buffer.nbytes)
    with pytest.raises(ValueError, match="live buffer"):
        runtime.read(forged, 8)
    with (
        OpenCLRuntime(library=os.environ.get("VIBEQC_OPENCL_LIBRARY")) as other,
        pytest.raises(ValueError, match="live buffer"),
    ):
        other.write(buffer, b"12345678")
    runtime.release(buffer)
    with pytest.raises(ValueError, match="live buffer"):
        runtime.read(buffer, 8)


def test_compile_failure_releases_failed_program_and_retains_diagnostics(runtime):
    before = len(runtime._resources)
    with pytest.raises(OpenCLError) as failed:
        runtime.compile("__kernel void broken( { definitely invalid")
    assert failed.value.operation == "clCompileProgram"
    assert failed.value.log
    assert len(runtime._resources) == before
    program = runtime.compile(SOURCE)
    executable = runtime.link((program,))
    assert runtime.resources(executable, "scale")["maximum_workgroup_threads"] >= 32


def test_cross_queue_events_retain_buffers_and_execute_partial_workgroups(
    runtime, monkeypatch
):
    program = runtime.compile(SOURCE)
    executable = runtime.link((program,))
    data = np.arange(17, dtype=np.float64)
    a, b, d = [runtime.allocate(data.nbytes) for _ in range(3)]
    runtime.write(a, data.tobytes())
    first = runtime.launch(
        executable, "scale", (a, b, c.c_uint64(17)), items=17, shape=ExecutionShape(32)
    )
    with pytest.raises(ValueError, match="retained"):
        runtime.release(a)
    stream = runtime.create_stream()
    flushed = []
    native_flush = runtime.api.clFlush

    def flush(queue):
        flushed.append(queue)
        return native_flush(queue)

    monkeypatch.setattr(runtime.api, "clFlush", flush)
    second = runtime.launch(
        executable,
        "scale",
        (b, d, c.c_uint64(17)),
        items=17,
        shape=ExecutionShape(16),
        stream=stream,
        wait_for=(first,),
    )
    assert runtime.elapsed_nanoseconds(second) >= 0
    assert runtime.queue.handle in flushed
    runtime.wait(first)
    np.testing.assert_array_equal(
        np.frombuffer(runtime.read(d, data.nbytes, stream=stream), dtype=np.float64),
        4 * data,
    )
    for item in (first, second, a, b, d, stream, executable, program):
        runtime.release(item)


def test_unsupported_graphs_and_invalid_launches_fail_before_enqueue(runtime):
    target = runtime.target_info()
    assert target.backend == "opencl" and target.maximum_resident_workgroups is None
    with pytest.raises(UnsupportedBackendFeature, match="GEMM"):
        runtime.library_provider().plan(LibraryRequest("gemm", (2, 3, 4)))
    program = runtime.compile(SOURCE)
    executable = runtime.link((program,))
    with pytest.raises(UnsupportedBackendFeature, match="graph"):
        runtime.launch(
            executable,
            "scale",
            (),
            items=1,
            shape=ExecutionShape(32, requires_graphs=True),
        )
    with pytest.raises(ValueError, match="positive"):
        runtime.launch(executable, "scale", (), items=0, shape=ExecutionShape(32))
    with pytest.raises(TypeError, match="typed scalar"):
        runtime.launch(executable, "scale", (1,), items=1, shape=ExecutionShape(32))
    # Failed kernel setup must release its transient cl_kernel handle.
    assert all(resource.kind != "kernel" for resource in runtime._resources.values())


@pytest.mark.parametrize("count,workgroup", [(1, 16), (17, 32), (8197, 64)])
def test_fp64_reduction_stays_on_device_across_partial_levels(
    runtime, count, workgroup
):
    data = np.arange(count, dtype=np.float64) - 11
    source = runtime.allocate(data.nbytes)
    runtime.write(source, data.tobytes())
    result = runtime.reduce_sum(source, count, workgroup=workgroup)
    assert result is not source
    assert np.frombuffer(runtime.read(result, 8), dtype=np.float64)[0] == data.sum()
    assert runtime.read(source, data.nbytes) == data.tobytes()
    runtime.release(result)
    runtime.release(source)


def test_close_attempts_all_releases_after_vendor_finish_error(runtime, monkeypatch):
    buffer = runtime.allocate(8)
    released = []
    native_release = runtime.api.clReleaseMemObject

    def release(handle):
        released.append(handle)
        return native_release(handle)

    monkeypatch.setattr(runtime.api, "clFinish", lambda _: -5)
    monkeypatch.setattr(runtime.api, "clReleaseMemObject", release)
    handle = buffer.handle
    with pytest.raises(OpenCLError, match="close"):
        runtime.close()
    assert released == [handle]
    assert buffer.handle == 0 and runtime._closed and not runtime._resources
    runtime.close()  # Idempotence must survive the failed cleanup call.


def test_reduction_terminal_execution_error_releases_intermediates(
    runtime, monkeypatch
):
    source = runtime.allocate(17 * 8)
    runtime.write(source, np.arange(17, dtype=np.float64).tobytes())
    before = set(runtime._resources)
    native_wait = runtime.api.clWaitForEvents

    def fail_after_completion(count, events):
        assert native_wait(count, events) == 0
        return -14

    monkeypatch.setattr(runtime.api, "clWaitForEvents", fail_after_completion)
    with pytest.raises(OpenCLError, match="clWaitForEvents"):
        runtime.reduce_sum(source, 17)
    assert set(runtime._resources) == before
    runtime.release(source)
