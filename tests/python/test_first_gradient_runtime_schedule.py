"""Schedule regressions for the shared CUDA first-gradient streaming owner."""

from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RUNTIME = REPOSITORY_ROOT / "src/integrals/first_gradient_runtime.cuh"


def _section(source: str, start: str, end: str) -> str:
    begin = source.index(start)
    return source[begin : source.index(end, begin)]


def test_append_fences_only_reusable_host_record_lifetime() -> None:
    source = RUNTIME.read_text()
    append = _section(source, "void append(Plan& p", "template <class Operation>")

    copy = "cudaMemcpyAsync(p.data() + p.record_offset"
    record = "cudaEventRecord(p.context.begin, p.context.stream)"
    launch = "execute<Program><<<"
    fence = "cudaEventSynchronize(p.context.begin)"

    assert (
        append.index(copy)
        < append.index(record)
        < append.index(launch)
        < append.index(fence)
    )
    successful, cleanup = append.split("} catch (...) {", maxsplit=1)
    assert "cudaStreamSynchronize(p.context.stream)" not in successful
    assert "(void)cudaStreamSynchronize(p.context.stream);" in cleanup
    assert cleanup.index("cudaStreamSynchronize") < cleanup.index("throw;")
    assert "cudaMemcpyDeviceToHost" not in append


def test_device_publication_keeps_generated_error_sticky() -> None:
    source = RUNTIME.read_text()
    output_device = _section(source, "const double* output_device", "void finish")

    assert "cudaMemsetAsync(context.error" not in output_device
    assert "validate_output<<<" in output_device
    assert "cudaMemcpyDeviceToHost" in output_device
    assert "cudaStreamSynchronize(context.stream)" in output_device
