"""Source contracts for CUDA trial/DIIS timing without extra stream fences."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "src/cc/cuda_solver.cu").read_text()


def test_trial_events_are_resolved_after_the_existing_diis_drain() -> None:
    block = SOURCE.split("if (options.diis_size) {")[-1]
    operations = (
        "cudaEventRecord(owner.trial_begin, owner.stream)",
        "owner.iteration()",
        "cudaEventRecord(owner.trial_end, owner.stream)",
        "run_diis(owner, options, trial)",
        "cudaEventElapsedTime(&trial_ms, owner.trial_begin, owner.trial_end)",
        "owner.diagnostic.iteration_seconds += trial_seconds",
        "owner.diagnostic.diis_seconds += trial_diis_seconds - trial_seconds",
    )
    positions = [block.index(operation) for operation in operations]
    assert positions == sorted(positions)
    assert "cudaStreamSynchronize" not in block.split("previous = status[0]")[0]
    assert "cudaEventSynchronize" not in SOURCE


def test_first_history_push_uses_the_same_outer_timer() -> None:
    diis = SOURCE.split("bool run_diis(", 1)[1].split("}  // namespace", 1)[0]
    first_history = diis.split("if (count == 1) {", 1)[1].split("}", 1)[0]
    checked = first_history.index("s.check_generated_error()")
    assert checked < first_history.index("return false;")
    assert "diis_seconds" not in diis
    assert "diis_started" not in diis
    assert (
        "std::clamp(static_cast<double>(trial_ms) * 1e-3, 0.0, trial_diis_seconds)"
        in SOURCE
    )


def test_timing_events_are_reused_and_exception_safe() -> None:
    constructor = SOURCE.split("Owner(const Problem&", 1)[1].split("~Owner()", 1)[0]
    allocation = constructor.split("if (options.diis_size) {", 1)[1].split("}", 1)[0]
    assert "cudaEventCreate(&trial_begin)" in allocation
    assert "cudaEventCreate(&trial_end)" in allocation
    assert "catch (...) {\n      cleanup();" in constructor
    cleanup = SOURCE.split("void cleanup() noexcept {", 1)[1].split(
        "template <class Output>", 1
    )[0]
    for event in ("trial_begin", "trial_end"):
        assert f"if ({event}) cudaEventDestroy({event});" in cleanup
        assert f"{event} = nullptr;" in cleanup


def test_scalar_status_checks_the_copied_error_destination() -> None:
    status = SOURCE.split("read_status(const Output& out)", 1)[1].split(
        "void advance(", 1
    )[0]
    assert "int error = 0;" in status
    assert "cudaMemcpyAsync(&error, state.error" in status
    assert "if (error)" in status
    assert "host_error" not in status
