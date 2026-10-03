"""Offline CUDA cost reports retain complete and target-matched PTXAS evidence."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _row(function: str, registers: int, architecture: str | None = None) -> str:
    header = (
        ""
        if architecture is None
        else f"ptxas info : Compiling entry function '{function}' for '{architecture}'\n"
    )
    return (
        header
        + f"ptxas info : Function properties for {function}\n"
        + "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
        + f"ptxas info : Used {registers} registers, 0 bytes smem\n"
    )


def _run(
    tmp_path: Path, log: str | None, *extra: str
) -> subprocess.CompletedProcess[str]:
    args = [
        sys.executable,
        str(_ROOT / "tools/analyze_cuda_cost.py"),
        "--arch",
        "sm_120",
        "--block-threads",
        "128",
    ]
    if log is not None:
        path = tmp_path / "ptxas.log"
        path.write_text(log, encoding="utf-8")
        args.extend(("--ptxas", str(path)))
    args.extend(extra)
    return subprocess.run(
        args,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(_ROOT / "python")},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


@pytest.mark.parametrize(
    ("log", "message"),
    [
        ("", "requires PTXAS resource rows"),
        (
            _row("light", 32, "sm_120")
            + "ptxas info : Function properties for heavy\n"
            + "    1024 bytes stack frame, 128 bytes spill stores, 128 bytes spill loads\n",
            "incomplete or unsupported resource rows",
        ),
        (
            _row("kernel", 32, "sm_120")
            + "ptxas info : Function properties for kernel\n"
            + "    1024 bytes stack frame, 128 bytes spill stores, 128 bytes spill loads\n",
            "incomplete or unsupported resource rows",
        ),
        (
            _row("light", 32, "sm_120")
            + "ptxas info : Compiling entry function 'heavy' for 'sm_120'\n",
            "incomplete or unsupported resource rows",
        ),
        (_row("kernel", 32, "sm_80"), "do not match requested target sm_120"),
        (
            _row("kernel", 255, "sm_80") + _row("kernel", 32, "sm_120"),
            "do not match requested target sm_120",
        ),
    ],
)
def test_cli_rejects_incomplete_or_mismatched_ptxas(
    tmp_path: Path, log: str, message: str
) -> None:
    result = _run(tmp_path, log)

    assert result.returncode != 0
    assert result.stdout == ""
    assert message in result.stderr


def test_cli_reports_complete_multi_kernel_provenance(tmp_path: Path) -> None:
    result = _run(tmp_path, _row("light", 32, "sm_120") + _row("heavy", 128, "sm_120"))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ptxas_evidence"] == {
        "architecture": "sm_120",
        "architecture_verified": True,
        "functions": ["light", "heavy"],
    }
    assert payload["evidence_stage"] == "compiled"
    assert payload["registers_per_thread"] == 128
    assert payload["occupancy_upper_bound"] == pytest.approx(1 / 3)
    assert payload["diagnostics"] == []


@pytest.mark.parametrize("first_architecture", [None, "sm_120"])
def test_cli_discloses_missing_architecture_headers(
    tmp_path: Path, first_architecture: str | None
) -> None:
    result = _run(tmp_path, _row("light", 32, first_architecture) + _row("heavy", 128))

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["architecture"] == "sm_120"
    assert payload["ptxas_evidence"]["architecture"] is None
    assert payload["ptxas_evidence"]["architecture_verified"] is False
    assert payload["registers_per_thread"] == 128
    assert any("PTXAS architecture is unverified" in d for d in payload["diagnostics"])


def test_static_cli_does_not_invent_ptxas_provenance(tmp_path: Path) -> None:
    result = _run(tmp_path, None)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["evidence_stage"] == "static"
    assert payload["ptxas_evidence"] is None
    assert "time_estimate" not in payload


def _calibration_file(tmp_path: Path) -> Path:
    """Retain synthetic arithmetic inputs; these are not device measurements."""
    from generativeqc_compiler.common.cuda_time_estimator import CudaTimingCalibration

    calibration = CudaTimingCalibration(
        device="synthetic-device",
        architecture="sm_120",
        sm_count=170,
        workload="synthetic FP64 streaming; FMA=2 ops",
        provenance="unit test only",
        effective_compute_ops_per_second=1e12,
        effective_memory_bytes_per_second=1e11,
        launch_seconds=1e-5,
        saturation_occupancy=0.5,
        uncertainty_fraction=0.25,
    )
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(calibration.to_payload()), encoding="utf-8")
    return path


_WORK = (
    "--operations",
    "1000000000",
    "--traffic-bytes",
    "100000000",
    "--launches",
    "2",
)


def test_calibrated_cli_uses_calibration_topology_and_retains_inputs(
    tmp_path: Path,
) -> None:
    path = _calibration_file(tmp_path)
    result = _run(
        tmp_path,
        _row("kernel", 64, "sm_120"),
        "--calibration",
        str(path),
        "--grid-blocks",
        "680",
        *_WORK,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    timing = payload["time_estimate"]
    assert payload["sm_count"] == 170
    assert timing["estimated_seconds"] == pytest.approx(0.00152)
    assert timing["parallelism_basis"] == "whole-device"
    assert timing["calibration"] == json.loads(path.read_text(encoding="utf-8"))
    assert timing["cost"]["launch_count"] == 2


@pytest.mark.parametrize("mismatched_schema", [False, True])
def test_refined_cli_uses_v2_formula_and_checks_schema(
    tmp_path: Path, mismatched_schema: bool
) -> None:
    path = _calibration_file(tmp_path)
    calibration = json.loads(path.read_text())
    calibration.update(
        schema="generativeqc.compiler.cuda-timing-calibration."
        + ("v1" if mismatched_schema else "v2"),
        model="roofline-calibrated-overlap.v2",
        memory_throughput_curve=[[0, 0], [1, 1]],
        crossover_penalty_curve=[[0, 0.5], [1, 0.5]],
        batch_seconds=2e-5,
    )
    path.write_text(json.dumps(calibration))
    result = _run(
        tmp_path,
        _row("kernel", 64, "sm_120"),
        "--calibration",
        str(path),
        "--grid-blocks",
        "680",
        *_WORK,
    )
    if mismatched_schema:
        assert result.returncode != 0
        assert "schema and model version disagree" in result.stderr
        return
    assert result.returncode == 0, result.stderr
    timing = json.loads(result.stdout)["time_estimate"]
    assert timing["schema"] == "generativeqc.compiler.cuda-time-estimate.v2"
    # Occupancy 1/3: C=1.5 ms, M=3 ms, k=0.5, fixed+launch=40 us.
    assert timing["estimated_seconds"] == pytest.approx(0.003 + 0.000375 + 0.00004)
    assert timing["batch_seconds"] == pytest.approx(2e-5)


@pytest.mark.parametrize("fallback", [False, True])
def test_calibrated_cli_missing_grid_requires_explicit_fallback(
    tmp_path: Path, fallback: bool
) -> None:
    result = _run(
        tmp_path,
        _row("kernel", 64, "sm_120"),
        "--calibration",
        str(_calibration_file(tmp_path)),
        *_WORK,
        *(("--allow-per-sm-fallback",) if fallback else ()),
    )
    assert result.returncode == 0, result.stderr
    timing = json.loads(result.stdout)["time_estimate"]
    assert (timing["estimated_seconds"] is not None) == fallback


@pytest.mark.parametrize("dynamic", [None, "0", "100000000"])
def test_static_cli_dynamic_spill_evidence_is_explicit(
    tmp_path: Path, dynamic: str | None
) -> None:
    result = _run(
        tmp_path,
        None,
        "--calibration",
        str(_calibration_file(tmp_path)),
        "--grid-blocks",
        "680",
        "--estimated-registers",
        "64",
        *_WORK,
        *(("--spill-traffic-bytes", dynamic) if dynamic is not None else ()),
    )
    assert result.returncode == 0, result.stderr
    timing = json.loads(result.stdout)["time_estimate"]
    if dynamic is None:
        assert timing["estimated_seconds"] is None
    else:
        assert timing["memory_seconds"] == pytest.approx(
            0.0015 if dynamic == "0" else 0.003
        )


def test_timing_rejects_multi_kernel_resource_aggregation(tmp_path: Path) -> None:
    result = _run(
        tmp_path,
        _row("first", 32, "sm_120") + _row("second", 128, "sm_120"),
        "--calibration",
        str(_calibration_file(tmp_path)),
        "--grid-blocks",
        "680",
        *_WORK,
    )
    assert result.returncode != 0
    assert result.stdout == ""
    assert "exactly one PTXAS kernel resource row" in result.stderr


@pytest.mark.parametrize(
    "contents",
    [
        "not JSON",
        "[]",
        "{}",
        '{"schema": "unsupported"}',
        '{"schema": "generativeqc.compiler.cuda-timing-calibration.v1"}',
    ],
)
def test_cli_rejects_malformed_calibration_without_output(
    tmp_path: Path, contents: str
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(contents, encoding="utf-8")
    result = _run(tmp_path, None, "--calibration", str(path))
    assert result.returncode != 0
    assert result.stdout == ""
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("options", [("--sm-count", "100"), ("--arch", "sm_80")])
def test_cli_rejects_calibration_target_mismatch(
    tmp_path: Path, options: tuple[str, ...]
) -> None:
    result = _run(
        tmp_path, None, "--calibration", str(_calibration_file(tmp_path)), *options
    )
    assert result.returncode != 0
    assert result.stdout == ""
    assert "does not match calibration" in result.stderr


@pytest.mark.parametrize(
    "options", [("--spill-traffic-bytes", "0"), ("--allow-per-sm-fallback",)]
)
def test_cli_timing_options_require_calibration(
    tmp_path: Path, options: tuple[str, ...]
) -> None:
    result = _run(tmp_path, None, *options)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "timing options require --calibration" in result.stderr
