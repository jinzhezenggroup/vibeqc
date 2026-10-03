"""CPU-only independent fitting oracle and train/holdout isolation checks."""

import copy
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.calibrate_cuda_time import collect, fit


def _measurement() -> dict:
    """Known piecewise curve; arithmetic is independent of the estimator code."""
    rows = []
    for split, grids in (("train", [3, 6, 12, 24]), ("holdout", [9, 18])):
        for grid in grids:
            for family in ("fma", "copy"):
                operations, traffic = (
                    (1_000_000, 8000) if family == "fma" else (0, 1_000_000)
                )
                # 4 SMs * 1536 threads/SM, 128 threads/block, threshold 1/4.
                scale = min(1, grid / 12)
                seconds = 10 * (max(operations / 1e9, traffic / 1e8) / scale + 1e-6)
                rows.append(
                    {
                        "id": f"{split}-{family}-{grid}",
                        "split": split,
                        "family": family,
                        "grid_blocks": grid,
                        "block_threads": 128,
                        "launch_count": 10,
                        "operations_per_launch": operations,
                        "bytes_per_launch": traffic,
                        "registers_per_thread": 32,
                        "shared_bytes": 0,
                        "local_bytes": 0,
                        "max_absolute_error": 0.0,
                        "wall_seconds": [seconds] * 3,
                    }
                )
    for split, launches in (("train", 256), ("train", 1024), ("holdout", 512)):
        rows.append(
            {
                "id": f"{split}-launch-{launches}",
                "split": split,
                "family": "launch",
                "grid_blocks": 1,
                "block_threads": 128,
                "launch_count": launches,
                "operations_per_launch": 0,
                "bytes_per_launch": 0,
                "registers_per_thread": 4,
                "shared_bytes": 0,
                "local_bytes": 0,
                "max_absolute_error": 0.0,
                "wall_seconds": [launches * 1e-6] * 3,
            }
        )
    return {
        "schema": "generativeqc.cuda-timing-probe.v1",
        "device": "synthetic",
        "architecture": "sm_120",
        "sm_count": 4,
        "runtime_version": 12090,
        "cases": rows,
    }


def test_fitter_recovers_known_rates_and_saturation() -> None:
    calibration, report = fit(_measurement(), "synthetic")
    assert calibration.saturation_occupancy == 0.25
    assert calibration.effective_compute_ops_per_second == pytest.approx(1e9)
    assert calibration.effective_memory_bytes_per_second == pytest.approx(1e8)
    assert calibration.launch_seconds == pytest.approx(1e-6)
    assert calibration.uncertainty_fraction == 0.10
    assert report["qualified"]
    assert report["holdout"]["max_relative_error"] < 1e-12


def test_holdout_failures_do_not_train_parameters_or_error_band() -> None:
    original = _measurement()
    altered = copy.deepcopy(original)
    for row in altered["cases"]:
        if row["split"] == "holdout":
            row["wall_seconds"] = [10 * value for value in row["wall_seconds"]]
    calibration, _ = fit(original, "same-input-identity-for-comparison")
    second, report = fit(altered, "same-input-identity-for-comparison")
    assert calibration == second
    assert not report["qualified"]
    assert report["holdout"]["max_relative_error"] == pytest.approx(0.9)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_absolute_error", 1e-3),
        ("max_absolute_error", float("nan")),
        ("local_bytes", 8),
        ("wall_seconds", [float("nan")]),
        ("wall_seconds", [0.0]),
        ("wall_seconds", []),
        ("split", "unknown"),
        ("launch_count", 0),
        ("launch_count", True),
    ],
)
def test_unusable_measurements_fail_closed(field: str, value: object) -> None:
    measurement = _measurement()
    measurement["cases"][0][field] = value
    with pytest.raises(ValueError):
        fit(measurement, "synthetic")


@pytest.mark.parametrize("missing", ["fma", "copy", "launch", "holdout"])
def test_calibration_needs_training_families_and_holdout(missing: str) -> None:
    measurement = _measurement()
    measurement["cases"] = [
        row
        for row in measurement["cases"]
        if row["family"] != missing and row["split"] != missing
    ]
    with pytest.raises(ValueError):
        fit(measurement, "synthetic")


def test_duplicate_case_identity_is_rejected() -> None:
    measurement = _measurement()
    measurement["cases"].append(copy.deepcopy(measurement["cases"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        fit(measurement, "synthetic")


def test_unidentifiable_single_occupancy_is_rejected() -> None:
    measurement = _measurement()
    for row in measurement["cases"]:
        row["grid_blocks"] = 3
    with pytest.raises(ValueError, match="multiple distinct occupancies"):
        fit(measurement, "synthetic")


def test_collection_refuses_to_bypass_scheduler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="Slurm GPU allocation"):
        collect(Path("nvcc"), "sm_120", 7, tmp_path)


@pytest.mark.parametrize("fail_holdout", [False, True])
@pytest.mark.parametrize("compressed", [False, True])
def test_cpu_replay_cli_preserves_samples_and_reports_failed_gate(
    tmp_path: Path, fail_holdout: bool, compressed: bool
) -> None:
    measurement = _measurement()
    if fail_holdout:
        for row in measurement["cases"]:
            if row["split"] == "holdout":
                row["wall_seconds"] = [value * 10 for value in row["wall_seconds"]]
    raw = json.dumps(measurement).encode()
    path = tmp_path / ("retained.json.gz" if compressed else "retained.json")
    path.write_bytes(gzip.compress(raw, mtime=0) if compressed else raw)
    destination = tmp_path / "replayed"
    runner = Path(__file__).resolve().parents[2] / "tools/calibrate_cuda_time.py"
    process = subprocess.run(
        [
            sys.executable,
            str(runner),
            "--measurement",
            str(path),
            "--output",
            str(destination),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert process.returncode == int(fail_holdout), process.stderr
    assert (destination / "measurement.json").read_bytes() == raw
    report = json.loads((destination / "qualification.json").read_text())
    assert report["measurement_sha256"] == hashlib.sha256(raw).hexdigest()
    assert report["qualified"] is not fail_holdout


def test_retained_rtx5090_measurements_reproduce_profile_and_qualification() -> None:
    """The published calibration must remain bound to its complete raw samples."""
    bundle = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/results/cuda-timing-rtx5090-20261003"
    )
    data = gzip.decompress((bundle / "measurement.json.gz").read_bytes())
    measurement = json.loads(data)
    expected = json.loads((bundle / "calibration.json").read_text())
    retained = json.loads(
        gzip.decompress((bundle / "qualification.json.gz").read_bytes())
    )
    calibration, qualification = fit(measurement, hashlib.sha256(data).hexdigest())
    for field, value in calibration.to_payload().items():
        assert expected[field] == (
            pytest.approx(value, rel=1e-12) if isinstance(value, float) else value
        )
    assert qualification["qualified"] is retained["qualified"] is True
    assert qualification["measurement_sha256"] == retained["measurement_sha256"]
    assert len(measurement["cases"]) == 40
    assert all(len(row["wall_seconds"]) == 9 for row in measurement["cases"])
    assert len(qualification["cases"]) == len(retained["cases"])
    for actual, expected_row in zip(
        qualification["cases"], retained["cases"], strict=True
    ):
        assert actual["id"] == expected_row["id"]
        assert actual["predicted_seconds"] == pytest.approx(
            expected_row["predicted_seconds"], rel=1e-12
        )
        assert actual["relative_error"] == pytest.approx(
            expected_row["relative_error"], rel=1e-12
        )
    sanitizer = json.loads((bundle / "sanitizer.json").read_text())
    assert sanitizer["binary_sha256"] == measurement["provenance"]["binary_sha256"]
    assert sanitizer["memory_errors"] == sanitizer["exit_code"] == 0
