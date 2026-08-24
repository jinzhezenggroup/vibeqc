import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _benchmark_support_module():
    """Load benchmark helpers without turning the scripts into a package."""

    path = REPOSITORY_ROOT / "benchmarks" / "_support.py"
    spec = importlib.util.spec_from_file_location("qce_benchmark_support", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batch_benchmark_writes_reproducible_json(tmp_path):
    """Keep benchmark artifacts tied to raw samples and exact source state."""

    output = tmp_path / "batch.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "python")
    environment["QCE_LIBRARY"] = str(REPOSITORY_ROOT / "build" / "libqce.so")
    completed = subprocess.run(
        (
            sys.executable,
            str(REPOSITORY_ROOT / "benchmarks" / "batch_throughput.py"),
            "--device",
            "cpu",
            "--batch",
            "2",
            "--repeats",
            "2",
            "--output",
            str(output),
        ),
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "JSON result:" in completed.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["benchmark"] == "batch_throughput"
    assert payload["settings"]["batch_size"] == 2
    assert len(payload["result"]["timings_seconds"]["warm_batches"]) == 2
    assert payload["result"]["executed_backend"] == "cpu_reference"
    assert payload["environment"]["git"]["commit"]
    assert payload["environment"]["packages"]["numpy"]


def test_gpu_comparison_help_does_not_require_an_allocated_device():
    """Keep benchmark discovery usable on scheduler login nodes."""

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPOSITORY_ROOT / "python")
    environment["CUDA_VISIBLE_DEVICES"] = ""
    for script in (
        "compare_gpu4pyscf.py",
        "compare_gpu4pyscf_batch.py",
    ):
        completed = subprocess.run(
            (
                sys.executable,
                str(REPOSITORY_ROOT / "benchmarks" / script),
                "--help",
            ),
            cwd=REPOSITORY_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "--output" in completed.stdout
        assert "oh-def2-svp-uhf" in completed.stdout
        if script == "compare_gpu4pyscf_batch.py":
            assert "--minimum-speedup" in completed.stdout
            assert "--maximum-energy-error" in completed.stdout
            assert "--maximum-force-error" in completed.stdout


def test_gpu_comparison_gate_reports_all_threshold_failures():
    """Keep allocated benchmark gates deterministic and independently testable."""

    support = _benchmark_support_module()
    failures = support.benchmark_gate_failures(
        speedup=3.0,
        maximum_energy_error=4.0e-12,
        maximum_force_error=5.0e-12,
        minimum_speedup=4.0,
        maximum_energy_error_limit=2.0e-12,
        maximum_force_error_limit=3.0e-12,
    )
    assert len(failures) == 3
    assert "speedup" in failures[0]
    assert "energy error" in failures[1]
    assert "force error" in failures[2]
    assert support.benchmark_gate_failures(
        speedup=4.0,
        maximum_energy_error=2.0e-12,
        maximum_force_error=3.0e-12,
        minimum_speedup=4.0,
        maximum_energy_error_limit=2.0e-12,
        maximum_force_error_limit=3.0e-12,
    ) == []
