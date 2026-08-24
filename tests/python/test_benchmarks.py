import json
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


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

