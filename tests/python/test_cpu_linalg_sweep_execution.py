"""A bounded sweep must reject empty work and terminate stalled probe processes."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "cpu_sweep_execution", ROOT / "benchmarks/cpu_linalg_sweep.py"
)
assert SPEC is not None and SPEC.loader is not None
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)


@pytest.mark.parametrize(
    "arguments",
    [
        {"sizes": ()},
        {"providers": ()},
        {"sizes": (0,)},
        {"sizes": (True,)},
        {"sizes": (16, 16)},
        {"providers": ("unknown",)},
        {"providers": ("auto", "auto")},
        {"repeats": True},
        {"threads": 1.5},
    ],
)
def test_invalid_work_is_rejected_before_starting_a_probe(
    arguments: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("invalid sweep started a process")

    monkeypatch.setattr(sweep.subprocess, "run", forbidden)
    with pytest.raises(ValueError):
        sweep.run_sweep(Path("probe"), **arguments)


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf"), 10**400])
def test_deadline_must_be_finite_and_positive(timeout: object) -> None:
    with pytest.raises(ValueError, match="timeout"):
        sweep.run_sweep(Path("probe"), timeout_seconds=timeout)


def test_default_sweep_applies_deadline_to_every_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def observe(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert kwargs.get("timeout") == 120.0
        calls.append(command)
        size, repeats, provider, threads = command[1:]
        payload = {
            "schema": sweep.PROBE_SCHEMA,
            "operation": "gemm",
            "m": int(size),
            "n": int(size),
            "k": int(size),
            "repeats": int(repeats),
            "provider_threads": int(threads),
            "transpose_a": "N",
            "transpose_b": "N",
            "requested_provider": "automatic" if provider == "auto" else provider,
            "provider": "scalar",
            "cpu_target": "test-cpu",
            "thread_ownership": "task_parallel",
            "seconds": 0.001,
            "gflops": 1.0,
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(sweep.subprocess, "run", observe)
    result = sweep.run_sweep(Path("probe"))
    assert len(calls) == len(result["records"]) == 6
    assert result["sizes"] == [16, 64, 256]
    assert result["requested_providers"] == ["auto", "scalar"]


def test_timeout_reports_exact_failed_sample(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def expire(command: list[str], **kwargs: object) -> None:
        calls.append(command)
        assert kwargs["timeout"] == 0.25
        raise subprocess.TimeoutExpired(command, 0.25)

    monkeypatch.setattr(sweep.subprocess, "run", expire)
    with pytest.raises(RuntimeError, match="provider=scalar size=16.*0.25"):
        sweep.run_sweep(Path("probe"), providers=("scalar",), timeout_seconds=0.25)
    assert len(calls) == 1


def test_real_stalled_probe_does_not_publish_evidence(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    source, probe = tmp_path / "slow.cpp", tmp_path / "slow"
    source.write_text(
        "#include <chrono>\n#include <thread>\n#include <iostream>\n"
        "int main() { std::this_thread::sleep_for(std::chrono::seconds(1)); "
        'std::cout << "{}\\n"; }\n',
        encoding="utf-8",
    )
    subprocess.run(
        [compiler, "-std=c++17", str(source), "-o", str(probe)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = tmp_path / "evidence.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarks/cpu_linalg_sweep.py"),
            "--probe",
            str(probe),
            "--sizes",
            "16",
            "--providers",
            "scalar",
            "--timeout-seconds",
            "0.05",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode != 0
    assert "probe timed out for provider=scalar size=16" in completed.stderr
    assert not output.exists()
