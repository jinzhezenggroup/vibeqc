"""Tests for the bounded CPU dense-linear-algebra benchmark sweep."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "cpu_linalg_sweep", ROOT / "benchmarks/cpu_linalg_sweep.py"
)
assert SPEC is not None and SPEC.loader is not None
cpu_linalg_sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cpu_linalg_sweep)


def test_default_sweep_is_small_medium_large_and_fail_closed() -> None:
    assert cpu_linalg_sweep.DEFAULT_SIZES == (16, 64, 256)
    assert cpu_linalg_sweep.DEFAULT_PROVIDERS == ("auto", "scalar")
    for value in ("", "0", "16,16", "16,nope"):
        with pytest.raises((argparse.ArgumentTypeError, ValueError)):
            cpu_linalg_sweep.parse_sizes(value)


def test_provider_parser_is_bounded() -> None:
    assert cpu_linalg_sweep.parse_providers("auto,scalar") == ("auto", "scalar")
    for value in ("", "auto,auto", "mkl"):
        with pytest.raises(argparse.ArgumentTypeError):
            cpu_linalg_sweep.parse_providers(value)


def test_sweep_records_exact_probe_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        size = int(command[1])
        provider = command[3]
        payload = {
            "schema": "vibeqc.cpu-linalg-probe.v1",
            "operation": "gemm",
            "m": size,
            "n": size,
            "k": size,
            "transpose_a": "N",
            "transpose_b": "N",
            "repeats": 3,
            "cpu_target": "x86_64-generic",
            "requested_provider": "automatic" if provider == "auto" else provider,
            "provider": "scalar",
            "provider_threads": 1,
            "thread_ownership": "task_parallel",
            "seconds": 0.001,
            "gflops": 2.0,
        }
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")

    monkeypatch.setattr(cpu_linalg_sweep.subprocess, "run", fake_run)
    result = cpu_linalg_sweep.run_sweep(
        Path("probe"), sizes=(16, 64, 256), providers=("auto",), repeats=3
    )
    assert result["schema"] == "vibeqc.cpu-linalg-sweep.v1"
    assert [record["m"] for record in result["records"]] == [16, 64, 256]
    assert calls == [
        ["probe", "16", "3", "auto", "1"],
        ["probe", "64", "3", "auto", "1"],
        ["probe", "256", "3", "auto", "1"],
    ]


def test_sweep_rejects_probe_identity_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "schema": "vibeqc.cpu-linalg-probe.v1",
        "operation": "gemm",
        "m": 15,
        "n": 16,
        "k": 16,
        "repeats": 1,
        "cpu_target": "x86_64-generic",
        "requested_provider": "scalar",
        "provider": "scalar",
        "provider_threads": 1,
        "seconds": 0.001,
        "gflops": 1.0,
    }
    monkeypatch.setattr(
        cpu_linalg_sweep.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, json.dumps(payload), ""
        ),
    )
    with pytest.raises(ValueError, match="unexpected m"):
        cpu_linalg_sweep.run_sweep(
            Path("probe"), sizes=(16,), providers=("scalar",), repeats=1
        )


def test_sweep_surfaces_probe_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cpu_linalg_sweep.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 1, "", "provider unavailable"
        ),
    )
    with pytest.raises(RuntimeError, match="provider unavailable"):
        cpu_linalg_sweep.run_sweep(
            Path("probe"), sizes=(16,), providers=("openblas",), repeats=1
        )


def test_sweep_rejects_transpose_or_thread_ownership_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = {
        "schema": "vibeqc.cpu-linalg-probe.v1",
        "operation": "gemm",
        "m": 16,
        "n": 16,
        "k": 16,
        "transpose_a": "N",
        "transpose_b": "N",
        "repeats": 1,
        "cpu_target": "x86_64-generic",
        "requested_provider": "scalar",
        "provider": "scalar",
        "provider_threads": 1,
        "thread_ownership": "task_parallel",
        "seconds": 0.001,
        "gflops": 1.0,
    }

    for key, value, match in (
        ("transpose_a", "T", "unexpected transpose_a"),
        ("thread_ownership", "provider_parallel", "must remain task-parallel"),
    ):
        payload = dict(base)
        payload[key] = value
        monkeypatch.setattr(
            cpu_linalg_sweep.subprocess,
            "run",
            lambda *args, payload=payload, **kwargs: subprocess.CompletedProcess(
                args[0], 0, json.dumps(payload), ""
            ),
        )
        with pytest.raises(ValueError, match=match):
            cpu_linalg_sweep.run_sweep(
                Path("probe"), sizes=(16,), providers=("scalar",), repeats=1
            )


def test_multithread_probe_requires_provider_owned_parallelism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "schema": "vibeqc.cpu-linalg-probe.v1",
        "operation": "gemm",
        "m": 16,
        "n": 16,
        "k": 16,
        "transpose_a": "N",
        "transpose_b": "N",
        "repeats": 1,
        "cpu_target": "x86_64-generic",
        "requested_provider": "scalar",
        "provider": "scalar",
        "provider_threads": 2,
        "thread_ownership": "task_parallel",
        "seconds": 0.001,
        "gflops": 1.0,
    }
    monkeypatch.setattr(
        cpu_linalg_sweep.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, json.dumps(payload), ""
        ),
    )
    with pytest.raises(ValueError, match="must own parallelism"):
        cpu_linalg_sweep.run_sweep(
            Path("probe"), sizes=(16,), providers=("scalar",), repeats=1, threads=2
        )
