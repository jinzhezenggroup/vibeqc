"""Malformed native observations must not become accepted CPU sweep evidence."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "cpu_linalg_sweep_record_validation", ROOT / "benchmarks/cpu_linalg_sweep.py"
)
assert SPEC is not None and SPEC.loader is not None
sweep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sweep)


def record(provider: str = "scalar", resolved: str = "scalar") -> dict[str, object]:
    return {
        "schema": sweep.PROBE_SCHEMA,
        "operation": "gemm",
        "m": 16,
        "n": 16,
        "k": 16,
        "transpose_a": "N",
        "transpose_b": "N",
        "repeats": 1,
        "cpu_target": "x86_64-generic",
        "requested_provider": "automatic" if provider == "auto" else provider,
        "provider": resolved,
        "provider_threads": 1,
        "thread_ownership": "task_parallel",
        "seconds": 0.001,
        "gflops": 0.008192,
    }


@pytest.mark.parametrize("field", ["seconds", "gflops"])
@pytest.mark.parametrize(
    "invalid",
    [float("nan"), float("inf"), float("-inf"), True, False, 0, -1, "1", 10**400],
)
def test_measurements_must_be_finite_positive_real_numbers(
    field: str, invalid: object
) -> None:
    payload = record()
    payload[field] = invalid
    with pytest.raises(ValueError, match=field):
        sweep._validated(payload, size=16, provider="scalar", repeats=1, threads=1)


@pytest.mark.parametrize("field", ["m", "n", "k", "repeats", "provider_threads"])
def test_integer_identity_rejects_float_alias(field: str) -> None:
    payload = record()
    payload[field] = float(payload[field])
    with pytest.raises(ValueError, match=field):
        sweep._validated(payload, size=16, provider="scalar", repeats=1, threads=1)


@pytest.mark.parametrize("field", ["repeats", "provider_threads"])
def test_integer_identity_rejects_boolean_alias(field: str) -> None:
    payload = record()
    payload[field] = True
    with pytest.raises(ValueError, match=field):
        sweep._validated(payload, size=16, provider="scalar", repeats=1, threads=1)


@pytest.mark.parametrize("field", ["cpu_target", "provider", "thread_ownership"])
@pytest.mark.parametrize("invalid", [True, ["scalar"], {"name": "scalar"}, " "])
def test_identity_requires_nonempty_strings(field: str, invalid: object) -> None:
    payload = record()
    payload[field] = invalid
    with pytest.raises(ValueError):
        sweep._validated(payload, size=16, provider="scalar", repeats=1, threads=1)


@pytest.mark.parametrize(
    "requested,resolved",
    [
        ("scalar", "openblas"),
        ("openblas", "scalar"),
        ("auto", "automatic"),
        ("auto", "unknown"),
    ],
)
def test_resolved_provider_cannot_contradict_request(
    requested: str, resolved: str
) -> None:
    with pytest.raises(ValueError, match="provider"):
        sweep._validated(
            record(requested, resolved),
            size=16,
            provider=requested,
            repeats=1,
            threads=1,
        )


@pytest.mark.parametrize(
    "requested,resolved",
    [
        ("scalar", "scalar"),
        ("openblas", "openblas"),
        ("auto", "scalar"),
        ("auto", "openblas"),
    ],
)
def test_legal_provider_results_are_copied_without_changes(
    requested: str, resolved: str
) -> None:
    payload = record(requested, resolved)
    before = deepcopy(payload)
    output = sweep._validated(
        payload, size=16, provider=requested, repeats=1, threads=1
    )
    assert output == before == payload
    assert output is not payload


def test_malformed_json_observation_is_rejected_by_complete_sweep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = record()
    payload["seconds"] = float("nan")
    monkeypatch.setattr(
        sweep.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, json.dumps(payload), ""
        ),
    )
    with pytest.raises(ValueError, match="seconds"):
        sweep.run_sweep(Path("probe"), sizes=(16,), providers=("scalar",), repeats=1)


@pytest.mark.parametrize("payload", [None, [], 1, "record"])
def test_non_object_probe_record_is_rejected(payload: object) -> None:
    with pytest.raises(ValueError, match="object"):
        sweep._validated(payload, size=16, provider="scalar", repeats=1, threads=1)
