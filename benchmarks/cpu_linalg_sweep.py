"""Run a bounded small/medium/large sweep through cpu_linalg_probe."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

try:
    from benchmarks._retention import raw_output_path
except ModuleNotFoundError:
    from _retention import raw_output_path

SWEEP_SCHEMA = "vibeqc.cpu-linalg-sweep.v1"
PROBE_SCHEMA = "vibeqc.cpu-linalg-probe.v1"
DEFAULT_SIZES = (16, 64, 256)
DEFAULT_PROVIDERS = ("auto", "scalar")


def parse_sizes(value: str) -> tuple[int, ...]:
    try:
        sizes = tuple(int(item) for item in value.split(",") if item)
    except ValueError as error:
        raise argparse.ArgumentTypeError("sizes must be integers") from error
    if not sizes or any(size <= 0 for size in sizes) or len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError("sizes must be positive and unique")
    return sizes


def parse_providers(value: str) -> tuple[str, ...]:
    providers = tuple(item for item in value.split(",") if item)
    allowed = {"auto", "scalar", "openblas"}
    if (
        not providers
        or any(provider not in allowed for provider in providers)
        or len(set(providers)) != len(providers)
    ):
        raise argparse.ArgumentTypeError(
            "providers must be unique selections from auto,scalar,openblas"
        )
    return providers


def _validated(
    record: dict[str, Any], *, size: int, provider: str, repeats: int, threads: int
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise TypeError("probe record must be a JSON object")
    expected_provider = "automatic" if provider == "auto" else provider
    expected = {
        "schema": PROBE_SCHEMA,
        "operation": "gemm",
        "m": size,
        "n": size,
        "k": size,
        "requested_provider": expected_provider,
        "repeats": repeats,
        "provider_threads": threads,
        "transpose_a": "N",
        "transpose_b": "N",
    }
    for key, value in expected.items():
        if type(record.get(key)) is not type(value) or record[key] != value:
            raise ValueError(f"probe record has unexpected {key}: {record.get(key)!r}")
    for key in ("cpu_target", "provider", "thread_ownership"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"probe record has invalid {key} identity")
    resolved = record["provider"]
    if resolved not in {"scalar", "openblas"} or (
        provider != "auto" and resolved != provider
    ):
        raise ValueError("probe record has inconsistent resolved provider identity")
    ownership = record.get("thread_ownership")
    if ownership not in {"task_parallel", "provider_parallel"}:
        raise ValueError("probe record omitted valid thread ownership")
    if threads > 1 and ownership != "provider_parallel":
        raise ValueError("multi-thread provider probe must own parallelism")
    if provider != "openblas" and threads == 1 and ownership != "task_parallel":
        raise ValueError("single-thread auto/scalar probe must remain task-parallel")
    for key in ("seconds", "gflops"):
        value = record.get(key)
        try:
            valid = type(value) in (int, float) and math.isfinite(value) and value > 0
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(f"probe record has invalid {key}")
    return dict(record)


def run_sweep(
    probe: Path,
    *,
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    providers: tuple[str, ...] = DEFAULT_PROVIDERS,
    repeats: int = 5,
    threads: int = 1,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    # Validate the public Python entry point as well as CLI-parsed arguments.
    sizes, providers = tuple(sizes), tuple(providers)
    if (
        not sizes
        or any(type(size) is not int or size <= 0 for size in sizes)
        or len(set(sizes)) != len(sizes)
    ):
        raise ValueError("sizes must be positive, unique integers")
    if (
        not providers
        or any(
            type(provider) is not str or provider not in {"auto", "scalar", "openblas"}
            for provider in providers
        )
        or len(set(providers)) != len(providers)
    ):
        raise ValueError("providers must be nonempty, supported and unique")
    if any(type(value) is not int or value <= 0 for value in (repeats, threads)):
        raise ValueError("repeats and threads must be positive integers")
    try:
        valid_timeout = (
            type(timeout_seconds) in (int, float)
            and math.isfinite(timeout_seconds)
            and timeout_seconds > 0
        )
    except OverflowError:
        valid_timeout = False
    if not valid_timeout:
        raise ValueError("timeout_seconds must be finite and positive")
    records: list[dict[str, Any]] = []
    for provider in providers:
        for size in sizes:
            try:
                completed = subprocess.run(
                    [str(probe), str(size), str(repeats), provider, str(threads)],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired as error:
                # subprocess.run kills and reaps this probe before raising.
                # Do not publish a completed sweep after a timed-out sample.
                raise RuntimeError(
                    f"probe timed out for provider={provider} size={size} "
                    f"after {timeout_seconds} seconds"
                ) from error
            if completed.returncode:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise RuntimeError(
                    f"probe failed for provider={provider} size={size}: {detail}"
                )
            records.append(
                _validated(
                    json.loads(completed.stdout),
                    size=size,
                    provider=provider,
                    repeats=repeats,
                    threads=threads,
                )
            )
    return {
        "schema": SWEEP_SCHEMA,
        "operation": "gemm",
        "sizes": list(sizes),
        "requested_providers": list(providers),
        "repeats": repeats,
        "provider_threads": threads,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--sizes", type=parse_sizes, default=DEFAULT_SIZES)
    parser.add_argument("--providers", type=parse_providers, default=DEFAULT_PROVIDERS)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="finite positive wall-time limit for each probe invocation (default: 120)",
    )
    parser.add_argument("--output", type=raw_output_path)
    args = parser.parse_args()
    payload = run_sweep(
        args.probe,
        sizes=args.sizes,
        providers=args.providers,
        repeats=args.repeats,
        threads=args.threads,
        timeout_seconds=args.timeout_seconds,
    )
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        destination = raw_output_path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
