"""Compare the complete native CPU RCCSD(T) force endpoint with/without tile bundles.

Run from an installed Linux checkout with VIBEQC_LIBRARY set to its CPU native
library. Separate child processes and fresh per-route caches keep cold and
warm-cache measurements distinct. The separate route disables only the new
triples-tile prewarm; all other production consumers and equations are identical.
"""

from __future__ import annotations

import argparse
import collections
import gc
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _checkpoint(path: Path, report: dict[str, object]) -> None:
    """Keep each completed child record even if a later process fails."""

    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _failure(
    error: BaseException,
    *,
    phase: str,
    mode: str | None = None,
    stdout: str = "",
    stderr: str = "",
) -> dict[str, str]:
    result = {
        "phase": phase,
        "exception": type(error).__name__,
        "message": str(error),
        "stdout_tail": stdout[-4096:],
        "stderr_tail": stderr[-16384:],
    }
    if mode is not None:
        result["mode"] = mode
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite child JSON constant: {value}")


def _rss() -> dict[str, int]:
    fields = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in {"VmRSS", "VmHWM"}:
            fields[key] = int(value.strip().split()[0]) * 1024
    return fields


def _run_child(mode: str, case: str, cache: Path) -> dict[str, object]:
    from vibeqc_compiler.tensor.cpu import emit_cpu

    from tools.cc_gradient_fixtures import inputs, source_arguments
    from tools.vibeqc_cc import (
        lambda_solver,
        native_tensor_cpu,
        rccsd_t_force,
        triples_complete_gradient,
        triples_lambda_response,
    )
    from tools.vibeqc_posthf.sources import NativeSource

    os.environ["VIBEQC_TENSOR_CACHE"] = str(cache)
    before = {path.resolve() for path in cache.rglob("runtime.so")}
    inventory: collections.Counter[str] = collections.Counter()
    programs = {}
    owners = []
    construction_seconds = 0.0
    original_single = native_tensor_cpu.NativeTensorProgram
    original_bundle = native_tensor_cpu.NativeTensorProgramBundle
    original_owner = lambda_solver.NativeCCTensorExecutor
    original_triples = triples_lambda_response.accumulate_tile_triples_vjp
    original_export = triples_complete_gradient.export_rhf

    def fixed_generation(source: object, **kwargs: object) -> object:
        # A new generation UUID is correct for ordinary live owners, but makes
        # cross-process response identities incomparable in this benchmark.
        return original_export(
            source, generation_id=f"ccsdt-cpu-bundle-{case}", **kwargs
        )

    def timed(constructor: object) -> object:
        def construct(*args: object, **kwargs: object) -> object:
            nonlocal construction_seconds
            start = time.perf_counter()
            result = constructor(*args, **kwargs)
            construction_seconds += time.perf_counter() - start
            return result

        return construct

    def owner(**kwargs: object) -> object:
        result = original_owner(**kwargs)
        owners.append(result)
        execute = result.execute

        def traced(program: object, feeds: object) -> object:
            identity = program.logical_hash
            inventory[identity] += 1
            programs[identity] = program
            return execute(program, feeds)

        result.execute = traced
        return result

    class SeparateTileExecutor:
        def __init__(self, underlying: object) -> None:
            self.underlying = underlying

        def execute(self, program: object, feeds: object) -> object:
            return self.underlying.execute(program, feeds)

    def separate_triples(*args: object, **kwargs: object) -> object:
        kwargs["executor"] = SeparateTileExecutor(kwargs["executor"])
        return original_triples(*args, **kwargs)

    with (
        patch.object(native_tensor_cpu, "NativeTensorProgram", timed(original_single)),
        patch.object(
            native_tensor_cpu, "NativeTensorProgramBundle", timed(original_bundle)
        ),
        patch.object(lambda_solver, "NativeCCTensorExecutor", owner),
        patch.object(triples_complete_gradient, "export_rhf", fixed_generation),
        patch.object(
            triples_lambda_response,
            "accumulate_tile_triples_vjp",
            separate_triples if mode == "separate" else original_triples,
        ),
    ):
        started = time.perf_counter()
        with NativeSource(**source_arguments(inputs(case))) as source:
            result = rccsd_t_force(source, vir_chunk_size=1)
        endpoint_wall = time.perf_counter() - started

    if len(owners) != 1:
        raise RuntimeError(f"expected one response executor, found {len(owners)}")
    executor = owners[0]
    artifacts = tuple(executor._artifacts())
    if len(artifacts) != executor.artifact_count:
        raise RuntimeError("artifact inventory does not match executor")
    fresh = tuple(
        artifact for artifact in artifacts if artifact.library.resolve() not in before
    )
    gc.collect()
    memory = _rss()
    scalar_work = 0
    required_max = 0
    for identity, program in programs.items():
        _, resource = emit_cpu(
            program,
            max_bytes=executor.max_bytes,
            max_work=executor.max_work,
            max_nodes=executor.max_nodes,
        )
        scalar_work += inventory[identity] * resource["scalar_work"]
        required_max = max(required_max, resource["required_bytes"])
    return {
        "mode": mode,
        "case": case,
        "wall_seconds": endpoint_wall,
        "artifact_construction_seconds": construction_seconds,
        "cold_compile_link_seconds": sum(
            float(artifact.metadata["compile_seconds"]) for artifact in fresh
        ),
        "new_artifact_count": len(fresh),
        "artifact_count": executor.artifact_count,
        "program_count": executor.compiled_program_count,
        "bundled_program_count": executor.bundled_program_count,
        "binary_bytes": executor.binary_bytes,
        "artifact_sha256": sorted(
            artifact.metadata["binary_sha256"] for artifact in artifacts
        ),
        "tensorir_execution_counts": dict(sorted(inventory.items())),
        "tensorir_scalar_work": scalar_work,
        "tensorir_max_required_bytes": required_max,
        "rss_after_bytes": memory["VmRSS"],
        "rss_peak_bytes": memory["VmHWM"],
        "total_energy": result.total_energy,
        "correlation_energy": result.correlation_energy,
        "gradient": result.gradient.tolist(),
        "cc_state_identity": result.cc_state_identity,
        "response_identity": result.response_identity,
        "z_operator_actions": result.diagnostics["z_operator_actions"],
        "logical_reserved_host_bytes": result.diagnostics[
            "logical_reserved_host_bytes"
        ],
        "compiler": str(executor.compiler.cxx),
        "compiler_version": subprocess.check_output(
            [str(executor.compiler.cxx), "--version"], text=True
        ).splitlines()[0],
    }


def _compare(records: dict[str, dict[str, dict[str, object]]], case: str) -> None:
    from tools.validate_ccsd_t_gradient import analytic_oracle

    expected = analytic_oracle(case)["analytic"]
    baseline = records["separate"]["cold"]
    for mode in ("separate", "bundled"):
        for phase in ("cold", "warm"):
            record = records[mode][phase]
            if (
                record["tensorir_execution_counts"]
                != baseline["tensorir_execution_counts"]
            ):
                raise RuntimeError("TensorIR program execution inventory changed")
            if record["tensorir_scalar_work"] != baseline["tensorir_scalar_work"]:
                raise RuntimeError("TensorIR semantic work changed")
            if record["cc_state_identity"] != baseline["cc_state_identity"]:
                raise RuntimeError("accepted CC state changed")
            if record["response_identity"] != baseline["response_identity"]:
                raise RuntimeError("response identity changed")
            if record["z_operator_actions"] != baseline["z_operator_actions"]:
                raise RuntimeError("orbital response work changed")
            if (
                record["logical_reserved_host_bytes"]
                != baseline["logical_reserved_host_bytes"]
            ):
                raise RuntimeError("logical response reservation changed")
            np.testing.assert_allclose(
                record["gradient"], expected["gradient"], atol=1e-6, rtol=0
            )
            np.testing.assert_allclose(
                record["total_energy"], expected["total_energy"], atol=1e-8, rtol=0
            )
            np.testing.assert_allclose(
                record["gradient"], baseline["gradient"], atol=1e-12, rtol=0
            )
            if phase == "warm" and record["new_artifact_count"] != 0:
                raise RuntimeError(
                    "warm-cache route unexpectedly compiled a new artifact"
                )
    if records["bundled"]["cold"]["artifact_count"] >= baseline["artifact_count"]:
        raise RuntimeError("triples bundle did not reduce artifact count")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("h2o", "nh3"), default="h2o")
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child-mode", choices=("separate", "bundled"))
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("the complete native CPU benchmark requires Linux")
    if args.child_mode:
        print(json.dumps(_run_child(args.child_mode, args.case, args.cache_root)))
        return
    if args.cache_root.exists() or args.output.exists():
        parser.error("cache root and output must not already exist")
    args.cache_root.mkdir(parents=True)
    records = {"separate": {}, "bundled": {}}
    report = {
        "schema": "vibeqc.ccsdt.cpu_bundle_endpoint/1",
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "case": args.case,
        "vir_chunk_size": 1,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "measurements": records,
        "oracle": "pinned PySCF 2.14.0 analytic RCCSD(T) gradient",
        "qualified": False,
    }
    _checkpoint(args.output, report)
    for mode in ("separate", "bundled"):
        cache = args.cache_root / mode
        for phase in ("cold", "warm"):
            completed = None
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--case",
                        args.case,
                        "--cache-root",
                        str(cache),
                        "--output",
                        str(args.output),
                        "--child-mode",
                        mode,
                    ],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                records[mode][phase] = json.loads(
                    completed.stdout, parse_constant=_reject_constant
                )
                _checkpoint(args.output, report)
            except BaseException as error:
                stdout = (
                    completed.stdout
                    if completed is not None
                    else getattr(error, "stdout", "")
                )
                stderr = (
                    completed.stderr
                    if completed is not None
                    else getattr(error, "stderr", "")
                )
                report["failure"] = _failure(
                    error,
                    mode=mode,
                    phase=phase,
                    stdout=stdout or "",
                    stderr=stderr or "",
                )
                _checkpoint(args.output, report)
                if isinstance(error, subprocess.CalledProcessError):
                    print(stdout, file=sys.stderr)
                    print(stderr, file=sys.stderr)
                raise
            print(
                f"{mode} {phase}: {records[mode][phase]['wall_seconds']:.3f}s, "
                f"{records[mode][phase]['artifact_count']} artifacts",
                file=sys.stderr,
            )
    try:
        _compare(records, args.case)
    except BaseException as error:
        report["failure"] = _failure(error, phase="qualification")
        _checkpoint(args.output, report)
        raise
    report["qualified"] = True
    _checkpoint(args.output, report)


if __name__ == "__main__":
    main()
