"""Qualify opt-in TensorIR CUB BlockReduce against generated CUDA.

This runner deliberately exercises one fixed FP64 reduction shape through the
existing complete TensorIR endpoint: host validation/staging, H2D, reduction,
D2H, and detached output allocation.  It records matched generated/CUB source,
compile, PTXAS, workspace, numerical, and ABBA timing evidence.  The run never
changes the production default; its reviewed result may only support a later
promotion decision for the measured target and shape.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

try:
    from benchmarks._support import environment_metadata, raw_output_path, write_result
except ModuleNotFoundError:
    from _support import environment_metadata, raw_output_path, write_result
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.common.evidence import (
    canonical_hash,
    file_hash,
    new_evidence,
    outcome,
    write_evidence,
)
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    input_tensor,
    reduce_sum,
)
from vibeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda
from vibeqc_compiler.tensor.cuda_tune import tune_cuda

SCHEMA = "vibeqc.tensor.cub-qualification.v1"
ROWS = 65
INNER = 4097
ATOL = 1e-11
RTOL = 1e-10


def _verified_archive_digest(path: Path, expected: str) -> str:
    """Bind the extracted snapshot to the caller's reviewed archive bytes."""

    if not path.is_file():
        raise ValueError("source archive is unavailable")
    actual = file_hash(path)
    if actual != expected:
        raise ValueError("source archive SHA-256 mismatch")
    return actual


def qualification_case(
    rows: int = ROWS, inner: int = INNER
) -> tuple[Program, tuple[dict[str, np.ndarray], ...]]:
    """Return one equation and two deterministic, layout-distinct fixtures."""

    if type(rows) is not int or type(inner) is not int or rows < 1 or inner < 32:
        raise ValueError("qualification requires positive rows and inner >= 32")
    i = Index("cub_rows", IndexSpace("cub_rows", "batch", rows))
    k = Index("cub_inner", IndexSpace("cub_inner", "batch", inner))
    x = input_tensor("x", TensorSpec((i, k), dtype="float64", role="input"))
    program = Program({"result": reduce_sum(x, (1,))})
    count = rows * inner
    ascending = np.linspace(-0.25, 0.75, count, dtype=np.float64).reshape(rows, inner)
    storage = np.linspace(1.0, -0.5, count * 2, dtype=np.float64).reshape(
        rows, inner * 2
    )
    strided = storage[:, ::2]
    if strided.flags.c_contiguous:
        raise RuntimeError("second qualification fixture must retain caller strides")
    return program, ({"x": ascending}, {"x": strided})


def _cub_candidate(evidence: dict[str, Any]) -> dict[str, Any]:
    matches = [
        row
        for row in evidence["candidates"]
        if row.get("plan", {}).get("schedule", {}).get("reduction_provider") == "cub"
    ]
    if len(matches) != 1:
        raise RuntimeError("qualification did not retain exactly one CUB candidate")
    candidate = matches[0]
    required = (
        "artifact",
        "gates",
        "max_absolute_error",
        "plan",
        "plan_identity",
        "profiles",
        "samples",
        "shared_gates",
    )
    missing = [name for name in required if name not in candidate]
    if missing:
        raise RuntimeError(
            "CUB candidate did not complete numerical/endpoint qualification: "
            + ", ".join(missing)
            + (f"; {candidate.get('reason')}" if candidate.get("reason") else "")
        )
    if candidate["status"] not in ("accepted", "rejected"):
        raise RuntimeError(
            f"unexpected CUB qualification status {candidate['status']!r}"
        )
    return candidate


def _flatten_samples(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    return [sample for fixture in candidate["samples"] for sample in fixture]


def _resource_metrics(candidate: dict[str, Any]) -> tuple[int, int]:
    metrics = [fixture["candidate"] for fixture in candidate["profiles"]]
    allocated = max(
        int(row["owned_device_bytes"]) + int(row["provider_retained_bytes"])
        for row in metrics
    )
    peak = max(int(row["predicted_peak_bytes"]) for row in metrics)
    return allocated, peak


def _validation_record(
    *,
    program: Program,
    fixtures: tuple[dict[str, np.ndarray], ...],
    environment: dict[str, Any],
    tuning: dict[str, Any],
    candidate: dict[str, Any],
    allocation_id: str,
    source_revision: str,
    source_archive_sha256: str,
) -> dict[str, Any]:
    fixture_identity = [
        {
            name: {
                "shape": list(value.shape),
                "strides": list(value.strides),
                "dtype": value.dtype.str,
                "values": canonical_hash(value.tolist()),
            }
            for name, value in sorted(fixture.items())
        }
        for fixture in fixtures
    ]
    rows, inner = fixtures[0]["x"].shape
    inputs_hash = canonical_hash(
        {"equation": program.logical_hash, "fixtures": fixture_identity}
    )
    artifact = candidate["artifact"]
    allocated, peak = _resource_metrics(candidate)
    record = new_evidence(
        tier="endpoint",
        subject="TensorIR FP64 CUB BlockReduce opt-in qualification",
        inputs_hash=inputs_hash,
    )
    record.update(
        revision=source_revision,
        hashes={
            "equation": program.logical_hash,
            "ir": canonical_hash(program.to_payload()),
            "source": artifact["identity"]["generated"],
            "schedule": candidate["plan_identity"],
        },
        device=tuning["device"],
        toolchain={
            "cuda": artifact["identity"]["toolchain"],
            "host_compiler": artifact["identity"]["host_compiler"],
        },
        settings={
            "device": "cuda",
            "fast_compile": False,
            "comparison_kind": "kernel",
            "fixed_state_hash": inputs_hash,
            "allocation_id": allocation_id,
            "source_archive_sha256": source_archive_sha256,
            "precision": "float64->float64",
            "rows": rows,
            "inner": inner,
            "fixtures": fixture_identity,
            "numerical_gate": {"atol": ATOL, "rtol": RTOL},
            "timing_scope": (
                "complete prepared TensorIR endpoint: caller validation/staging, "
                "H2D, identical reduction work, D2H, and detached output"
            ),
            "promotion_scope": (
                "opt-in candidate qualification only; generated CUDA remains the "
                "production default"
            ),
        },
        backend_selected="cuda",
        hardware=outcome(
            "pass",
            None,
            allocation_id=allocation_id,
            target=tuning["baseline_plan"]["target"],
            device=tuning["device"],
            execution_environment=environment,
        ),
        timings=_flatten_samples(candidate),
        memory={
            "allocated_bytes": allocated,
            "peak_bytes": peak,
            "reason": None,
        },
        compilation={
            "seconds": artifact["compile_seconds"],
            "reason": None,
            "source_bytes": artifact["generated_source_bytes"],
            "binary_bytes": artifact["binary_bytes"],
            "resources": artifact["resources"],
        },
        performance=outcome(
            "not-run",
            "matched endpoints cover one reduction shape and target; they do not authorize a general production-default promotion",
            candidate_status=candidate["status"],
            endpoint_gates=candidate["gates"],
            shared_gates=candidate["shared_gates"],
            endpoint_profitability=candidate["endpoint_profitability"],
        ),
    )
    record["stages"].update(
        representation=outcome(
            "pass",
            None,
            equation=program.logical_hash,
            matched_semantic_work={
                "input_elements": rows * inner,
                "output_elements": rows,
                "reduction_terms": rows * inner,
                "reduction_additions": rows * (inner - 1),
            },
        ),
        source=outcome(
            "pass",
            None,
            generated_source_bytes=tuning["baseline_artifact"][
                "generated_source_bytes"
            ],
            cub_source_bytes=artifact["generated_source_bytes"],
        ),
        compilation=outcome(
            "pass",
            None,
            generated_compile_seconds=tuning["baseline_artifact"]["compile_seconds"],
            generated_resources=tuning["baseline_artifact"]["resources"],
            cub_compile_seconds=artifact["compile_seconds"],
            cub_resources=artifact["resources"],
        ),
        numerical=outcome(
            "pass",
            None,
            independent_oracle="TensorIR CPU interpreter",
            fixtures=len(fixtures),
            repeats_per_implementation=len(candidate["samples"][0]) // 2,
            max_absolute_error=candidate["max_absolute_error"],
            atol=ATOL,
            rtol=RTOL,
        ),
        endpoint=outcome(
            "pass",
            None,
            fixtures=len(fixtures),
            samples=len(_flatten_samples(candidate)),
            synchronized=True,
            baseline="generated cooperative reduction",
            candidate="CUB BlockReduce",
        ),
        production=outcome(
            "not-run",
            "qualification does not modify the generated-CUDA production default",
        ),
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--architecture", default="sm_90")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--maximum-seconds", type=float, default=900.0)
    parser.add_argument("--allocation-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-archive-sha256", required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    if not 5 <= args.repeats <= 30:
        parser.error("--repeats must be in 5..30")
    if args.maximum_seconds <= 0:
        parser.error("--maximum-seconds must be positive")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("a finite scheduler allocation must set CUDA_VISIBLE_DEVICES")
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", args.source_revision):
        parser.error("--source-revision must be a full lowercase Git SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", args.source_archive_sha256):
        parser.error("--source-archive-sha256 must be a lowercase SHA-256")
    try:
        source_archive_sha256 = _verified_archive_digest(
            args.source_archive.resolve(), args.source_archive_sha256
        )
    except ValueError as error:
        parser.error(str(error))
    output = args.output.resolve()
    raw_output_path(output / "evidence.json")
    if output.exists():
        parser.error("--output must name a new directory")
    environment = environment_metadata(distributions={"numpy": ("numpy",)})
    output.mkdir(parents=True)

    program, fixtures = qualification_case()
    compiler = CudaCompilerAdapter(
        args.nvcc.resolve(), cuda_target_info(args.architecture)
    )
    baseline = plan_cuda(
        program,
        compiler.target,
        schedule=TensorSchedule(
            stream_reductions=True,
            reduction_provider="generated",
        ),
    )
    selection = tune_cuda(
        baseline,
        compiler,
        fixtures,
        output / "cache",
        schedules=(
            TensorSchedule(
                stream_reductions=True,
                reduction_provider="generated",
            ),
            TensorSchedule(stream_reductions=True, reduction_provider="cub"),
        ),
        screening=None,
        repeats=args.repeats,
        maximum_seconds=args.maximum_seconds,
        minimum_speedup=1.02,
        device=args.device,
    )
    candidate = _cub_candidate(selection.evidence)
    record = _validation_record(
        program=program,
        fixtures=fixtures,
        environment=environment,
        tuning=selection.evidence,
        candidate=candidate,
        allocation_id=args.allocation_id,
        source_revision=args.source_revision,
        source_archive_sha256=source_archive_sha256,
    )
    write_evidence(output / "evidence.json", record)
    detailed = selection.evidence_path.relative_to(output)
    summary = {
        "schema": SCHEMA,
        "revision": args.source_revision,
        "source_archive_sha256": source_archive_sha256,
        "allocation_id": args.allocation_id,
        "equation": program.logical_hash,
        "target": selection.evidence["baseline_plan"]["target"],
        "device": selection.evidence["device"],
        "generated_default_retained": True,
        "candidate_status": candidate["status"],
        "selected_schedule": selection.evidence["selected_schedule"],
        "max_absolute_error": candidate["max_absolute_error"],
        "generated_artifact": selection.evidence["baseline_artifact"],
        "cub_artifact": candidate["artifact"],
        "endpoint_gates": candidate["gates"],
        "shared_gates": candidate["shared_gates"],
        "detailed_tuning_evidence": detailed.as_posix(),
        "decision": (
            "CUB remains opt-in; this single-shape qualification does not change "
            "the generated-CUDA production default"
        ),
    }
    write_result(output / "summary.json", summary)
    print((output / "evidence.json").as_posix())


if __name__ == "__main__":
    main()
