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
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path, PurePosixPath
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


def _sha256_stream(stream: Any) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _verified_source_snapshot(
    path: Path, expected_digest: str, expected_revision: str, root: Path
) -> dict[str, Any]:
    """Bind the current execution tree to one exact Git archive revision."""

    if not path.is_file():
        raise ValueError("source archive is unavailable")
    actual = file_hash(path)
    if actual != expected_digest:
        raise ValueError("source archive SHA-256 mismatch")
    archived_paths: set[str] = set()
    file_count = 0
    total_bytes = 0
    with tarfile.open(path, "r:*") as archive:
        revision = archive.pax_headers.get("comment")
        if revision != expected_revision:
            raise ValueError("source archive Git revision mismatch")
        for member in archive.getmembers():
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or not relative.parts or ".." in relative.parts:
                raise ValueError("source archive contains an unsafe path")
            normalized = relative.as_posix()
            if normalized in archived_paths:
                raise ValueError("source archive contains a duplicate path")
            archived_paths.add(normalized)
            local = root.joinpath(*relative.parts)
            if member.isdir():
                if not local.is_dir():
                    raise ValueError("execution tree differs from source archive")
            elif member.isfile():
                if not local.is_file() or local.is_symlink():
                    raise ValueError("execution tree differs from source archive")
                archived = archive.extractfile(member)
                if archived is None:
                    raise ValueError("source archive member is unreadable")
                with archived, local.open("rb") as current:
                    if _sha256_stream(archived) != _sha256_stream(current):
                        raise ValueError("execution tree differs from source archive")
                file_count += 1
                total_bytes += member.size
            elif member.issym():
                if not local.is_symlink() or os.readlink(local) != member.linkname:
                    raise ValueError("execution tree differs from source archive")
            else:
                raise ValueError("source archive contains an unsupported member type")
    extras = []
    for candidate in root.rglob("*"):
        if candidate.is_dir():
            continue
        relative = candidate.relative_to(root)
        if (
            relative.parts[0] in {".artifacts", ".git", ".pytest_cache", ".ruff_cache"}
            or "__pycache__" in relative.parts
            or candidate.suffix == ".pyc"
        ):
            continue
        if relative.as_posix() not in archived_paths:
            extras.append(relative.as_posix())
    if extras:
        raise ValueError("execution tree contains files outside the source archive")
    return {
        "archive_sha256": actual,
        "git_revision": revision,
        "verified_files": file_count,
        "verified_bytes": total_bytes,
        "execution_root": str(root.resolve()),
    }


def _nvidia_smi_metadata(device: int, probed: dict[str, Any]) -> dict[str, Any]:
    """Record and cross-check the scheduler-visible NVIDIA device."""

    if type(device) is not int or device < 0:
        raise ValueError("CUDA device ordinal must be a nonnegative integer")
    executed_uuid = str(probed.get("uuid", "")).lower()
    if not re.fullmatch(r"[0-9a-f]{32}", executed_uuid):
        raise ValueError("executed CUDA device UUID is invalid")
    # CUDA_VISIBLE_DEVICES can remap logical ordinals; select the actual UUID.
    smi_id = (
        f"GPU-{executed_uuid[:8]}-{executed_uuid[8:12]}-"
        f"{executed_uuid[12:16]}-{executed_uuid[16:20]}-{executed_uuid[20:]}"
    )
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise ValueError("nvidia-smi is unavailable")
    result = subprocess.run(
        (
            executable,
            f"--id={smi_id}",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError("nvidia-smi did not identify exactly one device")
    fields = [value.strip() for value in rows[0].split(",")]
    if len(fields) != 4:
        raise ValueError("nvidia-smi metadata is incomplete")
    name, uuid, driver, memory_mib = fields
    normalized = uuid.removeprefix("GPU-").replace("-", "").lower()
    if normalized != executed_uuid:
        raise ValueError("nvidia-smi UUID differs from the executed CUDA device")
    try:
        memory = int(memory_mib)
    except ValueError as error:
        raise ValueError("nvidia-smi memory size is invalid") from error
    return {
        "name": name,
        "uuid": uuid,
        "driver_version": driver,
        "memory_total_mib": memory,
    }


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
    source_snapshot: dict[str, Any],
    nvidia_smi: dict[str, Any],
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
        revision=source_snapshot["git_revision"],
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
            "source_snapshot": source_snapshot,
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
            nvidia_smi=nvidia_smi,
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
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", args.source_revision):
        parser.error("--source-revision must be a full lowercase Git SHA")
    if not re.fullmatch(r"[0-9a-f]{64}", args.source_archive_sha256):
        parser.error("--source-archive-sha256 must be a lowercase SHA-256")
    try:
        source_snapshot = _verified_source_snapshot(
            args.source_archive.resolve(),
            args.source_archive_sha256,
            args.source_revision,
            ROOT,
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
    nvidia_smi = _nvidia_smi_metadata(args.device, selection.evidence["device"])
    record = _validation_record(
        program=program,
        fixtures=fixtures,
        environment=environment,
        tuning=selection.evidence,
        candidate=candidate,
        allocation_id=args.allocation_id,
        source_snapshot=source_snapshot,
        nvidia_smi=nvidia_smi,
    )
    write_evidence(output / "evidence.json", record)
    write_result(output / "tuning-evidence.json", selection.evidence)
    summary = {
        "schema": SCHEMA,
        "revision": source_snapshot["git_revision"],
        "source_snapshot": source_snapshot,
        "allocation_id": args.allocation_id,
        "equation": program.logical_hash,
        "target": selection.evidence["baseline_plan"]["target"],
        "device": selection.evidence["device"],
        "nvidia_smi": nvidia_smi,
        "generated_default_retained": True,
        "candidate_status": candidate["status"],
        "selected_schedule": selection.evidence["selected_schedule"],
        "max_absolute_error": candidate["max_absolute_error"],
        "generated_artifact": selection.evidence["baseline_artifact"],
        "cub_artifact": candidate["artifact"],
        "endpoint_gates": candidate["gates"],
        "shared_gates": candidate["shared_gates"],
        "detailed_tuning_evidence": "tuning-evidence.json",
        "decision": (
            "CUB remains opt-in; this single-shape qualification does not change "
            "the generated-CUDA production default"
        ),
    }
    write_result(output / "summary.json", summary)
    print((output / "evidence.json").as_posix())


if __name__ == "__main__":
    main()
