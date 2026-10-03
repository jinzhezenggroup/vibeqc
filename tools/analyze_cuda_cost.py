"""Report GPU-free CUDA cost evidence from static facts or PTXAS diagnostics."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

from generativeqc_compiler.common.cuda_cost_model import static_cuda_cost
from generativeqc_compiler.common.cuda_resources import (
    KernelResources,
    compiled_gpu_profitability,
    parse_resources,
)
from generativeqc_compiler.common.cuda_target import cuda_target_info
from generativeqc_compiler.common.cuda_time_estimator import (
    CudaTimingCalibration,
    estimate_cuda_time,
)
from generativeqc_compiler.common.gpu_profitability import GpuProfitability


def _ptxas_resources(
    diagnostics: str, architecture: str
) -> tuple[tuple[KernelResources, ...], bool]:
    """Reject incomplete rows and contradictory retained-log provenance."""

    resources = parse_resources(diagnostics)
    declared = re.findall(r"Function properties for (\S+)", diagnostics)
    if not resources:
        raise ValueError("CUDA cost analysis requires PTXAS resource rows")
    if sorted(row.function for row in resources) != sorted(declared):
        raise ValueError("PTXAS log contains incomplete or unsupported resource rows")

    entries = re.findall(
        r"Compiling entry function '([^']+)' for '([^']+)'", diagnostics
    )
    entry_counts = Counter(function for function, _ in entries)
    resource_counts = Counter(row.function for row in resources)
    if entry_counts - resource_counts:
        raise ValueError("PTXAS log contains incomplete or unsupported resource rows")
    architectures = {entry_architecture for _, entry_architecture in entries}
    if architectures - {architecture}:
        raise ValueError(
            f"PTXAS architecture declarations {sorted(architectures)} "
            f"do not match requested target {architecture}"
        )
    # Headerless resource snippets remain useful, but cannot certify which
    # architecture produced them. Every reported function must be accounted for.
    architecture_verified = bool(entries) and entry_counts == resource_counts
    return resources, architecture_verified


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a CUDA resource/parallelism screening report without probing "
            "or executing a GPU. Optional explicit calibration adds an experimental "
            "homogeneous-kernel timing estimate, never an endpoint prediction."
        )
    )
    parser.add_argument("--arch", required=True, help="CUDA architecture, e.g. sm_120")
    parser.add_argument("--block-threads", required=True, type=int)
    parser.add_argument(
        "--ptxas",
        type=Path,
        help="optional PTXAS -v diagnostics; compilation may happen elsewhere",
    )
    parser.add_argument("--grid-blocks", type=int)
    parser.add_argument(
        "--sm-count",
        type=int,
        help="optional target GPU SM count for global-grid saturation",
    )
    parser.add_argument("--traffic-bytes", type=int)
    parser.add_argument("--operations", type=int)
    parser.add_argument("--launches", type=int)
    parser.add_argument("--source-bytes", type=int)
    parser.add_argument(
        "--estimated-registers",
        type=int,
        help="pre-compilation registers/thread estimate when --ptxas is absent",
    )
    parser.add_argument(
        "--estimated-occupancy",
        type=float,
        help="pre-compilation occupancy upper bound in [0, 1]",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        help="cuda-timing-calibration.v1 or .v2 JSON with measured device/workload rates",
    )
    parser.add_argument(
        "--spill-traffic-bytes",
        type=int,
        help="total dynamic spill traffic across all launches, excluding semantic traffic",
    )
    parser.add_argument(
        "--allow-per-sm-fallback",
        action="store_true",
        help="explicitly allow optimistic timing without global underfill evidence",
    )
    return parser


def main() -> None:
    """Emit static evidence and, only on request, calibrated kernel timing."""
    parser = _parser()
    args = parser.parse_args()
    try:
        payload = _report(args)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def _report(args: argparse.Namespace) -> dict[str, object]:
    calibration = None
    if args.calibration is not None:
        calibration_payload = json.loads(args.calibration.read_text(encoding="utf-8"))
        schema = (
            calibration_payload.pop("schema", None)
            if isinstance(calibration_payload, dict)
            else None
        )
        if schema not in {
            "generativeqc.compiler.cuda-timing-calibration.v1",
            "generativeqc.compiler.cuda-timing-calibration.v2",
        }:
            raise ValueError(
                "calibration must be a cuda-timing-calibration.v1 or .v2 JSON object"
            )
        calibration = CudaTimingCalibration(**calibration_payload)
        if calibration.to_payload()["schema"] != schema:
            raise ValueError("calibration schema and model version disagree")
    elif args.spill_traffic_bytes is not None or args.allow_per_sm_fallback:
        raise ValueError("timing options require --calibration")
    target = cuda_target_info(args.arch)
    ptxas_evidence = None
    if args.ptxas is not None:
        resources, architecture_verified = _ptxas_resources(
            args.ptxas.read_text(encoding="utf-8"), target.architecture
        )
        # A max-resource summary is useful for screening, but timing mixed
        # kernels with one roofline/occupancy value loses sequential work.
        if calibration is not None and len(resources) != 1:
            raise ValueError("timing requires exactly one PTXAS kernel resource row")
        ptxas_evidence = {
            "architecture": target.architecture if architecture_verified else None,
            "architecture_verified": architecture_verified,
            "functions": [row.function for row in resources],
        }
        profitability = compiled_gpu_profitability(
            resources,
            target,
            args.block_threads,
        )
        profitability = replace(
            profitability,
            semantic_traffic_bytes=args.traffic_bytes,
            arithmetic_operation_count=args.operations,
            launch_count=args.launches,
            source_bytes=args.source_bytes,
        )
    else:
        profitability = GpuProfitability(
            semantic_traffic_bytes=args.traffic_bytes,
            arithmetic_operation_count=args.operations,
            estimated_registers_per_thread=args.estimated_registers,
            estimated_occupancy_upper_bound=args.estimated_occupancy,
            launch_count=args.launches,
            source_bytes=args.source_bytes,
        )

    report = static_cuda_cost(
        profitability,
        target,
        args.block_threads,
        grid_blocks=args.grid_blocks,
        sm_count=(
            calibration.sm_count
            if args.sm_count is None and calibration is not None
            else args.sm_count
        ),
    )
    if ptxas_evidence is not None and not ptxas_evidence["architecture_verified"]:
        report = replace(
            report,
            diagnostics=(
                *report.diagnostics,
                (
                    "PTXAS architecture is unverified for one or more resource rows; "
                    "the requested target is a caller assumption"
                ),
            ),
        )
    payload = report.to_payload()
    payload["ptxas_evidence"] = ptxas_evidence
    payload["screening_priority"] = report.screening_priority(0)
    payload["profitability"] = profitability.to_payload()
    if calibration is not None:
        payload["time_estimate"] = estimate_cuda_time(
            report,
            calibration,
            spill_traffic_bytes=args.spill_traffic_bytes,
            allow_per_sm_fallback=args.allow_per_sm_fallback,
        ).to_payload()
    return payload


if __name__ == "__main__":
    main()
