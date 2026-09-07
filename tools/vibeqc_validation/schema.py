"""Versioned evidence with fail-closed promotion and mathematical identities.

Unavailable measurements are explicit records. A successful command or source
emission cannot stand in for numerical, endpoint, or performance evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np

SCHEMA = "vibeqc.validation"
VERSION = 1
TIERS = ("cpu", "cuda-compile", "gpu-numerical", "endpoint")
STAGES = (
    "representation",
    "source",
    "compilation",
    "numerical",
    "endpoint",
    "production",
)
WORKLOADS = (
    "cold-start",
    "unchanged-geometry",
    "changed-geometry",
    "energy-only",
    "energy-plus-force",
)
GATES = {
    "integral_fp64": {"atol": 1e-11, "rtol": 1e-10},
    "cc_correlation_energy": {"atol": 1e-8, "rtol": 0.0},
    "cc_residual": {"atol": 1e-9, "rtol": 0.0},
    "gradient": {"atol": 1e-6, "rtol": 0.0, "target_atol": 1e-7},
}


def canonical_hash(value: object) -> str:
    """Hash finite JSON inputs independent of dictionary insertion order.

    Versions and conventions belong in mathematical inputs; dates, paths,
    machine identity, and generator versions belong in separate provenance.
    Array order is significant. Floats retain their exact JSON round trip.
    """
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    """Identify exact source, binary, or reference bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def outcome(status: str, reason: str | None = None, **measurements) -> dict:
    """Construct an explicit measured, failed, or unavailable stage."""
    result = {"status": status, "reason": reason, **measurements}
    validate_outcome(result)
    return result


def validate_outcome(value: dict) -> None:
    """Reject ambiguous skips and failure records without an explanation."""
    if not isinstance(value, dict) or value.get("status") not in {
        "pass",
        "fail",
        "not-run",
    }:
        raise ValueError("status must be pass, fail, or not-run")
    if "reason" not in value or (
        value["status"] != "pass"
        and (not isinstance(value["reason"], str) or not value["reason"].strip())
    ):
        raise ValueError("failure/not-run requires a reason")


def block_error(actual, reference, *, atol: float, rtol: float) -> dict:
    """Compare every entry using an absolute floor, including near-zero values."""
    if not all(math.isfinite(x) and x >= 0 for x in (atol, rtol)) or atol + rtol == 0:
        raise ValueError("tolerances must be finite, nonnegative, and not both zero")
    actual, reference = (
        np.asarray(actual, dtype=float),
        np.asarray(reference, dtype=float),
    )
    if actual.shape != reference.shape or actual.size == 0:
        raise ValueError("nonempty block shapes must match")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("non-finite numerical block")
    error = np.abs(actual - reference)
    limit = atol + rtol * np.abs(reference)
    scaled = np.divide(error, limit, out=np.full_like(error, np.inf), where=limit > 0)
    scaled[(error == 0) & (limit == 0)] = 0
    return {
        "passed": bool(np.all(error <= limit)),
        "atol": atol,
        "rtol": rtol,
        "max_absolute_error": float(error.max()),
        "max_scaled_error": float(scaled.max()),
        "rms_error": float(np.sqrt(np.mean(error**2))),
        "shape": list(actual.shape),
    }


def finite_difference(
    energy, coordinates, analytic_gradient, *, settings: dict, steps=(1e-2, 3e-3, 1e-3)
) -> dict:
    """Report the whole central-difference curve under one frozen method policy.

    The callback accepts ``(coordinates, settings)``. A fresh settings copy is
    passed each time so an evaluator cannot silently retune subsequent steps.
    Analytic input is a gradient, never a force. No best-step gate is inferred.
    """
    if len(set(steps)) < 3 or any(not math.isfinite(h) or h <= 0 for h in steps):
        raise ValueError("at least three distinct positive finite step sizes required")
    xyz = np.asarray(coordinates, dtype=float)
    policy = json.dumps(settings, allow_nan=False)
    rows = []
    for step in steps:
        derivative = np.empty_like(xyz)
        for index in np.ndindex(xyz.shape):
            plus, minus = xyz.copy(), xyz.copy()
            plus[index] += step
            minus[index] -= step
            derivative[index] = (
                energy(plus, json.loads(policy)) - energy(minus, json.loads(policy))
            ) / (2 * step)
        rows.append(
            {
                "step_bohr": step,
                "gradient": derivative.tolist(),
                "error": block_error(derivative, analytic_gradient, atol=1e-6, rtol=0),
            }
        )
    return {
        "settings": settings,
        "settings_hash": canonical_hash(settings),
        "samples": rows,
    }


def new_evidence(*, tier: str, subject: str, inputs_hash: str) -> dict:
    """Create a non-promotable record for a downstream method or kernel."""
    return {
        "schema": SCHEMA,
        "schema_version": VERSION,
        "tier": tier,
        "subject": subject,
        "inputs_hash": inputs_hash,
        "revision": None,
        "hashes": {key: None for key in ("equation", "ir", "source", "schedule")},
        "hash_reasons": {},
        "device": None,
        "toolchain": {},
        "settings": {},
        "backend_selected": None,
        "hardware": outcome("not-run", "hardware not probed"),
        "stages": {key: outcome("not-run", "no evidence registered") for key in STAGES},
        "timings": [],
        "memory": {
            "allocated_bytes": None,
            "peak_bytes": None,
            "reason": "not measured",
        },
        "compilation": {"seconds": None, "reason": "not measured"},
        "residuals": {},
        "block_errors": {},
        "solver_iterations": [],
        "solver_trace_reason": "not exposed by executor",
        "attachments": [],
        "performance": outcome("not-run", "no comparison registered"),
    }


def _digest(value) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def validate_evidence(record: dict) -> None:
    """Reject incomplete provenance and false GPU/performance promotions.

    Legacy autotune payloads remain attachments, with their original schema.
    This envelope adds cross-method requirements without reinterpreting old
    winner flags as independent validation or changing production selection.
    """
    required = set(new_evidence(tier="cpu", subject="example", inputs_hash=""))
    if not isinstance(record, dict) or not required <= record.keys():
        raise ValueError("incomplete validation record")
    if (
        record["schema"] != SCHEMA
        or type(record["schema_version"]) is not int
        or record["schema_version"] != VERSION
    ):
        raise ValueError("unsupported validation schema/version")
    if record["tier"] not in TIERS or not _digest(record["inputs_hash"]):
        raise ValueError("invalid tier or mathematical inputs hash")
    # Also rejects NaN/Inf hidden in nested errors, settings, or timing samples.
    canonical_hash(record)
    for stage in STAGES:
        validate_outcome(record["stages"][stage])
    validate_outcome(record["hardware"])
    validate_outcome(record["performance"])
    gpu = (
        record["tier"] == "gpu-numerical"
        or record["settings"].get("device") == "cuda"
        or record["backend_selected"] in {"cuda", "hybrid_cuda"}
    )
    measured = any(
        record["stages"][s]["status"] == "pass" for s in ("numerical", "endpoint")
    )
    if (
        gpu
        and measured
        and (
            record["hardware"]["status"] != "pass"
            or not record["device"]
            or record["backend_selected"] != "cuda"
        )
    ):
        raise ValueError(
            "GPU success requires present hardware and actual CUDA backend"
        )
    for sample in record["timings"]:
        if (
            sample["selection"] not in {"baseline", "candidate"}
            or sample["seconds"] <= 0
        ):
            raise ValueError("invalid raw timing sample")
        if sample["workload"] not in WORKLOADS:
            raise ValueError("unknown workload")
        if not _digest(sample["inputs_hash"]):
            raise ValueError("timing sample requires mathematical inputs hash")
    for field in ("allocated_bytes", "peak_bytes"):
        value = record["memory"][field]
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("memory bytes must be nonnegative integers")
    allocated, peak = (
        record["memory"]["allocated_bytes"],
        record["memory"]["peak_bytes"],
    )
    if allocated is not None and peak is not None and allocated > peak:
        raise ValueError("peak memory cannot be smaller than allocated memory")
    compile_seconds = record["compilation"]["seconds"]
    if (allocated is None or peak is None) and not record["memory"].get("reason"):
        raise ValueError("unavailable memory measurement requires a reason")
    if compile_seconds is None and not record["compilation"].get("reason"):
        raise ValueError("unavailable compilation measurement requires a reason")
    if compile_seconds is not None and (
        isinstance(compile_seconds, bool) or compile_seconds < 0
    ):
        raise ValueError("compilation cost must be nonnegative")
    if (
        record["performance"]["status"] == "pass"
        or record["stages"]["production"]["status"] == "pass"
    ):
        if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", record["revision"] or ""):
            raise ValueError("performance pass requires revision provenance")
        if not all(
            _digest(record["hashes"].get(k))
            for k in ("equation", "ir", "source", "schedule")
        ):
            raise ValueError(
                "performance pass requires equation/IR/source/schedule provenance"
            )
        if (
            not record["toolchain"]
            or not record["device"]
            or not record["backend_selected"]
        ):
            raise ValueError(
                "performance pass requires device/toolchain/backend provenance"
            )
        if gpu and record["settings"].get("fast_compile") is not False:
            raise ValueError(
                "GPU promotion requires explicit fast_compile=false build provenance"
            )
        if any(
            record["stages"][s]["status"] != "pass"
            for s in ("compilation", "numerical", "endpoint")
        ):
            raise ValueError(
                "performance pass requires compilation/numerical/endpoint evidence"
            )
        if (
            any(record["memory"][k] is None for k in ("allocated_bytes", "peak_bytes"))
            or record["compilation"]["seconds"] is None
        ):
            raise ValueError("promotion requires memory and compilation cost")
        limits = record["settings"].get("promotion_limits", {})
        if not all(
            k in limits
            and isinstance(limits[k], (int, float))
            and not isinstance(limits[k], bool)
            and limits[k] >= 0
            for k in ("peak_bytes", "compile_seconds")
        ):
            raise ValueError(
                "promotion requires explicit memory and compilation budgets"
            )
        if peak > limits["peak_bytes"] or compile_seconds > limits["compile_seconds"]:
            raise ValueError("promotion exceeds memory or compilation budget")
        if not record["block_errors"] or not all(
            e["passed"] is True and 0 <= e["max_scaled_error"] <= 1
            for e in record["block_errors"].values()
        ):
            raise ValueError("promotion requires passing per-block errors")
        from .performance import assess_comparison

        assessment = assess_comparison(record["timings"])
        if assessment["status"] != "pass":
            raise ValueError(
                "performance pass lacks a significant interleaved comparison"
            )
        kind = record["settings"].get("comparison_kind")
        if kind not in {"kernel", "solver"}:
            raise ValueError("promotion requires kernel or solver comparison semantics")
        if kind == "kernel" and not _digest(record["settings"].get("fixed_state_hash")):
            raise ValueError(
                "kernel promotion requires fixed density/amplitude state hash"
            )
        if kind == "solver":
            if not record["solver_iterations"] or not record["residuals"]:
                raise ValueError(
                    "solver promotion requires every iteration and final residual"
                )
            for index, sample in enumerate(record["timings"]):
                rows = [
                    r
                    for r in record["solver_iterations"]
                    if r.get("sample_index") == index
                ]
                count = sample["diagnostics"].get("iterations")
                if (
                    type(count) is not int
                    or count < 1
                    or [r.get("iteration") for r in rows] != list(range(1, count + 1))
                ):
                    raise ValueError(
                        "solver history must include every iteration for every sample"
                    )
                if any(
                    not all(
                        isinstance(r.get(k), (int, float))
                        for k in ("energy", "residual")
                    )
                    for r in rows
                ):
                    raise ValueError("solver iterations require energy and residual")
                final = record["residuals"].get(str(index), {})
                if final.get("independently_evaluated") is not True or not isinstance(
                    final.get("value"), (int, float)
                ):
                    raise ValueError(
                        "solver promotion requires independently evaluated final residuals"
                    )
                limit = record["settings"].get("residual_limit")
                if (
                    not isinstance(limit, (int, float))
                    or limit < 0
                    or not 0 <= final["value"] <= limit
                ):
                    raise ValueError(
                        "solver final residual exceeds the registered numerical gate"
                    )


def write_evidence(path: str | Path, record: dict) -> None:
    """Validate before writing so invalid pass records never become artifacts."""
    validate_evidence(record)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def attach_artifact(record: dict, path: str | Path, *, kind: str) -> None:
    """Register existing autotune/#135/endpoint evidence without copying runners."""
    payload = json.loads(Path(path).read_text())
    if "schema_version" not in payload:
        raise ValueError("attached evidence must identify its schema version")
    record["attachments"].append(
        {
            "kind": kind,
            "path": str(path),
            "sha256": file_hash(path),
            "schema_version": payload["schema_version"],
        }
    )
