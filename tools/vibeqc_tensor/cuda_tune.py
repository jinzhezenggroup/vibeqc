"""Bounded, opt-in selection using complete FP64 TensorIR endpoints.

No candidate becomes the default merely because it compiles or saves FLOPs.
Selection requires CPU/baseline parity and paired timing evidence on every
provided fixture. Rejected candidates and all raw samples remain in evidence.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from vibeqc.profiles import atomic_json, canonical_hash

from tools.vibeqc_codegen.cuda_adapter import CudaCompilerAdapter
from tools.vibeqc_validation.performance import assess_comparison, measure_interleaved

from .cuda_execute import CudaArtifact, PreparedCuda, compile_cuda
from .cuda_plan import TensorPlan, TensorSchedule, plan_cuda
from .interpreter import execute


def endpoint_gate(baseline, candidate, *, minimum_speedup=1.02) -> dict:
    """Require a paired median gain whose bootstrap lower bound exceeds one."""
    left, right = np.asarray(baseline), np.asarray(candidate)
    if left.ndim != 1 or left.shape != right.shape or not 5 <= left.size <= 30:
        raise ValueError("tensor endpoint gate requires 5..30 paired samples")
    if (
        not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or np.min(left) <= 0
        or np.min(right) <= 0
    ):
        raise ValueError("endpoint times must be positive and finite")
    if not np.isfinite(minimum_speedup) or minimum_speedup < 1:
        raise ValueError("minimum speedup must be finite and at least one")
    speedup = float(np.median(left) / np.median(right))
    indices = np.random.default_rng(146).integers(left.size, size=(4096, left.size))
    lower = float(
        np.quantile(
            np.median(left[indices], axis=1) / np.median(right[indices], axis=1), 0.05
        )
    )
    return {
        "passed": speedup >= minimum_speedup and lower > 1,
        "median_speedup": speedup,
        "bootstrap_lower_95": lower,
        "minimum_speedup": minimum_speedup,
    }


def candidate_schedules() -> tuple[TensorSchedule, ...]:
    """A small readable candidate set; no exhaustive installation-time search."""
    return (
        TensorSchedule(views=True),
        TensorSchedule(views=True, fuse=True),
        TensorSchedule(views=True, fuse=True, tile_m=64, tile_n=64, tile_k=64),
        TensorSchedule(views=True, fuse=True, recompute=True),
    )


@dataclass(frozen=True)
class TensorSelection:
    """Concrete compiled winner plus the complete selection audit artifact."""

    plan: TensorPlan
    artifact: CudaArtifact
    evidence: dict
    evidence_path: Path


def tune_cuda(
    baseline: TensorPlan,
    compiler: CudaCompilerAdapter,
    fixtures,
    cache: Path,
    *,
    schedules=None,
    repeats: int = 8,
    maximum_seconds: float = 600,
    minimum_speedup: float = 1.02,
    device: int = 0,
) -> TensorSelection:
    """Validate/time at most eight candidates, retaining baseline on regression.

    Each fixture uses this plan's shape bucket but may differ in values, scale
    and caller strides. Every complete endpoint includes transfers, packing,
    validation and output copies. A finite Slurm allocation remains the hard
    timeout for device work; this deadline stops launching further candidates.
    The CPU interpreter is an oracle during tuning, never a runtime fallback.
    """
    schedules = tuple(candidate_schedules() if schedules is None else schedules)
    fixtures = tuple(fixtures)
    if not 1 <= len(schedules) <= 8 or not 1 <= len(fixtures) <= 8:
        raise ValueError("tuning requires 1..8 candidates and 1..8 fixtures")
    if type(repeats) is not int or not 5 <= repeats <= 30:
        raise ValueError("tuning repeats must be in 5..30")
    if not np.isfinite(maximum_seconds) or maximum_seconds <= 0:
        raise ValueError("tuning duration must be positive and finite")
    if not np.isfinite(minimum_speedup) or minimum_speedup < 1:
        raise ValueError("minimum speedup must be finite and at least one")
    if baseline.schedule.views or baseline.schedule.fuse or baseline.schedule.recompute:
        raise ValueError("tuning requires an unfused CUDA baseline")
    started = time.monotonic()
    artifact = compile_cuda(baseline, compiler, cache)
    references = [execute(baseline.program, values).outputs for values in fixtures]
    feed_identities = []
    import hashlib

    for values in fixtures:
        feed_identities.append(
            {
                name: {
                    "shape": value.shape,
                    "strides": value.strides,
                    "dtype": value.dtype.str,
                    "values_sha256": hashlib.sha256(
                        value.tobytes(order="C")
                    ).hexdigest(),
                }
                for name, value in sorted(values.items())
            }
        )
    best_plan, best_artifact, best_score = baseline, artifact, 1.0
    candidates = []
    with PreparedCuda(baseline, artifact, device=device) as reference_cuda:
        identity = {
            "schema": 1,
            "baseline": reference_cuda.identity,
            "fixtures": feed_identities,
            "schedules": [asdict(s) for s in schedules],
            "repeats": repeats,
            "minimum_speedup": minimum_speedup,
            "numerical_gate": {"atol": 1e-11, "rtol": 1e-10},
        }
        key = canonical_hash(identity)
        # Library loading and first-kernel JIT are reported separately and are
        # never silently mixed into warmed selection samples.
        startup = []
        for feeds, expected in zip(fixtures, references, strict=True):
            result = reference_cuda.execute(feeds)
            _parity(result.outputs, expected)
            startup.append(result.metrics)
            reference_cuda.execute(feeds)
        for schedule in schedules:
            row = {"requested_schedule": asdict(schedule)}
            candidates.append(row)
            if time.monotonic() - started >= maximum_seconds:
                row.update(status="skipped", reason="tuning deadline exhausted")
                continue
            try:
                plan = plan_cuda(
                    baseline.program,
                    baseline.target,
                    max_bytes=baseline.max_bytes,
                    schedule=schedule,
                    reservations=baseline.reservations,
                    library_bytes=baseline.library_bytes,
                    provider_bytes=baseline.provider_bytes,
                )
                row.update(plan=plan.to_payload(), plan_identity=plan.identity)
                compiled = compile_cuda(plan, compiler, cache)
                row["artifact"] = compiled.metadata
                resources = compiled.metadata.get("resources", [])
                if not resources or any(
                    r["registers"] > plan.target.tuning_maximum_registers
                    or r["stack_bytes"] > plan.target.tuning_maximum_stack_bytes
                    or r["spill_store_bytes"]
                    or r["spill_load_bytes"]
                    or r["shared_bytes"] > plan.target.tuning_maximum_shared_bytes
                    for r in resources
                ):
                    raise ValueError("candidate fails the compiled resource gate")
                with PreparedCuda(plan, compiled, device=device) as candidate:
                    profiles, timings, errors = [], [], []
                    for fixture_index, (feeds, expected) in enumerate(
                        zip(fixtures, references, strict=True)
                    ):
                        result = candidate.execute(feeds)
                        errors.append(_parity(result.outputs, expected))
                        # Compare against the unfused GPU too, including all
                        # outputs, without assuming CPU/GPU associativity.
                        errors.append(
                            _parity(
                                result.outputs, reference_cuda.execute(feeds).outputs
                            )
                        )
                        candidate.execute(feeds)
                        latest = [None]

                        def before_sample(selection, latest=latest, expected=expected):
                            if time.monotonic() - started >= maximum_seconds:
                                raise TimeoutError("tuning deadline exhausted")
                            if latest[0] is not None:
                                _parity(latest[0].outputs, expected)

                        def evaluate(selection, latest=latest, feeds=feeds):
                            selected = (
                                reference_cuda if selection == "baseline" else candidate
                            )
                            latest[0] = selected.execute(feeds)
                            return latest[0].metrics

                        # execute() synchronizes its final event, so the shared
                        # runner needs no extra global-device barrier. Numerical
                        # comparisons stay outside its endpoint timing window.
                        pairs = measure_interleaved(
                            evaluate,
                            lambda: None,
                            prepare=before_sample,
                            repeats=repeats,
                            workload="unchanged-geometry",
                            inputs_hash=canonical_hash(
                                {
                                    "equation": baseline.program.logical_hash,
                                    "feeds": feed_identities[fixture_index],
                                }
                            ),
                        )
                        _parity(latest[0].outputs, expected)
                        timings.append(pairs)
                        profiles.append(
                            {
                                "baseline": reference_cuda.execute(
                                    feeds, profile=True
                                ).metrics,
                                "candidate": candidate.execute(
                                    feeds, profile=True
                                ).metrics,
                            }
                        )
                    gates = [
                        endpoint_gate(
                            [
                                p["seconds"]
                                for p in pairs
                                if p["selection"] == "baseline"
                            ],
                            [
                                p["seconds"]
                                for p in pairs
                                if p["selection"] == "candidate"
                            ],
                            minimum_speedup=minimum_speedup,
                        )
                        for pairs in timings
                    ]
                    shared_gates = [assess_comparison(pairs) for pairs in timings]
                    passed = all(g["passed"] for g in gates) and all(
                        g["status"] == "pass" for g in shared_gates
                    )
                    row.update(
                        status="accepted" if passed else "rejected",
                        gates=gates,
                        shared_gates=shared_gates,
                        samples=timings,
                        profiles=profiles,
                        max_absolute_error=max(errors),
                    )
                    score = min(g["median_speedup"] for g in gates)
                    if passed and score > best_score:
                        best_plan, best_artifact, best_score = plan, compiled, score
            except (ValueError, RuntimeError, TimeoutError) as error:
                row.update(status="rejected", reason=str(error))
        evidence = {
            "schema": "vibeqc.tensor.cuda.tuning",
            "schema_version": 1,
            "identity": identity,
            "key": key,
            "device": reference_cuda.device,
            "baseline_artifact": artifact.metadata,
            "baseline_plan": baseline.to_payload(),
            "baseline_startup": startup,
            "candidates": candidates,
            "selected_plan": best_plan.identity,
            "selected_artifact": best_artifact.metadata["key"],
            "selected_schedule": asdict(best_plan.schedule),
            "seconds": time.monotonic() - started,
            "graph_status": reference_cuda.graph_status,
        }
    path = Path(cache) / "selections" / key / "evidence.json"
    atomic_json(path, evidence)
    return TensorSelection(best_plan, best_artifact, evidence, path)


def _parity(actual, expected):
    error = 0.0
    for name, reference in expected.items():
        result = actual[name]
        if (
            result.shape != reference.shape
            or not np.isfinite(result).all()
            or not np.allclose(result, reference, atol=1e-11, rtol=1e-10)
        ):
            raise ValueError(f"CUDA tensor numerical gate failed for {name}")
        error = max(error, float(np.max(np.abs(result - reference), initial=0)))
    return error
