"""Bounded, opt-in selection using complete FP64 TensorIR endpoints.

No candidate becomes the default merely because it compiles or saves FLOPs.
Selection requires CPU/baseline parity and paired timing evidence on every
provided fixture. Rejected candidates and all raw samples remain in evidence.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

import numpy as np

from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.performance import assess_comparison, measure_interleaved
from vibeqc_compiler.common.provenance import atomic_json, canonical_hash
from vibeqc_compiler.common.specialization import (
    CompilationIdentity,
    GuardPredicate,
    ImplementationProfile,
    SpecializationGuard,
    TargetCapabilities,
    WorkloadSignature,
)

from .cuda_execute import CudaArtifact, PreparedCuda, compile_cuda
from .cuda_plan import TensorPlan, TensorSchedule
from .cuda_search import (
    DEFAULT_SEARCH_LIMITS,
    TensorScheduleSpace,
    TensorSearchLimits,
    plan_schedule_search,
    require_compiled_resources,
)
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


def candidate_schedules(
    space: TensorScheduleSpace | None = None, *, maximum: int = 128
) -> tuple[TensorSchedule, ...]:
    """Structured, reproducible prefix; still opt-in, never installation tuning."""
    return (TensorScheduleSpace() if space is None else space).generate(maximum)


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
    search_space: TensorScheduleSpace | None = None,
    search_limits: TensorSearchLimits = DEFAULT_SEARCH_LIMITS,
    repeats: int = 8,
    maximum_seconds: float = 600,
    minimum_speedup: float = 1.02,
    device: int = 0,
) -> TensorSelection:
    """Search bounded schedules, retaining baseline unless endpoints qualify.

    Each fixture uses this plan's shape bucket but may differ in values, scale
    and caller strides. Every complete endpoint includes transfers, packing,
    validation and output copies. A finite Slurm allocation remains the hard
    timeout for device work; this deadline stops launching further candidates.
    The CPU interpreter is an oracle during tuning, never a runtime fallback.
    """
    if baseline.precision != "fp64":
        raise ValueError(
            "automatic TensorIR schedule promotion is currently qualified only for FP64"
        )
    if not isinstance(search_limits, TensorSearchLimits):
        raise TypeError("search_limits must be TensorSearchLimits")
    if schedules is not None and search_space is not None:
        raise ValueError("provide schedules or search_space, not both")
    schedules = (
        candidate_schedules(search_space, maximum=search_limits.maximum_candidates)
        if schedules is None
        else tuple(islice(schedules, search_limits.maximum_candidates + 1))
    )
    fixtures = tuple(islice(fixtures, 9))
    if not 1 <= len(fixtures) <= 8:
        raise ValueError("tuning requires 1..8 fixtures")
    if type(repeats) is not int or not 5 <= repeats <= 30:
        raise ValueError("tuning repeats must be in 5..30")
    if not np.isfinite(maximum_seconds) or maximum_seconds <= 0:
        raise ValueError("tuning duration must be positive and finite")
    if not np.isfinite(minimum_speedup) or minimum_speedup < 1:
        raise ValueError("minimum speedup must be finite and at least one")
    if baseline.schedule.views or baseline.schedule.fuse or baseline.schedule.recompute:
        raise ValueError("tuning requires an unfused CUDA baseline")
    started = time.monotonic()
    search = plan_schedule_search(baseline, schedules, search_limits)
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
    compilation_attempts = 0
    selected_profiles = []
    with PreparedCuda(baseline, artifact, device=device) as reference_cuda:
        identity = {
            "schema": 2,
            "baseline": reference_cuda.identity,
            "fixtures": feed_identities,
            "schedules": [asdict(s) for s in schedules],
            "search_limits": asdict(search_limits),
            "maximum_seconds": maximum_seconds,
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
        for proposal in search:
            row = proposal.to_payload()
            candidates.append(row)
            if proposal.status != "ready":
                continue
            if time.monotonic() - started >= maximum_seconds:
                row.update(
                    status="skipped",
                    stage="deadline",
                    reason="tuning deadline exhausted",
                )
                continue
            if compilation_attempts >= search_limits.maximum_compilations:
                row.update(
                    status="skipped",
                    stage="compile-budget",
                    reason="candidate compilation budget exhausted",
                )
                continue
            try:
                plan = proposal.plan
                row["stage"] = "compile"
                compilation_attempts += 1
                compile_started = time.monotonic()
                try:
                    compiled = compile_cuda(plan, compiler, cache)
                finally:
                    row["compile_wall_seconds"] = time.monotonic() - compile_started
                row["artifact"] = compiled.metadata
                row["stage"] = "compiled-resource"
                require_compiled_resources(
                    plan,
                    compiled.metadata.get("resources", []),
                    minimum_resident_blocks=search_limits.minimum_resident_blocks,
                )
                if time.monotonic() - started >= maximum_seconds:
                    raise TimeoutError("tuning deadline exhausted")
                row["stage"] = "endpoint"
                with PreparedCuda(plan, compiled, device=device) as candidate:
                    profiles, timings, errors = [], [], []
                    # Keep completed fixture evidence even if a later fixture
                    # fails a numerical/resource/deadline gate.
                    row.update(samples=timings, profiles=profiles)
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
                    if passed:
                        row["promotion_profiles"] = _promotion_profiles(
                            plan,
                            compiled,
                            feed_identities,
                            canonical_hash(row),
                            baseline_execution=reference_cuda.identity,
                        )
                    if passed and score > best_score:
                        best_plan, best_artifact, best_score = plan, compiled, score
                        selected_profiles = row["promotion_profiles"]
            except (ValueError, RuntimeError, TimeoutError) as error:
                row.update(status="rejected", reason=str(error))
        evidence = {
            "schema": "vibeqc.tensor.cuda.tuning",
            "schema_version": 2,
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
            "selected_profiles": selected_profiles,
            "search_summary": {
                "generated": len(search),
                "pruned_before_compile": sum(p.status == "pruned" for p in search),
                "compilation_attempts": compilation_attempts,
                "endpoint_candidates": sum(
                    r.get("stage") == "endpoint" for r in candidates
                ),
                "accepted": sum(r["status"] == "accepted" for r in candidates),
            },
            "seconds": time.monotonic() - started,
            "graph_status": reference_cuda.graph_status,
        }
    path = Path(cache) / "selections" / key / "evidence.json"
    atomic_json(path, evidence)
    return TensorSelection(best_plan, best_artifact, evidence, path)


def _promotion_profiles(plan, artifact, feeds, evidence_hash, *, baseline_execution):
    """Declare only the measured layout domains using #459's shared records.

    No new profile database or runtime lookup is introduced. These records refer
    to #136's existing executable key and the candidate's complete evidence hash;
    they must not be treated as a general promotion to unmeasured inputs/targets.
    """
    identity = CompilationIdentity(
        plan.program.logical_hash,
        canonical_hash(
            {
                k: v
                for k, v in artifact.metadata["identity"].items()
                if k not in ("plan", "generated")
            }
        ),
    )
    target = TargetCapabilities(
        plan.target.target_info,
        (("tensor_target", canonical_hash(plan.target.to_payload())),),
    )
    profiles = {}
    for feed in feeds:
        layout = {
            name: {k: info[k] for k in ("shape", "strides", "dtype")}
            for name, info in feed.items()
        }
        workload = WorkloadSignature(
            "tensor-cuda-endpoint",
            (
                ("equation", plan.program.logical_hash),
                ("max_bytes", plan.max_bytes),
                ("reservations", canonical_hash(asdict(plan.reservations))),
                ("input_layout", canonical_hash(layout)),
                ("baseline_execution", baseline_execution),
            ),
        )
        correctness = SpecializationGuard(
            tuple(
                GuardPredicate("workload", name, "eq", value)
                for name, value in (("kind", workload.kind), *workload.features)
                if name not in ("input_layout", "baseline_execution")
            )
            + (
                GuardPredicate("target", "backend", "eq", "cuda"),
                GuardPredicate(
                    "target", "architecture", "eq", plan.target.architecture
                ),
                GuardPredicate(
                    "target",
                    "tensor_target",
                    "eq",
                    dict(target.features)["tensor_target"],
                ),
            )
        )
        performance = SpecializationGuard(
            correctness.predicates
            + (
                GuardPredicate(
                    "workload", "input_layout", "eq", canonical_hash(layout)
                ),
                GuardPredicate(
                    "workload", "baseline_execution", "eq", baseline_execution
                ),
            )
        )
        domain = canonical_hash(asdict(workload))
        profile = ImplementationProfile(
            name=f"tensor-{plan.identity[:12]}-{domain[:12]}",
            identity=identity,
            artifact_key=artifact.metadata["key"],
            schedule_hash=canonical_hash(asdict(plan.schedule)),
            profile_hash=evidence_hash,
            correctness=correctness,
            performance=performance,
        )
        profiles[domain] = {
            "workload": asdict(workload),
            "target": asdict(target),
            "profile": asdict(profile),
        }
    return [profiles[key] for key in sorted(profiles)]


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
