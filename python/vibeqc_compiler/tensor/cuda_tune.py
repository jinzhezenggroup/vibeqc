"""Bounded, opt-in selection from a strict FP64 baseline.

No schedule or precision candidate becomes the default merely because it
compiles, saves bytes, or saves FLOPs.
Selection requires CPU/baseline parity and paired timing evidence on every
provided fixture. Rejected candidates and all raw samples remain in evidence.
"""

from __future__ import annotations

import time
import typing
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path

import numpy as np

from vibeqc_compiler.common.gpu_profitability import GpuProfitability
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
from .cuda_search import (
    DEFAULT_SCREENING_POLICY,
    DEFAULT_SEARCH_LIMITS,
    TensorScheduleSpace,
    TensorScreeningPolicy,
    TensorSearchLimits,
    compiled_resource_calibration,
    plan_schedule_search,
    require_compiled_resources,
)
from .interpreter import execute
from .precision import describe_precision

if typing.TYPE_CHECKING:
    from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter

    from .cuda_plan import TensorPlan, TensorSchedule


def endpoint_gate(
    baseline: typing.Any, candidate: typing.Any, *, minimum_speedup: typing.Any = 1.02
) -> dict:
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
    space: TensorScheduleSpace | None = None, *, maximum: int = 256
) -> tuple[TensorSchedule, ...]:
    """Structured, reproducible prefix; still opt-in, never installation tuning."""
    return (TensorScheduleSpace() if space is None else space).generate(maximum)


def _static_compile_shortlist(
    search: typing.Any, maximum: int
) -> tuple[tuple[int, object], ...]:
    """Rank ready plans before expensive compilation using static audit facts.

    The score only allocates the finite compilation budget. It never promotes a
    candidate and deliberately avoids a learned/opaque cost model.
    """
    ready = []
    for index, proposal in enumerate(search):
        if proposal.status != "ready":
            continue
        estimate = proposal.estimates
        profitability = GpuProfitability(**estimate["profitability"]["static"])
        ready.append(
            (
                profitability.static_compile_priority(index),
                index,
                proposal,
            )
        )
    ready.sort(key=lambda item: item[0])
    return tuple((index, proposal) for _, index, proposal in ready[:maximum])


def _compile_cost_calibration(
    estimates: typing.Any, metadata: typing.Any, wall_seconds: float
) -> dict:
    source_bytes = estimates["generated_source_bytes"]
    compiler_seconds = metadata.get("compile_seconds")
    seconds_per_kib = None
    if (
        isinstance(compiler_seconds, (int, float))
        and not isinstance(compiler_seconds, bool)
        and np.isfinite(compiler_seconds)
        and compiler_seconds >= 0
        and source_bytes
    ):
        seconds_per_kib = float(compiler_seconds) / (source_bytes / 1024)
    return {
        "schema": "vibeqc.tensor.cuda.compile-calibration.v1",
        "source_bytes_proxy": source_bytes,
        "artifact_source_bytes": metadata.get("generated_source_bytes"),
        "compiler_seconds": compiler_seconds,
        "compile_wall_seconds": wall_seconds,
        "compiler_seconds_per_source_kib": seconds_per_kib,
        "scope": "compiler-reported build duration calibrates the source-size proxy; cache/load wall time is retained separately",
    }


def _compiled_profitability(
    estimates: typing.Any,
    resources: typing.Any,
    compilation: typing.Any,
    metadata: typing.Any,
) -> dict[str, object]:
    static = estimates["profitability"]["static"]
    return GpuProfitability(
        **static,
        compiled_registers_per_thread=resources["compiled_max_registers_per_thread"],
        spill_store_bytes=resources["compiled_max_spill_store_bytes"],
        spill_load_bytes=resources["compiled_max_spill_load_bytes"],
        local_bytes=resources["compiled_max_local_bytes"],
        shared_bytes=resources["compiled_max_shared_bytes"],
        compiled_occupancy_upper_bound=resources["compiled_occupancy_upper_bound"],
        object_bytes=metadata.get("binary_bytes"),
        compile_seconds=compilation["compiler_seconds"],
    ).to_payload()


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
    fixtures: typing.Any,
    cache: Path,
    *,
    schedules: typing.Any = None,
    search_space: TensorScheduleSpace | None = None,
    precision_programs: typing.Any = None,
    search_limits: TensorSearchLimits = DEFAULT_SEARCH_LIMITS,
    screening: TensorScreeningPolicy | None = DEFAULT_SCREENING_POLICY,
    repeats: int = 8,
    maximum_seconds: float = 600,
    minimum_speedup: float = 1.02,
    device: int = 0,
) -> TensorSelection:
    """Search bounded schedules, retaining baseline unless endpoints qualify.

    Each fixture uses this plan's shape bucket but may differ in values, scale
    and caller strides. Every complete endpoint includes transfers, packing,
    validation and output copies. Representative timing only ranks candidates;
    fresh qualification still checks all fixtures and the unchanged gates.
    Set screening=None to fully qualify every compiled candidate as before.
    A finite Slurm allocation remains the hard timeout for device work; this
    deadline stops further work. The CPU interpreter is a tuning oracle only.
    Precision variants are opt-in typed TensorIR programs that must preserve
    the baseline ABI and scientific source identity; they use this same search,
    compilation cache, numerical gate, and endpoint evidence path.
    """
    if baseline.precision != "fp64":
        raise ValueError(
            "automatic TensorIR schedule promotion is currently qualified only for FP64"
        )
    if not isinstance(search_limits, TensorSearchLimits):
        raise TypeError("search_limits must be TensorSearchLimits")
    if screening is not None and not isinstance(screening, TensorScreeningPolicy):
        raise TypeError("screening must be TensorScreeningPolicy or None")
    if schedules is not None and search_space is not None:
        raise ValueError("provide schedules or search_space, not both")
    precision_programs = (
        (baseline.program,)
        if precision_programs is None
        else tuple(islice(precision_programs, search_limits.maximum_candidates + 1))
    )
    if not 1 <= len(precision_programs) <= search_limits.maximum_candidates:
        raise ValueError(
            "precision variant count exceeds the candidate limit or is empty"
        )
    maximum_schedules = max(
        1, search_limits.maximum_candidates // len(precision_programs)
    )
    schedules = (
        candidate_schedules(search_space, maximum=maximum_schedules)
        if schedules is None
        else tuple(islice(schedules, search_limits.maximum_candidates + 1))
    )
    fixtures = tuple(islice(fixtures, 9))
    if not 1 <= len(fixtures) <= 8:
        raise ValueError("tuning requires 1..8 fixtures")
    if screening is not None and max(screening.fixture_indices) >= len(fixtures):
        raise ValueError("screening fixture index is outside the provided fixtures")
    if type(repeats) is not int or not 5 <= repeats <= 30:
        raise ValueError("tuning repeats must be in 5..30")
    if not np.isfinite(maximum_seconds) or maximum_seconds <= 0:
        raise ValueError("tuning duration must be positive and finite")
    if not np.isfinite(minimum_speedup) or minimum_speedup < 1:
        raise ValueError("minimum speedup must be finite and at least one")
    if (
        baseline.schedule.views
        or baseline.schedule.fuse
        or baseline.schedule.recompute
        or baseline.schedule.layouts
    ):
        raise ValueError("tuning requires an unfused CUDA baseline")
    started = time.monotonic()

    def check_deadline() -> None:
        if time.monotonic() - started >= maximum_seconds:
            raise TimeoutError("tuning deadline exhausted")

    search = plan_schedule_search(
        baseline,
        schedules,
        search_limits,
        precision_programs=precision_programs,
    )
    compile_shortlist = _static_compile_shortlist(
        search, search_limits.maximum_compilations
    )
    compile_indices = {index for index, _ in compile_shortlist}
    compile_ranks = {
        index: rank for rank, (index, _) in enumerate(compile_shortlist, 1)
    }
    screening_plan = _screening_plan(
        screening,
        len(compile_shortlist),
        len(fixtures),
        repeats,
    )
    screening_active = screening_plan["active"]
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
    inputs_hashes = [
        canonical_hash({"equation": baseline.program.logical_hash, "feeds": feeds})
        for feeds in feed_identities
    ]
    best_plan, best_artifact, best_score = baseline, artifact, 1.0
    candidates, screened = [], []
    compilation_attempts = 0
    selected_profiles = []
    with PreparedCuda(baseline, artifact, device=device) as reference_cuda:
        identity = {
            "schema": 3,
            "baseline": reference_cuda.identity,
            "fixtures": feed_identities,
            "schedules": [asdict(s) for s in schedules],
            "precision_schedules": [
                describe_precision(program).identity for program in precision_programs
            ],
            "search_limits": asdict(search_limits),
            "screening": asdict(screening) if screening is not None else None,
            "screening_active": screening_active,
            "maximum_seconds": maximum_seconds,
            "repeats": repeats,
            "minimum_speedup": minimum_speedup,
            "numerical_gate": {"atol": 1e-11, "rtol": 1e-10},
        }
        key = canonical_hash(identity)
        # First-kernel JIT remains separate from warmed selection samples.
        startup = []
        for feeds, expected in zip(fixtures, references, strict=True):
            result = reference_cuda.execute(feeds)
            _parity(result.outputs, expected)
            startup.append(result.metrics)
            reference_cuda.execute(feeds)

        def qualify(plan: typing.Any, compiled: typing.Any, row: typing.Any) -> None:
            nonlocal best_plan, best_artifact, best_score, selected_profiles
            try:
                check_deadline()
                row.update(stage="endpoint", endpoint_attempted=True)
                # Screening contexts have already been destroyed. Keep only
                # baseline + one candidate resident, independent of shortlist size.
                with PreparedCuda(plan, compiled, device=device) as candidate:
                    timings, profiles, errors = [], [], []
                    row.update(samples=timings, profiles=profiles)
                    for index, (feeds, expected) in enumerate(
                        zip(fixtures, references, strict=True)
                    ):
                        pairs, error, profile = _measure_fixture(
                            reference_cuda,
                            candidate,
                            feeds,
                            expected,
                            inputs_hash=inputs_hashes[index],
                            repeats=repeats,
                            check_deadline=check_deadline,
                            profile=True,
                        )
                        timings.append(_timing_evidence(pairs))
                        errors.append(error)
                        profiles.append(profile)
                    gates = [
                        endpoint_gate(
                            *_paired_seconds(pairs), minimum_speedup=minimum_speedup
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

        for index, proposal in enumerate(search):
            row = proposal.to_payload()
            candidates.append(row)
            if proposal.status != "ready":
                continue
            row["static_compile_priority"] = {
                **proposal.estimates["profitability"]["static"],
                "generation_index": index,
            }
            if index not in compile_indices:
                row.update(
                    status="skipped",
                    stage="compile-budget",
                    reason="outside static compile shortlist; ranked by shared GPU profitability (traffic, pressure/occupancy, launches, source) and generation order",
                )
                continue
            row["static_compile_rank"] = compile_ranks[index]
            if time.monotonic() - started >= maximum_seconds:
                row.update(
                    status="skipped",
                    stage="deadline",
                    reason="tuning deadline exhausted",
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
                row["compile_calibration"] = _compile_cost_calibration(
                    proposal.estimates,
                    compiled.metadata,
                    row["compile_wall_seconds"],
                )
                row["stage"] = "compiled-resource"
                resources = compiled.metadata.get("resources", [])
                require_compiled_resources(
                    plan,
                    resources,
                    minimum_resident_blocks=search_limits.minimum_resident_blocks,
                )
                row["resource_calibration"] = compiled_resource_calibration(
                    plan, proposal.estimates, resources
                )
                row["profitability"] = _compiled_profitability(
                    proposal.estimates,
                    row["resource_calibration"],
                    row["compile_calibration"],
                    compiled.metadata,
                )
                check_deadline()
                if not screening_active:
                    qualify(plan, compiled, row)
                    continue
                row["stage"] = "representative-timing"
                screen = {
                    "scope": "ranking only; not performance qualification",
                    "fixture_indices": list(screening.fixture_indices),
                    "repeats": screening.repeats,
                    "samples": [],
                }
                row["screening"] = screen
                errors = []
                with PreparedCuda(plan, compiled, device=device) as candidate:
                    for fixture_index in screening.fixture_indices:
                        pairs, error, _ = _measure_fixture(
                            reference_cuda,
                            candidate,
                            fixtures[fixture_index],
                            references[fixture_index],
                            inputs_hash=inputs_hashes[fixture_index],
                            repeats=screening.repeats,
                            check_deadline=check_deadline,
                            profile=False,
                        )
                        screen["samples"].append(_timing_evidence(pairs))
                        errors.append(error)
                screen.update(
                    score=min(_screening_speedup(pairs) for pairs in screen["samples"]),
                    max_absolute_error=max(errors),
                )
                row["status"] = "screened"
                screened.append((plan, compiled, row))
            except (ValueError, RuntimeError, TimeoutError) as error:
                row.update(status="rejected", reason=str(error))

        if screening_active:
            # Stable sorting breaks exact ties by original generation order.
            # No screen-speed threshold: noisy/negative screens can still reach
            # qualification. Only the unchanged complete gates can promote.
            ranked = sorted(screened, key=lambda item: -item[2]["screening"]["score"])
            for rank, (_, _, row) in enumerate(ranked, 1):
                row["shortlist_rank"] = rank
                if rank > screening.maximum_finalists:
                    row.update(
                        status="pruned",
                        stage="shortlist",
                        reason="outside representative-timing shortlist; ranking only",
                    )
            for plan, compiled, row in ranked[: screening.maximum_finalists]:
                qualify(plan, compiled, row)

        evidence = {
            "schema": "vibeqc.tensor.cuda.tuning",
            "schema_version": 3,
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
            "screening_plan": screening_plan,
            "search_summary": {
                "generated": len(search),
                "pruned_before_compile": sum(p.status == "pruned" for p in search),
                "static_compile_shortlist": len(compile_shortlist),
                "static_compile_budget_skips": sum(
                    r["stage"] == "compile-budget" for r in candidates
                ),
                "compilation_attempts": compilation_attempts,
                "screening_candidates": sum("screening" in r for r in candidates),
                "screened_candidates": len(screened),
                "shortlist_pruned": sum(r["stage"] == "shortlist" for r in candidates),
                "endpoint_candidates": sum(
                    r.get("endpoint_attempted", False) for r in candidates
                ),
                "accepted": sum(r["status"] == "accepted" for r in candidates),
            },
            "seconds": time.monotonic() - started,
            "graph_status": reference_cuda.graph_status,
        }
    path = Path(cache) / "selections" / key / "evidence.json"
    atomic_json(path, evidence)
    return TensorSelection(best_plan, best_artifact, evidence, path)


def _screening_plan(
    policy: typing.Any,
    candidate_budget: typing.Any,
    fixture_count: typing.Any,
    repeats: typing.Any,
) -> typing.Any:
    """Avoid a shortlist when its planned sample count cannot save any work.

    Counts are A/B pairs only, not predicted time. Startup, compilation and
    profiling costs are excluded; a lower count is not a speedup claim.
    """
    full = candidate_budget * fixture_count * repeats
    probe = (
        0
        if policy is None
        else candidate_budget * len(policy.fixture_indices) * policy.repeats
    )
    final = (
        full
        if policy is None
        else min(candidate_budget, policy.maximum_finalists) * fixture_count * repeats
    )
    if policy is None:
        reason = "screening explicitly disabled"
    elif candidate_budget <= policy.maximum_finalists:
        reason = "candidate budget already fits the finalist limit"
    elif probe + final >= full:
        reason = "screening would not reduce planned measurement pairs"
    else:
        reason = "representative shortlist reduces planned measurement pairs"
    return {
        "active": policy is not None and probe + final < full,
        "reason": reason,
        "candidate_budget": candidate_budget,
        "unfiltered_pairs": full,
        "screening_pairs": probe,
        "finalist_pairs": final,
        "scope": "paired timing samples only; excludes startup, profiling and compilation",
    }


def _timing_evidence(pairs: typing.Any) -> typing.Any:
    """Retain invalid clock samples without emitting nonstandard JSON NaN/Inf."""
    rows = []
    for sample in pairs:
        row = dict(sample)
        seconds = row["seconds"]
        if isinstance(seconds, (float, np.floating)) and not np.isfinite(seconds):
            row.update(seconds=None, invalid_seconds=repr(float(seconds)))
        rows.append(row)
    return rows


def _paired_seconds(pairs: typing.Any) -> typing.Any:
    return tuple(
        np.asarray(
            [row["seconds"] for row in pairs if row["selection"] == side], dtype=float
        )
        for side in ("baseline", "candidate")
    )


def _screening_speedup(pairs: typing.Any) -> typing.Any:
    """A finite descriptive ratio, not a promotion or statistical decision."""
    left, right = (np.asarray(values) for values in _paired_seconds(pairs))
    if (
        left.ndim != 1
        or left.shape != right.shape
        or not 5 <= left.size <= 30
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or np.min(left) <= 0
        or np.min(right) <= 0
    ):
        raise ValueError("screening requires 5..30 positive finite paired samples")
    ratio = float(np.median(left) / np.median(right))
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("screening speedup must be positive and finite")
    return ratio


def _measure_fixture(
    reference_cuda: typing.Any,
    candidate: typing.Any,
    feeds: typing.Any,
    expected: typing.Any,
    *,
    inputs_hash: typing.Any,
    repeats: typing.Any,
    check_deadline: typing.Any,
    profile: typing.Any,
) -> typing.Any:
    """Warm and measure one full endpoint, checking every returned output.

    Both screening and final qualification use the same synchronized runner and
    numerical gates. Final qualification calls this again for fresh samples.
    Neither comparisons nor optional profiling contaminate timing windows.
    """
    check_deadline()
    result = candidate.execute(feeds)
    error = _parity(result.outputs, expected)
    check_deadline()
    error = max(error, _parity(result.outputs, reference_cuda.execute(feeds).outputs))
    check_deadline()
    error = max(error, _parity(candidate.execute(feeds).outputs, expected))
    latest = [None]

    def before_sample(selection: typing.Any) -> None:
        nonlocal error
        check_deadline()
        if latest[0] is not None:
            error = max(error, _parity(latest[0].outputs, expected))

    def evaluate(selection: typing.Any) -> typing.Any:
        selected = reference_cuda if selection == "baseline" else candidate
        latest[0] = selected.execute(feeds)
        return latest[0].metrics

    pairs = measure_interleaved(
        evaluate,
        lambda: None,
        prepare=before_sample,
        repeats=repeats,
        workload="unchanged-geometry",
        inputs_hash=inputs_hash,
    )
    error = max(error, _parity(latest[0].outputs, expected))
    metrics = None
    if profile:
        metrics = {}
        for name, prepared in (("baseline", reference_cuda), ("candidate", candidate)):
            check_deadline()
            result = prepared.execute(feeds, profile=True)
            error = max(error, _parity(result.outputs, expected))
            metrics[name] = result.metrics
    return pairs, error, metrics


def _promotion_profiles(
    plan: typing.Any,
    artifact: typing.Any,
    feeds: typing.Any,
    evidence_hash: typing.Any,
    *,
    baseline_execution: typing.Any,
) -> typing.Any:
    """Declare only the measured layout domains using #459's shared records.

    No new profile database or runtime lookup is introduced. These records refer
    to #136's existing executable key and the candidate's complete evidence hash;
    they must not be treated as a general promotion to unmeasured inputs/targets.
    """
    precision = plan.precision_schedule
    identity = CompilationIdentity(
        precision.source_equation,
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
                ("equation", precision.source_equation),
                ("precision_schedule", precision.identity),
                ("math_mode", precision.math_mode),
                ("strict_audit_dtype", precision.strict_audit_dtype),
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
            schedule_hash=canonical_hash(
                {
                    "schedule": asdict(plan.schedule),
                    "precision_schedule": precision.identity,
                }
            ),
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


def _parity(actual: typing.Any, expected: typing.Any) -> typing.Any:
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
