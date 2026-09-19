"""Device-free schedule search and staged tuning regression contracts."""

from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    cuda_tune,
    einsum,
    input_tensor,
    multiply,
)
from vibeqc_compiler.tensor.cuda_plan import Reservations, TensorSchedule, plan_cuda
from vibeqc_compiler.tensor.cuda_search import (
    TensorScheduleSpace,
    TensorScreeningPolicy,
    TensorSearchLimits,
    estimate_schedule,
    execution_key,
    plan_schedule_search,
    require_compiled_resources,
)
from vibeqc_compiler.tensor.interpreter import execute

TARGET = cuda_target_info("sm_80")


def vector_program(size=65):
    i = Index("i", IndexSpace("axis", "batch", size))
    x = input_tensor("x", TensorSpec((i,), role="input"))
    return Program({"result": add(multiply(x, x), x)})


def gemm_program(*, packed=False):
    i = Index("i", IndexSpace("rows", "batch", 65))
    j = Index("j", IndexSpace("cols", "batch", 97))
    k = Index("k", IndexSpace("inner", "batch", 129))
    a = input_tensor("a", TensorSpec((i, k), role="input"))
    b = input_tensor("b", TensorSpec((k, j), role="input"))
    return Program({"result": einsum("ik,kj->ji" if packed else "ik,kj->ij", a, b)})


def test_structured_search_is_bounded_reproducible_and_covers_each_axis():
    space = TensorScheduleSpace()
    schedules = space.generate()
    assert len(schedules) == len(set(schedules)) == 128
    assert space.cardinality > len(schedules)
    assert schedules == space.generate()
    assert space.generate(7) == schedules[:7]
    for name, values in asdict(space).items():
        assert {getattr(s, name) for s in schedules} == set(values)
    custom = replace(space, threads=(32, 128, 512, 1024))
    assert {s.threads for s in custom.generate()} == {32, 128, 512, 1024}


@pytest.mark.parametrize(
    "options",
    [
        {"threads": ()},
        {"tile_m": (0,)},
        {"views": (1,)},
        {"threads": (128, 128)},
        {"tile_n": (False,)},
    ],
)
def test_bad_search_axes_fail_before_generation(options):
    with pytest.raises((ValueError, TypeError)):
        TensorScheduleSpace(**options)


@pytest.mark.parametrize(
    "options",
    [
        {"maximum_candidates": 0},
        {"maximum_compilations": True},
        {"maximum_candidates": 4097},
        {"maximum_source_bytes": -1},
        {"minimum_resident_blocks": 0},
        {"maximum_candidates": 2},
    ],
)
def test_search_limits_are_explicit_and_finite(options):
    with pytest.raises(ValueError):
        TensorSearchLimits(**options)


def test_effective_key_ignores_noop_tiles_but_preserves_real_execution_changes():
    baseline = plan_cuda(gemm_program(), TARGET)
    no_op = plan_cuda(
        baseline.program, TARGET, schedule=TensorSchedule(tile_m=32, views=True)
    )
    assert no_op.identity != baseline.identity
    assert execution_key(no_op) == execution_key(baseline)
    for schedule in (TensorSchedule(threads=64), TensorSchedule(direct_gemm=False)):
        candidate = plan_cuda(baseline.program, TARGET, schedule=schedule)
        assert execution_key(candidate) != execution_key(baseline)
    packed = plan_cuda(gemm_program(packed=True), TARGET)
    small = plan_cuda(packed.program, TARGET, schedule=TensorSchedule(tile_m=32))
    assert execution_key(small) != execution_key(packed)
    large = plan_cuda(packed.program, TARGET, schedule=TensorSchedule(tile_m=256))
    assert execution_key(large) == execution_key(packed)


def test_default_search_prunes_equivalent_plans_and_preserves_baseline():
    baseline = plan_cuda(vector_program(), TARGET)
    original = baseline.to_payload()
    candidates = plan_schedule_search(baseline, TensorScheduleSpace().generate())
    ready = [c for c in candidates if c.status == "ready"]
    assert len(candidates) == 128
    assert 0 < len(ready) <= 5
    assert len({execution_key(c.plan) for c in ready}) == len(ready)
    assert all(c.equivalent_to for c in candidates if c.stage == "duplicate")
    assert baseline.to_payload() == original
    assert all(
        c.plan.program.logical_hash == baseline.program.logical_hash for c in ready
    )


def test_duplicate_does_not_even_emit_source(monkeypatch):
    import vibeqc_compiler.tensor.cuda_search as search

    baseline = plan_cuda(vector_program(), TARGET)
    monkeypatch.setattr(search, "emit_cuda", lambda _: pytest.fail("duplicate emitted"))
    (candidate,) = plan_schedule_search(baseline, [TensorSchedule(views=True)])
    assert candidate.stage == "duplicate"
    assert candidate.equivalent_to == baseline.identity


def test_pruning_has_legality_source_register_and_occupancy_reasons():
    baseline = plan_cuda(vector_program(), TARGET)
    (invalid,) = plan_schedule_search(baseline, [TensorSchedule(threads=48)])
    assert invalid.stage == "legality"
    (source,) = plan_schedule_search(
        baseline,
        [TensorSchedule(fuse=True)],
        TensorSearchLimits(maximum_source_bytes=1),
    )
    assert source.status == "pruned"
    assert "compile-cost" in source.reason
    limited = plan_cuda(baseline.program, replace(TARGET, tuning_maximum_registers=16))
    (registers,) = plan_schedule_search(limited, [TensorSchedule(fuse=True)])
    assert "register pressure" in registers.reason
    (occupancy,) = plan_schedule_search(
        baseline,
        [TensorSchedule(threads=1024)],
        TensorSearchLimits(minimum_resident_blocks=3),
    )
    assert "resident-block" in occupancy.reason


def test_static_accounting_reuses_combined_numeric_budget_and_labels_unknowns():
    baseline = plan_cuda(
        gemm_program(packed=True),
        TARGET,
        reservations=Reservations(t=1024, concurrent=4096),
    )
    estimate = estimate_schedule(baseline)
    assert estimate["peak_numeric_bytes"] == baseline.device_bytes + baseline.host_bytes
    assert estimate["panel_bytes"] == baseline.panel_bytes > 0
    assert estimate["estimated_shared_bytes"] == 0  # global panels are not shared tiles
    assert estimate["estimated_local_bytes"] is None  # spills need PTXAS, not guesses
    assert estimate["generated_source_bytes"] > 0
    assert "excludes" in estimate["traffic_scope"]
    assert "not predicted seconds" in estimate["compile_cost_proxy"]


def resources(**changes):
    return {
        "function": "kernel",
        "registers": 32,
        "stack_bytes": 0,
        "spill_store_bytes": 0,
        "spill_load_bytes": 0,
        "shared_bytes": 0,
        **changes,
    }


@pytest.mark.parametrize(
    "records",
    [
        [],
        [{}],
        [resources(registers="32")],
        [resources(registers=-1)],
        [resources(registers=193)],
        [resources(stack_bytes=129)],
        [resources(spill_store_bytes=1)],
        [resources(spill_load_bytes=1)],
        [resources(shared_bytes=65536)],
    ],
)
def test_compiled_resource_gate_fails_closed(records):
    with pytest.raises(ValueError, match="compiled resource"):
        require_compiled_resources(plan_cuda(vector_program(), TARGET), records)


def test_compiled_register_block_cliff_is_rejected():
    plan = plan_cuda(vector_program(), TARGET, schedule=TensorSchedule(threads=1024))
    with pytest.raises(ValueError, match="compiled resource gate"):
        require_compiled_resources(plan, [resources(registers=96)])
    require_compiled_resources(plan, [resources(registers=32)])


@pytest.fixture
def fake_cuda(monkeypatch):
    """Exercise the actual tuner with CPU outputs and explicitly synthetic times."""
    calls = SimpleNamespace(
        compiled=[],
        prepared=[],
        measured=0,
        bad_resources=False,
        noisy=False,
        measurements=[],
        timings={},
        active=0,
        maximum_active=0,
    )

    def compile_plan(plan, compiler, cache):
        calls.compiled.append(plan)
        return SimpleNamespace(
            metadata={
                "key": canonical_hash({"artifact": plan.identity}),
                "identity": {
                    "source": "unit-test-only",
                    "plan": plan.identity,
                    "generated": "mock",
                },
                "resources": []
                if calls.bad_resources and len(calls.compiled) > 1
                else [resources()],
                "compile_seconds": 0.0,
                "binary_bytes": 0,
            }
        )

    class Prepared:
        graph_status = "unit-test-double; no CUDA execution"

        def __init__(self, plan, artifact, device=0):
            self.device = {"name": "synthetic"}
            self.plan = plan
            self.identity = canonical_hash(
                {"plan": plan.identity, "device": self.device}
            )
            calls.prepared.append(plan)

        def __enter__(self):
            calls.active += 1
            calls.maximum_active = max(calls.maximum_active, calls.active)
            return self

        def __exit__(self, *args):
            calls.active -= 1

        def execute(self, feeds, profile=False):
            return SimpleNamespace(
                outputs=execute(self.plan.program, feeds).outputs,
                metrics={
                    "threads": self.plan.schedule.threads,
                    "fixture": float(feeds["x"][0]),
                },
            )

    def measure(evaluate, synchronize, *, prepare, repeats, **kwargs):
        calls.measured += 1
        result = []
        for _ in range(repeats):
            for name, seconds in (("baseline", 10), ("candidate", 8)):
                prepare(name)
                metrics = evaluate(name)
                if name == "candidate":
                    seconds = calls.timings.get(
                        (metrics["threads"], metrics["fixture"]), seconds
                    )
                result.append({"selection": name, "seconds": seconds})
        calls.measurements.append({**metrics, "repeats": repeats, "samples": result})
        return result

    monkeypatch.setattr(cuda_tune, "compile_cuda", compile_plan)
    monkeypatch.setattr(cuda_tune, "PreparedCuda", Prepared)
    monkeypatch.setattr(cuda_tune, "measure_interleaved", measure)
    monkeypatch.setattr(
        cuda_tune,
        "assess_comparison",
        lambda _: {"status": "inconclusive" if calls.noisy else "pass"},
    )
    return calls


def run_fake_tuning(tmp_path, **options):
    baseline = plan_cuda(vector_program(), TARGET)
    return cuda_tune.tune_cuda(
        baseline,
        None,
        [{"x": np.linspace(-1, 1, 65)}],
        tmp_path,
        repeats=5,
        **options,
    )


def test_tuner_bounds_actual_compiles_and_emits_guarded_endpoint_profiles(
    tmp_path, fake_cuda
):
    result = run_fake_tuning(
        tmp_path,
        schedules=[TensorSchedule(threads=t) for t in (64, 256, 512)],
        search_limits=TensorSearchLimits(maximum_candidates=3, maximum_compilations=1),
    )
    assert len(fake_cuda.compiled) == 2  # one mandatory baseline and one candidate
    assert fake_cuda.measured == 1
    assert result.evidence["search_summary"]["compilation_attempts"] == 1
    assert [c["status"] for c in result.evidence["candidates"]] == [
        "accepted",
        "skipped",
        "skipped",
    ]
    assert result.plan.schedule.threads == 64
    assert result.evidence_path.exists()
    (profile,) = result.evidence["selected_profiles"]
    assert profile["profile"]["artifact_key"] == result.artifact.metadata["key"]
    assert profile["profile"]["schedule_hash"] == canonical_hash(
        asdict(result.plan.schedule)
    )
    assert any(
        p["feature"] == "input_layout"
        for p in profile["profile"]["performance"]["predicates"]
    )
    assert len(result.evidence["candidates"][0]["samples"][0]) == 10


def test_default_search_reuses_existing_cache_and_never_compiles_duplicates(
    tmp_path, fake_cuda
):
    result = run_fake_tuning(tmp_path)
    summary = result.evidence["search_summary"]
    assert summary["generated"] == 128
    assert summary["pruned_before_compile"] >= 123
    assert len(fake_cuda.compiled) <= 6
    assert result.evidence_path.parent.parent.name == "selections"


@pytest.mark.parametrize("failure", ["static", "resources", "noise", "timeout"])
def test_negative_evidence_keeps_baseline_without_promoting(
    tmp_path, fake_cuda, failure
):
    options = {"schedules": [TensorSchedule(fuse=True)]}
    if failure == "static":
        options["search_limits"] = TensorSearchLimits(maximum_source_bytes=1)
    elif failure == "resources":
        fake_cuda.bad_resources = True
    elif failure == "noise":
        fake_cuda.noisy = True
    else:
        options["maximum_seconds"] = 1e-12
    result = run_fake_tuning(tmp_path, **options)
    assert result.plan.schedule == TensorSchedule()
    assert result.evidence["selected_profiles"] == []
    assert result.evidence["candidates"][0]["status"] != "accepted"
    if failure != "noise":
        assert fake_cuda.measured == 0
    if failure in ("static", "timeout"):
        assert len(fake_cuda.compiled) == 1
    if failure == "resources":
        assert len(fake_cuda.prepared) == 1  # reject before loading/executing candidate


def test_candidate_overflow_fails_before_any_compilation(tmp_path, fake_cuda):
    with pytest.raises(ValueError, match="candidate limit"):
        run_fake_tuning(tmp_path, schedules=[TensorSchedule()] * 129)
    assert fake_cuda.compiled == []


def test_promotion_profiles_use_shared_selector_and_reject_unmeasured_layout(
    tmp_path, fake_cuda
):
    from vibeqc_compiler.common.backend import TargetInfo
    from vibeqc_compiler.common.specialization import (
        CompilationIdentity,
        GuardPredicate,
        ImplementationProfile,
        SpecializationGuard,
        TargetCapabilities,
        WorkloadSignature,
        select_specialization,
    )

    result = run_fake_tuning(tmp_path, schedules=[TensorSchedule(fuse=True)])
    (domain,) = result.evidence["selected_profiles"]
    raw = domain["profile"]
    identity = CompilationIdentity(**raw["identity"])
    profile = ImplementationProfile(
        **{
            **raw,
            "identity": identity,
            "correctness": SpecializationGuard(
                tuple(GuardPredicate(**p) for p in raw["correctness"]["predicates"])
            ),
            "performance": SpecializationGuard(
                tuple(GuardPredicate(**p) for p in raw["performance"]["predicates"])
            ),
        }
    )
    target = TargetCapabilities(
        TargetInfo(**domain["target"]["target"]), domain["target"]["features"]
    )
    workload = WorkloadSignature(**domain["workload"])
    fallback = replace(
        profile, name="baseline", artifact_key="b" * 64, performance=None
    )
    options = {
        "target": target,
        "identity": identity,
        "profiles": (profile,),
        "fallback": fallback,
    }
    assert select_specialization(workload=workload, **options).selected == profile
    for feature in ("input_layout", "baseline_execution"):
        changed = replace(
            workload,
            features=tuple(
                (key, "unmeasured" if key == feature else value)
                for key, value in workload.features
            ),
        )
        decision = select_specialization(workload=changed, **options)
        assert decision.status == "fallback"
        assert decision.selected == fallback
        assert any(
            feature in reason for reason in decision.evaluations[0].promotion_failures
        )
    missing_target = replace(target, features=())
    assert (
        select_specialization(
            workload=workload, **(options | {"target": missing_target})
        ).status
        == "unsupported"
    )


def test_multiple_measured_layouts_have_separate_profiles_and_no_value_guard(
    tmp_path, fake_cuda
):
    baseline = plan_cuda(vector_program(), TARGET)
    source = np.linspace(-1, 1, 130)
    fixtures = [
        {"x": source[::2]},
        {"x": source[1::2].copy()},
        {"x": source[1::2].copy() * 2},
    ]
    result = cuda_tune.tune_cuda(
        baseline,
        None,
        fixtures,
        tmp_path,
        repeats=5,
        schedules=[TensorSchedule(fuse=True)],
    )
    assert len(result.evidence["selected_profiles"]) == 2
    assert fake_cuda.measured == 3
    for domain in result.evidence["selected_profiles"]:
        assert all("values" not in name for name, _ in domain["workload"]["features"])


def test_candidate_numerical_failure_never_creates_a_profile(
    tmp_path, fake_cuda, monkeypatch
):
    prepared = cuda_tune.PreparedCuda

    class BadPrepared(prepared):
        def execute(self, feeds, profile=False):
            result = super().execute(feeds, profile=profile)
            if self.plan.schedule.fuse:
                result.outputs["result"][0] += 1
            return result

    monkeypatch.setattr(cuda_tune, "PreparedCuda", BadPrepared)
    result = run_fake_tuning(tmp_path, schedules=[TensorSchedule(fuse=True)])
    assert result.plan.schedule == TensorSchedule()
    assert result.evidence["selected_profiles"] == []
    assert "numerical gate failed" in result.evidence["candidates"][0]["reason"]
    assert fake_cuda.measured == 0


def test_unemittable_candidate_does_not_abort_other_candidates(monkeypatch):
    import vibeqc_compiler.tensor.cuda_search as search

    emit = search.emit_cuda

    def emit_supported(plan):
        if plan.schedule.fuse:
            raise ValueError("candidate emission unsupported")
        return emit(plan)

    monkeypatch.setattr(search, "emit_cuda", emit_supported)
    baseline = plan_cuda(vector_program(), TARGET)
    rejected, accepted = plan_schedule_search(
        baseline, [TensorSchedule(fuse=True), TensorSchedule(threads=64)]
    )
    assert rejected.status == "pruned"
    assert rejected.stage == "legality"
    assert "emission unsupported" in rejected.reason
    assert accepted.status == "ready"


@pytest.mark.parametrize(
    "options",
    [
        {"maximum_finalists": 0},
        {"maximum_finalists": True},
        {"maximum_finalists": 4097},
        {"repeats": 4},
        {"repeats": 31},
        {"repeats": 5.0},
        {"fixture_indices": ()},
        {"fixture_indices": (0, 0)},
        {"fixture_indices": (-1,)},
        {"fixture_indices": (8,)},
        {"fixture_indices": (True,)},
        {"fixture_indices": (0.0,)},
    ],
)
def test_screening_policy_rejects_invalid_or_unbounded_inputs(options):
    with pytest.raises(ValueError):
        TensorScreeningPolicy(**options)


def test_screening_policy_copies_fixture_indices():
    indices = [1, 0]
    policy = TensorScreeningPolicy(fixture_indices=indices)
    indices[0] = 2
    assert policy.fixture_indices == (1, 0)


@pytest.mark.parametrize(
    "screening", ["auto", False, TensorScreeningPolicy(fixture_indices=(1,))]
)
def test_invalid_screening_fails_before_compilation(tmp_path, fake_cuda, screening):
    with pytest.raises((ValueError, TypeError), match="screening"):
        run_fake_tuning(tmp_path, screening=screening)
    assert fake_cuda.compiled == []


def run_screened_tuning(tmp_path, *, fixtures=2, **options):
    return cuda_tune.tune_cuda(
        plan_cuda(vector_program(), TARGET),
        None,
        [{"x": np.linspace(i, i + 1, 65)} for i in range(fixtures)],
        tmp_path,
        schedules=[TensorSchedule(threads=t) for t in (64, 256, 512)],
        **(
            {"repeats": 7, "screening": TensorScreeningPolicy(maximum_finalists=1)}
            | options
        ),
    )


def test_shortlist_ranks_all_compiled_candidates_before_full_qualification(
    tmp_path, fake_cuda
):
    fake_cuda.timings = {(64, 0): 9, (256, 0): 5, (512, 0): 7}
    result = run_screened_tuning(tmp_path)
    summary = result.evidence["search_summary"]
    assert summary["compilation_attempts"] == summary["screened_candidates"] == 3
    assert summary["endpoint_candidates"] == 1
    assert summary["shortlist_pruned"] == 2
    assert result.evidence["schema_version"] == 3
    assert result.evidence["screening_plan"]["active"]
    assert result.plan.schedule.threads == 256  # not the first compiled candidate
    assert fake_cuda.measured == 5  # 3 screens + 2 fresh all-fixture measurements
    assert [m["repeats"] for m in fake_cuda.measurements] == [5, 5, 5, 7, 7]
    assert [m["fixture"] for m in fake_cuda.measurements] == [0, 0, 0, 0, 1]
    rows = result.evidence["candidates"]
    assert [r["shortlist_rank"] for r in rows] == [3, 1, 2]
    assert [r["status"] for r in rows] == ["pruned", "accepted", "pruned"]
    assert all(
        "promotion_profiles" not in r and "gates" not in r for r in (rows[0], rows[2])
    )
    assert len(rows[1]["samples"]) == 2
    assert rows[1]["samples"][0] is not rows[1]["screening"]["samples"][0]
    assert fake_cuda.active == 0 and fake_cuda.maximum_active == 2
    assert len(fake_cuda.compiled) == 4  # finalists reuse artifacts, not recompilation
    assert len(fake_cuda.prepared) == 5  # baseline, 3 screens, reopened finalist


def test_screen_success_cannot_bypass_a_later_fixture_performance_failure(
    tmp_path, fake_cuda
):
    fake_cuda.timings = {(64, 0): 1, (64, 1): 12}
    result = run_screened_tuning(tmp_path)
    assert result.plan.schedule == TensorSchedule()
    assert result.evidence["selected_profiles"] == []
    row = result.evidence["candidates"][0]
    assert row["screening"]["score"] == 10
    assert row["gates"][0]["passed"] and not row["gates"][1]["passed"]
    assert row["status"] == "rejected" and "promotion_profiles" not in row


def test_screen_success_cannot_bypass_later_fixture_numerical_failure(
    tmp_path, fake_cuda, monkeypatch
):
    prepared = cuda_tune.PreparedCuda

    class BadPrepared(prepared):
        def execute(self, feeds, profile=False):
            result = super().execute(feeds, profile=profile)
            if self.plan.schedule.threads == 64 and feeds["x"][0] == 1:
                result.outputs["result"][0] += 1
            return result

    monkeypatch.setattr(cuda_tune, "PreparedCuda", BadPrepared)
    result = run_screened_tuning(tmp_path)
    row = result.evidence["candidates"][0]
    assert row["screening"]["score"] > 1
    assert "numerical gate failed" in row["reason"]
    assert len(row["samples"]) == 1  # retain the first completed fixture
    assert result.evidence["selected_profiles"] == []
    assert fake_cuda.active == 0


def test_shared_noise_gate_still_controls_screened_winners(tmp_path, fake_cuda):
    fake_cuda.noisy = True
    result = run_screened_tuning(tmp_path)
    assert result.evidence["selected_profiles"] == []
    assert result.evidence["candidates"][0]["gates"][0]["passed"]
    assert result.evidence["candidates"][0]["status"] == "rejected"


def test_negative_screens_are_ranked_not_mistaken_for_promotions(tmp_path, fake_cuda):
    fake_cuda.timings = {(t, i): 11 for t in (64, 256, 512) for i in (0, 1)}
    result = run_screened_tuning(tmp_path)
    assert result.evidence["search_summary"]["endpoint_candidates"] == 1
    assert result.evidence["selected_profiles"] == []
    assert result.evidence["candidates"][0]["shortlist_rank"] == 1  # stable tie


def test_screening_uses_all_declared_representatives_and_worst_ratio(
    tmp_path, fake_cuda
):
    fake_cuda.timings = {(64, 0): 2, (64, 1): 12, (256, 0): 7, (256, 1): 7}
    result = run_screened_tuning(
        tmp_path,
        fixtures=4,
        repeats=10,
        screening=TensorScreeningPolicy(maximum_finalists=1, fixture_indices=(1, 0)),
    )
    assert result.plan.schedule.threads == 256
    assert [m["fixture"] for m in fake_cuda.measurements[:6]] == [1, 0] * 3
    assert result.evidence["candidates"][0]["screening"]["score"] == pytest.approx(
        10 / 12
    )
    assert fake_cuda.measured == 10  # six screens + four qualification fixtures


@pytest.mark.parametrize(
    "options, reason",
    [
        ({"screening": None}, "explicitly disabled"),
        ({"screening": TensorScreeningPolicy(maximum_finalists=3)}, "already fits"),
        ({"fixtures": 1, "repeats": 5}, "would not reduce"),
    ],
)
def test_screening_is_bypassed_when_disabled_or_not_cost_effective(
    tmp_path, fake_cuda, options, reason
):
    result = run_screened_tuning(tmp_path, **options)
    assert not result.evidence["screening_plan"]["active"]
    assert reason in result.evidence["screening_plan"]["reason"]
    assert result.evidence["search_summary"]["screening_candidates"] == 0
    assert result.evidence["search_summary"]["endpoint_candidates"] == 3
    assert all("screening" not in r for r in result.evidence["candidates"])


def test_screening_policy_and_selected_representatives_change_evidence_identity(
    tmp_path, fake_cuda
):
    original = run_screened_tuning(tmp_path)
    changed = run_screened_tuning(
        tmp_path,
        screening=TensorScreeningPolicy(maximum_finalists=1, fixture_indices=(1,)),
    )
    disabled = run_screened_tuning(tmp_path, screening=None)
    assert len({r.evidence["key"] for r in (original, changed, disabled)}) == 3
    assert original.plan.program.logical_hash == changed.plan.program.logical_hash


def test_compiled_resource_failures_never_reach_screening(tmp_path, fake_cuda):
    fake_cuda.bad_resources = True
    result = run_screened_tuning(tmp_path)
    assert fake_cuda.measured == 0
    assert result.evidence["search_summary"]["screening_candidates"] == 0
    assert result.evidence["selected_profiles"] == []


@pytest.mark.parametrize(
    "invalid", [float("nan"), float("inf"), -float("inf"), 0.0, -1.0]
)
def test_invalid_probe_samples_are_retained_but_never_ranked(
    tmp_path, fake_cuda, invalid
):
    fake_cuda.timings = {(t, 0): invalid for t in (64, 256, 512)}
    result = run_screened_tuning(tmp_path)
    assert result.evidence["selected_profiles"] == []
    assert result.evidence["search_summary"]["endpoint_candidates"] == 0
    assert result.evidence["search_summary"]["screened_candidates"] == 0
    assert all(
        "positive finite paired" in r["reason"] for r in result.evidence["candidates"]
    )


def test_deadline_after_screening_does_not_promote_or_reopen_finalist(
    tmp_path, fake_cuda, monkeypatch
):
    clock = [0.0]
    monkeypatch.setattr(cuda_tune.time, "monotonic", lambda: clock[0])
    measure = cuda_tune.measure_interleaved

    def expire_after_third_screen(*args, **kwargs):
        rows = measure(*args, **kwargs)
        if fake_cuda.measured == 3:
            clock[0] = 601.0
        return rows

    monkeypatch.setattr(cuda_tune, "measure_interleaved", expire_after_third_screen)
    result = run_screened_tuning(tmp_path)
    assert result.evidence["selected_profiles"] == []
    assert result.evidence["search_summary"]["endpoint_candidates"] == 0
    assert "deadline exhausted" in result.evidence["candidates"][0]["reason"]
    assert len(fake_cuda.prepared) == 4 and fake_cuda.active == 0
