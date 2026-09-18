"""Synthetic specialization contracts: no GPU probing or performance claims."""

import json
from dataclasses import FrozenInstanceError, replace

import pytest
from vibeqc_compiler.common.backend import TargetInfo
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.common.specialization import (
    CompilationIdentity,
    GuardPredicate,
    ImplementationProfile,
    SpecializationGuard,
    TargetCapabilities,
    WorkloadSignature,
    select_specialization,
)


def guard(*predicates):
    return SpecializationGuard(tuple(GuardPredicate(*p) for p in predicates))


def with_features(record, **updates):
    return replace(record, features=tuple((dict(record.features) | updates).items()))


@pytest.fixture
def case():
    identity = CompilationIdentity(
        canonical_hash("science"), canonical_hash("compiler")
    )
    correctness = guard(
        ("workload", "kind", "eq", "df_exchange"),
        ("target", "backend", "eq", "cuda"),
        ("target", "fp64", "eq", True),
    )
    fallback = ImplementationProfile(
        "generic",
        identity,
        canonical_hash("generic-artifact"),
        canonical_hash("generic-schedule"),
        canonical_hash("profile-v1"),
        correctness,
    )
    tuned = replace(
        fallback,
        name="bounded-example",
        artifact_key=canonical_hash("tuned-artifact"),
        schedule_hash=canonical_hash("tuned-schedule"),
        correctness=SpecializationGuard(
            correctness.predicates
            + guard(("workload", "resident", "eq", True)).predicates
        ),
        performance=guard(
            ("target", "architecture", "eq", "sm_120"),
            ("workload", "nbf", "ge", 256),
            ("workload", "nbf", "le", 1024),
            ("workload", "occupied_fraction", "le", 0.25),
        ),
    )
    return {
        "workload": WorkloadSignature(
            "df_exchange",
            (("nbf", 384), ("occupied_fraction", 80 / 384), ("resident", True)),
        ),
        "target": TargetCapabilities(
            TargetInfo("cuda", "sm_120", 32, 1024, 16), (("fp64", True),)
        ),
        "identity": identity,
        "profiles": (tuned,),
        "fallback": fallback,
    }


def test_exact_reuse_and_detached_diagnostics(case):
    decision = select_specialization(**case)
    assert decision.status == "specialized"
    assert decision.selected is case["profiles"][0]
    assert decision.evaluations[0].eligible and decision.evaluations[0].promoted
    assert decision.selected.artifact_key != decision.selection_key
    payload = decision.to_payload()
    assert payload["artifact_key"] == decision.selected.artifact_key
    assert json.loads(json.dumps(payload))["status"] == "specialized"
    payload["evaluations"][0]["name"] = "mutated copy"
    assert decision.evaluations[0].name == "bounded-example"
    assert select_specialization(**case).selection_key == decision.selection_key


@pytest.mark.parametrize(
    "nbf,nocc", [(256, 64), (384, 80), (416, 87), (768, 160), (1024, 256)]
)
def test_nearby_workloads_reuse_one_guarded_artifact(case, nbf, nocc):
    workload = with_features(case["workload"], nbf=nbf, occupied_fraction=nocc / nbf)
    result = select_specialization(**(case | {"workload": workload}))
    assert result.selected is case["profiles"][0]
    if nbf != 384:
        assert result.selection_key != select_specialization(**case).selection_key


@pytest.mark.parametrize(
    "updates", [{"nbf": 255}, {"nbf": 1025}, {"occupied_fraction": 0.3}]
)
def test_outside_promotion_domain_uses_generic(case, updates):
    result = select_specialization(
        **(case | {"workload": with_features(case["workload"], **updates)})
    )
    assert result.status == "fallback"
    assert result.selected is case["fallback"]
    assert result.evaluations[0].eligible
    assert not result.evaluations[0].promoted
    assert result.evaluations[0].promotion_failures


def test_correctness_and_promotion_are_independent(case):
    result = select_specialization(
        **(case | {"workload": with_features(case["workload"], resident=False)})
    )
    assert result.status == "fallback"
    assert not result.evaluations[0].eligible
    assert result.evaluations[0].promoted
    assert "workload.resident" in result.evaluations[0].eligibility_failures[0]
    experimental = replace(case["profiles"][0], performance=None)
    result = select_specialization(**(case | {"profiles": (experimental,)}))
    assert result.status == "fallback"
    assert result.evaluations[0].eligible and not result.evaluations[0].promoted
    assert result.evaluations[0].promotion_failures == ("profile is not promoted",)


def test_unknown_architecture_falls_back_without_product_name_policy(case):
    target = replace(
        case["target"], target=replace(case["target"].target, architecture="unknown")
    )
    result = select_specialization(**(case | {"target": target}))
    assert result.status == "fallback"
    assert "target.architecture" in result.evaluations[0].promotion_failures[0]
    assert result.selection_key != select_specialization(**case).selection_key


@pytest.mark.parametrize("fp64", [False, 1, "true"])
def test_missing_correctness_capability_never_uses_even_generic(case, fp64):
    result = select_specialization(
        **(case | {"target": with_features(case["target"], fp64=fp64)})
    )
    assert result.status == "unsupported"
    assert result.selected is None
    assert not result.fallback.eligible
    assert any(
        "target.fp64" in reason for reason in result.fallback.eligibility_failures
    )


def test_absent_facts_are_not_zero_false_or_cuda_defaults(case):
    target = replace(case["target"], features=())
    result = select_specialization(**(case | {"target": target}))
    assert result.selected is None
    assert result.fallback.eligibility_failures == (
        "target.fp64: missing required fact",
    )
    workload = replace(case["workload"], features=(("resident", True),))
    result = select_specialization(**(case | {"workload": workload}))
    assert result.status == "fallback"
    assert all(
        "missing required fact" in reason
        for reason in result.evaluations[0].promotion_failures
    )
    target = replace(
        case["target"], target=replace(case["target"].target, subgroup_size=None)
    )
    tuned = replace(
        case["profiles"][0], performance=guard(("target", "subgroup_size", "ge", 1))
    )
    assert (
        select_specialization(
            **(case | {"target": target, "profiles": (tuned,)})
        ).status
        == "fallback"
    )


@pytest.mark.parametrize("field", ["scientific_hash", "compiler_hash"])
def test_changed_identity_invalidates_candidates_and_stale_fallback(case, field):
    identity = replace(case["identity"], **{field: canonical_hash("changed")})
    result = select_specialization(**(case | {"identity": identity}))
    assert result.selected is None
    assert f"identity.{field}: mismatch" in result.fallback.eligibility_failures
    fallback = replace(
        case["fallback"], identity=identity, artifact_key=canonical_hash("new-generic")
    )
    result = select_specialization(
        **(case | {"identity": identity, "fallback": fallback})
    )
    assert result.status == "fallback" and result.selected is fallback


@pytest.mark.parametrize("field", ["artifact_key", "schedule_hash", "profile_hash"])
def test_existing_artifact_schedule_profile_hashes_affect_reselection(case, field):
    original = select_specialization(**case)
    profile = replace(case["profiles"][0], **{field: canonical_hash("changed")})
    result = select_specialization(**(case | {"profiles": (profile,)}))
    assert result.selection_key != original.selection_key
    assert result.selected is profile
    assert result.selected.artifact_key == profile.artifact_key


def test_priority_and_target_resources_participate_in_selection_identity(case):
    first = case["profiles"][0]
    second = replace(first, name="other", artifact_key=canonical_hash("other-artifact"))
    a = select_specialization(**(case | {"profiles": (first, second)}))
    b = select_specialization(**(case | {"profiles": (second, first)}))
    assert a.selected is first and b.selected is second
    assert a.selection_key != b.selection_key
    target = replace(
        case["target"],
        target=replace(case["target"].target, maximum_workgroup_threads=512),
    )
    assert (
        select_specialization(**(case | {"target": target})).selection_key
        != select_specialization(**case).selection_key
    )
    assert select_specialization(**(case | {"profiles": ()})).status == "fallback"
    with pytest.raises(ValueError, match="unique"):
        select_specialization(**(case | {"profiles": (first, first)}))
    with pytest.raises(ValueError, match="unique"):
        select_specialization(**(case | {"profiles": (case["fallback"],)}))


def test_feature_and_conjunction_order_do_not_change_identity(case):
    original = select_specialization(**case)
    profile = case["profiles"][0]
    reordered = replace(
        profile,
        performance=SpecializationGuard(
            tuple(reversed(profile.performance.predicates))
        ),
    )
    workload = replace(
        case["workload"], features=tuple(reversed(case["workload"].features))
    )
    result = select_specialization(
        **(case | {"workload": workload, "profiles": (reordered,)})
    )
    assert result.selection_key == original.selection_key


def test_frozen_records_copy_nested_input_pairs(case):
    pairs = [["nbf", 416], ["resident", True]]
    workload = WorkloadSignature("df_exchange", pairs)
    pairs[0][1] = 999
    assert dict(workload.features)["nbf"] == 416
    with pytest.raises(FrozenInstanceError):
        workload.kind = "other"
    with pytest.raises(TypeError):
        workload.features[0][1] = 100
    # JSON type identity survives Python's True == 1 == 1.0 equality.
    keys = {
        select_specialization(
            **(case | {"workload": with_features(case["workload"], nbf=n)})
        ).selection_key
        for n in (True, 1, 1.0)
    }
    assert len(keys) == 3


@pytest.mark.parametrize("value", [float("nan"), float("inf"), [], {}, None])
def test_nonportable_or_mutable_feature_values_are_rejected(value):
    with pytest.raises(ValueError, match="scalars"):
        WorkloadSignature("example", (("value", value),))
    with pytest.raises(ValueError, match="scalars"):
        GuardPredicate("workload", "value", "eq", value)


@pytest.mark.parametrize(
    "features", [(("nbf", 1), ("nbf", 2)), (("kind", "other"),), (("", 1),), ("ab",)]
)
def test_duplicate_reserved_or_malformed_features_are_rejected(features):
    with pytest.raises(ValueError):
        WorkloadSignature("example", features)


@pytest.mark.parametrize("operator,value", [("exec", 1), ("ge", True), ("le", "100")])
def test_invalid_predicate_definitions_are_rejected(operator, value):
    with pytest.raises(ValueError):
        GuardPredicate("workload", "nbf", operator, value)


def test_target_facts_cannot_override_target_info(case):
    with pytest.raises(ValueError, match="reserved"):
        replace(case["target"], features=(("backend", "opencl"),))
    with pytest.raises(ValueError, match="scope"):
        GuardPredicate("unknown", "nbf", "eq", 1)
    with pytest.raises(ValueError, match="SHA-256"):
        replace(case["identity"], compiler_hash="unversioned")
    with pytest.raises(ValueError, match="guards"):
        replace(case["profiles"][0], performance=True)


def test_other_backends_and_unknown_subgroups_need_no_cuda_assumptions(case):
    target = TargetCapabilities(
        TargetInfo("opencl", "portable", None, 64, None), (("fp64", True),)
    )
    correctness = guard(
        ("target", "backend", "eq", "opencl"), ("target", "fp64", "eq", True)
    )
    fallback = replace(case["fallback"], correctness=correctness)
    profile = replace(
        case["profiles"][0], correctness=correctness, performance=SpecializationGuard()
    )
    result = select_specialization(
        **(case | {"target": target, "fallback": fallback, "profiles": (profile,)})
    )
    assert result.status == "specialized"


def test_existing_compiled_artifact_key_and_loader_remain_the_authority(case, tmp_path):
    from vibeqc_compiler.integral.artifact_cache import LocalArtifactCache
    from vibeqc_compiler.integral.runtime_backend import CompiledArtifactIdentity

    artifact = CompiledArtifactIdentity(
        "cuda",
        "test-device",
        "runtime",
        "compiler",
        "driver",
        "cuda-c++",
        (),
        case["identity"].scientific_hash,
        canonical_hash("source"),
        canonical_hash("schedule"),
    )
    profile = replace(
        case["profiles"][0],
        artifact_key=artifact.key,
        schedule_hash=artifact.schedule_hash,
    )
    cache = LocalArtifactCache(tmp_path / "existing-cache")
    cache.install(artifact, b"synthetic cache bytes, not GPU evidence")
    result = select_specialization(**(case | {"profiles": (profile,)}))
    assert result.selected.artifact_key == artifact.key
    assert cache.load(artifact) == b"synthetic cache bytes, not GPU evidence"
    with pytest.raises(ValueError, match="incompatible"):
        artifact.require_compatible(replace(artifact, device="another-device"))


def test_record_equality_preserves_scalar_types(case):
    assert len({WorkloadSignature("kind", (("x", x),)) for x in (True, 1, 1.0)}) == 3
    assert len({GuardPredicate("workload", "x", "eq", x) for x in (True, 1, 1.0)}) == 3
    assert len({with_features(case["target"], fp64=x) for x in (True, 1, 1.0)}) == 3
    assert WorkloadSignature("kind", (("x", 1),)) != "not-a-record"


@pytest.mark.parametrize("value", [float("inf"), float("nan"), [1]])
def test_direct_guard_evaluation_rejects_nonfinite_or_mutable_facts(value):
    predicate = GuardPredicate("target", "memory_bytes", "ge", 1)
    assert "invalid" in predicate.failure({"target": {"memory_bytes": value}})


def test_backend_and_workload_mismatches_never_trigger_an_implicit_fallback(case):
    workload = replace(case["workload"], kind="unsupported-consumer")
    assert select_specialization(**(case | {"workload": workload})).selected is None
    target = replace(
        case["target"], target=replace(case["target"].target, backend="cpu")
    )
    assert select_specialization(**(case | {"target": target})).selected is None
