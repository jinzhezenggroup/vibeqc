"""Deterministic pass-pipeline execution and bisection contracts."""

import pytest
from vibeqc_compiler.common.pass_manager import PassManager, PassStage
from vibeqc_compiler.tensor import Program, constant, execute, multiply, optimize


def fingerprint(value: int) -> str:
    return str(value)


def test_pass_manager_runs_versioned_pipeline_and_records_changes() -> None:
    manager = PassManager(
        name="test.pipeline",
        version=2,
        stages=(
            PassStage("plus_one", 1, lambda value: value + 1, invalidates=("range",)),
            PassStage("identity", 3, lambda value: value),
            PassStage("double", 1, lambda value: value * 2, requires=("plus_one",)),
        ),
        fingerprint=fingerprint,
    )
    first = manager.run(4)
    second = manager.run(4)
    assert first == second
    assert first.value == 10
    assert len(first.pipeline_identity) == 64
    assert [record.name for record in first.records] == [
        "plus_one",
        "identity",
        "double",
    ]
    assert [record.changed for record in first.records] == [True, False, True]
    assert first.records[0].invalidated_analyses == ("range",)
    assert first.records[1].invalidated_analyses == ()


def test_pass_manager_supports_prefix_bisection_and_explicit_disables() -> None:
    manager = PassManager(
        name="test.bisect",
        version=1,
        stages=(
            PassStage("a", 1, lambda value: value + "a"),
            PassStage("b", 1, lambda value: value + "b"),
            PassStage("c", 1, lambda value: value + "c"),
        ),
        fingerprint=lambda value: value or "empty",
    )
    prefix = manager.run("", stop_after="b")
    skipped = manager.run("", disabled=("b",))
    assert prefix.value == "ab"
    assert prefix.stopped_after == "b"
    assert skipped.value == "ac"
    assert skipped.disabled == ("b",)
    assert prefix.pipeline_identity != skipped.pipeline_identity


def test_pass_manager_fails_closed_for_invalid_pipeline_or_bisection() -> None:
    with pytest.raises(ValueError, match="unique"):
        PassManager(
            name="duplicate",
            version=1,
            stages=(
                PassStage("same", 1, lambda value: value),
                PassStage("same", 1, lambda value: value),
            ),
            fingerprint=fingerprint,
        )
    with pytest.raises(ValueError, match="non-prior"):
        PassManager(
            name="dependency",
            version=1,
            stages=(PassStage("later", 1, lambda value: value, requires=("missing",)),),
            fingerprint=fingerprint,
        )

    manager = PassManager(
        name="required",
        version=1,
        stages=(
            PassStage("prepare", 1, lambda value: value + 1),
            PassStage("consume", 1, lambda value: value * 2, requires=("prepare",)),
        ),
        fingerprint=fingerprint,
    )
    with pytest.raises(ValueError, match="requires disabled"):
        manager.run(2, disabled=("prepare",))
    with pytest.raises(ValueError, match="unknown disabled"):
        manager.run(2, disabled=("absent",))


def test_tensor_optimizer_records_shared_pipeline_without_changing_equation() -> None:
    live = multiply(constant(2), constant(3))
    dead = multiply(constant(4), constant(5))
    program = Program({"value": live}, definitions=(dead,))
    optimized = optimize(program)
    provenance = optimized.provenance
    assert provenance["original_logical_hash"] == program.logical_hash
    assert (
        execute(optimized, {}).outputs["value"] == execute(program, {}).outputs["value"]
    )
    assert len(provenance["optimizer_identity"]) == 64
    assert [item["name"] for item in provenance["optimizer_passes"]] == [
        "dead_nodes",
        "identity_transposes",
        "view_canonicalization",
        "algebraic_canonicalization",
        "exact_cse",
        "scalar_constants",
        "post_fold_exact_cse",
        "post_fold_dead_nodes",
    ]
    assert provenance["rewrites"] == [
        "dead_nodes",
        "identity_transposes",
        "view_canonicalization",
        "algebraic_canonicalization",
        "exact_cse",
        "scalar_constants",
        "exact_cse",
        "dead_nodes",
    ]
