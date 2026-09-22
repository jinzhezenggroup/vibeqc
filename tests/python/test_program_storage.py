"""ProgramIR physical alias/effect overlay for shared #831 storage analysis."""

import copy

import pytest
from vibeqc_compiler.common.program import PlanCall, ProgramBuffer, ProgramIR
from vibeqc_compiler.common.program_storage import (
    BufferAliasBinding,
    CallDonationBinding,
    CallEffectBinding,
    ProgramStoragePlan,
)
from vibeqc_compiler.common.storage import AliasKind, MemoryEffect


def example() -> ProgramIR:
    return ProgramIR(
        "storage-overlay",
        (
            ProgramBuffer("x", 64),
            ProgramBuffer("a", 128),
            ProgramBuffer("a_view", 128),
            ProgramBuffer("b", 80),
            ProgramBuffer("out", 8),
        ),
        ("x",),
        (
            PlanCall("make_a", "provider.a", "v1", ("x",), ("a",)),
            PlanCall("slice", "provider.view", "v1", ("a",), ("a_view",)),
            PlanCall("consume", "provider.b", "v1", ("a_view",), ("b",)),
            PlanCall("finish", "provider.out", "v1", ("b",), ("out",)),
        ),
        ("out",),
    )


def test_view_binding_shares_owner_capacity() -> None:
    program = example()
    baseline = program.storage_analysis()
    plan = ProgramStoragePlan(
        program,
        (BufferAliasBinding("a_view", AliasKind.VIEW, "a"),),
    )
    analysis = plan.storage_analysis()
    assert baseline.peak_by_space["pageable"] == 320
    assert analysis.peak_by_space["pageable"] == 272
    assert analysis.slot_for("a_view") is None
    owner = next(item for item in analysis.ranges if item.owner == "a")
    assert owner.members == ("a", "a_view")
    assert owner.first_phase == 1 and owner.last_phase == 3
    assert plan.diagnostics()["elided_alias_allocations"] == ["a_view"]


def test_program_donation_reuses_same_call_slot_and_reduces_peak() -> None:
    program = ProgramIR(
        "donation-overlay",
        (
            ProgramBuffer("x", 64),
            ProgramBuffer("a", 64),
            ProgramBuffer("b", 64),
            ProgramBuffer("out", 8),
        ),
        ("x",),
        (
            PlanCall("make_a", "provider.a", "v1", ("x",), ("a",)),
            PlanCall("update", "provider.update", "v1", ("a",), ("b",)),
            PlanCall("finish", "provider.out", "v1", ("b",), ("out",)),
        ),
        ("out",),
    )
    baseline = ProgramStoragePlan(program).storage_analysis()
    plan = ProgramStoragePlan(
        program,
        donations=(CallDonationBinding("update", "a", "b"),),
    )
    donated = plan.storage_analysis()

    assert donated.donations == ((2, "a", "b"),)
    assert donated.slot_for("a") == donated.slot_for("b")
    assert baseline.peak_by_space["pageable"] == 192
    assert donated.peak_by_space["pageable"] == 136
    assert ProgramStoragePlan.from_payload(program, plan.to_payload()) == plan
    assert plan.identity != ProgramStoragePlan(program).identity


def test_program_donation_fails_closed_on_opaque_or_invalid_edges() -> None:
    program = ProgramIR(
        "donation-fail-closed",
        (
            ProgramBuffer("x", 64),
            ProgramBuffer("a", 64),
            ProgramBuffer("b", 64),
        ),
        ("x",),
        (
            PlanCall("make_a", "provider.a", "v1", ("x",), ("a",)),
            PlanCall("update", "provider.update", "v1", ("a",), ("b",)),
        ),
        ("b",),
    )
    donation = CallDonationBinding("update", "a", "b")
    with pytest.raises(ValueError, match="explicit memory effects"):
        ProgramStoragePlan(
            program,
            effects=(CallEffectBinding("update", MemoryEffect.OPAQUE),),
            donations=(donation,),
        )
    with pytest.raises(ValueError, match="call read to a call write"):
        ProgramStoragePlan(
            program,
            donations=(CallDonationBinding("update", "x", "b"),),
        )
    with pytest.raises(ValueError, match="unknown ProgramIR call"):
        ProgramStoragePlan(
            program,
            donations=(CallDonationBinding("missing", "a", "b"),),
        )


def test_schema_v1_replay_remains_supported_without_donations() -> None:
    program = example()
    payload = ProgramStoragePlan(program).to_payload()
    payload["schema_version"] = 1
    payload.pop("donations")
    replayed = ProgramStoragePlan.from_payload(program, payload)

    assert replayed.schema_version == 1
    assert replayed.donations == ()
    assert replayed.to_payload() == payload


def test_opaque_effect_retains_touched_owner() -> None:
    program = example()
    aliases = (BufferAliasBinding("a_view", AliasKind.VIEW, "a"),)
    explicit = ProgramStoragePlan(program, aliases).storage_analysis()
    opaque = ProgramStoragePlan(
        program,
        aliases,
        (CallEffectBinding("consume", MemoryEffect.OPAQUE),),
    ).storage_analysis()
    owner = next(item for item in opaque.ranges if item.owner == "a")
    assert owner.reusable is False
    assert owner.retained_reason == "opaque_effect"
    assert owner.last_phase == len(program.calls) + 1
    assert opaque.peak_by_space["pageable"] > explicit.peak_by_space["pageable"]


def test_identity_and_replay_bind_program_aliases_and_effects() -> None:
    program = example()
    plan = ProgramStoragePlan(
        program,
        (BufferAliasBinding("a_view", AliasKind.VIEW, "a"),),
        (CallEffectBinding("consume", MemoryEffect.OPAQUE),),
    )
    payload = copy.deepcopy(plan.to_payload())
    assert ProgramStoragePlan.from_payload(program, payload) == plan
    assert ProgramStoragePlan.from_payload(program, payload).identity == plan.identity
    assert ProgramStoragePlan(program).identity != plan.identity

    other = ProgramIR(
        "other",
        program.buffers,
        program.inputs,
        program.calls,
        program.outputs,
    )
    with pytest.raises(ValueError, match="does not match"):
        ProgramStoragePlan.from_payload(other, payload)


@pytest.mark.parametrize(
    "binding,match",
    [
        (BufferAliasBinding("missing", AliasKind.VIEW, "a"), "unknown"),
        (BufferAliasBinding("a_view", AliasKind.VIEW, "missing"), "unknown"),
        (BufferAliasBinding("a_view", AliasKind.UNKNOWN), "borrowed"),
    ],
)
def test_invalid_alias_bindings_fail_closed(
    binding: BufferAliasBinding, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        ProgramStoragePlan(example(), (binding,))


def test_invalid_effect_and_duplicate_bindings_fail_closed() -> None:
    program = example()
    alias = BufferAliasBinding("a_view", AliasKind.VIEW, "a")
    with pytest.raises(ValueError, match="duplicate"):
        ProgramStoragePlan(program, (alias, alias))
    effect = CallEffectBinding("consume", MemoryEffect.OPAQUE)
    with pytest.raises(ValueError, match="duplicate"):
        ProgramStoragePlan(program, effects=(effect, effect))
    with pytest.raises(ValueError, match="unknown"):
        ProgramStoragePlan(
            program, effects=(CallEffectBinding("missing", MemoryEffect.OPAQUE),)
        )


def test_unknown_borrowed_alias_blocks_space_reuse() -> None:
    plan = ProgramStoragePlan(
        example(),
        (BufferAliasBinding("x", AliasKind.UNKNOWN),),
    )
    analysis = plan.storage_analysis()
    assert analysis.blocked_spaces == ("pageable",)
    assert all(item.reusable is False for item in analysis.ranges)
    assert {item.retained_reason for item in analysis.ranges} == {
        "unknown_alias",
        "unknown_alias_space",
    }


def test_untrusted_replay_rejects_unknown_fields_and_kinds() -> None:
    program = example()
    payload = ProgramStoragePlan(program).to_payload()
    payload["extra"] = True
    with pytest.raises(ValueError, match="fields"):
        ProgramStoragePlan.from_payload(program, payload)

    payload = ProgramStoragePlan(program).to_payload()
    payload["aliases"] = [{"buffer": "x", "alias": "invented", "alias_of": None}]
    with pytest.raises(ValueError, match="kind"):
        ProgramStoragePlan.from_payload(program, payload)


def test_legacy_positional_schema_argument_is_not_a_donation() -> None:
    program = example()
    plan = ProgramStoragePlan(program, (), (), 1)
    assert plan.schema_version == 1
    assert plan.donations == ()
    assert "donations" not in plan.to_payload()
    assert ProgramStoragePlan.from_payload(program, plan.to_payload()) == plan
