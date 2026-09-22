"""Strict container validation for the ProgramIR donation replay boundary."""

import pytest
from vibeqc_compiler.common.program import PlanCall, ProgramBuffer, ProgramIR
from vibeqc_compiler.common.program_storage import (
    CallDonationBinding,
    ProgramStoragePlan,
)


def _program() -> ProgramIR:
    return ProgramIR(
        "donation-payload",
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


@pytest.mark.parametrize("malformed", ({}, "", ()))
def test_donation_replay_rejects_non_list_containers(malformed: object) -> None:
    program = _program()
    payload = ProgramStoragePlan(program).to_payload()
    payload["donations"] = malformed
    with pytest.raises(TypeError, match="donations"):
        ProgramStoragePlan.from_payload(program, payload)


@pytest.mark.parametrize("donate", (False, True))
def test_donation_replay_preserves_valid_list_payload(donate: bool) -> None:
    program = _program()
    edges = (CallDonationBinding("update", "a", "b"),) if donate else ()
    plan = ProgramStoragePlan(program, donations=edges)
    replayed = ProgramStoragePlan.from_payload(program, plan.to_payload())
    assert replayed == plan
    assert replayed.to_payload() == plan.to_payload()
    analysis = replayed.storage_analysis()
    assert analysis.peak_by_space["pageable"] == (136 if donate else 192)
    if donate:
        assert analysis.slot_for("a") == analysis.slot_for("b")


def test_donation_replay_preserves_legacy_schema_without_field() -> None:
    program = _program()
    legacy = ProgramStoragePlan(program, (), (), 1)
    payload = legacy.to_payload()
    assert "donations" not in payload
    assert ProgramStoragePlan.from_payload(program, payload) == legacy
