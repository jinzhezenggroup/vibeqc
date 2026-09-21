"""Shared storage analysis for ProgramIR and TensorIR compiler consumers."""

import typing

import pytest
from vibeqc_compiler.common.storage import (
    AliasKind,
    BufferOp,
    BufferValue,
    MemoryEffect,
    analyze_storage,
)


def explicit(key: typing.Any, reads: typing.Any, writes: typing.Any) -> BufferOp:
    return BufferOp(key, reads, writes, MemoryEffect.EXPLICIT)


def test_alias_lifetime_interference_and_deterministic_reuse() -> None:
    analysis = analyze_storage(
        (
            BufferValue("x", 64, "device", compiler_owned=False),
            BufferValue("a", 64, "device"),
            BufferValue(
                "a_view",
                32,
                "device",
                alias=AliasKind.VIEW,
                alias_of="a",
                compiler_owned=False,
            ),
            BufferValue("token", 8, "device"),
            BufferValue("b", 64, "device"),
        ),
        (
            explicit("make_a", ("x",), ("a",)),
            explicit("view_a", ("a",), ("a_view",)),
            explicit("consume_view", ("a_view",), ("token",)),
            explicit("make_b", ("token",), ("b",)),
        ),
        inputs=("x",),
        outputs=("b",),
    )

    ranges = {item.owner: item for item in analysis.ranges}
    assert ranges["a"].members == ("a", "a_view")
    assert (ranges["a"].first_phase, ranges["a"].last_phase) == (1, 3)
    assert ("a", "b") not in analysis.interference
    assert analysis.slot_for("a") == analysis.slot_for("b")
    assert analysis.peak_by_space["device"] == 136


def test_unknown_alias_space_and_opaque_effect_fail_closed() -> None:
    unknown = analyze_storage(
        (
            BufferValue(
                "external",
                1,
                "device",
                alias=AliasKind.UNKNOWN,
                compiler_owned=False,
            ),
            BufferValue("a", 64, "device"),
            BufferValue("token", 8, "device"),
            BufferValue("b", 64, "device"),
        ),
        (
            explicit("make_a", ("external",), ("a",)),
            explicit("consume_a", ("a",), ("token",)),
            explicit("make_b", ("token",), ("b",)),
        ),
        inputs=("external",),
        outputs=("b",),
    )
    ranges = {item.owner: item for item in unknown.ranges}
    assert unknown.blocked_spaces == ("device",)
    assert ranges["a"].retained_reason == "unknown_alias_space"
    assert ranges["a"].last_phase == 4
    assert unknown.slot_for("a") != unknown.slot_for("b")

    opaque = analyze_storage(
        (
            BufferValue("x", 8, "pageable", compiler_owned=False),
            BufferValue("a", 32, "pageable"),
            BufferValue("b", 32, "pageable"),
        ),
        (
            explicit("make_a", ("x",), ("a",)),
            BufferOp("opaque", ("a",), ("b",)),
        ),
        inputs=("x",),
        outputs=("b",),
    )
    opaque_ranges = {item.owner: item for item in opaque.ranges}
    assert opaque_ranges["a"].retained_reason == "opaque_effect"
    assert opaque_ranges["a"].last_phase == 3
    assert opaque.slot_for("a") != opaque.slot_for("b")


def test_alias_contract_rejects_undeclared_or_oversized_views() -> None:
    with pytest.raises(ValueError, match="undeclared"):
        analyze_storage(
            (
                BufferValue(
                    "view",
                    8,
                    "device",
                    alias=AliasKind.VIEW,
                    alias_of="missing",
                    compiler_owned=False,
                ),
            ),
            (),
            inputs=("view",),
            outputs=("view",),
        )

    with pytest.raises(ValueError, match="exceeds"):
        analyze_storage(
            (
                BufferValue("owner", 8, "device", compiler_owned=False),
                BufferValue(
                    "view",
                    16,
                    "device",
                    alias=AliasKind.VIEW,
                    alias_of="owner",
                    compiler_owned=False,
                ),
            ),
            (explicit("view", ("owner",), ("view",)),),
            inputs=("owner",),
            outputs=("view",),
        )
