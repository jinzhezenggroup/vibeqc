"""A physical owner must satisfy the base alignment of every declared view."""

import pytest
from vibeqc_compiler.common.layout import DenseLayout
from vibeqc_compiler.common.storage import (
    AliasKind,
    BufferOp,
    BufferValue,
    MemoryEffect,
    analyze_storage,
)


@pytest.mark.parametrize("nested", (False, True))
def test_view_alignment_is_carried_into_its_physical_slot(nested: bool) -> None:
    values = [BufferValue("owner", 64, "device", DenseLayout((8,), alignment=8))]
    ops = []
    parent = "owner"
    if nested:
        values.append(
            BufferValue(
                "middle",
                64,
                "device",
                DenseLayout((8,), alignment=32),
                alias=AliasKind.VIEW,
                alias_of=parent,
                compiler_owned=False,
            )
        )
        ops.append(
            BufferOp("middle_view", (parent,), ("middle",), MemoryEffect.EXPLICIT)
        )
        parent = "middle"
    values.append(
        BufferValue(
            "view",
            32,
            "device",
            DenseLayout((4,), alignment=128),
            alias=AliasKind.VIEW,
            alias_of=parent,
            compiler_owned=False,
        )
    )
    ops.append(BufferOp("last_view", (parent,), ("view",), MemoryEffect.EXPLICIT))
    result = analyze_storage(values, ops, inputs=("owner",), outputs=("view",))
    assert len(result.slots) == 1
    assert result.slots[0].alignment == 128
    assert result.slots[0].bytes == 64
    assert result.slot_for("owner") == 0
    assert result.slot_for("view") is None
