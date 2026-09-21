"""Serial provider-boundary liveness, strict replay and conservative ownership."""

import copy
import json
import typing
from dataclasses import FrozenInstanceError, replace

import pytest
from vibeqc_compiler.common.layout import DenseLayout
from vibeqc_compiler.common.program import PlanCall, ProgramBuffer, ProgramIR
from vibeqc_compiler.common.resources import MAX_BYTES, ResourceBudget, plan_resources


def example() -> typing.Any:
    return ProgramIR(
        "test",
        tuple(
            ProgramBuffer(k, v)
            for k, v in (("x", 100), ("a", 50), ("b", 30), ("c", 80), ("out", 8))
        ),
        ("x",),
        (
            PlanCall("a", "provider.A", "v1", ("x",), ("a",)),
            PlanCall("b", "provider.B", "v1", ("a",), ("b",)),
            PlanCall("c", "provider.C", "v1", ("x",), ("c",)),
            PlanCall("finish", "provider.D", "v1", ("b", "c"), ("out",)),
        ),
        ("out",),
    )


def test_last_use_is_after_call_not_before_its_output_allocation() -> None:
    p = example()
    storage = p.storage_analysis()
    assert storage.peak_by_space["pageable"] == 218
    assert storage.slot_for("a") == storage.slot_for("out")
    assert ("a", "b") in storage.interference
    assert ("a", "c") not in storage.interference
    intervals = {e.name: (e.first_phase, e.last_phase, e.kind) for e in p.lifetimes()}
    assert intervals == {
        "x": (0, 5, "persistent"),
        "a": (1, 2, "workspace"),
        "b": (2, 4, "workspace"),
        "c": (3, 4, "workspace"),
        "out": (4, 5, "output"),
    }
    assert p.release_after("a") == ()
    assert p.release_after("b") == ("a",)
    assert p.release_after("finish") == ("b", "c")
    optimized = plan_resources((p.resource_request(),), ResourceBudget())
    retained = plan_resources(
        (p.resource_request(retain_temporaries=True),), ResourceBudget()
    )
    assert optimized.peak_bytes["host"] == 218
    assert retained.peak_bytes["host"] == 268
    assert optimized.identity != retained.identity
    assert optimized.requests[0].scope_exclusions
    with pytest.raises(MemoryError):
        plan_resources(
            (p.resource_request(),), ResourceBudget(host_bytes=217)
        ).require_feasible()
    with pytest.raises(ValueError, match="unknown"):
        p.release_after("missing")
    with pytest.raises(ValueError, match="bool"):
        p.lifetimes(retain_temporaries=1)


def test_dense_layout_is_part_of_boundary_identity_and_strict_replay() -> None:
    layout = DenseLayout((2, 3), order=(1, 0), alignment=8)
    base = example()
    buffer = ProgramBuffer("x", 48, layout=layout, itemsize=8)
    p = replace(base, buffers=(buffer, *base.buffers[1:]))
    replayed = ProgramIR.from_payload(p.to_payload())
    assert replayed == p
    assert replayed.buffers[0].layout == layout
    assert replayed.buffers[0].layout.element_strides == (1, 2)
    assert p.identity != base.identity
    with pytest.raises(ValueError, match="capacity"):
        ProgramBuffer("matrix", 47, layout=layout, itemsize=8)
    with pytest.raises(ValueError, match="itemsize"):
        ProgramBuffer("matrix", 48, layout=layout)
    with pytest.raises(ValueError, match="itemsize"):
        ProgramBuffer("matrix", 48, itemsize=8)
    with pytest.raises(TypeError, match="DenseLayout"):
        ProgramBuffer("matrix", 48, layout="C", itemsize=8)


def test_serialization_is_detached_strict_and_deterministic() -> None:
    p = example()
    payload = json.loads(json.dumps(p.to_payload()))
    assert ProgramIR.from_payload(payload) == p
    assert ProgramIR.from_payload(payload).identity == p.identity
    payload["buffers"][0]["bytes"] = 101
    assert p.buffers[0].bytes == 100
    assert ProgramIR.from_payload(payload).identity != p.identity
    with pytest.raises(FrozenInstanceError):
        p.name = "other"
    with pytest.raises(FrozenInstanceError):
        p.calls[0].identity = "other"
    calls = list(p.calls)
    q = replace(
        p,
        calls=calls,
        inputs=list(p.inputs),
        outputs=list(p.outputs),
        buffers=list(p.buffers),
    )
    calls.clear()
    assert q == p
    assert (
        replace(p, calls=(replace(p.calls[0], identity="new"), *p.calls[1:])).identity
        != p.identity
    )
    assert (
        replace(
            p, buffers=(replace(p.buffers[0], space="pinned"), *p.buffers[1:])
        ).identity
        != p.identity
    )


def test_dead_outputs_are_released_without_pruning_opaque_calls() -> None:
    p = replace(example(), outputs=("b",))
    assert len(p.calls) == 4
    assert p.release_after("finish") == ("c", "out")
    assert next(e for e in p.lifetimes() if e.name == "b").last_phase == 5


def test_borrowed_output_is_not_donated() -> None:
    p = replace(example(), outputs=("x",))
    e = next(e for e in p.lifetimes() if e.name == "x")
    assert e.first_phase == 0 and e.last_phase == 5 and e.kind == "persistent"
    assert "x" not in sum((p.release_after(c.name) for c in p.calls), ())


def test_repeated_reads_are_not_duplicate_owners() -> None:
    p = example()
    p = replace(p, calls=(replace(p.calls[0], reads=("x", "x")), *p.calls[1:]))
    assert p.release_after("b") == ("a",)


@pytest.mark.parametrize("bytes", [-1, True, 1.2, MAX_BYTES + 1])
def test_invalid_buffer_sizes(bytes: typing.Any) -> None:
    with pytest.raises(ValueError):
        ProgramBuffer("x", bytes)


@pytest.mark.parametrize("space", ["", "cuda", "device:-1", "device:01", None])
def test_invalid_memory_spaces(space: typing.Any) -> None:
    with pytest.raises(ValueError):
        ProgramBuffer("x", 1, space)


def test_byte_overflow_and_device_space_use_existing_planner() -> None:
    p = ProgramIR(
        "overflow",
        (ProgramBuffer("x", MAX_BYTES, "device:0"), ProgramBuffer("y", 1, "device:0")),
        ("x",),
        (PlanCall("p", "provider", "v1", ("x",), ("y",)),),
        ("y",),
    )
    with pytest.raises(ValueError, match="bytes"):
        plan_resources((p.resource_request(),), ResourceBudget())
    p = replace(p, buffers=(ProgramBuffer("x", 4, "device:0"), p.buffers[1]))
    assert (
        plan_resources((p.resource_request(),), ResourceBudget()).peak_bytes["device:0"]
        == 5
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", ""),
        ("schema_version", 3),
        ("schema_version", True),
        ("inputs", "x"),
        ("inputs", ("x", "x")),
        ("inputs", ("missing",)),
        ("outputs", ()),
        ("outputs", ("missing",)),
        ("outputs", ("out", "out")),
        ("calls", ()),
        ("buffers", ()),
        ("calls", ("wrong",)),
        ("buffers", ("wrong",)),
    ],
)
def test_invalid_program_contract(field: typing.Any, value: typing.Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(example(), **{field: value})


def test_duplicate_and_missing_owners_fail() -> None:
    p = example()
    for buffers in (
        (*p.buffers, p.buffers[0]),
        (*p.buffers, ProgramBuffer("orphan", 3)),
    ):
        with pytest.raises(ValueError):
            replace(p, buffers=buffers)
    with pytest.raises(ValueError, match="duplicate calls"):
        replace(p, calls=(*p.calls, p.calls[0]))
    with pytest.raises(ValueError, match="in-place"):
        replace(p, calls=(replace(p.calls[0], writes=("x",)), *p.calls[1:]))
    with pytest.raises(ValueError, match="undeclared"):
        replace(p, calls=(replace(p.calls[0], writes=("missing",)), *p.calls[1:]))
    with pytest.raises(ValueError, match="cyclic"):
        replace(p, calls=(replace(p.calls[0], reads=("c",)), *p.calls[1:]))
    with pytest.raises(ValueError, match="dependency"):
        replace(p, calls=(replace(p.calls[0], reads=("missing",)), *p.calls[1:]))


@pytest.mark.parametrize(
    "changes",
    [
        {"name": ""},
        {"provider": ""},
        {"identity": ""},
        {"reads": "x"},
        {"reads": (None,)},
        {"writes": ()},
        {"writes": ("a", "a")},
    ],
)
def test_bad_calls(changes: typing.Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(example().calls[0], **changes)


@pytest.mark.parametrize(
    "location,field,value",
    [
        ("program", "extra", 1),
        ("program", "buffers", "not a list"),
        ("program", "calls", None),
        ("buffer", "alias_of", "x"),
        ("call", "async", True),
        ("call", "identity", ""),
        ("call", "reads", ["out"]),
        ("program", "schema_version", 0),
    ],
)
def test_untrusted_replay_is_fail_closed(
    location: typing.Any, field: typing.Any, value: typing.Any
) -> None:
    payload = copy.deepcopy(example().to_payload())
    target = {
        "program": payload,
        "buffer": payload["buffers"][0],
        "call": payload["calls"][0],
    }[location]
    target[field] = value
    with pytest.raises((TypeError, ValueError)):
        ProgramIR.from_payload(payload)


def test_replay_rejects_missing_fields_and_non_objects() -> None:
    with pytest.raises(ValueError):
        ProgramIR.from_payload([])
    payload = example().to_payload()
    del payload["calls"][0]["reads"]
    with pytest.raises(ValueError):
        ProgramIR.from_payload(payload)
