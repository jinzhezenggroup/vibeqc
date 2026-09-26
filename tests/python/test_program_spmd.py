"""Logical ProgramIR SPMD lowering, collective semantics and accounting."""

import copy
import math
from collections.abc import Callable

import pytest
from vibeqc_compiler.common.layout import DenseLayout
from vibeqc_compiler.common.program import PlanCall, ProgramBuffer, ProgramIR
from vibeqc_compiler.common.resources import ResourceBudget, plan_resources
from vibeqc_compiler.common.spmd import (
    BufferPlacement,
    CollectiveSpec,
    DeviceMesh,
    SpmdPlan,
    partition_bounds,
    reference_collective,
)


def _dense(name: str, shape: tuple[int, ...]) -> ProgramBuffer:
    return ProgramBuffer(
        name,
        8 * math.prod(shape),
        layout=DenseLayout(tuple(shape)),
        itemsize=8,
    )


def example() -> ProgramIR:
    return ProgramIR(
        "spmd_test",
        (_dense("x", (10, 4)), _dense("tmp", (10, 4)), _dense("out", (4,))),
        ("x",),
        (
            PlanCall("map", "provider.map", "v1", ("x",), ("tmp",)),
            PlanCall("reduce", "provider.reduce", "v1", ("tmp",), ("out",)),
        ),
        ("out",),
    )


def placements() -> tuple[BufferPlacement, ...]:
    return (
        BufferPlacement("x", "sharded", "data", 0),
        BufferPlacement("tmp", "sharded", "data", 0),
        BufferPlacement("out", "replicated"),
    )


def collective() -> CollectiveSpec:
    return CollectiveSpec(
        "sum_out", "all_reduce", "out", "data", "reduce", reduction="sum"
    )


def test_same_program_lowers_to_single_and_multi_device() -> None:
    program = example()
    single = SpmdPlan.lower(
        program, DeviceMesh((("data", 1),)), placements(), (collective(),)
    )
    multi = SpmdPlan.lower(
        program, DeviceMesh((("data", 4),)), placements(), (collective(),)
    )

    assert single.program_identity == multi.program_identity == program.identity
    assert single.collectives == ()
    assert single.required_collectives == ()
    assert single.local_shape("x", 0) == (10, 4)
    assert single.communication_resource_request() is None

    assert multi.required_collectives == ("all_reduce",)
    assert [multi.local_shape("x", rank) for rank in range(4)] == [
        (3, 4),
        (3, 4),
        (2, 4),
        (2, 4),
    ]
    assert [multi.local_bytes("x", rank) for rank in range(4)] == [
        96,
        96,
        64,
        64,
    ]
    assert single.identity != multi.identity
    assert len({multi.shard_identity(rank) for rank in range(4)}) == 4


def test_mesh_coordinates_and_balanced_partition_are_deterministic() -> None:
    mesh = DeviceMesh((("data", 2), ("model", 3)))
    assert mesh.size == 6
    assert mesh.coordinates(4) == (1, 1)
    assert mesh.axis_coordinate(4, "data") == 1
    assert mesh.axis_coordinate(4, "model") == 1
    assert [partition_bounds(10, 4, rank) for rank in range(4)] == [
        (0, 3),
        (3, 6),
        (6, 8),
        (8, 10),
    ]


def test_spmd_replay_identity_and_provenance_are_strict_and_detached() -> None:
    program = example()
    plan = SpmdPlan.lower(
        program, DeviceMesh((("data", 4),)), placements(), (collective(),)
    )
    payload = copy.deepcopy(plan.to_payload())
    replayed = SpmdPlan.from_payload(program, payload)
    assert replayed == plan
    assert replayed.identity == plan.identity
    provenance = plan.provenance()
    provenance["mesh"]["axes"][0][1] = 99
    assert plan.mesh.size == 4
    assert plan.provenance()["spmd_identity"] == plan.identity

    payload["mesh"]["axes"][0][1] = 2
    assert SpmdPlan.from_payload(program, payload).identity != plan.identity
    mismatched_program = ProgramIR(
        "other",
        program.buffers,
        program.inputs,
        program.calls,
        program.outputs,
    )
    with pytest.raises(ValueError, match="does not match"):
        SpmdPlan.from_payload(mismatched_program, plan.to_payload())
    bad = copy.deepcopy(plan.to_payload())
    bad["extra"] = True
    with pytest.raises(ValueError, match="fields"):
        SpmdPlan.from_payload(program, bad)


def test_collective_reference_semantics_and_capability_gate() -> None:
    assert reference_collective("all_reduce", ((1, 2), (3, 4))) == (
        (4, 6),
        (4, 6),
    )
    assert reference_collective(
        "reduce_scatter",
        ((1, 2, 3, 4), (10, 20, 30, 40)),
    ) == ((11, 22), (33, 44))
    assert reference_collective("all_gather", ((1, 2), (3,), (4, 5))) == (
        (1, 2, 3, 4, 5),
        (1, 2, 3, 4, 5),
        (1, 2, 3, 4, 5),
    )
    with pytest.raises(ValueError, match="equal local extents"):
        reference_collective("all_reduce", ((1,), (2, 3)))

    plan = SpmdPlan.lower(
        example(), DeviceMesh((("data", 4),)), placements(), (collective(),)
    )
    plan.require_collective_support(("all_reduce",))
    with pytest.raises(NotImplementedError, match="all_reduce"):
        plan.require_collective_support(())
    single = SpmdPlan.lower(
        example(), DeviceMesh((("data", 1),)), placements(), (collective(),)
    )
    single.require_collective_support(())


def test_communication_resources_and_profitability_are_explicit() -> None:
    plan = SpmdPlan.lower(
        example(), DeviceMesh((("data", 4),)), placements(), (collective(),)
    )
    costs = plan.collective_costs()
    assert len(costs) == 1
    assert costs[0].participants == 4
    assert costs[0].payload_bytes == 32
    assert costs[0].logical_work_bytes == 256

    request = plan.communication_resource_request()
    assert request is not None
    candidate = request.candidates[0]
    assert candidate.relative_cost == costs[0].logical_work_bytes
    assert len(candidate.estimates) == 4
    assert {estimate.bytes for estimate in candidate.estimates} == {32}
    assert {estimate.streamed_bytes for estimate in candidate.estimates} == {64}
    resource_plan = plan_resources((request,), ResourceBudget())
    assert resource_plan.peak_bytes["device"] == 128
    for rank in range(4):
        assert resource_plan.peak_bytes[f"device:{rank}"] == 32


def test_scatter_gather_representation_transitions_are_accounted() -> None:
    program = example()
    mesh = DeviceMesh((("data", 4),))
    gather = SpmdPlan.lower(
        program,
        mesh,
        placements(),
        (
            CollectiveSpec(
                "gather_out",
                "all_gather",
                "out",
                "data",
                "reduce",
                tensor_axis=0,
            ),
        ),
    )
    gather_cost = gather.collective_costs()[0]
    assert gather_cost.logical_work_bytes == 160
    gather_request = gather.communication_resource_request()
    assert gather_request is not None
    assert {row.bytes for row in gather_request.candidates[0].estimates} == {32}
    assert {row.streamed_bytes for row in gather_request.candidates[0].estimates} == {
        40
    }

    scattered = (*placements()[:-1], BufferPlacement("out", "sharded", "data", 0))
    scatter = SpmdPlan.lower(
        program,
        mesh,
        scattered,
        (
            CollectiveSpec(
                "scatter_out",
                "reduce_scatter",
                "out",
                "data",
                "reduce",
                reduction="sum",
                tensor_axis=0,
            ),
        ),
    )
    assert scatter.collective_costs()[0].logical_work_bytes == 160
    assert [scatter.local_bytes("out", rank) for rank in range(4)] == [8] * 4


def test_partial_shard_completion_cannot_publish() -> None:
    plan = SpmdPlan.lower(
        example(), DeviceMesh((("data", 4),)), placements(), (collective(),)
    )
    plan.require_complete_shards(range(4))
    with pytest.raises(RuntimeError, match="partial"):
        plan.require_complete_shards((0, 1, 2))
    with pytest.raises(RuntimeError, match="extra"):
        plan.require_complete_shards((0, 1, 2, 3, 4))
    with pytest.raises(ValueError, match="duplicate"):
        plan.require_complete_shards((0, 1, 1, 3))


def test_invalid_spmd_contracts_fail_closed() -> None:
    program = example()
    mesh = DeviceMesh((("data", 4),))
    with pytest.raises(ValueError, match="cover every"):
        SpmdPlan.lower(program, mesh, placements()[:-1], (collective(),))
    with pytest.raises(ValueError, match="duplicate"):
        SpmdPlan.lower(program, mesh, (*placements(), placements()[0]), (collective(),))
    with pytest.raises(ValueError, match="mesh axis"):
        SpmdPlan.lower(
            program,
            mesh,
            (
                BufferPlacement("x", "sharded", "missing", 0),
                *placements()[1:],
            ),
            (collective(),),
        )
    with pytest.raises(ValueError, match="tensor axis"):
        SpmdPlan.lower(
            program,
            mesh,
            (
                BufferPlacement("x", "sharded", "data", 9),
                *placements()[1:],
            ),
            (collective(),),
        )
    with pytest.raises(ValueError, match="unavailable"):
        SpmdPlan.lower(
            program,
            mesh,
            placements(),
            (
                CollectiveSpec(
                    "too_early",
                    "all_reduce",
                    "out",
                    "data",
                    "map",
                    reduction="sum",
                ),
            ),
        )

    opaque = ProgramIR(
        "opaque",
        (ProgramBuffer("x", 8), ProgramBuffer("out", 8)),
        ("x",),
        (PlanCall("copy", "provider", "v1", ("x",), ("out",)),),
        ("out",),
    )
    with pytest.raises(ValueError, match="dense layout"):
        SpmdPlan.lower(
            opaque,
            mesh,
            (
                BufferPlacement("x", "sharded", "data", 0),
                BufferPlacement("out", "replicated"),
            ),
        )


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: DeviceMesh(()),
        lambda: DeviceMesh((("data", 0),)),
        lambda: DeviceMesh((("data", 2), ("data", 2))),
        lambda: BufferPlacement("x", "replicated", "data", 0),
        lambda: BufferPlacement("x", "sharded", None, 0),
        lambda: CollectiveSpec("c", "all_reduce", "x", "data", "map"),
        lambda: CollectiveSpec("c", "all_gather", "x", "data", "map", "sum"),
    ],
)
def test_invalid_component_contracts(constructor: Callable[[], object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        constructor()
