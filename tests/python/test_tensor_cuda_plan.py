"""CPU proofs of storage invariants and deliberately infeasible CUDA plans."""

import typing
from dataclasses import replace

import pytest
from vibeqc_compiler.integral.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    broadcast,
    einsum,
    gather,
    input_tensor,
    multiply,
    reduce_sum,
    runtime_indexed_select,
    transpose,
)
from vibeqc_compiler.tensor.cuda_plan import (
    ALIGNMENT,
    Reservations,
    TensorSchedule,
    aligned,
    plan_cuda,
)

TARGET = cuda_target_info("sm_80")


def vector(size: typing.Any = 65, dtype: typing.Any = "float64") -> typing.Any:
    index = Index("i", IndexSpace("axis", "batch", size))
    return input_tensor("x", TensorSpec((index,), dtype=dtype, role="input"))


def assert_disjoint_live_allocations(plan: typing.Any) -> None:
    """Independently intersect lifetimes and byte ranges for every buffer pair."""
    for i, a in enumerate(plan.steps):
        if a.virtual or not a.node.spec.size:
            continue
        assert a.offset % ALIGNMENT == 0
        assert a.offset + aligned(a.node.spec.size * 8) <= plan.arena_bytes
        for j, b in enumerate(plan.steps[i + 1 :], i + 1):
            if b.virtual or not b.node.spec.size or a.last_use < j:
                continue
            assert (
                a.offset + aligned(a.node.spec.size * 8) <= b.offset
                or b.offset + aligned(b.node.spec.size * 8) <= a.offset
            ), (i, j)


def test_reuse_keeps_inputs_and_outputs_and_releases_dead_work() -> None:
    x = vector()
    chain = [x]
    for _ in range(12):
        chain.append(add(chain[-1], x))
    plan = plan_cuda(Program({"result": chain[-1], "retained": chain[2]}), TARGET)
    assert_disjoint_live_allocations(plan)
    assert plan.arena_bytes < sum(aligned(s.node.spec.size * 8) for s in plan.steps)
    assert len({s.offset for s in plan.steps}) < len(plan.steps)
    storage = plan.storage_analysis()
    assert storage.peak_by_space["device"] <= plan.arena_bytes
    assert any(len(slot.owners) > 1 for slot in storage.slots)
    for _, i in plan.outputs:
        assert plan.steps[i].last_use == len(plan.steps)
    assert plan.peak_bytes == plan.device_bytes + plan.host_bytes


def test_opt_in_inplace_donation_reuses_final_elementwise_owner() -> None:
    x = vector()
    transient = add(x, x)
    result = multiply(transient, x)
    program = Program({"result": result})

    baseline = plan_cuda(program, TARGET)
    donated = plan_cuda(
        program,
        TARGET,
        schedule=TensorSchedule(inplace_donation=True),
    )
    transient_index = next(
        i for i, step in enumerate(donated.steps) if step.node is transient
    )
    result_index = next(
        i for i, step in enumerate(donated.steps) if step.node is result
    )
    assert donated.steps[result_index].donated_from == transient_index
    assert donated.steps[result_index].offset == donated.steps[transient_index].offset
    assert donated.arena_bytes == baseline.arena_bytes - aligned(result.spec.size * 8)
    storage = donated.storage_analysis()
    assert storage.donations == ((result_index, transient_index, result_index),)
    assert storage.slot_for(transient_index) == storage.slot_for(result_index)
    assert storage.peak_by_space["device"] == donated.arena_bytes


def test_inplace_donation_fails_closed_with_layout_optimization() -> None:
    x = vector()
    with pytest.raises(ValueError, match="not yet qualified"):
        plan_cuda(
            Program({"result": add(x, x)}),
            TARGET,
            schedule=TensorSchedule(inplace_donation=True, layouts=True),
        )


def test_alias_lifetime_follows_materialized_ancestors() -> None:
    x = vector()
    matrix = broadcast(
        x, (x.spec.indices[0], Index("j", x.spec.indices[0].space)), (0,)
    )
    view = transpose(matrix, (1, 0))
    result = add(matrix, view)
    plan = plan_cuda(
        Program({"result": result}), TARGET, schedule=TensorSchedule(views=True)
    )
    assert sum(s.virtual for s in plan.steps) == 2
    assert_disjoint_live_allocations(plan)
    assert plan.steps[0].last_use == len(plan.steps)

    i = Index("row", IndexSpace("row", "batch", 7))
    j = Index("col", IndexSpace("col", "batch", 7))
    owner = input_tensor("matrix", TensorSpec((i, j), role="input"))
    alias = transpose(owner, (1, 0))
    alias_plan = plan_cuda(
        Program({"result": add(alias, alias)}),
        TARGET,
        schedule=TensorSchedule(views=True),
    )
    owner_index = next(
        k for k, step in enumerate(alias_plan.steps) if step.node is owner
    )
    alias_index = next(
        k for k, step in enumerate(alias_plan.steps) if step.node is alias
    )
    storage = alias_plan.storage_analysis()
    owner_range = next(item for item in storage.ranges if item.owner == owner_index)
    assert alias_index in owner_range.members
    assert alias_index not in dict(storage.assignments)


def test_fusion_preserves_checks_before_subsets() -> None:
    x = vector()
    square = multiply(x, x)
    zero_axis = Index("empty", IndexSpace("empty", "batch", 0))
    empty = broadcast(square, (x.spec.indices[0], zero_axis), (0,))
    plan = plan_cuda(
        Program({"empty": empty}),
        TARGET,
        schedule=TensorSchedule(views=True, fuse=True),
    )
    assert not next(s for s in plan.steps if s.node is square).virtual


def test_streaming_reduction_virtualizes_complete_runtime_domain() -> None:
    source_axis = Index("source", IndexSpace("stream_source", "batch", 137))
    inner = Index("inner", IndexSpace("stream_inner", "batch", 127))
    domain = Index("q", IndexSpace("stream_domain", "batch", 129))
    source = input_tensor(
        "stream_source",
        TensorSpec((source_axis, inner), role="input"),
    )
    coordinates = input_tensor(
        "stream_coordinates",
        TensorSpec((domain,), dtype="int64", role="input"),
    )
    selected = runtime_indexed_select(source, ((0, coordinates),), domain)
    squared = multiply(selected, selected)
    lane = reduce_sum(squared, (1,))
    total = reduce_sum(lane, (0,))
    program = Program({"total": total})

    baseline = plan_cuda(program, TARGET)
    streamed = plan_cuda(
        program,
        TARGET,
        schedule=TensorSchedule(stream_reductions=True),
    )

    assert streamed.identity != baseline.identity
    assert streamed.arena_bytes < baseline.arena_bytes
    for node in (selected, squared):
        assert next(step for step in streamed.steps if step.node is node).virtual
    # Keep the q-only reduction frontier materialized so CUDA launches one
    # parallel lane kernel before the final scalar reduction.
    assert not next(step for step in streamed.steps if step.node is lane).virtual


def test_streaming_reduction_stops_before_partial_source_consumption() -> None:
    x = vector(16)
    squared = multiply(x, x)
    subset = gather(squared, 0, (0, 3, 7))
    total = reduce_sum(subset, (0,))
    plan = plan_cuda(
        Program({"total": total}),
        TARGET,
        schedule=TensorSchedule(stream_reductions=True),
    )

    # The q-only frontier remains materialized, and the partial gather blocks
    # streaming of its source so arithmetic/bounds diagnostics are still
    # evaluated over the complete original domain.
    assert not next(step for step in plan.steps if step.node is subset).virtual
    assert not next(step for step in plan.steps if step.node is squared).virtual


def test_streaming_reduction_stops_when_einsum_sibling_domain_is_empty() -> None:
    q = Index("q_zero", IndexSpace("stream_q_zero", "batch", 8))
    k = Index("k_zero", IndexSpace("stream_k_zero", "batch", 0))
    x = input_tensor("stream_x_zero", TensorSpec((q,), role="input"))
    empty = input_tensor("stream_empty", TensorSpec((k,), role="input"))
    squared = multiply(x, x)
    contraction = einsum("q,k->q", squared, empty)
    total = reduce_sum(contraction, (0,))
    plan = plan_cuda(
        Program({"total": total}),
        TARGET,
        schedule=TensorSchedule(stream_reductions=True),
    )

    assert not next(step for step in plan.steps if step.node is squared).virtual


def test_constrained_plan_shrinks_panels_and_rejects_below_indivisible_minimum() -> (
    None
):
    i = Index("i", IndexSpace("rows", "batch", 17))
    j = Index("j", IndexSpace("cols", "batch", 19))
    k = Index("k", IndexSpace("inner", "batch", 23))
    a = input_tensor("a", TensorSpec((i, k), role="input"))
    b = input_tensor("b", TensorSpec((k, j), role="input"))
    program = Program({"c": einsum("ik,kj->ji", a, b)})
    schedule = TensorSchedule(direct_gemm=False)
    full = plan_cuda(program, TARGET, schedule=schedule, library_bytes=0)
    budget = full.peak_bytes - full.panel_bytes + ALIGNMENT
    tiny = plan_cuda(
        program, TARGET, schedule=schedule, library_bytes=0, max_bytes=budget
    )
    assert tiny.panel_bytes == ALIGNMENT
    assert tiny.schedule.tile_m < schedule.tile_m
    assert tiny.peak_bytes <= budget
    assert_disjoint_live_allocations(tiny)
    with pytest.raises(ValueError, match="infeasible"):
        plan_cuda(
            program, TARGET, schedule=schedule, library_bytes=0, max_bytes=budget - 1
        )


def test_reservations_are_real_capacities_and_change_identity() -> None:
    program = Program({"x": vector()})
    plain = plan_cuda(program, TARGET)
    reserved = plan_cuda(
        program,
        TARGET,
        reservations=Reservations(t=3, r=500, diis=4096, concurrent=10000),
    )
    assert reserved.device_bytes - plain.device_bytes == aligned(14599)
    assert reserved.identity != plain.identity
    with pytest.raises(ValueError, match="infeasible"):
        plan_cuda(
            program,
            TARGET,
            max_bytes=plain.peak_bytes,
            reservations=reserved.reservations,
        )


def test_recomputation_can_reduce_retention_with_more_work() -> None:
    x = vector(8192)
    # Breadth/depth scheduling otherwise keeps all these large independent
    # intermediates until their scalar root reductions have executed.
    roots = {
        f"r{i}": reduce_sum(add(x, x, coefficients=(1, i + 1)), (0,)) for i in range(6)
    }
    program = Program(roots)
    retained = plan_cuda(program, TARGET)
    recomputed = plan_cuda(program, TARGET, schedule=TensorSchedule(recompute=True))
    assert recomputed.arena_bytes < retained.arena_bytes
    assert_disjoint_live_allocations(recomputed)
    shared = multiply(x, x)
    program = Program(
        {"a": reduce_sum(shared, (0,)), "b": reduce_sum(add(shared, x), (0,))}
    )
    retained = plan_cuda(program, TARGET)
    recomputed = plan_cuda(program, TARGET, schedule=TensorSchedule(recompute=True))
    assert sum(s.node is shared for s in recomputed.steps) == 2
    assert recomputed.estimated_flops > retained.estimated_flops


@pytest.mark.parametrize(
    "equation,expected",
    [
        ("ik,kj->ij", "NN"),
        ("ki,kj->ij", "TN"),
        ("ik,jk->ij", "NT"),
        ("ki,jk->ij", "TT"),
        ("bik,bkj->bij", "NN"),
    ],
)
def test_direct_gemm_flags(equation: typing.Any, expected: typing.Any) -> None:
    from test_tensor_cuda_gemm import node_for

    plan = plan_cuda(
        Program({"c": node_for(equation, {"i": 3, "j": 5, "k": 7, "b": 2})}), TARGET
    )
    assert plan.steps[-1].gemm == "direct-" + expected
    assert plan.panel_bytes == 0


def test_invalid_types_overflow_and_float32_admission() -> None:
    fp32 = plan_cuda(Program({"x": vector(dtype="float32")}), TARGET)
    assert fp32.precision == "fp32"
    for value in (True, -1, 2**63):
        with pytest.raises(ValueError):
            Reservations(t=value)
        with pytest.raises(ValueError):
            TensorSchedule(tile_m=value)
    with pytest.raises(ValueError, match="signed-64-bit"):
        plan_cuda(Program({"x": vector(2**59)}), TARGET, max_bytes=2**63 - 1)
    with pytest.raises(ValueError, match="workgroup"):
        plan_cuda(
            Program({"x": vector()}),
            TARGET,
            schedule=replace(TensorSchedule(), threads=2048),
        )


def test_cublas_retained_storage_cannot_be_hidden_by_zero_user_workspace() -> None:
    from test_tensor_cuda_gemm import node_for

    program = Program({"c": node_for("ik,kj->ij", {"i": 3, "j": 5, "k": 7})})
    plan = plan_cuda(program, TARGET, library_bytes=0)
    assert plan.provider_bytes == 96 * 1024**2
    assert plan.device_bytes == plan.allocation_bytes + plan.provider_bytes
    with pytest.raises(ValueError, match="provider allowance"):
        plan_cuda(program, TARGET, library_bytes=0, provider_bytes=0)
    with pytest.raises(ValueError, match="infeasible"):
        plan_cuda(program, TARGET, library_bytes=0, max_bytes=plan.peak_bytes - 1)
    # Empty contractions never create a provider handle or charge an allowance.
    empty = Program({"c": node_for("ik,kj->ij", {"i": 3, "j": 5, "k": 0})})
    assert plan_cuda(empty, TARGET).provider_bytes == 0
