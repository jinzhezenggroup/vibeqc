"""CPU proofs of storage invariants and deliberately infeasible CUDA plans."""

from dataclasses import replace

import pytest

from tools.vibeqc_codegen.cuda_target import cuda_target_info
from tools.vibeqc_tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    broadcast,
    einsum,
    input_tensor,
    multiply,
    reduce_sum,
    transpose,
)
from tools.vibeqc_tensor.cuda_plan import (
    ALIGNMENT,
    Reservations,
    TensorSchedule,
    aligned,
    plan_cuda,
)

TARGET = cuda_target_info("sm_80")


def vector(size=65, dtype="float64"):
    index = Index("i", IndexSpace("axis", "batch", size))
    return input_tensor("x", TensorSpec((index,), dtype=dtype, role="input"))


def assert_disjoint_live_allocations(plan):
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


def test_reuse_keeps_inputs_and_outputs_and_releases_dead_work():
    x = vector()
    chain = [x]
    for _ in range(12):
        chain.append(add(chain[-1], x))
    plan = plan_cuda(Program({"result": chain[-1], "retained": chain[2]}), TARGET)
    assert_disjoint_live_allocations(plan)
    assert plan.arena_bytes < sum(aligned(s.node.spec.size * 8) for s in plan.steps)
    assert len({s.offset for s in plan.steps}) < len(plan.steps)
    for _, i in plan.outputs:
        assert plan.steps[i].last_use == len(plan.steps)
    assert plan.peak_bytes == plan.device_bytes + plan.host_bytes


def test_alias_lifetime_follows_materialized_ancestors():
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


def test_fusion_preserves_checks_before_subsets():
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


def test_constrained_plan_shrinks_panels_and_rejects_below_indivisible_minimum():
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


def test_reservations_are_real_capacities_and_change_identity():
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


def test_recomputation_can_reduce_retention_with_more_work():
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
def test_direct_gemm_flags(equation, expected):
    from test_tensor_cuda_gemm import node_for

    plan = plan_cuda(
        Program({"c": node_for(equation, {"i": 3, "j": 5, "k": 7, "b": 2})}), TARGET
    )
    assert plan.steps[-1].gemm == "direct-" + expected
    assert plan.panel_bytes == 0


def test_invalid_types_overflow_and_float32_fail_before_allocation():
    with pytest.raises(ValueError, match="float64"):
        plan_cuda(Program({"x": vector(dtype="float32")}), TARGET)
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


def test_cublas_retained_storage_cannot_be_hidden_by_zero_user_workspace():
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
