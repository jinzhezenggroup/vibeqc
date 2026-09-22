"""Dense layout proofs and producer/consumer planning without a CUDA runtime."""

import typing
from dataclasses import replace
from itertools import permutations

import numpy as np
import pytest
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.common.layout import SymmetricPairLayout
from vibeqc_compiler.tensor import (
    DenseLayout,
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    broadcast,
    divide,
    einsum,
    gather,
    input_tensor,
    multiply,
    reduce_sum,
    reshape,
    slice_tensor,
    transpose,
)
from vibeqc_compiler.tensor.cuda_emit import emit_cuda
from vibeqc_compiler.tensor.cuda_gemm import gemm_contract
from vibeqc_compiler.tensor.cuda_layout import MAX_LAYOUT_TRIALS, conversion_bytes
from vibeqc_compiler.tensor.cuda_plan import ALIGNMENT, TensorSchedule, plan_cuda

TARGET = cuda_target_info("sm_80")


def operand(name: typing.Any, labels: typing.Any, dimensions: typing.Any) -> typing.Any:
    return input_tensor(
        name,
        TensorSpec(
            tuple(
                Index(label, IndexSpace(label, "batch", dimensions[label]))
                for label in labels
            ),
            role="parameter",
            differentiable=True,
        ),
    )


def producer_case(
    kind: typing.Any = "multiply",
    *,
    dimensions: typing.Any = None,
    both: typing.Any = False,
) -> typing.Any:
    """Interleaved batch layout at a producer; GEMM needs batch-prefix storage."""
    dims = {"i": 3, "b": 2, "k": 7, "j": 5} if dimensions is None else dimensions
    x = operand("x", "ibk", dims)
    y = operand("y", "kjb" if both else "bkj", dims)
    if kind == "multiply":
        value = multiply(x, x)
    elif kind == "add":
        value = add(x, x, coefficients=(1, 2))
    elif kind == "divide":
        value = divide(x, add(x, x, coefficients=(1, 2)))
    elif kind == "transpose":
        x = operand("x", "bik", dims)
        value = transpose(x, (1, 0, 2))
    elif kind == "gather":
        value = gather(x, 0, (2, 0, 2))
    elif kind == "slice":
        value = slice_tensor(x, ((0, dims["i"]), (0, dims["b"]), (1, dims["k"])))
        y = slice_tensor(y, ((0, dims["b"]), (1, dims["k"]), (0, dims["j"])))
    elif kind == "broadcast":
        x = operand("x", "ik", dims)
        axes = operand("unused", "ibk", dims).spec.indices
        value = broadcast(x, axes, (0, 2))
    elif kind == "reshape":
        value = reshape(x, x.spec.indices)
    elif kind == "reduce":
        x = operand("x", "ibkr", {**dims, "r": 3})
        value = reduce_sum(x, (3,))
    else:
        raise ValueError(kind)
    right = multiply(y, y) if both else y
    equation = "ibk,kjb->bij" if both else "ibk,bkj->bij"
    program = Program({"out": einsum(equation, value, right)})
    rng = np.random.default_rng(509)
    feeds = {
        node.attrs["name"]: rng.uniform(0.2, 0.8, node.spec.shape)
        for node in program.live_nodes
        if node.op == "input"
    }
    return program, feeds, value


@pytest.mark.parametrize("shape", [(), (1,), (2, 3), (2, 1, 3), (2, 3, 5)])
def test_dense_layout_bijection_matches_independent_numpy(
    shape: typing.Any,
) -> None:
    reference = np.arange(np.prod(shape, dtype=int)).reshape(shape)
    for order in permutations(range(len(shape))):
        layout = DenseLayout(shape, order, 256)
        physical = np.ascontiguousarray(reference.transpose(order)).ravel()
        for logical, expected in enumerate(reference.ravel()):
            offset = layout.physical_index(logical)
            assert physical[offset] == expected
            assert layout.logical_index(offset) == logical
        # An independent ndarray stride check, including singleton axes.
        logical = physical.reshape(tuple(shape[i] for i in order)).transpose(
            np.argsort(order)
        )
        np.testing.assert_array_equal(logical, reference)
        for axes in permutations(range(len(shape))):
            view = layout.transpose(axes)
            expected = reference.transpose(axes).ravel()
            for index, value in enumerate(expected):
                assert physical[view.physical_index(index)] == value


def test_symmetric_pair_layout_matches_native_lower_triangle_and_costs() -> None:
    layout = SymmetricPairLayout(4, (7,), alignment=8)

    assert layout.logical_shape == (4, 4, 7)
    assert layout.storage_shape == (10, 7)
    assert layout.storage_elements == 70
    assert layout.dense_elements == 112
    assert layout.storage_bytes(8) == 560
    assert layout.dense_equivalent_bytes(8) == 896
    assert (
        layout.dense_materialization_bytes(8, rows=2, trailing_shape=(3,))
        == 2 * 4 * 3 * 8
    )
    assert (
        layout.unpack_traffic_bytes(8, rows=2, trailing_shape=(3,)) == 2 * 2 * 4 * 3 * 8
    )

    expected = {
        (high, low): high * (high + 1) // 2 + low
        for high in range(4)
        for low in range(high + 1)
    }
    for first in range(4):
        for second in range(4):
            high, low = max(first, second), min(first, second)
            assert layout.pair_index(first, second) == expected[(high, low)]
            assert layout.pair_index(first, second) == layout.pair_index(second, first)

    assert layout.to_payload()["exact"] is True
    assert layout.identity == SymmetricPairLayout(4, (7,), alignment=8).identity
    assert layout.identity != SymmetricPairLayout(5, (7,), alignment=8).identity


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SymmetricPairLayout(-1),
        lambda: SymmetricPairLayout(True),
        lambda: SymmetricPairLayout(2, (-1,)),
        lambda: SymmetricPairLayout(2, alignment=3),
    ],
)
def test_invalid_symmetric_pair_layouts_fail_closed(factory: typing.Any) -> None:
    with pytest.raises(ValueError):
        factory()

    layout = SymmetricPairLayout(3, (5,))
    with pytest.raises(ValueError, match="outside"):
        layout.pair_index(3, 0)
    with pytest.raises(ValueError, match="rank"):
        layout.dense_materialization_bytes(8, trailing_shape=())
    with pytest.raises(ValueError, match="exceed"):
        layout.dense_materialization_bytes(8, rows=4, trailing_shape=(1,))
    with pytest.raises(ValueError, match="exceed"):
        layout.dense_materialization_bytes(8, rows=1, trailing_shape=(6,))


@pytest.mark.parametrize("shape", [(0,), (2, 0, 3)])
def test_empty_layouts_have_no_valid_indices(shape: typing.Any) -> None:
    layout = DenseLayout(shape, tuple(reversed(range(len(shape)))))
    assert layout.is_c_contiguous
    with pytest.raises(ValueError, match="outside"):
        layout.physical_index(0)
    with pytest.raises(ValueError, match="outside"):
        layout.logical_index(0)


@pytest.mark.parametrize(
    "options",
    [
        {"shape": (2, -1)},
        {"shape": (True,)},
        {"shape": (2**62, 4)},
        {"shape": (2, 3), "order": (0, 0)},
        {"shape": (2, 3), "order": (False, 1)},
        {"shape": (), "alignment": 3},
        {"shape": (), "alignment": 0},
    ],
)
def test_invalid_layouts_fail_closed(options: typing.Any) -> None:
    with pytest.raises(ValueError):
        DenseLayout(**options)


def test_layout_descriptors_do_not_retain_mutable_sequences() -> None:
    shape, order = [2, 3], [1, 0]
    layout = DenseLayout(shape, order)
    shape[0], order[0] = 99, 0
    assert layout.shape == (2, 3)
    assert layout.order == (1, 0)
    assert DenseLayout((2, 1, 3), (1, 0, 2)).is_c_contiguous
    with pytest.raises(ValueError):
        layout.transpose((0, 0))


@pytest.mark.parametrize(
    "kind",
    [
        "multiply",
        "add",
        "divide",
        "transpose",
        "gather",
        "slice",
        "broadcast",
        "reshape",
        "reduce",
    ],
)
def test_producer_layout_removes_packing(kind: typing.Any) -> None:
    program, _, producer = producer_case(kind)
    logical_hash, serialized = program.logical_hash, program.dumps()
    baseline = plan_cuda(program, TARGET)
    plan = plan_cuda(program, TARGET, schedule=TensorSchedule(layouts=True))
    assert baseline.steps[-1].gemm == "packed"
    assert plan.steps[-1].gemm.startswith("direct-")
    assert baseline.panel_bytes > 0 and plan.panel_bytes == 0
    step = next(s for s in plan.steps if s.node is producer)
    assert step.layout.order == (1, 0, 2)
    assert not step.virtual
    assert program.logical_hash == logical_hash and program.dumps() == serialized
    assert baseline.layout_identity != plan.layout_identity
    assert baseline.identity != plan.identity
    assert plan.identity == plan_cuda(program, TARGET, schedule=plan.schedule).identity
    assert (
        plan.layout_decision.selected_cost
        < plan.layout_decision.baseline_conversion_bytes
    )
    assert plan.layout_decision.trials <= MAX_LAYOUT_TRIALS
    assert f"pack_{len(plan.steps) - 1}" not in emit_cuda(plan)
    for index in (*plan.inputs, *(i for _, i in plan.outputs)):
        assert plan.steps[index].layout.is_c_contiguous
    assert len(plan.steps) == len(baseline.steps)
    assert plan.arena_bytes == baseline.arena_bytes


def test_joint_operand_choice_avoids_a_greedy_single_producer_dead_end() -> None:
    program, _, _ = producer_case(both=True)
    baseline = plan_cuda(program, TARGET)
    plan = plan_cuda(program, TARGET, schedule=TensorSchedule(layouts=True))
    assert baseline.steps[-1].gemm == "packed"
    assert plan.steps[-1].gemm == "direct-NN"
    assert len(plan.layout_decision.changed_steps) == 2


@pytest.mark.parametrize("large_first", [False, True])
def test_conflicting_consumers_are_costed_together_without_duplicate_storage(
    large_first: typing.Any,
) -> None:
    dims = {
        "i": 3,
        "b": 2,
        "k": 7,
        "j": 31 if large_first else 3,
        "l": 3 if large_first else 31,
    }
    x = operand("x", "ibk", dims)
    value = multiply(x, x)
    one = einsum("ibk,bkj->bij", value, operand("y", "bkj", dims))
    two = einsum("ibk,kl->ibl", value, operand("z", "kl", dims))
    plan = plan_cuda(
        Program({"one": one, "two": two}), TARGET, schedule=TensorSchedule(layouts=True)
    )
    assert sum(s.node is value for s in plan.steps) == 1
    step = next(s for s in plan.steps if s.node is value)
    assert step.layout.is_c_contiguous is not large_first
    kinds = {s.node: s.gemm for s in plan.steps}
    assert kinds[one] == ("direct-NN" if large_first else "packed")
    assert kinds[two] == ("packed" if large_first else "direct-NN")


def gemm_producer_case(
    *, packed_producer: typing.Any = False, packed_consumer: typing.Any = False
) -> typing.Any:
    dims = {"i": 3, "b": 2, "k": 7, "j": 5, "l": 11}
    if packed_producer:
        value = einsum(
            "ibk,bkj->ibj", operand("x", "ibk", dims), operand("y", "bkj", dims)
        )
        result = einsum("ibj,bjl->bil", value, operand("z", "bjl", dims))
    else:
        value = einsum("ik,kj->ji", operand("x", "ik", dims), operand("y", "kj", dims))
        result = einsum(
            "ji,il->" + ("lj" if packed_consumer else "jl"),
            value,
            operand("z", "il", dims),
        )
    program = Program({"out": result})
    rng = np.random.default_rng(951)
    feeds = {
        n.attrs["name"]: rng.normal(size=n.spec.shape)
        for n in program.live_nodes
        if n.op == "input"
    }
    return program, feeds, value


@pytest.mark.parametrize(
    "packed_producer,packed_consumer", [(False, False), (True, False), (False, True)]
)
def test_gemm_output_layout_can_propagate_forward_and_backward(
    packed_producer: typing.Any, packed_consumer: typing.Any
) -> None:
    program, _, producer = gemm_producer_case(
        packed_producer=packed_producer, packed_consumer=packed_consumer
    )
    baseline = plan_cuda(program, TARGET)
    plan = plan_cuda(program, TARGET, schedule=TensorSchedule(layouts=True))
    step = next(s for s in plan.steps if s.node is producer)
    assert not step.layout.is_c_contiguous
    assert step.gemm == ("packed" if packed_producer else "direct-NN")
    assert plan.steps[-1].gemm == (
        "packed" if packed_consumer else "direct-NN" if packed_producer else "direct-TN"
    )
    assert plan.layout_decision.selected_cost < baseline.layout_decision.selected_cost
    assert emit_cuda(plan)


def test_views_outputs_and_explicit_fallback_keep_their_contracts() -> None:
    program, _, value = producer_case("transpose")
    for schedule in (
        TensorSchedule(views=True, layouts=True),
        TensorSchedule(direct_gemm=False, layouts=True),
    ):
        plan = plan_cuda(program, TARGET, schedule=schedule)
        assert plan.steps[-1].gemm == "packed"
    # Retaining the intermediate as a named output prevents changing its ABI.
    pinned = Program({**program.outputs, "retained": value, "alias": value})
    plan = plan_cuda(pinned, TARGET, schedule=TensorSchedule(layouts=True))
    step = next(s for s in plan.steps if s.node is value)
    assert step.layout.is_c_contiguous
    assert plan.layout_decision.changed_steps == ()
    with pytest.raises(TypeError, match="boolean"):
        TensorSchedule(layouts=1)


def test_panel_savings_enable_a_previously_infeasible_budget() -> None:
    from test_tensor_cuda_plan import assert_disjoint_live_allocations

    program, _, _ = producer_case()
    optimized = plan_cuda(program, TARGET, schedule=TensorSchedule(layouts=True))
    with pytest.raises(ValueError, match="infeasible"):
        plan_cuda(program, TARGET, max_bytes=optimized.peak_bytes)
    plan = plan_cuda(
        program, TARGET, schedule=optimized.schedule, max_bytes=optimized.peak_bytes
    )
    assert plan.peak_bytes == optimized.peak_bytes
    assert_disjoint_live_allocations(plan)
    with pytest.raises(ValueError, match="infeasible"):
        plan_cuda(
            program,
            TARGET,
            schedule=optimized.schedule,
            max_bytes=optimized.peak_bytes - 1,
        )


def test_conversion_accounting_uses_actual_budget_shrunk_tiles() -> None:
    program, _, _ = gemm_producer_case(packed_consumer=True)
    full = plan_cuda(program, TARGET, schedule=TensorSchedule(layouts=True))
    tiny = plan_cuda(
        program,
        TARGET,
        schedule=full.schedule,
        max_bytes=full.peak_bytes - full.panel_bytes + ALIGNMENT,
    )
    assert tiny.panel_bytes == ALIGNMENT
    expected = sum(
        conversion_bytes(gemm_contract(s.node), tiny.schedule, s.node.spec.itemsize)
        for s in tiny.steps
        if s.gemm == "packed"
    )
    assert tiny.layout_decision.selected_conversion_bytes == expected
    assert (
        tiny.layout_decision.selected_cost
        <= tiny.layout_decision.baseline_conversion_bytes
    )


def test_emitter_rejects_noncanonical_public_output_layout() -> None:
    program, _, _ = producer_case()
    plan = plan_cuda(program, TARGET)
    steps = list(plan.steps)
    steps[-1] = replace(
        steps[-1], layout=DenseLayout(steps[-1].node.spec.shape, (1, 0, 2))
    )
    with pytest.raises(ValueError, match="ABI"):
        emit_cuda(replace(plan, steps=tuple(steps)))


@pytest.mark.parametrize(
    "equation,kind",
    [
        ("ik,kj->ij", "NN"),
        ("ki,kj->ij", "TN"),
        ("ik,jk->ij", "NT"),
        ("ki,jk->ij", "TT"),
        ("bik,bkj->bij", "NN"),
    ],
)
def test_opt_in_keeps_existing_direct_transpose_flags(
    equation: typing.Any, kind: typing.Any
) -> None:
    from test_tensor_cuda_gemm import node_for

    plan = plan_cuda(
        Program({"out": node_for(equation, {"b": 2, "i": 3, "j": 5, "k": 7})}),
        TARGET,
        schedule=TensorSchedule(layouts=True),
    )
    assert plan.steps[-1].gemm == "direct-" + kind
    assert not plan.layout_decision.changed_steps


def self_product_case() -> typing.Any:
    x = operand("x", "ibk", {"i": 3, "b": 2, "k": 7})
    value = multiply(x, x)
    program = Program({"out": einsum("ibk,jbk->bij", value, value)})
    return program, {"x": np.random.default_rng(1509).normal(size=x.spec.shape)}, value


def test_same_producer_in_both_operand_slots_requires_one_compatible_layout() -> None:
    program, _, value = self_product_case()
    plan = plan_cuda(program, TARGET, schedule=TensorSchedule(layouts=True))
    assert plan.steps[-1].gemm == "direct-NT"
    assert len(plan.layout_decision.changed_steps) == 1
    assert sum(s.node is value for s in plan.steps) == 1


def grouped_output_case() -> typing.Any:
    from test_tensor_cuda_gemm import node_for

    value = node_for(
        "abef,ijef->ijab", {"a": 3, "b": 2, "e": 3, "f": 2, "i": 2, "j": 3}
    )
    program = Program({"out": add(value, value)})
    rng = np.random.default_rng(509)
    feeds = {
        n.attrs["name"]: rng.normal(size=n.spec.shape)
        for n in program.live_nodes
        if n.op == "input"
    }
    return program, feeds, value


def test_grouped_mnk_labels_preserve_physical_order() -> None:
    program, _, value = grouped_output_case()
    plan = plan_cuda(program, TARGET, schedule=TensorSchedule(layouts=True))
    step = next(s for s in plan.steps if s.node is value)
    assert step.gemm == "direct-NT"
    assert step.layout.order == (2, 3, 0, 1)
    assert plan.panel_bytes == 0


def test_layout_identity_can_fail_closed_in_shared_specialization_guards() -> None:
    from vibeqc_compiler.common.specialization import GuardPredicate

    program, _, _ = producer_case()
    baseline = plan_cuda(program, TARGET)
    plan = plan_cuda(program, TARGET, schedule=TensorSchedule(layouts=True))
    guard = GuardPredicate("workload", "layout_identity", "eq", plan.layout_identity)
    assert (
        guard.failure({"workload": {"layout_identity": plan.layout_identity}}) is None
    )
    assert guard.failure({"workload": {"layout_identity": baseline.layout_identity}})
    assert guard.failure({"workload": {}})


def test_region_search_has_an_explicit_trial_limit() -> None:
    dims = {"i": 2, "b": 2, "k": 2, "j": 2}
    right = operand("y", "bkj", dims)
    outputs = {}
    for i in range(100):
        x = operand(f"x{i}", "ibk", dims)
        outputs[f"out{i}"] = einsum("ibk,bkj->bij", multiply(x, x), right)
    plan = plan_cuda(Program(outputs), TARGET, schedule=TensorSchedule(layouts=True))
    assert plan.layout_decision.trials == MAX_LAYOUT_TRIALS
    assert plan.layout_decision.truncated
    assert (
        plan.layout_decision.selected_cost
        <= plan.layout_decision.baseline_conversion_bytes
    )


def test_fp32_layout_admission_does_not_change_ordinary_fp32() -> None:
    x = input_tensor("x", TensorSpec(dtype="float32", role="input"))
    program = Program({"out": add(x, x)})
    assert plan_cuda(program, TARGET).precision == "fp32"
    with pytest.raises(ValueError, match="layout.*float64"):
        plan_cuda(program, TARGET, schedule=TensorSchedule(layouts=True))


@pytest.mark.parametrize("other", [None, object(), (2, 3), 1, "layout"])
def test_dense_layout_equivalence_rejects_foreign_types(
    other: typing.Any,
) -> None:
    """Type annotations must not remove the existing runtime comparison guard."""
    from vibeqc_compiler.common.layout import DenseLayout

    assert DenseLayout((2, 3)).equivalent(other) is False
