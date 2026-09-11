"""Independent loop checks, view boundaries, packing metrics, and rewrites."""

import json
from dataclasses import replace
from fractions import Fraction
from itertools import pairwise

import numpy as np
import pytest
from vibeqc_compiler.tensor import (
    PASSES,
    PRIMITIVES,
    Index,
    IndexSpace,
    PackedLayout,
    Program,
    Symmetry,
    TensorSpec,
    add,
    broadcast,
    constant,
    divide,
    einsum,
    execute,
    gather,
    input_tensor,
    multiply,
    optimize,
    reduce_sum,
    reshape,
    rewrite,
    slice_tensor,
    transpose,
)
from vibeqc_compiler.tensor.examples import example_cases

from tools.vibeqc_validation.schema import GATES, block_error

FP64 = GATES["integral_fp64"]


def assert_gate(actual, reference):
    assert block_error(actual, reference, **FP64)["passed"]


@pytest.mark.parametrize("seed", [145, 321, 702])
def test_examples_against_loops_after_replay_and_each_rewrite(seed):
    for case in example_cases(seed):
        source = case.program.dumps()
        program = Program.loads(source)
        assert program.dumps() == source
        assert_gate(execute(program, case.inputs).outputs["value"], case.reference)
        for pass_name in PASSES:
            rewritten = rewrite(program, pass_name)
            assert_gate(
                execute(rewritten, case.inputs).outputs["value"], case.reference
            )
            program = rewritten
        optimized = optimize(case.program)
        assert_gate(
            execute(Program.loads(optimized.dumps()), case.inputs).outputs["value"],
            case.reference,
        )
        assert case.program.dumps() == source
        if case.packing:
            layout = PackedLayout.from_payload(
                json.loads(json.dumps(case.packing.to_payload()))
            )
            # Restricted output and spin-orbital inputs exercise distinct
            # packing contracts; no shape-based symmetry substitution occurs.
            dense = (
                case.reference
                if case.name == "restricted_pair_update"
                else case.inputs.get("g", case.inputs.get("t"))
            )
            assert_gate(layout.unpack(layout.pack(dense)), dense)


def test_views_gathers_reductions_and_broadcasts_against_coordinate_loops():
    o, v = IndexSpace("o", "occupied", 3), IndexSpace("v", "virtual", 4)
    i, a = Index("i", o), Index("a", v)
    x = input_tensor("x", TensorSpec((i, a), role="parameter", differentiable=True))
    tile = slice_tensor(x, ((1, 3), (1, 4)))
    selected = gather(tile, 1, (2, 0, 2))
    reduced = reduce_sum(selected, (1,))
    batch = Index("batch", IndexSpace("batch", "batch", 2))
    expanded = broadcast(reduced, (batch, reduced.spec.indices[0]), (1,))
    permuted = transpose(selected, (1, 0))
    flat_axis = Index("flat", IndexSpace("flat", "batch", 6))
    flat = reshape(permuted, (flat_axis,))
    # A nonmonotone broadcast axis map must also transpose the input values.
    reordered = broadcast(x, (a, batch, i), (2, 0))
    program = Program(
        {
            "selected": selected,
            "sum": reduced,
            "expanded": expanded,
            "flat": flat,
            "reordered": reordered,
            "original": x,
        }
    )
    values = np.arange(24.0).reshape(3, 8)[:, ::-2]
    assert not values.flags.c_contiguous
    before = values.copy()
    result = execute(Program.loads(program.dumps()), {"x": values}, debug=True)
    expected = np.zeros((2, 3))
    for local_i in range(2):
        for local_a, source_a in enumerate((3, 1, 3)):
            expected[local_i, local_a] = values[local_i + 1, source_a]
    assert_gate(result.outputs["selected"], expected)
    assert_gate(result.outputs["sum"], np.array([sum(row) for row in expected]))
    assert_gate(
        result.outputs["expanded"], np.array([[sum(row) for row in expected]] * 2)
    )
    assert_gate(
        result.outputs["flat"],
        np.array([expected[j, k] for k in range(3) for j in range(2)]),
    )
    for av, bv, iv in np.ndindex(4, 2, 3):
        assert result.outputs["reordered"][av, bv, iv] == values[iv, av]
    assert selected.spec.indices[1].selection == (3, 1, 3)
    assert selected.spec.indices[0].start == 1
    assert selected.spec.differentiable
    for output in result.outputs.values():
        assert not np.shares_memory(output, values)
    outputs = list(result.outputs.values()) + list(result.intermediates.values())
    for j, output in enumerate(outputs):
        assert all(not np.shares_memory(output, other) for other in outputs[j + 1 :])
    result.outputs["original"][:] = -20
    np.testing.assert_array_equal(values, before)
    assert values.flags.writeable


def test_gather_of_gather_and_sliced_partial_blocks_keep_global_identity():
    space = IndexSpace("o", "occupied", 7)
    x = input_tensor("x", TensorSpec((Index("i", space, 2, 6),), role="input"))
    g = slice_tensor(gather(gather(x, 0, (3, 0, 2)), 0, (2, 1, 2)), ((1, 3),))
    assert g.spec.indices[0].selection == (2, 4)
    actual = execute(
        Program.loads(Program({"g": g}).dumps()), {"x": np.arange(2.0, 6.0)}
    ).outputs["g"]
    np.testing.assert_array_equal(actual, [2.0, 4.0])
    with pytest.raises(ValueError, match="index domains"):
        add(gather(x, 0, (0, 1)), gather(x, 0, (1, 0)))
    with pytest.raises(ValueError, match="index domains"):
        add(slice_tensor(x, ((0, 2),)), slice_tensor(x, ((2, 4),)))


def test_repeated_einsum_indices_trace_scalar_and_prefactor_against_loops():
    o, v, aux = (
        IndexSpace("o", "occupied", 2),
        IndexSpace("v", "virtual", 3),
        IndexSpace("aux", "auxiliary", 4),
    )
    x = input_tensor(
        "x",
        TensorSpec(
            (Index("i", o), Index("a", v), Index("b", v), Index("P", aux)), role="input"
        ),
    )
    trace = einsum("ijjk->ik", x, coefficient=Fraction(-3, 4))
    values = np.random.default_rng(781).normal(size=x.spec.shape)
    reference = np.zeros((2, 4))
    for i in range(2):
        for p in range(4):
            for a in range(3):
                reference[i, p] -= 0.75 * values[i, a, a, p]
    assert_gate(
        execute(Program({"trace": trace}), {"x": values}).outputs["trace"], reference
    )
    scalar = constant("1/2")
    y = input_tensor("y", TensorSpec((Index("i", o),), role="input"))
    p = Program(
        {
            "scalar": einsum("->", scalar),
            "scale": einsum(",i->i", scalar, y),
            "outer": einsum("i,j->ij", y, y),
        }
    )
    result = execute(p, {"y": np.array([2.0, -3.0])}).outputs
    assert result["scalar"].shape == ()
    assert float(result["scalar"]) == 0.5
    np.testing.assert_array_equal(result["scale"], [1.0, -1.5])
    np.testing.assert_array_equal(result["outer"], [[4, -6], [-6, 9]])


@pytest.mark.parametrize("size", [0, 1])
def test_empty_and_singleton_axes_have_defined_reduction_and_broadcast(size):
    axis = Index("i", IndexSpace("o", "occupied", size))
    one = Index("b", IndexSpace("b", "batch", 1))
    x = input_tensor("x", TensorSpec((axis,), role="input"))
    empty = slice_tensor(x, ((0, 0),))
    p = Program(
        {
            "trace": einsum("i,i->", x, x),
            "empty": empty,
            "gather": gather(x, 0, ()),
            "sum": reduce_sum(empty, (0,)),
            "view": reshape(empty, (empty.spec.indices[0], one)),
            "expanded": broadcast(x, (axis, one), (0,)),
        }
    )
    actual = execute(Program.loads(p.dumps()), {"x": np.ones(size)}).outputs
    assert actual["trace"].shape == ()
    assert float(actual["trace"]) == size
    assert float(actual["sum"]) == 0
    assert actual["empty"].shape == actual["gather"].shape == (0,)
    assert actual["expanded"].shape == (size, 1)
    assert actual["view"].shape == (0, 1)
    layout = PackedLayout.from_spec(x.spec)
    assert layout.unpack(layout.pack(np.ones(size))).shape == (size,)


def test_float32_lowering_rounds_only_at_execution():
    axis = Index("i", IndexSpace("o", "occupied", 3))
    x = input_tensor("x", TensorSpec((axis,), dtype="float32", role="input"))
    result = execute(
        Program({"value": add(x, coefficients=("1/3",))}),
        {"x": np.array([1, 2, 3], dtype=np.float32)},
    ).outputs["value"]
    assert result.dtype == np.float32
    np.testing.assert_allclose(result, [1 / 3, 2 / 3, 1], atol=1e-7)


def test_packed_spatial_and_spin_orbital_metrics_are_distinct():
    o, v = IndexSpace("o", "occupied", 2), IndexSpace("v", "virtual", 3)
    indices = (Index("i", o), Index("j", o), Index("a", v), Index("b", v))
    spatial = TensorSpec(
        indices,
        representation="restricted_spatial",
        role="input",
        symmetries=(Symmetry((1, 0, 3, 2)),),
    )
    spin = replace(
        spatial,
        representation="spin_orbital",
        symmetries=(Symmetry((1, 0, 2, 3), -1), Symmetry((0, 1, 3, 2), -1)),
    )
    spatial_layout, spin_layout = PackedLayout(spatial), PackedLayout(spin)
    assert spatial_layout.size == 21
    assert spin_layout.size == 3
    assert set(spatial_layout.weights) == {1, 2}
    assert spin_layout.weights == (4, 4, 4)
    rng = np.random.default_rng(122)
    for layout in (spatial_layout, spin_layout):
        a, b = rng.normal(size=(2, layout.size))
        dense_a, dense_b = layout.unpack(a), layout.unpack(b)
        expected = sum(dense_a[c] * dense_b[c] for c in np.ndindex(layout.spec.shape))
        assert abs(layout.inner_product(a, b) - expected) <= 1e-11
        assert_gate(layout.pack(dense_a), a)
        for i, j, av, bv in np.ndindex(layout.spec.shape):
            assert dense_a[i, j, av, bv] == dense_a[j, i, bv, av]
            if layout is spin_layout:
                assert dense_a[i, j, av, bv] == -dense_a[j, i, av, bv]
                assert dense_a[i, j, av, bv] == -dense_a[i, j, bv, av]
        node = input_tensor("t", layout.spec)
        assert_gate(execute(Program({"t": node}), {"t": dense_a}).outputs["t"], dense_a)
        broken = dense_a.copy()
        broken[0, 1, 0, 1] += 0.5
        with pytest.raises(ValueError, match="symmetry"):
            layout.pack(broken)
        with pytest.raises(ValueError, match="symmetry"):
            execute(Program({"t": node}), {"t": broken})
    with pytest.raises(ValueError, match="budget"):
        PackedLayout.from_spec(spatial, max_elements=1)
    damaged = spatial_layout.to_payload()
    damaged["weights"][0] += 1
    with pytest.raises(ValueError, match="weights"):
        PackedLayout.from_payload(damaged)


def test_transpose_carries_symmetry_and_constant_zero_orbits():
    case = example_cases()[1]
    t = input_tensor("t", case.packing.spec)
    p = Program({"permuted": transpose(t, (2, 0, 3, 1))})
    value = case.inputs["g"]
    actual = execute(Program.loads(p.dumps()), {"t": value}).outputs["permuted"]
    for symmetry in p.outputs["permuted"].spec.symmetries:
        np.testing.assert_array_equal(
            actual, symmetry.sign * actual.transpose(symmetry.permutation)
        )
    axis = Index("i", IndexSpace("o", "occupied", 1))
    zeros = PackedLayout(TensorSpec((axis,), symmetries=(Symmetry((0,), -1),)))
    assert zeros.size == 0
    np.testing.assert_array_equal(zeros.unpack(np.empty(0)), [0.0])


def test_each_rewrite_has_an_effect_without_erasing_original_equation():
    case = example_cases()[0]
    left, right = case.program.outputs["value"].inputs
    first, duplicate = (
        einsum("pP,Pq->pq", left, right),
        einsum("ik,kj->ij", left, right),
    )
    result = add(transpose(first, (0, 1)), duplicate, coefficients=("1/2", "1/2"))
    folded = divide(add(constant("1/2"), constant("1/4")), constant(3))
    dead = multiply(constant(2), constant(3))
    source = Program({"value": result, "folded": folded}, definitions=(dead,))
    snapshot = source.dumps()
    p = source
    counts = [len(p.nodes)]
    for pass_name in PASSES:
        p = rewrite(p, pass_name)
        outputs = execute(p, case.inputs).outputs
        assert_gate(outputs["value"], case.reference)
        assert float(outputs["folded"]) == 0.25
        counts.append(len(p.nodes))
    assert all(after < before for before, after in pairwise(counts))
    assert p.outputs["folded"].op == "constant"
    assert source.dumps() == snapshot
    assert_gate(
        execute(Program.loads(optimize(source).dumps()), case.inputs).outputs["value"],
        case.reference,
    )


def test_constant_folding_preserves_roundoff_overflow_and_zero_division():
    cancellation = add(constant(10**16), constant(1), constant(-(10**16)))
    p = rewrite(Program({"value": cancellation}), "scalar_constants")
    assert p.outputs["value"].op == "add"
    assert float(execute(p, {}).outputs["value"]) == 0.0
    bad = divide(constant(1), constant(0))
    with pytest.raises(ValueError, match="division by zero"):
        execute(optimize(Program({"bad": bad})), {})
    huge = add(constant(10**400), constant(1))
    with pytest.raises(ValueError, match="non-finite"):
        execute(optimize(Program({"huge": huge})), {})


def test_cse_never_merges_different_spin_symmetry_or_parameter_roles():
    space = IndexSpace("o", "occupied", 2)
    indices = (Index("i", space), Index("j", space))
    specs = [
        TensorSpec(indices, role="input"),
        TensorSpec(indices, role="input", representation="spin_orbital"),
        TensorSpec(indices, role="input", symmetries=(Symmetry((1, 0)),)),
        TensorSpec(indices, role="parameter", differentiable=True),
    ]
    outputs = {f"o{i}": input_tensor(f"x{i}", spec) for i, spec in enumerate(specs)}
    p = rewrite(Program(outputs), "exact_cse")
    assert len(set(p.outputs.values())) == len(specs)
    assert len(
        {Program({"value": node}).logical_hash for node in outputs.values()}
    ) == len(specs)
    assert all(
        "real operands" in contract.differentiable_operands
        for contract in PRIMITIVES.values()
    )
    assert "scatter-add" in PRIMITIVES["gather"].accumulation
