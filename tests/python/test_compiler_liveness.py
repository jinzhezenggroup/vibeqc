"""Shared dead-code liveness and conservative effect semantics."""

import typing

import pytest
from vibeqc_compiler.common.liveness import (
    EffectKind,
    LivenessNode,
    analyze_liveness,
)
from vibeqc_compiler.tensor.ir import add, constant, multiply
from vibeqc_compiler.tensor.optimize import rewrite
from vibeqc_compiler.tensor.program import Program


def node(
    key: typing.Any,
    reads: typing.Any,
    writes: typing.Any,
    effect: EffectKind = EffectKind.PURE,
) -> LivenessNode:
    return LivenessNode(key, tuple(reads), tuple(writes), effect)


def test_backward_liveness_prunes_transitive_dead_pure_branch() -> None:
    graph = (
        node("a", ("x",), ("a",)),
        node("b", ("a",), ("b",)),
        node("dead_a", ("x",), ("dead_a",)),
        node("dead_b", ("dead_a",), ("dead_b",)),
        node("finish", ("b",), ("out",)),
    )
    result = analyze_liveness(graph, roots=("out",), external_values=("x", "unused"))
    assert result.live_node_keys == ("a", "b", "finish")
    assert result.removed_node_keys == ("dead_a", "dead_b")
    assert result.live_external_values == ("x",)
    assert result.live_values == ("x", "a", "b", "out")


@pytest.mark.parametrize("effect", [EffectKind.EFFECTFUL, EffectKind.OPAQUE])
def test_effectful_and_opaque_nodes_are_conservative_roots(
    effect: EffectKind,
) -> None:
    graph = (
        node("prep", ("x",), ("prepared",)),
        node("side", ("prepared",), ("token",), effect),
        node("main", ("y",), ("out",)),
    )
    result = analyze_liveness(
        graph,
        roots=("out",),
        external_values=("x", "y"),
    )
    assert result.live_node_keys == ("prep", "side", "main")
    assert result.live_external_values == ("x", "y")
    assert result.live_values == ("x", "y", "prepared", "token", "out")


def test_multi_output_node_is_retained_as_one_operation() -> None:
    graph = (
        node("split", ("x",), ("left", "right")),
        node("finish", ("left",), ("out",)),
    )
    result = analyze_liveness(graph, roots=("out",), external_values=("x",))
    assert result.live_node_keys == ("split", "finish")
    assert result.live_values == ("x", "left", "right", "out")


def test_empty_root_set_keeps_only_nonpure_effects() -> None:
    graph = (
        node("dead", ("x",), ("dead",)),
        node("opaque", ("x",), ("opaque",), EffectKind.OPAQUE),
    )
    result = analyze_liveness(graph, roots=(), external_values=("x",))
    assert result.live_node_keys == ("opaque",)
    assert result.removed_node_keys == ("dead",)
    assert result.live_values == ("x", "opaque")


@pytest.mark.parametrize(
    "graph,roots,external,match",
    [
        ((node("n", ("missing",), ("y",)),), ("y",), (), "dependency"),
        (
            (
                node("a", (), ("x",)),
                node("b", (), ("x",)),
            ),
            ("x",),
            (),
            "producer",
        ),
        ((node("a", (), ("x",)),), ("missing",), (), "root"),
    ],
)
def test_invalid_dependency_graphs_fail_closed(
    graph: typing.Any, roots: typing.Any, external: typing.Any, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        analyze_liveness(graph, roots=roots, external_values=external)


def test_duplicate_node_key_and_invalid_effect_fail_closed() -> None:
    graph = (
        node("same", (), ("x",)),
        node("same", ("x",), ("y",)),
    )
    with pytest.raises(ValueError, match="duplicate liveness node key"):
        analyze_liveness(graph, roots=("y",))
    with pytest.raises(TypeError, match="EffectKind"):
        LivenessNode("bad", (), ("x",), "pure")


def test_tensor_program_live_nodes_share_common_output_demand_analysis() -> None:
    x = constant(2)
    live_mid = multiply(x, constant(3))
    out = add(live_mid, constant(4))
    dead = multiply(constant(5), constant(6))
    program = Program({"out": out}, definitions=(live_mid, dead))
    live = set(program.live_nodes)
    assert out in live
    assert live_mid in live
    assert x in live
    assert dead not in live
    assert all(node in set(program.nodes) for node in live)

    pruned = rewrite(program, "dead_nodes")
    assert pruned.definitions == (live_mid,)
    assert dead not in pruned.nodes
    assert pruned.logical_hash == program.logical_hash
