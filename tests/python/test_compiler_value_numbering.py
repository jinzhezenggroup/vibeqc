"""Effect-aware GVN shared by TensorIR and generated integral algebra."""

import numpy as np
from vibeqc_compiler.common.liveness import EffectKind
from vibeqc_compiler.common.value_numbering import ValueNumberTable
from vibeqc_compiler.integral.expr import Graph
from vibeqc_compiler.tensor import (
    Program,
    TensorSpec,
    add,
    execute,
    input_tensor,
    multiply,
    optimize,
    transpose,
)
from vibeqc_compiler.tensor.optimize import _value_number


def _equivalent(left: tuple[str, str], right: tuple[str, str]) -> bool:
    return left == right


def test_value_numbering_fails_closed_and_rejects_bucket_collisions() -> None:
    table = ValueNumberTable[tuple[str, str]](equivalent=_equivalent)
    first = table.number("same-bucket", ("add", "a"), effect=EffectKind.PURE)
    duplicate = table.number("same-bucket", ("add", "a"), effect=EffectKind.PURE)
    collision = table.number("same-bucket", ("add", "b"), effect=EffectKind.PURE)
    opaque = table.number("same-bucket", ("add", "a"))
    effectful = table.number(
        "same-bucket",
        ("add", "a"),
        effect=EffectKind.EFFECTFUL,
    )
    copied = table.copy(collision.number, collision.representative)

    assert duplicate.number == first.number
    assert collision.number != first.number
    assert opaque.number not in {first.number, collision.number}
    assert effectful.number not in {first.number, collision.number, opaque.number}
    assert copied.number == collision.number
    diagnostics = table.diagnostics
    assert diagnostics.reused_values == 1
    assert diagnostics.copy_propagations == 1
    assert diagnostics.effect_barriers == 2
    assert diagnostics.rejected_key_collisions == 1


def test_tensor_gvn_merges_independent_pure_construction_paths() -> None:
    spec = TensorSpec(role="parameter")
    x = input_tensor("x", spec)
    y = input_tensor("y", spec)
    first = multiply(x, y)
    second = multiply(x, y)
    program = Program({"first": first, "second": second})

    rewritten, diagnostics = _value_number(program)
    optimized = optimize(program)

    assert rewritten.outputs["first"] is rewritten.outputs["second"]
    assert diagnostics.reused_values >= 1
    assert optimized.outputs["first"] is optimized.outputs["second"]
    recorded = optimized.provenance["optimizer_diagnostics"]["value_numbering"]
    assert recorded[0]["reused_values"] >= 1


def test_tensor_gvn_copy_propagates_proven_identity() -> None:
    spec = TensorSpec(role="parameter")
    x = input_tensor("x", spec)
    identity = transpose(x, ())
    program = Program({"original": x, "copy": identity})

    rewritten, diagnostics = _value_number(program)

    assert rewritten.outputs["original"] is rewritten.outputs["copy"]
    assert diagnostics.copy_propagations == 1


def test_tensor_gvn_preserves_ieee_sensitive_association() -> None:
    spec = TensorSpec(role="parameter")
    x = input_tensor("x", spec)
    y = input_tensor("y", spec)
    z = input_tensor("z", spec)
    left = add(add(x, y), z)
    right = add(x, add(y, z))
    program = Program({"left": left, "right": right})

    rewritten, _ = _value_number(program)

    assert rewritten.outputs["left"] is not rewritten.outputs["right"]
    values = {
        "x": np.array(1.0e16),
        "y": np.array(-1.0e16),
        "z": np.array(1.0),
    }
    outputs = execute(rewritten, values).outputs
    assert float(outputs["left"]) == 1.0
    assert float(outputs["right"]) == 0.0


def test_integral_graph_uses_shared_value_numbering_contract() -> None:
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    first = graph.add(x, y)
    second = graph.add(y, x)

    assert first.identifier == second.identifier
    diagnostics = graph.value_numbering_diagnostics
    assert diagnostics.reused_values >= 1
    assert diagnostics.effect_barriers == 0
