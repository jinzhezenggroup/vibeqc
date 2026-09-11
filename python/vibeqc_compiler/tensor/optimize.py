"""Conservative, individually testable rewrites of an immutable definition.

No reassociation, symmetry inference, spin conversion, contraction planning,
or AD occurs here. The original Program remains the mathematical reference.
"""

from __future__ import annotations

from fractions import Fraction

from .interpreter import execute
from .ir import Node, _infer, constant
from .program import Program, hash_node
from .types import TensorSpec

PASSES = ("dead_nodes", "identity_transposes", "exact_cse", "scalar_constants")


def _fold(node: Node) -> Node:
    if (
        node.spec.shape
        or node.op not in ("add", "multiply", "divide")
        or any(n.op != "constant" for n in node.inputs)
    ):
        return node
    values = [Fraction(*n.attrs["values"][0]) for n in node.inputs]
    if node.op == "add":
        value = sum(
            (Fraction(*c) * x for c, x in zip(node.attrs["coefficients"], values)),
            Fraction(0),
        )
    elif node.op == "multiply":
        value = values[0] * values[1]
    elif values[1]:
        value = values[0] / values[1]
    else:
        return node  # Preserve the original division-by-zero diagnostic.
    candidate = constant(
        value,
        TensorSpec(
            dtype=node.spec.dtype,
            representation=node.spec.representation,
            role="constant",
        ),
    )
    # Exact rational algebra alone is insufficient for floating-point folding:
    # e.g. 1e16 + 1 - 1e16 must not become 1 in an FP64 interpreter. Fold only
    # when the rounded literal agrees bit-for-bit with the original scalar DAG.
    try:
        before = execute(Program({"value": node}), {}).outputs["value"]
        after = execute(Program({"value": candidate}), {}).outputs["value"]
    except ValueError:
        return node
    return candidate if before.tobytes() == after.tobytes() else node


def rewrite(program: Program, pass_name: str) -> Program:
    """Apply one named pass, preserving the pre-rewrite program for comparison."""
    if pass_name not in PASSES:
        raise ValueError(f"unsupported tensor rewrite: {pass_name}")
    if pass_name == "dead_nodes":
        return Program(program.outputs, provenance=program.provenance)
    replacements, interned, hashes = {}, {}, {}
    for node in program.nodes:
        inputs = tuple(replacements[n] for n in node.inputs)
        updated = node
        if inputs != node.inputs:
            spec = _infer(node.op, inputs, node.attrs, node.spec)
            updated = Node(node.op, inputs, spec, node.attributes)
        if pass_name == "identity_transposes" and updated.op == "transpose":
            if updated.attrs["axes"] == tuple(range(len(updated.spec.indices))):
                updated = inputs[0]
        elif pass_name == "scalar_constants":
            updated = _fold(updated)
        elif pass_name == "exact_cse":
            # This content address contains the dtype, spaces/ranges/spins,
            # symmetry declarations, parameter role, and differentiability.
            # Equal shapes alone cannot intern two different amplitude types.
            hashes[updated] = hash_node(updated, hashes)
            updated = interned.setdefault(hashes[updated], updated)
        replacements[node] = updated
    return Program(
        {name: replacements[n] for name, n in program.outputs.items()},
        tuple(replacements[n] for n in program.definitions),
        program.provenance,
    )


def optimize(program: Program) -> Program:
    """Run the four initial passes; keep provenance back to the source DAG."""
    result = program
    for pass_name in PASSES:
        result = rewrite(result, pass_name)
    # Folding may expose duplicates; a final exact CSE/dead pass is still
    # conservative and does not reorder any arithmetic.
    result = rewrite(rewrite(result, "exact_cse"), "dead_nodes")
    return Program(
        result.outputs,
        provenance={
            **program.provenance,
            "original_logical_hash": program.logical_hash,
            "rewrites": list(PASSES) + ["exact_cse", "dead_nodes"],
        },
    )
