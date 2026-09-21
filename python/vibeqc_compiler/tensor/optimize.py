"""Conservative, individually testable rewrites of an immutable definition.

No reassociation, symmetry inference, spin conversion, contraction planning,
or AD occurs here. The original Program remains the mathematical reference.
"""

from __future__ import annotations

import hashlib
import typing
from fractions import Fraction

from vibeqc_compiler.common.pass_manager import PassManager, PassStage

from .interpreter import execute
from .ir import Node, _infer, constant
from .program import Program, hash_node
from .types import TensorSpec

PASSES = (
    "dead_nodes",
    "identity_transposes",
    "view_canonicalization",
    "exact_cse",
    "scalar_constants",
)


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


def _canonicalize_view(node: Node) -> Node:
    """Collapse representation-only chains without changing arithmetic order."""
    if len(node.inputs) != 1:
        return node
    value = node.inputs[0]
    if node.op == "cast" and node.attrs["dtype"] == value.spec.dtype:
        return value
    if node.op == "reshape":
        source = value.inputs[0] if value.op == "reshape" else value
        if node.spec.indices == source.spec.indices:
            return source
        if source is not value:
            return Node("reshape", (source,), node.spec)
    if node.op == "transpose" and value.op == "transpose":
        inner = value.attrs["axes"]
        outer = node.attrs["axes"]
        axes = tuple(inner[index] for index in outer)
        source = value.inputs[0]
        if axes == tuple(range(len(axes))):
            return source
        return Node("transpose", (source,), node.spec, (("axes", axes),))
    if node.op == "slice":
        full = tuple((0, index.extent) for index in value.spec.indices)
        if node.attrs["ranges"] == full:
            return value
    if node.op == "broadcast":
        axes = node.attrs["axes"]
        if (
            axes == tuple(range(len(value.spec.indices)))
            and node.spec.indices == value.spec.indices
        ):
            return value
    return node


def rewrite(program: Program, pass_name: str) -> Program:
    """Apply one named pass, preserving the pre-rewrite program for comparison."""
    if pass_name not in PASSES:
        raise ValueError(f"unsupported tensor rewrite: {pass_name}")
    if pass_name == "dead_nodes":
        live = set(program.live_nodes)
        definitions = tuple(node for node in program.definitions if node in live)
        return Program(
            program.outputs,
            definitions=definitions,
            provenance=program.provenance,
        )
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
        elif pass_name == "view_canonicalization":
            updated = _canonicalize_view(updated)
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


def _program_fingerprint(program: Program) -> str:
    """Track complete deterministic structure, including retained definitions."""
    return hashlib.sha256(program.dumps().encode()).hexdigest()


def _pass(pass_name: str) -> typing.Callable[[Program], Program]:
    def apply(program: Program) -> Program:
        return rewrite(program, pass_name)

    return apply


_OPTIMIZER = PassManager(
    name="tensor.optimize",
    version=1,
    stages=(
        PassStage("dead_nodes", 1, _pass("dead_nodes"), invalidates=("liveness",)),
        PassStage("identity_transposes", 1, _pass("identity_transposes")),
        PassStage(
            "view_canonicalization",
            1,
            _pass("view_canonicalization"),
            invalidates=("liveness",),
        ),
        PassStage("exact_cse", 1, _pass("exact_cse"), invalidates=("liveness",)),
        PassStage("scalar_constants", 1, _pass("scalar_constants")),
        # Folding may expose duplicates; final CSE/DCE remains conservative.
        PassStage(
            "post_fold_exact_cse", 1, _pass("exact_cse"), invalidates=("liveness",)
        ),
        PassStage(
            "post_fold_dead_nodes", 1, _pass("dead_nodes"), invalidates=("liveness",)
        ),
    ),
    fingerprint=_program_fingerprint,
)


def optimize(program: Program) -> Program:
    """Run the initial TensorIR pipeline through the shared pass manager."""
    run = _OPTIMIZER.run(program)
    result = run.value
    return Program(
        result.outputs,
        provenance={
            **program.provenance,
            "original_logical_hash": program.logical_hash,
            # Keep the established rewrite inventory for compatibility.
            "rewrites": list(PASSES) + ["exact_cse", "dead_nodes"],
            "optimizer_identity": run.pipeline_identity,
            "optimizer_passes": [
                {
                    "name": record.name,
                    "version": record.version,
                    "changed": record.changed,
                }
                for record in run.records
            ],
        },
    )
