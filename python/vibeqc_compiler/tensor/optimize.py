"""Conservative, individually testable rewrites of an immutable definition.

No reassociation, symmetry inference, spin conversion, contraction planning,
or AD occurs here. The original Program remains the mathematical reference.
"""

from __future__ import annotations

import hashlib
import typing
from dataclasses import replace
from fractions import Fraction

from vibeqc_compiler.common.pass_manager import PassManager, PassStage

from .interpreter import execute
from .ir import Node, _infer, constant
from .program import Program, hash_node

PASSES = (
    "dead_nodes",
    "identity_transposes",
    "view_canonicalization",
    "algebraic_canonicalization",
    "exact_cse",
    "scalar_constants",
)


def _constant_values(node: Node) -> tuple[Fraction, ...] | None:
    if node.op != "constant":
        return None
    return tuple(Fraction(*pair) for pair in node.attrs["values"])


def _fold(node: Node) -> Node:
    if node.op not in ("add", "multiply", "divide") or any(
        n.op != "constant" for n in node.inputs
    ):
        return node
    values = [_constant_values(n) for n in node.inputs]
    assert all(value is not None for value in values)
    columns = typing.cast("list[tuple[Fraction, ...]]", values)
    if node.op == "add":
        coefficients = tuple(Fraction(*pair) for pair in node.attrs["coefficients"])
        result = tuple(
            sum(
                (
                    coefficient * column[index]
                    for coefficient, column in zip(coefficients, columns, strict=True)
                ),
                Fraction(0),
            )
            for index in range(node.spec.size)
        )
    elif node.op == "multiply":
        result = tuple(
            columns[0][index] * columns[1][index] for index in range(node.spec.size)
        )
    elif all(columns[1][index] for index in range(node.spec.size)):
        result = tuple(
            columns[0][index] / columns[1][index] for index in range(node.spec.size)
        )
    else:
        return node  # Preserve the original division-by-zero diagnostic.
    candidate = constant(
        result,
        replace(
            node.spec,
            role="constant",
            differentiable=False,
            symmetries=(),
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


def _is_literal_one(node: Node) -> bool:
    values = _constant_values(node)
    return values is not None and all(value == 1 for value in values)


def _same_value_type(result: Node, value: Node) -> bool:
    """Allow identity removal only when all semantics except SSA role agree."""
    return replace(value.spec, role=result.spec.role) == result.spec


def _canonicalize_algebra(node: Node) -> Node:
    """Apply IEEE-safe algebraic identities without reassociation."""
    if node.op == "multiply":
        left, right = node.inputs
        if _is_literal_one(left) and _same_value_type(node, right):
            return right
        if _is_literal_one(right) and _same_value_type(node, left):
            return left
    if node.op == "divide":
        numerator, denominator = node.inputs
        if _is_literal_one(denominator) and _same_value_type(node, numerator):
            return numerator
    return node


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
        if value.op == "slice":
            inner = value.attrs["ranges"]
            outer = node.attrs["ranges"]
            ranges = tuple(
                (inner_start + outer_start, inner_start + outer_stop)
                for (inner_start, _), (outer_start, outer_stop) in zip(
                    inner, outer, strict=True
                )
            )
            return Node(
                "slice",
                (value.inputs[0],),
                node.spec,
                (("ranges", ranges),),
            )
        full = tuple((0, index.extent) for index in value.spec.indices)
        if node.attrs["ranges"] == full:
            return value
    if (
        node.op == "gather"
        and value.op == "gather"
        and node.attrs["axis"] == value.attrs["axis"]
    ):
        inner = value.attrs["positions"]
        positions = tuple(inner[index] for index in node.attrs["positions"])
        return Node(
            "gather",
            (value.inputs[0],),
            node.spec,
            (("axis", node.attrs["axis"]), ("positions", positions)),
        )
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
        elif pass_name == "algebraic_canonicalization":
            updated = _canonicalize_algebra(updated)
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
    version=2,
    stages=(
        PassStage("dead_nodes", 1, _pass("dead_nodes"), invalidates=("liveness",)),
        PassStage("identity_transposes", 1, _pass("identity_transposes")),
        PassStage(
            "view_canonicalization",
            2,
            _pass("view_canonicalization"),
            invalidates=("liveness",),
        ),
        PassStage(
            "algebraic_canonicalization",
            1,
            _pass("algebraic_canonicalization"),
            invalidates=("liveness",),
        ),
        PassStage("exact_cse", 1, _pass("exact_cse"), invalidates=("liveness",)),
        PassStage(
            "scalar_constants",
            2,
            _pass("scalar_constants"),
            invalidates=("liveness",),
        ),
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
