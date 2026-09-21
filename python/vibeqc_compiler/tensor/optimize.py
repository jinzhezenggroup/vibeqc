"""Conservative, individually testable rewrites of an immutable definition.

No reassociation, symmetry inference, spin conversion, contraction planning,
or AD occurs here. The original Program remains the mathematical reference.
"""

from __future__ import annotations

import hashlib
import typing
from dataclasses import replace
from fractions import Fraction

from vibeqc_compiler.common.liveness import EffectKind
from vibeqc_compiler.common.pass_manager import PassManager, PassStage
from vibeqc_compiler.common.value_numbering import (
    ValueNumberingDiagnostics,
    ValueNumberTable,
)

from .interpreter import execute
from .ir import PRIMITIVES, Node, _infer, constant
from .precision import precision_execution_contracts, remap_precision_execution
from .program import Program, hash_node
from .types import spec_to_payload

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


def _tensor_effect(node: Node) -> EffectKind:
    """Map validated TensorIR primitives onto the shared effect contract."""
    return (
        EffectKind.PURE
        if node.op in PRIMITIVES or node.op in {"input", "constant"}
        else EffectKind.OPAQUE
    )


def _same_semantic_value(left: Node, right: Node) -> bool:
    """Guard a value-number bucket hit with complete TensorIR semantics."""
    return (
        left.op == right.op
        and spec_to_payload(left.spec, logical=True)
        == spec_to_payload(right.spec, logical=True)
        and left.attributes == right.attributes
        and len(left.inputs) == len(right.inputs)
        and all(a is b for a, b in zip(left.inputs, right.inputs, strict=True))
    )


def _propagate_identity(node: Node) -> Node:
    """Return a proven value-preserving source, without IEEE reassociation."""
    if node.op == "transpose" and node.attrs["axes"] == tuple(
        range(len(node.spec.indices))
    ):
        return node.inputs[0]
    return _canonicalize_algebra(_canonicalize_view(node))


def _value_number(program: Program) -> tuple[Program, ValueNumberingDiagnostics]:
    """Run copy propagation plus effect-aware GVN over one TensorIR program."""
    table = ValueNumberTable[Node](equivalent=_same_semantic_value)
    replacements: dict[Node, Node] = {}
    hashes: dict[Node, str] = {}
    numbers: dict[Node, int] = {}
    execution_contracts = precision_execution_contracts(program)

    for node in program.nodes:
        inputs = tuple(replacements[child] for child in node.inputs)
        updated = node
        if inputs != node.inputs:
            spec = _infer(node.op, inputs, node.attrs, node.spec)
            updated = Node(node.op, inputs, spec, node.attributes)

        updated = _propagate_identity(updated)
        source_number = numbers.get(updated)
        if source_number is not None:
            decision = table.copy(source_number, updated)
        else:
            digest = hash_node(updated, hashes)
            decision = table.number(
                (digest, execution_contracts.get(node)),
                updated,
                effect=_tensor_effect(updated),
            )
            if not decision.reused:
                hashes[decision.representative] = digest
                numbers[decision.representative] = decision.number

        replacements[node] = decision.representative
        numbers.setdefault(decision.representative, decision.number)

    outputs = {name: replacements[node] for name, node in program.outputs.items()}
    definitions = tuple(replacements[node] for node in program.definitions)
    return (
        Program(
            outputs,
            definitions,
            remap_precision_execution(program, replacements, outputs, definitions),
        ),
        table.diagnostics,
    )


def rewrite(program: Program, pass_name: str) -> Program:
    """Apply one named pass, preserving the pre-rewrite program for comparison."""
    if pass_name not in PASSES:
        raise ValueError(f"unsupported tensor rewrite: {pass_name}")
    if pass_name == "dead_nodes":
        live = set(program.live_nodes)
        definitions = tuple(node for node in program.definitions if node in live)
        replacements = {node: node for node in program.nodes}
        outputs = dict(program.outputs)
        provenance = remap_precision_execution(
            program, replacements, outputs, definitions
        )
        return Program(outputs, definitions, provenance)
    if pass_name == "exact_cse":
        return _value_number(program)[0]
    replacements = {}
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
        replacements[node] = updated
    outputs = {name: replacements[n] for name, n in program.outputs.items()}
    definitions = tuple(replacements[n] for n in program.definitions)
    return Program(
        outputs,
        definitions,
        remap_precision_execution(program, replacements, outputs, definitions),
    )


def _program_fingerprint(program: Program) -> str:
    """Track complete deterministic structure, including retained definitions."""
    return hashlib.sha256(program.dumps().encode()).hexdigest()


def _pass(
    pass_name: str,
    diagnostics: list[ValueNumberingDiagnostics] | None = None,
) -> typing.Callable[[Program], Program]:
    def apply(program: Program) -> Program:
        if pass_name == "exact_cse":
            result, stats = _value_number(program)
            if diagnostics is not None:
                diagnostics.append(stats)
            return result
        return rewrite(program, pass_name)

    return apply


def _project_requested_outputs(
    program: Program,
    requested_outputs: typing.Any,
) -> Program:
    """Make explicit output demand a compiler fact before liveness/scheduling."""

    if requested_outputs is None:
        return program
    if not isinstance(requested_outputs, (tuple, list)):
        raise TypeError("requested outputs must be a sequence")
    requested = tuple(requested_outputs)
    if not requested:
        raise ValueError("TensorIR specialization requires at least one output")
    if any(not isinstance(name, str) for name in requested):
        raise TypeError("requested output names must be strings")
    if len(set(requested)) != len(requested):
        raise ValueError("TensorIR specialization contains duplicate outputs")
    missing = tuple(name for name in requested if name not in program.outputs)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"TensorIR specialization requests unknown outputs: {names}")
    if set(requested) == set(program.outputs):
        return program
    outputs = {name: program.outputs[name] for name in requested}
    candidate = Program(outputs, definitions=program.definitions)
    provenance = remap_precision_execution(
        program,
        {node: node for node in candidate.live_nodes},
        outputs,
        program.definitions,
        allow_pruned=True,
    )
    return Program(outputs, definitions=program.definitions, provenance=provenance)


def _input_names(program: Program) -> tuple[str, ...]:
    return tuple(
        sorted(node.attrs["name"] for node in program.nodes if node.op == "input")
    )


def _pruning_diagnostics(
    before: Program,
    requested: Program,
    after: Program,
) -> dict[str, typing.Any]:
    """Summarize #673 output/input/DCE pruning before backend lowering."""

    before_nodes = before.nodes
    after_nodes = after.nodes
    after_live_nodes = after.live_nodes
    available_outputs = tuple(before.outputs)
    requested_outputs = tuple(requested.outputs)
    retained_outputs = tuple(after.outputs)
    if requested_outputs != retained_outputs:
        raise ValueError("TensorIR optimizer changed requested output names")
    if after_nodes != after_live_nodes:
        raise ValueError("TensorIR optimizer left dead definitions before lowering")
    before_inputs = _input_names(before)
    after_inputs = _input_names(after)
    if not set(after_inputs) <= set(before_inputs):
        raise ValueError("TensorIR optimizer introduced a new external input")
    removed_outputs = tuple(
        name for name in available_outputs if name not in set(requested_outputs)
    )
    removed_inputs = tuple(
        name for name in before_inputs if name not in set(after_inputs)
    )
    return {
        "schema": "vibeqc.compiler.pruning.v1",
        "nodes_before": len(before_nodes),
        "nodes_after": len(after_nodes),
        "nodes_removed": len(before_nodes) - len(after_nodes),
        "definitions_before": len(before.definitions),
        "definitions_after": len(after.definitions),
        "definitions_removed": len(before.definitions) - len(after.definitions),
        "available_outputs": list(available_outputs),
        "requested_outputs": list(requested_outputs),
        "retained_outputs": list(retained_outputs),
        "removed_outputs": list(removed_outputs),
        "inputs_before": list(before_inputs),
        "inputs_after": list(after_inputs),
        "removed_inputs": list(removed_inputs),
        "minimal_before_lowering": True,
    }


def _optimizer(
    diagnostics: list[ValueNumberingDiagnostics] | None = None,
) -> PassManager[Program]:
    return PassManager(
        name="tensor.optimize",
        version=4,
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
            PassStage(
                "exact_cse",
                2,
                _pass("exact_cse", diagnostics),
                invalidates=("liveness",),
            ),
            PassStage(
                "scalar_constants",
                2,
                _pass("scalar_constants"),
                invalidates=("liveness",),
            ),
            # Folding may expose new value-number and liveness opportunities.
            PassStage(
                "post_fold_exact_cse",
                2,
                _pass("exact_cse", diagnostics),
                invalidates=("liveness",),
            ),
            PassStage(
                "post_fold_dead_nodes",
                1,
                _pass("dead_nodes"),
                invalidates=("liveness",),
            ),
        ),
        fingerprint=_program_fingerprint,
    )


def optimize(program: Program, *, requested_outputs: typing.Any = None) -> Program:
    """Run the initial TensorIR pipeline through the shared pass manager."""
    specialized = _project_requested_outputs(program, requested_outputs)
    value_numbering_diagnostics: list[ValueNumberingDiagnostics] = []
    run = _optimizer(value_numbering_diagnostics).run(specialized)
    result = run.value
    pruning = _pruning_diagnostics(program, specialized, result)
    return Program(
        result.outputs,
        provenance={
            **result.provenance,
            "original_logical_hash": program.logical_hash,
            "specialized_logical_hash": specialized.logical_hash,
            "pruning_diagnostics": pruning,
            # Keep the established rewrite inventory for compatibility.
            "rewrites": list(PASSES) + ["exact_cse", "dead_nodes"],
            "optimizer_identity": run.pipeline_identity,
            "optimizer_diagnostics": {
                "nodes_before": len(program.nodes),
                "nodes_after": len(result.nodes),
                "value_numbering": [
                    stats.to_payload() for stats in value_numbering_diagnostics
                ],
            },
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
