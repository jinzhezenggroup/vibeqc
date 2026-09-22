"""Resolved precision schedules and explicit cast-cost provenance.

This module describes precision semantics already present in a TensorIR
program. It never silently rewrites an equation or promotes a low-precision
candidate. Method controllers may generate explicit cast DAGs and use this
schedule as the stable compiler/audit identity consumed by the existing
planner, tuner, cache, and evidence pipeline.
"""

from __future__ import annotations

import typing
from collections.abc import Mapping
from dataclasses import dataclass, replace

from vibeqc_compiler.common.precision import (
    DTYPES,
    STRICT_MATH_MODE,
    ExecutionPrecisionSchedule,
    PrecisionDirective,
)

from .ir import TRANSCENDENTALS, Node, cast
from .program import Program, _hash
from .types import checked_size

REDUCTION_OPS = frozenset(("reduce", "einsum", "scatter_add", "segment_sum"))
MIXED_ACCUMULATION_OPS = frozenset(("reduce", "einsum"))
MIXED_ACCUMULATION_SCHEMA = "vibeqc.tensor.precision-execution.v1"
SENSITIVE_OPS = frozenset(("divide", "scaled_bilinear")) | TRANSCENDENTALS
AUTO_FP32_OPS = frozenset(
    ("add", "multiply", "transpose", "reshape", "slice", "gather", "broadcast")
)


def _dtype(value: typing.Any, label: str) -> str:
    if value not in DTYPES:
        raise ValueError(f"{label} must be float32 or float64")
    return value


@dataclass(frozen=True)
class ValuePrecision:
    """Resolved storage/compute/accumulation semantics for one SSA value."""

    name: str
    op: str
    storage_dtype: str
    compute_dtype: str
    accumulation_dtype: str
    sensitivity: str
    math_mode: str = STRICT_MATH_MODE

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("precision value requires a stable node name")
        if not isinstance(self.op, str) or not self.op:
            raise ValueError("precision value requires an operation name")
        for label in ("storage_dtype", "compute_dtype", "accumulation_dtype"):
            _dtype(getattr(self, label), label)
        if self.sensitivity not in (
            "ordinary",
            "reduction",
            "sensitive",
            "cast",
        ):
            raise ValueError("invalid precision sensitivity class")
        if self.math_mode != STRICT_MATH_MODE:
            raise ValueError("unsupported TensorIR arithmetic mode")

    def to_payload(self) -> dict:
        return {
            "name": self.name,
            "op": self.op,
            "storage_dtype": self.storage_dtype,
            "compute_dtype": self.compute_dtype,
            "accumulation_dtype": self.accumulation_dtype,
            "sensitivity": self.sensitivity,
            "math_mode": self.math_mode,
        }


@dataclass(frozen=True)
class CastBoundary:
    """One explicit SSA conversion and its unavoidable logical byte traffic."""

    name: str
    source_dtype: str
    target_dtype: str
    elements: int
    read_bytes: int
    write_bytes: int

    def __post_init__(self) -> None:
        _dtype(self.source_dtype, "cast source dtype")
        _dtype(self.target_dtype, "cast target dtype")
        for label in ("elements", "read_bytes", "write_bytes"):
            checked_size(getattr(self, label), label)

    @property
    def simultaneous_bytes(self) -> int:
        return checked_size(
            self.read_bytes + self.write_bytes, "cast simultaneous bytes"
        )

    def to_payload(self) -> dict:
        return {
            "name": self.name,
            "source_dtype": self.source_dtype,
            "target_dtype": self.target_dtype,
            "elements": self.elements,
            "read_bytes": self.read_bytes,
            "write_bytes": self.write_bytes,
            "simultaneous_bytes": self.simultaneous_bytes,
        }


@dataclass(frozen=True)
class PrecisionSchedule:
    """Complete precision identity for one already-resolved TensorIR DAG.

    source_equation remains the scientific/source identity when a controller
    records it in provenance; lowered_equation is the concrete typed DAG.
    Strict FP64 audit/refinement remains an external method-controller gate and
    is deliberately not claimed by a compiler schedule alone.
    """

    source_equation: str
    lowered_equation: str
    values: tuple[ValuePrecision, ...]
    casts: tuple[CastBoundary, ...]
    strict_audit_dtype: str = "float64"
    audit_owner: str = "method-controller"
    math_mode: str = STRICT_MATH_MODE
    request_identity: str | None = None
    qualification_scope: tuple[tuple[str, str], ...] = ()
    execution_scope: tuple[tuple[str, str], ...] = ()
    parent_schedule_identity: str | None = None

    def __post_init__(self) -> None:
        _dtype(self.strict_audit_dtype, "strict audit dtype")
        if self.request_identity is not None and (
            not isinstance(self.request_identity, str)
            or len(self.request_identity) != 64
            or any(c not in "0123456789abcdef" for c in self.request_identity)
        ):
            raise ValueError("invalid precision request identity")
        if self.parent_schedule_identity is not None and (
            not isinstance(self.parent_schedule_identity, str)
            or len(self.parent_schedule_identity) != 64
            or any(c not in "0123456789abcdef" for c in self.parent_schedule_identity)
        ):
            raise ValueError("invalid parent precision schedule identity")
        if (
            self.qualification_scope or self.execution_scope
        ) and self.request_identity is None:
            raise ValueError(
                "precision qualification/execution scope requires a request identity"
            )
        value_names = {value.name for value in self.values}
        qualified_sources = {name for name, _ in self.qualification_scope}
        if (
            len({source for source, _ in self.execution_scope})
            != len(self.execution_scope)
            or len({lowered for _, lowered in self.execution_scope})
            != len(self.execution_scope)
            or any(
                source not in qualified_sources for source, _ in self.execution_scope
            )
            or any(lowered not in value_names for _, lowered in self.execution_scope)
        ):
            raise ValueError("invalid precision execution scope")
        if self.audit_owner != "method-controller":
            raise ValueError("TensorIR precision audit owner must be method-controller")
        if self.math_mode != STRICT_MATH_MODE:
            raise ValueError("unsupported TensorIR arithmetic mode")
        if len({value.name for value in self.values}) != len(self.values):
            raise ValueError("precision schedule node names must be unique")
        if any(value.math_mode != self.math_mode for value in self.values):
            raise ValueError("per-value and schedule arithmetic modes must agree")

    @property
    def cast_read_bytes(self) -> int:
        return checked_size(
            sum(cast.read_bytes for cast in self.casts), "cast read bytes"
        )

    @property
    def cast_write_bytes(self) -> int:
        return checked_size(
            sum(cast.write_bytes for cast in self.casts), "cast write bytes"
        )

    @property
    def maximum_cast_live_bytes(self) -> int:
        return max((cast.simultaneous_bytes for cast in self.casts), default=0)

    @property
    def execution_precision(self) -> ExecutionPrecisionSchedule:
        """Project TensorIR precision into the shared cross-IR contract."""

        qualification_by_source = dict(self.qualification_scope)
        source_by_lowered = {
            lowered: source for source, lowered in self.execution_scope
        }
        return ExecutionPrecisionSchedule(
            tuple(
                (
                    value.name,
                    PrecisionDirective(
                        storage_dtype=value.storage_dtype,
                        compute_dtype=value.compute_dtype,
                        accumulation_dtype=value.accumulation_dtype,
                        qualification=qualification_by_source.get(
                            source_by_lowered.get(value.name, "")
                        ),
                        math_mode=value.math_mode,
                    ),
                )
                for value in self.values
            ),
            strict_audit_dtype=self.strict_audit_dtype,
            audit_owner=self.audit_owner,
            math_mode=self.math_mode,
        )

    def to_payload(self) -> dict:
        return {
            "schema": "vibeqc.tensor.precision-schedule.v3",
            "precision_request_identity": self.request_identity,
            "parent_precision_schedule_identity": self.parent_schedule_identity,
            "qualification_scope": [
                {"source_value": name, "qualification": qualification}
                for name, qualification in self.qualification_scope
            ],
            "execution_scope": [
                {"source_value": source, "lowered_value": lowered}
                for source, lowered in self.execution_scope
            ],
            "source_equation": self.source_equation,
            "lowered_equation": self.lowered_equation,
            "strict_audit_dtype": self.strict_audit_dtype,
            "audit_owner": self.audit_owner,
            "math_mode": self.math_mode,
            "values": [value.to_payload() for value in self.values],
            "casts": [cast.to_payload() for cast in self.casts],
            "cast_read_bytes": self.cast_read_bytes,
            "cast_write_bytes": self.cast_write_bytes,
            "maximum_cast_live_bytes": self.maximum_cast_live_bytes,
            "promotion": "requires-independent-numerical-and-endpoint-evidence",
        }

    @property
    def identity(self) -> str:
        return _hash(self.to_payload())


def _sensitivity(op: str) -> str:
    if op == "cast":
        return "cast"
    if op in REDUCTION_OPS:
        return "reduction"
    if op in SENSITIVE_OPS:
        return "sensitive"
    return "ordinary"


def _execution_bindings(
    program: Program,
) -> tuple[dict[str, PrecisionDirective], tuple[tuple[str, str], ...]]:
    """Validate lowered mixed-accumulation bindings and return current directives."""
    provenance = program.provenance
    execution = provenance.get("precision_execution")
    if execution is None:
        return {}, ()
    if (
        not isinstance(execution, dict)
        or execution.get("schema") != MIXED_ACCUMULATION_SCHEMA
    ):
        raise ValueError("unsupported precision execution schema")
    request = provenance.get("precision_request")
    request_identity = provenance.get("precision_request_identity")
    if (
        not isinstance(request, dict)
        or request.get("schema") != "vibeqc.tensor.precision-request.v1"
        or request_identity != _hash(request)
        or execution.get("precision_request_identity") != request_identity
    ):
        raise ValueError("precision execution is not bound to its validated request")
    directives = request.get("directives")
    values = execution.get("values")
    if not isinstance(directives, dict) or not isinstance(values, dict):
        raise TypeError("precision execution requires directive/value mappings")
    names = program.debug_names
    live = {names[node]: node for node in program.live_nodes}
    result: dict[str, PrecisionDirective] = {}
    scope = []
    for name, binding in sorted(values.items()):
        if (
            not isinstance(name, str)
            or name not in live
            or not isinstance(binding, dict)
        ):
            raise ValueError("precision execution references an unknown live value")
        if set(binding) != {
            "source_value",
            "compute_dtype",
            "accumulation_dtype",
            "qualification",
        }:
            raise ValueError("invalid precision execution binding")
        source = binding["source_value"]
        payload = directives.get(source)
        if not isinstance(source, str) or not isinstance(payload, dict):
            raise TypeError("precision execution source directive is missing")
        directive = PrecisionDirective(**payload)
        node = live[name]
        if (
            node.op not in MIXED_ACCUMULATION_OPS
            or node.spec.dtype != "float32"
            or directive.storage_dtype != "float32"
            or directive.compute_dtype != "float32"
            or directive.accumulation_dtype != "float64"
            or binding["compute_dtype"] != directive.compute_dtype
            or binding["accumulation_dtype"] != directive.accumulation_dtype
            or binding["qualification"] != directive.qualification
        ):
            raise ValueError("invalid mixed-accumulation execution binding")
        result[name] = directive
        scope.append((source, name))
    return result, tuple(scope)


def precision_execution_contracts(
    program: Program,
) -> dict[Node, tuple[str, str, str, str]]:
    """Return rewrite-equivalence keys for explicit execution precision.

    Mathematical node identity intentionally excludes execution precision. Optimizer
    CSE therefore needs this separate key whenever a qualified accumulation contract
    is attached through provenance. Qualification labels do not change arithmetic and
    are deliberately excluded so equivalent qualified programs may still CSE.
    """
    execution, _ = _execution_bindings(program)
    if not execution:
        return {}
    names = program.debug_names
    result = {}
    for node in program.live_nodes:
        directive = execution.get(names[node])
        if directive is None:
            continue
        result[node] = (
            node.spec.dtype,
            directive.compute_dtype,
            directive.accumulation_dtype,
            directive.math_mode,
        )
    return result


def remap_precision_execution(
    program: Program,
    replacements: Mapping[Node, Node],
    outputs: Mapping[str, Node],
    definitions: tuple[Node, ...],
    *,
    allow_pruned: bool = False,
) -> dict:
    """Transport validated mixed-accumulation bindings through a rewrite.

    Precision execution is keyed by content-addressed debug names while optimizer
    rewrites may change those names without changing the requested arithmetic.
    Equivalent qualified nodes may merge, but incompatible contracts must never do so.
    """
    provenance = program.provenance
    execution = provenance.get("precision_execution")
    if execution is None:
        return provenance

    _execution_bindings(program)
    old_names = program.debug_names
    values = execution["values"]

    base = dict(provenance)
    base.pop("precision_execution", None)
    interim = Program(outputs, definitions, provenance=base)
    new_names = interim.debug_names
    remapped: dict[str, dict] = {}
    contracts: dict[str, tuple[str, str, str]] = {}

    for node in program.live_nodes:
        row = values.get(old_names[node])
        if row is None:
            continue
        target = replacements.get(node)
        # Only explicit output projection may intentionally discard bindings.
        # Ordinary rewrites must still account for every formerly live value.
        if target is None and allow_pruned:
            continue
        if target is None or target not in new_names:
            raise ValueError("optimizer dropped a precision-bound live value")
        new_name = new_names[target]
        contract = (
            target.spec.dtype,
            row["compute_dtype"],
            row["accumulation_dtype"],
        )
        previous_contract = contracts.get(new_name)
        if previous_contract is not None and previous_contract != contract:
            raise ValueError(
                "optimizer merged incompatible precision execution contracts"
            )
        contracts[new_name] = contract

        candidate = dict(row)
        previous = remapped.get(new_name)
        if previous is None or (
            candidate["source_value"],
            candidate.get("qualification") or "",
        ) < (previous["source_value"], previous.get("qualification") or ""):
            remapped[new_name] = candidate

    if remapped:
        base["precision_execution"] = {
            "schema": MIXED_ACCUMULATION_SCHEMA,
            "precision_request_identity": execution["precision_request_identity"],
            "values": remapped,
        }
    return base


def lower_precision(
    program: Program,
    directives: Mapping[str, PrecisionDirective],
    *,
    strict_audit_dtype: str = "float64",
) -> Program:
    """Lower explicit per-value precision requests into a typed cast DAG.

    External input/output dtypes remain unchanged. Qualified FP32 reductions
    may accumulate in FP64; every other distinct compute/accumulation contract
    fails closed rather than being silently approximated. Sensitive/reduction
    FP32 requests require an external qualification id.
    """
    if not isinstance(program, Program):
        raise TypeError("precision lowering requires a TensorIR Program")
    if not isinstance(directives, Mapping):
        raise TypeError("precision directives must be a mapping")
    _dtype(strict_audit_dtype, "strict audit dtype")
    names = program.debug_names
    live = {names[node]: node for node in program.live_nodes}
    inherited, _ = _execution_bindings(program)
    normalized: dict[str, PrecisionDirective] = dict(inherited)
    for name, directive in directives.items():
        if not isinstance(name, str) or name not in live:
            raise ValueError(f"unknown live precision value: {name!r}")
        if not isinstance(directive, PrecisionDirective):
            raise TypeError("precision directive values must be PrecisionDirective")
        node = live[name]
        if node.op in ("input", "constant", "cast"):
            raise ValueError("precision directives target computed non-cast values")
        if directive.compute_dtype != directive.accumulation_dtype and (
            node.op not in MIXED_ACCUMULATION_OPS
            or directive.storage_dtype != "float32"
            or directive.compute_dtype != "float32"
            or directive.accumulation_dtype != "float64"
        ):
            raise ValueError(
                "separate compute/accumulation dtype lowering supports only "
                "qualified FP32 storage/compute with FP64 accumulation on reductions"
            )
        sensitivity = _sensitivity(node.op)
        if (
            sensitivity in ("reduction", "sensitive")
            and any(
                dtype != "float64"
                for dtype in (
                    directive.storage_dtype,
                    directive.compute_dtype,
                    directive.accumulation_dtype,
                )
            )
            and directive.qualification is None
        ):
            raise ValueError(
                "sensitive/reduction FP32 lowering requires a qualification id"
            )
        normalized[name] = directive

    mapping: dict[Node, Node] = {}
    computed: dict[Node, Node] = {}
    cast_cache: dict[tuple[Node, str], Node] = {}

    def ensure_dtype(node: Node, dtype: str) -> Node:
        if node.spec.dtype == dtype:
            return node
        key = (node, dtype)
        converted = cast_cache.get(key)
        if converted is None:
            converted = cast(node, dtype)
            cast_cache[key] = converted
        return converted

    for node in program.nodes:
        if node.op in ("input", "constant"):
            mapping[node] = node
            continue
        if node.op == "cast":
            mapping[node] = cast(mapping[node.inputs[0]], node.attrs["dtype"])
            continue
        directive = normalized.get(names[node])
        compute_dtype = (
            node.spec.dtype if directive is None else directive.compute_dtype
        )
        storage_dtype = (
            node.spec.dtype if directive is None else directive.storage_dtype
        )
        if node.op == "runtime_indexed_select":
            inputs = (
                ensure_dtype(mapping[node.inputs[0]], compute_dtype),
                *(mapping[child] for child in node.inputs[1:]),
            )
        else:
            inputs = tuple(
                ensure_dtype(mapping[child], compute_dtype) for child in node.inputs
            )
        declared = replace(
            node.spec,
            dtype=compute_dtype,
            role="intermediate",
        )
        rebuilt = Node(node.op, inputs, declared, node.attributes)
        computed[node] = rebuilt
        mapping[node] = ensure_dtype(rebuilt, storage_dtype)

    outputs = {
        name: ensure_dtype(mapping[node], node.spec.dtype)
        for name, node in program.outputs.items()
    }
    source_equation = program.provenance.get(
        "precision_source_equation", program.logical_hash
    )
    request = {
        "schema": "vibeqc.tensor.precision-request.v1",
        "parent_precision_schedule_identity": describe_precision(program).identity,
        "source_equation": source_equation,
        "strict_audit_dtype": strict_audit_dtype,
        "math_mode": STRICT_MATH_MODE,
        "directives": {
            name: directive.to_payload()
            for name, directive in sorted(normalized.items())
        },
    }
    request_identity = _hash(request)
    definitions = tuple(mapping[node] for node in program.definitions)
    provenance = {
        **program.provenance,
        "precision_source_equation": source_equation,
        "precision_request": request,
        "precision_request_identity": request_identity,
    }
    provenance.pop("precision_execution", None)
    interim = Program(outputs, definitions, provenance=provenance)
    lowered_names = interim.debug_names
    execution_values = {}
    for source_name, directive in sorted(normalized.items()):
        if directive.compute_dtype == directive.accumulation_dtype:
            continue
        source_node = live[source_name]
        lowered_node = computed[source_node]
        execution_values[lowered_names[lowered_node]] = {
            "source_value": source_name,
            "compute_dtype": directive.compute_dtype,
            "accumulation_dtype": directive.accumulation_dtype,
            "qualification": directive.qualification,
        }
    if execution_values:
        provenance["precision_execution"] = {
            "schema": MIXED_ACCUMULATION_SCHEMA,
            "precision_request_identity": request_identity,
            "values": execution_values,
        }
    return Program(outputs, definitions, provenance=provenance)


def conservative_precision_variants(program: Program) -> tuple[Program, ...]:
    """Return strict plus one opt-in FP32 ordinary-subgraph candidate.

    Reductions/contractions, quotient-like operations, transcendental
    operations, inputs, constants, and existing casts retain their source
    precision. This only generates a benchmark candidate; promotion still
    belongs to the existing endpoint/numerical evidence gate.
    """
    if not isinstance(program, Program):
        raise TypeError("precision variants require a TensorIR Program")
    names = program.debug_names
    directives = {
        names[node]: PrecisionDirective("float32", "float32", "float32")
        for node in program.live_nodes
        if node.spec.dtype == "float64" and node.op in AUTO_FP32_OPS
    }
    if not directives:
        return (program,)
    lowered = lower_precision(program, directives)
    if lowered.logical_hash == program.logical_hash:
        return (program,)
    return (program, lowered)


def _request_scope(
    program: Program, source: str
) -> tuple[str | None, tuple[tuple[str, str], ...]]:
    """Bind external qualification labels to source/value scope, not just provenance."""
    provenance = program.provenance
    request = provenance.get("precision_request")
    recorded = provenance.get("precision_request_identity")
    if request is None:
        if recorded is not None:
            raise ValueError("precision request identity has no request")
        return None, ()
    if (
        not isinstance(request, dict)
        or request.get("schema") != "vibeqc.tensor.precision-request.v1"
    ):
        raise ValueError("unsupported precision request schema")
    if request.get("source_equation") != source or recorded != _hash(request):
        raise ValueError("precision request scope or identity mismatch")
    parent = request.get("parent_precision_schedule_identity")
    if parent is not None and (
        not isinstance(parent, str)
        or len(parent) != 64
        or any(c not in "0123456789abcdef" for c in parent)
    ):
        raise ValueError("invalid parent precision request identity")
    directives = request.get("directives")
    if not isinstance(directives, dict) or any(
        not isinstance(name, str) for name in directives
    ):
        raise ValueError("precision request requires named directives")
    qualifications = []
    for name, payload in sorted(directives.items()):
        if not isinstance(payload, dict):
            raise TypeError("precision request directive must be an object")
        directive = PrecisionDirective(**payload)
        if directive.qualification is not None:
            qualifications.append((name, directive.qualification))
    return recorded, tuple(qualifications)


def describe_precision(
    program: Program, *, strict_audit_dtype: str | None = None
) -> PrecisionSchedule:
    """Resolve dtype/sensitivity/cast facts without inventing a promotion policy."""
    if not isinstance(program, Program):
        raise TypeError("precision scheduling requires a TensorIR Program")
    if strict_audit_dtype is None:
        request = program.provenance.get("precision_request")
        strict_audit_dtype = (
            request.get("strict_audit_dtype", "float64")
            if isinstance(request, dict)
            else "float64"
        )
    _dtype(strict_audit_dtype, "strict audit dtype")
    names = program.debug_names
    execution, execution_scope = _execution_bindings(program)
    values = []
    casts = []
    for node in program.live_nodes:
        if node.spec.dtype == "int64":
            continue
        sensitivity = _sensitivity(node.op)
        binding = execution.get(names[node])
        accumulation_dtype = (
            node.spec.dtype if binding is None else binding.accumulation_dtype
        )
        values.append(
            ValuePrecision(
                names[node],
                node.op,
                node.spec.dtype,
                node.spec.dtype,
                accumulation_dtype,
                sensitivity,
            )
        )
        if node.op == "cast":
            source = node.inputs[0].spec
            read_bytes = checked_size(
                source.size * source.itemsize, "cast source bytes"
            )
            write_bytes = checked_size(
                node.spec.size * node.spec.itemsize, "cast target bytes"
            )
            casts.append(
                CastBoundary(
                    names[node],
                    source.dtype,
                    node.spec.dtype,
                    node.spec.size,
                    read_bytes,
                    write_bytes,
                )
            )
    provenance = program.provenance
    source = provenance.get("precision_source_equation", program.logical_hash)
    if not isinstance(source, str):
        raise TypeError("precision_source_equation provenance must be a string")
    request_identity, scope = _request_scope(program, source)
    return PrecisionSchedule(
        source,
        program.logical_hash,
        tuple(values),
        tuple(casts),
        strict_audit_dtype,
        request_identity=request_identity,
        qualification_scope=scope,
        execution_scope=execution_scope,
        parent_schedule_identity=provenance.get("precision_parent_schedule_identity"),
    )


def derivative_precision_provenance(program: Program) -> dict[str, str]:
    """Retain a parent's precision/evidence identity without qualifying its AD.

    Strict programs without explicit precision provenance stay unchanged. A
    qualified primal or derived program carries a stable schedule fingerprint
    into generated AD and subsequent precision lowering. This is identity
    partitioning, not a claim that primal evidence qualifies a derivative.
    """
    provenance = program.provenance
    if any(
        provenance.get(key) is not None
        for key in (
            "precision_request",
            "precision_request_identity",
            "precision_parent_schedule_identity",
        )
    ):
        return {
            "precision_parent_schedule_identity": describe_precision(program).identity
        }
    return {}
