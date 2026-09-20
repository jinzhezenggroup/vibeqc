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

from .ir import TRANSCENDENTALS, Node, cast
from .program import Program, _hash
from .types import checked_size

DTYPES = frozenset(("float32", "float64"))
STRICT_MATH_MODE = "ieee-rn-no-tf32"
REDUCTION_OPS = frozenset(("reduce", "einsum", "scatter_add", "segment_sum"))
SENSITIVE_OPS = frozenset(("divide", "scaled_bilinear")) | TRANSCENDENTALS
AUTO_FP32_OPS = frozenset(
    ("add", "multiply", "transpose", "reshape", "slice", "gather", "broadcast")
)


def _dtype(value: typing.Any, label: str) -> str:
    if value not in DTYPES:
        raise ValueError(f"{label} must be float32 or float64")
    return value


@dataclass(frozen=True)
class PrecisionDirective:
    """Requested per-value storage, compute, and accumulation precision."""

    storage_dtype: str
    compute_dtype: str
    accumulation_dtype: str
    qualification: str | None = None
    math_mode: str = STRICT_MATH_MODE

    def __post_init__(self) -> None:
        for label in ("storage_dtype", "compute_dtype", "accumulation_dtype"):
            _dtype(getattr(self, label), label)
        if self.math_mode != STRICT_MATH_MODE:
            raise ValueError("unsupported TensorIR arithmetic mode")
        if self.qualification is not None and (
            not isinstance(self.qualification, str) or not self.qualification.strip()
        ):
            raise ValueError("precision qualification must be a nonempty string")

    def to_payload(self) -> dict:
        return {
            "storage_dtype": self.storage_dtype,
            "compute_dtype": self.compute_dtype,
            "accumulation_dtype": self.accumulation_dtype,
            "qualification": self.qualification,
            "math_mode": self.math_mode,
        }


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
        if self.qualification_scope and self.request_identity is None:
            raise ValueError(
                "precision qualification scope requires a request identity"
            )
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

    def to_payload(self) -> dict:
        return {
            "schema": "vibeqc.tensor.precision-schedule.v2",
            "precision_request_identity": self.request_identity,
            "parent_precision_schedule_identity": self.parent_schedule_identity,
            "qualification_scope": [
                {"source_value": name, "qualification": qualification}
                for name, qualification in self.qualification_scope
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


def _ensure_dtype(node: Node, dtype: str) -> Node:
    return node if node.spec.dtype == dtype else cast(node, dtype)


def lower_precision(
    program: Program,
    directives: Mapping[str, PrecisionDirective],
    *,
    strict_audit_dtype: str = "float64",
) -> Program:
    """Lower explicit per-value precision requests into a typed cast DAG.

    External input/output dtypes remain unchanged. Current ordinary-stream
    backends require compute and accumulation dtype to match; unsupported
    combinations fail closed rather than being silently approximated.
    Sensitive/reduction FP32 requests require an external qualification id.
    """
    if not isinstance(program, Program):
        raise TypeError("precision lowering requires a TensorIR Program")
    if not isinstance(directives, Mapping):
        raise TypeError("precision directives must be a mapping")
    _dtype(strict_audit_dtype, "strict audit dtype")
    names = program.debug_names
    live = {names[node]: node for node in program.live_nodes}
    normalized: dict[str, PrecisionDirective] = {}
    for name, directive in directives.items():
        if not isinstance(name, str) or name not in live:
            raise ValueError(f"unknown live precision value: {name!r}")
        if not isinstance(directive, PrecisionDirective):
            raise TypeError("precision directive values must be PrecisionDirective")
        node = live[name]
        if node.op in ("input", "constant", "cast"):
            raise ValueError("precision directives target computed non-cast values")
        if directive.compute_dtype != directive.accumulation_dtype:
            raise ValueError(
                "separate compute/accumulation dtype lowering is not qualified"
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
        inputs = tuple(
            ensure_dtype(mapping[child], compute_dtype) for child in node.inputs
        )
        declared = replace(
            node.spec,
            dtype=compute_dtype,
            role="intermediate",
        )
        rebuilt = Node(node.op, inputs, declared, node.attributes)
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
    return Program(
        outputs,
        tuple(mapping[node] for node in program.definitions),
        provenance={
            **program.provenance,
            "precision_source_equation": source_equation,
            "precision_request": request,
            "precision_request_identity": _hash(request),
        },
    )


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
    values = []
    casts = []
    for node in program.live_nodes:
        sensitivity = _sensitivity(node.op)
        values.append(
            ValuePrecision(
                names[node],
                node.op,
                node.spec.dtype,
                node.spec.dtype,
                node.spec.dtype,
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
