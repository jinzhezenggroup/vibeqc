"""Scientific admission for experimental selective-precision XC arithmetic.

This module classifies an already-built XC scalar DAG.  It does not change the
functional, insert casts, or promote mixed precision into production.  The
compiler may consume the result as a candidate schedule only after the method
owner supplies independent numerical and endpoint evidence.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

from vibeqc_compiler.common.provenance import canonical_hash

from .program import XCProgram

SCHEMA = "vibeqc.xc.precision-admission.v1"
STRICT_MATH_MODE = "ieee-rn-no-tf32"
_LEAF_OPS = frozenset(("constant", "variable"))
_CANDIDATE_OPS = frozenset(("add", "multiply"))
_SENSITIVE_OPS = frozenset(
    (
        "reciprocal",
        "power",
        "exp",
        "expm1",
        "log",
        "log1p",
        "atan",
        "asinh",
        "erf",
        "select_le",
    )
)
_KNOWN_COMPUTE_OPS = _CANDIDATE_OPS | _SENSITIVE_OPS


@dataclass(frozen=True, slots=True)
class XCNodePrecision:
    """One reachable scalar operation's admitted compute precision."""

    identifier: int
    operation: str
    compute_dtype: str
    reason: str

    def __post_init__(self) -> None:
        if self.compute_dtype not in ("float32", "float64"):
            raise ValueError("XC compute dtype must be float32 or float64")
        if not self.operation or not self.reason:
            raise ValueError("XC precision entries require operation and reason")

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "identifier": self.identifier,
            "operation": self.operation,
            "compute_dtype": self.compute_dtype,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class XCPrecisionAdmission:
    """Immutable scientific admission for one XC expression/observable."""

    expression_hash: str
    functional: str
    functional_identity: str
    spin: str
    observable: str
    values: tuple[XCNodePrecision, ...]
    reasons: tuple[str, ...] = ()
    storage_dtype: str = "float64"
    reduction_dtype: str = "float64"
    strict_audit_dtype: str = "float64"
    math_mode: str = STRICT_MATH_MODE

    def __post_init__(self) -> None:
        if self.observable not in ("energy", "potential", "response", "geometry"):
            raise ValueError("unsupported XC precision observable")
        if (
            self.storage_dtype != "float64"
            or self.reduction_dtype != "float64"
            or self.strict_audit_dtype != "float64"
        ):
            raise ValueError("XC selective-precision admission keeps FP64 boundaries")
        if self.math_mode != STRICT_MATH_MODE:
            raise ValueError("XC precision admission forbids implicit fast math/TF32")
        if len({value.identifier for value in self.values}) != len(self.values):
            raise ValueError("XC precision admission contains duplicate nodes")

    @property
    def enabled(self) -> bool:
        """Whether this admission contains any lower-precision compute candidate."""

        return any(value.compute_dtype == "float32" for value in self.values)

    @property
    def candidate_nodes(self) -> tuple[int, ...]:
        return tuple(
            value.identifier
            for value in self.values
            if value.compute_dtype == "float32"
        )

    @property
    def strict_nodes(self) -> tuple[int, ...]:
        return tuple(
            value.identifier
            for value in self.values
            if value.compute_dtype == "float64"
        )

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "schema": SCHEMA,
            "expression_hash": self.expression_hash,
            "functional": self.functional,
            "functional_identity": self.functional_identity,
            "spin": self.spin,
            "observable": self.observable,
            "storage_dtype": self.storage_dtype,
            "reduction_dtype": self.reduction_dtype,
            "strict_audit_dtype": self.strict_audit_dtype,
            "math_mode": self.math_mode,
            "values": [value.to_payload() for value in self.values],
            "reasons": list(self.reasons),
            "promotion": "requires-independent-numerical-and-endpoint-evidence",
        }

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())


def _strict_admission(
    program: XCProgram, observable: str, reason: str
) -> XCPrecisionAdmission:
    reachable = program.graph.topological_order(program.roots)
    values = tuple(
        XCNodePrecision(identifier, node.operation, "float64", reason)
        for identifier in reachable
        if (node := program.graph.nodes[identifier]).operation not in _LEAF_OPS
    )
    return XCPrecisionAdmission(
        expression_hash=program.expression_hash,
        functional=program.spec.identifier,
        functional_identity=program.spec.identity,
        spin=program.spec.spin,
        observable=observable,
        values=values,
        reasons=(reason,),
    )


def admit_selective_precision(
    program: XCProgram, *, observable: str
) -> XCPrecisionAdmission:
    """Classify a conservative PBE FP32-compute candidate with FP64 boundaries.

    Sensitive operations and every arithmetic dependency feeding them stay
    FP64.  Final requested roots stay FP64 as the local output boundary.  Only
    remaining add/multiply nodes are candidates; this is not a promotion claim.
    """

    if not isinstance(program, XCProgram):
        raise TypeError("XC precision admission requires an XCProgram")
    if observable not in ("energy", "potential", "response", "geometry"):
        raise ValueError("unsupported XC precision observable")

    expected_order = {"energy": 0, "potential": 1}.get(observable)
    if program.spec.identifier != "PBE":
        return _strict_admission(
            program,
            observable,
            f"{program.spec.identifier} has no selective-precision qualification",
        )
    if expected_order is None or program.order != expected_order:
        return _strict_admission(
            program,
            observable,
            "selective precision is qualified only for PBE energy/potential DAGs",
        )

    graph = program.graph
    reachable = graph.topological_order(program.roots)
    compute = [
        identifier
        for identifier in reachable
        if graph.nodes[identifier].operation not in _LEAF_OPS
    ]
    unknown = sorted(
        {
            graph.nodes[identifier].operation
            for identifier in compute
            if graph.nodes[identifier].operation not in _KNOWN_COMPUTE_OPS
        }
    )
    if unknown:
        return _strict_admission(
            program,
            observable,
            "unclassified XC operations: " + ", ".join(unknown),
        )

    sensitive = {
        identifier
        for identifier in compute
        if graph.nodes[identifier].operation in _SENSITIVE_OPS
    }

    strict = set(sensitive)
    stack = list(sensitive)
    while stack:
        identifier = stack.pop()
        for argument in graph.nodes[identifier].arguments:
            if argument not in strict:
                strict.add(argument)
                stack.append(argument)

    roots = {root.identifier for root in program.roots}
    strict.update(roots)

    values: list[XCNodePrecision] = []
    for identifier in compute:
        node = graph.nodes[identifier]
        if identifier not in strict and node.operation in _CANDIDATE_OPS:
            dtype, reason = "float32", "ordinary-arithmetic-candidate"
        elif identifier in sensitive:
            dtype, reason = "float64", "sensitive-operation"
        elif identifier in roots:
            dtype, reason = "float64", "requested-output-root"
        else:
            dtype, reason = "float64", "feeds-sensitive-operation"
        values.append(XCNodePrecision(identifier, node.operation, dtype, reason))

    admission = XCPrecisionAdmission(
        expression_hash=program.expression_hash,
        functional=program.spec.identifier,
        functional_identity=program.spec.identity,
        spin=program.spec.spin,
        observable=observable,
        values=tuple(values),
    )
    if admission.enabled:
        return admission
    return _strict_admission(
        program,
        observable,
        "sensitivity closure leaves no lower-precision arithmetic candidate",
    )
