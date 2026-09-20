"""Resolved precision schedules and explicit cast-cost provenance.

This module describes precision semantics already present in a TensorIR
program. It never silently rewrites an equation or promotes a low-precision
candidate. Method controllers may generate explicit cast DAGs and use this
schedule as the stable compiler/audit identity consumed by the existing
planner, tuner, cache, and evidence pipeline.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

from .program import Program, _hash
from .types import checked_size

DTYPES = frozenset(("float32", "float64"))
STRICT_MATH_MODE = "ieee-rn-no-tf32"
REDUCTION_OPS = frozenset(("reduce", "einsum"))
SENSITIVE_OPS = frozenset(("divide", "scaled_bilinear", "log", "sqrt", "power"))


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
        if self.sensitivity not in ("ordinary", "reduction", "sensitive", "cast"):
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

    def __post_init__(self) -> None:
        _dtype(self.strict_audit_dtype, "strict audit dtype")
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
        return checked_size(sum(cast.read_bytes for cast in self.casts), "cast read bytes")

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
            "schema": "vibeqc.tensor.precision-schedule.v1",
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


def describe_precision(
    program: Program, *, strict_audit_dtype: str = "float64"
) -> PrecisionSchedule:
    """Resolve dtype/sensitivity/cast facts without inventing a promotion policy."""
    if not isinstance(program, Program):
        raise TypeError("precision scheduling requires a TensorIR Program")
    _dtype(strict_audit_dtype, "strict audit dtype")
    names = program.debug_names
    values = []
    casts = []
    for node in program.live_nodes:
        if node.op == "cast":
            sensitivity = "cast"
        elif node.op in REDUCTION_OPS:
            sensitivity = "reduction"
        elif node.op in SENSITIVE_OPS:
            sensitivity = "sensitive"
        else:
            sensitivity = "ordinary"
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
        raise ValueError("precision_source_equation provenance must be a string")
    return PrecisionSchedule(
        source,
        program.logical_hash,
        tuple(values),
        tuple(casts),
        strict_audit_dtype,
    )
