"""Versioned scientific IR payloads, separate from the unchanged CUDA ABI.

Decoding is strict at every record: an old or unknown schema/layout must not
silently acquire today's defaults and then alias an existing compiled cache.
Legacy production profiles continue to use their existing schema and loader.
"""

from .blocks import RawBlock, TensorLayout, WeightDescriptor, WeightedDerivative
from .ir import (
    ContractionSpec,
    DerivativeSpec,
    IntegralIR,
    NuclearCenter,
    NuclearCoordinates,
    OperatorSpec,
    TranslationInvariant,
)
from .shell_signature import BasisShell, CenterBinding, ShellSignature
from .shell_spec import ShellClassSpec

INTEGRAL_SCHEMA_VERSION = 1
INTEGRAL_SCHEMA = "vibeqc.integral_ir"


def _record(payload, fields):
    if not isinstance(payload, dict):
        raise TypeError("IR record must be an object")
    unknown = set(payload) - set(fields)
    missing = set(fields) - set(payload)
    if unknown or missing:
        raise ValueError(
            f"IR record has unknown fields {sorted(unknown)} or missing fields {sorted(missing)}"
        )
    return payload


def _invariant_payload(invariant):
    centers = invariant.parameters.centers
    return {
        "centers": centers if centers == "all" else list(centers),
        "dependent_center": invariant.dependent_center,
    }


def _coordinates(payload):
    return NuclearCoordinates(payload if payload == "all" else tuple(payload))


def _invariant(payload):
    _record(payload, ("centers", "dependent_center"))
    return TranslationInvariant(
        _coordinates(payload["centers"]), payload["dependent_center"]
    )


def _layout(payload):
    _record(payload, ("indices", "shape", "strides", "dtype"))
    if payload["dtype"] != "float64":
        raise ValueError("unsupported tensor scalar type; expected float64")
    return TensorLayout(
        tuple(payload["indices"]), tuple(payload["shape"]), tuple(payload["strides"])
    )


def _consumer_payload(consumer):
    if isinstance(consumer, ContractionSpec):
        return {
            "consumer": consumer.consumer.value,
            "density": sorted(d.value for d in consumer.density),
            "output": consumer.output.value,
        }
    if isinstance(consumer, RawBlock):
        return {
            "consumer": consumer.consumer,
            "layout": consumer.layout.to_payload(),
            "memory_budget_bytes": consumer.memory_budget_bytes,
            "output_sign": consumer.output_sign,
        }
    return {
        "consumer": consumer.consumer,
        "weights": {
            "source": consumer.weights.source,
            "layout": consumer.weights.layout.to_payload(),
            "sign": consumer.weights.sign,
            "prefactor": consumer.weights.prefactor,
        },
        "output_layout": consumer.output_layout.to_payload(),
        "output": consumer.output.value,
        "memory_budget_bytes": consumer.memory_budget_bytes,
        "output_sign": consumer.output_sign,
    }


def _consumer(payload):
    if not isinstance(payload, dict) or "consumer" not in payload:
        raise ValueError("consumer record requires a consumer tag")
    kind = payload["consumer"]
    if kind in ("direct_fock", "direct_force"):
        _record(payload, ("consumer", "density", "output"))
        return ContractionSpec(kind, tuple(payload["density"]), payload["output"])
    if kind == "raw_block":
        _record(payload, ("consumer", "layout", "memory_budget_bytes", "output_sign"))
        return RawBlock(
            _layout(payload["layout"]),
            payload["memory_budget_bytes"],
            payload["output_sign"],
        )
    if kind != "weighted_derivative":
        raise ValueError(f"unknown consumer tag {kind!r}")
    _record(
        payload,
        (
            "consumer",
            "weights",
            "output_layout",
            "output",
            "memory_budget_bytes",
            "output_sign",
        ),
    )
    w = _record(payload["weights"], ("source", "layout", "sign", "prefactor"))
    return WeightedDerivative(
        WeightDescriptor(w["source"], _layout(w["layout"]), w["sign"], w["prefactor"]),
        _layout(payload["output_layout"]),
        payload["memory_budget_bytes"],
        payload["output"],
        payload["output_sign"],
    )


def integral_to_payload(integral: IntegralIR) -> dict[str, object]:
    """Serialize scientific intent deterministically without executable callbacks."""
    signature = integral.signature
    if isinstance(integral.spec, ShellClassSpec):
        spec = {
            "kind": "shell_class",
            "name": integral.spec.name,
            "angular": list(integral.spec.angular),
        }
    else:
        spec = {
            "kind": "shell_signature",
            "legacy_class": signature.legacy_class,
            "shells": [
                {
                    "slot": s.slot,
                    "center": s.center,
                    "angular": s.angular,
                    "role": s.role.value,
                    "convention": s.convention.value,
                }
                for s in signature.shells
            ],
            "center_bindings": [
                {"center": b.center, "atom_index": b.atom_index}
                for b in signature.center_bindings
            ],
        }
    operator = integral.operator
    derivative = integral.derivative
    return {
        "schema": INTEGRAL_SCHEMA,
        "schema_version": INTEGRAL_SCHEMA_VERSION,
        "spec": spec,
        "operator": {
            "family": operator.family.value,
            "centers": list(operator.centers),
            "invariants": [_invariant_payload(i) for i in operator.invariants],
            "external_centers": [
                {"center": c.center, "charge": c.charge}
                for c in operator.external_centers
            ],
            "permutations": [list(p) for p in operator.permutations],
        },
        "derivative": None
        if derivative is None
        else {
            "order": derivative.order,
            "centers": derivative.parameters.centers
            if derivative.parameters.centers == "all"
            else list(derivative.parameters.centers),
            "invariants": [_invariant_payload(i) for i in derivative.invariants],
        },
        "contractions": [_consumer_payload(c) for c in integral.contractions],
        "recurrence": integral.recurrence,
    }


def integral_from_payload(payload: dict[str, object]) -> IntegralIR:
    """Reject unknown schemas/fields rather than guessing an ABI or layout."""
    _record(
        payload,
        (
            "schema",
            "schema_version",
            "spec",
            "operator",
            "derivative",
            "contractions",
            "recurrence",
        ),
    )
    if (
        payload["schema"] != INTEGRAL_SCHEMA
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != INTEGRAL_SCHEMA_VERSION
    ):
        raise ValueError("unsupported integral IR schema")
    s = payload["spec"]
    if not isinstance(s, dict):
        raise TypeError("shell specification must be an object")
    if s.get("kind") == "shell_class":
        _record(s, ("kind", "name", "angular"))
        spec = ShellClassSpec(s["name"], tuple(s["angular"]))
    elif s.get("kind") == "shell_signature":
        _record(s, ("kind", "legacy_class", "shells", "center_bindings"))
        shells = tuple(
            BasisShell(
                **_record(x, ("slot", "center", "angular", "role", "convention"))
            )
            for x in s["shells"]
        )
        bindings = tuple(
            CenterBinding(**_record(x, ("center", "atom_index")))
            for x in s["center_bindings"]
        )
        spec = ShellSignature(shells, bindings, s["legacy_class"])
    else:
        raise ValueError("unknown shell specification kind")
    o = _record(
        payload["operator"],
        ("family", "centers", "invariants", "external_centers", "permutations"),
    )
    operator = OperatorSpec(
        o["family"],
        tuple(o["centers"]),
        tuple(_invariant(i) for i in o["invariants"]),
        tuple(
            NuclearCenter(**_record(c, ("center", "charge")))
            for c in o["external_centers"]
        ),
        tuple(tuple(p) for p in o["permutations"]),
    )
    d = payload["derivative"]
    derivative = None
    if d is not None:
        _record(d, ("order", "centers", "invariants"))
        derivative = DerivativeSpec(
            d["order"],
            _coordinates(d["centers"]),
            tuple(_invariant(i) for i in d["invariants"]),
        )
    return IntegralIR(
        spec,
        operator,
        derivative,
        tuple(_consumer(c) for c in payload["contractions"]),
        payload["recurrence"],
    )
