"""First-order implicit-solve rules expressed as ordinary TensorIR programs.

For R(x, q) = 0, solve J_x^* lambda = -bar_x and return
bar_q_direct + R_q^T W_R lambda. State cotangents use W_x; parameter
cotangents are Euclidean. A numerical solver is an explicit runtime callback,
not a differentiated iteration graph. No solver or native runtime is imported.
"""

from __future__ import annotations

import math
import typing
from dataclasses import dataclass, replace
from fractions import Fraction
from types import MappingProxyType
from typing import ClassVar

from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.tensor import (
    Program,
    TensorSpec,
    add,
    constant,
    input_tensor,
    linearize,
    multiply,
    transpose_program,
)
from vibeqc_compiler.tensor.ad_program import _rebuild

if typing.TYPE_CHECKING:
    from collections.abc import Mapping

SCHEMA = "vibeqc.method.implicit_solve"
VERSION = 1
PREFIX = "__implicit_"
RULE = "first-order-implicit-negative-adjoint-v1"
SOLVER_CONTRACT = "opaque-euclidean-transpose-true-residual-v1"


def _inputs(program: Program) -> dict:
    return {
        node.attrs["name"]: node for node in program.live_nodes if node.op == "input"
    }


def _metric(values: typing.Any, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be an immutable tuple")
    if values and len(values) != size:
        raise ValueError(f"{name} size does not match its independent coordinates")
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value <= 0
        for value in values
    ):
        raise ValueError(f"{name} must contain finite positive real weights")
    return tuple(float(value) for value in values)


def _scaled(
    node: typing.Any, weights: typing.Any, *, inverse: typing.Any = False
) -> typing.Any:
    """Emit square-root metric scaling, never a dense metric/Jacobian."""
    if not weights:
        return node
    values = tuple(
        1.0 / math.sqrt(value) if inverse else math.sqrt(value) for value in weights
    )
    factor = constant(
        tuple(Fraction(value) for value in values),
        replace(node.spec, role="constant", differentiable=False, symmetries=()),
    )
    return multiply(node, factor)


def _seed(name: str, spec: TensorSpec) -> typing.Any:
    return input_tensor(
        PREFIX + name, replace(spec, role="input", differentiable=False)
    )


def _substitute(program: Program, name: str, replacement: typing.Any) -> Program:
    # Reuse the AD frontend's primitive reconstruction, including its legality
    # checks. Multiple definitions of one named input must all be replaced.
    substitutions = {
        node: replacement
        for node in program.nodes
        if node.op == "input" and node.attrs["name"] == name
    }
    rebuilt = _rebuild(program, substitutions)
    return Program(rebuilt.outputs, provenance=program.provenance)


@dataclass(frozen=True)
class ImplicitSolveSpec:
    """Typed residual in explicitly independent real FP64 coordinates.

    Empty metric tuples mean Euclidean coordinates. Otherwise positive diagonal
    weights define the state/residual inner products. Packed callers provide
    their existing packing-map identity and orbit weights; the residual itself
    must already be expressed in independent coordinates. Redundant symmetric
    dense states are rejected, not silently flattened or gauge-fixed.
    """

    program: Program
    state_name: str
    parameter_names: tuple[str, ...]
    operator_identity: str
    residual_name: str = "residual"
    state_layout: str = "independent-c-order"
    residual_layout: str = "independent-c-order"
    gauge: str = "nonredundant"
    state_metric: tuple[float, ...] = ()
    residual_metric: tuple[float, ...] = ()
    kind: ClassVar[str] = "implicit_solve"

    def __post_init__(self) -> None:
        if not isinstance(self.program, Program):
            raise TypeError("implicit residual must be a TensorIR Program")
        if set(self.program.outputs) != {self.residual_name}:
            raise ValueError("implicit program must expose exactly its named residual")
        for name in ("operator_identity", "state_layout", "residual_layout", "gauge"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty identity")
        if not isinstance(self.parameter_names, tuple) or not self.parameter_names:
            raise TypeError("parameter_names must be a nonempty immutable tuple")
        if any(not isinstance(name, str) for name in self.parameter_names):
            raise TypeError("parameter names must be strings")
        if len(set(self.parameter_names)) != len(self.parameter_names):
            raise ValueError("duplicate implicit parameter name")
        if self.state_name in self.parameter_names:
            raise ValueError("implicit state cannot also be a parameter")
        inputs = _inputs(self.program)
        if any(name.startswith(PREFIX) for name in inputs):
            raise ValueError("residual uses a reserved implicit input name")
        for name in (self.state_name, *self.parameter_names):
            if name not in inputs or not inputs[name].spec.differentiable:
                raise ValueError(
                    f"{name!r} must be a live differentiable residual input"
                )
            if inputs[name].spec.symmetries:
                raise NotImplementedError(
                    "implicit inputs require explicit independent coordinates; "
                    "reduce packed/symmetric inputs through their declared map"
                )
        if any(node.spec.dtype != "float64" for node in self.program.live_nodes):
            raise NotImplementedError(
                "implicit-solve first rule supports real FP64 only"
            )
        residual = self.program.outputs[self.residual_name].spec
        state = inputs[self.state_name].spec
        if residual.symmetries:
            raise NotImplementedError(
                "implicit residual must use independent coordinates"
            )
        if state.size != residual.size or state.size < 1:
            raise ValueError(
                "implicit state and residual must have equal nonzero dimension"
            )
        for name, size in (
            ("state_metric", state.size),
            ("residual_metric", residual.size),
        ):
            object.__setattr__(self, name, _metric(getattr(self, name), size, name))
        object.__setattr__(self, "parameter_names", tuple(sorted(self.parameter_names)))

    @property
    def inputs(self) -> Mapping[str, TensorSpec]:
        return MappingProxyType(
            {name: node.spec for name, node in _inputs(self.program).items()}
        )

    @property
    def state_spec(self) -> TensorSpec:
        return self.inputs[self.state_name]

    @property
    def residual_spec(self) -> TensorSpec:
        return self.program.outputs[self.residual_name].spec

    @property
    def dimension(self) -> int:
        return self.state_spec.size

    @property
    def identity(self) -> str:
        # Provenance is documentary, not mathematical. The residual equation
        # hash already includes input, layout and primitive semantics.
        return canonical_hash(
            {
                **self._contract(),
                "residual_equation": self.program.logical_hash,
            }
        )

    def _contract(self) -> dict:
        return {
            "schema": SCHEMA,
            "version": VERSION,
            "kind": self.kind,
            "state_name": self.state_name,
            "parameter_names": list(self.parameter_names),
            "residual_name": self.residual_name,
            "operator_identity": self.operator_identity,
            "state_layout": self.state_layout,
            "residual_layout": self.residual_layout,
            "gauge": self.gauge,
            "state_metric": list(self.state_metric),
            "residual_metric": list(self.residual_metric),
            "parameter_metric": "euclidean",
            "rule": RULE,
            "derivative_orders": [1],
        }

    def to_payload(self) -> dict:
        return {**self._contract(), "program": self.program.to_payload()}

    @classmethod
    def from_payload(cls, payload: dict) -> ImplicitSolveSpec:
        if not isinstance(payload, dict):
            raise TypeError("implicit payload must be an object")
        fields = {
            "program",
            "state_name",
            "parameter_names",
            "operator_identity",
            "residual_name",
            "state_layout",
            "residual_layout",
            "gauge",
            "state_metric",
            "residual_metric",
        }
        if set(payload) != fields | {
            "schema",
            "version",
            "kind",
            "parameter_metric",
            "rule",
            "derivative_orders",
        }:
            raise ValueError("implicit payload has missing or unknown fields")
        arguments = {name: payload[name] for name in fields}
        arguments["program"] = Program.from_payload(arguments["program"])
        for name in ("parameter_names", "state_metric", "residual_metric"):
            if not isinstance(arguments[name], list):
                raise TypeError(f"serialized {name} must be an array")
            arguments[name] = tuple(arguments[name])
        spec = cls(**arguments)
        if canonical_hash(payload) != canonical_hash(spec.to_payload()):
            raise ValueError("unsupported or noncanonical implicit derivative contract")
        return spec

    def compile(self) -> ImplicitVJPPlan:
        """Generate one first-order rule; this does not grant higher derivatives."""
        return ImplicitVJPPlan(self)


@dataclass(frozen=True, init=False)
class ImplicitVJPPlan:
    """Generated operator/RHS/source programs around an opaque transpose solve.

    In Euclidean solver coordinates y = sqrt(W_R) lambda, the callback solves
    [inv_sqrt(W_x) J^T sqrt(W_R)] y = -sqrt(W_x) bar_x.
    No callback is serialized. Runtime binds its contract and current state.
    """

    spec: ImplicitSolveSpec
    programs: Mapping[str, Program]
    identity: str

    def __init__(self, spec: ImplicitSolveSpec) -> None:
        if not isinstance(spec, ImplicitSolveSpec):
            raise TypeError("implicit plan requires ImplicitSolveSpec")
        state, residual = spec.state_spec, spec.residual_spec
        dx = _seed("vector", state)
        dy = _seed("vector", residual)
        seed = _seed("seed", state)
        forward = linearize(spec.program, [spec.state_name])
        forward = _substitute(
            forward.program,
            f"d_{spec.state_name}",
            _scaled(dx, spec.state_metric, inverse=True),
        )
        reverse = transpose_program(
            spec.program,
            [spec.residual_name],
            inputs=[spec.state_name, *spec.parameter_names],
        )
        reverse = _substitute(
            reverse.program,
            f"bar_{spec.residual_name}",
            _scaled(dy, spec.residual_metric),
        )
        roots = {
            "primal": {
                "value": _scaled(
                    spec.program.outputs[spec.residual_name], spec.residual_metric
                )
            },
            "jacobian": {
                "value": _scaled(
                    forward.outputs[f"d_{spec.residual_name}"], spec.residual_metric
                )
            },
            "transpose": {
                "value": _scaled(
                    reverse.outputs[f"bar_{spec.state_name}"],
                    spec.state_metric,
                    inverse=True,
                )
            },
            "rhs": {"value": add(_scaled(seed, spec.state_metric), coefficients=(-1,))},
            "adjoint": {"value": _scaled(dy, spec.residual_metric, inverse=True)},
            "source": {
                f"bar_{name}": add(
                    reverse.outputs[f"bar_{name}"],
                    _seed(f"direct_{name}", spec.inputs[name]),
                )
                for name in spec.parameter_names
            },
        }
        programs = {
            name: Program(
                outputs,
                provenance={
                    "implicit_spec": spec.identity,
                    "rule": RULE,
                    "stage": name,
                },
            )
            for name, outputs in roots.items()
        }
        identity = canonical_hash(
            {
                "spec": spec.identity,
                "programs": {
                    name: program.logical_hash for name, program in programs.items()
                },
                "solver_coordinates": "sqrt-weighted-euclidean",
                "solver_contract": SOLVER_CONTRACT,
                "derivative_orders": [1],
            }
        )
        object.__setattr__(self, "spec", spec)
        object.__setattr__(self, "programs", MappingProxyType(programs))
        object.__setattr__(self, "identity", identity)

    @property
    def reference_workspace_bytes(self) -> int:
        """Conservative simultaneous logical arrays, excluding solver storage.

        This reference admission is NOT a native peak/RSS claim. Opaque
        NumPy/BLAS temporaries and Python objects are outside its scope.
        Runtime adds the separately declared solver/executor reservations.
        """
        retained = sum(
            value.size * value.itemsize for value in self.spec.inputs.values()
        )
        parameters = sum(
            self.spec.inputs[name].size * 8 for name in self.spec.parameter_names
        )
        graph = max(
            sum(node.spec.size * node.spec.itemsize for node in program.live_nodes)
            + sum(
                node.spec.size * node.spec.itemsize for node in program.outputs.values()
            )
            for program in self.programs.values()
        )
        return 2 * retained + 4 * parameters + 16 * self.spec.dimension * 8 + graph

    def to_payload(self) -> dict:
        return {
            "schema": "vibeqc.method.implicit_vjp_plan",
            "version": VERSION,
            "solver_contract": SOLVER_CONTRACT,
            "spec": self.spec.to_payload(),
            "programs": {
                name: program.to_payload() for name, program in self.programs.items()
            },
            "identity": self.identity,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> ImplicitVJPPlan:
        if not isinstance(payload, dict) or set(payload) != {
            "schema",
            "version",
            "spec",
            "programs",
            "identity",
            "solver_contract",
        }:
            raise ValueError("invalid implicit plan payload fields")
        plan = cls(ImplicitSolveSpec.from_payload(payload["spec"]))
        # Regeneration rejects substituted/stale derivative programs and rules.
        if canonical_hash(payload) != canonical_hash(plan.to_payload()):
            raise ValueError(
                "implicit plan does not match its generated derivative rule"
            )
        return plan
