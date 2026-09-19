"""StationaryProblem composition and source-boundary plans (#181 A).

The mathematical cut is explicit: E(x,q), R(x,q)=0 and L=E+lambda.R.
TensorIR owns differentiation; #465 owns implicit solves and live state binding.
Provider edges are an auditable pullback *inventory*, not executable derivatives.
This module neither solves a stationary problem nor publishes molecular forces.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from graphlib import CycleError, TopologicalSorter
from types import MappingProxyType

from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.tensor import (
    Program,
    add,
    constant,
    input_tensor,
    multiply,
    reduce_sum,
    transpose_program,
)

from .implicit import ImplicitSolveSpec, ImplicitVJPPlan

SCHEMA = "vibeqc.stationary_problem"
VERSION = 2
PLAN_VERSION = 2
INNER_PRODUCT = "real-euclidean-independent-v1"
CONVENTION = "L=E+lambda^T R; R_x^T lambda=-E_x; weights=E_q+R_q^T lambda"
_PREFIX = "stationary_"


def _identifier(value, label):
    if not isinstance(value, str) or not value.isidentifier():
        raise ValueError(f"{label} must be an identifier")
    if value.startswith(_PREFIX):
        raise ValueError(f"{label} uses reserved prefix {_PREFIX}")


def _identity(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty identity")


def _names(values, label):
    if isinstance(values, str):
        raise TypeError(f"{label} must be a sequence, not a string")
    values = tuple(values)
    for value in values:
        _identifier(value, label)
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {label}")
    return tuple(sorted(values))


def _inputs(program):
    return {
        node.attrs["name"]: node for node in program.live_nodes if node.op == "input"
    }


def _dependencies(node):
    """Actual SSA input dependencies, never a caller's guessed equation list."""
    return tuple(sorted(_inputs(Program({"value": node}))))


def _record(record_type, payload):
    if not isinstance(payload, dict) or set(payload) != {
        f.name for f in fields(record_type)
    }:
        raise ValueError(f"invalid {record_type.__name__} fields")
    return record_type(**payload)


def _load_json(source):
    def unique(pairs):
        result = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"duplicate JSON field: {name}")
            result[name] = value
        return result

    return json.loads(source, object_pairs_hook=unique)


@dataclass(frozen=True)
class StationaryState:
    """One independent state block and one equally sized physical equation.

    A KKT multiplier is a state too: bind it to its constraint residual with
    ``kind="constraint"``. The other blocks must explicitly contain the KKT
    stationarity equations. Neither constraints nor gauges are inferred from E.
    Redundant symmetric/packed state coordinates need a separately qualified
    mapping and are rejected by this first composition boundary.
    """

    name: str
    residual: str
    coordinate_identity: str
    gauge_identity: str
    kind: str = "physical"
    inner_product: str = INNER_PRODUCT
    implicit_operator_identity: str | None = None
    residual_layout_identity: str | None = None

    def __post_init__(self):
        _identifier(self.name, "state name")
        _identifier(self.residual, "residual output")
        _identity(self.coordinate_identity, "state coordinate")
        _identity(self.gauge_identity, "state gauge")
        if self.kind not in ("physical", "constraint"):
            raise ValueError("state equation kind must be physical or constraint")
        if self.inner_product != INNER_PRODUCT:
            raise ValueError("unsupported state inner product; no implicit flattening")
        if self.implicit_operator_identity is None:
            if self.residual_layout_identity is not None:
                raise ValueError(
                    "residual layout identity requires an implicit operator identity"
                )
        else:
            _identity(self.implicit_operator_identity, "implicit operator")
            layout = (
                self.coordinate_identity
                if self.residual_layout_identity is None
                else self.residual_layout_identity
            )
            _identity(layout, "residual layout")
            object.__setattr__(self, "residual_layout_identity", layout)


@dataclass(frozen=True)
class ParameterSource:
    """One physical source field; its name binds a TensorIR input when present.

    Ancestors (for example geometry upstream of overlap) need not be TensorIR
    inputs. ``identity`` identifies a physical *field*, not just its provider:
    aliasing the same field under two input names would double count its weight.
    A derived field must name its pullback contract. Naming a contract does not
    register or execute that custom rule; generated weights stop at input fields.
    """

    name: str
    identity: str
    dependencies: tuple[str, ...] = ()
    pullback_identity: str | None = None

    def __post_init__(self):
        _identifier(self.name, "parameter source")
        _identity(self.identity, "parameter source")
        dependencies = _names(self.dependencies, "source dependency")
        object.__setattr__(self, "dependencies", dependencies)
        if dependencies:
            _identity(self.pullback_identity, "derived source pullback")
        elif self.pullback_identity is not None:
            raise ValueError("root source cannot declare a provider pullback")


@dataclass(frozen=True)
class StationaryProblem:
    """Immutable method-level schema over existing TensorIR equations.

    Sources exactly cover all non-state live inputs and their provider ancestors.
    Every equation output is classified as objective or a declared residual.
    This catches missing declared sources and constraints, but cannot discover
    physics omitted from BOTH the manifest and the equations. A declared gauge
    and square residual layout do not certify invertibility or convergence.
    """

    equations: Program
    objective: str
    states: tuple[StationaryState, ...]
    sources: tuple[ParameterSource, ...]
    model_identity: str
    solver_contract: str

    def __post_init__(self):
        if not isinstance(self.equations, Program):
            raise TypeError("stationary equations must be a TensorIR Program")
        _identifier(self.objective, "objective output")
        _identity(self.model_identity, "stationary model")
        _identity(self.solver_contract, "implicit solver contract")
        states, sources = tuple(self.states), tuple(self.sources)
        if not states or any(not isinstance(s, StationaryState) for s in states):
            raise ValueError("stationary problem requires typed state blocks")
        if any(not isinstance(s, ParameterSource) for s in sources):
            raise TypeError("stationary sources must be ParameterSource records")
        states = tuple(sorted(states, key=lambda s: s.name))
        sources = tuple(sorted(sources, key=lambda s: s.name))
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "sources", sources)
        names = _names((s.name for s in states), "state name")
        residuals = _names((s.residual for s in states), "residual output")
        source_names = _names((s.name for s in sources), "parameter source")
        if set(names) & set(source_names):
            raise ValueError("state and parameter source names must be disjoint")
        if len({s.identity for s in sources}) != len(sources):
            raise ValueError("duplicate physical source identity; use one input field")
        if self.objective in residuals or set(self.equations.outputs) != {
            self.objective,
            *residuals,
        }:
            raise ValueError("outputs must be exactly objective and declared residuals")
        if self.equations.outputs[self.objective].spec.shape != ():
            raise ValueError("stationary objective must be scalar")
        inputs = _inputs(self.equations)
        for name in (*inputs, *self.equations.outputs):
            _identifier(name, "equation input/output")
        if any(n.spec.dtype != "float64" for n in self.equations.live_nodes):
            raise ValueError("stationary composition currently requires float64")
        for state in states:
            if state.name not in inputs or not inputs[state.name].spec.differentiable:
                raise ValueError("every state must be a live differentiable input")
            spec = inputs[state.name].spec
            if spec.symmetries or not spec.size:
                raise ValueError("states require nonempty independent coordinates")
            residual = self.equations.outputs[state.residual]
            if residual.spec.signature != spec.signature:
                raise ValueError("residual and state coordinate semantics must match")
            if not set(_dependencies(residual)) & set(names):
                raise ValueError("each residual must depend on a declared state")
        for name in set(inputs) - set(names):
            if name not in source_names:
                raise ValueError(f"missing parameter source: {name}")
        source_map = {s.name: s for s in sources}
        for source in sources:
            if not set(source.dependencies) <= set(source_names):
                raise ValueError(
                    "missing provider dependency or undeclared implicit cycle"
                )
            if (
                source.name in inputs
                and source.dependencies
                and not inputs[source.name].spec.differentiable
            ):
                raise ValueError(
                    "derived input source cannot silently freeze its pullback"
                )
        try:
            tuple(
                TopologicalSorter(
                    {s.name: s.dependencies for s in sources}
                ).static_order()
            )
        except CycleError as exc:
            raise ValueError(
                "provider dependency cycle; declare an implicit region instead"
            ) from exc
        reachable = set(inputs) - set(names)
        pending = list(reachable)
        while pending:
            for name in source_map[pending.pop()].dependencies:
                if name not in reachable:
                    reachable.add(name)
                    pending.append(name)
        if reachable != set(source_names):
            raise ValueError(
                "unused declared parameter source; possible omitted equation branch"
            )

    @property
    def state_names(self):
        return tuple(state.name for state in self.states)

    @property
    def boundary_sources(self):
        inputs = _inputs(self.equations)
        return tuple(source.name for source in self.sources if source.name in inputs)

    @property
    def differentiable_sources(self):
        inputs = _inputs(self.equations)
        return tuple(
            name for name in self.boundary_sources if inputs[name].spec.differentiable
        )

    @property
    def dependency_graph(self):
        """Data-only provider DAG plus the explicitly cut implicit equation region."""
        order = tuple(
            TopologicalSorter(
                {s.name: s.dependencies for s in self.sources}
            ).static_order()
        )
        derived_sources = {s.name for s in self.sources if s.dependencies}
        return {
            "providers": {s.name: list(s.dependencies) for s in self.sources},
            "provider_order": list(order),
            "implicit_region": {
                "states": list(self.state_names),
                "solver_contract": self.solver_contract,
                "residual_dependencies": {
                    s.residual: list(_dependencies(self.equations.outputs[s.residual]))
                    for s in self.states
                },
                "constraints": [
                    s.residual for s in self.states if s.kind == "constraint"
                ],
            },
            "objective_dependencies": list(
                _dependencies(self.equations.outputs[self.objective])
            ),
            "boundary_sources": list(self.boundary_sources),
            "provider_pullback_order": [
                name for name in reversed(order) if name in derived_sources
            ],
        }

    def _semantic_payload(self):
        return {
            "schema": SCHEMA,
            "schema_version": VERSION,
            "convention": CONVENTION,
            "equation_hash": self.equations.logical_hash,
            "objective": self.objective,
            "states": [asdict(s) for s in self.states],
            "sources": [asdict(s) for s in self.sources],
            "model_identity": self.model_identity,
            "solver_contract": self.solver_contract,
        }

    @property
    def identity(self):
        return canonical_hash(self._semantic_payload())

    def to_payload(self):
        payload = self._semantic_payload()
        payload.pop("equation_hash")
        return {
            **payload,
            "equations": self.equations.to_payload(),
            "identity": self.identity,
        }

    def dumps(self):
        return (
            json.dumps(self.to_payload(), sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        )

    @classmethod
    def from_payload(cls, payload):
        fields = {
            "schema",
            "schema_version",
            "convention",
            "equations",
            "objective",
            "states",
            "sources",
            "model_identity",
            "solver_contract",
            "identity",
        }
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ValueError("invalid stationary problem fields")
        if (
            payload["schema"] != SCHEMA
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != VERSION
            or payload["convention"] != CONVENTION
        ):
            raise ValueError("unsupported stationary problem version/convention")
        try:
            result = cls(
                Program.from_payload(payload["equations"]),
                payload["objective"],
                tuple(_record(StationaryState, s) for s in payload["states"]),
                tuple(_record(ParameterSource, s) for s in payload["sources"]),
                payload["model_identity"],
                payload["solver_contract"],
            )
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("malformed stationary problem") from exc
        if result.identity != payload["identity"]:
            raise ValueError("stationary problem identity mismatch")
        return result

    @classmethod
    def loads(cls, source):
        return cls.from_payload(_load_json(source))

    def compile(self, *, max_elements=1_000_000):
        """Generate first-order fragments and declared implicit VJP plans; do not solve."""
        return compile_stationary(self, max_elements=max_elements)


def _unit_reverse(primal, output, inputs, max_elements):
    """Use #151, specializing only the scalar seed (no new derivative rules)."""
    reverse = transpose_program(
        primal, (output,), inputs=inputs, max_elements=max_elements
    )
    mapping = {}
    for node in reverse.program.live_nodes:
        if node.op == "input" and node.attrs["name"] == f"bar_{output}":
            mapping[node] = constant(
                1, replace(node.spec, role="constant", differentiable=False)
            )
        else:
            mapping[node] = replace(node, inputs=tuple(mapping[n] for n in node.inputs))
    return {name: mapping[node] for name, node in reverse.program.outputs.items()}


@dataclass(frozen=True)
class StationaryDerivativePlan:
    """Generated adjoint RHS and Lagrangian partials at independent source fields.

    With multipliers set to a qualified adjoint solution, ``weights`` are
    E_q+R_q^T lambda; before that they are only partial contractions. Nonzero
    ``stationarity`` outputs diagnose the adjoint equation, not primal validity.
    No execution adapter here can certify a converged/stale/native state.
    """

    problem: StationaryProblem
    lagrangian: Program
    rhs: Program
    partials: Program
    multiplier_inputs: Mapping[str, str]
    stationarity_outputs: Mapping[str, str]
    weight_outputs: Mapping[str, str]
    implicit_plans: Mapping[str, ImplicitVJPPlan]

    def __post_init__(self):
        for name in (
            "multiplier_inputs",
            "stationarity_outputs",
            "weight_outputs",
            "implicit_plans",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))

    @property
    def identity(self):
        return canonical_hash(
            {
                "schema": "vibeqc.stationary_derivative_plan",
                "version": PLAN_VERSION,
                "problem": self.problem.identity,
                "lagrangian": self.lagrangian.logical_hash,
                "rhs": self.rhs.logical_hash,
                "partials": self.partials.logical_hash,
                "multipliers": dict(self.multiplier_inputs),
                "stationarity": dict(self.stationarity_outputs),
                "weights": dict(self.weight_outputs),
                "implicit_plans": {
                    name: plan.identity for name, plan in self.implicit_plans.items()
                },
            }
        )

    def to_payload(self):
        """Persist an inspectable plan using ordinary TensorIR artifacts."""
        return {
            "schema": "vibeqc.stationary_derivative_plan",
            "schema_version": PLAN_VERSION,
            "problem": self.problem.to_payload(),
            "programs": {
                "lagrangian": self.lagrangian.to_payload(),
                "rhs": self.rhs.to_payload(),
                "partials": self.partials.to_payload(),
            },
            "multiplier_inputs": dict(self.multiplier_inputs),
            "stationarity_outputs": dict(self.stationarity_outputs),
            "weight_outputs": dict(self.weight_outputs),
            "implicit_plans": {
                name: plan.to_payload() for name, plan in self.implicit_plans.items()
            },
            "identity": self.identity,
        }

    def dumps(self):
        return (
            json.dumps(self.to_payload(), sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        )

    @classmethod
    def from_payload(cls, payload, *, max_elements=1_000_000):
        """Regenerate from equations and reject altered derivative fragments.

        Reuse the caller's #151 constant-generation limit during regeneration.
        A claimed hash alone is not evidence that a supplied graph is the
        derivative of the stationary problem. No serialized solver is invoked.
        """
        if not isinstance(payload, dict) or "problem" not in payload:
            raise ValueError("invalid stationary derivative plan fields")
        problem = StationaryProblem.from_payload(payload["problem"])
        plan = problem.compile(max_elements=max_elements)
        # Canonical JSON also distinguishes bools from integer version fields,
        # while normalizing tuple/list representation of TensorIR attributes.
        if json.dumps(payload, sort_keys=True, allow_nan=False) != json.dumps(
            plan.to_payload(), sort_keys=True, allow_nan=False
        ):
            raise ValueError(
                "stationary derivative plan differs from regenerated equations"
            )
        return plan

    @classmethod
    def loads(cls, source, *, max_elements=1_000_000):
        return cls.from_payload(_load_json(source), max_elements=max_elements)

    @property
    def provider_pullbacks(self):
        """Outstanding provider rules after method-level implicit dispatch."""
        order = self.problem.dependency_graph["provider_pullback_order"]
        sources = {s.name: s for s in self.problem.sources}
        return tuple(sources[name] for name in order)


def _compile_implicit_state(problem, state):
    """Compile one declared independent state through #465.

    A state is dispatched only when its operator identity is explicit. Coupled
    state blocks must first be represented as one independent composite state;
    silently solving one row of a KKT/CC system would be mathematically wrong.
    """
    if state.implicit_operator_identity is None:
        return None
    residual = problem.equations.outputs[state.residual]
    residual_program = Program(
        {state.residual: residual},
        provenance={
            "stationary_problem": problem.identity,
            "state": state.name,
            "role": "implicit-residual",
        },
    )
    inputs = _inputs(residual_program)
    coupled = (set(inputs) & set(problem.state_names)) - {state.name}
    if coupled:
        raise NotImplementedError(
            "automatic implicit dispatch requires one independent state block; "
            f"{state.name!r} is coupled to {sorted(coupled)!r}"
        )
    parameters = tuple(
        name for name in problem.differentiable_sources if name in inputs
    )
    if not parameters:
        raise ValueError(
            f"implicit state {state.name!r} has no differentiable residual source"
        )
    return ImplicitSolveSpec(
        residual_program,
        state.name,
        parameters,
        state.implicit_operator_identity,
        residual_name=state.residual,
        state_layout=state.coordinate_identity,
        residual_layout=state.residual_layout_identity,
        gauge=state.gauge_identity,
    ).compile()


def compile_stationary(problem, *, max_elements=1_000_000):
    """Compose L and generate -E_x and (L_x,L_q) using the shared TensorIR AD.

    Residual values use the declared physical equations, never shifted solver
    denominators or iteration history. Source weights remain separate from
    upstream provider pullbacks and nuclear-coordinate derivative contractions.
    """
    if not isinstance(problem, StationaryProblem):
        raise TypeError("compile_stationary requires StationaryProblem")
    if type(max_elements) is not int or max_elements < 0:
        raise ValueError("max_elements must be a nonnegative integer")
    objective = problem.equations.outputs[problem.objective]
    terms, multipliers = [objective], {}
    for state in problem.states:
        residual = problem.equations.outputs[state.residual]
        name = f"{_PREFIX}lambda_{state.name}"
        multipliers[state.name] = name
        multiplier = input_tensor(
            name, replace(residual.spec, role="input", differentiable=False)
        )
        term = multiply(multiplier, residual)
        if term.spec.indices:
            term = reduce_sum(term, tuple(range(len(term.spec.indices))))
        terms.append(term)
    provenance = {"stationary_problem": problem.identity, "derivative_order": 1}
    lagrangian = Program(
        {f"{_PREFIX}lagrangian": add(*terms)},
        definitions=problem.equations.live_nodes,
        provenance=provenance,
    )
    selected = (*problem.state_names, *problem.differentiable_sources)
    reverse = _unit_reverse(lagrangian, f"{_PREFIX}lagrangian", selected, max_elements)
    state_outputs = {name: f"{_PREFIX}state_{name}" for name in problem.state_names}
    weight_outputs = {
        name: f"{_PREFIX}weight_{name}" for name in problem.differentiable_sources
    }
    partials = Program(
        {
            **{out: reverse[f"bar_{name}"] for name, out in state_outputs.items()},
            **{out: reverse[f"bar_{name}"] for name, out in weight_outputs.items()},
        },
        provenance={**provenance, "role": "lagrangian-partials-at-source-boundary"},
    )
    # Keep the original multi-output graph: #151 selects live input names,
    # not retained dead definitions. Seed ONLY E, so constraint multipliers
    # absent from E still receive an exact zero from the shared AD machinery.
    objective_reverse = _unit_reverse(
        problem.equations,
        problem.objective,
        problem.state_names,
        max_elements,
    )
    rhs = Program(
        {
            name: add(objective_reverse[f"bar_{name}"], coefficients=(-1,))
            for name in problem.state_names
        },
        provenance={**provenance, "role": "negative-objective-state-gradient"},
    )
    implicit_plans = {
        state.name: plan
        for state in problem.states
        if (plan := _compile_implicit_state(problem, state)) is not None
    }
    return StationaryDerivativePlan(
        problem,
        lagrangian,
        rhs,
        partials,
        multipliers,
        state_outputs,
        weight_outputs,
        implicit_plans,
    )
