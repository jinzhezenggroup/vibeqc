"""Method-neutral electronic-structure orchestration IR.

This layer sits above the existing DFT MethodIR, TensorIR and IntegralIR.  It is
representation-only: constructing an ElectronicMethodIR does not select a
backend, allocate resources, or change any production execution path.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter

from vibeqc_compiler.common.provenance import canonical_hash

SCHEMA = "vibeqc.electronic_method_ir"
VERSION = 1

_FAMILIES = ("hf", "ks", "cc")
_REFERENCES = ("restricted", "unrestricted")
_STATE_KINDS = ("density", "amplitude")
_ENERGY_KINDS = ("total", "correlation")


def _identifier(value: typing.Any, label: str) -> str:
    if not isinstance(value, str) or not value.isidentifier():
        raise ValueError(f"{label} must be a Python identifier")
    return value


def _identity(value: typing.Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty identity")
    return value


def _names(values: typing.Any, label: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{label} must be a sequence, not a string")
    result = tuple(_identifier(value, label) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"duplicate {label}")
    return tuple(sorted(result))


@dataclass(frozen=True)
class StateSpec:
    """One persistent fixed-point state block."""

    name: str
    kind: str
    coordinate_identity: str

    def __post_init__(self) -> None:
        _identifier(self.name, "state name")
        if self.kind not in _STATE_KINDS:
            raise ValueError(f"unsupported state kind {self.kind!r}")
        _identity(self.coordinate_identity, "state coordinate")

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "coordinate_identity": self.coordinate_identity,
        }


@dataclass(frozen=True)
class OperatorSpec:
    """One method-level operator with explicit symbolic dataflow.

    ir_identity references an already-owned lower-level IR when one exists;
    it never embeds or copies that graph.
    """

    name: str
    kind: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    ir_identity: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.name, "operator name")
        _identifier(self.kind, "operator kind")
        object.__setattr__(self, "inputs", _names(self.inputs, "operator input"))
        object.__setattr__(self, "outputs", _names(self.outputs, "operator output"))
        if not self.outputs:
            raise ValueError("operator requires at least one output")
        if self.ir_identity is not None:
            _identity(self.ir_identity, "operator IR")

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            **({"ir_identity": self.ir_identity} if self.ir_identity else {}),
        }


@dataclass(frozen=True)
class EnergySpec:
    """Named energy output and whether it is total or correlation-only."""

    output: str
    kind: str = "total"

    def __post_init__(self) -> None:
        _identifier(self.output, "energy output")
        if self.kind not in _ENERGY_KINDS:
            raise ValueError(f"unsupported energy kind {self.kind!r}")

    def to_payload(self) -> dict[str, typing.Any]:
        return {"output": self.output, "kind": self.kind}


@dataclass(frozen=True)
class IterationSpec:
    """Fixed-point ownership without prescribing one concrete solver."""

    states: tuple[str, ...]
    residuals: tuple[tuple[str, str], ...]
    updates: tuple[tuple[str, str], ...]
    solver_contract: str

    def __post_init__(self) -> None:
        states = _names(self.states, "iteration state")
        object.__setattr__(self, "states", states)
        _identity(self.solver_contract, "solver contract")

        def mapping(values: typing.Any, label: str) -> tuple[tuple[str, str], ...]:
            if isinstance(values, dict):
                values = tuple(values.items())
            values = tuple(values)
            checked: list[tuple[str, str]] = []
            for item in values:
                if not isinstance(item, tuple) or len(item) != 2:
                    raise ValueError(f"{label} must contain (state, value) pairs")
                state, value = item
                checked.append(
                    (_identifier(state, f"{label} state"), _identifier(value, label))
                )
            if {state for state, _ in checked} != set(states):
                raise ValueError(
                    f"{label} must cover every iteration state exactly once"
                )
            if len({state for state, _ in checked}) != len(checked):
                raise ValueError(f"duplicate {label} state")
            return tuple(sorted(checked))

        object.__setattr__(self, "residuals", mapping(self.residuals, "residual"))
        object.__setattr__(self, "updates", mapping(self.updates, "update"))

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "states": list(self.states),
            "residuals": [list(item) for item in self.residuals],
            "updates": [list(item) for item in self.updates],
            "solver_contract": self.solver_contract,
        }


@dataclass(frozen=True)
class ResponseSpec:
    """Optional response boundary owned by another compiler/runtime layer."""

    name: str
    operator_identity: str
    states: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.name, "response name")
        _identity(self.operator_identity, "response operator")
        object.__setattr__(self, "states", _names(self.states, "response state"))
        if not self.states:
            raise ValueError("response requires at least one state")

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "name": self.name,
            "operator_identity": self.operator_identity,
            "states": list(self.states),
        }


@dataclass(frozen=True)
class DerivativeSpec:
    """Optional derivative contract above a response or stationary plan."""

    order: int
    output: str
    plan_identity: str

    def __post_init__(self) -> None:
        if type(self.order) is not int or self.order not in (1, 2):
            raise ValueError("electronic method derivative order must be one or two")
        _identifier(self.output, "derivative output")
        _identity(self.plan_identity, "derivative plan")

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "order": self.order,
            "output": self.output,
            "plan_identity": self.plan_identity,
        }


@dataclass(frozen=True)
class ElectronicMethodIR:
    """Canonical method-level graph shared by HF, KS and coupled cluster.

    This first slice is descriptive only.  It captures persistent state,
    operator/data dependencies, energy ownership and iteration contracts while
    retaining lower-level compiler graphs by identity.
    """

    identifier: str
    family: str
    reference: str
    sources: tuple[str, ...]
    states: tuple[StateSpec, ...]
    operators: tuple[OperatorSpec, ...]
    energy: EnergySpec
    iteration: IterationSpec | None = None
    response: ResponseSpec | None = None
    derivative: DerivativeSpec | None = None
    composition_identity: str | None = None
    version: int = VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.identifier, str) or not self.identifier.strip():
            raise ValueError("electronic method identifier must be nonempty")
        if self.family not in _FAMILIES:
            raise ValueError(f"unsupported electronic method family {self.family!r}")
        if self.reference not in _REFERENCES:
            raise ValueError(f"unsupported reference {self.reference!r}")
        if type(self.version) is not int or self.version != VERSION:
            raise ValueError("unsupported electronic MethodIR version")
        if self.composition_identity is not None:
            _identity(self.composition_identity, "composition")

        sources = _names(self.sources, "method source")
        states = tuple(self.states)
        operators = tuple(self.operators)
        if not states or any(not isinstance(item, StateSpec) for item in states):
            raise ValueError("electronic method requires typed persistent state")
        if not operators or any(
            not isinstance(item, OperatorSpec) for item in operators
        ):
            raise ValueError("electronic method requires typed operators")
        states = tuple(sorted(states, key=lambda item: item.name))
        operators = tuple(sorted(operators, key=lambda item: item.name))
        if len({item.name for item in states}) != len(states):
            raise ValueError("duplicate state name")
        if not isinstance(self.energy, EnergySpec):
            raise TypeError("energy must be EnergySpec")
        if self.iteration is not None and not isinstance(self.iteration, IterationSpec):
            raise TypeError("iteration must be IterationSpec")
        if self.response is not None and not isinstance(self.response, ResponseSpec):
            raise TypeError("response must be ResponseSpec")
        if self.derivative is not None and not isinstance(
            self.derivative, DerivativeSpec
        ):
            raise TypeError("derivative must be DerivativeSpec")
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "operators", operators)

        state_names = {item.name for item in states}
        if state_names & set(sources):
            raise ValueError("method sources and persistent states must be disjoint")
        operator_names = [item.name for item in operators]
        if len(set(operator_names)) != len(operator_names):
            raise ValueError("duplicate operator name")

        owner: dict[str, str] = {}
        for operator in operators:
            for output in operator.outputs:
                if output in state_names or output in sources:
                    raise ValueError(
                        f"operator output shadows declared value {output!r}"
                    )
                if output in owner:
                    raise ValueError(f"duplicate operator output {output!r}")
                owner[output] = operator.name

        available = set(sources) | state_names | set(owner)
        for operator in operators:
            dangling = set(operator.inputs) - available
            if dangling:
                raise ValueError(
                    f"operator {operator.name!r} has dangling inputs {tuple(sorted(dangling))}"
                )

        graph: dict[str, set[str]] = {}
        for operator in operators:
            deps = {
                owner[name]
                for name in operator.inputs
                if name in owner and owner[name] != operator.name
            }
            if any(owner.get(name) == operator.name for name in operator.inputs):
                raise ValueError(f"operator {operator.name!r} reads its own output")
            graph[operator.name] = deps
        try:
            tuple(TopologicalSorter(graph).static_order())
        except CycleError as exc:
            raise ValueError("electronic method operator dependency cycle") from exc

        if self.energy.output not in owner:
            raise ValueError("energy output must be produced by a declared operator")

        if self.iteration is not None:
            if set(self.iteration.states) - state_names:
                raise ValueError("iteration references undeclared state")
            for _, value in (*self.iteration.residuals, *self.iteration.updates):
                if value not in owner:
                    raise ValueError(
                        "iteration residual/update must be operator output"
                    )

        if self.response is not None and set(self.response.states) - state_names:
            raise ValueError("response references undeclared state")

    @property
    def dependency_graph(self) -> dict[str, tuple[str, ...]]:
        """Operator dependencies inferred from symbolic inputs."""
        owner = {
            output: operator.name
            for operator in self.operators
            for output in operator.outputs
        }
        return {
            operator.name: tuple(
                sorted(
                    {
                        owner[name]
                        for name in operator.inputs
                        if name in owner and owner[name] != operator.name
                    }
                )
            )
            for operator in self.operators
        }

    @property
    def operator_order(self) -> tuple[str, ...]:
        """Deterministic topological order for inspection and future lowering."""
        return tuple(TopologicalSorter(self.dependency_graph).static_order())

    @property
    def structural_payload(self) -> dict[str, typing.Any]:
        return {
            "schema": SCHEMA,
            "version": self.version,
            "family": self.family,
            "reference": self.reference,
            "sources": list(self.sources),
            "states": [item.to_payload() for item in self.states],
            "operators": [item.to_payload() for item in self.operators],
            "energy": self.energy.to_payload(),
            **(
                {"iteration": self.iteration.to_payload()}
                if self.iteration is not None
                else {}
            ),
            **(
                {"response": self.response.to_payload()}
                if self.response is not None
                else {}
            ),
            **(
                {"derivative": self.derivative.to_payload()}
                if self.derivative is not None
                else {}
            ),
            **(
                {"composition_identity": self.composition_identity}
                if self.composition_identity is not None
                else {}
            ),
        }

    @property
    def identity(self) -> str:
        """Stable structural hash independent of descriptive method aliases."""
        return canonical_hash(self.structural_payload)

    @property
    def manifest_identity(self) -> str:
        return canonical_hash(self.to_payload())

    def to_payload(self) -> dict[str, typing.Any]:
        return {"identifier": self.identifier, **self.structural_payload}


def rhf_electronic_method_ir() -> ElectronicMethodIR:
    """Describe the restricted HF fixed-point graph without selecting a backend."""
    return ElectronicMethodIR(
        identifier="RHF",
        family="hf",
        reference="restricted",
        sources=(
            "core_hamiltonian",
            "eri_provider",
            "nuclear_repulsion",
            "occupations",
            "overlap",
        ),
        states=(StateSpec("density", "density", "restricted-ao-density-v1"),),
        operators=(
            OperatorSpec(
                "build_fock",
                "fock",
                ("core_hamiltonian", "density", "eri_provider"),
                ("fock",),
            ),
            OperatorSpec(
                "diagonalize_fock",
                "eigensolve",
                ("fock", "overlap"),
                ("orbital_coefficients", "orbital_energies"),
            ),
            OperatorSpec(
                "build_density",
                "density_update",
                ("occupations", "orbital_coefficients"),
                ("density_next",),
            ),
            OperatorSpec(
                "density_residual",
                "fixed_point_residual",
                ("density", "density_next"),
                ("density_error",),
            ),
            OperatorSpec(
                "hf_energy",
                "energy",
                ("core_hamiltonian", "density", "fock", "nuclear_repulsion"),
                ("total_energy",),
            ),
        ),
        energy=EnergySpec("total_energy"),
        iteration=IterationSpec(
            ("density",),
            (("density", "density_error"),),
            (("density", "density_next"),),
            "scf-fixed-point-v1",
        ),
    )


def rks_electronic_method_ir(method: typing.Any) -> ElectronicMethodIR:
    """Project an existing restricted DFT MethodIR into the common orchestration IR."""
    from .spec import MethodIR, resolve_method

    method = resolve_method(method) if isinstance(method, str) else method
    if not isinstance(method, MethodIR):
        raise TypeError("RKS projection requires a resolved DFT MethodIR")
    if method.reference != "restricted":
        raise ValueError("RKS projection requires an unpolarized/restricted MethodIR")

    operators = tuple(method.requirements["operators"])
    supported = {"semilocal-xc", "full-range-exchange"}
    if not set(operators) <= supported or "semilocal-xc" not in operators:
        raise ValueError(
            "first RKS ElectronicMethodIR slice supports semilocal XC with optional full-range exchange"
        )
    has_exchange = "full-range-exchange" in operators
    fock_inputs = ["core_hamiltonian", "hartree", "xc_potential"]
    energy_inputs = [
        "core_hamiltonian",
        "density",
        "hartree_energy",
        "nuclear_repulsion",
        "xc_energy",
    ]
    nodes: list[OperatorSpec] = [
        OperatorSpec(
            "build_hartree",
            "coulomb",
            ("density", "eri_provider"),
            ("hartree", "hartree_energy"),
        ),
        OperatorSpec(
            "evaluate_xc",
            "semilocal_xc",
            ("basis", "density", "quadrature"),
            ("xc_energy", "xc_potential"),
        ),
    ]
    if has_exchange:
        nodes.append(
            OperatorSpec(
                "build_exchange",
                "exact_exchange",
                ("density", "eri_provider"),
                ("exchange", "exchange_energy"),
                ir_identity=method.identity,
            )
        )
        fock_inputs.append("exchange")
        energy_inputs.append("exchange_energy")
    nodes.extend(
        (
            OperatorSpec(
                "assemble_fock",
                "fock",
                tuple(fock_inputs),
                ("fock",),
            ),
            OperatorSpec(
                "diagonalize_fock",
                "eigensolve",
                ("fock", "overlap"),
                ("orbital_coefficients", "orbital_energies"),
            ),
            OperatorSpec(
                "build_density",
                "density_update",
                ("occupations", "orbital_coefficients"),
                ("density_next",),
            ),
            OperatorSpec(
                "density_residual",
                "fixed_point_residual",
                ("density", "density_next"),
                ("density_error",),
            ),
            OperatorSpec(
                "ks_energy",
                "energy",
                tuple(energy_inputs),
                ("total_energy",),
            ),
        )
    )
    return ElectronicMethodIR(
        identifier=method.identifier,
        family="ks",
        reference="restricted",
        sources=(
            "basis",
            "core_hamiltonian",
            "eri_provider",
            "nuclear_repulsion",
            "occupations",
            "overlap",
            "quadrature",
        ),
        states=(StateSpec("density", "density", "restricted-ao-density-v1"),),
        operators=tuple(nodes),
        energy=EnergySpec("total_energy"),
        iteration=IterationSpec(
            ("density",),
            (("density", "density_error"),),
            (("density", "density_next"),),
            "ks-fixed-point-v1",
        ),
        composition_identity=method.identity,
    )


def rccsd_electronic_method_ir(program: typing.Any) -> ElectronicMethodIR:
    """Wrap audited RCCSD TensorIR equations without importing tools into compiler."""
    from vibeqc_compiler.tensor import Program

    if not isinstance(program, Program):
        raise TypeError("RCCSD projection requires a TensorIR Program")
    required = {"correlation_energy", "singles_residual", "doubles_residual"}
    if not required <= set(program.outputs):
        raise ValueError(
            "RCCSD TensorIR must expose energy, singles and doubles residuals"
        )
    inputs = {node.attrs["name"] for node in program.live_nodes if node.op == "input"}
    if not {"t1", "t2"} <= inputs:
        raise ValueError("RCCSD TensorIR must expose t1 and t2 state inputs")
    sources = tuple(sorted((inputs - {"t1", "t2"}) | {"denominators"}))
    return ElectronicMethodIR(
        identifier="RCCSD",
        family="cc",
        reference="restricted",
        sources=sources,
        states=(
            StateSpec("t1", "amplitude", "restricted-spatial-t1-v1"),
            StateSpec("t2", "amplitude", "restricted-spatial-t2-v1"),
        ),
        operators=(
            OperatorSpec(
                "rccsd_equations",
                "tensor_program",
                tuple(sorted(inputs)),
                (
                    "correlation_energy",
                    "doubles_residual",
                    "singles_residual",
                ),
                ir_identity=program.logical_hash,
            ),
            OperatorSpec(
                "rccsd_update",
                "amplitude_update",
                (
                    "denominators",
                    "doubles_residual",
                    "singles_residual",
                    "t1",
                    "t2",
                ),
                ("next_t1", "next_t2"),
            ),
        ),
        energy=EnergySpec("correlation_energy", kind="correlation"),
        iteration=IterationSpec(
            ("t1", "t2"),
            (
                ("t1", "singles_residual"),
                ("t2", "doubles_residual"),
            ),
            (("t1", "next_t1"), ("t2", "next_t2")),
            "rccsd-jacobi-diis-v1",
        ),
        composition_identity=program.logical_hash,
    )
