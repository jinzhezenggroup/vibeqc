"""Structured bounded solver regions above serial ProgramIR.

A SolverRegion describes loop-carried values, predicates, observation boundaries,
and registered custom derivatives. It does not execute a solver, choose
convergence thresholds, or differentiate iteration history. The runtime owner
remains responsible for those semantics.
"""

from __future__ import annotations

import json
import typing
from dataclasses import asdict, dataclass

from .program import ProgramIR
from .provenance import canonical_hash
from .resources import ResourceCandidate, ResourceIdentity, ResourceRequest

_BOUNDARIES = frozenset(("entry", "iteration", "success", "failure", "exit"))
_COMPLETION_MODES = frozenset(("scalar", "per_item_mask"))
_DERIVATIVE_MODES = frozenset(
    ("implicit_vjp", "implicit_jvp", "stationary_vjp", "stationary_jvp")
)


def _text(value: typing.Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _names(values: typing.Any, name: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, (tuple, list)):
        raise TypeError(f"{name} must be a sequence of names")
    result = tuple(_text(value, name) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"duplicate {name}")
    return result


def _keys(payload: typing.Any, expected: typing.Any, name: str) -> None:
    if not isinstance(payload, dict) or set(payload) != set(expected):
        raise ValueError(f"invalid {name} fields")


@dataclass(frozen=True)
class RegionCarry:
    """One immutable body input whose next value is an explicit body output."""

    current: str
    next: str

    def __post_init__(self) -> None:
        _text(self.current, "carry current")
        _text(self.next, "carry next")
        if self.current == self.next:
            raise ValueError("loop carry must use distinct SSA current/next values")


@dataclass(frozen=True)
class RegionPredicate:
    """Opaque predicate semantics owned by the solver/runtime subsystem."""

    buffer: str
    identity: str

    def __post_init__(self) -> None:
        _text(self.buffer, "predicate buffer")
        _text(self.identity, "predicate identity")


@dataclass(frozen=True)
class RegionCheckpoint:
    """Explicit observation/publication boundary; host visibility is never implied."""

    name: str
    boundary: str
    buffers: tuple[str, ...]
    host_visible: bool = False

    def __post_init__(self) -> None:
        _text(self.name, "checkpoint name")
        if self.boundary not in _BOUNDARIES:
            raise ValueError("unsupported solver-region checkpoint boundary")
        object.__setattr__(self, "buffers", _names(self.buffers, "checkpoint buffers"))
        if not self.buffers:
            raise ValueError("checkpoint must observe at least one buffer")
        if type(self.host_visible) is not bool:
            raise TypeError("checkpoint host_visible must be bool")


@dataclass(frozen=True)
class RegionDerivative:
    """Identity of a custom derivative rule; never an iteration-history tape."""

    mode: str
    identity: str
    order: int = 1

    def __post_init__(self) -> None:
        if self.mode not in _DERIVATIVE_MODES:
            raise ValueError("unsupported solver-region derivative mode")
        _text(self.identity, "derivative identity")
        if type(self.order) is not int or self.order != 1:
            raise ValueError("solver-region derivative order is not registered")


@dataclass(frozen=True)
class RegionCompletion:
    """Scalar completion now; ragged completion requires an explicit active mask."""

    mode: str = "scalar"
    active_mask: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in _COMPLETION_MODES:
            raise ValueError("unsupported solver-region completion mode")
        if self.mode == "scalar":
            if self.active_mask is not None:
                raise ValueError("scalar completion cannot bind an active mask")
        else:
            _text(self.active_mask, "active mask")


@dataclass(frozen=True)
class SolverRegion:
    """Bounded loop semantics around one serial ProgramIR body.

    The body remains SSA. State mutation across iterations is represented only
    by explicit current-to-next carries, so the existing ProgramIR v2 validation
    and resource model remain unchanged.
    """

    name: str
    body: ProgramIR
    invariants: tuple[str, ...]
    carries: tuple[RegionCarry, ...]
    results: tuple[str, ...]
    max_steps: int
    converged: RegionPredicate
    failed: RegionPredicate | None = None
    completion: RegionCompletion = RegionCompletion()
    checkpoints: tuple[RegionCheckpoint, ...] = ()
    derivatives: tuple[RegionDerivative, ...] = ()
    derivative_policy: str = "unsupported"
    schema_version: int = 1

    def __post_init__(self) -> None:
        _text(self.name, "solver-region name")
        if not isinstance(self.body, ProgramIR):
            raise TypeError("solver-region body must be ProgramIR")
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("solver-region max_steps must be a positive integer")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported SolverRegion schema")
        object.__setattr__(
            self, "invariants", _names(self.invariants, "region invariants")
        )
        object.__setattr__(self, "results", _names(self.results, "region results"))
        if not self.results:
            raise ValueError("solver-region requires exported results")
        for field, item_type in (
            ("carries", RegionCarry),
            ("checkpoints", RegionCheckpoint),
            ("derivatives", RegionDerivative),
        ):
            values = getattr(self, field)
            if not isinstance(values, (tuple, list)):
                raise TypeError(f"{field} must be a structured sequence")
            if not all(isinstance(value, item_type) for value in values):
                raise TypeError(f"{field} must contain {item_type.__name__}")
            object.__setattr__(self, field, tuple(values))
        if not self.carries:
            raise ValueError("solver-region requires loop-carried state")
        if not isinstance(self.converged, RegionPredicate):
            raise TypeError("solver-region requires a convergence predicate")
        if self.failed is not None and not isinstance(self.failed, RegionPredicate):
            raise TypeError("solver-region failure predicate must be RegionPredicate")
        if not isinstance(self.completion, RegionCompletion):
            raise TypeError("solver-region completion must be RegionCompletion")

        current = tuple(carry.current for carry in self.carries)
        following = tuple(carry.next for carry in self.carries)
        if len(set(current)) != len(current) or len(set(following)) != len(following):
            raise ValueError("duplicate solver-region carry endpoint")
        if set(current) & set(self.invariants):
            raise ValueError("loop-carried inputs cannot also be invariants")
        if set(self.body.inputs) != set(self.invariants) | set(current):
            raise ValueError(
                "body inputs must be exactly invariants plus carried state"
            )
        buffers = {buffer.name: buffer for buffer in self.body.buffers}
        for carry in self.carries:
            before, after = buffers[carry.current], buffers[carry.next]
            before_contract = (
                before.bytes,
                before.space,
                before.layout,
                before.itemsize,
            )
            after_contract = (
                after.bytes,
                after.space,
                after.layout,
                after.itemsize,
            )
            if before_contract != after_contract:
                raise ValueError(
                    "loop-carried current/next buffers must have identical contracts"
                )

        body_outputs = set(self.body.outputs)
        required_outputs = set(following) | set(self.results) | {self.converged.buffer}
        if self.failed is not None:
            required_outputs.add(self.failed.buffer)
        if self.completion.active_mask is not None:
            required_outputs.add(self.completion.active_mask)
        if not required_outputs <= body_outputs:
            raise ValueError("region semantics reference a non-output body buffer")
        if body_outputs != required_outputs:
            raise ValueError("every body output must have an explicit region role")
        if self.failed is not None and self.failed.buffer == self.converged.buffer:
            raise ValueError("convergence and failure predicates must be distinct")
        available = {buffer.name for buffer in self.body.buffers}
        checkpoint_names = tuple(checkpoint.name for checkpoint in self.checkpoints)
        if len(set(checkpoint_names)) != len(checkpoint_names):
            raise ValueError("duplicate solver-region checkpoint")
        for checkpoint in self.checkpoints:
            observed = set(checkpoint.buffers)
            if not observed <= available:
                raise ValueError("checkpoint references an undeclared body buffer")
            if checkpoint.boundary == "entry":
                if not observed <= set(self.body.inputs):
                    raise ValueError("entry checkpoint can observe body inputs only")
            elif not observed <= body_outputs:
                raise ValueError("post-step checkpoint can observe body outputs only")
            if checkpoint.boundary == "failure" and self.failed is None:
                raise ValueError("failure checkpoint requires a failure predicate")
        derivative_keys = tuple((rule.mode, rule.order) for rule in self.derivatives)
        if len(set(derivative_keys)) != len(derivative_keys):
            raise ValueError("duplicate solver-region derivative registration")
        if self.derivative_policy not in ("unsupported", "custom"):
            raise ValueError("unknown solver-region derivative policy")
        if self.derivative_policy == "unsupported" and self.derivatives:
            raise ValueError("unsupported derivative policy cannot register rules")
        if self.derivative_policy == "custom" and not self.derivatives:
            raise ValueError("custom derivative policy requires registered rules")

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())

    def derivative_rule(self, mode: str, *, order: int = 1) -> RegionDerivative:
        """Return an explicitly registered rule or fail closed."""

        if mode not in _DERIVATIVE_MODES:
            raise ValueError("unsupported solver-region derivative mode")
        if type(order) is not int or order < 1:
            raise ValueError("derivative order must be a positive integer")
        for rule in self.derivatives:
            if rule.mode == mode and rule.order == order:
                return rule
        raise NotImplementedError(
            f"{mode} order {order} is not registered for solver region {self.name!r}"
        )

    def lifetimes(self) -> typing.Any:
        """Reuse one body allocation across bounded steps; do not multiply by max_steps."""

        return self.body.lifetimes()

    def resource_request(self) -> ResourceRequest:
        """Expose body capacities to the shared planner without executing the loop."""

        return ResourceRequest(
            self.name,
            ResourceIdentity(
                "structured_solver_region",
                "SolverRegion",
                "described",
                "described",
                json.dumps({"region": self.identity}),
                self.results,
                "bounded_body_reuse",
            ),
            (
                ResourceCandidate(
                    "body",
                    "resident",
                    self.lifetimes(),
                    decisions=(
                        ("max_steps", str(self.max_steps)),
                        ("completion", self.completion.mode),
                    ),
                ),
            ),
            (
                "Opaque provider-internal scratch, copies, and library workspaces",
                "Runtime convergence/mixing policy and solver-history implementation",
                "Checkpoint payloads retained by callers after region publication",
            ),
        )

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "name": self.name,
            "body": self.body.to_payload(),
            "invariants": list(self.invariants),
            "carries": [asdict(value) for value in self.carries],
            "results": list(self.results),
            "max_steps": self.max_steps,
            "converged": asdict(self.converged),
            "failed": None if self.failed is None else asdict(self.failed),
            "completion": asdict(self.completion),
            "checkpoints": [
                {**asdict(value), "buffers": list(value.buffers)}
                for value in self.checkpoints
            ],
            "derivatives": [asdict(value) for value in self.derivatives],
            "derivative_policy": self.derivative_policy,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_payload(cls, payload: typing.Any) -> SolverRegion:
        """Strict data-only replay; regenerate all validation and identities."""

        _keys(payload, cls.__dataclass_fields__, "SolverRegion")
        data = dict(payload)
        data["body"] = ProgramIR.from_payload(data["body"])
        for name in ("invariants", "results"):
            if not isinstance(data[name], list):
                raise TypeError(f"serialized {name} must be an array")
            data[name] = tuple(data[name])
        for name, record_type in (
            ("carries", RegionCarry),
            ("checkpoints", RegionCheckpoint),
            ("derivatives", RegionDerivative),
        ):
            if not isinstance(data[name], list):
                raise TypeError(f"serialized {name} must be an array")
            records = []
            for value in data[name]:
                _keys(value, record_type.__dataclass_fields__, record_type.__name__)
                item = dict(value)
                if record_type is RegionCheckpoint:
                    if not isinstance(item["buffers"], list):
                        raise TypeError(
                            "serialized checkpoint buffers must be an array"
                        )
                    item["buffers"] = tuple(item["buffers"])
                records.append(record_type(**item))
            data[name] = tuple(records)
        _keys(
            data["converged"],
            RegionPredicate.__dataclass_fields__,
            "convergence predicate",
        )
        data["converged"] = RegionPredicate(**data["converged"])
        if data["failed"] is not None:
            _keys(
                data["failed"],
                RegionPredicate.__dataclass_fields__,
                "failure predicate",
            )
            data["failed"] = RegionPredicate(**data["failed"])
        _keys(
            data["completion"],
            RegionCompletion.__dataclass_fields__,
            "completion policy",
        )
        data["completion"] = RegionCompletion(**data["completion"])
        region = cls(**data)
        if canonical_hash(payload) != canonical_hash(region.to_payload()):
            raise ValueError("noncanonical SolverRegion payload")
        return region
