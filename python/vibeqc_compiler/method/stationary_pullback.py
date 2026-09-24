"""Fail-closed provider-pullback planning for stationary derivatives (#181).

StationaryProblem generates cotangents only at live TensorIR source fields. This
module resolves the declared provider DAG against an explicit registry of
first-order pullback contracts and records how direct and propagated cotangents
must be accumulated before each provider rule runs. It is deliberately a plan:
provider-specific mathematics and live-state execution remain with the provider.
"""

from __future__ import annotations

import typing
from dataclasses import asdict, dataclass

from vibeqc_compiler.common.provenance import canonical_hash

from .stationary import StationaryDerivativePlan


@dataclass(frozen=True)
class ProviderPullbackRule:
    """One registered first-order provider rule with exact field identities."""

    identity: str
    source_identity: str
    dependency_identities: tuple[str, ...]
    derivative_order: int = 1

    def __post_init__(self) -> None:
        for label, value in (
            ("provider rule", self.identity),
            ("provider source", self.source_identity),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label} identity must be nonempty")
        dependencies = tuple(self.dependency_identities)
        if any(not isinstance(value, str) or not value.strip() for value in dependencies):
            raise ValueError("provider dependency identities must be nonempty")
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("duplicate provider dependency identity")
        if type(self.derivative_order) is not int or self.derivative_order != 1:
            raise ValueError("stationary provider rules currently require derivative order 1")
        object.__setattr__(self, "dependency_identities", dependencies)


@dataclass(frozen=True)
class ProviderPullbackStep:
    """One reverse-DAG rule invocation after cotangent accumulation."""

    source: str
    source_identity: str
    rule_identity: str
    dependencies: tuple[str, ...]
    dependency_identities: tuple[str, ...]
    direct_weight_output: str | None
    propagated_from: tuple[str, ...]

    @property
    def contribution_count(self) -> int:
        return int(self.direct_weight_output is not None) + len(self.propagated_from)


@dataclass(frozen=True)
class ProviderRootCotangent:
    """Final cotangent accumulation at one root physical source field."""

    source: str
    source_identity: str
    direct_weight_output: str | None
    propagated_from: tuple[str, ...]

    @property
    def contribution_count(self) -> int:
        return int(self.direct_weight_output is not None) + len(self.propagated_from)


@dataclass(frozen=True)
class StationaryProviderPullbackPlan:
    """Inspectable provider-rule dispatch and accumulation topology."""

    stationary_plan_identity: str
    steps: tuple[ProviderPullbackStep, ...]
    roots: tuple[ProviderRootCotangent, ...]

    @property
    def identity(self) -> str:
        return canonical_hash(
            {
                "schema": "vibeqc.stationary_provider_pullback_plan",
                "version": 1,
                "stationary_plan": self.stationary_plan_identity,
                "steps": [asdict(step) for step in self.steps],
                "roots": [asdict(root) for root in self.roots],
            }
        )

    def to_payload(self) -> dict[str, typing.Any]:
        return {
            "schema": "vibeqc.stationary_provider_pullback_plan",
            "schema_version": 1,
            "stationary_plan_identity": self.stationary_plan_identity,
            "steps": [asdict(step) for step in self.steps],
            "roots": [asdict(root) for root in self.roots],
            "identity": self.identity,
        }


def compile_provider_pullback_plan(
    plan: StationaryDerivativePlan,
    rules: typing.Iterable[ProviderPullbackRule],
) -> StationaryProviderPullbackPlan:
    """Resolve active provider cotangent paths against registered rule contracts.

    A rule becomes active only when its source has a generated direct weight or
    receives a propagated cotangent from an active downstream source. This keeps
    deliberately nondifferentiable/frozen provider branches frozen. For active
    paths, every rule must match the exact physical field and dependency-field
    identities declared by the StationaryProblem.
    """
    if not isinstance(plan, StationaryDerivativePlan):
        raise TypeError("provider pullback planning requires StationaryDerivativePlan")

    registry: dict[str, ProviderPullbackRule] = {}
    for rule in rules:
        if not isinstance(rule, ProviderPullbackRule):
            raise TypeError("provider pullback registry requires ProviderPullbackRule")
        if rule.identity in registry:
            raise ValueError(f"duplicate provider pullback rule: {rule.identity}")
        registry[rule.identity] = rule

    sources = {source.name: source for source in plan.problem.sources}
    propagated: dict[str, list[str]] = {name: [] for name in sources}
    steps: list[ProviderPullbackStep] = []

    for name in plan.problem.dependency_graph["provider_pullback_order"]:
        source = sources[name]
        direct = plan.weight_outputs.get(name)
        incoming = tuple(sorted(propagated[name]))
        if direct is None and not incoming:
            continue

        rule_identity = source.pullback_identity
        if rule_identity is None or rule_identity not in registry:
            raise ValueError(f"missing active provider pullback rule for {name!r}")
        rule = registry[rule_identity]
        if rule.source_identity != source.identity:
            raise ValueError(f"provider pullback source identity mismatch for {name!r}")
        expected_dependency_identities = tuple(
            sources[dependency].identity for dependency in source.dependencies
        )
        if rule.dependency_identities != expected_dependency_identities:
            raise ValueError(
                f"provider pullback dependency identity mismatch for {name!r}"
            )

        step = ProviderPullbackStep(
            source=name,
            source_identity=source.identity,
            rule_identity=rule.identity,
            dependencies=source.dependencies,
            dependency_identities=expected_dependency_identities,
            direct_weight_output=direct,
            propagated_from=incoming,
        )
        if step.contribution_count < 1:
            raise AssertionError("active provider step lost its cotangent")
        steps.append(step)
        for dependency in source.dependencies:
            propagated[dependency].append(name)

    roots = []
    for name, source in sorted(sources.items()):
        if source.dependencies:
            continue
        direct = plan.weight_outputs.get(name)
        incoming = tuple(sorted(propagated[name]))
        if direct is None and not incoming:
            continue
        roots.append(
            ProviderRootCotangent(
                source=name,
                source_identity=source.identity,
                direct_weight_output=direct,
                propagated_from=incoming,
            )
        )

    return StationaryProviderPullbackPlan(plan.identity, tuple(steps), tuple(roots))
