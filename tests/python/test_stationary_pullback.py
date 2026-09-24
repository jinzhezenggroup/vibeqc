"""Provider-pullback planning checks for #181."""

import typing
from dataclasses import FrozenInstanceError, replace

import pytest
from vibeqc_compiler.method.stationary import (
    ParameterSource,
    StationaryProblem,
    StationaryState,
)
from vibeqc_compiler.method.stationary_pullback import (
    ProviderPullbackRule,
    compile_provider_pullback_plan,
)
from vibeqc_compiler.tensor import Program, TensorSpec, add, input_tensor, multiply


def _input(name: str, *, diff: bool = True) -> typing.Any:
    return input_tensor(
        name,
        TensorSpec((), role="parameter", differentiable=diff, dtype="float64"),
    )


def _problem() -> StationaryProblem:
    x = _input("x")
    factor = _input("factor")
    overlap = _input("overlap")
    geometry = _input("geometry")
    residual = add(x, factor, overlap)
    energy = add(multiply(x, x), factor, geometry)
    return StationaryProblem(
        Program({"energy": energy, "residual": residual}),
        "energy",
        (StationaryState("x", "residual", "x-v1", "gauge-v1"),),
        (
            ParameterSource("factor", "factor-field-v1", ("metric",), "factor-rule-v1"),
            ParameterSource("geometry", "geometry-field-v1"),
            ParameterSource("metric", "metric-field-v1", ("geometry",), "metric-rule-v1"),
            ParameterSource("overlap", "overlap-field-v1", ("geometry",), "overlap-rule-v1"),
        ),
        "provider-pullback-test-v1",
        "test-solver-v1",
    )


def _rules() -> tuple[ProviderPullbackRule, ...]:
    return (
        ProviderPullbackRule("factor-rule-v1", "factor-field-v1", ("metric-field-v1",)),
        ProviderPullbackRule("metric-rule-v1", "metric-field-v1", ("geometry-field-v1",)),
        ProviderPullbackRule("overlap-rule-v1", "overlap-field-v1", ("geometry-field-v1",)),
    )


def test_direct_and_propagated_cotangents_are_explicit() -> None:
    derivative = _problem().compile()
    plan = compile_provider_pullback_plan(derivative, _rules())
    steps = {step.source: step for step in plan.steps}

    assert set(steps) == {"factor", "metric", "overlap"}
    assert steps["factor"].direct_weight_output == derivative.weight_outputs["factor"]
    assert steps["factor"].propagated_from == ()
    assert steps["metric"].direct_weight_output is None
    assert steps["metric"].propagated_from == ("factor",)
    assert steps["overlap"].direct_weight_output == derivative.weight_outputs["overlap"]

    assert len(plan.roots) == 1
    root = plan.roots[0]
    assert root.source == "geometry"
    assert root.direct_weight_output == derivative.weight_outputs["geometry"]
    assert root.propagated_from == ("metric", "overlap")
    assert root.contribution_count == 3


def test_reverse_steps_follow_provider_dependencies() -> None:
    plan = compile_provider_pullback_plan(_problem().compile(), _rules())
    position = {step.source: index for index, step in enumerate(plan.steps)}
    assert position["factor"] < position["metric"]
    assert all(step.contribution_count >= 1 for step in plan.steps)


def test_active_rule_registry_fails_closed() -> None:
    derivative = _problem().compile()
    with pytest.raises(ValueError, match="missing active provider pullback rule"):
        compile_provider_pullback_plan(derivative, _rules()[:-1])

    bad_source = replace(_rules()[0], source_identity="wrong-field")
    with pytest.raises(ValueError, match="source identity mismatch"):
        compile_provider_pullback_plan(derivative, (bad_source, *_rules()[1:]))

    bad_dependencies = replace(
        _rules()[0], dependency_identities=("geometry-field-v1",)
    )
    with pytest.raises(ValueError, match="dependency identity mismatch"):
        compile_provider_pullback_plan(derivative, (bad_dependencies, *_rules()[1:]))


def test_rule_contract_and_plan_identity_are_immutable() -> None:
    derivative = _problem().compile()
    plan = compile_provider_pullback_plan(derivative, _rules())
    changed_problem = replace(
        derivative.problem, model_identity="provider-pullback-test-v2"
    )
    changed = compile_provider_pullback_plan(changed_problem.compile(), _rules())
    assert plan.identity != changed.identity

    with pytest.raises(ValueError, match="missing active provider pullback rule"):
        compile_provider_pullback_plan(
            derivative,
            (replace(_rules()[0], identity="factor-rule-v2"), *_rules()[1:]),
        )

    with pytest.raises(FrozenInstanceError):
        plan.steps[0].source = "other"  # type: ignore[misc]


def test_rule_contract_validation_and_duplicate_registry() -> None:
    with pytest.raises(ValueError, match="derivative order 1"):
        ProviderPullbackRule("rule", "field", (), derivative_order=2)
    with pytest.raises(ValueError, match="duplicate provider dependency identity"):
        ProviderPullbackRule("rule", "field", ("root", "root"))
    with pytest.raises(ValueError, match="nonempty"):
        ProviderPullbackRule("", "field", ())

    derivative = _problem().compile()
    with pytest.raises(ValueError, match="duplicate provider pullback rule"):
        compile_provider_pullback_plan(derivative, (*_rules(), _rules()[0]))
    with pytest.raises(TypeError, match="ProviderPullbackRule"):
        compile_provider_pullback_plan(derivative, (*_rules(), object()))
    with pytest.raises(TypeError, match="StationaryDerivativePlan"):
        compile_provider_pullback_plan(object(), _rules())
