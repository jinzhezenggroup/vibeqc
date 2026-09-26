"""Method-neutral stationary second-order executor gates (#180/#932)."""

import typing
from types import SimpleNamespace

import numpy as np
import pytest
from vibeqc_compiler.method import (
    StationaryHVPPlan,
    StationaryMeanField,
    resolve_method,
)
from vibeqc_compiler.method.stationary_gradient import SCF_POINT_MODEL

from vibeqc.second_order import (
    StationaryHVPContext,
    StationaryHVPContributor,
    StationaryPerturbationProvider,
    StationaryResponseDriver,
    StationarySecondOrderExecutor,
)





def test_tools_compatibility_shim_reexports_production_owner() -> None:
    """Repository tools must not retain a second second-order implementation."""
    from tools.vibeqc_hessian import stationary_executor as compatibility

    assert compatibility.StationaryHVPContext is StationaryHVPContext
    assert compatibility.StationaryHVPContributor is StationaryHVPContributor
    assert compatibility.StationaryPerturbationProvider is StationaryPerturbationProvider
    assert compatibility.StationaryResponseDriver is StationaryResponseDriver
    assert compatibility.StationarySecondOrderExecutor is StationarySecondOrderExecutor


def _pbe_plan() -> StationaryHVPPlan:
    return StationaryHVPPlan(
        resolve_method("PBE", spin="unpolarized"),
        StationaryMeanField(SCF_POINT_MODEL),
    )


def _contributors(
    source_names: tuple[str, ...],
    events: list[typing.Any],
    *,
    scale: float = 1.0,
) -> tuple[StationaryHVPContributor, ...]:
    result = []
    for index, source in enumerate(source_names, start=1):

        def evaluate(
            context: StationaryHVPContext,
            source: str = source,
            index: int = index,
        ) -> np.ndarray:
            events.append(("source", source, id(context.response)))
            return scale * index * context.direction

        result.append(
            StationaryHVPContributor(source, f"{source}-contributor-v1", evaluate)
        )
    return tuple(reversed(result))


def test_methodir_plan_executes_one_response_and_complete_source_inventory() -> None:
    plan = _pbe_plan()
    events = []

    def build(direction: np.ndarray) -> dict[str, typing.Any]:
        events.append(("build", direction.copy()))
        return {"direction": direction, "token": object()}

    def solve(perturbation: typing.Any) -> dict[str, typing.Any]:
        response = {"perturbation": perturbation, "token": object()}
        events.append(("solve", id(response)))
        return response

    executor = StationarySecondOrderExecutor(
        plan,
        natoms=2,
        perturbation=StationaryPerturbationProvider("pbe-nuclear-rhs-v1", build),
        response=StationaryResponseDriver("shared-cpks-v1", solve),
        contributors=_contributors(plan.source_names, events),
    )
    direction = np.arange(6, dtype=float).reshape(2, 3) / 7
    result = executor.apply(direction)

    coefficient = sum(range(1, len(plan.source_names) + 1))
    np.testing.assert_allclose(result.value, coefficient * direction)
    assert tuple(result.components) == plan.source_names
    assert [item[1] for item in events if item[0] == "source"] == list(
        plan.source_names
    )
    assert sum(item[0] == "build" for item in events) == 1
    assert sum(item[0] == "solve" for item in events) == 1
    response_ids = {item[2] for item in events if item[0] == "source"}
    assert response_ids == {id(result.response)}
    assert result.diagnostics["complete_source_coverage"] is True
    assert result.diagnostics["method_dispatch"] == "none"
    assert not result.value.flags.writeable
    assert not result.direction.flags.writeable
    assert all(not value.flags.writeable for value in result.components.values())
    with pytest.raises(TypeError):
        result.components["extra"] = np.zeros((2, 3))


def test_same_executor_accepts_hf_like_structural_plan_without_method_branch() -> None:
    plan = SimpleNamespace(
        identity="synthetic-rhf-second-order-plan-v1",
        source_names=(
            "one_electron",
            "coulomb",
            "exact_exchange",
            "overlap_pulay",
            "nuclear",
        ),
    )
    events = []
    executor = StationarySecondOrderExecutor(
        plan,
        natoms=1,
        perturbation=StationaryPerturbationProvider(
            "rhf-nuclear-rhs-v1",
            lambda direction: {"direction": direction},
        ),
        response=StationaryResponseDriver(
            "shared-cphf-v1",
            lambda perturbation: {"density_response": perturbation["direction"]},
        ),
        contributors=_contributors(plan.source_names, events, scale=0.5),
    )
    direction = np.array([[1.0, -2.0, 3.0]])
    result = executor.apply(direction)
    coefficient = 0.5 * sum(range(1, len(plan.source_names) + 1))
    np.testing.assert_allclose(result.value, coefficient * direction)
    assert executor.source_names == plan.source_names
    assert result.diagnostics["method_dispatch"] == "none"


@pytest.mark.parametrize(
    ("contributors", "match"),
    [
        (
            (
                StationaryHVPContributor(
                    "one_electron", "one-v1", lambda context: context.direction
                ),
            ),
            "missing=",
        ),
        (
            (
                StationaryHVPContributor(
                    "one_electron", "one-v1", lambda context: context.direction
                ),
                StationaryHVPContributor(
                    "unknown", "unknown-v1", lambda context: context.direction
                ),
            ),
            "extra=",
        ),
    ],
)
def test_executor_rejects_partial_or_extra_source_coverage(
    contributors: tuple[StationaryHVPContributor, ...],
    match: str,
) -> None:
    plan = SimpleNamespace(identity="plan-v1", source_names=("one_electron", "nuclear"))
    with pytest.raises(ValueError, match=match):
        StationarySecondOrderExecutor(
            plan,
            natoms=1,
            perturbation=StationaryPerturbationProvider("rhs-v1", lambda value: value),
            response=StationaryResponseDriver("response-v1", lambda value: value),
            contributors=contributors,
        )


def test_failure_does_not_publish_partial_result_and_valid_replay_succeeds() -> None:
    plan = SimpleNamespace(identity="plan-v1", source_names=("a", "b"))
    state = {"bad": True, "calls": []}

    def first(context: StationaryHVPContext) -> np.ndarray:
        state["calls"].append("a")
        return context.direction

    def second(context: StationaryHVPContext) -> np.ndarray:
        state["calls"].append("b")
        if state["bad"]:
            return np.full_like(context.direction, np.nan)
        return 2 * context.direction

    executor = StationarySecondOrderExecutor(
        plan,
        natoms=1,
        perturbation=StationaryPerturbationProvider("rhs-v1", lambda value: value),
        response=StationaryResponseDriver("response-v1", lambda value: value),
        contributors=(
            StationaryHVPContributor("a", "a-v1", first),
            StationaryHVPContributor("b", "b-v1", second),
        ),
    )
    direction = np.array([[0.25, -0.5, 1.0]])
    with pytest.raises(ValueError, match="finite real"):
        executor.apply(direction)
    assert state["calls"] == ["a", "b"]

    state["bad"] = False
    result = executor.apply(direction)
    np.testing.assert_allclose(result.value, 3 * direction)
    assert state["calls"] == ["a", "b", "a", "b"]


@pytest.mark.parametrize(
    "direction",
    [
        np.zeros((3,)),
        np.array([[np.inf, 0.0, 0.0]]),
        np.array([[1.0j, 0.0, 0.0]]),
    ],
)
def test_executor_rejects_invalid_directions_before_build(
    direction: typing.Any,
) -> None:
    calls = []
    plan = SimpleNamespace(identity="plan-v1", source_names=("a",))
    executor = StationarySecondOrderExecutor(
        plan,
        natoms=1,
        perturbation=StationaryPerturbationProvider(
            "rhs-v1", lambda value: calls.append(value)
        ),
        response=StationaryResponseDriver("response-v1", lambda value: value),
        contributors=(
            StationaryHVPContributor("a", "a-v1", lambda context: context.direction),
        ),
    )
    with pytest.raises(ValueError, match="HVP direction"):
        executor.apply(direction)
    assert calls == []


def test_execution_identity_tracks_plan_and_adapter_semantics() -> None:
    plan = SimpleNamespace(identity="plan-v1", source_names=("a",))
    contributor = StationaryHVPContributor(
        "a", "component-v1", lambda context: context.direction
    )
    left = StationarySecondOrderExecutor(
        plan,
        natoms=1,
        perturbation=StationaryPerturbationProvider("rhs-v1", lambda value: value),
        response=StationaryResponseDriver("response-v1", lambda value: value),
        contributors=(contributor,),
    )
    right = StationarySecondOrderExecutor(
        plan,
        natoms=1,
        perturbation=StationaryPerturbationProvider("rhs-v2", lambda value: value),
        response=StationaryResponseDriver("response-v1", lambda value: value),
        contributors=(contributor,),
    )
    assert left.identity != right.identity
    assert (
        left.apply([[1.0, 0.0, 0.0]]).identity != left.apply([[0.0, 1.0, 0.0]]).identity
    )
