"""Bounded structured solver regions, strict replay and derivative boundaries."""

import copy
import json
from dataclasses import replace

import pytest
from vibeqc_compiler.common.program import PlanCall, ProgramBuffer, ProgramIR
from vibeqc_compiler.common.resources import ResourceBudget, plan_resources
from vibeqc_compiler.common.solver_region import (
    RegionCarry,
    RegionCheckpoint,
    RegionCompletion,
    RegionDerivative,
    RegionPredicate,
    SolverRegion,
)


def body() -> ProgramIR:
    return ProgramIR(
        "iteration",
        (
            ProgramBuffer("parameters", 64),
            ProgramBuffer("state", 32),
            ProgramBuffer("trial", 32),
            ProgramBuffer("next_state", 32),
            ProgramBuffer("converged", 1),
            ProgramBuffer("failed", 1),
            ProgramBuffer("observable", 8),
        ),
        ("parameters", "state"),
        (
            PlanCall(
                "step",
                "solver.step",
                "equation-v1",
                ("parameters", "state"),
                ("trial",),
            ),
            PlanCall(
                "control",
                "solver.control",
                "policy-v1",
                ("parameters", "state", "trial"),
                ("next_state", "converged", "failed", "observable"),
            ),
        ),
        ("next_state", "converged", "failed", "observable"),
    )


def region() -> SolverRegion:
    return SolverRegion(
        "bounded-test",
        body(),
        ("parameters",),
        (RegionCarry("state", "next_state"),),
        ("next_state", "observable"),
        12,
        RegionPredicate("converged", "energy+residual-v1"),
        RegionPredicate("failed", "finite-and-provider-status-v1"),
        checkpoints=(
            RegionCheckpoint("control", "iteration", ("observable",), True),
            RegionCheckpoint("publish", "success", ("next_state",), False),
            RegionCheckpoint("failure", "failure", ("failed", "next_state"), True),
        ),
        derivatives=(RegionDerivative("implicit_vjp", "implicit-plan-v1"),),
        derivative_policy="custom",
    )


def test_roundtrip_identity_is_strict_and_control_semantics_are_hashed() -> None:
    item = region()
    payload = json.loads(json.dumps(item.to_payload()))
    assert SolverRegion.from_payload(item.to_payload()) == item
    replayed = SolverRegion.from_payload(payload)
    assert replayed == item
    assert replayed.identity == item.identity

    assert replace(item, max_steps=13).identity != item.identity
    assert (
        replace(item, converged=RegionPredicate("converged", "other-policy")).identity
        != item.identity
    )
    assert (
        replace(
            item,
            checkpoints=(
                replace(item.checkpoints[0], host_visible=False),
                *item.checkpoints[1:],
            ),
        ).identity
        != item.identity
    )
    assert (
        replace(
            item,
            derivatives=(RegionDerivative("implicit_vjp", "implicit-plan-v2"),),
        ).identity
        != item.identity
    )


def test_derivatives_are_registered_explicitly_and_fail_closed() -> None:
    item = region()
    assert item.derivative_policy == "custom"
    assert item.derivative_rule("implicit_vjp").identity == "implicit-plan-v1"
    with pytest.raises(NotImplementedError, match="not registered"):
        item.derivative_rule("implicit_jvp")
    with pytest.raises(NotImplementedError, match="order 2"):
        item.derivative_rule("implicit_vjp", order=2)
    with pytest.raises(ValueError, match="unsupported"):
        item.derivative_rule("tape_vjp")
    unsupported = replace(item, derivatives=(), derivative_policy="unsupported")
    with pytest.raises(NotImplementedError, match="not registered"):
        unsupported.derivative_rule("implicit_vjp")
    with pytest.raises(ValueError, match="cannot register"):
        replace(item, derivative_policy="unsupported")
    with pytest.raises(ValueError, match="requires registered"):
        replace(item, derivatives=())


def test_body_resources_are_reused_across_bounded_steps() -> None:
    item = region()
    request = item.resource_request()
    planned = plan_resources((request,), ResourceBudget())
    longer = plan_resources(
        (replace(item, max_steps=120).resource_request(),), ResourceBudget()
    )
    assert planned.peak_bytes == longer.peak_bytes
    assert request.candidates[0].decisions == (
        ("max_steps", "12"),
        ("completion", "scalar"),
    )
    lifetimes = {value.name: value for value in item.lifetimes()}
    assert lifetimes["parameters"].kind == "persistent"
    assert lifetimes["state"].kind == "persistent"
    assert lifetimes["next_state"].kind == "output"
    assert lifetimes["trial"].kind == "workspace"


def test_all_body_inputs_and_outputs_have_explicit_region_roles() -> None:
    item = region()
    with pytest.raises(ValueError, match="exactly invariants"):
        replace(item, invariants=())
    with pytest.raises(ValueError, match="explicit region role"):
        replace(item, results=("next_state",))
    with pytest.raises(ValueError, match="non-output"):
        replace(item, converged=RegionPredicate("trial", "wrong"))
    with pytest.raises(ValueError, match="duplicate"):
        replace(
            item,
            carries=(
                RegionCarry("state", "next_state"),
                RegionCarry("state", "observable"),
            ),
        )
    changed = tuple(
        replace(buffer, bytes=16) if buffer.name == "next_state" else buffer
        for buffer in item.body.buffers
    )
    with pytest.raises(ValueError, match="identical contracts"):
        replace(item, body=replace(item.body, buffers=changed))


def test_checkpoint_and_completion_boundaries_are_explicit() -> None:
    item = region()
    with pytest.raises(ValueError, match="failure checkpoint"):
        replace(item, failed=None, results=("next_state", "observable", "failed"))
    with pytest.raises(ValueError, match="undeclared"):
        replace(
            item,
            checkpoints=(RegionCheckpoint("bad", "exit", ("missing",), True),),
        )
    with pytest.raises(ValueError, match="entry checkpoint"):
        replace(
            item,
            checkpoints=(RegionCheckpoint("bad", "entry", ("observable",), True),),
        )
    with pytest.raises(ValueError, match="post-step checkpoint"):
        replace(
            item,
            checkpoints=(RegionCheckpoint("bad", "exit", ("parameters",), True),),
        )
    with pytest.raises(ValueError, match="active mask"):
        RegionCompletion("scalar", "mask")
    with pytest.raises(ValueError, match="non-output"):
        replace(item, completion=RegionCompletion("per_item_mask", "trial"))
    with pytest.raises(TypeError, match="bool"):
        RegionCheckpoint("bad", "exit", ("next_state",), 1)


def test_per_item_completion_uses_an_explicit_body_output_mask() -> None:
    item = region()
    masked_body = replace(
        item.body,
        buffers=(*item.body.buffers, ProgramBuffer("active", 4)),
        calls=(
            item.body.calls[0],
            replace(
                item.body.calls[1],
                writes=(*item.body.calls[1].writes, "active"),
            ),
        ),
        outputs=(*item.body.outputs, "active"),
    )
    masked = replace(
        item,
        body=masked_body,
        completion=RegionCompletion("per_item_mask", "active"),
    )
    assert masked.completion.active_mask == "active"
    assert SolverRegion.from_payload(masked.to_payload()) == masked


@pytest.mark.parametrize("max_steps", [0, -1, True, 1.5])
def test_iteration_bound_is_positive_integer(max_steps: object) -> None:
    with pytest.raises(ValueError, match="max_steps"):
        replace(region(), max_steps=max_steps)


def test_untrusted_replay_rejects_unknown_or_noncanonical_fields() -> None:
    payload = copy.deepcopy(region().to_payload())
    payload["extra"] = True
    with pytest.raises(ValueError, match="fields"):
        SolverRegion.from_payload(payload)

    payload = copy.deepcopy(region().to_payload())
    payload["carries"][0]["extra"] = True
    with pytest.raises(ValueError, match="fields"):
        SolverRegion.from_payload(payload)

    payload = copy.deepcopy(region().to_payload())
    payload["checkpoints"][0]["buffers"] = "observable"
    with pytest.raises(TypeError, match="array"):
        SolverRegion.from_payload(payload)

    payload = copy.deepcopy(region().to_payload())
    payload["body"]["calls"][0]["identity"] = ""
    with pytest.raises(ValueError):
        SolverRegion.from_payload(payload)
