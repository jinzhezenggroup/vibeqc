"""TensorIR alternatives must fit the whole plan before any device access."""

import json

import pytest
from vibeqc.resources import (
    ResourceBudget,
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourceRequest,
    plan_resources,
)
from vibeqc_compiler.integral.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    input_tensor,
    reduce_sum,
)
from vibeqc_compiler.tensor.resources import tensor_resource_choices


def fragment():
    index = Index("i", IndexSpace("axis", "batch", 8192))
    x = input_tensor("x", TensorSpec((index,), role="input"))
    return Program(
        {
            f"r{i}": reduce_sum(add(x, x, coefficients=(1, i + 1)), (0,))
            for i in range(6)
        }
    )


def test_tensor_global_budget_selects_real_recompute_plan_without_device(monkeypatch):
    import ctypes

    def no_device(*args, **kwargs):
        pytest.fail("planning must not load a device library")

    monkeypatch.setattr(ctypes, "CDLL", no_device)
    choices = tensor_resource_choices(
        fragment(), cuda_target_info("sm_120"), first_phase=1, last_phase=2
    )
    fast = choices.plans[0][1]
    recomputed = min(
        (p for _, p in choices.plans if p.schedule.recompute),
        key=lambda p: p.device_bytes,
    )
    assert recomputed.device_bytes < fast.device_bytes
    retained = 4 << 20
    state = ResourceRequest(
        "hf-state",
        ResourceIdentity(
            "rhf",
            "native-hf",
            "cuda",
            "fp64",
            json.dumps({"nbf": 128}),
            ("energy",),
            "fixed",
        ),
        (
            ResourceCandidate(
                "fixed",
                "resident",
                (
                    ResourceEstimate(
                        "density", retained, "device:0", 0, 3, kind="persistent"
                    ),
                ),
            ),
        ),
    )
    budget = ResourceBudget(
        host_bytes=fast.host_bytes, device_bytes=retained + recomputed.device_bytes
    )
    global_plan = plan_resources([state, choices.request], budget)
    selected = choices.selected(global_plan)
    assert selected.schedule.recompute
    assert global_plan.peak_bytes["device"] <= budget.device_bytes
    assert global_plan.peak_bytes["device"] == retained + selected.device_bytes
    assert (
        plan_resources(
            [state, choices.request],
            ResourceBudget(
                host_bytes=fast.host_bytes,
                device_bytes=retained + recomputed.device_bytes - 1,
            ),
        ).status
        == "infeasible"
    )


def test_tensor_indivisible_minimum_is_an_infeasible_provider_request():
    choices = tensor_resource_choices(
        fragment(), cuda_target_info("sm_120"), sub_budget_bytes=1
    )
    plan = plan_resources([choices.request], ResourceBudget(device_bytes=1))
    assert plan.status == "infeasible"
    assert "provider sub-budget" in plan.diagnostic


def test_tensor_foreign_plan_is_rejected_before_artifact_loading():
    choices = tensor_resource_choices(fragment(), cuda_target_info("sm_120"))
    foreign = tensor_resource_choices(fragment(), cuda_target_info("sm_120"), device=1)
    plan = plan_resources([foreign.request], ResourceBudget())
    with pytest.raises(ValueError, match="this tensor provider"):
        choices.prepare(plan, None)
