"""Adapt existing TensorIR storage plans to the common resource budget.

No CUDA call, compilation or tensor allocation occurs here. Alternatives are
produced by TensorIR's own liveness/tiling planner; the global planner chooses
among those actual executable plans and never edits the tensor equation.
"""

import json
from dataclasses import asdict, dataclass, replace

from vibeqc.resources import (
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourceRequest,
    checked_bytes,
)

from .cuda_plan import TensorSchedule, plan_cuda


@dataclass(frozen=True)
class TensorResourceChoices:
    """Immutable provider request and the exact corresponding TensorIR plans."""

    request: ResourceRequest
    plans: tuple

    def selected(self, resource_plan):
        """Reject foreign or modified requests before loading a native artifact."""
        resource_plan.require_feasible()
        requests = {r.name: r for r in resource_plan.requests}
        if requests.get(self.request.name) != self.request:
            raise ValueError(
                "global plan does not contain this tensor provider request"
            )
        name = dict(resource_plan.selections)[self.request.name]
        return dict(self.plans)[name]

    def prepare(self, resource_plan, artifact, *, device=0):
        """Allocate the selected provider under the stored complete resource plan.

        The orchestrator must respect its declared lifetime and close the
        object before a later phase reuses the capacity. The existing native
        provider verifies allocation bytes and retained library capacity.
        """
        from .cuda_execute import PreparedCuda

        plan = self.selected(resource_plan)
        expected_device = json.loads(self.request.identity.topology)["device"]
        if device != expected_device:
            raise ValueError("tensor resource plan device differs from execution")
        return PreparedCuda(
            plan,
            artifact,
            device=device,
            resource_plan=resource_plan,
            resource_owner=self.request.name,
        )

    def factory(self, compiler, cache_root, *, device=0):
        """Bind this provider to a ResourceSession, including retry artifacts.

        Compilation selects the exact enumerated plan and does not execute
        the tensor equation. Native preparation releases partial CUDA objects
        before reporting typed allocation failures to the session.
        """
        from .cuda_execute import compile_cuda

        def prepare_selected(resource_plan):
            plan = self.selected(resource_plan)
            artifact = compile_cuda(plan, compiler, cache_root)
            return self.prepare(resource_plan, artifact, device=device)

        return prepare_selected


def tensor_resource_choices(
    program,
    target,
    *,
    name="tensor",
    first_phase=0,
    last_phase=0,
    device=0,
    sub_budget_bytes=256 << 20,
    schedule=None,
    allow_recompute=True,
    **planner_options,
):
    """Obtain finite tile/recompute alternatives within an explicit sub-budget.

    The outer budget still accounts for all concurrently live providers. A
    locally fitting plan is only a candidate. Full input/output tensors stay
    resident, as required by the existing provider; panel tiling is not
    described as out-of-core output support. The sub-budget has TensorIR's
    established combined host/device numeric-buffer meaning.
    """
    checked_bytes(device, "device ordinal")
    checked_bytes(sub_budget_bytes, "tensor sub-budget")
    if type(allow_recompute) is not bool:
        raise ValueError("allow_recompute must be boolean")
    schedule = TensorSchedule() if schedule is None else schedule
    schedules = [schedule]
    for tile in (32, 8, 1):
        smaller = replace(
            schedule,
            tile_m=min(tile, schedule.tile_m),
            tile_n=min(tile, schedule.tile_n),
            tile_k=min(tile, schedule.tile_k),
        )
        if smaller not in schedules:
            schedules.append(smaller)
    if allow_recompute:
        schedules += [
            replace(s, recompute=True) for s in tuple(schedules) if not s.recompute
        ]
    candidates, plans, failures = [], [], []
    seen = set()
    for preference, variant in enumerate(schedules):
        try:
            plan = plan_cuda(
                program,
                target,
                max_bytes=sub_budget_bytes,
                schedule=variant,
                **planner_options,
            )
        except ValueError as error:
            # Only the specialized planner's explicit infeasibility is a
            # memory alternative failure. Invalid equations/options propagate.
            if "infeasible" not in str(error):
                raise
            failures.append(str(error))
            continue
        if plan.identity in seen:
            continue
        seen.add(plan.identity)
        candidate_name = f"tensor-{preference}"
        estimates = (
            ResourceEstimate(
                "native arena and workspaces",
                plan.allocation_bytes,
                f"device:{device}",
                first_phase,
                last_phase,
                kind="persistent",
                recomputable=plan.schedule.recompute,
            ),
            ResourceEstimate(
                "retained library allowance",
                plan.provider_bytes,
                f"device:{device}",
                first_phase,
                last_phase,
                kind="library",
                accounting="runtime_allowance",
            ),
            ResourceEstimate(
                "host staging, validation and detached output",
                plan.host_bytes,
                "pageable",
                first_phase,
                last_phase,
                kind="persistent",
            ),
        )
        mode = (
            "recomputed"
            if plan.schedule.recompute
            else "tiled"
            if plan.panel_bytes
            else "resident"
        )
        candidates.append(
            ResourceCandidate(
                candidate_name,
                mode,
                estimates,
                relative_cost=preference,
                decisions=(("tensor_plan", plan.identity),),
            )
        )
        plans.append((candidate_name, plan))
    identity = ResourceIdentity(
        "tensor_contraction",
        "tensorir-cuda",
        "cuda",
        "fp64",
        json.dumps(
            {
                "equation": program.logical_hash,
                "target": target.to_payload(),
                "device": device,
                "outputs": list(program.outputs),
            }
        ),
        ("tensor_outputs",),
        json.dumps(asdict(schedule), sort_keys=True),
    )
    request = ResourceRequest(
        name,
        identity,
        tuple(candidates),
        scope_exclusions=(
            "caller-owned inputs and previously retained outputs",
            "Python/code objects, CUDA context/module/stack and allocator rounding",
        ),
        infeasible_reason=(
            "no TensorIR candidate fits the provider sub-budget: " + "; ".join(failures)
        )
        if not candidates
        else None,
    )
    return TensorResourceChoices(request, tuple(plans))
