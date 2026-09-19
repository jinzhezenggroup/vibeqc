"""Explicit generated CUDA actions with the existing host-controlled GMRES.

Six TensorIR stages share one admitted resource plan. This is not a native
Krylov loop or a device-resident molecular force owner: each generated action
uses the existing CUDA staging boundary. Compilation/preparation is explicit;
there is no CPU scientific fallback and no provider allowance is waived.
"""

from __future__ import annotations

from dataclasses import replace

from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.common.resources import (
    ResourceBudget,
    ResourceSession,
    plan_resources,
)
from vibeqc_compiler.method import ImplicitVJPPlan
from vibeqc_compiler.tensor.resources import tensor_resource_choices

from .implicit import BoundImplicitState, ResponseGMRES


class PreparedImplicitCuda:
    """Own generated native stages under one simultaneous device/host budget.

    ``budget`` includes the explicit host snapshot/solver reservation. Context,
    module loading, Python objects, allocator rounding and opaque host BLAS
    scratch remain outside numeric-buffer accounting. No speedup or full
    process peak-memory qualification is implied.
    """

    backend = "cuda-fp64-ordinary-stream"

    def __init__(self, plan, solver, compiler, cache, *, budget, device=0):
        if not isinstance(plan, ImplicitVJPPlan) or not isinstance(
            solver, ResponseGMRES
        ):
            raise TypeError(
                "CUDA implicit execution requires a plan and the host ResponseGMRES callback"
            )
        if solver.dimension != plan.spec.dimension:
            raise ValueError("implicit CUDA solver dimension mismatch")
        if (
            not isinstance(budget, ResourceBudget)
            or budget.host_bytes is None
            or budget.device_bytes is None
        ):
            raise ValueError("implicit CUDA requires explicit host and device budgets")
        self._session = None
        self._closed = True
        self.plan = plan
        self.solver = solver
        self.plan_identity = plan.identity
        self.host_capacity = budget.limits()["host"]
        reserved = plan.reference_workspace_bytes + solver.workspace_bytes
        effective_budget = replace(
            budget, host_reserve_bytes=budget.host_reserve_bytes + reserved
        )
        choices = {
            stage: tensor_resource_choices(
                program,
                compiler.target,
                name=stage,
                device=device,
                allow_recompute=False,
            )
            for stage, program in plan.programs.items()
        }
        admission = plan_resources(
            [value.request for value in choices.values()], effective_budget
        ).require_feasible()
        self.resource_plan = admission
        self._session = ResourceSession(
            admission,
            {
                stage: value.factory(compiler, cache, device=device)
                for stage, value in choices.items()
            },
        )
        try:
            self._session.advance(0)
            # ResourceSession owns retry/rollback. Identify its actual selected
            # plan and providers, not an earlier candidate admission.
            self.resource_plan = self._session.plan
            self.workspace_bytes = sum(
                self._session.provider(stage).plan.host_bytes for stage in choices
            )
            self.identity = canonical_hash(
                {
                    "implicit": plan.identity,
                    "solver": solver.identity,
                    "resources": self.resource_plan.identity,
                    "providers": {
                        stage: self._session.provider(stage).identity
                        for stage in choices
                    },
                    "backend": self.backend,
                }
            )
            self._closed = False
            self.last_metrics = {}
        except BaseException:
            self._session.close()
            raise

    def bind(
        self, feeds, *, reference_identity, current_reference=None, primal_atol=1e-10
    ):
        """Bind a fresh host snapshot to these already-admitted CUDA programs."""
        if self._closed:
            raise RuntimeError("implicit CUDA executor is closed")
        return BoundImplicitState(
            self.plan,
            feeds,
            reference_identity=reference_identity,
            solver=self.solver,
            executor=self,
            current_reference=current_reference,
            primal_atol=primal_atol,
            max_bytes=self.host_capacity,
        )

    def execute(self, stage, feeds):
        """Execute the generated native program; never call a CPU interpreter."""
        if self._closed:
            raise RuntimeError("implicit CUDA executor is closed")
        if stage not in self.plan.programs:
            raise ValueError("unknown implicit CUDA stage")
        result = self._session.provider(stage).execute(feeds)
        self.last_metrics[stage] = result.metrics
        return result

    def close(self):
        """Release every prepared provider, including after an execution failure."""
        if self._session is not None:
            self._session.close()
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("implicit CUDA executor is closed")
        return self

    def __exit__(self, *unused):
        self.close()
