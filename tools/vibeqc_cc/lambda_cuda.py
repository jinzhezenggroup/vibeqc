"""CUDA execution owner for generated RCCSD Lambda actions.

The scientific equations remain owned by lambda_equations. This module prepares
shared and expanded primal/RHS/transpose TensorIR programs under one explicit
host/device resource plan, then reuses #179's host GMRES control. It does not
claim a device-resident Krylov loop or a public force capability.

Rationale: .agents/notes/implemented/architecture/2026-09-20-generated-cuda-lambda-owner.md
"""

from __future__ import annotations

import typing
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
from typing_extensions import Self
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.common.resources import (
    ResourceBudget,
    ResourceSession,
    plan_resources,
)
from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.tensor.resources import tensor_resource_choices

from tools.vibeqc_response.implicit import (
    ImplicitSolveError,
    _immutable,
    checked_transpose_solve,
)
from tools.vibeqc_response.krylov import _vector_norm
from tools.vibeqc_response.problem import ResponseCompatibilityError

from .lambda_solver import BoundCCSDLambda, CCSDLambdaResult, LambdaOptions


class PreparedCUDALambda:
    """Prepare and execute generated Lambda actions on one CUDA device.

    The CC/reference snapshot and solver contract are first bound by the
    existing CPU oracle. All Lambda RHS/J^T actions and both final stationarity
    checks then execute through generated CUDA plans with no CPU scientific
    fallback. The GMRES iteration remains the shared host controller.
    """

    backend = "cuda-fp64-ordinary-stream"

    def __init__(
        self,
        snapshot: typing.Any,
        cc_result: typing.Any,
        compiler: typing.Any,
        cache: typing.Any,
        *,
        budget: typing.Any,
        options: typing.Any = None,
        solver: typing.Any = None,
        current_reference: typing.Any = None,
        device: typing.Any = 0,
    ) -> None:
        if not isinstance(compiler, CudaCompilerAdapter):
            raise TypeError("CUDA Lambda requires a CudaCompilerAdapter")
        if not isinstance(cache, Path):
            raise TypeError("CUDA Lambda cache must be a pathlib.Path")
        if type(device) is not int or device < 0:
            raise ValueError("CUDA Lambda device must be a nonnegative ordinal")
        if (
            not isinstance(budget, ResourceBudget)
            or budget.host_bytes is None
            or budget.device_bytes is None
        ):
            raise ValueError("CUDA Lambda requires explicit host and device budgets")

        self.bound = BoundCCSDLambda(
            snapshot,
            cc_result,
            options=LambdaOptions() if options is None else options,
            solver=solver,
            current_reference=current_reference,
        )
        self.compiler = compiler
        self.cache = cache
        self.device = device
        self._closed = True
        self._session = None
        self.last_metrics: dict[str, typing.Any] = {}
        self.execution_metrics: dict[str, dict[str, int]] = {}

        programs = {
            "shared_primal": self.bound.programs.primal,
            "shared_rhs": self.bound.programs.energy_vjp.program,
            "shared_transpose": self.bound.programs.residual_vjp.program,
            "independent_primal": self.bound.independent.primal,
            "independent_rhs": self.bound.independent.energy_vjp.program,
            "independent_transpose": self.bound.independent.residual_vjp.program,
        }
        limits = budget.limits()
        sub_budget = min(limits["host"], limits["device"])
        if sub_budget <= 0:
            raise ImplicitSolveError("CUDA Lambda budget leaves no tensor capacity")
        choices = {
            name: tensor_resource_choices(
                program,
                compiler.target,
                name=name,
                device=device,
                sub_budget_bytes=sub_budget,
                allow_recompute=False,
            )
            for name, program in programs.items()
        }
        effective_budget = replace(
            budget,
            host_reserve_bytes=(
                budget.host_reserve_bytes + self.bound.logical_reserved_host_bytes
            ),
        )
        admission = plan_resources(
            [choice.request for choice in choices.values()], effective_budget
        ).require_feasible()
        self.resource_plan = admission
        self._session = ResourceSession(
            admission,
            {
                name: choice.factory(compiler, cache, device=device)
                for name, choice in choices.items()
            },
        )
        try:
            self._session.advance(0)
            self.resource_plan = self._session.plan
            self.identity = canonical_hash(
                {
                    "bound_equations": self.bound.equation_identity,
                    "reference": self.bound.reference_identity,
                    "cc_state": self.bound.cc_state_identity,
                    "solver": dict(self.bound._solver_contract),
                    "resources": self.resource_plan.identity,
                    "providers": {
                        name: self._session.provider(name).identity for name in programs
                    },
                    "backend": self.backend,
                }
            )
            self._closed = False
            self._validate_primal_cuda(cc_result)
        except BaseException:
            self._session.close()
            self._closed = True
            raise

    @property
    def logical_reserved_host_bytes(self) -> int:
        return self.bound.logical_reserved_host_bytes + int(
            self.resource_plan.peak_bytes.get("host", 0)
        )

    def _assert_current(self, reference_identity: str) -> None:
        if self._closed or self._session is None:
            raise RuntimeError("CUDA Lambda owner is closed")
        self.bound._assert_current(reference_identity)

    def _execute(
        self,
        stage: str,
        extra: typing.Mapping[str, typing.Any] | None = None,
    ) -> typing.Mapping[str, np.ndarray]:
        self._assert_current(self.bound.reference_identity)
        result = self._session.provider(stage).execute(
            {**self.bound.feeds, **({} if extra is None else dict(extra))}
        )
        if result.backend != self.backend:
            raise ResponseCompatibilityError(
                "CUDA Lambda tensor backend changed; no CPU fallback allowed"
            )
        self.last_metrics[stage] = dict(result.metrics)
        counters = self.execution_metrics.setdefault(
            stage,
            {
                "calls": 0,
                "observed_host_to_device_bytes": 0,
                "observed_device_to_host_bytes": 0,
                "peak_tracked_device_bytes": 0,
                "peak_tracked_host_bytes": 0,
            },
        )
        counters["calls"] += 1
        counters["observed_host_to_device_bytes"] += int(
            result.metrics.get("observed_host_to_device_bytes", 0)
        )
        counters["observed_device_to_host_bytes"] += int(
            result.metrics.get("observed_device_to_host_bytes", 0)
        )
        counters["peak_tracked_device_bytes"] = max(
            counters["peak_tracked_device_bytes"],
            int(result.metrics.get("tracked_device_bytes", 0)),
        )
        counters["peak_tracked_host_bytes"] = max(
            counters["peak_tracked_host_bytes"],
            int(result.metrics.get("tracked_host_bytes", 0)),
        )
        self._assert_current(self.bound.reference_identity)
        return result.outputs

    def _validate_primal_cuda(self, cc_result: typing.Any) -> None:
        outputs = [
            self._execute("shared_primal"),
            self._execute("independent_primal"),
        ]
        for out in outputs:
            energy = float(out["correlation_energy"])
            if abs(energy - float(cc_result.correlation_energy)) > 1e-10:
                raise ResponseCompatibilityError(
                    "CUDA Lambda primal energy differs from converged CC state"
                )
            if (
                max(
                    float(np.max(np.abs(out["singles_residual"]))),
                    float(np.max(np.abs(out["doubles_residual"]))),
                )
                > self.bound.options.cc_tolerance
            ):
                raise ImplicitSolveError(
                    "CUDA Lambda primal state fails physical CC residual gate"
                )
        for name in ("singles_residual", "doubles_residual"):
            if not np.allclose(
                outputs[0][name], outputs[1][name], atol=1e-11, rtol=1e-10
            ):
                raise ImplicitSolveError(
                    "shared/expanded CUDA CC primal replay disagrees"
                )

    def _rhs(self, prefix: str) -> np.ndarray:
        out = self._execute(
            f"{prefix}_rhs", {"bar_correlation_energy": np.asarray(-1.0)}
        )
        return self.bound.sqrt_weights * self.bound._pack(
            (out["bar_t1"], out["bar_t2"])
        )

    def _transpose(self, prefix: str, vector: np.ndarray) -> np.ndarray:
        l1, l2 = self.bound._unpack(vector / self.bound.sqrt_weights)
        out = self._execute(
            f"{prefix}_transpose",
            {"bar_singles_residual": l1, "bar_doubles_residual": l2},
        )
        return self.bound.sqrt_weights * self.bound._pack(
            (out["bar_t1"], out["bar_t2"])
        )

    def solve(self, *, reference_identity: str) -> CCSDLambdaResult:
        """Solve Lambda with generated CUDA actions and independent GPU replay."""
        with self.bound._lock:
            self._assert_current(reference_identity)
            owner = self

            class Operator:
                dimension = len(owner.bound.sqrt_weights)

                def apply(self, vector: typing.Any) -> np.ndarray:
                    return owner._transpose("shared", np.asarray(vector))

            result = checked_transpose_solve(
                Operator(),
                self._rhs("shared"),
                solver=self.bound.solver,
                assert_current=lambda: self._assert_current(reference_identity),
            )
            independent = self._transpose("independent", result.solution) - self._rhs(
                "independent"
            )
            independent_norm = _vector_norm(independent)
            independent_max = float(
                np.max(np.abs(independent / self.bound.sqrt_weights))
            )
            if (
                max(result.residual_norm, independent_norm, independent_max)
                > self.bound.options.lambda_tolerance
            ):
                raise ImplicitSolveError(
                    "CUDA Lambda independent physical residual gate failed"
                )
            l1, l2 = self.bound._unpack(result.solution / self.bound.sqrt_weights)
            self._assert_current(reference_identity)
            transfer = {
                stage: dict(row) for stage, row in self.execution_metrics.items()
            }
            return CCSDLambdaResult(
                _immutable(l1),
                _immutable(l2),
                self.bound.reference_identity,
                self.bound.cc_state_identity,
                self.bound.reference.scf_residual,
                self.bound.cc_r1_max,
                self.bound.cc_r2_max,
                self.bound.cc_residual_norm,
                result.residual_norm,
                independent_norm,
                independent_max,
                result.iterations,
                result.operator_actions,
                self.logical_reserved_host_bytes,
                MappingProxyType(
                    {
                        "equation_identity": self.bound.equation_identity,
                        "lagrangian": "E_corr + <lambda, R>",
                        "inner_product": (
                            "dense Frobenius; sqrt-orbit-weighted independent "
                            "solver coordinates"
                        ),
                        "tensor_backend": self.backend,
                        "solver_backend": self.bound._solver_contract["backend"],
                        "execution_owner_identity": self.identity,
                        "resource_plan_identity": self.resource_plan.identity,
                        "resource_peak_bytes": dict(self.resource_plan.peak_bytes),
                        "transfer_metrics": transfer,
                        "reference_binding": (
                            "live-reference-callback"
                            if self.bound._current_reference is not None
                            else "detached-immutable-snapshot"
                        ),
                        "scope": (
                            "generated CUDA amplitude response with host-controlled "
                            "GMRES; no resident Krylov, RDM or nuclear-force capability"
                        ),
                    }
                ),
            )

    def close(self) -> None:
        if self._session is not None:
            self._session.close()
        self._closed = True

    def __enter__(self) -> Self:
        if self._closed:
            raise RuntimeError("CUDA Lambda owner is closed")
        return self

    def __exit__(self, *unused: object) -> None:
        self.close()
