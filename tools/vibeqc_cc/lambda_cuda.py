"""CUDA execution owner for generated RCCSD Lambda actions.

The scientific equations remain owned by lambda_equations. Shared/expanded
primal, RHS and transpose TensorIR programs are planned under one explicit
host/device budget. The repeated shared J^T action keeps the bound CC inputs
resident on device while #179's checked GMRES remains the host controller.
This is not a device-resident Krylov loop or a public force capability.

Rationale: .agents/notes/implemented/architecture/2026-09-20-generated-cuda-lambda-owner.md
Resident phase/transfer contract: .agents/notes/implemented/architecture/2026-09-20-resident-lambda-actions.md
"""

from __future__ import annotations

import typing
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.common.resources import (
    ResourceBudget,
    ResourceSession,
    plan_resources,
)
from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.tensor.cuda_resident import PreparedResident, compile_resident
from vibeqc_compiler.tensor.resources import tensor_resource_choices

from tools.vibeqc_response.implicit import (
    ImplicitSolveError,
    _immutable,
    checked_transpose_solve,
)
from tools.vibeqc_response.krylov import _vector_norm
from tools.vibeqc_response.problem import ResponseCompatibilityError

from .lambda_solver import BoundCCSDLambda, CCSDLambdaResult, LambdaOptions

if typing.TYPE_CHECKING:
    from typing_extensions import Self


class PreparedCUDALambda:
    """Own generated CUDA Lambda actions with a resident repeated J^T stage.

    The existing CPU consumer first verifies the immutable reference/CC state
    and solver contract. The two primal checks execute on generated CUDA in
    phase 0. Phase 1 releases those owners, prepares RHS/final-check programs,
    and retains the shared transpose program with its CC/Fock/integral inputs
    on device across all GMRES actions. Only the changing Lambda cotangent and
    resulting T cotangent cross the host boundary per action.

    GMRES itself remains the shared #179 host implementation. Provenance and
    transfer counters therefore distinguish resident *action state* from a
    fully device-resident Krylov solver.
    """

    backend = "cuda-fp64-resident-actions"
    ordinary_stage_backend = "cuda-fp64-ordinary-stream"

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
        self.execution_metrics: dict[str, dict[str, typing.Any]] = {}
        self._resident_initial_h2d_bytes = 0
        self._resident_action_input_bytes = 0
        self._resident_action_output_bytes = 0

        programs = {
            "shared_primal": self.bound.programs.primal,
            "shared_rhs": self.bound.programs.energy_vjp.program,
            "shared_transpose": self.bound.programs.residual_vjp.program,
            "independent_primal": self.bound.independent.primal,
            "independent_rhs": self.bound.independent.energy_vjp.program,
            "independent_transpose": self.bound.independent.residual_vjp.program,
        }
        self._program_hashes = {
            name: program.logical_hash for name, program in programs.items()
        }
        limits = budget.limits()
        sub_budget = min(limits["host"], limits["device"])
        if sub_budget <= 0:
            raise ImplicitSolveError("CUDA Lambda budget leaves no tensor capacity")

        choices = {}
        for name, program in programs.items():
            phase = 0 if name.endswith("_primal") else 1
            choices[name] = tensor_resource_choices(
                program,
                compiler.target,
                name=name,
                first_phase=phase,
                last_phase=phase,
                device=device,
                sub_budget_bytes=sub_budget,
                allow_recompute=False,
            )
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

        factories = {}
        for name, choice in choices.items():
            if name != "shared_transpose":
                factories[name] = choice.factory(compiler, cache, device=device)
                continue

            def prepare_resident(
                resource_plan: typing.Any,
                *,
                _choice: typing.Any = choice,
                _name: str = name,
            ) -> PreparedResident:
                plan = _choice.selected(resource_plan)
                artifact = compile_resident(plan, compiler, cache)
                return PreparedResident(
                    plan,
                    artifact,
                    device=device,
                    resource_plan=resource_plan,
                    resource_owner=_name,
                )

            factories[name] = prepare_resident

        self._session = ResourceSession(admission, factories)
        try:
            self._session.advance(0)
            self.resource_plan = self._session.plan
            self._closed = False
            self._validate_primal_cuda(cc_result)
            self._session.advance(1)
            self.resource_plan = self._session.plan
            self._prepare_resident_transpose()
            self.identity = canonical_hash(
                {
                    "bound_equations": self.bound.equation_identity,
                    "reference": self.bound.reference_identity,
                    "cc_state": self.bound.cc_state_identity,
                    "solver": dict(self.bound._solver_contract),
                    "programs": self._program_hashes,
                    "resources": self.resource_plan.identity,
                    "live_providers": {
                        name: self._session.provider(name).identity
                        for name in (
                            "shared_rhs",
                            "shared_transpose",
                            "independent_rhs",
                            "independent_transpose",
                        )
                    },
                    "backend": self.backend,
                }
            )
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

    def _record_ordinary_metrics(
        self, stage: str, metrics: typing.Mapping[str, typing.Any]
    ) -> None:
        self.last_metrics[stage] = dict(metrics)
        counters = self.execution_metrics.setdefault(
            stage,
            {
                "mode": "ordinary-stream",
                "calls": 0,
                "observed_host_to_device_bytes": 0,
                "observed_device_to_host_bytes": 0,
                "peak_tracked_device_bytes": 0,
                "peak_tracked_host_bytes": 0,
            },
        )
        counters["calls"] += 1
        counters["observed_host_to_device_bytes"] += int(
            metrics.get("observed_host_to_device_bytes", 0)
        )
        counters["observed_device_to_host_bytes"] += int(
            metrics.get("observed_device_to_host_bytes", 0)
        )
        counters["peak_tracked_device_bytes"] = max(
            int(counters["peak_tracked_device_bytes"]),
            int(metrics.get("tracked_device_bytes", 0)),
        )
        counters["peak_tracked_host_bytes"] = max(
            int(counters["peak_tracked_host_bytes"]),
            int(metrics.get("tracked_host_bytes", 0)),
        )

    def _execute(
        self,
        stage: str,
        extra: typing.Mapping[str, typing.Any] | None = None,
    ) -> typing.Mapping[str, np.ndarray]:
        self._assert_current(self.bound.reference_identity)
        result = self._session.provider(stage).execute(
            {**self.bound.feeds, **({} if extra is None else dict(extra))}
        )
        if result.backend != self.ordinary_stage_backend:
            raise ResponseCompatibilityError(
                "CUDA Lambda tensor backend changed; no CPU fallback allowed"
            )
        self._record_ordinary_metrics(stage, result.metrics)
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

    def _prepare_resident_transpose(self) -> None:
        resident = self._session.provider("shared_transpose")
        if not isinstance(resident, PreparedResident):
            raise TypeError(
                "shared Lambda transpose must use the resident TensorIR owner"
            )
        l1 = np.zeros(self.bound.layouts[0].spec.shape, dtype=np.float64)
        l2 = np.zeros(self.bound.layouts[1].spec.shape, dtype=np.float64)
        resident.upload(
            {
                **self.bound.feeds,
                "bar_singles_residual": l1,
                "bar_doubles_residual": l2,
            }
        )
        self._resident_initial_h2d_bytes = int(resident.transfers["h2d_bytes"])
        self._resident_action_input_bytes = int(l1.nbytes + l2.nbytes)
        outputs = dict(resident.plan.outputs)
        self._resident_action_output_bytes = sum(
            resident.plan.steps[outputs[name]].node.spec.size
            * resident.plan.steps[outputs[name]].node.spec.itemsize
            for name in ("bar_t1", "bar_t2")
        )
        self._record_resident_metrics(resident)

    def _record_resident_metrics(self, resident: PreparedResident) -> None:
        transfers = resident.transfers
        calls = int(transfers["runs"])
        dynamic_h2d = int(transfers["h2d_bytes"]) - self._resident_initial_h2d_bytes
        expected_h2d = calls * self._resident_action_input_bytes
        expected_d2h = calls * (4 + self._resident_action_output_bytes)
        if dynamic_h2d != expected_h2d:
            raise RuntimeError(
                "resident Lambda action re-uploaded static scientific inputs"
            )
        if int(transfers["d2h_bytes"]) != expected_d2h:
            raise RuntimeError("resident Lambda action transfer accounting drifted")
        self.execution_metrics["shared_transpose"] = {
            "mode": "resident-static-cc-feeds",
            "calls": calls,
            "initial_static_h2d_bytes": self._resident_initial_h2d_bytes,
            "dynamic_h2d_bytes_total": dynamic_h2d,
            "dynamic_h2d_bytes_per_call": self._resident_action_input_bytes,
            "output_tensor_d2h_bytes_per_call": self._resident_action_output_bytes,
            "observed_host_to_device_bytes": int(transfers["h2d_bytes"]),
            "observed_device_to_host_bytes": int(transfers["d2h_bytes"]),
            "synchronizations": int(transfers["synchronizations"]),
            "planned_device_arena_bytes": int(resident.plan.allocation_bytes),
            "planned_provider_bytes": int(resident.plan.provider_bytes),
        }

    def _rhs(self, prefix: str) -> np.ndarray:
        out = self._execute(
            f"{prefix}_rhs", {"bar_correlation_energy": np.asarray(-1.0)}
        )
        return self.bound.sqrt_weights * self.bound._pack(
            (out["bar_t1"], out["bar_t2"])
        )

    def _resident_transpose(self, vector: np.ndarray) -> np.ndarray:
        resident = self._session.provider("shared_transpose")
        if not isinstance(resident, PreparedResident):
            raise TypeError("shared Lambda transpose resident owner changed")
        l1, l2 = self.bound._unpack(vector / self.bound.sqrt_weights)
        resident.upload(
            {
                "bar_singles_residual": np.ascontiguousarray(l1, dtype=np.float64),
                "bar_doubles_residual": np.ascontiguousarray(l2, dtype=np.float64),
            }
        )
        leases, metrics = resident.run()
        bar_t1 = resident.download(leases["bar_t1"])
        bar_t2 = resident.download(leases["bar_t2"])
        self.last_metrics["shared_transpose"] = dict(metrics)
        self._record_resident_metrics(resident)
        self._assert_current(self.bound.reference_identity)
        return self.bound.sqrt_weights * self.bound._pack((bar_t1, bar_t2))

    def _transpose(self, prefix: str, vector: np.ndarray) -> np.ndarray:
        if prefix == "shared":
            return self._resident_transpose(vector)
        l1, l2 = self.bound._unpack(vector / self.bound.sqrt_weights)
        out = self._execute(
            f"{prefix}_transpose",
            {"bar_singles_residual": l1, "bar_doubles_residual": l2},
        )
        return self.bound.sqrt_weights * self.bound._pack(
            (out["bar_t1"], out["bar_t2"])
        )

    def solve(self, *, reference_identity: str) -> CCSDLambdaResult:
        """Solve Lambda with resident generated J^T actions and GPU replay."""
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
                            "generated CUDA amplitude response; CC/Fock/integral "
                            "feeds stay resident for repeated shared J^T actions; "
                            "GMRES remains host-controlled; no device Krylov, RDM "
                            "or nuclear-force capability"
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
