"""Native directional nuclear RHS and one CPHF solve, not a molecular HVP.

First-integral execution/direction/weight contractions select explicit CPU or
CUDA providers. AO/MO and shared Krylov execution remain host-side. Metric and
response J/K independently select the qualified direct CUDA adapter. The
original NativeRHFState small-system admission remains unchanged.
"""

import time
import typing
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from hashlib import sha256

import numpy as np
from vibeqc.profiles import canonical_hash

from tools.vibeqc_posthf.reference import immutable
from tools.vibeqc_response import (
    CudaDirectJKBackend,
    GMRESOptions,
    NativeJKBackend,
    RHFResponseOperator,
)

from .first_order import checked_direction, generated_directional_first_order
from .native import NativeRHFState
from .perturbation import RHFNuclearResponse, solve_rhf_nuclear_perturbation


@dataclass(frozen=True, eq=False)
class DirectionalRHFResponse:
    """Owned directional matrices/response bound to a specific SCF snapshot."""

    direction: np.ndarray
    frozen_fock_derivative: np.ndarray
    overlap_derivative: np.ndarray
    response: RHFNuclearResponse
    identity: str
    _diagnostics: dict = field(repr=False)

    @property
    def diagnostics(self) -> typing.Any:
        return deepcopy(self._diagnostics)


def directional_rhf_response(
    state: typing.Any,
    direction: typing.Any,
    *,
    jk_backend: typing.Any = "cpu",
    device_id: typing.Any = 0,
    device_budget_bytes: typing.Any = 64 << 20,
    response_execution: typing.Any = "host",
    response_device_budget_bytes: typing.Any = 128 << 20,
    solver_options: typing.Any = None,
    first_backend: typing.Any = "cpu",
    first_compiler: typing.Any = None,
    first_budget_bytes: typing.Any = 64 << 20,
) -> typing.Any:
    """Build directional H1/S1 and solve one complete canonical RHF response.

    ``direction`` is (atom,xyz), is not normalized, and expands to every
    mathematical integral center before its local derivative contraction.
    No all-coordinate first-order matrices or ERI derivative tensors are
    allocated. A fresh successful result is published only after the response
    solve and density/energy-weighted-density reconstruction finish.

    ``jk_backend="cuda"`` selects CUDA J/K, including metric-density and final
    density response. Independently, ``first_backend="cuda"`` with an explicit
    ``first_compiler`` selects generated device first-integral/direction/weight
    contractions with one final matrix download. AO/MO and Krylov remain host-
    side. Provider/solver budgets retain distinct scopes; no global peak-memory
    guarantee or complete molecular HVP follows from either CUDA selection.
    """
    if not isinstance(state, NativeRHFState):
        raise TypeError("directional response requires NativeRHFState")
    state.validate()
    vector = checked_direction(direction, state.nat)
    if jk_backend not in ("cpu", "cuda"):
        raise ValueError("jk_backend must be cpu or cuda")
    if first_backend not in ("cpu", "cuda"):
        raise ValueError("first_backend must be cpu or cuda")
    if response_execution not in ("host", "cuda-resident"):
        raise ValueError("response_execution must be host or cuda-resident")
    if response_execution == "cuda-resident" and jk_backend != "cuda":
        raise ValueError("cuda-resident response requires jk_backend='cuda'")
    if (
        type(response_device_budget_bytes) is not int
        or not 0 < response_device_budget_bytes < 2**63
    ):
        raise ValueError("response_device_budget_bytes must be a positive int64")
    if first_backend == "cuda":
        from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter

        if not isinstance(first_compiler, CudaCompilerAdapter):
            raise TypeError("first_compiler must be an explicit CudaCompilerAdapter")
    elif first_compiler is not None:
        raise ValueError("first_compiler is only meaningful for CUDA first derivatives")
    if solver_options is not None and not isinstance(solver_options, GMRESOptions):
        raise TypeError("solver_options must be GMRESOptions")
    # Tight absolute tolerance is needed for translation and other nearly zero
    # RHSs; a purely relative rule can demand resolving cancellation noise.
    options = solver_options or GMRESOptions(rtol=1e-11, atol=1e-12)
    with ExitStack() as stack:
        if jk_backend == "cuda":
            backend = stack.enter_context(
                CudaDirectJKBackend(
                    state.source,
                    device_id=device_id,
                    device_budget_bytes=device_budget_bytes,
                )
            )
        else:
            backend = NativeJKBackend(state.source)
        problem = RHFResponseOperator.build_problem(state.reference, backend)
        operator = RHFResponseOperator(problem, backend)
        resident_owner = None
        if response_execution == "cuda-resident":
            retained_provider = backend.device_resident_bytes
            if retained_provider >= response_device_budget_bytes:
                raise ValueError(
                    "direct J/K retained storage leaves no resident response budget"
                )
            restart = min(operator.dimension, options.restart, options.max_iterations)
            vector_slots = min(4096, max(8, 3 * restart + 32))
            resident_owner = stack.enter_context(
                backend.resident_response(
                    problem,
                    vector_slots=vector_slots,
                    device_budget_bytes=response_device_budget_bytes
                    - retained_provider,
                )
            )
            if (
                retained_provider + resident_owner.workspace_bytes
                > response_device_budget_bytes
            ):
                raise RuntimeError(
                    "combined retained J/K and resident response storage exceeds budget"
                )
            operator._krylov_engine = resident_owner
        first_diagnostics = None
        first_started = time.perf_counter()
        if first_backend == "cuda":
            from .first_order_cuda import generated_directional_first_order_cuda

            frozen, overlap, first_diagnostics = generated_directional_first_order_cuda(
                state,
                vector,
                first_compiler,
                device_id=device_id,
                budget_bytes=first_budget_bytes,
            )
        else:
            frozen, overlap = generated_directional_first_order(state, vector)
        first_seconds = time.perf_counter() - first_started
        response_started = time.perf_counter()
        response = solve_rhf_nuclear_perturbation(
            operator, frozen, overlap, options=options
        )
        response_seconds = time.perf_counter() - response_started
        resident_diagnostics = (
            resident_owner.diagnostics if resident_owner is not None else None
        )
        # Check the live state again before publishing detached outputs.
        state.validate()
        identity = canonical_hash(
            {
                "schema": "vibeqc.rhf-directional-response/v1",
                "source": state.source.identity,
                "reference": state.reference.identity,
                "operator": problem.operator_identity,
                "solver_options": asdict(options),
                "first_backend": first_backend,
                "response_execution": response_execution,
                "resident_response": resident_owner.identity
                if resident_owner is not None
                else None,
                "first_programs": first_diagnostics["program_identities"]
                if first_diagnostics
                else None,
                "first_artifacts": first_diagnostics["native_artifacts"]
                if first_diagnostics
                else None,
                "direction": sha256(
                    vector.astype("<f8", copy=False).tobytes()
                ).hexdigest(),
            }
        )
        diag = {
            "source_identity": state.source.identity,
            "reference_identity": state.reference.identity,
            "operator_identity": problem.operator_identity,
            "hamiltonian_id": state.reference.hamiltonian_id,
            "first_integral_derivatives": "cuda-generated"
            if first_backend == "cuda"
            else "cpu-generated",
            "direction_density_reduction": "cuda-generated"
            if first_backend == "cuda"
            else "host",
            "jk_backend": jk_backend,
            "response_execution": response_execution,
            "ao_mo_transforms": (
                "mixed: rhs/reconstruction host, operator cuda-resident"
                if resident_owner is not None
                else "host"
            ),
            "operator_ao_mo_transforms": (
                "cuda-resident" if resident_owner is not None else "host"
            ),
            "rhs_reconstruction": "host",
            "response_vector_storage": (
                "cuda-resident" if resident_owner is not None else "host"
            ),
            "krylov_execution": (
                "cuda-resident-host-controlled"
                if resident_owner is not None
                else "host"
            ),
            "small_least_squares": "host",
            "molecular_hvp": False,
            "nuclear_response_solves": 1,
            "rhs_count": 1,
            "first_order_matrix_bytes": frozen.nbytes + overlap.nbytes,
            "first_order_matrix_layout": "AO,AO",
            "residual_norm": response.solve_result.residual_norm,
            "relative_residual": response.solve_result.relative_residual,
            "iterations": response.solve_result.iterations,
            "operator_actions": response.solve_result.operator_actions,
            "first_source_seconds": first_seconds,
            "response_solve_reconstruct_seconds": response_seconds,
            "response_operator_seconds": response.solve_result.operator_seconds,
            "response_orthogonalization_seconds": (
                response.solve_result.orthogonalization_seconds
            ),
            "solver_options": asdict(options),
            "solver_workspace_bytes": response.solve_result.workspace_bytes,
            "jk_statistics": deepcopy(backend.statistics),
            "memory_scope": "matrix and solver diagnostics only; not total live peak",
        }
        if first_diagnostics is not None:
            diag["first_derivative_provider"] = first_diagnostics
        if jk_backend == "cuda":
            diag["jk_provider"] = backend.diagnostics
        if resident_diagnostics is not None:
            diag["resident_response"] = resident_diagnostics
            diag["retained_response_device_bytes"] = (
                backend.device_resident_bytes
                + resident_diagnostics["owned_device_bytes"]
            )
            diag["response_device_budget_bytes"] = response_device_budget_bytes
    return DirectionalRHFResponse(
        immutable(vector),
        immutable(frozen),
        immutable(overlap),
        response,
        identity,
        diag,
    )
