"""Matrix-free conventional RHF molecular Hessian-vector products.

This bounded tools integration composes #178 directional second-integral
consumers with one #179 directional CPHF solve and shell-local first-integral
relaxation contractions. It intentionally does not expose a Calculator Hessian
API and does not claim an all-device response path.
"""

import time
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256

import numpy as np
from vibeqc.profiles import canonical_hash

from tools.vibeqc_posthf.reference import immutable

from .analytic import nuclear_hvp, provider_hvp_components
from .directional import DirectionalRHFResponse, directional_rhf_response
from .first_order import checked_direction, generated_rhf_relaxation_contraction
from .native import NativeRHFState


@dataclass(frozen=True, eq=False)
class RHFHVPResult:
    """Detached complete conventional RHF HVP and its scientific components."""

    direction: np.ndarray
    value: np.ndarray
    nuclear: np.ndarray
    core: np.ndarray
    pulay: np.ndarray
    two_electron: np.ndarray
    relaxation: np.ndarray
    directional_response: DirectionalRHFResponse = field(repr=False)
    identity: str
    _diagnostics: dict = field(repr=False)

    @property
    def diagnostics(self):
        return deepcopy(self._diagnostics)

    @property
    def components(self):
        return {
            "nuclear": self.nuclear,
            "core": self.core,
            "pulay": self.pulay,
            "two_electron": self.two_electron,
            "relaxation": self.relaxation,
        }


def rhf_hvp(
    state,
    direction,
    *,
    jk_backend="cpu",
    device_id=0,
    device_budget_bytes=64 << 20,
    response_execution="host",
    response_device_budget_bytes=128 << 20,
    solver_options=None,
    first_backend="cpu",
    first_compiler=None,
    first_budget_bytes=64 << 20,
    relaxation_backend="cpu",
    relaxation_compiler=None,
    relaxation_budget_bytes=64 << 20,
):
    """Apply the complete conventional RHF molecular Hessian to one direction.

    The second-integral skeleton is generated directly as weighted HVPs. One
    directional CPHF solve supplies D1(v) and W1(v); generated first derivatives
    then contract Tr[H1_R D1(v)] - Tr[S1_R W1(v)] shell-locally for every
    output coordinate. No molecular Hessian, all-coordinate H1/S1 tensor, or
    coordinate-by-coordinate ERI derivative tensor is allocated.

    CPU remains the qualified second-integral HVP backend. CUDA may be selected
    independently for directional H1/S1, direct J/K/response residency and the
    first-integral relaxation contraction. The CUDA relaxation path uploads the
    solved D1/W1/P0 AO weights, keeps primitive derivatives and AO-weight
    products on device, and downloads only the final Cartesian contraction.
    Second-integral HVPs and final molecular assembly remain host-side.
    """
    if not isinstance(state, NativeRHFState):
        raise TypeError("RHF HVP requires NativeRHFState")
    state.validate()
    vector = checked_direction(direction, state.nat)
    if relaxation_backend not in ("cpu", "cuda"):
        raise ValueError("relaxation_backend must be cpu or cuda")
    if relaxation_backend == "cuda":
        from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter

        if not isinstance(relaxation_compiler, CudaCompilerAdapter):
            raise TypeError("CUDA relaxation requires an explicit CudaCompilerAdapter")
        if (
            type(relaxation_budget_bytes) is not int
            or not 0 < relaxation_budget_bytes < 2**63
        ):
            raise ValueError("relaxation_budget_bytes must be a positive int64")
        from vibeqc_compiler.integral.first_gradient_execute import (
            first_gradient_storage,
        )

        storage = first_gradient_storage(state.nbf, state.nat, 3, 128)
        if storage["numeric_peak_bytes"] > relaxation_budget_bytes:
            raise MemoryError(
                "CUDA relaxation numeric storage exceeds relaxation_budget_bytes"
            )
    elif relaxation_compiler is not None:
        raise ValueError("relaxation_compiler is only meaningful for CUDA relaxation")

    total_started = time.perf_counter()

    response_started = time.perf_counter()
    response = directional_rhf_response(
        state,
        vector,
        jk_backend=jk_backend,
        device_id=device_id,
        device_budget_bytes=device_budget_bytes,
        response_execution=response_execution,
        response_device_budget_bytes=response_device_budget_bytes,
        solver_options=solver_options,
        first_backend=first_backend,
        first_compiler=first_compiler,
        first_budget_bytes=first_budget_bytes,
    )
    response_seconds = time.perf_counter() - response_started

    relaxation_started = time.perf_counter()
    if relaxation_backend == "cuda":
        from .first_order_cuda import generated_rhf_relaxation_contraction_cuda

        relaxation, relaxation_provider = generated_rhf_relaxation_contraction_cuda(
            state,
            response.response.density_derivative,
            response.response.energy_weighted_density_derivative,
            relaxation_compiler,
            device_id=device_id,
            budget_bytes=relaxation_budget_bytes,
        )
    else:
        relaxation = generated_rhf_relaxation_contraction(
            state,
            response.response.density_derivative,
            response.response.energy_weighted_density_derivative,
        )
        relaxation_provider = {
            "backend": "cpu-generated-weighted-contraction",
            "device_transfers": 0,
        }
    relaxation_seconds = time.perf_counter() - relaxation_started

    second_started = time.perf_counter()
    second = provider_hvp_components(state, vector)
    second_seconds = time.perf_counter() - second_started

    nuclear_started = time.perf_counter()
    nuclear = nuclear_hvp(state, vector)
    nuclear_seconds = time.perf_counter() - nuclear_started

    assembly_started = time.perf_counter()
    total = (
        nuclear + second["core"] + second["pulay"] + second["two_electron"] + relaxation
    )
    assembly_seconds = time.perf_counter() - assembly_started

    if not np.isfinite(total).all():
        raise FloatingPointError("nonfinite RHF HVP; no result published")
    state.validate()

    total_seconds = time.perf_counter() - total_started
    identity = canonical_hash(
        {
            "schema": "vibeqc.rhf-hvp/v1",
            "source": state.source.identity,
            "reference": state.reference.identity,
            "directional_response": response.identity,
            "direction": sha256(vector.astype("<f8", copy=False).tobytes()).hexdigest(),
            "second_integrals": "cpu-generated-weighted-hvp",
            "relaxation_first_integrals": relaxation_provider["backend"],
            "relaxation_programs": relaxation_provider.get("program_identities"),
        }
    )

    response_diag = response.diagnostics
    residency = (
        "mixed-host-device-resident-response"
        if response_execution == "cuda-resident"
        else "mixed-host-device"
        if first_backend == "cuda"
        or jk_backend == "cuda"
        or relaxation_backend == "cuda"
        else "host"
    )
    first_transfer = response_diag.get("first_derivative_provider")
    if first_transfer is None:
        first_transfer = {"backend": "cpu-generated", "device_transfers": 0}
    jk_transfer = response_diag.get("jk_provider")
    if jk_transfer is None:
        jk_transfer = {"backend": "cpu-native", "device_transfers": 0}
    resident_transfer = response_diag.get("resident_response")
    if resident_transfer is None:
        resident_transfer = {
            "backend": "host",
            "h2d_bytes": 0,
            "d2h_bytes": 0,
            "synchronizations": 0,
            "operator_actions": 0,
        }

    diagnostics = {
        "source_identity": state.source.identity,
        "reference_identity": state.reference.identity,
        "molecular_hvp": True,
        "full_molecular_hessian_allocated": False,
        "all_coordinate_first_integrals_allocated": False,
        "second_integral_backend": "cpu-generated-weighted-hvp",
        "relaxation_first_integral_backend": relaxation_provider["backend"],
        "relaxation_provider": deepcopy(relaxation_provider),
        "response_first_backend": first_backend,
        "response_jk_backend": jk_backend,
        "response_execution": response_execution,
        "ao_mo_transforms": response_diag["ao_mo_transforms"],
        "krylov_execution": response_diag["krylov_execution"],
        "execution_residency": residency,
        "nuclear_response_solves": response_diag["nuclear_response_solves"],
        "response_residual_norm": response_diag["residual_norm"],
        "response_relative_residual": response_diag["relative_residual"],
        "response_iterations": response_diag["iterations"],
        "response_operator_actions": response_diag["operator_actions"],
        "timings_seconds": {
            "directional_response": response_seconds,
            "first_source": response_diag["first_source_seconds"],
            "response_solve_reconstruct": response_diag[
                "response_solve_reconstruct_seconds"
            ],
            "response_operator": response_diag["response_operator_seconds"],
            "response_orthogonalization": response_diag[
                "response_orthogonalization_seconds"
            ],
            "relaxation_first_integrals": relaxation_seconds,
            "second_integral_hvp": second_seconds,
            "nuclear": nuclear_seconds,
            "final_assembly": assembly_seconds,
            "complete_hvp": total_seconds,
        },
        "transfers": {
            "directional_first": deepcopy(first_transfer),
            "response_jk": deepcopy(jk_transfer),
            "resident_response": deepcopy(resident_transfer),
            "second_integral_hvp": "host-only; no device transfers",
            "relaxation_first_integrals": deepcopy(relaxation_provider),
            "nuclear": "host-only; no device transfers",
        },
        "solver_workspace_bytes": response_diag["solver_workspace_bytes"],
        "retained_response_device_bytes": response_diag.get(
            "retained_response_device_bytes", 0
        ),
        "response_device_budget_bytes": response_diag.get(
            "response_device_budget_bytes", 0
        ),
        "published_hvp_bytes": int(total.nbytes),
        "memory_scope": (
            "published HVP + directional-provider/solver diagnostics only; "
            "not a combined SCF/J/K/compiler/CUDA-context peak"
        ),
    }

    arrays = (
        vector,
        total,
        nuclear,
        second["core"],
        second["pulay"],
        second["two_electron"],
        relaxation,
    )
    return RHFHVPResult(
        *(immutable(np.array(value, copy=True)) for value in arrays),
        response,
        identity,
        diagnostics,
    )
