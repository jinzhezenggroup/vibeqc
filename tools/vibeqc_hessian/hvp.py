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

from .analytic import (
    _checked_second_hvp_options,
    accumulate_provider_hvp_cuda,
    nuclear_hvp,
    provider_hvp_components,
)
from .directional import DirectionalRHFResponse, directional_rhf_response
from .first_order import checked_direction, generated_rhf_relaxation_contraction
from .native import NativeRHFState


@dataclass(frozen=True, eq=False)
class RHFHVPResult:
    """Detached complete conventional RHF HVP and its scientific components."""

    direction: np.ndarray
    value: np.ndarray
    nuclear: np.ndarray | None
    core: np.ndarray | None
    pulay: np.ndarray | None
    two_electron: np.ndarray | None
    relaxation: np.ndarray | None
    directional_response: DirectionalRHFResponse = field(repr=False)
    identity: str
    _diagnostics: dict = field(repr=False)

    @property
    def diagnostics(self) -> dict[str, object]:
        return deepcopy(self._diagnostics)

    @property
    def components(self) -> dict[str, np.ndarray | None]:
        return {
            "nuclear": self.nuclear,
            "core": self.core,
            "pulay": self.pulay,
            "two_electron": self.two_electron,
            "relaxation": self.relaxation,
        }


def _rhf_hvp_cuda_assembly(
    state: NativeRHFState,
    vector: np.ndarray,
    *,
    jk_backend: str,
    device_id: int,
    device_budget_bytes: int,
    response_execution: str,
    response_device_budget_bytes: int,
    solver_options: object,
    first_backend: str,
    first_compiler: object,
    first_budget_bytes: int,
    second_backend: str,
    second_compiler: object,
    second_budget_bytes: int,
    relaxation_backend: str,
    relaxation_compiler: object,
    relaxation_budget_bytes: int,
    assembly_budget_bytes: int,
) -> RHFHVPResult:
    """Assemble frozen, nuclear and relaxation HVP terms on one CUDA owner."""
    from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter

    from .device_assembly import CudaHVPAccumulator, compile_hvp_assembly
    from .first_order_cuda import generated_rhf_relaxation_contraction_cuda

    if (
        jk_backend != "cuda"
        or response_execution != "cuda-resident"
        or second_backend != "cuda"
        or relaxation_backend != "cuda"
    ):
        raise ValueError(
            "CUDA final assembly requires cuda J/K, resident response, "
            "CUDA second integrals and CUDA relaxation"
        )
    if not isinstance(second_compiler, CudaCompilerAdapter) or not isinstance(
        relaxation_compiler, CudaCompilerAdapter
    ):
        raise TypeError("CUDA final assembly requires explicit CUDA compilers")
    if type(assembly_budget_bytes) is not int or not 0 < assembly_budget_bytes < 2**63:
        raise ValueError("assembly_budget_bytes must be a positive int64")

    total_started = time.perf_counter()
    artifact = compile_hvp_assembly(
        second_compiler, state.cache / "final-hvp-assembly-cuda"
    )
    with CudaHVPAccumulator(
        artifact,
        natoms=state.nat,
        device_id=device_id,
        budget_bytes=assembly_budget_bytes,
    ) as owner:
        nuclear_started = time.perf_counter()
        owner.reset_nuclear(state.coords, state.Z, vector)
        nuclear_seconds = time.perf_counter() - nuclear_started
        resident_result: dict[str, object] = {}

        def consume_resident_response(weights: object) -> None:
            started = time.perf_counter()
            _, provider = generated_rhf_relaxation_contraction_cuda(
                state,
                None,
                None,
                relaxation_compiler,
                resident_weights=weights,
                device_output_consumer=lambda pointer, count: (
                    owner.add_device(pointer)
                    if count == 3 * state.nat
                    else (_ for _ in ()).throw(
                        RuntimeError("relaxation device output size mismatch")
                    )
                ),
                publish_host=False,
                device_id=device_id,
                budget_bytes=relaxation_budget_bytes,
            )
            resident_result["provider"] = provider
            resident_result["seconds"] = time.perf_counter() - started

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
            resident_reconstruction_consumer=consume_resident_response,
        )
        relaxation_seconds = float(resident_result.get("seconds", 0.0))
        response_seconds = max(
            0.0, time.perf_counter() - response_started - relaxation_seconds
        )
        if "provider" not in resident_result:
            raise RuntimeError("resident CUDA relaxation did not reach final assembly")

        second_started = time.perf_counter()
        second_provider = accumulate_provider_hvp_cuda(
            state,
            vector,
            owner,
            compiler=second_compiler,
            device_id=device_id,
            budget_bytes=second_budget_bytes,
        )
        second_seconds = time.perf_counter() - second_started

        assembly_started = time.perf_counter()
        total = owner.finish()
        assembly_seconds = time.perf_counter() - assembly_started
        assembly_diagnostics = owner.diagnostics

    if not np.isfinite(total).all():
        raise FloatingPointError("nonfinite CUDA-assembled RHF HVP")
    state.validate()
    response_diag = response.diagnostics
    relaxation_provider = resident_result["provider"]
    identity = canonical_hash(
        {
            "schema": "vibeqc.rhf-hvp/v1",
            "source": state.source.identity,
            "reference": state.reference.identity,
            "directional_response": response.identity,
            "direction": sha256(vector.astype("<f8", copy=False).tobytes()).hexdigest(),
            "second_integrals": second_provider["backend"],
            "second_integral_programs": second_provider["program_identities"],
            "relaxation_first_integrals": relaxation_provider["backend"],
            "relaxation_programs": relaxation_provider.get("program_identities"),
            "final_assembly": "cuda-device-resident",
        }
    )
    diagnostics = {
        "source_identity": state.source.identity,
        "reference_identity": state.reference.identity,
        "molecular_hvp": True,
        "component_publication": "suppressed",
        "final_assembly_backend": "cuda-device-resident",
        "execution_residency": "device-final-assembly-with-host-response-publication",
        "response_execution": response_execution,
        "response_jk_backend": jk_backend,
        "response_first_backend": first_backend,
        "second_integral_backend": second_provider["backend"],
        "second_integral_provider": deepcopy(second_provider),
        "relaxation_first_integral_backend": relaxation_provider["backend"],
        "relaxation_provider": deepcopy(relaxation_provider),
        "final_assembly": deepcopy(assembly_diagnostics),
        "timings_seconds": {
            "directional_response": response_seconds,
            "relaxation_first_integrals": relaxation_seconds,
            "second_integral_hvp": second_seconds,
            "nuclear": nuclear_seconds,
            "final_download": assembly_seconds,
            "complete_hvp": time.perf_counter() - total_started,
        },
        "transfers": {
            "resident_response": deepcopy(response_diag.get("resident_response", {})),
            "second_integral_hvp": deepcopy(second_provider),
            "relaxation_first_integrals": deepcopy(relaxation_provider),
            "final_assembly": deepcopy(assembly_diagnostics),
        },
        "published_hvp_bytes": int(total.nbytes),
        "published_component_bytes": 0,
        "remaining_host_boundaries": (
            "directional H1/S1 publication, RHS preparation/small least-squares, "
            "compatibility response publication"
        ),
        "memory_scope": (
            "final CUDA accumulator + provider-local budgets; not a combined "
            "SCF/compiler/CUDA-context global peak"
        ),
    }
    return RHFHVPResult(
        immutable(vector),
        immutable(np.array(total, copy=True)),
        None,
        None,
        None,
        None,
        None,
        response,
        identity,
        diagnostics,
    )


def rhf_hvp(
    state: NativeRHFState,
    direction: np.ndarray,
    *,
    jk_backend: str = "cpu",
    device_id: int = 0,
    device_budget_bytes: int = 64 << 20,
    response_execution: str = "host",
    response_device_budget_bytes: int = 128 << 20,
    solver_options: object = None,
    first_backend: str = "cpu",
    first_compiler: object = None,
    first_budget_bytes: int = 64 << 20,
    second_backend: str = "cpu",
    second_compiler: object = None,
    second_budget_bytes: int = 64 << 20,
    relaxation_backend: str = "cpu",
    relaxation_compiler: object = None,
    relaxation_budget_bytes: int = 64 << 20,
    assembly_backend: str = "host",
    assembly_budget_bytes: int = 8 << 20,
) -> RHFHVPResult:
    """Apply the complete conventional RHF molecular Hessian to one direction.

    The second-integral skeleton is generated directly as weighted HVPs. One
    directional CPHF solve supplies D1(v) and W1(v); generated first derivatives
    then contract Tr[H1_R D1(v)] - Tr[S1_R W1(v)] shell-locally for every
    output coordinate. No molecular Hessian, all-coordinate H1/S1 tensor, or
    coordinate-by-coordinate ERI derivative tensor is allocated.

    CPU remains the default second-integral HVP backend. CUDA may be selected
    independently for directional H1/S1, direct J/K/response residency, the
    #178 second-integral weighted HVP provider and first-integral relaxation.
    The ordinary CUDA second-integral path streams packed shell primitive/weight
    records and downloads only contracted coordinate HVP tiles. With
    ``assembly_backend="cuda"``, those compact tiles and CUDA relaxation are
    consumed device-to-device by a bounded final HVP accumulator; only the final
    molecular HVP is published to host.
    """
    if not isinstance(state, NativeRHFState):
        raise TypeError("RHF HVP requires NativeRHFState")
    state.validate()
    vector = checked_direction(direction, state.nat)
    if assembly_backend not in ("host", "cuda"):
        raise ValueError("assembly_backend must be host or cuda")
    if assembly_backend == "cuda":
        _checked_second_hvp_options(
            second_backend, second_compiler, device_id, second_budget_bytes
        )
        return _rhf_hvp_cuda_assembly(
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
            second_backend=second_backend,
            second_compiler=second_compiler,
            second_budget_bytes=second_budget_bytes,
            relaxation_backend=relaxation_backend,
            relaxation_compiler=relaxation_compiler,
            relaxation_budget_bytes=relaxation_budget_bytes,
            assembly_budget_bytes=assembly_budget_bytes,
        )
    _checked_second_hvp_options(
        second_backend, second_compiler, device_id, second_budget_bytes
    )
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
    resident_relaxation = (
        response_execution == "cuda-resident" and relaxation_backend == "cuda"
    )
    resident_result = {}

    def consume_resident_response(weights: object) -> None:
        from .first_order_cuda import generated_rhf_relaxation_contraction_cuda

        started = time.perf_counter()
        value, provider = generated_rhf_relaxation_contraction_cuda(
            state,
            None,
            None,
            relaxation_compiler,
            resident_weights=weights,
            device_id=device_id,
            budget_bytes=relaxation_budget_bytes,
        )
        resident_result["value"] = value
        resident_result["provider"] = provider
        resident_result["seconds"] = time.perf_counter() - started

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
        resident_reconstruction_consumer=(
            consume_resident_response if resident_relaxation else None
        ),
    )
    response_elapsed = time.perf_counter() - response_started
    relaxation_seconds = float(resident_result.get("seconds", 0.0))
    response_seconds = max(0.0, response_elapsed - relaxation_seconds)

    relaxation_started = time.perf_counter()
    if resident_relaxation:
        if "value" not in resident_result or "provider" not in resident_result:
            raise RuntimeError("resident RHF relaxation result was not published")
        relaxation = resident_result["value"]
        relaxation_provider = resident_result["provider"]
    elif relaxation_backend == "cuda":
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
    if not resident_relaxation:
        relaxation_seconds = time.perf_counter() - relaxation_started

    second_started = time.perf_counter()
    second, second_provider = provider_hvp_components(
        state,
        vector,
        backend=second_backend,
        compiler=second_compiler,
        device_id=device_id,
        budget_bytes=second_budget_bytes,
        return_diagnostics=True,
    )
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
            "second_integrals": second_provider["backend"],
            "second_integral_programs": second_provider["program_identities"],
            "relaxation_first_integrals": relaxation_provider["backend"],
            "relaxation_programs": relaxation_provider.get("program_identities"),
        }
    )

    response_diag = response.diagnostics
    residency = (
        "mixed-host-device-resident-response-relaxation"
        if resident_relaxation
        else "mixed-host-device-resident-response"
        if response_execution == "cuda-resident"
        else "mixed-host-device"
        if first_backend == "cuda"
        or jk_backend == "cuda"
        or relaxation_backend == "cuda"
        or second_backend == "cuda"
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
        "second_integral_backend": second_provider["backend"],
        "second_integral_provider": deepcopy(second_provider),
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
            "second_integral_hvp": deepcopy(second_provider),
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
        "second_integral_budget_bytes": second_budget_bytes,
        "resident_relaxation_overlap_device_bytes": (
            response_diag.get("retained_response_device_bytes", 0)
            + relaxation_provider.get("storage", {}).get("device_bytes", 0)
            if resident_relaxation
            else 0
        ),
        "published_hvp_bytes": int(total.nbytes),
        "memory_scope": (
            "resident response + CUDA relaxation simultaneous device storage is reported "
            "when active; still not a combined SCF/compiler/CUDA-context global peak"
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
