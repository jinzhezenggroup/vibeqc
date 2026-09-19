"""Native directional nuclear RHS and one CPHF solve, not a molecular HVP.

First-integral execution and direction/density contractions are CPU-side.
The same metric and response J/K actions can use the qualified direct CUDA
adapter. The original NativeRHFState small-system admission remains unchanged.
"""

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
    def diagnostics(self):
        return deepcopy(self._diagnostics)


def directional_rhf_response(
    state,
    direction,
    *,
    jk_backend="cpu",
    device_id=0,
    device_budget_bytes=64 << 20,
    solver_options=None,
):
    """Build directional H1/S1 and solve one complete canonical RHF response.

    ``direction`` is (atom,xyz), is not normalized, and expands to every
    mathematical integral center before its local derivative contraction.
    No all-coordinate first-order matrices or ERI derivative tensors are
    allocated. A fresh successful result is published only after the response
    solve and density/energy-weighted-density reconstruction finish.

    ``jk_backend="cuda"`` selects CUDA J/K only, including metric-density and
    final density response. It does not move first derivatives, AO/MO transforms
    or Krylov execution off the host. The device/solver budgets retain their
    existing distinct scopes; this is not a global peak-memory guarantee.
    """
    if not isinstance(state, NativeRHFState):
        raise TypeError("directional response requires NativeRHFState")
    state.validate()
    vector = checked_direction(direction, state.nat)
    if jk_backend not in ("cpu", "cuda"):
        raise ValueError("jk_backend must be cpu or cuda")
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
        frozen, overlap = generated_directional_first_order(state, vector)
        response = solve_rhf_nuclear_perturbation(
            operator, frozen, overlap, options=options
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
            "first_integral_derivatives": "cpu-generated",
            "direction_density_reduction": "host",
            "jk_backend": jk_backend,
            "ao_mo_transforms": "host",
            "krylov_execution": "host",
            "molecular_hvp": False,
            "nuclear_response_solves": 1,
            "rhs_count": 1,
            "first_order_matrix_bytes": frozen.nbytes + overlap.nbytes,
            "first_order_matrix_layout": "AO,AO",
            "residual_norm": response.solve_result.residual_norm,
            "relative_residual": response.solve_result.relative_residual,
            "iterations": response.solve_result.iterations,
            "solver_options": asdict(options),
            "solver_workspace_bytes": response.solve_result.workspace_bytes,
            "jk_statistics": deepcopy(backend.statistics),
            "memory_scope": "matrix and solver diagnostics only; not total live peak",
        }
        if jk_backend == "cuda":
            diag["jk_provider"] = backend.diagnostics
    return DirectionalRHFResponse(
        immutable(vector),
        immutable(frozen),
        immutable(overlap),
        response,
        identity,
        diag,
    )
