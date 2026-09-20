"""Bounded block and full-Hessian assembly from the conventional RHF HVP.

This tools-only B4 path shares one response operator per block and delegates
the nonredundant orbital solves to #179 solve_many. It never substitutes a
partial/diagonal Hessian when a full output cannot fit its declared budget.
"""

import time
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256

import numpy as np
from vibeqc.profiles import canonical_hash

from tools.vibeqc_posthf.reference import immutable
from tools.vibeqc_response import (
    CudaDirectJKBackend,
    GMRESOptions,
    NativeJKBackend,
    RHFResponseOperator,
    resident_vector_slots,
)

from .analytic import (
    _checked_second_hvp_options,
    nuclear_hvp,
    provider_hvp_components,
)
from .first_order import (
    generated_directional_first_order,
    generated_rhf_relaxation_contraction,
)
from .native import NativeRHFState
from .perturbation import (
    RHFNuclearBatchResponse,
    solve_rhf_nuclear_perturbations,
)


@dataclass(frozen=True, eq=False)
class RHFHVPBlockResult:
    """A bounded ordered set of complete RHF Hessian-vector products."""

    directions: np.ndarray
    values: np.ndarray
    nuclear: np.ndarray
    core: np.ndarray
    pulay: np.ndarray
    two_electron: np.ndarray
    relaxation: np.ndarray
    response_batch: RHFNuclearBatchResponse = field(repr=False)
    identity: str
    _diagnostics: dict = field(repr=False)

    @property
    def diagnostics(self) -> dict[str, object]:
        return deepcopy(self._diagnostics)

    @property
    def components(self) -> dict[str, np.ndarray]:
        return {
            "nuclear": self.nuclear,
            "core": self.core,
            "pulay": self.pulay,
            "two_electron": self.two_electron,
            "relaxation": self.relaxation,
        }


@dataclass(frozen=True, eq=False)
class RHFHessianResult:
    """A bounded raw Cartesian RHF Hessian in canonical atom/xyz order."""

    matrix: np.ndarray
    identity: str
    _diagnostics: dict = field(repr=False)

    @property
    def diagnostics(self) -> dict[str, object]:
        return deepcopy(self._diagnostics)


def _checked_directions(directions: np.ndarray, natoms: int) -> np.ndarray:
    values = np.asarray(directions)
    if (
        values.ndim != 3
        or values.shape[1:] != (natoms, 3)
        or values.shape[0] < 1
        or values.dtype.kind not in "iuf"
        or not np.isfinite(values).all()
    ):
        raise ValueError("directions must be finite real with shape (nrhs, natoms, 3)")
    result = np.array(values, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("directions must be representable in FP64")
    return result


def _checked_budget(value: int, name: str) -> int:
    if type(value) is not int or not 0 < value < 2**63:
        raise ValueError(f"{name} must be a positive int64 byte count")
    return value


def _block_persistent_bound(state: NativeRHFState, nrhs: int) -> dict[str, int]:
    """Conservative numeric storage retained outside solve_many workspace."""
    nmo, nocc = state.nbf, state.nocc
    nvir = nmo - nocc
    dim = nocc * nvir
    coords = 3 * state.nat
    f8 = 8
    directions = nrhs * coords * f8
    # AO H1/S1 inputs coexist with validation copies and prepared MO H1/S1
    # during batch preparation and reconstruction. Also reserve the live
    # transforms, metric/Fock intermediates and immutable validation copies.
    first_and_prepared = 16 * nrhs * nmo * nmo * f8
    # Prepared RHS vectors and the packed multi-RHS matrix can coexist. The
    # solver's validated RHS copy is charged inside solve_many workspace.
    prepared_rhs = 2 * nrhs * dim * f8
    # Published per-RHS response: rhs, C1, e1, D1 and W1.
    responses = 2 * nrhs * (dim + nmo * nocc + nocc * nocc + 2 * nmo * nmo) * f8
    # total plus five separately retained scientific components. Directions
    # are charged above as their own immutable publication.
    hvp_outputs = nrhs * coords * 6 * f8
    # Stacked component sources, sum temporaries and immutable bytes-backed
    # publications can coexist with the working output arrays.
    assembly_publication = 2 * (directions + hvp_outputs)
    return {
        "directions": directions,
        "first_and_prepared_matrices": first_and_prepared,
        "prepared_rhs": prepared_rhs,
        "published_responses": responses,
        "hvp_outputs": hvp_outputs,
        "assembly_publication_scratch": assembly_publication,
        "total": (
            directions
            + first_and_prepared
            + prepared_rhs
            + responses
            + hvp_outputs
            + assembly_publication
        ),
    }


def rhf_hvp_many(
    state: NativeRHFState,
    directions: np.ndarray,
    *,
    strategy: str = "recycled",
    total_budget_bytes: int = 128 << 20,
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
) -> RHFHVPBlockResult:
    """Apply the complete conventional RHF Hessian to a bounded direction block.

    First-integral sources are generated independently per direction, while one
    shared response operator and one #179 solve_many call own the orbital solve.
    Second-integral HVPs and relaxation contractions remain directional and are
    assembled only after all response solutions converge. The #178 HVP provider
    may execute on CPU or CUDA independently of response and relaxation.

    total_budget_bytes is a conservative numeric live-storage bound for this
    block. It includes retained first/prepared matrices, published response and
    HVP arrays plus the solve_many peak workspace. Compiler/runtime metadata,
    native call stacks and CUDA context memory remain explicit exclusions.
    """
    if not isinstance(state, NativeRHFState):
        raise TypeError("RHF HVP block requires NativeRHFState")
    state.validate()
    vectors = _checked_directions(directions, state.nat)
    total_budget_bytes = _checked_budget(total_budget_bytes, "total_budget_bytes")
    device_budget_bytes = _checked_budget(device_budget_bytes, "device_budget_bytes")
    first_budget_bytes = _checked_budget(first_budget_bytes, "first_budget_bytes")
    _checked_second_hvp_options(
        second_backend, second_compiler, device_id, second_budget_bytes
    )
    if strategy not in ("sequential", "blocked", "recycled"):
        raise ValueError("strategy must be sequential, blocked or recycled")
    if jk_backend not in ("cpu", "cuda"):
        raise ValueError("jk_backend must be cpu or cuda")
    if response_execution not in ("host", "cuda-resident"):
        raise ValueError("response_execution must be host or cuda-resident")
    if response_execution == "cuda-resident" and jk_backend != "cuda":
        raise ValueError("cuda-resident response requires jk_backend='cuda'")
    response_device_budget_bytes = _checked_budget(
        response_device_budget_bytes, "response_device_budget_bytes"
    )
    if first_backend not in ("cpu", "cuda"):
        raise ValueError("first_backend must be cpu or cuda")
    if solver_options is not None and not isinstance(solver_options, GMRESOptions):
        raise TypeError("solver_options must be GMRESOptions")
    if relaxation_backend not in ("cpu", "cuda"):
        raise ValueError("relaxation_backend must be cpu or cuda")
    relaxation_storage = None
    if relaxation_backend == "cuda":
        from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
        from vibeqc_compiler.integral.first_gradient_execute import (
            first_gradient_storage,
        )

        if not isinstance(relaxation_compiler, CudaCompilerAdapter):
            raise TypeError("CUDA relaxation requires an explicit CudaCompilerAdapter")
        relaxation_budget_bytes = _checked_budget(
            relaxation_budget_bytes, "relaxation_budget_bytes"
        )
        relaxation_storage = first_gradient_storage(state.nbf, state.nat, 3, 128)
        if relaxation_storage["numeric_peak_bytes"] > relaxation_budget_bytes:
            raise MemoryError(
                "CUDA relaxation numeric storage exceeds relaxation_budget_bytes"
            )
    elif relaxation_compiler is not None:
        raise ValueError("relaxation_compiler is only meaningful for CUDA relaxation")
    if first_backend == "cuda":
        from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter

        if not isinstance(first_compiler, CudaCompilerAdapter):
            raise TypeError("first_compiler must be an explicit CudaCompilerAdapter")
    elif first_compiler is not None:
        raise ValueError("first_compiler is only meaningful for CUDA first derivatives")

    storage = _block_persistent_bound(state, len(vectors))
    if storage["total"] >= total_budget_bytes:
        raise ValueError(
            "HVP block persistent numeric storage exceeds total_budget_bytes"
        )
    if (
        relaxation_storage is not None
        and storage["total"] + relaxation_storage["numeric_peak_bytes"]
        > total_budget_bytes
    ):
        raise ValueError(
            "HVP block plus CUDA relaxation numeric storage exceeds total_budget_bytes"
        )
    options = solver_options or GMRESOptions(rtol=1e-11, atol=1e-12)
    solver_budget = total_budget_bytes - storage["total"]
    bounded_options = replace(
        options,
        max_workspace_bytes=min(options.max_workspace_bytes, solver_budget),
    )

    total_started = time.perf_counter()
    first_diagnostics = []
    with ExitStack() as stack:
        if jk_backend == "cuda":
            backend = stack.enter_context(
                CudaDirectJKBackend(
                    state.source,
                    device_id=device_id,
                    device_budget_bytes=min(
                        device_budget_bytes,
                        total_budget_bytes - storage["total"],
                        response_device_budget_bytes
                        if response_execution == "cuda-resident"
                        else device_budget_bytes,
                    ),
                )
            )
        else:
            backend = NativeJKBackend(state.source)
        problem = RHFResponseOperator.build_problem(state.reference, backend)
        operator = RHFResponseOperator(problem, backend)
        resident_owner = None
        retained_response_bytes = (
            backend.device_resident_bytes if jk_backend == "cuda" else 0
        )
        if response_execution == "cuda-resident":
            slots = resident_vector_slots(
                operator.dimension,
                bounded_options,
                rhs_count=len(vectors),
                strategy=strategy,
            )
            if slots > 4096:
                raise ValueError(
                    "resident block vector-slot plan exceeds native capacity"
                )
            remaining = (
                min(response_device_budget_bytes, total_budget_bytes - storage["total"])
                - retained_response_bytes
            )
            if remaining <= 0:
                raise ValueError(
                    "Coulomb/exchange leaves no resident response device budget"
                )
            resident_owner = stack.enter_context(
                backend.resident_response(
                    problem, vector_slots=max(8, slots), device_budget_bytes=remaining
                )
            )
            retained_response_bytes += resident_owner.workspace_bytes
            operator._krylov_engine = resident_owner
        # The response phase coexists with the fixed device arena. Reserve it
        # before first-integral work, in addition to the solver's logical buffers.
        response_available = (
            total_budget_bytes - storage["total"] - retained_response_bytes
        )
        if response_available <= 0:
            raise ValueError("retained response storage exceeds total_budget_bytes")
        bounded_options = replace(
            bounded_options,
            max_workspace_bytes=min(
                bounded_options.max_workspace_bytes, response_available
            ),
        )

        first_started = time.perf_counter()
        frozen = np.empty((len(vectors), state.nbf, state.nbf))
        overlap = np.empty_like(frozen)
        for index, vector in enumerate(vectors):
            if first_backend == "cuda":
                from .first_order_cuda import generated_directional_first_order_cuda

                h1, s1, diag = generated_directional_first_order_cuda(
                    state,
                    vector,
                    first_compiler,
                    device_id=device_id,
                    budget_bytes=first_budget_bytes,
                )
                first_diagnostics.append(diag)
            else:
                h1, s1 = generated_directional_first_order(state, vector)
            frozen[index] = h1
            overlap[index] = s1
        first_seconds = time.perf_counter() - first_started

        response_started = time.perf_counter()
        batch = solve_rhf_nuclear_perturbations(
            operator,
            frozen,
            overlap,
            strategy=strategy,
            options=bounded_options,
        )
        response_seconds = time.perf_counter() - response_started
        jk_statistics = deepcopy(backend.statistics)
        jk_provider = backend.diagnostics if jk_backend == "cuda" else None
        resident_diagnostic = (
            resident_owner.diagnostics if resident_owner is not None else None
        )
        resident_identity = (
            resident_owner.identity if resident_owner is not None else None
        )

    state.validate()
    if (
        storage["total"]
        + retained_response_bytes
        + batch.solve_result.peak_workspace_bytes
        > total_budget_bytes
    ):
        raise RuntimeError("multi-RHS response exceeded the declared total budget")

    relaxation_started = time.perf_counter()
    relaxation_diagnostics = []
    if relaxation_backend == "cuda":
        from .first_order_cuda import generated_rhf_relaxation_contraction_cuda

        relaxation_items = []
        for response in batch.responses:
            item, diagnostic = generated_rhf_relaxation_contraction_cuda(
                state,
                response.density_derivative,
                response.energy_weighted_density_derivative,
                relaxation_compiler,
                device_id=device_id,
                budget_bytes=relaxation_budget_bytes,
            )
            relaxation_items.append(item)
            relaxation_diagnostics.append(diagnostic)
        relaxation = np.stack(relaxation_items)
    else:
        relaxation = np.stack(
            [
                generated_rhf_relaxation_contraction(
                    state,
                    response.density_derivative,
                    response.energy_weighted_density_derivative,
                )
                for response in batch.responses
            ]
        )
    relaxation_seconds = time.perf_counter() - relaxation_started

    second_started = time.perf_counter()
    second_items = [
        provider_hvp_components(
            state,
            vector,
            backend=second_backend,
            compiler=second_compiler,
            device_id=device_id,
            budget_bytes=second_budget_bytes,
            return_diagnostics=True,
        )
        for vector in vectors
    ]
    second_components = [item[0] for item in second_items]
    second_diagnostics = [item[1] for item in second_items]
    core = np.stack([item["core"] for item in second_components])
    pulay = np.stack([item["pulay"] for item in second_components])
    two_electron = np.stack([item["two_electron"] for item in second_components])
    second_seconds = time.perf_counter() - second_started

    nuclear_started = time.perf_counter()
    nuclear = np.stack([nuclear_hvp(state, vector) for vector in vectors])
    nuclear_seconds = time.perf_counter() - nuclear_started

    assembly_started = time.perf_counter()
    values = nuclear + core + pulay + two_electron + relaxation
    assembly_seconds = time.perf_counter() - assembly_started
    if not np.isfinite(values).all():
        raise FloatingPointError("nonfinite RHF HVP block; no result published")
    state.validate()

    total_seconds = time.perf_counter() - total_started
    response_phase_bound = (
        storage["total"]
        + retained_response_bytes
        + batch.solve_result.peak_workspace_bytes
    )
    relaxation_phase_bound = storage["total"] + (
        0 if relaxation_storage is None else relaxation_storage["numeric_peak_bytes"]
    )
    second_host_peak = max(
        (item["peak_host_bytes"] for item in second_diagnostics), default=0
    )
    second_device_peak = max(
        (item["peak_device_bytes"] for item in second_diagnostics), default=0
    )
    second_phase_bound = storage["total"] + second_host_peak + second_device_peak
    if second_phase_bound > total_budget_bytes:
        raise ValueError(
            "HVP block plus second-integral provider storage exceeds total_budget_bytes"
        )
    actual_bound = max(response_phase_bound, relaxation_phase_bound, second_phase_bound)
    first_programs = (
        tuple(
            sorted(
                {
                    program
                    for diagnostic in first_diagnostics
                    for program in diagnostic["program_identities"]
                }
            )
        )
        if first_diagnostics
        else None
    )
    identity = canonical_hash(
        {
            "schema": "vibeqc.rhf-hvp-block/v1",
            "source": state.source.identity,
            "reference": state.reference.identity,
            "operator": problem.operator_identity,
            "strategy": strategy,
            "solver_options": asdict(bounded_options),
            "nrhs": len(vectors),
            "directions": sha256(
                vectors.astype("<f8", copy=False).tobytes()
            ).hexdigest(),
            "jk_backend": jk_backend,
            "response_execution": response_execution,
            "resident_response": resident_identity,
            "first_backend": first_backend,
            "first_programs": first_programs,
            "second_backend": second_backend,
            "second_programs": tuple(
                sorted(
                    {
                        program
                        for diagnostic in second_diagnostics
                        for program in diagnostic["program_identities"]
                    }
                )
            ),
            "relaxation_backend": relaxation_backend,
            "relaxation_programs": (
                tuple(
                    sorted(
                        {
                            program
                            for diagnostic in relaxation_diagnostics
                            for program in diagnostic["program_identities"]
                        }
                    )
                )
                if relaxation_diagnostics
                else None
            ),
        }
    )
    diagnostics = {
        "source_identity": state.source.identity,
        "reference_identity": state.reference.identity,
        "operator_identity": problem.operator_identity,
        "nrhs": len(vectors),
        "strategy": strategy,
        "rhs_rank": batch.solve_result.rhs_rank,
        "rank_deficient_rhs": batch.solve_result.rank_deficient_rhs,
        "response_operator_actions": batch.solve_result.operator_actions,
        "response_peak_workspace_bytes": batch.solve_result.peak_workspace_bytes,
        "persistent_numeric_bound": storage,
        "relaxation_numeric_bound": deepcopy(relaxation_storage),
        "response_phase_numeric_bound_bytes": response_phase_bound,
        "relaxation_phase_numeric_bound_bytes": relaxation_phase_bound,
        "second_integral_phase_numeric_bound_bytes": second_phase_bound,
        "complete_numeric_peak_bound_bytes": actual_bound,
        "total_budget_bytes": total_budget_bytes,
        "solver_options": asdict(bounded_options),
        "jk_backend": jk_backend,
        "response_execution": response_execution,
        "resident_response": deepcopy(resident_diagnostic),
        "retained_response_device_bytes": retained_response_bytes,
        "response_device_budget_bytes": response_device_budget_bytes,
        "first_backend": first_backend,
        "second_backend": second_backend,
        "second_integral_budget_bytes": second_budget_bytes,
        "second_integral_provider": deepcopy(second_diagnostics),
        "relaxation_backend": relaxation_backend,
        "relaxation_provider": deepcopy(relaxation_diagnostics),
        "execution_residency": (
            "mixed-host-device"
            if jk_backend == "cuda"
            or first_backend == "cuda"
            or relaxation_backend == "cuda"
            or second_backend == "cuda"
            else "host"
        ),
        "jk_statistics": jk_statistics,
        "jk_provider": deepcopy(jk_provider),
        "first_provider": deepcopy(first_diagnostics),
        "full_molecular_hessian_allocated": False,
        "timings_seconds": {
            "first_sources": first_seconds,
            "response_prepare_solve_reconstruct": response_seconds,
            "response_solve_many": batch.solve_result.seconds,
            "response_operator": batch.solve_result.operator_seconds,
            "response_orthogonalization": batch.solve_result.orthogonalization_seconds,
            "response_recycling": batch.solve_result.recycling_seconds,
            "relaxation_first_integrals": relaxation_seconds,
            "second_integral_hvps": second_seconds,
            "nuclear": nuclear_seconds,
            "final_assembly": assembly_seconds,
            "complete_block": total_seconds,
        },
        "memory_scope_exclusions": (
            "compiler metadata/native call stacks/CUDA context and library code"
        ),
    }
    arrays = (
        vectors,
        values,
        nuclear,
        core,
        pulay,
        two_electron,
        relaxation,
    )
    return RHFHVPBlockResult(
        *(immutable(value) for value in arrays),
        batch,
        identity,
        diagnostics,
    )


def rhf_hessian(
    state: NativeRHFState,
    *,
    block_size: int | None = None,
    strategy: str = "recycled",
    total_budget_bytes: int = 128 << 20,
    **hvp_kwargs: object,
) -> RHFHessianResult:
    """Assemble the raw full Cartesian RHF Hessian in bounded direction blocks.

    Columns are independent canonical atom/xyz unit directions. The output is
    never symmetrized; raw symmetry is diagnostic evidence. A request whose
    full output plus one block cannot fit the declared budget fails explicitly.
    """
    if not isinstance(state, NativeRHFState):
        raise TypeError("RHF Hessian requires NativeRHFState")
    state.validate()
    total_budget_bytes = _checked_budget(total_budget_bytes, "total_budget_bytes")
    coordinates = 3 * state.nat
    if block_size is None:
        block_size = min(4, coordinates)
    if type(block_size) is not int or not 1 <= block_size <= coordinates:
        raise ValueError("block_size must be between 1 and 3*natoms")
    output_bytes = coordinates * coordinates * 8
    # Raw-symmetry checking can hold matrix, matrix-matrix.T and abs(diff)
    # simultaneously; immutable publication needs the original plus one copy.
    output_peak_bound = 3 * output_bytes
    if output_peak_bound > total_budget_bytes:
        raise ValueError(
            "full Hessian output and publication exceed total_budget_bytes"
        )
    # This buffer belongs to the full assembler, not rhf_hvp_many: its own
    # validated direction copy is already included in the block inventory.
    caller_direction_bytes = block_size * coordinates * 8
    block_budget = total_budget_bytes - output_bytes - caller_direction_bytes
    if _block_persistent_bound(state, block_size)["total"] >= block_budget:
        raise ValueError(
            "full Hessian output plus block persistent numeric storage exceeds "
            "total_budget_bytes"
        )

    started = time.perf_counter()
    matrix = np.empty((coordinates, coordinates), dtype=np.float64)
    block_diagnostics = []
    for begin in range(0, coordinates, block_size):
        end = min(coordinates, begin + block_size)
        directions = np.zeros((end - begin, coordinates), dtype=np.float64)
        for local, column in enumerate(range(begin, end)):
            directions[local, column] = 1.0
        result = rhf_hvp_many(
            state,
            directions.reshape(end - begin, state.nat, 3),
            strategy=strategy,
            total_budget_bytes=block_budget,
            **hvp_kwargs,
        )
        matrix[:, begin:end] = result.values.reshape(end - begin, coordinates).T
        block_diagnostics.append(
            {
                "begin": begin,
                "end": end,
                "identity": result.identity,
                "diagnostics": result.diagnostics,
            }
        )
        # Assignment of the next call would otherwise keep the old RHS alive
        # throughout that call, allowing two complete blocks to overlap.
        del result, directions
    if not np.isfinite(matrix).all():
        raise FloatingPointError("nonfinite RHF Hessian; no result published")
    state.validate()
    symmetry_error = float(np.max(np.abs(matrix - matrix.T), initial=0.0))
    identity = canonical_hash(
        {
            "schema": "vibeqc.rhf-hessian-block/v1",
            "source": state.source.identity,
            "reference": state.reference.identity,
            "block_size": block_size,
            "strategy": strategy,
            "blocks": tuple(item["identity"] for item in block_diagnostics),
        }
    )
    diagnostics = {
        "source_identity": state.source.identity,
        "reference_identity": state.reference.identity,
        "block_size": block_size,
        "block_count": len(block_diagnostics),
        "strategy": strategy,
        "output_bytes": output_bytes,
        "output_peak_bound_bytes": output_peak_bound,
        "caller_direction_bytes": caller_direction_bytes,
        "complete_numeric_peak_bound_bytes": max(
            output_peak_bound,
            output_bytes
            + caller_direction_bytes
            + max(
                item["diagnostics"]["complete_numeric_peak_bound_bytes"]
                for item in block_diagnostics
            ),
        ),
        "block_budget_bytes": block_budget,
        "total_budget_bytes": total_budget_bytes,
        "raw_symmetry_error": symmetry_error,
        "posthoc_symmetrization": False,
        "blocks": block_diagnostics,
        "seconds": time.perf_counter() - started,
        "public_calculator_endpoint": False,
    }
    return RHFHessianResult(immutable(matrix), identity, diagnostics)
