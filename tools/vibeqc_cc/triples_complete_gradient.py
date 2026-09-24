"""Complete conventional RCCSD(T) analytic-gradient validation endpoint.

This module consumes the already-qualified total orbital/metric response from
``triples_orbital_response`` and reuses the existing RCCSD derivative consumers.
No second CC, Lambda, Z-vector, or nuclear-derivative equation stack is introduced.
The endpoint remains internal: public Calculator/native RCCSD(T) force capability
is deliberately unchanged.
"""

from __future__ import annotations

import time
import typing
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np
from vibeqc_compiler.common.evidence import canonical_hash

from tools.vibeqc_posthf.export import export_rhf
from tools.vibeqc_posthf.providers import ConventionalProvider
from tools.vibeqc_response.implicit import (
    ImplicitSolveError,
    _array,
    _checked_bytes,
    _immutable,
)
from tools.vibeqc_response.problem import ResponseCompatibilityError

from .complete_gradient import (
    BoundCCSDGradient,
    CCSDGradientOptions,
    CCSDGradientResult,
    _derivative_bytes,
    _validate_source,
)
from .gradient_equations import (
    build_ao_eri_weight_block_program,
    build_ao_one_electron_weight_program,
    build_ao_weight_program,
)
from .lambda_solver import BoundCCSDLambda, _graph_bytes
from .solver import solve
from .triples import triples_energy
from .triples_lambda_response import (
    BoundCCSDTResponse,
    _triples_arrays,
    solve_corrected_lambda,
)
from .triples_orbital_response import BoundCCSDTOrbitalResponse


@dataclass(frozen=True, init=False, eq=False, repr=False)
class BoundCCSDTGradient(BoundCCSDGradient):
    """Attach shared AO/nuclear derivative consumers to one total RCCSD(T) response.

    ``BoundCCSDTOrbitalResponse`` already owns the complete HF + CCSD + (T)
    Lagrangian weights and the single physical occupied-virtual Z solve.  This
    class therefore prepares only the AO back-transform/derivative-consumer
    stage and inherits the proven CPU/CUDA contractions from ``BoundCCSDGradient``.
    """

    def __init__(
        self,
        response: BoundCCSDTOrbitalResponse,
        *,
        options: CCSDGradientOptions | None = None,
    ) -> None:
        started = time.perf_counter()
        if not isinstance(response, BoundCCSDTOrbitalResponse):
            raise TypeError("RCCSD(T) gradient requires a total orbital-response owner")
        options = response.options if options is None else options
        if not isinstance(options, CCSDGradientOptions):
            raise TypeError("RCCSD(T) gradient options must be CCSDGradientOptions")
        if options != response.options:
            raise ResponseCompatibilityError(
                "RCCSD(T) derivative consumer options must match the prepared response"
            )

        provider = response.provider
        source = response.baseline.source
        reference = response.reference
        _validate_source(source)
        if provider is not response.baseline.provider or provider.source is not source:
            raise ResponseCompatibilityError(
                "RCCSD(T) gradient provider differs from the prepared orbital response"
            )

        put = lambda name, value: object.__setattr__(self, name, value)
        for name, value in (
            ("response", response),
            ("provider", provider),
            ("source", source),
            ("reference", reference),
            ("reference_identity", reference.identity),
            ("source_identity", source.identity),
            ("options", options),
            ("weights", response.weights),
            ("component_weights", response.component_weights),
            ("z_result", response.z_result),
            ("independent_z_residual", response.independent_z_residual),
            ("orbital_stationarity", response.orbital_stationarity),
            ("minimum_orbital_curvature", response.minimum_orbital_curvature),
            ("same_space_stationarity", response.same_space_stationarity),
            ("operator_identity", response.baseline.operator_identity),
            ("tensor_executor", response.response.parameter_executor),
        ):
            put(name, value)
        self._assert_current()

        n = reference.nmo
        ao_program = build_ao_weight_program(n)
        ao_one_program = build_ao_one_electron_weight_program(n)
        largest_shell = max(source.shell_sizes)
        ao_eri_block_program = (
            build_ao_eri_weight_block_program(n, (largest_shell,) * 4)
            if options.derivative_backend == "cuda"
            and options.eri_weight_mode == "shell"
            else None
        )
        put("ao_program", ao_program)
        put("ao_one_program", ao_one_program)
        put("ao_eri_block_program", ao_eri_block_program)

        ao_numeric_programs = (
            (ao_one_program, ao_eri_block_program)
            if ao_eri_block_program is not None
            else (ao_program,)
        )
        retained_weight_bytes = sum(
            np.asarray(response.weights[name]).nbytes
            for name in ("hcore", "eri", "overlap")
        )
        retained_component_bytes = sum(
            np.asarray(weights[name]).nbytes
            for weights in response.component_weights.values()
            for name in ("hcore", "eri", "overlap")
        )
        required = (
            retained_weight_bytes
            + retained_component_bytes
            + 2 * max(_graph_bytes(program) for program in ao_numeric_programs)
            + (
                3 * _derivative_bytes(source)
                if options.derivative_backend == "cpu"
                else 0
            )
        )
        _checked_bytes(required, "RCCSD(T) derivative-contraction logical reservation")
        if required > options.max_bytes:
            raise ImplicitSolveError(
                "RCCSD(T) derivative-contraction host budget exceeded before derivatives"
            )
        put("logical_reserved_host_bytes", required)
        put(
            "weight_identity",
            canonical_hash(
                {
                    "total_orbital_response": response.response_identity,
                    "ao_transform": ao_program.logical_hash,
                    "ao_one_electron_transform": ao_one_program.logical_hash,
                    "ao_eri_block_transform": (
                        None
                        if ao_eri_block_program is None
                        else ao_eri_block_program.logical_hash
                    ),
                    "derivative_backend": options.derivative_backend,
                    "eri_weight_mode": options.eri_weight_mode,
                    "one_electron_schedule": options.one_electron_schedule,
                    "scope": "complete RCCSD(T) AO/nuclear derivative contraction",
                }
            ),
        )
        put(
            "timings",
            MappingProxyType(
                {"prepare_derivative_consumer_seconds": time.perf_counter() - started}
            ),
        )
        self._assert_current()

    def _run(self, program: typing.Any, feeds: typing.Any) -> typing.Any:
        if self.tensor_executor is None:
            return super()._run(program, feeds)
        self._assert_current()
        outputs = self.tensor_executor.execute(program, feeds)
        if set(outputs) != set(program.outputs):
            raise ResponseCompatibilityError(
                "RCCSD(T) CUDA tensor executor returned an incomplete output set"
            )
        result = MappingProxyType(
            {
                name: _immutable(_array(outputs[name], node.spec.shape, name))
                for name, node in program.outputs.items()
            }
        )
        self._assert_current()
        return result

    def _assert_current(self) -> None:
        self.response._assert_current()
        if (
            self.provider is not self.response.provider
            or self.source is not self.response.baseline.source
            or self.reference is not self.response.reference
            or self.source.identity != self.source_identity
            or self.reference.identity != self.reference_identity
            or self.response.baseline.operator_identity != self.operator_identity
        ):
            raise ResponseCompatibilityError(
                "RCCSD(T) derivative consumer is stale relative to its prepared response"
            )

    def _result(
        self,
        gradient: typing.Any,
        physical: typing.Any,
        integral: typing.Any,
        diagnostics: typing.Any,
    ) -> CCSDGradientResult:
        state = self.response
        bound = state.response.bound
        correlation_ccsd = float(
            state.response.baseline._execute_tensor(
                bound.independent.primal,
                bound.feeds,
            )["correlation_energy"]
        )
        nocc = self.reference.nocc
        nvir = self.reference.nmo - nocc
        triples = float(triples_energy(nocc, nvir, *_triples_arrays(bound)))
        correlation = correlation_ccsd + triples
        corrected = state.response.corrected
        self._assert_current()
        return CCSDGradientResult(
            self.reference.reference_energy + correlation,
            correlation,
            np.asarray(gradient).reshape(-1, 3),
            {k: np.asarray(v).reshape(-1, 3) for k, v in physical.items()},
            {k: np.asarray(v).reshape(-1, 3) for k, v in integral.items()},
            self.reference.scf_residual,
            max(bound.cc_r1_max, bound.cc_r2_max),
            max(
                corrected.lambda_residual_norm,
                corrected.independent_lambda_residual_norm,
                corrected.independent_lambda_residual_max,
            ),
            state.z_residual,
            state.orbital_stationarity,
            state.minimum_orbital_curvature,
            self.reference_identity,
            state.cc_state_identity,
            state.response_identity,
            self.operator_identity,
            self.source_identity,
            {
                "weight_identity": self.weight_identity,
                "method": "standard-canonical-rccsd(t)",
                "hamiltonian_id": self.reference.hamiltonian_id,
                "logical_reserved_host_bytes": self.logical_reserved_host_bytes,
                "provider_budget_bytes": self.provider.budget_bytes,
                "native_hf_backend": self.reference.hf_backend,
                "tensor_backend": (
                    bound.tensor_backend
                    if self.tensor_executor is None
                    else self.tensor_executor.backend
                ),
                "orbital_backend": self.response.baseline.response_backend.identity,
                "orbital_solver": "shared-response-gmres",
                "response_execution": self.response.response_execution,
                "dense_orbital_curvature_check": True,
                "dense_cc_jacobian": False,
                "dense_mo_eri_and_weights": True,
                "native_public_force_capability": False,
                "triples_gradient": True,
                "triples_energy": triples,
                "ccsd_correlation_energy": correlation_ccsd,
                "corrected_lambda_identity": state.response.corrected_lambda_identity,
                "fixed_orbital_response_identity": state.fixed_orbital_response_identity,
                "total_orbital_response_identity": state.response_identity,
                "same_space_stationarity": state.same_space_stationarity,
                "minimum_same_space_gap": state.minimum_same_space_gap,
                "z_iterations": state.z_result.iterations,
                "z_operator_actions": state.z_result.operator_actions,
                "memory_boundary": (
                    "prepared response is caller-owned; logical reservation counts retained "
                    "h/g/S weights plus derivative-consumer numeric state"
                ),
                **dict(self.timings),
                **diagnostics,
            },
        )


def complete_ccsdt_gradient_validation(
    source: typing.Any,
    *,
    options: CCSDGradientOptions | None = None,
    vir_chunk_size: int | None = 1,
) -> CCSDGradientResult:
    """Run fresh RHF -> RCCSD -> corrected Lambda -> total Z -> RCCSD(T) gradient.

    This is still an internal small-system conventional-RHF qualification endpoint.
    Public/native RCCSD(T) forces, DF, frozen core, open shell and ECP remain
    unsupported.  ``derivative_backend='cuda'`` moves only the final AO/nuclear
    derivative consumers to CUDA; the validated response/control chain remains CPU-owned.
    """

    started = time.perf_counter()
    options = CCSDGradientOptions() if options is None else options
    if not isinstance(options, CCSDGradientOptions):
        raise TypeError("RCCSD(T) gradient options must be CCSDGradientOptions")
    _validate_source(source)
    if (
        options.derivative_backend == "cpu"
        and 3 * _derivative_bytes(source) > options.max_bytes
    ):
        raise ImplicitSolveError(
            "RCCSD(T) derivative state budget exceeded before HF execution"
        )

    reference, _ = export_rhf(
        source,
        tolerance=options.scf_tolerance,
        max_iterations=options.scf_max_iterations,
    )
    if reference.scf_residual > 1e-9:
        raise ImplicitSolveError(
            "RCCSD(T) gradient requires a strictly converged physical RHF reference"
        )
    with ConventionalProvider(
        reference, source, budget_bytes=options.provider_budget_bytes
    ) as provider:
        cc = solve(reference, provider, options=options.cc_options)
        if not cc.converged:
            raise ImplicitSolveError(f"RCCSD(T) gradient primal failed: {cc.reason}")

        def current_reference() -> str:
            source._check_open()
            if provider._closed:
                raise ResponseCompatibilityError("RCCSD(T) gradient provider closed")
            return provider.snapshot.identity

        bound = BoundCCSDLambda(
            reference,
            cc,
            options=options.lambda_options,
            current_reference=current_reference,
            backend="native-cpu",
        )
        baseline_lambda = bound.solve(reference_identity=reference.identity)
        corrected = solve_corrected_lambda(
            bound,
            baseline_lambda,
            vir_chunk_size=vir_chunk_size,
        )
        fixed_response = BoundCCSDTResponse(
            bound,
            baseline_lambda,
            corrected,
            vir_chunk_size=vir_chunk_size,
        )
        orbital_response = BoundCCSDTOrbitalResponse(
            fixed_response,
            provider,
            options=options,
        )
        gradient = BoundCCSDTGradient(
            orbital_response,
            options=options,
        ).gradient()

    return replace(
        gradient,
        diagnostics={
            **gradient.diagnostics,
            "total_endpoint_seconds": time.perf_counter() - started,
        },
    )
