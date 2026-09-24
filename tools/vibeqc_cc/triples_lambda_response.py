"""Corrected RCCSD(T) Lambda and fixed-orbital response weights for #154 B.

The (T) energy remains the audited TensorIR primal.  Its generated amplitude
cotangents shift the ordinary RCCSD Lambda right-hand side; the Jacobian is
still the RCCSD residual Jacobian because standard (T) is noniterative.

Rationale: .agents/notes/implemented/numerics/2026-09-20-corrected-triples-lambda-binding.md
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.tensor import PackedLayout

from tools.vibeqc_response.implicit import (
    ImplicitSolveError,
    _immutable,
    checked_transpose_solve,
)
from tools.vibeqc_response.krylov import _vector_norm
from tools.vibeqc_response.problem import ResponseCompatibilityError

from .lambda_equations import PARAMETERS, build_parameter_vjp
from .lambda_response import BoundCCSDResponse
from .lambda_solver import BoundCCSDLambda, CCSDLambdaResult, _feed_hash
from .triples import build_triples_program
from .triples_response import accumulate_tile_triples_vjp

_DIRECT_TRIPLES_PARAMETERS = frozenset(("fov", "ovov", "ovvv", "ovoo"))


@dataclass(frozen=True)
class CorrectedLambdaResult:
    """Total multipliers for E_CCSD + E_(T), plus the explicit correction."""

    lambda1: np.ndarray
    lambda2: np.ndarray
    delta_lambda1: np.ndarray
    delta_lambda2: np.ndarray
    reference_identity: str
    cc_state_identity: str
    triples_source_identity: str
    lambda_residual_norm: float
    independent_lambda_residual_norm: float
    independent_lambda_residual_max: float
    iterations: int
    operator_actions: int
    provenance: typing.Mapping

    def __post_init__(self) -> None:
        for name in ("lambda1", "lambda2", "delta_lambda1", "delta_lambda2"):
            object.__setattr__(self, name, _immutable(getattr(self, name)))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))


def _triples_arrays(bound: BoundCCSDLambda) -> tuple[np.ndarray, ...]:
    o = bound.reference.nocc
    eps = bound.reference.orbital_energies
    return (
        bound.feeds["ovvv"],
        bound.feeds["ovoo"],
        bound.feeds["ovov"],
        bound.feeds["fov"],
        bound.feeds["t1"],
        bound.feeds["t2"],
        eps[:o],
        eps[o:],
    )


def _triples_sources(
    bound: BoundCCSDLambda,
    inputs: tuple[str, ...],
    *,
    vir_chunk_size: int | None,
    cuda_response: typing.Any = None,
) -> dict[str, np.ndarray]:
    """Return exact (T) VJP blocks without relabeling a CPU replay as CUDA."""

    bound._assert_current(bound.reference_identity)
    if cuda_response is None:
        values = accumulate_tile_triples_vjp(
            bound.reference.nocc,
            bound.reference.nmo - bound.reference.nocc,
            *_triples_arrays(bound),
            vir_chunk_size=vir_chunk_size,
            inputs=inputs,
            executor=bound.tensor_executor,
        )
        bound._assert_current(bound.reference_identity)
        return values

    # Local import avoids a module cycle: the CUDA owner imports the corrected-
    # Lambda result and mathematical source-identity helpers from this module.
    from .triples_response_cuda import CudaTriplesResponseResult

    if not isinstance(cuda_response, CudaTriplesResponseResult):
        raise TypeError("external triples response must be CudaTriplesResponseResult")
    nocc = bound.reference.nocc
    nvir = bound.reference.nmo - nocc
    if cuda_response.nocc != nocc or cuda_response.nvir != nvir:
        raise ResponseCompatibilityError(
            "CUDA triples response shape belongs to another CC state"
        )
    if cuda_response.vir_chunk_size != vir_chunk_size:
        raise ResponseCompatibilityError(
            "CUDA triples response tile schedule differs from the bound response"
        )
    provenance = cuda_response.provenance
    if (
        provenance.get("backend") != "cuda-fp64-resident-triples-vjp"
        or provenance.get("cpu_fallback") is not False
    ):
        raise ResponseCompatibilityError(
            "CUDA triples response provenance permits an unsupported execution path"
        )

    arrays = _triples_arrays(bound)
    expected_inputs = dict(
        zip(
            ("ovvv", "ovoo", "ovov", "fov", "t1", "t2", "eps_o", "eps_v"),
            arrays,
            strict=True,
        )
    )
    if cuda_response.input_identity != _feed_hash(expected_inputs):
        raise ResponseCompatibilityError(
            "CUDA triples response inputs belong to another CC state"
        )
    required = set(inputs)
    missing = sorted(
        (required - set(cuda_response.inputs)) | (required - set(cuda_response.sources))
    )
    if missing:
        raise ResponseCompatibilityError(
            f"CUDA triples response is missing required source blocks: {missing}"
        )

    values = {}
    for name in inputs:
        value = np.asarray(cuda_response.sources[name])
        expected = np.asarray(expected_inputs[name])
        if (
            value.dtype != np.float64
            or value.shape != expected.shape
            or not np.isfinite(value).all()
        ):
            raise ResponseCompatibilityError(
                f"CUDA triples response block {name!r} has invalid shape, dtype, or values"
            )
        values[name] = _immutable(value)
    bound._assert_current(bound.reference_identity)
    return values


def _source_identity(
    bound: BoundCCSDLambda,
    sources: dict[str, np.ndarray],
    projected_sources: dict[str, np.ndarray],
    vir_chunk_size: int | None,
) -> str:
    return canonical_hash(
        {
            "triples_primal": build_triples_program(
                bound.reference.nocc, bound.reference.nmo - bound.reference.nocc
            ).logical_hash,
            "dense_amplitude_sources": _feed_hash(sources),
            "projected_amplitude_sources": _feed_hash(projected_sources),
            "vir_chunk_size": vir_chunk_size,
        }
    )


def solve_corrected_lambda(
    bound: BoundCCSDLambda,
    baseline: CCSDLambdaResult,
    *,
    vir_chunk_size: int | None = None,
) -> CorrectedLambdaResult:
    """Solve J_CCSD^T lambda_total = -d(E_CCSD + E_(T))/dt."""

    # Reuse the existing response gate as independent evidence that the supplied
    # baseline multipliers really belong to this immutable CC/reference state.
    baseline_response = BoundCCSDResponse(bound, baseline)
    sources = _triples_sources(bound, ("t1", "t2"), vir_chunk_size=vir_chunk_size)
    # The (T) primal intentionally accepts a dense t2 tensor, while RCCSD
    # solves only independent simultaneous-pair-exchange coordinates. Apply
    # the adjoint of that unpacking map before entering Lambda coordinates.
    t2_layout = bound.layouts[1]
    projected_t2 = t2_layout.unpack(t2_layout.unpack_transpose(sources["t2"]))
    projected_sources = {"t1": sources["t1"], "t2": projected_t2}
    source = bound.sqrt_weights * bound._pack(
        (projected_sources["t1"], projected_sources["t2"])
    )
    shared_rhs = bound._rhs(bound.programs) - source
    independent_rhs = bound._rhs(bound.independent) - source
    owner = bound

    class Operator:
        dimension = len(owner.sqrt_weights)

        def apply(self, vector: np.ndarray) -> np.ndarray:
            return owner._transpose(owner.programs, vector)

    solved = checked_transpose_solve(
        Operator(),
        shared_rhs,
        solver=bound.solver,
        assert_current=lambda: bound._assert_current(bound.reference_identity),
    )
    independent = bound._transpose(bound.independent, solved.solution) - independent_rhs
    independent_norm = _vector_norm(independent)
    independent_max = float(np.max(np.abs(independent / bound.sqrt_weights)))
    if (
        max(solved.residual_norm, independent_norm, independent_max)
        > bound.options.lambda_tolerance
    ):
        raise ImplicitSolveError("corrected RCCSD(T) Lambda residual gate failed")

    total1, total2 = bound._unpack(solved.solution / bound.sqrt_weights)
    delta1 = total1 - baseline.lambda1
    delta2 = total2 - baseline.lambda2
    source_identity = _source_identity(
        bound, sources, projected_sources, vir_chunk_size
    )
    bound._assert_current(bound.reference_identity)
    return CorrectedLambdaResult(
        total1,
        total2,
        delta1,
        delta2,
        bound.reference_identity,
        bound.cc_state_identity,
        source_identity,
        solved.residual_norm,
        independent_norm,
        independent_max,
        solved.iterations,
        solved.operator_actions,
        {
            "lagrangian": "E_CCSD + E_(T) + <lambda_total, R_CCSD>",
            "jacobian": "RCCSD residual Jacobian; no iterative T3 equations",
            "triples_source_identity": source_identity,
            "baseline_lambda_identity": baseline_response.lambda_identity,
            "equation_identity": bound.equation_identity,
            "vir_chunk_size": vir_chunk_size,
            "tensor_backend": bound.tensor_backend,
            "orbital_response": "excluded",
        },
    )


@dataclass(frozen=True)
class CCSDTParameterWeight:
    """One combined fixed-orbital mathematical-input cotangent block."""

    parameter: str
    values: np.ndarray
    baseline_ccsd: np.ndarray
    direct_triples: np.ndarray
    delta_lambda: np.ndarray
    spec: typing.Any
    reference_identity: str
    cc_state_identity: str
    corrected_lambda_identity: str
    response_identity: str
    independent_delta_max_abs_error: float
    provenance: typing.Mapping

    def __post_init__(self) -> None:
        layout = PackedLayout.from_spec(self.spec)
        for name in ("values", "baseline_ccsd", "direct_triples", "delta_lambda"):
            value = np.asarray(getattr(self, name))
            layout.pack(value)  # validate shape, real dtype, finiteness and symmetry
            object.__setattr__(self, name, _immutable(value))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    def contract(self, direction: np.ndarray) -> float:
        direction = np.asarray(direction)
        PackedLayout.from_spec(self.spec).pack(direction)
        value = float(np.sum(self.values * direction))
        if not np.isfinite(value):
            raise FloatingPointError("nonfinite CCSD(T) parameter-weight contraction")
        return value


@dataclass(frozen=True, init=False, eq=False, repr=False)
class BoundCCSDTResponse:
    """Combine CCSD response, direct (T), and delta-Lambda contributions."""

    bound: BoundCCSDLambda
    baseline: BoundCCSDResponse
    corrected: CorrectedLambdaResult
    vir_chunk_size: int | None
    triples_response: typing.Any
    corrected_lambda_identity: str
    response_identity: str

    def __init__(
        self,
        bound: BoundCCSDLambda,
        baseline: CCSDLambdaResult,
        corrected: CorrectedLambdaResult,
        *,
        vir_chunk_size: int | None = None,
        triples_response: typing.Any = None,
    ) -> None:
        if (
            not isinstance(bound, BoundCCSDLambda)
            or not isinstance(baseline, CCSDLambdaResult)
            or not isinstance(corrected, CorrectedLambdaResult)
        ):
            raise TypeError(
                "RCCSD(T) response requires a bound RCCSD state, baseline Lambda, "
                "and corrected Lambda result"
            )
        object.__setattr__(self, "bound", bound)
        object.__setattr__(self, "baseline", BoundCCSDResponse(bound, baseline))
        object.__setattr__(self, "triples_response", triples_response)
        if (
            corrected.reference_identity != bound.reference_identity
            or corrected.cc_state_identity != bound.cc_state_identity
            or corrected.provenance.get("baseline_lambda_identity")
            != self.baseline.lambda_identity
            or corrected.provenance.get("equation_identity") != bound.equation_identity
            or corrected.provenance.get("vir_chunk_size") != vir_chunk_size
            or corrected.provenance.get("lagrangian")
            != "E_CCSD + E_(T) + <lambda_total, R_CCSD>"
        ):
            raise ResponseCompatibilityError(
                "corrected Lambda belongs to another CC state, schedule, or convention"
            )

        # Do not trust convergence flags or stored residual diagnostics. Rebuild
        # the triples amplitude source and independently replay both generated
        # CCSD transpose forms against the supplied total multipliers.
        sources = _triples_sources(
            bound,
            ("t1", "t2"),
            vir_chunk_size=vir_chunk_size,
            cuda_response=triples_response,
        )
        t2_layout = bound.layouts[1]
        projected_t2 = t2_layout.unpack(t2_layout.unpack_transpose(sources["t2"]))
        expected_source = _source_identity(
            bound, sources, {"t1": sources["t1"], "t2": projected_t2}, vir_chunk_size
        )
        if (
            corrected.triples_source_identity != expected_source
            or corrected.provenance.get("triples_source_identity") != expected_source
        ):
            raise ResponseCompatibilityError(
                "corrected Lambda source identity mismatch"
            )
        source = bound.sqrt_weights * bound._pack((sources["t1"], projected_t2))
        total_vector = bound.sqrt_weights * bound._pack(
            (corrected.lambda1, corrected.lambda2)
        )
        delta_vector = bound.sqrt_weights * bound._pack(
            (corrected.delta_lambda1, corrected.delta_lambda2)
        )
        baseline_vector = bound.sqrt_weights * bound._pack(
            (self.baseline.lambda1, self.baseline.lambda2)
        )
        if not np.allclose(
            delta_vector, total_vector - baseline_vector, atol=1e-12, rtol=1e-10
        ):
            raise ResponseCompatibilityError(
                "corrected Lambda delta does not match total-minus-baseline multipliers"
            )
        replay_norms = []
        for programs in (bound.programs, bound.independent):
            residual = bound._transpose(programs, total_vector) - (
                bound._rhs(programs) - source
            )
            replay_norms.extend(
                (
                    _vector_norm(residual),
                    float(np.max(np.abs(residual / bound.sqrt_weights))),
                )
            )
        if (
            not all(np.isfinite(value) for value in replay_norms)
            or max(replay_norms) > bound.options.lambda_tolerance
        ):
            raise ImplicitSolveError(
                "RCCSD(T) response requires a freshly verified corrected Lambda solution"
            )
        bound._assert_current(bound.reference_identity)

        object.__setattr__(self, "corrected", corrected)
        object.__setattr__(self, "vir_chunk_size", vir_chunk_size)
        corrected_lambda_identity = canonical_hash(
            {
                "cc_state": bound.cc_state_identity,
                "triples_source": corrected.triples_source_identity,
                "multipliers": _feed_hash(
                    {
                        "lambda1": corrected.lambda1,
                        "lambda2": corrected.lambda2,
                    }
                ),
            }
        )
        object.__setattr__(self, "corrected_lambda_identity", corrected_lambda_identity)
        response_identity = canonical_hash(
            {
                "baseline": self.baseline.response_identity,
                "corrected_lambda": corrected_lambda_identity,
                "triples_response": (
                    None
                    if triples_response is None
                    else triples_response.source_identity
                ),
                "scope": "fixed-orbital RCCSD(T) mathematical-input weights",
            }
        )

        object.__setattr__(self, "response_identity", response_identity)

    @property
    def parameters(self) -> tuple[str, ...]:
        return PARAMETERS

    def _delta_weight(self, parameter: str) -> tuple[np.ndarray, float]:
        extra = {
            "bar_correlation_energy": np.asarray(0.0),
            "bar_singles_residual": self.corrected.delta_lambda1,
            "bar_doubles_residual": self.corrected.delta_lambda2,
        }
        values = []
        for programs in (self.bound.programs, self.bound.independent):
            reverse = build_parameter_vjp(programs.primal, parameter)
            outputs = self.bound._tensor_execute(
                reverse.program,
                {**self.bound.feeds, **extra},
            )
            values.append(np.asarray(outputs[f"bar_{parameter}"]))
        if not np.allclose(values[0], values[1], atol=1e-12, rtol=1e-10):
            raise ImplicitSolveError(
                "independent corrected-Lambda parameter-weight check failed"
            )
        return values[0], float(np.max(np.abs(values[0] - values[1])))

    def weight(
        self, parameter: str, *, reference_identity: str
    ) -> CCSDTParameterWeight:
        if parameter not in PARAMETERS:
            raise ValueError(f"unsupported CCSD(T) response parameter {parameter!r}")
        self.bound._assert_current(reference_identity)
        baseline = self.baseline.weight(
            parameter, reference_identity=reference_identity
        )
        delta, error = self._delta_weight(parameter)
        direct = np.zeros_like(baseline.values)
        if parameter in _DIRECT_TRIPLES_PARAMETERS:
            dense_direct = _triples_sources(
                self.bound,
                (parameter,),
                vir_chunk_size=self.vir_chunk_size,
                cuda_response=self.triples_response,
            )[parameter]
            if dense_direct.shape != baseline.values.shape:
                raise ResponseCompatibilityError(
                    "triples and CCSD parameter block conventions disagree"
                )
            # The triples primal uses dense mathematical blocks. Project its
            # cotangent through the CCSD block's declared symmetry before the
            # two response conventions are combined.
            layout = PackedLayout.from_spec(baseline.spec)
            direct = layout.unpack(layout.unpack_transpose(dense_direct))
        values = baseline.values + direct + delta
        self.bound._assert_current(reference_identity)
        return CCSDTParameterWeight(
            parameter,
            values,
            baseline.values,
            direct,
            delta,
            baseline.spec,
            self.bound.reference_identity,
            self.bound.cc_state_identity,
            self.corrected_lambda_identity,
            self.response_identity,
            error,
            {
                "baseline_response": baseline.response_identity,
                "corrected_triples_source": self.corrected.triples_source_identity,
                "triples_response_backend": (
                    self.bound.tensor_backend
                    if self.triples_response is None
                    else self.triples_response.provenance["backend"]
                ),
                "decomposition": "CCSD baseline + direct (T) + delta-Lambda * dR_CCSD/dq",
                "orbital_response": "excluded",
                "physical_rdm": False,
            },
        )

    def orbital_energy_weights(
        self, *, reference_identity: str
    ) -> typing.Mapping[str, np.ndarray]:
        """Return direct dE_(T)/d eps_o,eps_v sources for later orbital response."""

        self.bound._assert_current(reference_identity)
        sources = _triples_sources(
            self.bound,
            ("eps_o", "eps_v"),
            vir_chunk_size=self.vir_chunk_size,
            cuda_response=self.triples_response,
        )
        result = MappingProxyType(
            {name: _immutable(np.asarray(sources[name])) for name in ("eps_o", "eps_v")}
        )
        self.bound._assert_current(reference_identity)
        return result
