"""Total RCCSD(T) orbital/metric response for issue #155 slice A.

This module connects the fixed-orbital standard-(T) response from #154 to the
validated conventional RCCSD raw-Hamiltonian and RHF Z-vector chain from #153.
It intentionally stops before nuclear integral derivatives: no force capability
or complete CCSD(T) analytic gradient is published here.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from tools.vibeqc_response.implicit import (
    ImplicitSolveError,
    ResponseGMRES,
    _immutable,
    checked_transpose_solve,
)
from tools.vibeqc_response.krylov import _vector_norm
from tools.vibeqc_response.problem import ResponseCompatibilityError
from tools.vibeqc_validation.schema import canonical_hash

from .complete_gradient import BoundCCSDGradient, CCSDGradientOptions
from .gradient_equations import build_fock_weight_program
from .lambda_equations import PARAMETERS
from .lambda_solver import _feed_hash
from .triples_lambda_response import BoundCCSDTResponse

_RESPONSE_FIELDS = (
    "hcore",
    "eri",
    "overlap",
    "rotation_gradient",
    "stationarity",
    "orbital_rhs",
)
_CANONICAL_FOCK_TOLERANCE = 1.0e-9
_MINIMUM_SAME_SPACE_GAP = 1.0e-10


def _sum_weight_maps(*values: typing.Mapping[str, np.ndarray]) -> typing.Mapping:
    if not values:
        raise ValueError("at least one response-weight map is required")
    result = {}
    for name in _RESPONSE_FIELDS:
        arrays = [np.asarray(value[name], dtype=np.float64) for value in values]
        total = np.array(arrays[0], copy=True)
        for array in arrays[1:]:
            if array.shape != total.shape:
                raise ResponseCompatibilityError(
                    f"response component {name!r} has inconsistent shapes"
                )
            total += array
        if not np.isfinite(total).all():
            raise ImplicitSolveError(f"nonfinite total response component {name!r}")
        result[name] = _immutable(total)
    return MappingProxyType(result)


def _minimum_same_space_gap(energies: np.ndarray, nocc: int) -> float:
    """Return the smallest non-redundant canonical gap within o or v spaces."""

    energies = np.asarray(energies, dtype=np.float64)
    values = []
    for block in (energies[:nocc], energies[nocc:]):
        if len(block) < 2:
            continue
        delta = np.abs(block[:, None] - block[None, :])
        values.extend(delta[np.triu_indices(len(block), 1)].tolist())
    return min(values, default=float("inf"))


@dataclass(frozen=True, init=False, eq=False, repr=False)
class BoundCCSDTOrbitalResponse:
    """Bind standard-(T) sources to the complete conventional RHF response chain.

    The fixed-orbital parameter weights already contain the CCSD baseline,
    direct (T), and corrected-Lambda pieces.  The missing standard-(T)
    denominator source is interpreted as a cotangent of the canonical Fock
    diagonal and reverse-differentiated through the exact #153 Hamiltonian
    primal.  The resulting raw h/g/metric and orbital terms are then combined
    before solving one fresh total RHF Z-vector.

    This is an internal small-system validation boundary.  It produces no
    nuclear-coordinate derivative and does not enable public force support.
    """

    def __init__(
        self,
        response: BoundCCSDTResponse,
        provider: typing.Any,
        *,
        options: CCSDGradientOptions | None = None,
    ) -> None:
        if not isinstance(response, BoundCCSDTResponse):
            raise TypeError("RCCSD(T) orbital response requires BoundCCSDTResponse")
        options = CCSDGradientOptions() if options is None else options
        if not isinstance(options, CCSDGradientOptions):
            raise TypeError(
                "RCCSD(T) orbital response options must be CCSDGradientOptions"
            )

        # Reuse the already-qualified #153 raw Hamiltonian, RHF operator,
        # curvature checks and lifetime gates.  Constructing this validation
        # owner does not evaluate nuclear integral derivatives.
        baseline = BoundCCSDGradient(response.baseline, provider, options=options)
        reference = baseline.reference
        response.bound._assert_current(reference.identity)
        if (
            response.bound is not response.baseline.bound
            or response.bound.reference_identity != reference.identity
            or response.bound.cc_state_identity
            != response.baseline.bound.cc_state_identity
        ):
            raise ResponseCompatibilityError(
                "RCCSD(T) response and RCCSD gradient owner refer to different states"
            )

        primal = baseline._run(baseline.programs.primal, baseline.raw_inputs)
        fock = np.asarray(primal["fock"])
        off_diagonal = fock - np.diag(np.diag(fock))
        if float(
            np.max(np.abs(off_diagonal))
        ) > _CANONICAL_FOCK_TOLERANCE or not np.allclose(
            np.diag(fock),
            reference.orbital_energies,
            atol=_CANONICAL_FOCK_TOLERANCE,
            rtol=0,
        ):
            raise ResponseCompatibilityError(
                "standard (T) denominator response requires a canonical RHF Fock basis"
            )
        minimum_same_space_gap = _minimum_same_space_gap(
            reference.orbital_energies, reference.nocc
        )
        if minimum_same_space_gap <= _MINIMUM_SAME_SPACE_GAP:
            raise ResponseCompatibilityError(
                "degenerate canonical occupied/virtual subspaces are not yet "
                "qualified for RCCSD(T) orbital response"
            )

        parameter_weights = tuple(
            response.weight(name, reference_identity=reference.identity)
            for name in PARAMETERS
        )

        zero_seeds = {
            "bar_" + name: _immutable(
                np.zeros_like(response.bound.feeds[name], dtype=np.float64)
            )
            for name in PARAMETERS
        }
        zero_seeds["bar_reference_electronic_energy"] = _immutable(np.asarray(0.0))

        def parameter_pullback(field: str | None) -> typing.Mapping:
            seeds = dict(zero_seeds)
            for weight in parameter_weights:
                seeds["bar_" + weight.parameter] = (
                    weight.values if field is None else getattr(weight, field)
                )
            return baseline._pullback(seeds)

        combined_parameters = parameter_pullback(None)
        direct_triples = parameter_pullback("direct_triples")
        delta_lambda = parameter_pullback("delta_lambda")
        baseline_correlation = baseline.component_weights["correlation"]
        reconstructed_parameters = _sum_weight_maps(
            baseline_correlation, direct_triples, delta_lambda
        )
        for name in _RESPONSE_FIELDS:
            if not np.allclose(
                combined_parameters[name],
                reconstructed_parameters[name],
                atol=2e-11,
                rtol=2e-11,
            ):
                raise ImplicitSolveError(
                    "RCCSD(T) parameter response decomposition failed"
                )

        orbital_energy_weights = response.orbital_energy_weights(
            reference_identity=reference.identity
        )
        n, o = reference.nmo, reference.nocc
        bar_fock = np.zeros((n, n), dtype=np.float64)
        diagonal = np.concatenate(
            (
                np.asarray(orbital_energy_weights["eps_o"]),
                np.asarray(orbital_energy_weights["eps_v"]),
            )
        )
        bar_fock[np.arange(n), np.arange(n)] = diagonal
        fock_program = build_fock_weight_program(o, n - o)
        denominator = baseline._run(
            fock_program,
            {**baseline.raw_inputs, "bar_fock": bar_fock},
        )
        correlation = _sum_weight_maps(combined_parameters, denominator)

        same_space = max(
            float(np.max(np.abs(correlation["stationarity"][:o, :o]))),
            float(np.max(np.abs(correlation["stationarity"][o:, o:]))),
        )
        if same_space > options.stationarity_tolerance:
            raise ImplicitSolveError(
                "RCCSD(T) same-space canonical stationarity is not satisfied"
            )

        rhs = _immutable(np.asarray(correlation["orbital_rhs"]).reshape(-1))
        operator = baseline.operator
        owner = baseline

        class Transpose:
            dimension = operator.dimension

            def apply(self, vector: typing.Any) -> typing.Any:
                return operator.apply_transpose(vector)

        solver = ResponseGMRES(operator.dimension, options.z_options)
        z = checked_transpose_solve(
            Transpose(),
            rhs,
            solver=solver,
            assert_current=owner._assert_current,
        )
        independent_z_residual = _vector_norm(
            baseline.orbital_matrix.T @ z.solution - rhs
        )
        if independent_z_residual > options.orbital_residual_tolerance:
            raise ImplicitSolveError(
                "independent RCCSD(T) physical Z-vector residual failed"
            )

        z_seeds = dict(zero_seeds)
        z_seeds["bar_fov"] = -z.solution.reshape(o, n - o)
        orbital = baseline._pullback(z_seeds)
        hf = baseline.component_weights["hf"]
        total = _sum_weight_maps(hf, correlation, orbital)

        stationarity = float(np.max(np.abs(total["stationarity"])))
        if stationarity > options.stationarity_tolerance:
            raise ImplicitSolveError(
                "complete HF+RCCSD(T)+Z orbital stationarity failed"
            )

        components = MappingProxyType(
            {
                "hf": hf,
                "ccsd_baseline": baseline_correlation,
                "direct_triples": direct_triples,
                "delta_lambda": delta_lambda,
                "triples_denominator": denominator,
                "orbital_response": orbital,
            }
        )
        reconstructed_total = _sum_weight_maps(*components.values())
        for name in _RESPONSE_FIELDS:
            if not np.allclose(
                total[name], reconstructed_total[name], atol=2e-11, rtol=2e-11
            ):
                raise ImplicitSolveError(
                    "RCCSD(T) total orbital-response component sum failed"
                )

        response_identity = canonical_hash(
            {
                "fixed_orbital_response": response.response_identity,
                "baseline_gradient_weights": baseline.weight_identity,
                "fock_response_program": fock_program.logical_hash,
                "operator": baseline.operator_identity,
                "parameter_weights": _feed_hash(
                    {weight.parameter: weight.values for weight in parameter_weights}
                ),
                "orbital_energy_weights": _feed_hash(dict(orbital_energy_weights)),
                "z_state": _feed_hash({"rhs": rhs, "solution": z.solution}),
                "scope": "RCCSD(T) total orbital/metric response; no nuclear derivatives",
            }
        )

        put = lambda name, value: object.__setattr__(self, name, value)
        for name, value in (
            ("response", response),
            ("provider", provider),
            ("baseline", baseline),
            ("reference", reference),
            ("options", options),
            ("weights", total),
            ("correlation_weights", correlation),
            ("component_weights", components),
            ("parameter_weights", parameter_weights),
            ("orbital_energy_weights", orbital_energy_weights),
            ("orbital_rhs", rhs),
            ("z_result", z),
            ("independent_z_residual", independent_z_residual),
            ("same_space_stationarity", same_space),
            ("orbital_stationarity", stationarity),
            ("minimum_orbital_curvature", baseline.minimum_orbital_curvature),
            ("minimum_same_space_gap", minimum_same_space_gap),
            ("fock_response_program", fock_program),
            ("reference_identity", reference.identity),
            ("cc_state_identity", response.bound.cc_state_identity),
            ("fixed_orbital_response_identity", response.response_identity),
            ("response_identity", response_identity),
        ):
            put(name, value)
        self._assert_current()

    @property
    def z_residual(self) -> float:
        return max(self.z_result.residual_norm, self.independent_z_residual)

    def _assert_current(self) -> None:
        self.baseline._assert_current()
        self.response.bound._assert_current(self.reference_identity)
        if (
            self.response.response_identity != self.fixed_orbital_response_identity
            or self.response.bound.cc_state_identity != self.cc_state_identity
        ):
            raise ResponseCompatibilityError(
                "RCCSD(T) orbital response is stale relative to its fixed-orbital state"
            )
