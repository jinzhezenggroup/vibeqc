"""Closed-shell stationary nuclear perturbations through the shared solver.

RHF and semilocal RKS/CPKS consumers share the metric, occupied-response and
energy-weighted-density conventions here. Method-specific density-to-Fock
physics belongs to the response operator, not to the Hessian consumer.
"""

import typing
from dataclasses import dataclass

import numpy as np

from tools.vibeqc_posthf.reference import immutable
from tools.vibeqc_response import (
    GMRESOptions,
    MultiRHSResult,
    SolveResult,
    solve,
    solve_many,
)

from .response import build_stationary_nuclear_rhs, metric_density_response_mo


@dataclass(frozen=True, eq=False)
class StationaryNuclearResponse:
    """Detached response of occupied orbitals and densities to one perturbation."""

    rhs: np.ndarray
    coefficient_derivative: np.ndarray
    occupied_energy_derivative: np.ndarray
    density_derivative: np.ndarray
    energy_weighted_density_derivative: np.ndarray
    solve_result: SolveResult


@dataclass(frozen=True, eq=False)
class StationaryNuclearBatchResponse:
    """Responses to a bounded set of perturbations from one shared multi-RHS solve."""

    responses: tuple[StationaryNuclearResponse, ...]
    solve_result: MultiRHSResult

    @property
    def converged(self) -> typing.Any:
        return self.solve_result.converged


@dataclass(frozen=True, eq=False)
class _PreparedNuclearPerturbation:
    rhs: np.ndarray
    frozen_mo: np.ndarray
    overlap_mo: np.ndarray


def _matrix(value: typing.Any, n: typing.Any, name: typing.Any) -> typing.Any:
    array = np.asarray(value)
    if (
        array.shape != (n, n)
        or array.dtype.kind not in "iuf"
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} must be a finite real AO matrix")
    array = np.array(array, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be representable in FP64")
    if not np.allclose(array, array.T, atol=2e-10, rtol=2e-12):
        raise ValueError(f"{name} must be symmetric")
    return array


def _validate_operator(operator: typing.Any) -> typing.Any:
    problem = getattr(operator, "problem", None)
    induced_fock = getattr(operator, "induced_fock", None)
    if (
        problem is None
        or problem.method not in ("rhf", "cpks")
        or not callable(induced_fock)
    ):
        raise TypeError(
            "expected a closed-shell RHF/CPKS response operator with induced_fock"
        )
    ref = problem.reference
    nmo, nocc = ref.nmo, ref.nocc
    layout = problem.layout
    if layout.occupied != tuple(range(nocc)) or layout.virtual != tuple(
        range(nocc, nmo)
    ):
        raise ValueError(
            "nuclear response requires the complete canonical closed-shell space"
        )
    validate_current = getattr(operator, "validate_current", None)
    if callable(validate_current):
        validate_current()
    validate = getattr(operator.backend, "validate_reference", None)
    if callable(validate):
        validate(ref)
    return ref, layout


def _prepare_stationary_nuclear_perturbation(
    operator: typing.Any, frozen_fock: typing.Any, overlap: typing.Any
) -> typing.Any:
    """Build one nuclear RHS without running Krylov or publishing a response."""
    ref, _ = _validate_operator(operator)
    nmo, nocc = ref.nmo, ref.nocc
    frozen = _matrix(frozen_fock, nmo, "frozen Fock derivative")
    s1 = _matrix(overlap, nmo, "overlap derivative")
    C, eps = ref.coefficients, ref.orbital_energies

    frozen_mo = C.T @ frozen @ C
    overlap_mo = C.T @ s1 @ C
    metric_dm_mo = metric_density_response_mo(overlap_mo, nocc=nocc)
    metric_fock_mo = C.T @ operator.induced_fock(C @ metric_dm_mo @ C.T) @ C
    rhs = build_stationary_nuclear_rhs(
        frozen_mo, overlap_mo, metric_fock_mo, eps, nocc=nocc
    )
    values = (rhs, frozen_mo, overlap_mo)
    if not all(np.isfinite(value).all() for value in values):
        raise FloatingPointError("nonfinite nuclear perturbation preparation")
    return _PreparedNuclearPerturbation(*(immutable(value) for value in values))


def _reconstruct_stationary_nuclear_response(
    operator: typing.Any, prepared: typing.Any, result: typing.Any
) -> typing.Any:
    """Reconstruct occupied and density responses from one solved rotation vector."""
    if not isinstance(prepared, _PreparedNuclearPerturbation):
        raise TypeError("expected a prepared stationary nuclear perturbation")
    if not isinstance(result, SolveResult):
        raise TypeError("expected a SolveResult")
    result.require_converged()
    ref, layout = _validate_operator(operator)
    C, eps = ref.coefficients, ref.orbital_energies
    nocc = ref.nocc
    mocc, e_i = C[:, :nocc], eps[:nocc]

    x_ia = layout.as_ia(result.solution)
    mo1 = -0.5 * prepared.overlap_mo[:, :nocc]
    mo1[nocc:, :] += x_ia.T
    c1 = C @ mo1
    density = 2.0 * (c1 @ mocc.T + mocc @ c1.T)
    hs = prepared.frozen_mo[:, :nocc] - prepared.overlap_mo[:, :nocc] * e_i[None, :]
    hs += C.T @ operator.induced_fock(density) @ mocc
    e1 = hs[:nocc, :] + mo1[:nocc, :] * (e_i[:, None] - e_i[None, :])
    left = (c1 * e_i[None, :]) @ mocc.T
    energy_density = 2.0 * (left + left.T + mocc @ e1 @ mocc.T)
    values = (
        prepared.rhs,
        c1,
        e1,
        density,
        energy_density,
    )
    if not all(np.isfinite(value).all() for value in values):
        raise FloatingPointError("nonfinite nuclear response; no result published")
    return StationaryNuclearResponse(*(immutable(value) for value in values), result)


def solve_stationary_nuclear_perturbation(
    operator: typing.Any,
    frozen_fock: typing.Any,
    overlap: typing.Any,
    *,
    options: typing.Any = None,
    resident_reconstruction_consumer: typing.Any = None,
) -> typing.Any:
    """Solve A x = -b once, retaining occupied metric and energy responses."""
    if options is not None and not isinstance(options, GMRESOptions):
        raise TypeError("options must be GMRESOptions")
    if resident_reconstruction_consumer is not None and not callable(
        resident_reconstruction_consumer
    ):
        raise TypeError("resident_reconstruction_consumer must be callable")
    prepared = _prepare_stationary_nuclear_perturbation(operator, frozen_fock, overlap)
    _, layout = _validate_operator(operator)

    def consume_resident_solution(engine: typing.Any, solution: typing.Any) -> None:
        reconstruct = getattr(engine, "reconstruct_nuclear_response", None)
        if reconstruct is None:
            raise ValueError(
                "resident reconstruction requires a compatible resident response engine"
            )
        resident_reconstruction_consumer(
            reconstruct(solution, prepared.frozen_mo, prepared.overlap_mo)
        )

    result = solve(
        operator,
        layout.pack(-prepared.rhs),
        options=options or GMRESOptions(),
        raise_on_failure=True,
        collect_basis=False,
        solution_consumer=(
            consume_resident_solution
            if resident_reconstruction_consumer is not None
            else None
        ),
    )
    return _reconstruct_stationary_nuclear_response(operator, prepared, result)


def solve_stationary_nuclear_perturbations(
    operator: typing.Any,
    frozen_focks: typing.Any,
    overlaps: typing.Any,
    *,
    strategy: typing.Any = "recycled",
    options: typing.Any = None,
    resident_reconstruction_consumers: typing.Any = None,
) -> typing.Any:
    """Solve closed-shell stationary perturbations with one multi-RHS call.

    Inputs have shape (nrhs, nmo, nmo). RHS construction retains the exact
    single-perturbation metric convention; only the nonredundant Krylov solve is
    shared/recycled. Published responses preserve input order.
    """
    if strategy not in ("sequential", "blocked", "recycled"):
        raise ValueError("strategy must be sequential, blocked or recycled")
    if options is not None and not isinstance(options, GMRESOptions):
        raise TypeError("options must be GMRESOptions")
    ref, layout = _validate_operator(operator)
    frozen_values = np.asarray(frozen_focks)
    overlap_values = np.asarray(overlaps)
    expected_tail = (ref.nmo, ref.nmo)
    if (
        frozen_values.ndim != 3
        or overlap_values.ndim != 3
        or frozen_values.shape != overlap_values.shape
        or frozen_values.shape[1:] != expected_tail
        or frozen_values.shape[0] < 1
    ):
        raise ValueError(
            "multi-RHS frozen Fock/overlap inputs must share shape "
            "(nrhs, nmo, nmo) with nrhs >= 1"
        )
    prepared = tuple(
        _prepare_stationary_nuclear_perturbation(operator, frozen, overlap)
        for frozen, overlap in zip(frozen_values, overlap_values, strict=True)
    )
    consumers = (
        (None,) * len(prepared)
        if resident_reconstruction_consumers is None
        else tuple(resident_reconstruction_consumers)
    )
    if len(consumers) != len(prepared) or any(
        consumer is not None and not callable(consumer) for consumer in consumers
    ):
        raise ValueError(
            "resident_reconstruction_consumers must match perturbation count"
        )

    def solution_consumer(
        item: _PreparedNuclearPerturbation, consumer: typing.Any
    ) -> typing.Any:
        if consumer is None:
            return None

        def consume(engine: typing.Any, solution: typing.Any) -> None:
            reconstruct = getattr(engine, "reconstruct_nuclear_response", None)
            if reconstruct is None:
                raise ValueError(
                    "resident reconstruction requires a compatible resident response engine"
                )
            consumer(reconstruct(solution, item.frozen_mo, item.overlap_mo))

        return consume

    packed = np.column_stack([layout.pack(-item.rhs) for item in prepared])
    multi = solve_many(
        operator,
        packed,
        strategy=strategy,
        options=options or GMRESOptions(),
        raise_on_failure=True,
        collect_basis=False,
        solution_consumers=tuple(
            solution_consumer(item, consumer)
            for item, consumer in zip(prepared, consumers, strict=True)
        ),
    )
    responses = tuple(
        _reconstruct_stationary_nuclear_response(operator, item, result)
        for item, result in zip(prepared, multi.results, strict=True)
    )
    return StationaryNuclearBatchResponse(responses, multi)


# Compatibility aliases keep existing RHF consumers and annotations stable.
RHFNuclearResponse = StationaryNuclearResponse
RHFNuclearBatchResponse = StationaryNuclearBatchResponse


def _require_rhf_operator(operator: typing.Any) -> None:
    problem = getattr(operator, "problem", None)
    if problem is None or problem.method != "rhf":
        raise TypeError("RHF nuclear response requires an RHFResponseOperator")


def solve_rhf_nuclear_perturbation(
    operator: typing.Any,
    frozen_fock: typing.Any,
    overlap: typing.Any,
    *,
    options: typing.Any = None,
    resident_reconstruction_consumer: typing.Any = None,
) -> typing.Any:
    """Backward-compatible RHF wrapper over the stationary response consumer."""
    _require_rhf_operator(operator)
    return solve_stationary_nuclear_perturbation(
        operator,
        frozen_fock,
        overlap,
        options=options,
        resident_reconstruction_consumer=resident_reconstruction_consumer,
    )


def solve_rhf_nuclear_perturbations(
    operator: typing.Any,
    frozen_focks: typing.Any,
    overlaps: typing.Any,
    *,
    strategy: typing.Any = "recycled",
    options: typing.Any = None,
    resident_reconstruction_consumers: typing.Any = None,
) -> typing.Any:
    """Backward-compatible RHF multi-RHS wrapper over the stationary consumer."""
    _require_rhf_operator(operator)
    return solve_stationary_nuclear_perturbations(
        operator,
        frozen_focks,
        overlaps,
        strategy=strategy,
        options=options,
        resident_reconstruction_consumers=resident_reconstruction_consumers,
    )
