"""Bounded analytic semilocal RKS molecular Hessian-vector products.

This is the first complete LDA/PBE RKS consumer of the shared #178 second-
integral providers, #179 CPKS solver, #958 grid/AO geometry response and #964
mixed XC contraction. It is deliberately a tools endpoint: CPU, all-electron,
closed-shell Cartesian RKS, at most 12 AOs/four atoms. Public Calculator
Hessian capability and production-size qualification remain separate work.
"""

from __future__ import annotations

import time
import typing
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import numpy as np
from vibeqc._dft_gradient import StationaryDerivativeContract, _native_ao_atoms
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryMeanField,
)
from vibeqc_compiler.method.stationary_hvp import StationaryHVPPlan
from vibeqc_compiler.xc.contractions import ContractionProgram
from vibeqc_compiler.xc.grid_response import (
    partition_mixed_response,
    partition_response,
)

from tools.vibeqc_posthf.reference import immutable
from tools.vibeqc_response import NativeRKSResponse

from .analytic import (
    _provider_data,
    _run_eri,
    _run_one_electron,
    _second_provider_diagnostics,
    nuclear_hvp,
)
from .first_order import (
    checked_direction,
    generated_coulomb_directional_first_order,
    generated_coulomb_relaxation_components,
)
from .native import NativeRHFState
from .perturbation import solve_stationary_nuclear_perturbation
from .stationary_executor import (
    StationaryHVPContributor,
    StationaryPerturbationProvider,
    StationaryResponseDriver,
    StationarySecondOrderExecutor,
)


@dataclass(frozen=True, eq=False)
class _NativeRKSIntegralState(NativeRHFState):
    """Adapt one live RKS/CPKS lease to the shared integral derivative providers."""

    response: typing.Any = field(default=None, repr=False)

    @classmethod
    def from_response(
        cls, response: typing.Any, *, cache: typing.Any = None
    ) -> typing.Self:
        if not isinstance(response, NativeRKSResponse):
            raise TypeError("RKS HVP requires NativeRKSResponse")
        return cls(
            response._source,
            response.problem.reference,
            Path(cache) if cache is not None else Path(".artifacts"),
            response,
        )

    def validate(self) -> typing.Self:
        if not isinstance(self.response, NativeRKSResponse):
            raise TypeError("RKS integral state requires a NativeRKSResponse lease")
        self.response.validate_current()
        state = self.response.state
        StationaryDerivativeContract(state.identity).validate(state)
        if (
            self.source is not self.response._source
            or self.reference is not self.response.problem.reference
        ):
            raise ValueError("RKS integral state/source/reference identity mismatch")
        if state._source.backend != "cpu":
            raise NotImplementedError("first complete RKS HVP slice is CPU-only")
        if state.identity.method not in ("lda-rks", "pbe-rks"):
            raise ValueError("RKS HVP requires native all-electron LDA/PBE RKS")
        if state._source.hamiltonian != "all-electron":
            raise ValueError("RKS HVP requires an all-electron Hamiltonian")
        if state._source.coefficients != (1.0, 1.0, 0.0):
            raise ValueError("RKS HVP requires unscaled semilocal LDA/PBE")
        if (
            self.source.representation != "cartesian"
            or self.source.multiplicity != 1
            or self.source.electron_count % 2
            or self.source.auxiliary_shells
        ):
            raise ValueError(
                "RKS HVP requires direct Cartesian closed-shell all-electron state"
            )
        if not 1 <= self.source.nbf <= 12 or not 1 <= len(self.source.atoms) <= 4:
            raise ValueError("RKS HVP is bounded to 12 AOs and four atoms")
        if any(shell.angular_momentum > 3 for shell in self.source.shells):
            raise ValueError("RKS HVP derivative providers support s/p/d/f shells")
        if state._source.grid_spec is None or state._source.atomic_weights is None:
            raise ValueError("RKS HVP requires retained native grid prescription")
        if self.reference.algorithm != "KS":
            raise ValueError("RKS HVP requires a KS reference snapshot")
        if self.reference.nmo != self.source.nbf:
            raise ValueError("RKS HVP source/reference AO dimension mismatch")
        gap = (
            self.reference.orbital_energies[self.reference.nocc]
            - self.reference.orbital_energies[self.reference.nocc - 1]
        )
        if gap <= 1e-8:
            raise ValueError("RKS HVP requires a nonzero occupied/virtual gap")
        density = np.asarray(state.density[0])
        weighted = np.asarray(state.weighted_density[0])
        if not np.allclose(self.P0, density, atol=2e-11, rtol=2e-11):
            raise ValueError("RKS HVP density/reference mismatch")
        expected_weighted = (
            self.C[:, : self.nocc]
            * (2.0 * self.eps[: self.nocc])[None, :]
        ) @ self.C[:, : self.nocc].T
        if not np.allclose(expected_weighted, weighted, atol=2e-11, rtol=2e-11):
            raise ValueError("RKS HVP weighted-density/reference mismatch")
        return self


@dataclass(frozen=True)
class _RKSPerturbation:
    direction: np.ndarray
    frozen_fock: np.ndarray
    overlap: np.ndarray
    diagnostics: typing.Mapping[str, typing.Any]


@dataclass(frozen=True)
class _RKSResolvedDirection:
    stationary_response: typing.Any
    components: typing.Mapping[str, np.ndarray]
    diagnostics: typing.Mapping[str, typing.Any]


@dataclass(frozen=True, eq=False)
class RKSHVPResult:
    """Detached complete bounded semilocal RKS HVP."""

    direction: np.ndarray
    value: np.ndarray
    components: typing.Mapping[str, np.ndarray]
    response: typing.Any = field(repr=False)
    plan_identity: str
    identity: str
    _diagnostics: typing.Mapping[str, typing.Any] = field(repr=False)

    @property
    def diagnostics(self) -> dict[str, typing.Any]:
        return deepcopy(dict(self._diagnostics))


def _tile_range(count: int, tile_points: int) -> typing.Iterator[tuple[int, int]]:
    for begin in range(0, count, tile_points):
        yield begin, min(begin + tile_points, count)


def _grid_sources(state: _NativeRKSIntegralState) -> tuple[typing.Any, ...]:
    response = state.response
    ks = response.state
    basis = response.xc_kernel.basis
    grid = ks.grid
    spec = ks._source.grid_spec
    atomic_weights = np.asarray(ks._source.atomic_weights, dtype=np.float64)
    if len(atomic_weights) != len(grid.points):
        raise ValueError("RKS HVP atomic/grid weight length mismatch")
    centers = state.coords
    ao_atoms = _native_ao_atoms(basis)
    return ks, basis, grid, spec, atomic_weights, centers, ao_atoms


def _xc_frozen_fock_direction(
    state: _NativeRKSIntegralState,
    direction: np.ndarray,
    *,
    tile_points: int,
) -> tuple[np.ndarray, dict[str, typing.Any]]:
    """Differentiate the explicit semilocal XC AO potential at fixed D."""
    ks, basis, grid, spec, atomic_weights, centers, ao_atoms = _grid_sources(state)
    program = ContractionProgram(state.response.xc_kernel.spec, "geometry")
    density = np.asarray(ks.density[0])
    result = np.zeros((state.nbf, state.nbf), dtype=np.float64)
    branches: list[str] = []
    for begin, end in _tile_range(len(grid.points), tile_points):
        points = np.asarray(grid.points[begin:end])
        owners = np.asarray(grid.owners[begin:end], dtype=np.int64)
        point_motion = direction[owners]
        partition = partition_response(
            points,
            centers,
            point_motion=point_motion,
            center_motion=direction,
            iterations=spec.partition_iterations,
            coincident_tolerance=spec.coincident_tolerance,
        )
        selected = (np.arange(end - begin), owners)
        weight_motion = atomic_weights[begin:end] * partition.directional[selected]
        jets = basis.evaluate(
            points, program.contract.ingredients.ao_order + 1
        )
        matrices = program.potential_geometry_directional(
            jets,
            density,
            np.asarray(grid.weights[begin:end]),
            ao_atoms=ao_atoms,
            center_motion=direction,
            point_motion=point_motion,
            weight_motion=weight_motion,
        )
        if np.asarray(matrices).shape != (1, state.nbf, state.nbf):
            raise ValueError("unpolarized XC geometry JVP returned invalid AO layout")
        result += np.asarray(matrices[0])
        branches.append(partition.branch_identity)
    if not np.isfinite(result).all():
        raise FloatingPointError("nonfinite RKS frozen XC Fock direction")
    return result, {
        "tiles": len(branches),
        "partition_branch_identities": tuple(branches),
    }


def _build_perturbation(
    state: _NativeRKSIntegralState,
    direction: np.ndarray,
    *,
    tile_points: int,
) -> _RKSPerturbation:
    started = time.perf_counter()
    integral_started = time.perf_counter()
    frozen, overlap = generated_coulomb_directional_first_order(state, direction)
    integral_seconds = time.perf_counter() - integral_started
    xc_started = time.perf_counter()
    xc_frozen, xc_diag = _xc_frozen_fock_direction(
        state, direction, tile_points=tile_points
    )
    xc_seconds = time.perf_counter() - xc_started
    frozen = np.asarray(frozen) + xc_frozen
    if not np.isfinite(frozen).all() or not np.isfinite(overlap).all():
        raise FloatingPointError("nonfinite RKS nuclear perturbation")
    return _RKSPerturbation(
        immutable(direction),
        immutable(frozen),
        immutable(overlap),
        MappingProxyType(
            {
                "integral_first_seconds": integral_seconds,
                "xc_frozen_fock_seconds": xc_seconds,
                "complete_perturbation_seconds": time.perf_counter() - started,
                "xc": xc_diag,
            }
        ),
    )


def _second_integral_components(
    state: _NativeRKSIntegralState,
    direction: np.ndarray,
    *,
    budget_bytes: int,
) -> tuple[dict[str, np.ndarray], dict[str, typing.Any]]:
    data = _provider_data(state, backend="cpu", budget_bytes=budget_bytes)
    density = state.P0
    one_electron = _run_one_electron(
        data, "kinetic", density, direction=direction
    ) + _run_one_electron(
        data, "nuclear_attraction", density, direction=direction
    )
    overlap = -_run_one_electron(
        data, "overlap", data["W_e"], direction=direction
    )
    coulomb = _run_eri(
        data,
        density,
        direction=direction,
        exchange_energy_coefficient=0.0,
    )
    return (
        {
            "one_electron": one_electron,
            "coulomb": coulomb,
            "overlap_pulay": overlap,
        },
        _second_provider_diagnostics(data),
    )


def _xc_mixed_components(
    state: _NativeRKSIntegralState,
    direction: np.ndarray,
    density_response: np.ndarray,
    *,
    tile_points: int,
) -> tuple[dict[str, np.ndarray], dict[str, typing.Any]]:
    """Differentiate each plan-owned XC gradient source along one right direction."""
    ks, basis, grid, spec, atomic_weights, centers, ao_atoms = _grid_sources(state)
    program = ContractionProgram(state.response.xc_kernel.spec, "geometry")
    density = np.asarray(ks.density[0])
    delta_density = np.asarray(density_response)
    components = {
        name: np.zeros((state.nat, 3), dtype=np.float64)
        for name in ("xc_ao", "xc_grid", "xc_weight")
    }
    branch_pairs: list[tuple[str, str]] = []
    zero_centers = np.zeros_like(centers)

    for begin, end in _tile_range(len(grid.points), tile_points):
        points = np.asarray(grid.points[begin:end])
        weights = np.asarray(grid.weights[begin:end])
        owners = np.asarray(grid.owners[begin:end], dtype=np.int64)
        atom_weights = atomic_weights[begin:end]
        right_points = direction[owners]
        right_partition = partition_response(
            points,
            centers,
            point_motion=right_points,
            center_motion=direction,
            iterations=spec.partition_iterations,
            coincident_tolerance=spec.coincident_tolerance,
        )
        selected = (np.arange(end - begin), owners)
        right_weights = atom_weights * right_partition.directional[selected]
        jets = basis.evaluate(
            points, program.contract.ingredients.ao_order + 2
        )
        zero_points = np.zeros_like(points)
        zero_weights = np.zeros(end - begin)

        for atom in range(state.nat):
            for axis in range(3):
                left = np.zeros_like(centers)
                left[atom, axis] = 1.0
                left_points = left[owners]

                ao_value = program.mixed_geometry_directional(
                    jets,
                    density,
                    weights,
                    ao_atoms=ao_atoms,
                    left_centers=left,
                    left_points=zero_points,
                    left_weights=zero_weights,
                    right_centers=direction,
                    right_points=right_points,
                    right_weights=right_weights,
                    mixed_weights=zero_weights,
                    delta_density=delta_density,
                )
                components["xc_ao"][atom, axis] += ao_value.total

                grid_value = program.mixed_geometry_directional(
                    jets,
                    density,
                    weights,
                    ao_atoms=ao_atoms,
                    left_centers=zero_centers,
                    left_points=left_points,
                    left_weights=zero_weights,
                    right_centers=direction,
                    right_points=right_points,
                    right_weights=right_weights,
                    mixed_weights=zero_weights,
                    delta_density=delta_density,
                )
                components["xc_grid"][atom, axis] += grid_value.total

                mixed_partition = partition_mixed_response(
                    points,
                    centers,
                    left_point_motion=left_points,
                    left_center_motion=left,
                    right_point_motion=right_points,
                    right_center_motion=direction,
                    iterations=spec.partition_iterations,
                    coincident_tolerance=spec.coincident_tolerance,
                )
                left_weights = atom_weights * mixed_partition.left[selected]
                mixed_weights = atom_weights * mixed_partition.mixed[selected]
                weight_value = program.mixed_geometry_directional(
                    jets,
                    density,
                    weights,
                    ao_atoms=ao_atoms,
                    left_centers=zero_centers,
                    left_points=zero_points,
                    left_weights=left_weights,
                    right_centers=direction,
                    right_points=right_points,
                    right_weights=right_weights,
                    mixed_weights=mixed_weights,
                    delta_density=delta_density,
                )
                components["xc_weight"][atom, axis] += weight_value.total
                branch_pairs.append(
                    (
                        right_partition.branch_identity,
                        mixed_partition.branch_identity,
                    )
                )

    if not all(np.isfinite(value).all() for value in components.values()):
        raise FloatingPointError("nonfinite RKS XC mixed HVP contribution")
    return components, {
        "tiles": (len(grid.points) + tile_points - 1) // tile_points,
        "coordinate_directions": 3 * state.nat,
        "mixed_contractions_per_coordinate": 3,
        "partition_branch_pairs": tuple(branch_pairs),
        "dense_molecular_hessian_allocated": False,
    }


def _resolve_direction(
    state: _NativeRKSIntegralState,
    perturbation: _RKSPerturbation,
    *,
    tile_points: int,
    second_budget_bytes: int,
) -> _RKSResolvedDirection:
    started = time.perf_counter()
    response_started = time.perf_counter()
    stationary = solve_stationary_nuclear_perturbation(
        state.response,
        perturbation.frozen_fock,
        perturbation.overlap,
    )
    response_seconds = time.perf_counter() - response_started

    second_started = time.perf_counter()
    second, second_diag = _second_integral_components(
        state,
        perturbation.direction,
        budget_bytes=second_budget_bytes,
    )
    second_seconds = time.perf_counter() - second_started

    relaxation_started = time.perf_counter()
    relaxation = generated_coulomb_relaxation_components(
        state,
        stationary.density_derivative,
        stationary.energy_weighted_density_derivative,
    )
    relaxation_seconds = time.perf_counter() - relaxation_started

    xc_started = time.perf_counter()
    xc, xc_diag = _xc_mixed_components(
        state,
        perturbation.direction,
        stationary.density_derivative,
        tile_points=tile_points,
    )
    xc_seconds = time.perf_counter() - xc_started

    nuclear_started = time.perf_counter()
    nuclear = nuclear_hvp(state, perturbation.direction)
    nuclear_seconds = time.perf_counter() - nuclear_started

    components = {
        "one_electron": second["one_electron"] + relaxation["one_electron"],
        "coulomb": second["coulomb"] + relaxation["coulomb"],
        **xc,
        "overlap_pulay": second["overlap_pulay"] + relaxation["overlap_pulay"],
        "nuclear": nuclear,
    }
    if tuple(components) != state_plan(state).source_names:
        raise RuntimeError("RKS HVP source assembly diverged from StationaryHVPPlan")
    return _RKSResolvedDirection(
        stationary,
        MappingProxyType(
            {name: immutable(value) for name, value in components.items()}
        ),
        MappingProxyType(
            {
                "perturbation": dict(perturbation.diagnostics),
                "second_integrals": second_diag,
                "xc": xc_diag,
                "timings_seconds": {
                    "cpks_response": response_seconds,
                    "second_integral_hvp": second_seconds,
                    "first_integral_relaxation": relaxation_seconds,
                    "xc_mixed_hvp": xc_seconds,
                    "nuclear_hvp": nuclear_seconds,
                    "complete_resolution": time.perf_counter() - started,
                },
            }
        ),
    )


def state_plan(state: _NativeRKSIntegralState) -> StationaryHVPPlan:
    """Resolve the exact live MethodIR into the common semilocal HVP plan."""
    state.validate()
    return StationaryHVPPlan(
        state.response.state._source.method_ir,
        StationaryMeanField(SCF_POINT_MODEL),
    )


def rks_hvp(
    response: NativeRKSResponse,
    direction: typing.Any,
    *,
    cache: typing.Any = None,
    tile_points: int = 128,
    second_budget_bytes: int = 64 << 20,
) -> RKSHVPResult:
    """Apply the complete bounded analytic LDA/PBE RKS Hessian to one direction.

    One shared CPKS solve supplies D1(v) and W1(v). Generated #178 providers
    supply fixed-density one-electron/Coulomb/overlap second derivatives; their
    generated first derivatives supply the response-weight contractions.
    #958/#964 supply the explicit XC AO/grid/partition mixed terms. No dense
    molecular Hessian or finite-difference quantity enters the result.
    """
    if type(tile_points) is not int or not 1 <= tile_points <= 4096:
        raise ValueError("tile_points must be an integer in [1,4096]")
    if (
        type(second_budget_bytes) is not int
        or not 1 <= second_budget_bytes < 2**63
    ):
        raise ValueError("second_budget_bytes must be a positive int64")
    state = _NativeRKSIntegralState.from_response(response, cache=cache)
    vector = checked_direction(direction, state.nat)
    plan = state_plan(state)
    state_identity = canonical_hash(state.response.state.identity.to_payload())

    perturbation_provider = StationaryPerturbationProvider(
        canonical_hash(
            {
                "schema": "vibeqc.rks-hvp-perturbation/v1",
                "state": state_identity,
                "plan": plan.identity,
                "functional": response.xc_kernel.spec.identity,
            }
        ),
        lambda value: _build_perturbation(
            state, value, tile_points=tile_points
        ),
    )
    response_driver = StationaryResponseDriver(
        canonical_hash(
            {
                "schema": "vibeqc.rks-hvp-response/v1",
                "state": state_identity,
                "operator": response.problem.operator_identity,
            }
        ),
        lambda prepared: _resolve_direction(
            state,
            prepared,
            tile_points=tile_points,
            second_budget_bytes=second_budget_bytes,
        ),
    )
    contributors = tuple(
        StationaryHVPContributor(
            source,
            canonical_hash(
                {
                    "schema": "vibeqc.rks-hvp-contributor/v1",
                    "state": state_identity,
                    "plan": plan.identity,
                    "source": source,
                }
            ),
            lambda context, source=source: context.response.components[source],
        )
        for source in plan.source_names
    )
    executor = StationarySecondOrderExecutor(
        plan,
        natoms=state.nat,
        perturbation=perturbation_provider,
        response=response_driver,
        contributors=contributors,
    )
    raw = executor.apply(vector)
    resolved = raw.response
    state.validate()
    diagnostics = {
        **dict(raw.diagnostics),
        **dict(resolved.diagnostics),
        "schema": "vibeqc.rks-hvp/v1",
        "state_identity": state_identity,
        "method": response.state.identity.method,
        "backend": "cpu",
        "molecular_hvp": True,
        "full_molecular_hessian_allocated": False,
        "public_calculator_capability": False,
        "scope": "bounded all-electron Cartesian LDA/PBE RKS",
    }
    return RKSHVPResult(
        raw.direction,
        raw.value,
        raw.components,
        resolved.stationary_response,
        raw.plan_identity,
        raw.identity,
        MappingProxyType(diagnostics),
    )
