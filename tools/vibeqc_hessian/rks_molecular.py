"""Complete bounded CPU semilocal RKS molecular Hessian-vector products.

This is the first native LDA/PBE RKS composition of the MethodIR-derived
StationaryHVPPlan, one real CPKS nuclear response, generated first/second
integral providers, analytic Becke mixed response and native SCF-domain XC
Hessian contractions. It remains a tools endpoint: direct all-electron
Cartesian CPU RKS, at most 12 AOs/four atoms, with no public Calculator Hessian
capability inferred.
"""

from __future__ import annotations

import typing
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from types import MappingProxyType

import numpy as np
from vibeqc.profiles import canonical_hash
from vibeqc_compiler.method import StationaryHVPPlan, StationaryMeanField
from vibeqc_compiler.method.stationary_gradient import SCF_POINT_MODEL
from vibeqc_compiler.tensor import execute

from tools.vibeqc_posthf.reference import immutable
from tools.vibeqc_response import GMRESOptions, NativeRKSResponse

from .analytic import (
    generated_weighted_second_integral_hvp,
    nuclear_hvp_from_source,
)
from .first_order import (
    checked_direction,
    generated_weighted_first_integral_gradient,
)
from .rks_directional import (
    DirectionalRKSBatchResponse,
    DirectionalRKSResponse,
    directional_rks_response,
    directional_rks_responses,
    native_rks_xc_hvp_components,
)
from .stationary_executor import (
    StationaryHVPContributor,
    StationaryPerturbationProvider,
    StationaryResponseDriver,
    StationarySecondOrderExecutor,
)


@dataclass(frozen=True, eq=False)
class RKSHVPResult:
    """Detached complete semilocal RKS HVP and its plan-owned source split."""

    direction: np.ndarray
    value: np.ndarray
    components: typing.Mapping[str, np.ndarray]
    directional_response: DirectionalRKSResponse
    plan_identity: str
    identity: str
    _diagnostics: typing.Mapping[str, typing.Any]

    @property
    def diagnostics(self) -> dict[str, typing.Any]:
        return deepcopy(dict(self._diagnostics))


@dataclass(frozen=True, eq=False)
class RKSHVPBatchResult:
    """Detached complete HVPs assembled from one shared RKS multi-RHS solve."""

    directions: np.ndarray
    values: np.ndarray
    results: tuple[RKSHVPResult, ...]
    directional_responses: DirectionalRKSBatchResponse
    plan_identity: str
    identity: str
    _diagnostics: typing.Mapping[str, typing.Any]

    @property
    def diagnostics(self) -> dict[str, typing.Any]:
        return deepcopy(dict(self._diagnostics))


def _checked_plan(operator: NativeRKSResponse) -> StationaryHVPPlan:
    operator.validate_current()
    state = operator.state
    if state._source.backend != "cpu":
        raise NotImplementedError("semilocal RKS molecular HVP is CPU-only")
    if (
        operator._source.representation != "cartesian"
        or operator._source.auxiliary_shells
        or not 1 <= operator._source.nbf <= 12
        or not 1 <= len(operator._source.atoms) <= 4
    ):
        raise ValueError(
            "semilocal RKS molecular HVP requires direct Cartesian all-electron "
            "sources bounded to 12 AOs and four atoms"
        )
    plan = StationaryHVPPlan(
        state._source.method_ir,
        StationaryMeanField(SCF_POINT_MODEL),
    )
    expected = (
        "one_electron",
        "coulomb",
        "xc_ao",
        "xc_grid",
        "xc_weight",
        "overlap_pulay",
        "nuclear",
    )
    if plan.source_names != expected:
        raise ValueError(
            "semilocal RKS HVP source inventory is not the qualified slice"
        )
    return plan


def _pair_plan_weights(
    plan: StationaryHVPPlan,
    source_name: str,
    response: DirectionalRKSResponse,
    operator: NativeRKSResponse,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate fixed and response weights for one ordered AO-pair source."""
    nbf = operator._source.nbf
    rows, cols = np.indices((nbf, nbf))
    rows, cols = rows.ravel(), cols.ravel()
    state = operator.state
    if source_name == "one_electron":
        base = np.asarray(state.density[0])
        delta = np.asarray(response.response.density_derivative)
        feeds = {
            "density_left": base[None, rows, cols],
            "d_density_left": delta[None, rows, cols],
        }
    elif source_name == "overlap_pulay":
        base = np.asarray(state.weighted_density[0])
        delta = np.asarray(response.response.energy_weighted_density_derivative)
        feeds = {
            "weighted_density": base[None, rows, cols],
            "d_weighted_density": delta[None, rows, cols],
        }
    else:
        raise ValueError("pair weight generation requires one-electron or overlap")
    block = plan.integral_block(source_name, terms=nbf * nbf, coordinates=1)
    fixed = execute(block.weights, feeds).outputs["weights"].reshape(nbf, nbf)
    moving = (
        execute(block.response_weights, feeds)
        .outputs["response_weights"]
        .reshape(nbf, nbf)
    )
    return immutable(fixed), immutable(moving)


def _coulomb_shell_plan_weights(
    plan: StationaryHVPPlan,
    response: DirectionalRKSResponse,
    operator: NativeRKSResponse,
    slots: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one ordered shell-quartet Coulomb weight and its response JVP."""
    offsets = np.cumsum((0, *operator._source.shell_sizes))
    ranges = [range(offsets[s], offsets[s + 1]) for s in slots]
    tuples = tuple(product(*ranges))
    if not tuples:
        raise ValueError("empty Coulomb shell block")
    ids = np.asarray(tuples, dtype=np.int64)
    state = operator.state
    density = np.asarray(state.density[0])
    delta = np.asarray(response.response.density_derivative)
    feeds = {
        "density_left": density[None, ids[:, 0], ids[:, 1]],
        "density_right": density[None, ids[:, 2], ids[:, 3]],
        "d_density_left": delta[None, ids[:, 0], ids[:, 1]],
        "d_density_right": delta[None, ids[:, 2], ids[:, 3]],
    }
    block = plan.integral_block("coulomb", terms=len(tuples), coordinates=1)
    shape = tuple(offsets[s + 1] - offsets[s] for s in slots)
    fixed = execute(block.weights, feeds).outputs["weights"].reshape(shape)
    moving = (
        execute(block.response_weights, feeds)
        .outputs["response_weights"]
        .reshape(shape)
    )
    return immutable(fixed), immutable(moving)


def _integral_source_hvp(
    source_name: str,
    plan: StationaryHVPPlan,
    operator: NativeRKSResponse,
    response: DirectionalRKSResponse,
    direction: np.ndarray,
    cache: Path,
) -> tuple[np.ndarray, dict[str, typing.Any]]:
    """Apply one plan-owned integral source as d(weight)dI + weight d2I(v)."""
    if source_name in ("one_electron", "overlap_pulay"):
        fixed, moving = _pair_plan_weights(plan, source_name, response, operator)
        first = generated_weighted_first_integral_gradient(
            operator._source,
            source_name,
            pair_weights=moving,
            cache=cache,
        )
        second, diagnostic = generated_weighted_second_integral_hvp(
            operator._source,
            source_name,
            direction,
            pair_weights=fixed,
            cache=cache,
        )
    elif source_name == "coulomb":

        def response_weights(slots: tuple[int, int, int, int]) -> np.ndarray:
            return _coulomb_shell_plan_weights(plan, response, operator, slots)[1]

        def fixed_weights(slots: tuple[int, int, int, int]) -> np.ndarray:
            return _coulomb_shell_plan_weights(plan, response, operator, slots)[0]

        first = generated_weighted_first_integral_gradient(
            operator._source,
            source_name,
            eri_shell_weights=response_weights,
            cache=cache,
        )
        second, diagnostic = generated_weighted_second_integral_hvp(
            operator._source,
            source_name,
            direction,
            eri_shell_weights=fixed_weights,
            cache=cache,
        )
    else:
        raise ValueError("unknown semilocal RKS integral HVP source")
    total = np.asarray(first) + np.asarray(second)
    if not np.isfinite(total).all():
        raise FloatingPointError("nonfinite semilocal RKS integral HVP source")
    diagnostic = {
        **diagnostic,
        "response_first_integral": "generated-plan-weighted",
        "stationary_hvp_plan": plan.identity,
    }
    return immutable(total), diagnostic


def _xc_hvp_components(
    operator: NativeRKSResponse,
    response: DirectionalRKSResponse,
    direction: np.ndarray,
) -> dict[str, np.ndarray]:
    """Reuse the native SCF point model and the already solved CPKS direction."""
    if not np.array_equal(response.direction, direction):
        raise ValueError("XC HVP direction does not match the solved response")
    sources = native_rks_xc_hvp_components(operator, response)
    return {name: getattr(sources, name) for name in ("xc_ao", "xc_grid", "xc_weight")}


def _rks_hvp_with_response(
    operator: NativeRKSResponse,
    vector: np.ndarray,
    directional: DirectionalRKSResponse,
    *,
    plan: StationaryHVPPlan,
    cache: Path,
    response_driver_identity: str,
    nuclear_response_solves: int,
    execution: str,
) -> RKSHVPResult:
    """Assemble one complete HVP from an already solved directional response."""
    if not np.array_equal(directional.direction, vector):
        raise ValueError("RKS HVP direction does not match the supplied response")
    provider_diagnostics: dict[str, typing.Any] = {}
    xc_cache: dict[str, np.ndarray] = {}

    def integral(source_name: str) -> typing.Callable[[typing.Any], np.ndarray]:
        def evaluate(context: typing.Any) -> np.ndarray:
            value, diagnostic = _integral_source_hvp(
                source_name,
                plan,
                operator,
                context.response,
                context.direction,
                cache,
            )
            provider_diagnostics[source_name] = diagnostic
            return value

        return evaluate

    def xc(source_name: str) -> typing.Callable[[typing.Any], np.ndarray]:
        def evaluate(context: typing.Any) -> np.ndarray:
            if not xc_cache:
                xc_cache.update(
                    _xc_hvp_components(operator, context.response, context.direction)
                )
            return xc_cache[source_name]

        return evaluate

    def reuse_response(value: typing.Any) -> DirectionalRKSResponse:
        candidate = checked_direction(value, operator.xc_kernel.basis.natom)
        if not np.array_equal(candidate, directional.direction):
            raise ValueError("stationary executor requested a different RKS direction")
        return directional

    contributors = (
        StationaryHVPContributor(
            "one_electron",
            "native-rks-plan-weighted-one-electron-v1",
            integral("one_electron"),
        ),
        StationaryHVPContributor(
            "coulomb",
            "native-rks-plan-weighted-coulomb-v1",
            integral("coulomb"),
        ),
        StationaryHVPContributor(
            "xc_ao", "native-rks-xc-scf-domain-ao-v2", xc("xc_ao")
        ),
        StationaryHVPContributor(
            "xc_grid", "native-rks-xc-scf-domain-grid-v2", xc("xc_grid")
        ),
        StationaryHVPContributor(
            "xc_weight", "native-rks-xc-scf-domain-weight-v2", xc("xc_weight")
        ),
        StationaryHVPContributor(
            "overlap_pulay",
            "native-rks-plan-weighted-overlap-v1",
            integral("overlap_pulay"),
        ),
        StationaryHVPContributor(
            "nuclear",
            "native-rks-nuclear-hvp-v1",
            lambda context: nuclear_hvp_from_source(
                operator._source, context.direction
            ),
        ),
    )
    executor = StationarySecondOrderExecutor(
        plan,
        natoms=operator.xc_kernel.basis.natom,
        perturbation=StationaryPerturbationProvider(
            "native-rks-cartesian-direction-v1",
            lambda value: immutable(np.asarray(value, dtype=np.float64)),
        ),
        response=StationaryResponseDriver(
            response_driver_identity,
            reuse_response,
        ),
        contributors=contributors,
    )
    executed = executor.apply(vector)
    operator.validate_current()
    if executed.response is not directional:
        raise RuntimeError("RKS HVP executor did not reuse the supplied response")
    identity = canonical_hash(
        {
            "schema": "vibeqc.rks-hvp/v1",
            "state": operator.state.identity.to_payload(),
            "plan": plan.identity,
            "executor_result": executed.identity,
            "directional_response": directional.identity,
        }
    )
    diagnostics = MappingProxyType(
        {
            **dict(executed.diagnostics),
            "molecular_hvp": True,
            "method": operator.state.identity.method,
            "nuclear_response_solves": nuclear_response_solves,
            "response_iterations": directional.response.solve_result.iterations,
            "response_residual_norm": directional.response.solve_result.residual_norm,
            "integral_providers": deepcopy(provider_diagnostics),
            "xc_second_order": "native-scf-point-response/analytic-grid-mixed",
            "full_molecular_hessian_allocated": False,
            "full_ao_rank_four_weights": False,
            "execution": execution,
        }
    )
    return RKSHVPResult(
        direction=executed.direction,
        value=executed.value,
        components=MappingProxyType(dict(executed.components)),
        directional_response=directional,
        plan_identity=plan.identity,
        identity=identity,
        _diagnostics=diagnostics,
    )


def rks_hvp(
    operator: typing.Any,
    direction: typing.Any,
    *,
    cache: typing.Any = ".artifacts",
    solver_options: typing.Any = None,
) -> RKSHVPResult:
    """Apply the complete bounded direct LDA/PBE RKS molecular Hessian once."""
    if not isinstance(operator, NativeRKSResponse):
        raise TypeError("RKS molecular HVP requires NativeRKSResponse")
    if solver_options is not None and not isinstance(solver_options, GMRESOptions):
        raise TypeError("solver_options must be GMRESOptions")
    plan = _checked_plan(operator)
    vector = checked_direction(direction, operator.xc_kernel.basis.natom)
    cache_path = Path(cache)
    directional = directional_rks_response(
        operator,
        vector,
        cache=cache_path,
        solver_options=solver_options,
    )
    return _rks_hvp_with_response(
        operator,
        vector,
        directional,
        plan=plan,
        cache=cache_path,
        response_driver_identity="native-rks-shared-cpks-direction-v1",
        nuclear_response_solves=1,
        execution="bounded-cpu-native-rks-hvp-v2",
    )


def rks_hvp_many(
    operator: typing.Any,
    directions: typing.Any,
    *,
    cache: typing.Any = ".artifacts",
    strategy: str = "recycled",
    solver_options: typing.Any = None,
) -> RKSHVPBatchResult:
    """Apply complete RKS HVPs after one shared sequential/blocked/recycled solve."""
    if not isinstance(operator, NativeRKSResponse):
        raise TypeError("RKS molecular HVP block requires NativeRKSResponse")
    if strategy not in ("sequential", "blocked", "recycled"):
        raise ValueError("strategy must be sequential, blocked or recycled")
    if solver_options is not None and not isinstance(solver_options, GMRESOptions):
        raise TypeError("solver_options must be GMRESOptions")
    plan = _checked_plan(operator)
    natom = operator.xc_kernel.basis.natom
    raw = np.asarray(directions)
    if (
        raw.ndim != 3
        or raw.shape[0] < 1
        or raw.shape[1:] != (natom, 3)
        or raw.dtype.kind not in "iuf"
        or np.iscomplexobj(raw)
        or not np.isfinite(raw).all()
    ):
        raise ValueError(
            "RKS HVP block directions must be finite real with shape (nrhs, natoms, 3)"
        )
    vectors = tuple(checked_direction(item, natom) for item in raw)
    cache_path = Path(cache)
    directional = directional_rks_responses(
        operator,
        np.stack(vectors),
        cache=cache_path,
        strategy=strategy,
        solver_options=solver_options,
    )
    results = tuple(
        _rks_hvp_with_response(
            operator,
            vector,
            response,
            plan=plan,
            cache=cache_path,
            response_driver_identity="native-rks-shared-cpks-multi-rhs-v1",
            nuclear_response_solves=0,
            execution="bounded-cpu-native-rks-hvp-multi-rhs-v1",
        )
        for vector, response in zip(vectors, directional.responses, strict=True)
    )
    values = immutable(np.stack([item.value for item in results]))
    published_directions = immutable(np.stack(vectors))
    identity = canonical_hash(
        {
            "schema": "vibeqc.rks-hvp-batch/v1",
            "state": operator.state.identity.to_payload(),
            "plan": plan.identity,
            "strategy": strategy,
            "directional_response_batch": directional.identity,
            "results": tuple(item.identity for item in results),
        }
    )
    diagnostics = MappingProxyType(
        {
            "molecular_hvp": True,
            "nrhs": len(results),
            "strategy": strategy,
            "multi_rhs_calls": 1,
            "response_operator_actions": directional.solve_result.operator_actions,
            "response_peak_workspace_bytes": directional.solve_result.peak_workspace_bytes,
            "rhs_rank": directional.solve_result.rhs_rank,
            "rank_deficient_rhs": directional.solve_result.rank_deficient_rhs,
            "full_molecular_hessian_allocated": False,
            "full_ao_rank_four_weights": False,
            "execution": "bounded-cpu-native-rks-hvp-multi-rhs-v1",
        }
    )
    return RKSHVPBatchResult(
        directions=published_directions,
        values=values,
        results=results,
        directional_responses=directional,
        plan_identity=plan.identity,
        identity=identity,
        _diagnostics=diagnostics,
    )
