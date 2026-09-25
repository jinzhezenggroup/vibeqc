"""Complete bounded CPU semilocal RKS molecular Hessian-vector products.

This is the first native LDA/PBE RKS composition of the MethodIR-derived
StationaryHVPPlan, one real CPKS nuclear response, generated first/second
integral providers, analytic Becke mixed response and generated XC feature
Hessian contractions.  It remains a tools endpoint: direct all-electron
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
from vibeqc._dft_gradient import _native_ao_atoms
from vibeqc.profiles import canonical_hash
from vibeqc_compiler.method import StationaryHVPPlan, StationaryMeanField
from vibeqc_compiler.method.stationary_gradient import SCF_POINT_MODEL
from vibeqc_compiler.tensor import execute
from vibeqc_compiler.xc.contractions import ExternalPointContraction
from vibeqc_compiler.xc.grid_response import partition_mixed_response

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
    DirectionalRKSResponse,
    _validate_partition_provenance,
    directional_rks_response,
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
    """Differentiate each left XC gradient owner along one full RKS direction."""
    operator.validate_current()
    state = operator.state
    source = state._source
    if source.atomic_weights is None or source.grid_spec is None:
        raise ValueError("semilocal RKS HVP requires retained grid provenance")
    basis = operator.xc_kernel.basis
    spec = operator.xc_kernel.spec
    if spec.spin != "unpolarized" or spec.ingredients not in (
        ("rho",),
        ("rho", "sigma"),
    ):
        raise NotImplementedError("semilocal RKS HVP supports LDA/GGA only")

    contraction = ExternalPointContraction(spec, "geometry")
    order = contraction.contract.ingredients.ao_order
    grid = state.grid
    all_points = np.asarray(grid.points)
    all_owners = np.asarray(grid.owners, dtype=np.int64)
    atomic_weights = np.asarray(source.atomic_weights, dtype=np.float64)
    centers = np.asarray([atom.position for atom in basis.atoms], dtype=np.float64)
    natom = basis.natom
    if (
        all_owners.shape != (len(all_points),)
        or atomic_weights.shape != (len(all_points),)
        or np.any(all_owners < 0)
        or np.any(all_owners >= natom)
    ):
        raise ValueError("semilocal RKS HVP grid provenance is malformed")
    ao_atoms = _native_ao_atoms(basis)
    delta_density = np.asarray(response.response.density_derivative)
    result = {
        "xc_ao": np.zeros((natom, 3), dtype=np.float64),
        "xc_grid": np.zeros((natom, 3), dtype=np.float64),
        "xc_weight": np.zeros((natom, 3), dtype=np.float64),
    }
    grid_spec = source.grid_spec
    tile_points = int(getattr(operator.xc_kernel, "tile_points", 256))
    zero_centers = np.zeros((natom, 3), dtype=np.float64)

    for begin in range(0, len(all_points), tile_points):
        end = min(begin + tile_points, len(all_points))
        points = all_points[begin:end]
        owners = all_owners[begin:end]
        weights = np.asarray(grid.weights[begin:end], dtype=np.float64)
        atomic = atomic_weights[begin:end]
        jets = basis.evaluate(points, order + 2)
        right_points = direction[owners]
        zero_points = np.zeros_like(points)
        zero_weights = np.zeros(end - begin, dtype=np.float64)

        for atom in range(natom):
            for axis in range(3):
                left = np.zeros((natom, 3), dtype=np.float64)
                left[atom, axis] = 1.0
                left_points = left[owners]
                partition = partition_mixed_response(
                    points,
                    centers,
                    left_point_motion=left_points,
                    left_center_motion=left,
                    right_point_motion=right_points,
                    right_center_motion=direction,
                    iterations=grid_spec.partition_iterations,
                    coincident_tolerance=grid_spec.coincident_tolerance,
                )
                if partition.branch_identity != response.grid_branch_identity:
                    raise ValueError("RKS HVP changed the qualified Becke branch")
                selected = (np.arange(end - begin), owners)
                _validate_partition_provenance(
                    atomic, weights, partition.weights[selected]
                )
                left_weight = atomic * partition.left[selected]
                right_weight = atomic * partition.right[selected]
                mixed_weight = atomic * partition.mixed[selected]

                ao = contraction.mixed_geometry_directional(
                    jets,
                    state.density[0],
                    weights,
                    ao_atoms=ao_atoms,
                    left_centers=left,
                    left_points=zero_points,
                    left_weights=zero_weights,
                    right_centers=direction,
                    right_points=right_points,
                    right_weights=right_weight,
                    mixed_weights=zero_weights,
                    delta_density=delta_density,
                )
                point = contraction.mixed_geometry_directional(
                    jets,
                    state.density[0],
                    weights,
                    ao_atoms=ao_atoms,
                    left_centers=zero_centers,
                    left_points=left_points,
                    left_weights=zero_weights,
                    right_centers=direction,
                    right_points=right_points,
                    right_weights=right_weight,
                    mixed_weights=zero_weights,
                    delta_density=delta_density,
                )
                weight = contraction.mixed_geometry_directional(
                    jets,
                    state.density[0],
                    weights,
                    ao_atoms=ao_atoms,
                    left_centers=zero_centers,
                    left_points=zero_points,
                    left_weights=left_weight,
                    right_centers=direction,
                    right_points=right_points,
                    right_weights=right_weight,
                    mixed_weights=mixed_weight,
                    delta_density=delta_density,
                )
                result["xc_ao"][atom, axis] += ao.total
                result["xc_grid"][atom, axis] += point.total
                result["xc_weight"][atom, axis] += weight.total

    operator.validate_current()
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError("nonfinite semilocal XC HVP source")
    return {name: immutable(value) for name, value in result.items()}


def rks_hvp(
    operator: typing.Any,
    direction: typing.Any,
    *,
    cache: typing.Any = ".artifacts",
    solver_options: typing.Any = None,
) -> RKSHVPResult:
    """Apply the complete bounded direct LDA/PBE RKS molecular Hessian once.

    The MethodIR-derived plan owns source inventory and integral weights. Exactly
    one real nuclear CPKS solve supplies D'(v)/W'(v). Integral contributors use
    generated first/second derivative providers, while XC contributors use the
    generated feature Hessian plus analytic AO/grid/Becke mixed directions.
    """
    if not isinstance(operator, NativeRKSResponse):
        raise TypeError("RKS molecular HVP requires NativeRKSResponse")
    if solver_options is not None and not isinstance(solver_options, GMRESOptions):
        raise TypeError("solver_options must be GMRESOptions")
    plan = _checked_plan(operator)
    vector = checked_direction(direction, operator.xc_kernel.basis.natom)
    cache = Path(cache)
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
        StationaryHVPContributor("xc_ao", "native-rks-xc-mixed-ao-v1", xc("xc_ao")),
        StationaryHVPContributor(
            "xc_grid", "native-rks-xc-mixed-grid-v1", xc("xc_grid")
        ),
        StationaryHVPContributor(
            "xc_weight", "native-rks-xc-mixed-weight-v1", xc("xc_weight")
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
            "native-rks-shared-cpks-direction-v1",
            lambda value: directional_rks_response(
                operator,
                value,
                cache=cache,
                solver_options=solver_options,
            ),
        ),
        contributors=contributors,
    )
    executed = executor.apply(vector)
    operator.validate_current()
    directional = executed.response
    if not isinstance(directional, DirectionalRKSResponse):
        raise TypeError("RKS HVP response driver returned an invalid response")
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
            "nuclear_response_solves": 1,
            "response_iterations": directional.response.solve_result.iterations,
            "response_residual_norm": directional.response.solve_result.residual_norm,
            "integral_providers": deepcopy(provider_diagnostics),
            "xc_second_order": "generated-feature-hessian/analytic-grid-mixed",
            "full_molecular_hessian_allocated": False,
            "full_ao_rank_four_weights": False,
            "execution": "bounded-cpu-native-rks-hvp-v1",
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
