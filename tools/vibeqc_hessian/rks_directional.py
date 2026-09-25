"""Real semilocal RKS nuclear directions for the matrix-free DFT Hessian line.

This module composes existing owners only: generated one-/two-electron geometry
derivatives, the native SCF-domain XC point model, analytic AO/grid JVPs, and the
shared CPKS nuclear perturbation solve. It also exposes the three semilocal XC
HVP source actions while leaving complete molecular HVP assembly to the generic
stationary second-order executor.
"""

import typing
from dataclasses import dataclass

import numpy as np
from vibeqc._dft_gradient import _native_ao_atoms
from vibeqc.profiles import canonical_hash
from vibeqc_compiler.xc.contractions import ExternalPointContraction
from vibeqc_compiler.xc.grid_response import (
    partition_mixed_response,
    partition_response,
)
from vibeqc_compiler.xc.potential import assemble_coefficients_directional

from tools.vibeqc_posthf.reference import immutable
from tools.vibeqc_response import GMRESOptions, NativeRKSResponse

from .first_order import (
    checked_direction,
    generated_directional_semilocal_rks_integral_first_order,
)
from .perturbation import (
    StationaryNuclearResponse,
    solve_stationary_nuclear_perturbation,
)


@dataclass(frozen=True, eq=False)
class DirectionalRKSResponse:
    """One complete first-order semilocal RKS nuclear response direction."""

    identity: str
    direction: np.ndarray
    integral_frozen_fock_derivative: np.ndarray
    xc_frozen_fock_derivative: np.ndarray
    frozen_fock_derivative: np.ndarray
    overlap_derivative: np.ndarray
    response: StationaryNuclearResponse
    grid_branch_identity: str

    @property
    def diagnostics(self) -> dict[str, typing.Any]:
        return {
            "molecular_hvp": False,
            "nuclear_response_solves": 1,
            "xc_geometry_point_model": "native-scf-domain",
            "xc_geometry_execution": "cpu",
            "grid_response": "analytic-becke-jvp",
            "response_operator": "shared-native-rks-cpks",
            "response_iterations": self.response.solve_result.iterations,
            "response_residual_norm": self.response.solve_result.residual_norm,
        }


@dataclass(frozen=True, eq=False)
class RKSXCHVPComponents:
    """Native semilocal XC contribution to one RKS molecular HVP direction."""

    identity: str
    xc_ao: np.ndarray
    xc_grid: np.ndarray
    xc_weight: np.ndarray
    grid_branch_identity: str

    @property
    def total(self) -> np.ndarray:
        value = (
            np.asarray(self.xc_ao)
            + np.asarray(self.xc_grid)
            + np.asarray(self.xc_weight)
        )
        return immutable(value)

    @property
    def diagnostics(self) -> dict[str, typing.Any]:
        return {
            "complete_molecular_hvp": False,
            "source_names": ("xc_ao", "xc_grid", "xc_weight"),
            "xc_point_model": "native-scf-domain",
            "xc_execution": "cpu",
            "grid_response": "analytic-becke-mixed-jvp",
            "response_reused": True,
            "additional_response_solves": 0,
        }


def _validate_partition_provenance(
    atomic_weights: typing.Any, weights: typing.Any, fractions: typing.Any
) -> None:
    """Compare dimensionless ownership, not radially amplified raw measures.

    Native hypot and generated scaled norms can differ by a few ULPs. A remote
    point's large atomic measure must not amplify that partition roundoff into
    a false provenance failure. Actual AO integration still uses native weights.
    """
    raw, actual, expected = map(np.asarray, (atomic_weights, weights, fractions))
    if (
        raw.ndim != 1
        or actual.shape != raw.shape
        or expected.shape != raw.shape
        or any(value.dtype.kind not in "iuf" for value in (raw, actual, expected))
        or any(not np.isfinite(value).all() for value in (raw, actual, expected))
        or np.any(raw < 0)
        or np.any(actual < 0)
        or np.any(expected < 0)
        or np.any(expected > 1)
    ):
        raise ValueError("invalid native grid partition measures")
    live = raw > 0
    if np.any(actual[~live] != 0):
        raise ValueError("zero atomic measure has nonzero grid weight")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        observed = actual[live] / raw[live]
    if not np.isfinite(observed).all() or not np.allclose(
        expected[live], observed, atol=2e-14, rtol=2e-13
    ):
        raise ValueError("native grid partition provenance does not reproduce weights")


def _native_rks_xc_geometry_direction(
    operator: NativeRKSResponse, direction: np.ndarray
) -> tuple[np.ndarray, str]:
    """Differentiate the exact native LDA/PBE XC Fock at fixed AO density."""
    operator.validate_current()
    state = operator.state
    source = state._source
    if source.backend != "cpu":
        raise NotImplementedError(
            "native RKS XC geometry direction is qualified on CPU only"
        )
    if source.atomic_weights is None or source.grid_spec is None:
        raise ValueError("native RKS XC geometry requires retained grid provenance")

    basis = operator.xc_kernel.basis
    grid = state.grid
    spec = operator.xc_kernel.spec
    if spec.spin != "unpolarized" or spec.ingredients not in (
        ("rho",),
        ("rho", "sigma"),
    ):
        raise NotImplementedError(
            "native RKS XC geometry direction supports semilocal LDA/GGA only"
        )

    points = np.asarray(grid.points)
    owners = np.asarray(grid.owners, dtype=np.int64)
    centers = np.asarray([atom.position for atom in basis.atoms], dtype=np.float64)
    if (
        owners.shape != (len(points),)
        or np.any(owners < 0)
        or np.any(owners >= len(centers))
    ):
        raise ValueError("native grid has invalid point ownership")
    point_motion = direction[owners]
    grid_spec = source.grid_spec
    partition = partition_response(
        points,
        centers,
        point_motion=point_motion,
        center_motion=direction,
        iterations=grid_spec.partition_iterations,
        coincident_tolerance=grid_spec.coincident_tolerance,
    )
    selected = (np.arange(len(points)), owners)
    atomic_weights = np.asarray(source.atomic_weights)
    if atomic_weights.shape != (len(points),):
        raise ValueError("native grid atomic measure has invalid shape")
    _validate_partition_provenance(
        atomic_weights, grid.weights, partition.weights[selected]
    )
    weight_motion = atomic_weights * partition.directional[selected]

    contraction = ExternalPointContraction(spec, "geometry")
    order = contraction.contract.ingredients.ao_order
    full_jets = basis.evaluate(points, order + 1)
    base_jets, directional_jets, features, feature_direction = (
        contraction.geometry_feature_direction(
            full_jets,
            state.density[0],
            ao_atoms=_native_ao_atoms(basis),
            center_motion=direction,
            point_motion=point_motion,
        )
    )

    zero_gradient = np.zeros((2, len(points), 3))
    point_values = source.evaluate_xc_points(
        spec,
        features["rho"],
        features.get("gradient", zero_gradient),
    )
    base_coefficients = {
        "rho": immutable(0.5 * (point_values["rho"][0] + point_values["rho"][1]))[None]
    }
    if "sigma" in spec.ingredients:
        base_coefficients["gradient"] = immutable(
            0.5 * (point_values["gradient"][0] + point_values["gradient"][1])
        )[None]

    directional_coefficients = source.evaluate_rks_response_points(
        "sigma" in spec.ingredients,
        features["rho"].sum(axis=0),
        features.get("gradient", zero_gradient).sum(axis=0),
        feature_direction["rho"].sum(axis=0),
        feature_direction.get("gradient", zero_gradient).sum(axis=0),
    )
    if "sigma" not in spec.ingredients:
        directional_coefficients.pop("gradient")

    value = assemble_coefficients_directional(
        base_jets,
        directional_jets,
        base_coefficients,
        directional_coefficients,
        grid.weights,
        weight_motion,
    )
    operator.validate_current()
    if value.shape != (1, basis.nao, basis.nao):
        raise ValueError("native RKS XC geometry JVP returned invalid AO shape")
    return immutable(value[0]), partition.branch_identity


def native_rks_xc_hvp_components(
    operator: typing.Any,
    directional: typing.Any,
) -> RKSXCHVPComponents:
    """Differentiate the three stationary semilocal XC gradient sources.

    The directional input must be the already-solved nuclear response for the
    same live operator. The right direction therefore reuses its CPKS density
    derivative and performs no second response solve. Each left nuclear
    coordinate is split exactly as the stationary gradient: AO-center motion,
    owner-grid-point motion, and Becke partition-weight motion.
    """
    if not isinstance(operator, NativeRKSResponse):
        raise TypeError("native RKS XC HVP requires NativeRKSResponse")
    if not isinstance(directional, DirectionalRKSResponse):
        raise TypeError("native RKS XC HVP requires DirectionalRKSResponse")
    operator.validate_current()
    state = operator.state
    source = state._source
    if source.backend != "cpu":
        raise NotImplementedError("native RKS XC HVP is qualified on CPU only")
    if source.atomic_weights is None or source.grid_spec is None:
        raise ValueError("native RKS XC HVP requires retained grid provenance")

    basis = operator.xc_kernel.basis
    grid = state.grid
    spec = operator.xc_kernel.spec
    if spec.spin != "unpolarized" or spec.ingredients not in (
        ("rho",),
        ("rho", "sigma"),
    ):
        raise NotImplementedError("native RKS XC HVP supports semilocal LDA/GGA only")
    vector = checked_direction(directional.direction, basis.natom)
    expected_directional_identity = canonical_hash(
        {
            "schema": "vibeqc.directional-rks-nuclear-response/v1",
            "state": state.identity.to_payload(),
            "response_operator": operator.problem.operator_identity,
            "direction": vector.tolist(),
            "grid_branch": directional.grid_branch_identity,
            "sources": (
                "one-electron",
                "direct-coulomb",
                "native-xc-geometry",
                "overlap-metric",
                "cpks-density-response",
            ),
        }
    )
    if directional.identity != expected_directional_identity:
        raise ValueError("directional RKS response does not belong to this operator")

    points = np.asarray(grid.points)
    owners = np.asarray(grid.owners, dtype=np.int64)
    centers = np.asarray([atom.position for atom in basis.atoms], dtype=np.float64)
    if (
        owners.shape != (len(points),)
        or np.any(owners < 0)
        or np.any(owners >= len(centers))
    ):
        raise ValueError("native grid has invalid point ownership")
    atomic_weights = np.asarray(source.atomic_weights)
    if atomic_weights.shape != (len(points),):
        raise ValueError("native grid atomic measure has invalid shape")
    selected = (np.arange(len(points)), owners)
    right_points = vector[owners]
    grid_spec = source.grid_spec
    right_partition = partition_response(
        points,
        centers,
        point_motion=right_points,
        center_motion=vector,
        iterations=grid_spec.partition_iterations,
        coincident_tolerance=grid_spec.coincident_tolerance,
    )
    _validate_partition_provenance(
        atomic_weights, grid.weights, right_partition.weights[selected]
    )
    if right_partition.branch_identity != directional.grid_branch_identity:
        raise ValueError("directional RKS response changed Becke branch")
    right_weights = atomic_weights * right_partition.directional[selected]

    contraction = ExternalPointContraction(spec, "geometry")
    order = contraction.contract.ingredients.ao_order
    full_jets = basis.evaluate(points, order + 2)
    _, _, features, right_feature = contraction.geometry_feature_direction(
        full_jets,
        state.density[0],
        ao_atoms=_native_ao_atoms(basis),
        center_motion=vector,
        point_motion=right_points,
        delta_density=directional.response.density_derivative,
    )
    zero_gradient = np.zeros((2, len(points), 3))
    point_values = source.evaluate_xc_points(
        spec,
        features["rho"],
        features.get("gradient", zero_gradient),
    )
    rho = 0.5 * (point_values["rho"][0] + point_values["rho"][1])
    gradient = (
        0.5 * (point_values["gradient"][0] + point_values["gradient"][1])
        if "sigma" in spec.ingredients
        else None
    )
    directional_coefficients = source.evaluate_rks_response_points(
        "sigma" in spec.ingredients,
        features["rho"].sum(axis=0),
        features.get("gradient", zero_gradient).sum(axis=0),
        right_feature["rho"].sum(axis=0),
        right_feature.get("gradient", zero_gradient).sum(axis=0),
    )
    drho = directional_coefficients["rho"][0]
    dgradient = (
        directional_coefficients["gradient"][0] if "sigma" in spec.ingredients else None
    )

    ao_atoms = _native_ao_atoms(basis)
    natom = basis.natom
    zeros_centers = np.zeros((natom, 3))
    zeros_points = np.zeros_like(points)
    zeros_weights = np.zeros(len(points))
    components = {
        "xc_ao": np.zeros((natom, 3)),
        "xc_grid": np.zeros((natom, 3)),
        "xc_weight": np.zeros((natom, 3)),
    }

    def contract(
        *,
        left_centers: np.ndarray,
        left_points: np.ndarray,
        left_weights: np.ndarray,
        mixed_weights: np.ndarray,
    ) -> float:
        result = contraction.mixed_geometry_from_rks_cartesian_coefficients(
            full_jets,
            state.density[0],
            grid.weights,
            ao_atoms=ao_atoms,
            left_centers=left_centers,
            left_points=left_points,
            left_weights=left_weights,
            right_centers=vector,
            right_points=right_points,
            right_weights=right_weights,
            mixed_weights=mixed_weights,
            energy=point_values["energy"],
            rho_coefficients=rho,
            directional_rho_coefficients=drho,
            gradient_coefficients=gradient,
            directional_gradient_coefficients=dgradient,
            delta_density=directional.response.density_derivative,
        )
        return result.total

    for atom in range(natom):
        for axis in range(3):
            unit = np.zeros((natom, 3))
            unit[atom, axis] = 1.0

            components["xc_ao"][atom, axis] = contract(
                left_centers=unit,
                left_points=zeros_points,
                left_weights=zeros_weights,
                mixed_weights=zeros_weights,
            )

            point_unit = unit[owners]
            components["xc_grid"][atom, axis] = contract(
                left_centers=zeros_centers,
                left_points=point_unit,
                left_weights=zeros_weights,
                mixed_weights=zeros_weights,
            )

            left_partition = partition_response(
                points,
                centers,
                point_motion=point_unit,
                center_motion=unit,
                iterations=grid_spec.partition_iterations,
                coincident_tolerance=grid_spec.coincident_tolerance,
            )
            mixed_partition = partition_mixed_response(
                points,
                centers,
                left_point_motion=point_unit,
                left_center_motion=unit,
                right_point_motion=right_points,
                right_center_motion=vector,
                iterations=grid_spec.partition_iterations,
                coincident_tolerance=grid_spec.coincident_tolerance,
            )
            for branch in (
                left_partition.branch_identity,
                mixed_partition.branch_identity,
            ):
                if branch != directional.grid_branch_identity:
                    raise ValueError("XC HVP crossed a Becke response branch")
            left_weight_motion = atomic_weights * left_partition.directional[selected]
            mixed_weight_motion = atomic_weights * mixed_partition.mixed[selected]
            components["xc_weight"][atom, axis] = contract(
                left_centers=zeros_centers,
                left_points=zeros_points,
                left_weights=left_weight_motion,
                mixed_weights=mixed_weight_motion,
            )

    operator.validate_current()
    arrays = tuple(
        immutable(components[name]) for name in ("xc_ao", "xc_grid", "xc_weight")
    )
    identity = canonical_hash(
        {
            "schema": "vibeqc.native-rks-xc-hvp/v1",
            "directional_response": directional.identity,
            "grid_branch": directional.grid_branch_identity,
            "sources": ("xc_ao", "xc_grid", "xc_weight"),
        }
    )
    return RKSXCHVPComponents(
        identity,
        arrays[0],
        arrays[1],
        arrays[2],
        directional.grid_branch_identity,
    )


def directional_rks_response(
    operator: typing.Any,
    direction: typing.Any,
    *,
    cache: typing.Any = ".artifacts",
    solver_options: typing.Any = None,
) -> DirectionalRKSResponse:
    """Solve one all-electron direct LDA/PBE RKS nuclear perturbation.

    The frozen Fock direction is hcore'(v) + J'(D;v) + Vxc'(D,R;v). The shared
    CPKS operator owns induced J/fxc response to D'(v); overlap/metric terms are
    handled by the common stationary nuclear reconstruction.
    """
    if not isinstance(operator, NativeRKSResponse):
        raise TypeError("directional RKS response requires NativeRKSResponse")
    if solver_options is not None and not isinstance(solver_options, GMRESOptions):
        raise TypeError("solver_options must be GMRESOptions")
    operator.validate_current()
    state = operator.state
    basis = operator.xc_kernel.basis
    vector = checked_direction(direction, basis.natom)
    integral, overlap = generated_directional_semilocal_rks_integral_first_order(
        operator._source,
        state.density[0],
        vector,
        cache=cache,
    )
    xc, branch_identity = _native_rks_xc_geometry_direction(operator, vector)
    frozen = immutable(integral + xc)
    solved = solve_stationary_nuclear_perturbation(
        operator,
        frozen,
        overlap,
        options=solver_options,
    )
    operator.validate_current()
    identity = canonical_hash(
        {
            "schema": "vibeqc.directional-rks-nuclear-response/v1",
            "state": state.identity.to_payload(),
            "response_operator": operator.problem.operator_identity,
            "direction": vector.tolist(),
            "grid_branch": branch_identity,
            "sources": (
                "one-electron",
                "direct-coulomb",
                "native-xc-geometry",
                "overlap-metric",
                "cpks-density-response",
            ),
        }
    )
    return DirectionalRKSResponse(
        identity=identity,
        direction=immutable(vector),
        integral_frozen_fock_derivative=immutable(integral),
        xc_frozen_fock_derivative=xc,
        frozen_fock_derivative=frozen,
        overlap_derivative=immutable(overlap),
        response=solved,
        grid_branch_identity=branch_identity,
    )
