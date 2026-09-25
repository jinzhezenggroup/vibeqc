"""Real semilocal RKS nuclear directions for the matrix-free DFT Hessian line.

This module composes existing owners only: generated one-/two-electron geometry
derivatives, the native SCF-domain XC point model, analytic AO/grid JVPs, and the
shared CPKS nuclear perturbation solve. It does not assemble a molecular HVP.
"""

import typing
from dataclasses import dataclass

import numpy as np
from vibeqc._dft_gradient import _native_ao_atoms
from vibeqc.profiles import canonical_hash
from vibeqc_compiler.xc.contractions import ExternalPointContraction
from vibeqc_compiler.xc.grid_response import partition_response
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
