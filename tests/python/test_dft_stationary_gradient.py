from dataclasses import replace

import numpy as np
import pytest
from vibeqc import _native
from vibeqc._dft_gradient import (
    StableGridMotion,
    StationaryDerivativeContract,
    StationaryKsIdentity,
    StationaryKsState,
    _fixed_density_xc_geometry,
    _native_ao_atoms,
    bind_generated_xc_geometry,
    native_ao_geometry_identity,
    xc_geometry_topology_identity,
    xc_regularization_identity,
)
from vibeqc._ks_snapshot import _scf_xc_points
from vibeqc_compiler.dft import NativeAO
from vibeqc_compiler.dft.fixtures import basis_arguments
from vibeqc_compiler.xc import functional
from vibeqc_compiler.xc.contractions import ContractionProgram
from vibeqc_compiler.xc.integration_fixtures import load_integration_fixture

from tools.vibeqc_validation.dft_gradient import finite_difference_xc_directional


def identity(method="pbe-rks"):
    return StationaryKsIdentity(
        method=method,
        model_identity="model-pbe-rks-v1",
        geometry_identity="geometry-1",
        basis_identity="basis-1",
        overlap_identity="overlap-1",
        grid_identity="grid-1",
        topology_identity="topology-1",
        functional_identity="functional-pbe-unpolarized-v1",
        regularization_identity="regularization-1",
        provider_identity="direct-j-cpu-v1",
        owner=17,
        solve_epoch=3,
        density_generation=7,
        fock_generation=9,
        orbital_generation=11,
    )


def state(method="pbe-rks", occupations=None, overlap=None):
    spins = 1 if method.endswith("rks") else 2
    overlap = np.diag([1.0, 2.0]) if overlap is None else overlap
    eigenvalues, vectors = np.linalg.eigh(overlap)
    coefficients = (vectors / np.sqrt(eigenvalues)) @ vectors.T
    energies = np.array([[-0.8, 0.5]] * spins)
    if occupations is None:
        occupations = (
            np.array([[2.0, 0.0]]) if spins == 1 else np.array([[1.0, 0.0], [1.0, 0.0]])
        )
    occupations = np.asarray(occupations, dtype=float)
    density = np.stack(
        [(coefficients * occupations[spin]) @ coefficients.T for spin in range(spins)]
    )
    weighted = np.stack(
        [
            (coefficients * (occupations[spin] * energies[spin])) @ coefficients.T
            for spin in range(spins)
        ]
    )
    fock = np.stack(
        [
            overlap @ coefficients @ np.diag(row) @ coefficients.T @ overlap
            for row in energies
        ]
    )
    return StationaryKsState(
        identity=identity(method),
        density=density,
        fock=fock,
        coefficients=np.stack([coefficients] * spins),
        orbital_energies=energies,
        occupations=occupations,
        weighted_density=weighted,
        overlap=overlap,
        physical_residual=0.0,
        successful=True,
        converged=True,
        physical=True,
    )


def bound_h2(method, name, spin, *, occupations=None):
    spec = functional(name, spin=spin)
    meta, _, grid = load_integration_fixture("h2")
    args = basis_arguments(meta)
    with NativeAO(**args) as basis:
        # This is a manufactured fixed-density algebra fixture, not a solved
        # KS state. Use the true H2 metric nonetheless, so the oracle never
        # contracts matrices relabeled from a different AO space.
        from tools.vibeqc_validation.dft_gradient import h2_overlap

        value = state(method, occupations=occupations, overlap=h2_overlap(basis))
        value = replace(
            value,
            identity=replace(
                value.identity,
                basis_identity=basis.identity,
                geometry_identity=native_ao_geometry_identity(basis),
                grid_identity=grid.identity,
                topology_identity=xc_geometry_topology_identity(basis, grid),
                functional_identity=spec.identity,
                regularization_identity=xc_regularization_identity(spec),
            ),
        )
        bound = _fixed_density_xc_geometry(
            StationaryDerivativeContract(value.identity), value, spec, basis, grid
        )
        natom = basis.natom
    density = value.density[0] if spin == "unpolarized" else value.density
    return spec, value, args, grid, density, bound, natom


@pytest.mark.parametrize(
    "method,name,spin",
    [
        ("lda-rks", "LDA_XC_PW", "unpolarized"),
        ("pbe-rks", "PBE", "unpolarized"),
        ("lda-uks", "LDA_XC_PW", "polarized"),
        ("pbe-uks", "PBE", "polarized"),
    ],
)
def test_cartesian_point_coefficient_pullback_matches_generated_interior(
    method, name, spin
):
    spec, _, args, grid, density, bound, _ = bound_h2(method, name, spin)
    with NativeAO(**args) as basis:
        program = ContractionProgram(spec, "geometry")
        jets = basis.evaluate(grid.points, program.contract.ao_order)
        features = program.features(jets, density)
        rows = program.scalar_values(features)
        v = program._gradient(rows, len(grid.points))
        gradient = features.get("gradient")
        functional_gradient = (
            gradient
            if gradient is None or spec.spin == "polarized"
            else gradient.sum(axis=0)
        )
        compact = program.coefficients.evaluate(functional_gradient, v)
        rho = compact["rho"]
        cartesian_gradient = compact.get("gradient")
        if spec.spin == "unpolarized":
            rho = np.repeat(rho, 2, axis=0)
            if cartesian_gradient is not None:
                cartesian_gradient = np.repeat(cartesian_gradient, 2, axis=0)
        actual = program.geometry_from_cartesian_coefficients(
            jets,
            density,
            grid.weights,
            rows[()],
            rho,
            cartesian_gradient,
            ao_atoms=_native_ao_atoms(basis),
            natom=basis.natom,
        )
    np.testing.assert_allclose(actual.centers, bound.partials.centers, atol=2e-12)
    np.testing.assert_allclose(actual.points, bound.partials.points, atol=2e-12)
    np.testing.assert_allclose(actual.weights, bound.partials.weights, atol=2e-12)


@pytest.mark.parametrize(
    "method,name,spin",
    [
        ("lda-rks", "LDA_XC_PW", "unpolarized"),
        ("pbe-rks", "PBE", "unpolarized"),
        ("lda-uks", "LDA_XC_PW", "polarized"),
        ("pbe-uks", "PBE", "polarized"),
    ],
)
def test_scf_domain_pullback_matches_independent_displaced_energy(method, name, spin):
    spec, value, args, grid, density, _, _ = bound_h2(
        method,
        name,
        spin,
        occupations=([[1.0, 0.0], [0.7, 0.2]] if spin == "polarized" else None),
    )
    library = _native.load_library(device="cpu")
    pbe = method.startswith("pbe")

    def point_energy(features):
        gradient = features.get("gradient")
        if gradient is None:
            gradient = np.zeros((2, len(grid.points), 3))
        return _scf_xc_points(library, pbe, features["rho"], gradient)["energy"]

    with NativeAO(**args) as basis:
        program = ContractionProgram(spec, "geometry")
        jets = basis.evaluate(grid.points, program.contract.ao_order)
        features = program.features(jets, density)
        gradient = features.get("gradient")
        if gradient is None:
            gradient = np.zeros((2, len(grid.points), 3))
        point = _scf_xc_points(library, pbe, features["rho"], gradient)
        partials = program.geometry_from_cartesian_coefficients(
            jets,
            density,
            grid.weights,
            point["energy"],
            point["rho"],
            point["gradient"] if pbe else None,
            ao_atoms=_native_ao_atoms(basis),
            natom=basis.natom,
        )

    rng = np.random.default_rng(16301)
    motion = StableGridMotion(
        topology_identity=value.identity.topology_identity,
        centers=rng.normal(size=partials.centers.shape) * 0.05,
        points=rng.normal(size=partials.points.shape) * 0.03,
        weights=rng.normal(size=partials.weights.shape) * 0.0005,
    )
    expected = (
        np.sum(partials.centers * motion.centers)
        + np.sum(partials.points * motion.points)
        + np.sum(partials.weights * motion.weights)
    )
    oracle = finite_difference_xc_directional(
        spec,
        args,
        grid.points,
        grid.weights,
        density,
        motion,
        steps=(1e-3, 3e-4, 1e-4),
        point_energy=point_energy,
    )
    assert oracle.spread < 2e-7
    np.testing.assert_allclose(oracle.stable_estimate, expected, atol=2e-8)


@pytest.mark.parametrize("method", ["lda-rks", "pbe-rks", "lda-uks", "pbe-uks"])
def test_stationary_contract_accepts_consistent_rks_and_uks(method):
    value = state(
        method,
        occupations=([[1.0, 0.0], [0.7, 0.2]] if method == "pbe-uks" else None),
    )
    contract = StationaryDerivativeContract(value.identity)

    assert contract._validate_arrays(value) is value
    assert contract.spin == ("unpolarized" if method.endswith("rks") else "polarized")
    assert contract.family == ("lda" if method.startswith("lda") else "gga")
    assert contract.to_payload()["force_capability"] == "unsupported"
    assert contract.identity == StationaryDerivativeContract(value.identity).identity


@pytest.mark.parametrize(
    "field",
    [
        "model_identity",
        "geometry_identity",
        "basis_identity",
        "overlap_identity",
        "grid_identity",
        "topology_identity",
        "functional_identity",
        "regularization_identity",
        "provider_identity",
        "owner",
        "solve_epoch",
        "density_generation",
        "fock_generation",
        "orbital_generation",
    ],
)
def test_stationary_contract_rejects_every_stale_identity_axis(field):
    value = state()
    old = getattr(value.identity, field)
    changed = old + 1 if isinstance(old, int) else old + "-stale"
    stale = replace(value, identity=replace(value.identity, **{field: changed}))

    with pytest.raises(ValueError, match="identity mismatch"):
        StationaryDerivativeContract(value.identity)._validate_arrays(stale)


@pytest.mark.parametrize("flag", ["successful", "converged", "physical"])
def test_stationary_contract_rejects_unsuccessful_state(flag):
    value = state()
    with pytest.raises(ValueError, match="successful converged physical"):
        StationaryDerivativeContract(value.identity)._validate_arrays(
            replace(value, **{flag: False})
        )


@pytest.mark.parametrize(
    "field,delta,message",
    [
        ("density", 1e-4, "density reconstruction"),
        ("weighted_density", 1e-4, "weighted-density reconstruction"),
        ("fock", 1e-4, "Fock eigen residual"),
        ("coefficients", 1e-4, "S-orthonormal"),
    ],
)
def test_stationary_contract_rejects_inconsistent_orbital_state(field, delta, message):
    value = state()
    changed = np.array(getattr(value, field), copy=True)
    changed[0, 0, 0] += delta
    with pytest.raises(ValueError, match=message):
        StationaryDerivativeContract(value.identity)._validate_arrays(
            replace(value, **{field: changed})
        )


def test_stationary_contract_requires_weighted_density_and_true_residual():
    value = state()
    with pytest.raises(ValueError, match="weighted density"):
        StationaryDerivativeContract(value.identity)._validate_arrays(
            replace(value, weighted_density=np.empty((0, 2, 2)))
        )
    with pytest.raises(ValueError, match="physical residual"):
        StationaryDerivativeContract(value.identity)._validate_arrays(
            replace(value, physical_residual=1e-5)
        )
    with pytest.raises(ValueError, match="physical residual"):
        StationaryDerivativeContract(value.identity)._validate_arrays(
            replace(value, physical_residual=-1e-10)
        )


def test_stationary_state_owns_read_only_snapshot_arrays():
    density = np.array(state().density, copy=True)
    value = replace(state(), density=density)
    density[:] = 99.0

    assert not np.any(value.density == 99.0)
    with pytest.raises(ValueError, match="read-only"):
        value.density[0, 0, 0] = 1.0


@pytest.mark.parametrize("field", ["successful", "converged", "physical"])
def test_stationary_state_rejects_truthy_nonboolean_gates(field):
    with pytest.raises(TypeError, match="boolean"):
        replace(state(), **{field: "false"})


@pytest.mark.parametrize("residual", [True, np.nan, np.array([0.0])])
def test_stationary_state_requires_finite_scalar_physical_residual(residual):
    with pytest.raises((TypeError, ValueError), match="physical residual"):
        replace(state(), physical_residual=residual)


@pytest.mark.parametrize("method", ["b3lyp-rks", "pbe0-rks", "pbe-rhf"])
def test_stationary_contract_rejects_unsupported_method_domain(method):
    with pytest.raises(ValueError, match="LDA/PBE RKS/UKS"):
        replace(identity(), method=method)


@pytest.mark.parametrize(
    "method,name,spin",
    [
        ("lda-rks", "LDA_XC_PW", "unpolarized"),
        ("pbe-rks", "PBE", "unpolarized"),
        ("lda-uks", "LDA_XC_PW", "polarized"),
        ("pbe-uks", "PBE", "polarized"),
    ],
)
def test_generated_xc_geometry_is_bound_to_stationary_identity(method, name, spin):
    spec, value, _, grid, _, bound, natom = bound_h2(
        method,
        name,
        spin,
        occupations=([[1.0, 0.0], [0.7, 0.2]] if method == "pbe-uks" else None),
    )

    assert bound.state_identity == value.identity
    assert bound.functional_identity == spec.identity
    assert bound.density_generation == value.identity.density_generation
    assert bound.partials.centers.shape == (natom, 3)
    assert bound.partials.points.shape == grid.points.shape
    assert bound.partials.weights.shape == grid.weights.shape
    assert bound.force_capability == "unsupported"


def test_generated_xc_binding_rejects_actual_source_or_method_mismatch():
    from dataclasses import replace as dc_replace
    from fractions import Fraction

    spec, value, args, grid, _, _, _ = bound_h2("pbe-rks", "PBE", "unpolarized")
    with NativeAO(**args) as basis:
        stale = dc_replace(
            value,
            identity=dc_replace(value.identity, basis_identity="stale-basis"),
        )
        with pytest.raises(ValueError, match="basis identity"):
            _fixed_density_xc_geometry(
                StationaryDerivativeContract(stale.identity), stale, spec, basis, grid
            )

        stale_regularization = dc_replace(
            value,
            identity=dc_replace(
                value.identity, regularization_identity="clipped-density-v1"
            ),
        )
        with pytest.raises(ValueError, match="regularization identity"):
            _fixed_density_xc_geometry(
                StationaryDerivativeContract(stale_regularization.identity),
                stale_regularization,
                spec,
                basis,
                grid,
            )

    wrong = dc_replace(spec, components=(("GGA_X_PBE", Fraction(1)),))
    with NativeAO(**args) as basis:
        relabeled = dc_replace(
            value,
            identity=dc_replace(value.identity, functional_identity=wrong.identity),
        )
        with pytest.raises(ValueError, match="canonical PBE"):
            _fixed_density_xc_geometry(
                StationaryDerivativeContract(relabeled.identity),
                relabeled,
                wrong,
                basis,
                grid,
            )


def test_generated_xc_binding_rejects_out_of_range_grid_owner():
    from vibeqc_compiler.dft import ExplicitGrid

    spec, value, args, grid, _, _, _ = bound_h2("pbe-rks", "PBE", "unpolarized")
    invalid = ExplicitGrid(
        grid.points,
        grid.weights,
        (2,) + grid.owners[1:],
        {"test": "invalid owner"},
    )
    with NativeAO(**args) as basis:
        relabeled = replace(
            value,
            identity=replace(
                value.identity,
                grid_identity=invalid.identity,
                topology_identity="invalid-owner-topology",
            ),
        )
        with pytest.raises(ValueError, match="grid owner"):
            _fixed_density_xc_geometry(
                StationaryDerivativeContract(relabeled.identity),
                relabeled,
                spec,
                basis,
                invalid,
            )


def test_stable_motion_reports_each_xc_component_once_and_translation():
    _, value, _, _, _, bound, _ = bound_h2("pbe-rks", "PBE", "unpolarized")
    rng = np.random.default_rng(163)
    centers = rng.normal(size=bound.partials.centers.shape)
    points = rng.normal(size=bound.partials.points.shape)
    weights = rng.normal(size=bound.partials.weights.shape)
    result = bound.directional(
        StableGridMotion(
            topology_identity=value.identity.topology_identity,
            centers=centers,
            points=points,
            weights=weights,
        )
    )
    np.testing.assert_allclose(result.center, np.sum(bound.partials.centers * centers))
    np.testing.assert_allclose(result.point, np.sum(bound.partials.points * points))
    np.testing.assert_allclose(result.weight, np.sum(bound.partials.weights * weights))
    np.testing.assert_allclose(
        result.total, result.center + result.point + result.weight
    )

    translation = np.array([0.2, -0.1, 0.3])
    rigid = bound.directional(
        StableGridMotion(
            topology_identity=value.identity.topology_identity,
            centers=np.broadcast_to(translation, bound.partials.centers.shape),
            points=np.broadcast_to(translation, bound.partials.points.shape),
            weights=np.zeros_like(bound.partials.weights),
        )
    )
    np.testing.assert_allclose(rigid.total, 0, atol=1e-12)


@pytest.mark.parametrize(
    "change,message",
    [
        ({"topology_identity": "other"}, "topology identity"),
        ({"topology_changed": True}, "topology change"),
        ({"centers": np.zeros((1, 3))}, "center direction"),
        ({"points": np.array([[np.nan, 0.0, 0.0]])}, "point direction"),
    ],
)
def test_generated_xc_geometry_rejects_stale_invalid_or_changing_motion(
    change, message
):
    from vibeqc_compiler.xc.contractions import GeometryPartials

    value = state()
    spec = functional("PBE", spin="unpolarized")
    value = replace(
        value,
        identity=replace(value.identity, functional_identity=spec.identity),
    )
    partials = GeometryPartials(np.zeros((2, 3)), np.zeros((3, 3)), np.zeros(3))
    from vibeqc._dft_gradient import FixedDensityXcGeometry

    bound = FixedDensityXcGeometry(
        state_identity=value.identity,
        discrete_contract_identity="generated-contract",
        basis_identity=value.identity.basis_identity,
        geometry_identity=value.identity.geometry_identity,
        grid_identity=value.identity.grid_identity,
        topology_identity=value.identity.topology_identity,
        functional_identity=value.identity.functional_identity,
        regularization_identity=value.identity.regularization_identity,
        density_generation=value.identity.density_generation,
        partials=partials,
    )
    values = {
        "topology_identity": value.identity.topology_identity,
        "topology_changed": False,
        "centers": np.zeros((2, 3)),
        "points": np.zeros((3, 3)),
        "weights": np.zeros(3),
    }
    values.update(change)
    with pytest.raises(ValueError, match=message):
        bound.directional(StableGridMotion(**values))


def test_generated_xc_geometry_owns_read_only_partial_arrays():
    from vibeqc._dft_gradient import FixedDensityXcGeometry
    from vibeqc_compiler.xc.contractions import GeometryPartials

    value = state()
    centers = np.zeros((2, 3))
    bound = FixedDensityXcGeometry(
        state_identity=value.identity,
        discrete_contract_identity="generated-contract",
        basis_identity=value.identity.basis_identity,
        geometry_identity=value.identity.geometry_identity,
        grid_identity=value.identity.grid_identity,
        topology_identity=value.identity.topology_identity,
        functional_identity=value.identity.functional_identity,
        regularization_identity=value.identity.regularization_identity,
        density_generation=value.identity.density_generation,
        partials=GeometryPartials(centers, np.zeros((3, 3)), np.zeros(3)),
    )
    centers[:] = np.inf

    assert np.isfinite(bound.partials.centers).all()
    with pytest.raises(ValueError, match="read-only"):
        bound.partials.centers[0, 0] = 1.0


@pytest.mark.parametrize(
    "method,name,spin",
    [
        ("lda-rks", "LDA_XC_PW", "unpolarized"),
        ("pbe-rks", "PBE", "unpolarized"),
        ("lda-uks", "LDA_XC_PW", "polarized"),
        ("pbe-uks", "PBE", "polarized"),
    ],
)
def test_stationary_xc_directions_match_independent_multistep_oracle(
    method, name, spin
):
    spec, value, args, grid, density, bound, _ = bound_h2(
        method,
        name,
        spin,
        occupations=([[1.0, 0.0], [0.7, 0.2]] if spin == "polarized" else None),
    )
    rng = np.random.default_rng(1634)
    centers = rng.normal(size=(len(args["atoms"]), 3)) * 0.07
    points = rng.normal(size=grid.points.shape) * 0.04
    weights = rng.normal(size=grid.weights.shape) * 0.001

    zero_centers = np.zeros_like(centers)
    zero_points = np.zeros_like(points)
    zero_weights = np.zeros_like(weights)
    motions = (
        (centers, zero_points, zero_weights),
        (zero_centers, points, zero_weights),
        (zero_centers, zero_points, weights),
        (centers, points, weights),
    )
    for dc, dp, dw in motions:
        motion = StableGridMotion(
            topology_identity=value.identity.topology_identity,
            centers=dc,
            points=dp,
            weights=dw,
        )
        expected = bound.directional(motion).total
        oracle = finite_difference_xc_directional(
            spec,
            args,
            grid.points,
            grid.weights,
            density,
            motion,
            steps=(1e-3, 3e-4, 1e-4),
        )
        assert oracle.spread < 3e-8
        np.testing.assert_allclose(oracle.stable_estimate, expected, atol=3e-9)


def test_stationary_xc_oracle_detects_omission_and_sign_reversal():
    spec, value, args, grid, density, bound, _ = bound_h2(
        "pbe-rks", "PBE", "unpolarized"
    )
    rng = np.random.default_rng(1635)
    motion = StableGridMotion(
        topology_identity=value.identity.topology_identity,
        centers=rng.normal(size=(len(args["atoms"]), 3)) * 0.07,
        points=rng.normal(size=grid.points.shape) * 0.04,
        weights=rng.normal(size=grid.weights.shape) * 0.001,
    )
    components = bound.directional(motion)
    oracle = finite_difference_xc_directional(
        spec,
        args,
        grid.points,
        grid.weights,
        density,
        motion,
    )
    assert abs(oracle.stable_estimate - components.total) < 3e-9
    for omitted in (
        components.center + components.point,
        components.center + components.weight,
        components.point + components.weight,
    ):
        assert abs(oracle.stable_estimate - omitted) > 1e-7
    assert abs(oracle.stable_estimate + components.total) > 1e-7


def test_manufactured_state_cannot_authorize_stationary_derivatives():
    value = state()
    with pytest.raises(ValueError, match="current native #162 snapshot"):
        StationaryDerivativeContract(value.identity).validate(value)
    spec, value, args, grid, _, _, _ = bound_h2("pbe-rks", "PBE", "unpolarized")
    with (
        NativeAO(**args) as basis,
        pytest.raises(ValueError, match="current native #162 snapshot"),
    ):
        bind_generated_xc_geometry(
            StationaryDerivativeContract(value.identity), value, spec, basis, grid
        )


def test_finite_xc_components_cannot_publish_an_overflowed_total():
    from vibeqc._dft_gradient import XcDirectionalComponents

    with pytest.raises(ArithmeticError, match="nonfinite total"):
        _ = XcDirectionalComponents(1e308, 1e308, 0.0).total
