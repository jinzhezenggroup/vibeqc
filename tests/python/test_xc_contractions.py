"""Independent E/V fixtures and directional checks of compact XC contractions."""

from functools import lru_cache

import numpy as np
import pytest
from vibeqc_compiler.dft import NativeAO
from vibeqc_compiler.dft.fixtures import basis_arguments
from vibeqc_compiler.method import resolve_method
from vibeqc_compiler.xc import UnsupportedXC, functional
from vibeqc_compiler.xc.contractions import ContractionProgram, ExternalPointContraction
from vibeqc_compiler.xc.integration_fixtures import load_integration_fixture as fixture


@lru_cache(maxsize=16)
def program(
    name: str, spin: str = "polarized", observable: str = "potential"
) -> ContractionProgram:
    return ContractionProgram(functional(name, spin=spin), observable)


@pytest.mark.parametrize("case", ["h2", "f_cartesian", "f_spherical"])
@pytest.mark.parametrize("name", ["LDA_XC_PW", "PBE"])
@pytest.mark.parametrize(
    "layout,spin",
    [("total", "unpolarized"), ("total", "polarized"), ("spin", "polarized")],
)
def test_minimal_contractions_preserve_independent_fixtures(
    case: str, name: str, layout: str, spin: str
) -> None:
    meta, data, grid = fixture(case)
    consumer = program(name, spin)
    with NativeAO(**basis_arguments(meta)) as basis:
        jets = basis.evaluate(grid.points, consumer.contract.ao_order)
        values = consumer.evaluate(jets, data[f"density_{layout}"], grid.weights)
    potential = values["potential"]
    if layout == "total":
        potential = potential.mean(axis=0)
    np.testing.assert_allclose(
        values["energy"], data[f"{name}_{layout}_energy"][0], atol=1e-11, rtol=1e-10
    )
    np.testing.assert_allclose(
        potential, data[f"{name}_{layout}_potential"], atol=1e-11, rtol=1e-10
    )
    assert "tau" not in consumer.features(jets, data[f"density_{layout}"])
    if name.startswith("LDA"):
        assert jets.shape[0] == 1
        assert consumer.contract.scalar_outputs == (
            ((), (0,), (1,)) if spin == "polarized" else ((), (0,))
        )


def test_r2scan_vtau_potential_matches_complete_density_directional_derivative() -> (
    None
):
    rng = np.random.default_rng(164)
    jets = rng.normal(size=(4, 13, 3))
    density = np.stack((np.eye(3), 0.7 * np.eye(3)))
    direction = rng.normal(size=density.shape)
    direction = 0.03 * (direction + direction.swapaxes(-1, -2))
    weights = rng.uniform(0.1, 1.0, jets.shape[1])
    consumer = program("R2SCAN")
    features = consumer.features(jets, density)
    assert set(features) >= {"rho", "gradient", "sigma", "tau"}
    rows = consumer.scalar_values(features)
    assert np.max(np.abs(rows[(5,)])) > 1e-8
    assert np.max(np.abs(rows[(6,)])) > 1e-8
    value = consumer.evaluate(jets, density, weights)
    analytic = np.sum(value["potential"] * direction)
    errors = []
    for step in (1e-4, 3e-5, 1e-5):
        plus = consumer.evaluate(jets, density + step * direction, weights)["energy"]
        minus = consumer.evaluate(jets, density - step * direction, weights)["energy"]
        errors.append(abs((plus - minus) / (2 * step) - analytic))
    assert np.all(np.asarray(errors) < [2e-7, 3e-8, 5e-9]), errors


def test_b3lyp_methodir_geometry_matches_moved_collocation() -> None:
    meta, data, grid = fixture("h2")
    args = basis_arguments(meta)
    density = data["density_total"]
    functional_spec = (
        resolve_method("B3LYP", spin="unpolarized").primitives[0].functional
    )
    geometry = ContractionProgram(functional_spec, "geometry")
    energy = ContractionProgram(functional_spec, "energy")
    with NativeAO(**args) as basis:
        ao_atoms = np.repeat(
            [shell.atom_index for shell in basis.shells],
            [
                2 * shell.angular_momentum + 1
                if basis.representation == "real_spherical"
                else (shell.angular_momentum + 1) * (shell.angular_momentum + 2) // 2
                for shell in basis.shells
            ],
        )
        partials = geometry.evaluate(
            basis.evaluate(grid.points, geometry.contract.ao_order),
            density,
            grid.weights,
            ao_atoms=ao_atoms,
            natom=basis.natom,
        )["geometry"]

    centers = np.array([[0.013, -0.009, 0.011], [-0.007, 0.012, -0.005]])
    points = np.tile(np.array([[0.002, -0.001, 0.003]]), (len(grid.points), 1))
    measure = np.linspace(-1.5e-5, 1.5e-5, len(grid.weights))
    expected = partials.directional(centers=centers, points=points, weights=measure)
    errors = []
    for step in (2e-4, 7e-5, 2e-5):
        values = []
        for sign in (1, -1):
            moved = [
                (atom, np.asarray(position) + sign * step * delta)
                for (atom, position), delta in zip(args["atoms"], centers, strict=True)
            ]
            with NativeAO(**{**args, "atoms": moved}) as basis:
                moved_jets = basis.evaluate(
                    grid.points + sign * step * points, energy.contract.ao_order
                )
                values.append(
                    energy.evaluate(
                        moved_jets,
                        density,
                        grid.weights + sign * step * measure,
                    )["energy"]
                )
        errors.append(abs((values[0] - values[1]) / (2 * step) - expected))
    assert np.all(np.asarray(errors) < [3e-7, 4e-8, 8e-9]), errors


def test_r2scan_unvalidated_density_response_fails_closed() -> None:
    with pytest.raises(UnsupportedXC, match="tau-dependent density response"):
        program("R2SCAN", observable="response")


@pytest.mark.parametrize("spin", ["polarized", "unpolarized"])
def test_r2scan_geometry_includes_tau_and_matches_moved_collocation(spin: str) -> None:
    meta, data, grid = fixture("h2")
    args = basis_arguments(meta)
    density = data["density_spin" if spin == "polarized" else "density_total"]
    geometry = program("R2SCAN", spin, "geometry")
    energy = program("R2SCAN", spin, "energy")
    with NativeAO(**args) as basis:
        ao_atoms = np.repeat(
            [shell.atom_index for shell in basis.shells],
            [
                2 * shell.angular_momentum + 1
                if basis.representation == "real_spherical"
                else (shell.angular_momentum + 1) * (shell.angular_momentum + 2) // 2
                for shell in basis.shells
            ],
        )
        jets = basis.evaluate(grid.points, geometry.contract.ao_order)
        assert geometry.contract.ao_order == 2
        assert set(geometry.features(jets, density)) >= {
            "rho",
            "gradient",
            "sigma",
            "tau",
        }
        partials = geometry.evaluate(
            jets,
            density,
            grid.weights,
            ao_atoms=ao_atoms,
            natom=basis.natom,
        )["geometry"]
    centers = np.array([[0.017, -0.011, 0.013], [-0.009, 0.014, -0.007]])
    points = np.tile(np.array([[0.003, -0.002, 0.001]]), (len(grid.points), 1))
    measure = np.linspace(-2e-5, 2e-5, len(grid.weights))
    expected = partials.directional(centers=centers, points=points, weights=measure)
    errors = []
    for step in (2e-4, 7e-5, 2e-5):
        values = []
        for sign in (1, -1):
            moved = [
                (atom, np.asarray(position) + sign * step * delta)
                for (atom, position), delta in zip(args["atoms"], centers, strict=True)
            ]
            with NativeAO(**{**args, "atoms": moved}) as basis:
                moved_jets = basis.evaluate(
                    grid.points + sign * step * points, energy.contract.ao_order
                )
                values.append(
                    energy.evaluate(
                        moved_jets,
                        density,
                        grid.weights + sign * step * measure,
                    )["energy"]
                )
        errors.append(abs((values[0] - values[1]) / (2 * step) - expected))
    assert np.all(np.asarray(errors) < [2e-7, 3e-8, 6e-9]), errors


@pytest.mark.parametrize("name", ["LDA_XC_PW", "PBE"])
def test_spin_resolved_response_finite_differences_transpose_and_exchange(
    name: str,
) -> None:
    meta, data, grid = fixture("h2")
    primal, response = program(name), program(name, observable="response")
    density = data["density_spin"]
    rng = np.random.default_rng(236)
    directions = rng.normal(size=(2, *density.shape)) * 0.04
    directions = (directions + directions.swapaxes(-1, -2)) / 2
    with NativeAO(**basis_arguments(meta)) as basis:
        jets = basis.evaluate(grid.points, primal.contract.ao_order)
    actions = [
        response.evaluate(jets, density, grid.weights, delta_density=d)["response"]
        for d in directions
    ]
    for direction, action in zip(directions, actions, strict=True):
        errors = []
        for step in (1e-3, 3e-4, 1e-4):
            plus = primal.evaluate(jets, density + step * direction, grid.weights)[
                "potential"
            ]
            minus = primal.evaluate(jets, density - step * direction, grid.weights)[
                "potential"
            ]
            errors.append(np.max(np.abs((plus - minus) / (2 * step) - action)))
        assert np.all(np.asarray(errors) < [3e-8, 3e-9, 4e-10]), errors
    np.testing.assert_allclose(
        np.sum(directions[0] * actions[1]),
        np.sum(actions[0] * directions[1]),
        atol=1e-12,
        rtol=1e-10,
    )
    # Check the actual shared svec coordinates, including sqrt(2) on the
    # off-diagonal entries, rather than a second unweighted triangular dot.
    from tools.vibeqc_posthf.pair_space import PairSpace

    pairs = PairSpace(density.shape[-1])
    packed_directions = [np.concatenate([pairs.pack(s) for s in d]) for d in directions]
    packed_actions = [np.concatenate([pairs.pack(s) for s in a]) for a in actions]
    np.testing.assert_allclose(
        packed_directions[0] @ packed_actions[1],
        packed_actions[0] @ packed_directions[1],
        atol=1e-12,
        rtol=1e-10,
    )
    # Pin the metric against the full trace independently of reciprocal
    # packing, so a self-consistent wrong off-diagonal scale cannot pass.
    np.testing.assert_allclose(
        packed_directions[0] @ packed_actions[1],
        np.sum(directions[0] * actions[1]),
        atol=1e-12,
        rtol=1e-10,
    )
    swapped = response.evaluate(
        jets, density[::-1], grid.weights, delta_density=directions[0, ::-1]
    )["response"]
    np.testing.assert_allclose(swapped, actions[0][::-1], atol=1e-12, rtol=1e-10)
    # A beta-only direction must retain alpha response from cross-spin terms.
    beta = directions[0].copy()
    beta[0] = 0
    cross = response.evaluate(jets, density, grid.weights, delta_density=beta)[
        "response"
    ]
    assert np.max(np.abs(cross[0])) > 1e-8


@pytest.mark.parametrize("name", ["LDA_XC_PW", "PBE"])
@pytest.mark.parametrize("spin", ["polarized", "unpolarized"])
@pytest.mark.parametrize("case", ["h2", "f_cartesian", "f_spherical"])
def test_explicit_geometry_sources_against_moved_native_collocation(
    name: str, spin: str, case: str
) -> None:
    meta, data, grid = fixture(case)
    args = basis_arguments(meta)
    density = data["density_spin" if spin == "polarized" else "density_total"]
    geometry = program(name, spin, "geometry")
    energy = program(name, spin, "energy")
    rng = np.random.default_rng(2361)
    with NativeAO(**args) as basis:
        ao_atoms = np.repeat(
            [s.atom_index for s in basis.shells],
            [
                2 * s.angular_momentum + 1
                if basis.representation == "real_spherical"
                else (s.angular_momentum + 1) * (s.angular_momentum + 2) // 2
                for s in basis.shells
            ],
        )
        partials = geometry.evaluate(
            basis.evaluate(grid.points, geometry.contract.ao_order),
            density,
            grid.weights,
            ao_atoms=ao_atoms,
            natom=basis.natom,
        )["geometry"]
    centers = rng.normal(size=(len(args["atoms"]), 3)) * 0.07
    points = rng.normal(size=grid.points.shape) * 0.04
    weights = rng.normal(size=grid.weights.shape) * 0.001
    # Test all sources separately as well as together, so omitted or repeated
    # grid/weight motion cannot cancel a wrong AO-center sign.
    for motion in (
        (centers, points * 0, weights * 0),
        (centers * 0, points, weights * 0),
        (centers * 0, points * 0, weights),
        (centers, points, weights),
    ):
        dc, dp, dw = motion
        expected = partials.directional(centers=dc, points=dp, weights=dw)
        errors = []
        for h in (1e-3, 3e-4, 1e-4):
            values = []
            for sign in (1, -1):
                moved = [
                    (atom, np.asarray(position) + sign * h * delta)
                    for (atom, position), delta in zip(args["atoms"], dc, strict=True)
                ]
                with NativeAO(**{**args, "atoms": moved}) as basis:
                    jets = basis.evaluate(
                        grid.points + sign * h * dp, energy.contract.ao_order
                    )
                    values.append(
                        energy.evaluate(jets, density, grid.weights + sign * h * dw)[
                            "energy"
                        ]
                    )
            errors.append(abs((values[0] - values[1]) / (2 * h) - expected))
        assert np.all(np.asarray(errors) < [2e-7, 2e-8, 3e-9]), errors
    np.testing.assert_allclose(
        partials.centers.sum(axis=0) + partials.points.sum(axis=0), 0, atol=1e-12
    )


@pytest.mark.parametrize("name", ["LDA_XC_PW", "PBE"])
@pytest.mark.parametrize("spin", ["polarized", "unpolarized"])
def test_xc_potential_geometry_jvp_matches_displaced_potential(
    name: str, spin: str
) -> None:
    meta, data, grid = fixture("h2")
    args = basis_arguments(meta)
    density = data["density_spin" if spin == "polarized" else "density_total"]
    geometry = program(name, spin, "geometry")
    potential = program(name, spin, "potential")
    rng = np.random.default_rng(180179 if spin == "polarized" else 180180)
    delta_density = rng.normal(size=density.shape) * 0.0015
    delta_density = 0.5 * (delta_density + np.swapaxes(delta_density, -1, -2))
    center_motion = rng.normal(size=(len(args["atoms"]), 3)) * 0.025
    point_motion = rng.normal(size=grid.points.shape) * 0.018
    weight_motion = rng.normal(size=grid.weights.shape) * 1.5e-4

    with NativeAO(**args) as basis:
        counts = [
            2 * shell.angular_momentum + 1
            if basis.representation == "real_spherical"
            else (shell.angular_momentum + 1) * (shell.angular_momentum + 2) // 2
            for shell in basis.shells
        ]
        ao_atoms = np.repeat([shell.atom_index for shell in basis.shells], counts)
        full_jets = basis.evaluate(
            grid.points, geometry.contract.ingredients.ao_order + 1
        )
        actual = geometry.potential_geometry_directional(
            full_jets,
            density,
            grid.weights,
            ao_atoms=ao_atoms,
            center_motion=center_motion,
            point_motion=point_motion,
            weight_motion=weight_motion,
            delta_density=delta_density,
        )

    errors = []
    for step in (2e-4, 7e-5, 2e-5):
        values = []
        for sign in (1, -1):
            moved_atoms = [
                (
                    atom,
                    np.asarray(position) + sign * step * delta,
                )
                for (atom, position), delta in zip(
                    args["atoms"], center_motion, strict=True
                )
            ]
            with NativeAO(**{**args, "atoms": moved_atoms}) as basis:
                moved_jets = basis.evaluate(
                    grid.points + sign * step * point_motion,
                    potential.contract.ao_order,
                )
                values.append(
                    potential.evaluate(
                        moved_jets,
                        density + sign * step * delta_density,
                        grid.weights + sign * step * weight_motion,
                    )["potential"]
                )
        fd = (values[0] - values[1]) / (2 * step)
        errors.append(float(np.max(np.abs(fd - actual))))
    assert max(errors) < 2e-9, errors
    np.testing.assert_allclose(actual, actual.swapaxes(-1, -2), atol=2e-13, rtol=0)


@pytest.mark.parametrize("name", ["LDA_XC_PW", "PBE"])
@pytest.mark.parametrize("spin", ["polarized", "unpolarized"])
def test_mixed_xc_geometry_matches_directional_derivative_of_analytic_gradient(
    name: str, spin: str
) -> None:
    meta, data, grid = fixture("h2")
    args = basis_arguments(meta)
    density = data["density_spin" if spin == "polarized" else "density_total"]
    geometry = program(name, spin, "geometry")
    rng = np.random.default_rng(180236 if spin == "polarized" else 180237)
    delta_density = rng.normal(size=density.shape) * 0.002
    delta_density = 0.5 * (delta_density + np.swapaxes(delta_density, -1, -2))
    left_centers = rng.normal(size=(len(args["atoms"]), 3)) * 0.03
    right_centers = rng.normal(size=left_centers.shape) * 0.025
    left_points = rng.normal(size=grid.points.shape) * 0.02
    right_points = rng.normal(size=grid.points.shape) * 0.018
    left_weights = rng.normal(size=grid.weights.shape) * 2e-4
    right_weights = rng.normal(size=grid.weights.shape) * 1.5e-4
    mixed_weights = rng.normal(size=grid.weights.shape) * 4e-5

    with NativeAO(**args) as basis:
        counts = [
            2 * shell.angular_momentum + 1
            if basis.representation == "real_spherical"
            else (shell.angular_momentum + 1) * (shell.angular_momentum + 2) // 2
            for shell in basis.shells
        ]
        ao_atoms = np.repeat([shell.atom_index for shell in basis.shells], counts)
        full_jets = basis.evaluate(
            grid.points, geometry.contract.ingredients.ao_order + 2
        )
        actual = geometry.mixed_geometry_directional(
            full_jets,
            density,
            grid.weights,
            ao_atoms=ao_atoms,
            left_centers=left_centers,
            left_points=left_points,
            left_weights=left_weights,
            right_centers=right_centers,
            right_points=right_points,
            right_weights=right_weights,
            mixed_weights=mixed_weights,
            delta_density=delta_density,
        )

    errors = []
    for step in (2e-4, 7e-5, 2e-5):
        directional = []
        for sign in (1, -1):
            moved_atoms = [
                (
                    atom,
                    np.asarray(position) + sign * step * delta,
                )
                for (atom, position), delta in zip(
                    args["atoms"], right_centers, strict=True
                )
            ]
            with NativeAO(**{**args, "atoms": moved_atoms}) as basis:
                moved_jets = basis.evaluate(
                    grid.points + sign * step * right_points,
                    geometry.contract.ao_order,
                )
                partials = geometry.evaluate(
                    moved_jets,
                    density + sign * step * delta_density,
                    grid.weights + sign * step * right_weights,
                    ao_atoms=ao_atoms,
                    natom=basis.natom,
                )["geometry"]
            directional.append(
                partials.directional(
                    centers=left_centers,
                    points=left_points,
                    weights=left_weights + sign * step * mixed_weights,
                )
            )
        errors.append(
            abs((directional[0] - directional[1]) / (2 * step) - actual.total)
        )
    # These differences reach the floating-point floor already at the
    # coarsest displacement, so monotonic O(h^2) convergence is not a useful
    # gate here. Pin the absolute analytic agreement instead.
    assert max(errors) < 1e-10, errors
    assert abs(actual.feature_mixed) > 1e-8

    frozen = geometry.mixed_geometry_directional(
        full_jets,
        density,
        grid.weights,
        ao_atoms=ao_atoms,
        left_centers=left_centers,
        left_points=left_points,
        left_weights=left_weights,
        right_centers=right_centers,
        right_points=right_points,
        right_weights=right_weights,
        mixed_weights=mixed_weights,
    )
    swapped = geometry.mixed_geometry_directional(
        full_jets,
        density,
        grid.weights,
        ao_atoms=ao_atoms,
        left_centers=right_centers,
        left_points=right_points,
        left_weights=right_weights,
        right_centers=left_centers,
        right_points=left_points,
        right_weights=left_weights,
        mixed_weights=mixed_weights,
    )
    np.testing.assert_allclose(frozen.total, swapped.total, atol=2e-11, rtol=2e-10)


@pytest.mark.parametrize("name", ["LDA_XC_PW", "PBE"])
def test_external_rks_mixed_cartesian_coefficients_match_generated_graph(
    name: str,
) -> None:
    meta, data, grid = fixture("h2")
    args = basis_arguments(meta)
    density = data["density_total"]
    spec = functional(name, spin="unpolarized")
    generated = ContractionProgram(spec, "geometry")
    external = ExternalPointContraction(spec, "geometry")
    second = ContractionProgram(spec, "response")
    rng = np.random.default_rng(180964)
    delta_density = rng.normal(size=density.shape) * 0.002
    delta_density = 0.5 * (delta_density + delta_density.T)
    left_centers = rng.normal(size=(len(args["atoms"]), 3)) * 0.03
    right_centers = rng.normal(size=left_centers.shape) * 0.025
    left_points = rng.normal(size=grid.points.shape) * 0.02
    right_points = rng.normal(size=grid.points.shape) * 0.018
    left_weights = rng.normal(size=grid.weights.shape) * 2e-4
    right_weights = rng.normal(size=grid.weights.shape) * 1.5e-4
    mixed_weights = rng.normal(size=grid.weights.shape) * 4e-5

    with NativeAO(**args) as basis:
        counts = [
            2 * shell.angular_momentum + 1
            if basis.representation == "real_spherical"
            else (shell.angular_momentum + 1) * (shell.angular_momentum + 2) // 2
            for shell in basis.shells
        ]
        ao_atoms = np.repeat([shell.atom_index for shell in basis.shells], counts)
        full_jets = basis.evaluate(
            grid.points, generated.contract.ingredients.ao_order + 2
        )
        expected = generated.mixed_geometry_directional(
            full_jets,
            density,
            grid.weights,
            ao_atoms=ao_atoms,
            left_centers=left_centers,
            left_points=left_points,
            left_weights=left_weights,
            right_centers=right_centers,
            right_points=right_points,
            right_weights=right_weights,
            mixed_weights=mixed_weights,
            delta_density=delta_density,
        )
        _, _, features, right = external.geometry_feature_direction(
            full_jets,
            density,
            ao_atoms=ao_atoms,
            center_motion=right_centers,
            point_motion=right_points,
            delta_density=delta_density,
        )

    rows = second.scalar_values(features)
    gradient = second._gradient(rows, len(grid.points))
    base_gradient = features.get("gradient")
    if base_gradient is not None:
        base_gradient = base_gradient.sum(axis=0)
    base_coefficients = external.coefficients.evaluate(base_gradient, gradient)

    right_packed = external.pack_features(right)
    directional_gradient = np.zeros_like(gradient)
    indices = second.contract.ingredients.feature_indices
    for i in indices:
        for j in indices:
            directional_gradient[i] += (
                rows[(min(i, j), max(i, j))] * right_packed[j]
            )
    response_coefficients = second.response_coefficients
    assert response_coefficients is not None
    right_cartesian_gradient = right.get("gradient")
    if right_cartesian_gradient is not None:
        right_cartesian_gradient = right_cartesian_gradient.sum(axis=0)
    directional_coefficients = response_coefficients.evaluate(
        base_gradient,
        gradient,
        delta_gradient=right_cartesian_gradient,
        delta_v=directional_gradient,
    )
    actual = external.mixed_geometry_from_rks_cartesian_coefficients(
        full_jets,
        density,
        grid.weights,
        ao_atoms=ao_atoms,
        left_centers=left_centers,
        left_points=left_points,
        left_weights=left_weights,
        right_centers=right_centers,
        right_points=right_points,
        right_weights=right_weights,
        mixed_weights=mixed_weights,
        energy=rows[()],
        rho_coefficients=base_coefficients["rho"][0],
        directional_rho_coefficients=directional_coefficients["rho"][0],
        gradient_coefficients=base_coefficients.get("gradient", [None])[0],
        directional_gradient_coefficients=directional_coefficients.get(
            "gradient", [None]
        )[0],
        delta_density=delta_density,
    )

    np.testing.assert_allclose(
        [
            actual.mixed_measure,
            actual.left_measure_right_feature,
            actual.right_measure_left_feature,
            actual.feature_mixed,
            actual.total,
        ],
        [
            expected.mixed_measure,
            expected.left_measure_right_feature,
            expected.right_measure_left_feature,
            expected.feature_mixed,
            expected.total,
        ],
        atol=2e-12,
        rtol=2e-11,
    )


def test_contraction_requests_reject_unsupported_axes_domains_and_directions() -> None:
    from vibeqc_compiler.xc.contracts import DerivativeRequest, IngredientContract
    from vibeqc_compiler.xc.spec import UnsupportedXC

    with pytest.raises(UnsupportedXC):
        DerivativeRequest("hessian")
    mgga = IngredientContract(family="mgga")
    assert mgga.feature_indices == tuple(range(7))
    assert mgga.ao_order == 1
    assert IngredientContract(spin="unpolarized", family="mgga").feature_indices == (
        0,
        1,
        2,
    )
    meta, data, grid = fixture("h2")
    with NativeAO(**basis_arguments(meta)) as basis:
        jets = basis.evaluate(grid.points, 1)
    potential, response = program("PBE"), program("PBE", observable="response")
    d = data["density_spin"]
    with pytest.raises(ValueError, match="direction"):
        potential.evaluate(jets, d, grid.weights, delta_density=d)
    with pytest.raises(ValueError, match="map"):
        potential.evaluate(jets, d, grid.weights, ao_atoms=[0, 1])
    with pytest.raises(ValueError, match="domain"):
        potential.evaluate(jets[:1], d, grid.weights)
    invalid = d.copy()
    invalid[0, 0, 1] += 0.1
    with pytest.raises(ValueError, match="symmetric"):
        response.evaluate(jets, d, grid.weights, delta_density=invalid)
    with pytest.raises(UnsupportedXC, match="equal spin"):
        program("PBE", "unpolarized", "response").evaluate(
            jets, data["density_total"], grid.weights, delta_density=d
        )
    with pytest.raises(UnsupportedXC):
        response.evaluate(jets, d * 0, grid.weights, delta_density=d)
    complete = potential.evaluate(jets, d, grid.weights)["energy"]
    partial = sum(
        potential.evaluate(jets, fraction * d, grid.weights)["energy"]
        for fraction in (0.4, 0.6)
    )
    assert abs(complete - partial) > 1e-3  # Nonlinearity must follow full-D reduction.
