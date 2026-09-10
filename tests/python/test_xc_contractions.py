"""Independent E/V fixtures and directional checks of compact XC contractions."""

from functools import lru_cache

import numpy as np
import pytest
from vibeqc_compiler.dft import NativeAO
from vibeqc_compiler.dft.fixtures import basis_arguments
from vibeqc_compiler.xc import functional
from vibeqc_compiler.xc.contractions import ContractionProgram
from vibeqc_compiler.xc.integration_fixtures import load_integration_fixture as fixture


@lru_cache(maxsize=16)
def program(name, spin="polarized", observable="potential"):
    return ContractionProgram(functional(name, spin=spin), observable)


@pytest.mark.parametrize("case", ["h2", "f_cartesian", "f_spherical"])
@pytest.mark.parametrize("name", ["LDA_XC_PW", "PBE"])
@pytest.mark.parametrize(
    "layout,spin",
    [("total", "unpolarized"), ("total", "polarized"), ("spin", "polarized")],
)
def test_minimal_contractions_preserve_independent_fixtures(case, name, layout, spin):
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


@pytest.mark.parametrize("name", ["LDA_XC_PW", "PBE"])
def test_spin_resolved_response_finite_differences_transpose_and_exchange(name):
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
def test_explicit_geometry_sources_against_moved_native_collocation(name, spin, case):
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


def test_contraction_requests_reject_unsupported_axes_domains_and_directions():
    from vibeqc_compiler.xc.contracts import DerivativeRequest, IngredientContract
    from vibeqc_compiler.xc.spec import UnsupportedXC

    with pytest.raises(UnsupportedXC):
        DerivativeRequest("hessian")
    with pytest.raises(UnsupportedXC):
        IngredientContract(family="mgga")
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
