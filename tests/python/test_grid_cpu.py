"""Independent spatial-jet, quadrature, spin-factor and ownership gates."""

import json
from dataclasses import replace

import numpy as np
import pytest

from tools.vibeqc_dft import (
    ExplicitGrid,
    GridSpec,
    MolecularGrid,
    NativeAO,
    density_features,
    jet_indices,
    orbital_features,
    partition_weights,
)
from tools.vibeqc_dft.fixtures import NAMES, ROOT, basis_arguments, load_fixture
from tools.vibeqc_validation.schema import block_error


def check(actual, expected):
    result = block_error(actual, expected, atol=1e-11, rtol=1e-10)
    assert result["passed"], result


@pytest.mark.parametrize("name", NAMES)
def test_every_jet_and_spin_feature_against_libcint(name):
    meta, arrays = load_fixture(name)
    explicit = ExplicitGrid.read(ROOT / f"{name}-grid.json")
    check(explicit.points, arrays["partitioned_grid_points"])
    check(explicit.weights, arrays["partitioned_grid_weights"])
    with NativeAO(**basis_arguments(meta)) as basis:
        jets = basis.evaluate(arrays["points"], 3)
        check(jets, arrays["ao_jets"])
        feature = density_features(jets, arrays["density"])
        orbital = orbital_features(jets, arrays["coefficients"], arrays["occupations"])
        for key in feature:
            check(feature[key], arrays[key])
            check(orbital[key], arrays[key])
        # Independent Laplacian contraction exercises mixed second derivatives
        # in the pinned jet dictionary, without a generated derivative oracle.
        lap_ao = jets[4] + jets[7] + jets[9]
        lap = []
        for d in arrays["density"]:
            lap.append(
                2 * np.sum(lap_ao * (jets[0] @ d), axis=1)
                + 2 * sum(np.sum(jets[k] * (jets[k] @ d), axis=1) for k in range(1, 4))
            )
        check(lap, arrays["laplacian"])
        # Arbitrary partial AO and point tiles, including actual f components.
        for begin in range(0, basis.nao, 3):
            count = min(3, basis.nao - begin)
            check(
                basis.evaluate(
                    arrays["points"][-7:], 3, ao_begin=begin, ao_count=count
                ),
                jets[:, -7:, begin : begin + count],
            )
        assert basis.evaluate(np.empty((0, 3)), 3).shape == (20, 0, basis.nao)
        assert basis.evaluate(arrays["points"][:1], 2, ao_begin=basis.nao).shape == (
            10,
            1,
            0,
        )


def test_jet_dictionary_and_spatial_center_chain_rule():
    assert jet_indices(1) == ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    meta, data = load_fixture("f_spherical")
    args = basis_arguments(meta)
    points = data["points"][-19:-14]
    with NativeAO(**args) as basis:
        full = basis.evaluate(points, 3)
        for axis in range(3):
            errors = []
            for step in (2e-3, 2e-4, 2e-5):
                shift = np.eye(3)[axis] * step
                fd = (
                    basis.evaluate(points + shift, 2)
                    - basis.evaluate(points - shift, 2)
                ) / (2 * step)
                expected = full[
                    [
                        jet_indices(3).index(
                            tuple(p + int(k == axis) for k, p in enumerate(index))
                        )
                        for index in jet_indices(2)
                    ]
                ]
                errors.append(np.max(abs(fd - expected)))
            assert errors[-1] < 2e-8 and errors[-1] < errors[0] / 100
        shifted_atoms = [
            (z, np.asarray(xyz) + [1e-5, 0, 0]) for z, xyz in args["atoms"]
        ]
        with NativeAO(**{**args, "atoms": shifted_atoms}) as shifted:
            check(shifted.evaluate(points + [1e-5, 0, 0], 3), full)


def test_partition_unity_coincidence_extremes_and_permutation():
    points = np.array(
        [[0, 0, 0], [0.1, -0.2, 0.5], [100, -200, 300], [1e12, 2e12, -1e12]]
    )
    for centers in (
        np.array([[0, 0, 0], [0, 0, 0], [0.4, 0.2, 0.1]]),
        np.array([[0, 0, 0], [1e-14, 0, 0], [1e9, 0, 0]]),
        np.array([[0, 0, 0], [1e-8, 0, 0], [0.4, 0.2, 0.1]]),
    ):
        weights = partition_weights(points, centers)
        assert np.isfinite(weights).all() and np.min(weights) >= 0
        np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-15)
        order = [2, 0, 1]
        np.testing.assert_allclose(
            partition_weights(points, centers[order]), weights[:, order], atol=2e-15
        )
    same = partition_weights(points, np.zeros((3, 3)))
    np.testing.assert_allclose(same, 1 / 3, atol=1e-15)


@pytest.mark.parametrize("alpha", [0.01, 1, 100])
def test_known_radial_integrals_without_renormalization(alpha):
    radius = 1 / np.sqrt(alpha)
    grid = MolecularGrid((("He", (0, 0, 0)),), GridSpec(96, 8, 16, ((2, radius),)))
    gaussian = sum(
        np.dot(t.weights, np.exp(-alpha * np.sum(t.points * t.points, axis=1)))
        for t in grid.tiles(127)
    )
    assert abs(gaussian / (np.pi / alpha) ** 1.5 - 1) < 2e-12
    exponent = np.sqrt(alpha)
    slater = sum(
        np.dot(t.weights, np.exp(-exponent * np.linalg.norm(t.points, axis=1)))
        for t in grid.tiles(131)
    )
    assert abs(slater / (8 * np.pi / exponent**3) - 1) < 2e-12
    w, d = grid.angular_weights, grid.directions
    np.testing.assert_allclose(np.sum(w), 4 * np.pi, atol=1e-14)
    np.testing.assert_allclose(w @ (d * d), np.full(3, 4 * np.pi / 3), atol=1e-14)


@pytest.mark.parametrize(
    "name,radius", [("water", 1), ("f_spherical", 1), ("diffuse", 10), ("tight", 0.04)]
)
def test_electron_integral_converges_to_overlap_trace(name, radius):
    meta, data = load_fixture(name)
    args = basis_arguments(meta)
    total = data["density"].sum(axis=0)
    exact = np.einsum("ij,ji->", total, data["overlap"])
    errors = []
    radii = tuple((z, radius) for z in sorted(set(meta["inputs"]["atomic_numbers"])))
    with NativeAO(**args) as basis:
        for radial, polar in ((16, 8), (32, 12), (64, 20)):
            grid = MolecularGrid(
                basis.atoms,
                GridSpec(radial, polar, 2 * polar, radii),
                multiplicity=args["multiplicity"],
            )
            result = 0.0
            for tile in grid.tiles(251):
                value = basis.evaluate(tile.points, 0)[0]
                result += np.dot(tile.weights, np.sum(value * (value @ total), axis=1))
            errors.append(abs(result - exact))
    assert errors[-1] < 1e-5 and errors[-1] < errors[0], (name, errors)


def test_explicit_grid_hash_ownership_and_round_trip(tmp_path):
    grid = ExplicitGrid.read(ROOT / "h2-grid.json")
    path = tmp_path / "grid.json"
    grid.write(path)
    assert ExplicitGrid.read(path).identity == grid.identity
    raw = json.loads(path.read_text())
    raw["weights_bohr3"][0] += 1
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="hash mismatch"):
        ExplicitGrid.read(path)
    with pytest.raises(ValueError):
        grid.points.setflags(write=True)
    with pytest.raises(ValueError):
        MolecularGrid((("H", (0, 0, 0)),)).explicit(max_points=10)


def test_grid_identity_motion_and_atom_order():
    atoms = (("H", (0, 0, 0)), ("He", (0.3, 0.2, 1.4)))
    grid = MolecularGrid(atoms, GridSpec(3, 3, 6), multiplicity=2)
    changed = replace(grid, atoms=(("H", (0.1, 0, 0)), atoms[1]))
    assert changed.identity != grid.identity
    assert not np.array_equal(changed.explicit().weights, grid.explicit().weights)
    moved = replace(
        grid, atoms=tuple((z, np.array(p) + [0.2, 0.3, 0.4]) for z, p in atoms)
    )
    np.testing.assert_allclose(
        moved.explicit().points, grid.explicit().points + [0.2, 0.3, 0.4], atol=1e-14
    )
    np.testing.assert_allclose(
        moved.explicit().weights, grid.explicit().weights, atol=1e-13
    )
    reordered = replace(grid, atoms=atoms[::-1]).explicit()
    original = grid.explicit()
    half = len(original.points) // 2
    np.testing.assert_allclose(
        reordered.weights, np.roll(original.weights, half), atol=1e-13
    )
    assert (
        len(
            {
                grid.identity,
                replace(grid, charge=1).identity,
                replace(grid, multiplicity=1).identity,
                replace(grid, spec=replace(grid.spec, partition_iterations=2)).identity,
            }
        )
        == 4
    )


def test_density_factor_validation_and_native_lifetime():
    meta, data = load_fixture("h2")
    basis = NativeAO(**basis_arguments(meta))
    points = data["points"][:3]
    jets = basis.evaluate(points)
    total = data["density"].sum(axis=0)
    a = density_features(jets, total)
    b = density_features(jets, np.stack((total / 2, total / 2)))
    for key in a:
        check(a[key], b[key])
    negative = density_features(jets, -np.eye(basis.nao))
    assert np.all(negative["rho"] < 0)
    with pytest.raises(ValueError, match="symmetric"):
        density_features(jets, [[1, 2], [0, 1]])
    with pytest.raises(ValueError):
        density_features(jets, np.eye(basis.nao) * 1j)
    with pytest.raises(ValueError, match="budget"):
        basis.evaluate(points, 3, budget_bytes=1)
    with pytest.raises(ValueError):
        basis.evaluate(points, 4)
    with pytest.raises(AttributeError):
        basis.identity = "changed"
    basis.close()
    with pytest.raises(RuntimeError, match="closed"):
        basis.evaluate(points)
    assert np.isfinite(jets).all()


@pytest.mark.parametrize(
    "changes",
    [
        {"radial_points": True},
        {"angular_azimuth": 2},
        {"pruning": "unknown"},
        {"units": "Angstrom"},
        {"element_radii": ((1, -1),)},
        {"element_radii": ((1, 1), (1, 2))},
    ],
)
def test_unsupported_grid_rules_fail(changes):
    with pytest.raises(ValueError):
        GridSpec(**changes)
