"""Independent fixed-D energy/matrix gates and discriminating factor regressions."""

from dataclasses import replace
from fractions import Fraction
from functools import lru_cache

import numpy as np
import pytest
from vibeqc_compiler.dft import (
    ExplicitGrid,
    GridSpec,
    MolecularGrid,
    NativeAO,
    density_features,
)
from vibeqc_compiler.dft.fixtures import basis_arguments
from vibeqc_compiler.xc import FixedDensityXC, UnsupportedXC, functional
from vibeqc_compiler.xc.integration_fixtures import CASES
from vibeqc_compiler.xc.integration_fixtures import load_integration_fixture as fixture
from vibeqc_compiler.xc.potential import assemble_potential


def check(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-10)


@lru_cache(maxsize=4)
def consumer(name, spin="polarized"):
    return FixedDensityXC(functional(name, spin=spin))


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("name", ["LDA_XC_PW", "PBE"])
@pytest.mark.parametrize(
    "layout,spin",
    [("total", "unpolarized"), ("total", "polarized"), ("spin", "polarized")],
)
def test_identical_grid_independent_energy_and_every_matrix_element(
    case, name, layout, spin
):
    meta, data, grid = fixture(case)
    density = data[f"density_{layout}"]
    prefix = f"{name}_{layout}"
    with NativeAO(**basis_arguments(meta)) as basis:
        result = consumer(name, spin).integrate(basis, grid, density, tile_points=7)
    check(result.energy, data[f"{prefix}_energy"][0])
    check(result.potential, data[f"{prefix}_potential"])
    electrons = result.electrons.sum() if layout == "total" else result.electrons
    check(electrons, data[f"{prefix}_electrons"])
    assert result.potential.shape == density.shape
    check(result.potential, result.potential.swapaxes(-1, -2))
    assert (
        result.points == len(grid.weights)
        and result.tiles == (len(grid.weights) + 6) // 7
    )
    assert result.backend == "cpu"
    assert not result.potential.flags.writeable


@pytest.mark.parametrize("name", ["LDA_XC_PW", "PBE"])
@pytest.mark.parametrize(
    "layout,spin",
    [("total", "unpolarized"), ("total", "polarized"), ("spin", "polarized")],
)
def test_trace_variation_diagonal_offdiagonal_and_mixed_spins(name, layout, spin):
    meta, data, grid = fixture("water")
    density = data[f"density_{layout}"]
    integrator = consumer(name, spin)
    directions = []
    for a, b in ((0, 0), (1, 4)):
        for channel in range(2) if layout == "spin" else (None,):
            direction = np.zeros_like(density)
            matrix = direction if channel is None else direction[channel]
            matrix[a, b] = matrix[b, a] = 1
            directions.append(direction)
    if layout == "spin":
        mixed = directions[2] - 0.7 * directions[3]
        directions.append(mixed)
    with NativeAO(**basis_arguments(meta)) as basis:
        result = integrator.integrate(basis, grid, density, tile_points=7)
        for direction in directions:
            expected = np.sum(result.potential * direction.swapaxes(-1, -2))
            errors = []
            for step in (1e-3, 3e-4, 1e-4):
                plus = integrator.integrate(
                    basis, grid, density + step * direction, tile_points=11
                )
                minus = integrator.integrate(
                    basis, grid, density - step * direction, tile_points=11
                )
                fd = (plus.energy - minus.energy) / (2 * step)
                errors.append(abs(fd - expected))
            # Check every step, including its second-order truncation bound;
            # the small-step target does not select the most favorable step.
            assert np.all(np.asarray(errors) < [2e-5, 2e-6, 3e-7]), errors
            assert errors[-1] < errors[0] / 20 + 2e-10, errors


@pytest.mark.parametrize("name", ["LDA_XC_PW", "PBE"])
def test_tile_boundaries_weight_linearity_and_spin_layouts(name):
    meta, data, grid = fixture("h2")
    density = data["density_total"]
    with NativeAO(**basis_arguments(meta)) as basis:
        reference = consumer(name, "unpolarized").integrate(basis, grid, density)
        for tile_points in (1, 7, 16, 31, 32, 33):
            result = consumer(name).integrate(
                basis, grid, density, tile_points=tile_points
            )
            check(result.energy, reference.energy)
            check(result.potential, reference.potential)
        equal = consumer(name, "unpolarized").integrate(
            basis, grid, np.stack((density / 2, density / 2))
        )
        check(equal.energy, reference.energy)
        check(equal.potential, np.stack((reference.potential, reference.potential)))
        unequal = data["density_spin"]
        first = consumer(name).integrate(basis, grid, unequal)
        swapped = consumer(name).integrate(basis, grid, unequal[::-1])
        check(swapped.energy, first.energy)
        check(swapped.potential, first.potential[::-1])
        scaled_grid = replace(grid, weights=grid.weights * 2.7)
        scaled = consumer(name, "unpolarized").integrate(basis, scaled_grid, density)
        check(scaled.energy, 2.7 * reference.energy)
        check(scaled.potential, 2.7 * reference.potential)
        assert scaled.identity != reference.identity
        assert not np.isclose(scaled.energy, 2.7**2 * reference.energy)


def test_assembly_tau_half_and_cross_spin_factors_against_direct_bilinears():
    rng = np.random.default_rng(162)
    jets = rng.normal(size=(4, 9, 3))
    d = np.stack((np.eye(3), 0.6 * np.eye(3)))
    features = density_features(jets, d)
    v = rng.normal(size=(7, 9))
    w = rng.uniform(0.1, 2, size=9)
    spec = functional("PBE")
    matrix = assemble_potential(spec, jets, features["gradient"], v, w)
    direction = rng.normal(size=d.shape)
    direction = (direction + direction.swapaxes(1, 2)) / 2

    def energy(dm):
        f = density_features(jets, dm)
        return np.sum(
            w * np.sum(v * np.concatenate((f["rho"], f["sigma"], f["tau"])), axis=0)
        )

    for h in (1e-3, 1e-4, 1e-5):
        np.testing.assert_allclose(
            (energy(d + h * direction) - energy(d - h * direction)) / (2 * h),
            np.sum(matrix * direction),
            atol=2e-7,
            rtol=1e-9,
        )


def test_geometry_grid_basis_density_and_functional_changes_do_not_reuse_results():
    meta, data, explicit = fixture("h2")
    args = basis_arguments(meta)
    dm = data["density_total"]
    integrator = consumer("PBE", "unpolarized")
    with NativeAO(**args) as basis:
        grid = MolecularGrid(basis.atoms, GridSpec(2, 2, 4))
        original = integrator.integrate(basis, grid, dm)
        changed_grid = replace(grid, spec=GridSpec(2, 3, 5))
        changed = integrator.integrate(basis, changed_grid, dm)
        assert changed.identity != original.identity
        assert abs(changed.energy - original.energy) > 1e-7
        check(integrator.integrate(basis, grid, dm).energy, original.energy)
        assert integrator.integrate(basis, grid, dm * 1.2).identity != original.identity
        assert (
            consumer("LDA_XC_PW").integrate(basis, grid, dm).identity
            != original.identity
        )
        other_args = {
            **args,
            "atoms": [
                (z, np.asarray(xyz) + ([0.2, 0, 0] if i else [0, 0, 0]))
                for i, (z, xyz) in enumerate(args["atoms"])
            ],
        }
        with NativeAO(**other_args) as moved:
            with pytest.raises(ValueError, match="stale molecular grid"):
                integrator.integrate(moved, grid, dm)
            moved_grid = MolecularGrid(moved.atoms, grid.spec)
            fresh = integrator.integrate(moved, moved_grid, dm)
            assert (
                fresh.identity != original.identity
                and abs(fresh.energy - original.energy) > 1e-7
            )
        shells = list(args["basis"])
        shells[0] = replace(
            shells[0],
            primitives=tuple(
                replace(p, exponent=p.exponent * 1.1) for p in shells[0].primitives
            ),
        )
        with NativeAO(**{**args, "basis": shells}) as changed_basis:
            new = integrator.integrate(changed_basis, explicit, dm)
            old = integrator.integrate(basis, explicit, dm)
            assert (
                new.basis_identity != old.basis_identity
                and abs(new.energy - old.energy) > 1e-7
            )


def test_unsupported_domain_invalid_density_and_empty_grid_are_explicit():
    meta, data, grid = fixture("h2")
    with NativeAO(**basis_arguments(meta)) as basis:
        d = data["density_total"]
        for bad in (np.zeros_like(d), d * 1e-20, -d):
            with pytest.raises(UnsupportedXC, match="XC tile starting"):
                consumer("PBE").integrate(basis, grid, bad, tile_points=7)
        with pytest.raises(UnsupportedXC, match="equal spin matrices"):
            consumer("PBE", "unpolarized").integrate(basis, grid, data["density_spin"])
        with pytest.raises(ValueError, match="symmetric"):
            consumer("PBE").integrate(basis, grid, d + [[0, 0.1], [0, 0]])
        for bad in (d.astype(complex) + 1j, d * np.nan, np.ones((3, 2, 2))):
            with pytest.raises(ValueError):
                consumer("PBE").integrate(basis, grid, bad)
        for size in (0, -1, True, 1.5):
            with pytest.raises(ValueError):
                consumer("PBE").integrate(basis, grid, d, tile_points=size)
        tail = ExplicitGrid([[1e3, 0, 0]], [0.0], (0,), {})
        with pytest.raises(UnsupportedXC):
            consumer("PBE").integrate(basis, tail, d)
        empty = ExplicitGrid(np.empty((0, 3)), [], (), {})
        result = consumer("PBE").integrate(basis, empty, d)
        assert result.energy == 0 and result.points == result.tiles == 0
        assert np.count_nonzero(result.potential) == 0


@pytest.mark.parametrize("fault", ["double_weight", "spin_factor", "double_potential"])
def test_independent_gate_rejects_deliberate_factor_faults(monkeypatch, fault):
    from vibeqc_compiler.xc import integration

    meta, data, grid = fixture("water")
    if fault == "double_weight":
        original = integration._tiles

        def wrong_tiles(grid, tile_points):
            for tile in original(grid, tile_points):
                yield replace(tile, weights=tile.weights**2)

        monkeypatch.setattr(integration, "_tiles", wrong_tiles)
    elif fault == "spin_factor":
        original = integration.spin_densities
        monkeypatch.setattr(
            integration, "spin_densities", lambda d, n: 2 * original(d, n)
        )
    else:
        from vibeqc_compiler.xc import contractions

        original = contractions.assemble_coefficients
        monkeypatch.setattr(
            contractions, "assemble_coefficients", lambda *args: 2 * original(*args)
        )
    with NativeAO(**basis_arguments(meta)) as basis:
        bad = consumer("PBE").integrate(
            basis, grid, data["density_spin"], tile_points=7
        )
    with pytest.raises(AssertionError):
        check(bad.potential, data["PBE_spin_potential"])
    if fault != "double_potential":
        with pytest.raises(AssertionError):
            check(bad.energy, data["PBE_spin_energy"][0])


def test_consumer_rejects_nonsemilocal_metadata_and_assembly_shapes():
    with pytest.raises(TypeError, match="FunctionalSpec"):
        FixedDensityXC("PBE")
    with pytest.raises(UnsupportedXC, match="semilocal"):
        FixedDensityXC(replace(functional("PBE"), exact_exchange=Fraction(1, 4)))
    spec = functional("PBE")
    jets, gradient, v, w = (
        np.ones((4, 3, 2)),
        np.ones((2, 3, 3)),
        np.ones((7, 3)),
        np.ones(3),
    )
    for bad_jets, bad_v, bad_w in (
        (jets[0], v, w),
        (jets, v[:, :2], w),
        (jets, v, w[:2]),
    ):
        with pytest.raises(ValueError):
            assemble_potential(spec, bad_jets, gradient, bad_v, bad_w)
