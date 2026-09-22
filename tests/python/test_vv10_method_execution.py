"""Executable fixed-density VV10/rVV10 potential/gradient gates for #491 B/C."""

import typing
from fractions import Fraction

import numpy as np
import pytest
from vibeqc._dft_gradient import resolve_nonlocal_nuclear_sources
from vibeqc.fock import FockPlan
from vibeqc.mean_field import FixedDensityMeanField, compile_fixed_density_method
from vibeqc_compiler.dft import (
    ExplicitGrid,
    FixedDensityNonlocalCorrelation,
    GridSpec,
    MolecularGrid,
    NativeAO,
)
from vibeqc_compiler.dft.fixtures import basis_arguments
from vibeqc_compiler.method import (
    MethodSpec,
    NonlocalCorrelationPrimitive,
    UnsupportedMethod,
    original_nonlocal_correlation,
    resolve_method,
)
from vibeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)
from vibeqc_compiler.xc.integration_fixtures import load_integration_fixture


def _pbe_nonlocal_spec(variant: typing.Any) -> typing.Any:
    return MethodSpec(
        f"PBE+{variant}",
        (("GGA_X_PBE", Fraction(1)), ("GGA_C_PBE", Fraction(1))),
        nonlocal_correlation=original_nonlocal_correlation(variant),
    )


@pytest.mark.parametrize("variant", ["vv10", "rvv10"])
@pytest.mark.parametrize(
    ("spin", "density_key"),
    (("unpolarized", "density_total"), ("polarized", "density_spin")),
)
def test_fixed_density_methodir_nonlocal_potential_matches_energy_derivative(
    spin: typing.Any, density_key: typing.Any, variant: typing.Any
) -> None:
    meta, data, grid = load_integration_fixture("h2")
    density = data[density_key]
    graph = resolve_method(_pbe_nonlocal_spec(variant), spin=spin)
    executable = compile_fixed_density_method(graph)
    with (
        NativeAO(**basis_arguments(meta)) as basis,
        FockPlan(basis, executable.fock_spec, device="cpu") as provider,
    ):
        consumer = FixedDensityMeanField.from_method(provider, graph)
        result = consumer.integrate(grid, density, tile_points=7)
        assert result.nonlocal_identity is not None
        assert result.nonlocal_energy != 0.0
        assert result.method_identity == graph.identity

        standalone = FixedDensityNonlocalCorrelation(
            original_nonlocal_correlation(variant)
        ).integrate(basis, grid, density, tile_points=7)
        assert result.nonlocal_energy == pytest.approx(standalone.energy, abs=1e-15)
        if spin == "polarized":
            np.testing.assert_allclose(
                standalone.potential[0], standalone.potential[1], atol=0.0, rtol=0.0
            )

        direction = 0.007 * density
        step = 1e-5
        plus = consumer.integrate(grid, density + step * direction, tile_points=7)
        minus = consumer.integrate(grid, density - step * direction, tile_points=7)
        finite_difference = (plus.energy - minus.energy) / (2.0 * step)
        np.testing.assert_allclose(
            finite_difference,
            np.sum(result.fock * direction),
            atol=4e-8,
            rtol=0.0,
        )


def test_nonlocal_reference_execution_has_explicit_grid_admission_gate() -> None:
    meta, data, grid = load_integration_fixture("h2")
    with NativeAO(**basis_arguments(meta)) as basis:
        executor = FixedDensityNonlocalCorrelation(
            original_nonlocal_correlation("rvv10"),
            max_points=len(grid.points) - 1,
        )
        with pytest.raises(ValueError, match="max_points"):
            executor.integrate(basis, grid, data["density_total"], tile_points=7)


def test_compile_rejects_nonlocal_only_graph_for_mean_field_execution() -> None:
    graph = resolve_method(
        MethodSpec(
            "VV10-only",
            (),
            nonlocal_correlation=original_nonlocal_correlation("vv10"),
        )
    )
    with pytest.raises(UnsupportedMethod, match="semilocal XC"):
        compile_fixed_density_method(graph)


@pytest.mark.parametrize("variant", ["vv10", "rvv10"])
def test_nonlocal_geometry_matches_independent_center_point_weight_difference(
    variant: typing.Any,
) -> None:
    meta, data, grid = load_integration_fixture("h2")
    args = basis_arguments(meta)
    density = data["density_total"]
    rng = np.random.default_rng(491)
    center_direction = rng.normal(size=(2, 3)) * 0.03
    point_direction = rng.normal(size=grid.points.shape) * 0.02
    weight_direction = rng.normal(size=grid.weights.shape) * 2e-4
    executor = FixedDensityNonlocalCorrelation(original_nonlocal_correlation(variant))
    with NativeAO(**args) as basis:
        geometry = executor.geometry(basis, grid, density, tile_points=7)
        predicted = geometry.directional(
            centers=center_direction,
            points=point_direction,
            weights=weight_direction,
        )

    errors = []
    for step in (1e-3, 3e-4, 1e-4):
        displaced = []
        for sign in (1.0, -1.0):
            atoms = [
                (z, np.asarray(position) + sign * step * direction)
                for (z, position), direction in zip(
                    args["atoms"], center_direction, strict=True
                )
            ]
            moved_grid = ExplicitGrid(
                grid.points + sign * step * point_direction,
                grid.weights + sign * step * weight_direction,
                grid.owners,
                {"oracle": "independent-vv10-geometry-fd"},
            )
            with NativeAO(**{**args, "atoms": atoms}) as basis:
                displaced.append(
                    executor.integrate(basis, moved_grid, density, tile_points=7).energy
                )
        errors.append(abs((displaced[0] - displaced[1]) / (2 * step) - predicted))
    assert max(errors) < 8e-10


@pytest.mark.parametrize("variant", ["vv10", "rvv10"])
def test_complete_nonlocal_nuclear_sources_match_rebuilt_molecular_grid(
    variant: typing.Any,
) -> None:
    meta, data, _ = load_integration_fixture("h2")
    args = basis_arguments(meta)
    density = data["density_total"]
    grid_spec = GridSpec(
        radial_points=3,
        angular_polar=2,
        angular_azimuth=4,
        element_radii=((1, 0.8),),
    )
    grid = MolecularGrid(args["atoms"], grid_spec)
    executor = FixedDensityNonlocalCorrelation(
        original_nonlocal_correlation(variant), max_points=100
    )
    primitive = NonlocalCorrelationPrimitive(executor.spec, executor.coefficient)
    direction = np.random.default_rng(4911).normal(size=(2, 3)) * 0.05
    with NativeAO(**args) as basis:
        geometry = executor.geometry(basis, grid, density, tile_points=5)
        integral = executor.integrate(basis, grid, density, tile_points=5)
        components = resolve_nonlocal_nuclear_sources(
            geometry, basis, grid, density, primitive=primitive, tile_points=5
        )
        with pytest.raises(ValueError, match="density identity"):
            resolve_nonlocal_nuclear_sources(
                geometry,
                basis,
                grid,
                density * 1.0001,
                primitive=primitive,
                tile_points=5,
            )
        displaced_atoms = list(args["atoms"])
        z, position = displaced_atoms[0]
        displaced_atoms[0] = (
            z,
            np.asarray(position) + np.array([1e-4, 0.0, 0.0]),
        )
        stale = MolecularGrid(tuple(displaced_atoms), grid_spec)
        with pytest.raises(ValueError, match="grid identity"):
            resolve_nonlocal_nuclear_sources(
                geometry, basis, stale, density, primitive=primitive, tile_points=5
            )
    assert geometry.grid_identity == integral.grid_identity == grid.identity
    assert (
        geometry.quadrature_identity
        == integral.quadrature_identity
        == grid.explicit(max_points=100).identity
    )
    assert tuple(components) == (
        "nonlocal_ao",
        "nonlocal_grid",
        "nonlocal_weight",
    )
    assert all(np.linalg.norm(value) > 1e-6 for value in components.values())
    gradient = sum(components.values())
    plan = StationaryGradientPlan(
        resolve_method(_pbe_nonlocal_spec(variant)),
        StationaryMeanField(SCF_POINT_MODEL),
    )
    assembled = {name: np.zeros_like(gradient) for name in plan.source_names}
    assembled.update(components)
    np.testing.assert_allclose(
        plan.reduce_diagnostic(assembled, atoms=2), gradient, atol=0.0, rtol=0.0
    )
    np.testing.assert_allclose(gradient.sum(axis=0), 0.0, atol=3e-15, rtol=0.0)
    predicted = float(np.sum(gradient * direction))

    estimates = []
    for step in (1e-3, 3e-4, 1e-4):
        energies = []
        for sign in (1.0, -1.0):
            atoms = [
                (z, np.asarray(position) + sign * step * delta)
                for (z, position), delta in zip(args["atoms"], direction, strict=True)
            ]
            moved_grid = MolecularGrid(tuple(atoms), grid_spec)
            with NativeAO(**{**args, "atoms": atoms}) as basis:
                energies.append(
                    executor.integrate(basis, moved_grid, density, tile_points=5).energy
                )
        estimates.append((energies[0] - energies[1]) / (2 * step))
    np.testing.assert_allclose(estimates, predicted, atol=8e-10, rtol=0.0)


def test_nonlocal_geometry_uses_total_density_for_rks_and_uks() -> None:
    meta, data, grid = load_integration_fixture("h2")
    args = basis_arguments(meta)
    executor = FixedDensityNonlocalCorrelation(original_nonlocal_correlation("vv10"))
    with NativeAO(**args) as basis:
        restricted = executor.geometry(
            basis, grid, data["density_total"], tile_points=7
        )
        unrestricted = executor.geometry(
            basis, grid, data["density_spin"], tile_points=7
        )
    for name in ("centers", "points", "weights"):
        np.testing.assert_allclose(
            getattr(restricted, name), getattr(unrestricted, name), atol=2e-15, rtol=0.0
        )
