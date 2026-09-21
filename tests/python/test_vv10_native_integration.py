"""Native VV10/rVV10 integration gates across MethodIR potential and gradient sources."""

from fractions import Fraction

import numpy as np
import pytest
from vibeqc._dft_gradient import resolve_nonlocal_nuclear_sources
from vibeqc.nonlocal_runtime import NativeNonlocalPairProvider, NonlocalFixedGridPlan
from vibeqc_compiler.dft import (
    FixedDensityNonlocalCorrelation,
    GridSpec,
    MolecularGrid,
    NativeAO,
)
from vibeqc_compiler.dft.fixtures import basis_arguments
from vibeqc_compiler.method import (
    NonlocalCorrelationPrimitive,
    original_nonlocal_correlation,
)
from vibeqc_compiler.xc.integration_fixtures import load_integration_fixture


def _provider_or_skip(device: str) -> NativeNonlocalPairProvider:
    if device == "cuda":
        try:
            with NonlocalFixedGridPlan(
                original_nonlocal_correlation("vv10"),
                1,
                device="cuda",
                device_id=0,
                maximum_bytes=1 << 20,
            ):
                pass
        except (NotImplementedError, RuntimeError) as error:
            pytest.skip(f"CUDA nonlocal provider unavailable: {error}")
    try:
        return NativeNonlocalPairProvider(
            device=device,
            device_id=0,
            memory_budget_bytes=64 << 20,
        )
    except (NotImplementedError, RuntimeError) as error:
        if device == "cuda":
            pytest.skip(f"CUDA nonlocal provider unavailable: {error}")
        raise


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("variant", ("vv10", "rvv10"))
def test_native_fixed_density_geometry_matches_reference_and_rks_uks(
    device: str, variant: str
) -> None:
    meta, data, grid = load_integration_fixture("h2")
    args = basis_arguments(meta)
    spec = original_nonlocal_correlation(variant)
    provider = _provider_or_skip(device)
    native_executor = FixedDensityNonlocalCorrelation(spec, pair_provider=provider)
    reference_executor = FixedDensityNonlocalCorrelation(spec)
    with NativeAO(**args) as basis:
        native = native_executor.geometry(
            basis, grid, data["density_total"], tile_points=7
        )
        reference = reference_executor.geometry(
            basis, grid, data["density_total"], tile_points=5
        )
        native_uks = native_executor.geometry(
            basis, grid, data["density_spin"], tile_points=7
        )
        integral = native_executor.integrate(
            basis, grid, data["density_total"], tile_points=7
        )

    assert native.backend == integral.backend == f"native-{device}"
    assert native.provider_identity == integral.provider_identity == provider.identity
    assert native_uks.provider_identity == provider.identity
    assert native.pair_evaluations == len(grid.points) ** 2
    if device == "cuda":
        assert native.device_workspace_bytes > 0
    else:
        assert native.device_workspace_bytes == 0
    for name in ("centers", "points", "weights"):
        np.testing.assert_allclose(
            getattr(native, name),
            getattr(reference, name),
            rtol=3e-12,
            atol=3e-13,
        )
        np.testing.assert_allclose(
            getattr(native_uks, name),
            getattr(native, name),
            rtol=3e-12,
            atol=3e-13,
        )


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("variant", ("vv10", "rvv10"))
def test_native_complete_gradient_matches_rebuilt_grid_multistep_fd(
    device: str, variant: str
) -> None:
    meta, data, _ = load_integration_fixture("h2")
    args = basis_arguments(meta)
    density = data["density_total"]
    spec = original_nonlocal_correlation(variant)
    grid_spec = GridSpec(
        radial_points=3,
        angular_polar=2,
        angular_azimuth=4,
        element_radii=((1, 0.8),),
    )
    grid = MolecularGrid(args["atoms"], grid_spec)
    provider = _provider_or_skip(device)
    executor = FixedDensityNonlocalCorrelation(spec, pair_provider=provider)
    primitive = NonlocalCorrelationPrimitive(spec, Fraction(1))
    direction = np.random.default_rng(4912).normal(size=(2, 3)) * 0.04

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
        displaced_atoms = [
            (z, np.asarray(position) + 1.0e-4 * delta)
            for (z, position), delta in zip(args["atoms"], direction, strict=True)
        ]
        stale_grid = MolecularGrid(tuple(displaced_atoms), grid_spec)
        with pytest.raises(ValueError, match="grid identity"):
            resolve_nonlocal_nuclear_sources(
                geometry,
                basis,
                stale_grid,
                density,
                primitive=primitive,
                tile_points=5,
            )

    assert geometry.backend == integral.backend == f"native-{device}"
    assert geometry.provider_identity == integral.provider_identity == provider.identity
    assert tuple(components) == (
        "nonlocal_ao",
        "nonlocal_grid",
        "nonlocal_weight",
    )
    gradient_force = sum(components.values())
    predicted = float(np.sum(gradient_force * direction))

    estimates = []
    identities = []
    for step in (1e-3, 3e-4, 1e-4):
        energies = []
        for sign in (1.0, -1.0):
            atoms = [
                (z, np.asarray(position) + sign * step * delta)
                for (z, position), delta in zip(args["atoms"], direction, strict=True)
            ]
            moved_grid = MolecularGrid(tuple(atoms), grid_spec)
            with NativeAO(**{**args, "atoms": atoms}) as moved_basis:
                moved = executor.integrate(
                    moved_basis, moved_grid, density, tile_points=5
                )
            energies.append(moved.energy)
            identities.append(moved.identity)
        estimates.append((energies[0] - energies[1]) / (2.0 * step))

    assert len(set(identities)) == len(identities)
    np.testing.assert_allclose(estimates, predicted, atol=1.2e-9, rtol=0.0)
    np.testing.assert_allclose(gradient_force.sum(axis=0), 0.0, atol=4e-14, rtol=0.0)


def test_native_ragged_batch_is_bounded_and_atomic() -> None:
    provider = _provider_or_skip("cpu")
    points = np.array(
        [[0.0, 0.0, 0.0], [0.6, 0.1, -0.2], [-0.2, 0.7, 0.3], [0.9, -0.4, 0.5]],
        dtype=np.float64,
    )
    weights = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
    density = np.array([0.4, 0.3, 0.2, 0.25], dtype=np.float64)
    gradient = np.full((4, 3), 0.01, dtype=np.float64)
    spec = original_nonlocal_correlation("rvv10")
    small = (
        points[:2],
        weights[:2],
        density[:2],
        gradient[:2],
        spec,
        Fraction(1),
    )
    large = (points, weights, density, gradient, spec, Fraction(2, 3))

    batch = provider.evaluate_batch((small, large), tile_points=2, geometry=True)
    assert tuple(len(result.vrho) for result in batch.results) == (2, 4)
    assert batch.pair_evaluations == 2**2 + 4**2
    assert batch.peak_owned_workspace_bytes == 12 * 4 * 8

    poisoned_density = density[:3].copy()
    poisoned_density[1] = 0.0
    poisoned = (
        points[:3],
        weights[:3],
        poisoned_density,
        gradient[:3],
        spec,
        Fraction(1),
    )
    with pytest.raises(RuntimeError, match="ragged member 1 failed"):
        provider.evaluate_batch((small, poisoned, large), tile_points=2)
    replay = provider.evaluate_batch((large,), tile_points=2)
    assert replay.results[0].energy == pytest.approx(batch.results[1].energy)
