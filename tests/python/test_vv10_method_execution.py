"""Executable fixed-density VV10/rVV10 potential gates for #491 slice B."""

import typing
from fractions import Fraction

import numpy as np
import pytest
from vibeqc.fock import FockPlan
from vibeqc.mean_field import FixedDensityMeanField, compile_fixed_density_method
from vibeqc_compiler.dft import FixedDensityNonlocalCorrelation, NativeAO
from vibeqc_compiler.dft.fixtures import basis_arguments
from vibeqc_compiler.method import (
    MethodSpec,
    UnsupportedMethod,
    original_nonlocal_correlation,
    resolve_method,
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
