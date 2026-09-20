"""Nonlocal geometry cannot be replayed under another current MethodIR kernel."""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
from vibeqc._dft_gradient import resolve_nonlocal_nuclear_sources
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


@pytest.mark.parametrize("variant", ("vv10", "rvv10"))
def test_nonlocal_geometry_rejects_changed_kernel_or_coefficient(variant: str) -> None:
    meta, data, _ = load_integration_fixture("h2")
    args = basis_arguments(meta)
    spec = original_nonlocal_correlation(variant)
    coefficient = Fraction(3, 2)
    primitive = NonlocalCorrelationPrimitive(spec, coefficient)
    grid = MolecularGrid(
        args["atoms"],
        GridSpec(
            radial_points=3,
            angular_polar=2,
            angular_azimuth=4,
            element_radii=((1, 0.8),),
        ),
    )
    with NativeAO(**args) as basis:
        geometry = FixedDensityNonlocalCorrelation(
            spec, coefficient=coefficient, max_points=100
        ).geometry(basis, grid, data["density_total"], tile_points=5)
        accepted = resolve_nonlocal_nuclear_sources(
            geometry,
            basis,
            grid,
            data["density_total"],
            primitive=primitive,
            tile_points=5,
        )
        assert geometry.coefficient == coefficient
        assert all(np.isfinite(value).all() for value in accepted.values())
        other = "rvv10" if variant == "vv10" else "vv10"
        changed_specs = (
            original_nonlocal_correlation(other),
            replace(spec, b=spec.b + Fraction(1)),
            replace(spec, c=spec.c + Fraction(1, 100)),
        )
        for changed in changed_specs:
            with pytest.raises(ValueError, match="specification identity"):
                resolve_nonlocal_nuclear_sources(
                    geometry,
                    basis,
                    grid,
                    data["density_total"],
                    primitive=NonlocalCorrelationPrimitive(changed, coefficient),
                )
        with pytest.raises(ValueError, match="coefficient identity"):
            resolve_nonlocal_nuclear_sources(
                geometry,
                basis,
                grid,
                data["density_total"],
                primitive=NonlocalCorrelationPrimitive(spec, Fraction(1)),
            )
        with pytest.raises(TypeError, match="primitive"):
            resolve_nonlocal_nuclear_sources(
                geometry, basis, grid, data["density_total"]
            )
