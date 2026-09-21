"""Self-consistent native VV10/rVV10 execution through the generic KS plan (#491)."""

from fractions import Fraction

import numpy as np
import pytest
from vibeqc import Calculator, GridSpec, KsOptions, estimate_ks_resources
from vibeqc._dft_gradient import StationaryKsState
from vibeqc.fock import FockPlan
from vibeqc.mean_field import FixedDensityMeanField, compile_fixed_density_method
from vibeqc_compiler.dft import NativeAO
from vibeqc_compiler.method import (
    MethodIR,
    MethodSpec,
    original_nonlocal_correlation,
    resolve_method,
)

H2 = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
GRID = GridSpec(radial_points=3, angular_polar=2, angular_azimuth=4)


def _graph(variant: str, spin: str) -> MethodIR:
    return resolve_method(
        MethodSpec(
            f"PBE+{variant}",
            (("GGA_X_PBE", Fraction(1)), ("GGA_C_PBE", Fraction(1))),
            nonlocal_correlation=original_nonlocal_correlation(variant),
        ),
        spin=spin,
    )


def _options(graph: MethodIR) -> KsOptions:
    return KsOptions(
        composition=graph,
        grid=GRID,
        tile_points=16,
        nonlocal_memory_budget_bytes=1 << 20,
    )


@pytest.mark.parametrize("variant", ("vv10", "rvv10"))
def test_rks_self_consistent_nonlocal_matches_independent_fixed_density_owner(
    variant: str,
) -> None:
    graph = _graph(variant, "unpolarized")
    calculator = Calculator(
        method=graph,
        basis="sto-3g",
        device="cpu",
        ks_options=KsOptions(
            grid=GRID,
            tile_points=16,
            nonlocal_memory_budget_bytes=1 << 20,
        ),
        max_iterations=100,
    )
    with calculator.prepare_batch([H2]) as batch, NativeAO(H2) as basis:
        native = batch.execute(strict=True).items[0]
        state = StationaryKsState.from_native(batch, basis)
        density = state.density[0]
        physical_fock = state.fock[0]
        plan = compile_fixed_density_method(graph)
        with FockPlan(basis, plan.fock_spec, device="cpu") as provider:
            fixed = FixedDensityMeanField.from_method(
                provider, graph, nonlocal_memory_budget_bytes=1 << 20
            ).integrate(state.grid, density, tile_points=16)

    assert native.converged
    assert state._source.method_ir.identity == graph.identity
    assert fixed.nonlocal_identity is not None
    assert abs(fixed.nonlocal_energy) > 1e-12
    assert native.energy == pytest.approx(fixed.energy, abs=2e-13)
    np.testing.assert_allclose(physical_fock, fixed.fock, atol=8e-13, rtol=0.0)


@pytest.mark.parametrize("variant", ("vv10", "rvv10"))
def test_uks_self_consistent_nonlocal_converges_on_live_total_density(
    variant: str,
) -> None:
    graph = _graph(variant, "polarized")
    corrected = Calculator(
        method=graph,
        basis="sto-3g",
        device="cpu",
        ks_options=KsOptions(
            grid=GRID,
            tile_points=16,
            nonlocal_memory_budget_bytes=1 << 20,
        ),
        max_iterations=100,
    ).singlepoint(H2, charge=1, multiplicity=2)
    plain = Calculator(
        method="pbe-uks",
        basis="sto-3g",
        device="cpu",
        ks_options=KsOptions(grid=GRID, tile_points=16),
        max_iterations=100,
    ).singlepoint(H2, charge=1, multiplicity=2)

    assert corrected.converged
    assert corrected.physical_residual_rms < 1e-10
    assert np.isfinite(corrected.energy)
    assert abs(corrected.energy - plain.energy) > 1e-8


def test_nonlocal_resource_plan_counts_pair_owner_and_rejects_tiny_budget() -> None:
    graph = _graph("vv10", "unpolarized")
    options = _options(graph)
    corrected = estimate_ks_resources([H2], method="pbe-rks", ks_options=options)
    plain = estimate_ks_resources(
        [H2],
        method="pbe-rks",
        ks_options=KsOptions(grid=GRID, tile_points=16),
    )
    assert corrected.resident_bytes["host"] > plain.resident_bytes["host"]
    assert corrected.peak_bytes["host"] > plain.peak_bytes["host"]

    with pytest.raises(ValueError, match="nonlocal provider workspace exceeds"):
        estimate_ks_resources(
            [H2],
            method="pbe-rks",
            ks_options=KsOptions(
                composition=graph,
                grid=GRID,
                tile_points=16,
                nonlocal_memory_budget_bytes=1024,
            ),
        )
