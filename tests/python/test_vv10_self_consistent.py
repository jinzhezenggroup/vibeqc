"""Self-consistent native VV10/rVV10 execution through the generic KS plan (#491)."""

import json
from fractions import Fraction

import numpy as np
import pytest
from vibeqc import (
    BasisProvenance,
    BasisSet,
    BasisShell,
    Calculator,
    ElementBasis,
    GridSpec,
    KsOptions,
    estimate_ks_resources,
)
from vibeqc._dft_gradient import StationaryKsState
from vibeqc._stationary_cpu import complete_rks_gradient_diagnostic
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
FORCE_ATOMS = ((11, (0.0, 0.0, 0.0)), (1, (0.0, 0.0, 2.2)))
FORCE_DIRECTION = np.array(((0.17, -0.11, 0.07), (-0.09, 0.06, 0.13)))


def _force_basis() -> BasisSet:
    na = ElementBasis(
        11,
        (BasisShell(0, ("0.7",), (("1",),)),),
        ecp_core_electrons=10,
        ecp_data=json.dumps(
            [
                {
                    "ecp_type": "scalar_ecp",
                    "angular_momentum": [0],
                    "r_exponents": [2],
                    "gaussian_exponents": ["0.8"],
                    "coefficients": [["-2"]],
                }
            ]
        ),
    )
    hydrogen = ElementBasis(1, (BasisShell(0, ("1.0",), (("1",),)),))
    return BasisSet(
        "synthetic NaH ECP force fixture",
        (na, hydrogen),
        BasisProvenance("synthetic test", "1", "CC0", "0" * 64),
        "cartesian",
    )


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


@pytest.mark.parametrize("variant", ("vv10", "rvv10"))
@pytest.mark.parametrize(
    ("spin", "selector", "charge", "multiplicity"),
    (
        ("unpolarized", "pbe-rks", 0, 1),
        ("polarized", "pbe-uks", 0, 1),
        ("polarized", "pbe-uks", 1, 2),
    ),
)
def test_public_cpu_nonlocal_force_matches_recomputed_energy_differences(
    variant: str, spin: str, selector: str, charge: int, multiplicity: int
) -> None:
    graph = _graph(variant, spin)
    calculator = Calculator(
        method=graph,
        basis=_force_basis(),
        device="cpu",
        ks_options=KsOptions(
            grid=GRID,
            tile_points=5,
            nonlocal_memory_budget_bytes=16 << 20,
        ),
        max_iterations=120,
        energy_tolerance=1e-11,
        density_tolerance=1e-9,
    )
    assert calculator._method_name == selector
    assert "forces" in calculator._capabilities.supported_properties
    result = calculator.singlepoint(
        FORCE_ATOMS,
        charge=charge,
        multiplicity=multiplicity,
        properties=("energy", "forces"),
    )
    assert result.converged and result.forces is not None
    assert result.physical_residual_rms is not None
    assert result.physical_residual_rms < 1e-12
    np.testing.assert_allclose(result.forces.sum(axis=0), 0.0, atol=5e-13, rtol=0.0)

    if multiplicity == 2:
        with (
            calculator.prepare_batch(
                [FORCE_ATOMS], charges=[charge], multiplicities=[multiplicity]
            ) as batch,
            NativeAO(
                FORCE_ATOMS,
                basis=_force_basis(),
                charge=charge,
                multiplicity=multiplicity,
            ) as basis,
        ):
            batch.execute(strict=True, properties=("energy",))
            state = StationaryKsState.from_native(batch, basis)
            try:
                assert np.linalg.norm(state.density[0] - state.density[1]) > 0.1
            finally:
                state._source.close()

    projection = -float(np.sum(result.forces * FORCE_DIRECTION))
    base = np.asarray([position for _, position in FORCE_ATOMS], dtype=np.float64)
    labels = tuple(number for number, _ in FORCE_ATOMS)
    estimates = []
    for step in (3e-4, 1e-4):
        energies = []
        for sign in (1, -1):
            moved = base + sign * step * FORCE_DIRECTION
            atoms = tuple(
                (labels[index], tuple(moved[index])) for index in range(len(labels))
            )
            energies.append(
                calculator.singlepoint(
                    atoms,
                    charge=charge,
                    multiplicity=multiplicity,
                    properties=("energy",),
                ).energy
            )
        estimates.append((energies[0] - energies[1]) / (2 * step))
    assert abs(estimates[-1] - projection) < 5e-10
    assert abs(estimates[-1] - projection) < abs(estimates[0] - projection)


def test_stationary_reduction_consumes_all_nonlocal_geometry_sources(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    basis_record = _force_basis()
    graph = _graph("vv10", "unpolarized")
    calculator = Calculator(
        method=graph,
        basis=basis_record,
        device="cpu",
        ks_options=KsOptions(
            grid=GRID,
            tile_points=5,
            nonlocal_memory_budget_bytes=16 << 20,
        ),
        max_iterations=120,
        energy_tolerance=1e-10,
        density_tolerance=1e-8,
    )
    with (
        calculator.prepare_batch([FORCE_ATOMS]) as batch,
        NativeAO(FORCE_ATOMS, basis=basis_record) as basis,
    ):
        batch.execute(strict=True, properties=("energy",))
        state = StationaryKsState.from_native(batch, basis)
        try:
            result = complete_rks_gradient_diagnostic(
                state,
                basis,
                cache=tmp_path,
                execution="native",
                max_host_bytes=256 << 20,
                tile_points=5,
            )
            point_count = len(state.grid.points)
            from vibeqc import _stationary_cpu

            def forbidden_provider(*args: object, **kwargs: object) -> object:
                pytest.fail(
                    "work admission reached derivative compilation/provider execution"
                )

            monkeypatch.setattr(_stationary_cpu, "compile_runtime", forbidden_provider)
            monkeypatch.setattr(
                _stationary_cpu, "NativeNonlocalPairProvider", forbidden_provider
            )
            with pytest.raises(ValueError, match="grid pair work budget exceeded"):
                complete_rks_gradient_diagnostic(
                    state,
                    basis,
                    cache=tmp_path,
                    execution="native",
                    tile_points=5,
                    max_host_bytes=256 << 20,
                    max_grid_pair_visits=result.work["grid_pair_work_bound"] - 1,
                )

        finally:
            state._source.close()

    for source in ("nonlocal_ao", "nonlocal_grid", "nonlocal_weight"):
        assert np.max(np.abs(result.components[source])) > 1e-6
    assert result.work["nonlocal_pair_evaluations"] == point_count**2
    assert result.work["nonlocal_partition_pair_work"] > 0
