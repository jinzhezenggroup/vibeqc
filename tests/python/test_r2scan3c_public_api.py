"""Public named r2SCAN-3c selectors preserve the canonical MethodIR contract."""

import numpy as np
import pytest
from vibeqc import (
    Calculator,
    GridSpec,
    KsOptions,
    load_r2scan3c_basis,
    method_capabilities,
)
from vibeqc_compiler.method import resolve_method

H2 = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
GRID = GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)


@pytest.mark.parametrize(
    ("name", "spin", "electronic"),
    [
        ("r2scan-3c", "unpolarized", "r2scan-rks"),
        ("r2scan-3c-rks", "unpolarized", "r2scan-rks"),
        ("r2scan-3c-uks", "polarized", "r2scan-uks"),
    ],
)
def test_named_selector_resolves_canonical_graph_and_basis(
    name: str, spin: str, electronic: str
) -> None:
    calculator = Calculator(method=name, ks_options=KsOptions(grid=GRID))
    expected = resolve_method("R2SCAN-3c", spin=spin)
    assert calculator.method_ir == expected
    assert calculator._method_name == electronic
    assert calculator._basis.identity == load_r2scan3c_basis().identity
    assert calculator._basis.representation == "spherical"


@pytest.mark.parametrize("name", ["r2scan-3c", "r2scan-3c-rks", "r2scan-3c-uks"])
def test_named_selector_has_conservative_public_capabilities(name: str) -> None:
    capabilities = method_capabilities(name)
    assert capabilities.method == name
    assert capabilities.family == "density_functional"
    assert capabilities.available
    assert capabilities.supports_batch
    assert capabilities.supported_properties == frozenset({"energy"})


def test_named_selector_rejects_changed_defining_basis() -> None:
    with pytest.raises(ValueError, match="expected basis"):
        Calculator(
            method="r2scan-3c-rks",
            basis="sto-3g",
            ks_options=KsOptions(grid=GRID),
        )


def test_named_selector_energy_batch_and_changed_geometry_match_fresh_execution() -> (
    None
):
    calculator = Calculator(
        method="r2scan-3c-rks",
        device="cpu",
        ks_options=KsOptions(grid=GRID),
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    direct = calculator.singlepoint(H2, properties=("energy",))
    moved = np.asarray([atom[1] for atom in H2], dtype=np.float64)
    moved[1] += [0.02, -0.03, 0.06]
    moved_atoms = [
        (symbol, tuple(xyz)) for (symbol, _), xyz in zip(H2, moved, strict=True)
    ]
    fresh = calculator.singlepoint(moved_atoms, properties=("energy",))

    with calculator.prepare_batch([H2, H2], warm_start=False) as batch:
        first = batch.execute(strict=True, properties=("energy",))
        replay = batch.execute(
            coordinates=[moved, None], strict=True, properties=("energy",)
        )

    assert first.items[0].energy == pytest.approx(direct.energy, abs=2e-10)
    assert replay.items[0].energy == pytest.approx(fresh.energy, abs=2e-10)
    assert replay.items[1].energy == pytest.approx(direct.energy, abs=2e-10)
