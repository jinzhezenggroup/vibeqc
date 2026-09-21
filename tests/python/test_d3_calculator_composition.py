"""Calculator-level electronic + D3(BJ) composition gates for #492."""

import typing
from dataclasses import replace

import numpy as np
import pytest
from vibeqc import Calculator, GridSpec, KsOptions, evaluate_d3_correction
from vibeqc_compiler.method import DispersionCorrectionPrimitive, resolve_method

H2 = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
WATER = [
    ("O", (0.1, -0.1, 0.2)),
    ("H", (1.5, 0.3, 0.4)),
    ("H", (-0.5, 1.4, 0.1)),
]
GRID = GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)


def _numbers_and_coordinates(atoms: typing.Any) -> tuple[list[int], np.ndarray]:
    from vibeqc.calculator import Atom

    normalized = tuple(Atom.from_value(atom) for atom in atoms)
    return (
        [atom.atomic_number for atom in normalized],
        np.asarray([atom.position for atom in normalized], dtype=np.float64),
    )


def _electronic(graph: typing.Any) -> typing.Any:
    return replace(
        graph,
        identifier=f"{graph.identifier}/electronic",
        primitives=tuple(
            primitive
            for primitive in graph.primitives
            if not isinstance(primitive, DispersionCorrectionPrimitive)
        ),
    )


def test_pbe_methodir_singlepoint_composes_energy_and_force_once() -> None:
    # CPU public KS forces are intentionally qualified only for the bounded ECP
    # domain. Reuse that admitted electronic endpoint rather than widening force
    # capability merely to test dispersion composition.
    from test_ecp import fixture

    atoms, basis, _ = fixture(representation="cartesian")
    graph = resolve_method("PBE-D3(BJ)", spin="unpolarized")
    plain = Calculator(
        method="pbe-rks",
        basis=basis,
        ks_options=KsOptions(grid=GRID),
        max_iterations=200,
    ).singlepoint(atoms, properties=("energy", "forces"))
    corrected_calculator = Calculator(
        method=graph,
        basis=basis,
        ks_options=KsOptions(grid=GRID),
        max_iterations=200,
    )
    corrected = corrected_calculator.singlepoint(atoms, properties=("energy", "forces"))
    numbers, coordinates = _numbers_and_coordinates(atoms)
    d3 = evaluate_d3_correction(graph, numbers, coordinates)

    assert corrected_calculator.method_ir == graph
    assert corrected_calculator.ks_options.method_ir == _electronic(graph)
    assert corrected.dispersion is not None
    assert corrected.dispersion.energy == pytest.approx(d3.energy, abs=2e-15)
    assert corrected.energy == pytest.approx(plain.energy + d3.energy, abs=2e-12)
    np.testing.assert_allclose(
        corrected.forces,
        plain.forces - d3.gradient,
        atol=3e-10,
        rtol=0.0,
    )


def test_pbe0_methodir_composes_energy_without_named_execution_branch() -> None:
    graph = resolve_method("PBE0-D3(BJ)", spin="unpolarized")
    plain = Calculator(
        method="pbe0-rks", ks_options=KsOptions(grid=GRID), max_iterations=200
    ).singlepoint(H2, properties=("energy",))
    corrected = Calculator(
        method=graph, ks_options=KsOptions(grid=GRID), max_iterations=200
    ).singlepoint(H2, properties=("energy",))
    numbers, coordinates = _numbers_and_coordinates(H2)
    d3 = evaluate_d3_correction(graph, numbers, coordinates)

    assert corrected.dispersion is not None
    assert corrected.energy == pytest.approx(plain.energy + d3.energy, abs=2e-12)


def test_existing_native_selector_can_bind_full_d3_composition() -> None:
    graph = resolve_method("PBE-D3(BJ)", spin="unpolarized")
    calculator = Calculator(
        method="pbe-rks",
        ks_options=KsOptions(composition=graph, grid=GRID),
        max_iterations=200,
    )
    assert calculator.method_ir == graph
    assert not any(
        isinstance(node, DispersionCorrectionPrimitive)
        for node in calculator.ks_options.method_ir.primitives
    )


def test_composed_ragged_replay_preserves_per_item_failure_boundary() -> None:
    graph = resolve_method("PBE-D3(BJ)", spin="unpolarized")
    corrected = Calculator(
        method=graph, ks_options=KsOptions(grid=GRID), max_iterations=200
    )
    plain = Calculator(
        method="pbe-rks", ks_options=KsOptions(grid=GRID), max_iterations=200
    )
    moved = np.asarray([position for _, position in H2], dtype=np.float64)
    moved[1, 2] += 0.07
    malformed_water = np.zeros((2, 3), dtype=np.float64)

    with corrected.prepare_batch([H2, WATER], warm_start=False) as batch:
        result = batch.execute(
            [moved, malformed_water], strict=False, properties=("energy",)
        )
        diagnostic = batch.dispersion_diagnostic
    with plain.prepare_batch([H2], warm_start=False) as plain_batch:
        electronic = plain_batch.execute(
            [moved], strict=True, properties=("energy",)
        ).items[0]
    numbers, _ = _numbers_and_coordinates(H2)
    d3 = evaluate_d3_correction(graph, numbers, moved)

    assert result.items[0].succeeded
    assert not result.items[1].succeeded
    assert result.items[0].energy == pytest.approx(
        electronic.energy + d3.energy, abs=2e-12
    )
    assert result.items[0].dispersion is not None
    assert diagnostic.method_ir_identity == graph.identity
    assert diagnostic.system_count == 2


def test_global_resource_plan_fails_closed_for_composed_d3_owner() -> None:
    graph = resolve_method("PBE-D3(BJ)", spin="unpolarized")
    calculator = Calculator(method=graph, ks_options=KsOptions(grid=GRID))
    with pytest.raises(NotImplementedError, match="D3 owner"):
        calculator.estimate_resources([H2])


def test_d3_does_not_promote_unqualified_pbe0_forces() -> None:
    graph = resolve_method("PBE0-D3(BJ)", spin="unpolarized")
    calculator = Calculator(method=graph, ks_options=KsOptions(grid=GRID))
    assert "forces" not in calculator._capabilities.supported_properties
    with pytest.raises(ValueError, match="does not support properties: forces"):
        calculator.singlepoint(H2, properties=("energy", "forces"))
