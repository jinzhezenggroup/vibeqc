"""End-to-end canonical r2SCAN-3c composition gates for #172."""

import os
import typing

import numpy as np
import pytest
from vibeqc import (
    Calculator,
    GridSpec,
    KsOptions,
    R2SCAN3CCorrectionBatch,
    evaluate_r2scan3c_correction,
    evaluate_r2scan3c_gcp,
    load_r2scan3c_basis,
)
from vibeqc_compiler.method import UnsupportedMethod, resolve_method

H2 = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
GRID = GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)


def _numbers_and_coordinates(atoms: typing.Any) -> tuple[list[int], np.ndarray]:
    from vibeqc.calculator import Atom

    normalized = tuple(Atom.from_value(atom) for atom in atoms)
    return (
        [atom.atomic_number for atom in normalized],
        np.asarray([atom.position for atom in normalized], dtype=np.float64),
    )


def test_native_gcp_public_wrapper_matches_independent_reference() -> None:
    from tools.vibeqc_gcp.reference import evaluate_r2scan3c_gcp as reference

    graph = resolve_method("R2SCAN-3c")
    numbers, xyz = _numbers_and_coordinates(H2)
    native = evaluate_r2scan3c_gcp(graph, numbers, xyz)
    oracle = reference(tuple(numbers), xyz)

    assert native.backend == "cpu"
    assert native.provider_identity == "r2scan3c-gcp-cpu-v1"
    assert native.energy == pytest.approx(oracle.energy, abs=2e-14)
    np.testing.assert_allclose(native.gradient, oracle.gradient, atol=2e-13, rtol=0)


def test_r2scan3c_correction_batch_composes_d4_and_gcp_once_on_replay() -> None:
    graph = resolve_method("R2SCAN-3c")
    numbers, xyz = _numbers_and_coordinates(H2)
    moved = xyz.copy()
    moved[1, 2] += 0.05

    with R2SCAN3CCorrectionBatch(graph, [(numbers, xyz, 0.0)], device="cpu") as batch:
        first = batch.execute(gradients=True)[0]
        replay = batch.execute([moved], gradients=True)[0]
        diagnostic = batch.diagnostic()

    direct = evaluate_r2scan3c_correction(
        graph, numbers, moved, total_charge=0.0, device="cpu"
    )
    assert first.ok and replay.ok
    assert replay.energy == pytest.approx(
        replay.d4.energy + replay.gcp.energy, abs=2e-15
    )
    np.testing.assert_allclose(
        replay.gradient, replay.d4.gradient + replay.gcp.gradient, atol=2e-14, rtol=0
    )
    assert replay.energy == pytest.approx(direct.energy, abs=2e-15)
    np.testing.assert_allclose(replay.gradient, direct.gradient, atol=2e-14, rtol=0)
    assert first.energy != replay.energy
    assert diagnostic.method_ir_identity == graph.identity
    assert diagnostic.d4.profile == "r2scan3c"
    assert diagnostic.gcp_backend == "cpu"


def test_calculator_composes_canonical_r2scan3c_energy_and_binds_basis() -> None:
    graph = resolve_method("R2SCAN-3c")
    basis = load_r2scan3c_basis()
    plain = Calculator(
        method="r2scan-rks",
        basis=basis,
        ks_options=KsOptions(grid=GRID),
        max_iterations=200,
    ).singlepoint(H2, properties=("energy",))
    calculator = Calculator(
        method=graph,
        ks_options=KsOptions(grid=GRID),
        max_iterations=200,
    )
    corrected = calculator.singlepoint(H2, properties=("energy",))
    numbers, xyz = _numbers_and_coordinates(H2)
    correction = evaluate_r2scan3c_correction(
        graph, numbers, xyz, total_charge=0.0, device="cpu", gradients=False
    )

    assert calculator.method_ir == graph
    assert calculator._basis.identity == basis.identity
    assert calculator._basis.representation == "spherical"
    assert corrected.dispersion is not None
    assert corrected.dispersion.energy == pytest.approx(correction.energy, abs=2e-15)
    assert corrected.dispersion.d4.energy == pytest.approx(
        correction.d4.energy, abs=2e-15
    )
    assert corrected.dispersion.gcp.energy == pytest.approx(
        correction.gcp.energy, abs=2e-15
    )
    assert corrected.energy == pytest.approx(
        plain.energy + correction.energy, abs=2e-11
    )


def test_r2scan3c_rejects_wrong_basis_and_out_of_domain_element() -> None:
    graph = resolve_method("R2SCAN-3c")
    with pytest.raises(ValueError, match="expected basis"):
        Calculator(method=graph, basis="sto-3g", ks_options=KsOptions(grid=GRID))

    calculator = Calculator(method=graph, ks_options=KsOptions(grid=GRID))
    with pytest.raises(UnsupportedMethod, match="does not support atomic numbers"):
        calculator.singlepoint([("K", (0.0, 0.0, 0.0))], properties=("energy",))


@pytest.mark.skipif(
    os.environ.get("VIBEQC_DFT_CUDA_TEST") != "1",
    reason="explicit Slurm CUDA gate",
)
def test_cuda_r2scan3c_public_force_is_electronic_plus_d4_plus_gcp() -> None:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require a Slurm allocation"
    graph = resolve_method("R2SCAN-3c")
    basis = load_r2scan3c_basis()
    controls = {
        "device": "cuda",
        "ks_options": KsOptions(grid=GRID),
        "max_iterations": 200,
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
    }
    plain = Calculator(method="r2scan-rks", basis=basis, **controls).singlepoint(
        H2, properties=("energy", "forces")
    )
    corrected = Calculator(method=graph, **controls).singlepoint(
        H2, properties=("energy", "forces")
    )
    numbers, xyz = _numbers_and_coordinates(H2)
    correction = evaluate_r2scan3c_correction(
        graph, numbers, xyz, total_charge=0.0, device="cuda"
    )

    assert corrected.dispersion is not None
    assert corrected.dispersion.backend == "cuda+cpu"
    assert corrected.energy == pytest.approx(plain.energy + correction.energy, abs=2e-9)
    np.testing.assert_allclose(
        corrected.forces,
        plain.forces - correction.gradient,
        atol=2e-7,
        rtol=0,
    )
