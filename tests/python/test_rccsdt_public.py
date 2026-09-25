"""Public native standard canonical RCCSD(T) energy-owner acceptance for #155 C."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from vibeqc import Calculator, method_capabilities

from tools.validate_ccsd_t_gradient import analytic_oracle
from tools.vibeqc_cc.triples import INVENTORY_HASH

ROOT = Path(__file__).resolve().parents[2]
GRADIENTS = ROOT / "tests/reference_data/cc/gradients"
TRIPLES = ROOT / "tests/reference_data/cc/rccsd-t.json"


def _reference_case(name: str) -> tuple[list[tuple[int, list[float]]], dict, float]:
    record = json.loads(
        (GRADIENTS / f"{name if name != 'h2' else 'h2_shifted'}.json").read_text()
    )
    triples = {
        item["name"]: item["et_ground_truth"]
        for item in json.loads(TRIPLES.read_text())["molecules"]
    }
    inputs = record["inputs"]
    atoms = list(zip(inputs["atomic_numbers"], inputs["coordinates"], strict=True))
    return atoms, record, triples[name]


def _calculator(**kwargs: object) -> Calculator:
    options: dict[str, object] = {
        "method": "ccsd(t)",
        "basis": "sto-3g",
        "device": "cpu",
        "max_iterations": 200,
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-11,
        "ccsd_max_iterations": 150,
        "ccsd_energy_tolerance": 1e-13,
        "ccsd_residual_tolerance": 1e-11,
    }
    options.update(kwargs)
    return Calculator(**options)


@pytest.fixture(params=("cpu", "cuda"))
def energy_device(request: pytest.FixtureRequest) -> str:
    if request.param == "cuda" and os.environ.get("VIBEQC_RCCSDT_CUDA_TEST") != "1":
        pytest.skip("requires explicitly allocated CUDA native library")
    return request.param


@pytest.fixture
def cuda_device() -> str:
    if os.environ.get("VIBEQC_RCCSDT_CUDA_TEST") != "1":
        pytest.skip("requires explicitly allocated CUDA native library")
    return "cuda"


def test_native_rccsdt_capability_is_energy_forces_batch() -> None:
    caps = method_capabilities("rccsd(t)")
    alias = method_capabilities("ccsd(t)")
    assert caps.available and caps.supports_batch
    assert caps.family == "coupled_cluster"
    assert caps.supported_properties == frozenset({"energy", "forces"})
    assert alias.available and alias.supported_properties == caps.supported_properties


@pytest.mark.parametrize("case", ("h2", "h2o", "nh3"))
def test_public_native_rccsdt_matches_pinned_standard_triples(
    energy_device: str, case: str
) -> None:
    atoms, reference, triples = _reference_case(case)
    result = _calculator(device=energy_device).singlepoint(
        atoms, properties=("energy",)
    )
    diag = result.correlation
    assert result.converged and result.forces is None
    assert result.executed_backend == (
        "cuda" if energy_device == "cuda" else "cpu_reference"
    )
    assert diag is not None
    assert diag.ccsd_t_triples_energy == pytest.approx(triples, abs=2e-9)
    assert result.energy == pytest.approx(reference["total_energy"] + triples, abs=3e-9)
    assert diag.ccsd_correlation_energy == pytest.approx(
        reference["correlation_energy"], abs=2e-9
    )
    assert diag.ccsd_t_equation_hash == INVENTORY_HASH
    assert diag.ccsd_t_virtual_triples > 0
    assert diag.ccsd_t_workspace_bytes > 0
    assert diag.ccsd_replay_singles_residual_max <= 1e-11
    assert diag.ccsd_replay_doubles_residual_max <= 1e-11
    if energy_device == "cuda":
        assert diag.correlation_owned_device_bytes >= diag.ccsd_t_workspace_bytes
        assert diag.ccsd_setup_h2d_bytes > 0
        assert diag.ccsd_amplitude_d2h_bytes > 0
        assert diag.mo_host_staging


@pytest.mark.parametrize("case", ("h2o", "nh3"))
def test_public_native_rccsdt_force_matches_pyscf_analytic_gradient(case: str) -> None:
    pytest.importorskip(
        "pyscf", reason="independent RCCSD(T) gradient reference requires PySCF"
    )
    pytest.importorskip("threadpoolctl")
    atoms, _, _ = _reference_case(case)
    expected = np.asarray(
        analytic_oracle(case)["analytic"]["gradient"], dtype=np.float64
    )
    result = _calculator().singlepoint(atoms, properties=("energy", "forces"))
    assert result.converged
    assert result.forces is not None
    np.testing.assert_allclose(result.forces, -expected, atol=1.0e-6, rtol=0)
    diag = result.correlation
    assert diag is not None
    assert diag.response_absolute_residual <= 1.0e-9
    assert diag.response_iterations > 0
    assert diag.force_provenance_flags & 0x1


def test_public_native_rccsdt_cuda_force_matches_pyscf_analytic_gradient(
    cuda_device: str,
) -> None:
    pytest.importorskip(
        "pyscf", reason="independent RCCSD(T) gradient reference requires PySCF"
    )
    pytest.importorskip("threadpoolctl")
    atoms, _, _ = _reference_case("h2o")
    expected = np.asarray(
        analytic_oracle("h2o")["analytic"]["gradient"], dtype=np.float64
    )
    result = _calculator(device=cuda_device).singlepoint(
        atoms, properties=("energy", "forces")
    )
    assert result.converged
    assert result.forces is not None
    np.testing.assert_allclose(result.forces, -expected, atol=1.0e-6, rtol=0)
    diag = result.correlation
    assert diag is not None
    assert diag.force_provenance_flags & 0x8
    assert diag.response_absolute_residual <= 1.0e-9
    assert diag.response_iterations > 0


def test_public_native_rccsdt_force_matches_three_step_energy_finite_difference() -> (
    None
):
    atoms, _, _ = _reference_case("h2o")
    rng = np.random.default_rng(15521)
    direction = rng.normal(size=(len(atoms), 3))
    direction -= direction.mean(axis=0, keepdims=True)
    direction /= np.linalg.norm(direction)

    calc = _calculator()
    center = calc.singlepoint(atoms, properties=("energy", "forces"))
    assert center.forces is not None
    analytic = -float(np.vdot(np.asarray(center.forces), direction))
    errors: list[float] = []
    for step in (1.0e-3, 3.0e-4, 1.0e-4):
        energies: list[float] = []
        for sign in (-1.0, 1.0):
            displaced = [
                (z, (np.asarray(xyz) + sign * step * delta).tolist())
                for (z, xyz), delta in zip(atoms, direction, strict=True)
            ]
            energies.append(calc.singlepoint(displaced, properties=("energy",)).energy)
        finite = (energies[1] - energies[0]) / (2.0 * step)
        errors.append(abs(finite - analytic))
    assert min(errors[1:]) < 1.0e-6, errors
    assert errors[-1] < 2.0e-6, errors


def test_public_native_rccsdt_rejects_df_and_frozen_core() -> None:
    with pytest.raises(NotImplementedError, match=r"density fitting"):
        _calculator(density_fitting="cpu")
    with pytest.raises(NotImplementedError, match=r"frozen-core"):
        _calculator(ccsd_frozen_core=1)


def test_public_native_rccsdt_nonconvergence_never_publishes_triples(
    energy_device: str,
) -> None:
    atoms, _, _ = _reference_case("h2o")
    calc = _calculator(device=energy_device, ccsd_max_iterations=1)
    with pytest.raises(RuntimeError, match=r"maximum RCCSD iterations|conver"):
        calc.singlepoint(atoms, properties=("energy",))


def test_public_native_rccsdt_homogeneous_batch_repeats_and_moves_geometry() -> None:
    atoms, reference, triples = _reference_case("h2")
    moved = [(z, [xyz[0], xyz[1], xyz[2] + 0.02]) for z, xyz in atoms]
    calc = _calculator()
    with calc.prepare_batch([atoms, moved]) as prepared:
        first = prepared.execute(strict=True)
        assert all(item.converged for item in first.items)
        assert first.items[0].energy == pytest.approx(
            reference["total_energy"] + triples, abs=3e-9
        )
        assert all(
            item.correlation is not None
            and item.correlation.ccsd_t_equation_hash == INVENTORY_HASH
            for item in first.items
        )
        forced = prepared.execute(properties=("energy", "forces"), strict=True)
        assert all(item.forces is not None for item in forced.items)
        repeated = prepared.execute(strict=True)
        np.testing.assert_allclose(
            [item.energy for item in repeated.items],
            [item.energy for item in first.items],
            atol=2e-10,
            rtol=0,
        )


def test_public_native_rccsdt_cuda_batch_rebuild_and_failure_isolation(
    cuda_device: str,
) -> None:
    atoms, _, _ = _reference_case("h2o")
    moved = [
        (z, [xyz[0], xyz[1], xyz[2] + (0.03 if i == 1 else 0.0)])
        for i, (z, xyz) in enumerate(atoms)
    ]
    calc = _calculator(device=cuda_device)
    expected = calc.singlepoint(moved, properties=("energy",)).energy
    with calc.prepare_batch([atoms, atoms]) as batch:
        first = batch.execute(properties=("energy",), strict=True)
        assert all(item.succeeded and item.forces is None for item in first.items)
        updated = batch.execute(
            coordinates=[None, [xyz for _, xyz in moved]],
            properties=("energy",),
            strict=True,
        )
        assert updated.items[1].energy == pytest.approx(expected, abs=2e-9)
        invalid = batch.execute(
            coordinates=[None, [0.0]], properties=("energy",), strict=False
        )
        assert invalid.items[0].succeeded and not invalid.items[1].succeeded
        assert invalid.items[1].correlation is None


def test_public_native_rccsdt_cuda_batch_forces(
    cuda_device: str,
) -> None:
    atoms, _, _ = _reference_case("h2")
    calc = _calculator(device=cuda_device)
    with calc.prepare_batch([atoms, atoms]) as batch:
        result = batch.execute(properties=("energy", "forces"), strict=True)
    assert all(item.succeeded and item.forces is not None for item in result.items)
    # Independent CUDA derivative reductions may differ by a few FP64 ulps.
    np.testing.assert_allclose(
        result.items[0].forces, result.items[1].forces, atol=1e-12, rtol=0
    )
    assert all(
        item.correlation is not None and item.correlation.force_provenance_flags & 0x8
        for item in result.items
    )


def test_public_force_admits_reported_endpoint_budget() -> None:
    """The reported total includes preparation and retained response lifetimes."""
    atoms, _, _ = _reference_case("h2o")
    reference = _calculator().singlepoint(atoms, properties=("energy", "forces"))
    peak = reference.correlation.planned_endpoint_peak_bytes
    assert peak > 0
    exact = _calculator(correlation_memory_budget_bytes=peak).singlepoint(
        atoms, properties=("energy", "forces")
    )
    assert exact.correlation.numeric_capacity_bytes <= peak
    np.testing.assert_allclose(exact.forces, reference.forces, rtol=0, atol=1e-12)
    with pytest.raises(RuntimeError, match="error 7|host budget|memory budget"):
        _calculator(correlation_memory_budget_bytes=peak - 1).singlepoint(
            atoms, properties=("energy", "forces")
        )
