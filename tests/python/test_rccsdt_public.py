"""Public native standard canonical RCCSD(T) energy-owner acceptance for #155 C."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from vibeqc import Calculator, method_capabilities

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


def test_native_rccsdt_capability_is_energy_only_batch() -> None:
    caps = method_capabilities("rccsd(t)")
    alias = method_capabilities("ccsd(t)")
    assert caps.available and caps.supports_batch
    assert caps.family == "coupled_cluster"
    assert caps.supported_properties == frozenset({"energy"})
    assert alias.available and alias.supported_properties == caps.supported_properties


@pytest.mark.parametrize("case", ("h2", "h2o"))
def test_public_native_rccsdt_matches_pinned_standard_triples(case: str) -> None:
    atoms, reference, triples = _reference_case(case)
    result = _calculator().singlepoint(atoms, properties=("energy",))
    diag = result.correlation
    assert result.converged and result.forces is None
    assert result.executed_backend == "cpu_reference"
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


def test_public_native_rccsdt_rejects_unpromoted_force_cuda_df_and_frozen_core() -> (
    None
):
    atoms, _, _ = _reference_case("h2")
    with pytest.raises(ValueError, match=r"does not support.*forces"):
        _calculator().singlepoint(atoms, properties=("energy", "forces"))
    with pytest.raises(NotImplementedError, match=r"CUDA owner"):
        _calculator(device="cuda")
    with pytest.raises(NotImplementedError, match=r"density fitting"):
        _calculator(density_fitting="cpu")
    with pytest.raises(NotImplementedError, match=r"frozen-core"):
        _calculator(ccsd_frozen_core=1)


def test_public_native_rccsdt_nonconvergence_never_publishes_triples() -> None:
    atoms, _, _ = _reference_case("h2o")
    calc = _calculator(ccsd_max_iterations=1)
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
        repeated = prepared.execute(strict=True)
        np.testing.assert_allclose(
            [item.energy for item in repeated.items],
            [item.energy for item in first.items],
            atol=2e-10,
            rtol=0,
        )
