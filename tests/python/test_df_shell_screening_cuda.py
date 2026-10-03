"""Whole-shell force screening preserves the observable across response layouts."""

import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from generativeqc import Calculator

from benchmarks._retained_basis import load_retained_basis
from benchmarks.df_component_ledger import read_trace

pytestmark = pytest.mark.skipif(
    os.environ.get("GENERATIVEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires a finite Slurm GPU allocation",
)


def separated_fragments(spin: int = 0) -> list:
    """Keep local chemistry while exposing negligible interfragment AO products."""
    water = [("O", (0, 0, 0)), ("H", (0, 0, 1.8)), ("H", (1.7, 0, -0.6))]
    remote = [("H", (0, 0, 0))] if spin else water
    return water + [(z, (r[0] + 24, r[1], r[2])) for z, r in remote]


def controls(monkeypatch: Any, pairs: str = "symmetric") -> None:
    """Select actual production consumers and isolate the whole-shell budget."""
    for key, value in {
        "RESPONSE_STORAGE": "jk-scratch",
        "RESIDENT_EXCHANGE": "full",
        "RESPONSE_SPACE": "occupied",
        "EXCHANGE": "occupied",
        "FINAL_EXCHANGE": "occupied",
        "WEIGHTED_EXECUTION": "shell",
        "PRIMITIVE_BUCKETS": "packet",
        "SHELL_POLICY": "candidate",
        "DERIVATIVE_PAIRS": pairs,
        "SHELL_COUNTERS": "1",
        "FORCE_SCREEN_ABS": "off",
    }.items():
        monkeypatch.setenv("GENERATIVEQC_DF_" + key, value)


def calculator(**kwargs: Any) -> Calculator:
    """Auxiliary f functions exercise a different shell/component domain."""
    auxiliary = load_retained_basis(
        Path(__file__).resolve().parents[2]
        / "benchmarks/results/issue206-practical-auxiliary/identity/cc-pvdz-jkfit.json"
    )
    representation = kwargs.get("basis_representation", "cartesian")
    # The large Cartesian JKFIT metric crosses the native spectral cutoff;
    # the ordinary PySCF Cholesky oracle then represents a different model.
    # Keep that independent oracle full-rank with the smaller Cartesian set.
    auxiliary = (
        replace(auxiliary, representation=representation)
        if representation == "spherical"
        else "def2-svp"
    )
    return Calculator(
        basis=kwargs.pop("basis", "def2-svp"),
        auxiliary_basis=auxiliary,
        device="cuda",
        density_fitting="cuda",
        max_iterations=100,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        **kwargs,
    )


@pytest.mark.parametrize("pairs", ["full", "symmetric", "packed"])
@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize("method", ["rhf", "uhf"])
@pytest.mark.parametrize("basis", ["def2-svp", "def2-tzvp"])
def test_shell_budget_and_skipped_work(
    pairs: str,
    representation: str,
    method: str,
    basis: str,
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    """Validate complete forces and conserved primitive work on identical states."""
    from pyscf import gto, scf

    assert os.environ.get("SLURM_JOB_ID")
    spin = int(method == "uhf")
    atoms = separated_fragments(spin)
    mol = gto.M(
        atom=atoms,
        basis=basis,
        unit="Bohr",
        spin=spin,
        cart=representation == "cartesian",
        verbose=0,
    )
    auxbasis = "cc-pvdz-jkfit" if representation == "spherical" else "def2-svp"
    ref = (scf.UHF if spin else scf.RHF)(mol).density_fit(auxbasis=auxbasis)
    ref.conv_tol, ref.conv_tol_grad, ref.max_cycle = 1e-13, 1e-10, 100
    ref.kernel()
    assert ref.converged
    forces = -ref.nuc_grad_method().kernel()
    controls(monkeypatch, pairs)
    calc = calculator(method=method, basis=basis, basis_representation=representation)
    with calc.prepare_batch([atoms], multiplicities=[spin + 1]) as owner:
        monkeypatch.setenv("GENERATIVEQC_DF_SHELL_SCREEN_ABS", "off")
        owner.execute(strict=True)
        answers, counts = [], []
        for index, budget in enumerate((0, 1e-10, 1e-8, 0)):
            monkeypatch.setenv("GENERATIVEQC_DF_SHELL_SCREEN_ABS", str(budget))
            trace = tmp_path / f"shell-screen-{index}.jsonl"
            monkeypatch.setenv("GENERATIVEQC_DF_TRACE", str(trace))
            result = owner.execute(strict=True).items[0]
            assert result.energy == pytest.approx(ref.e_tot, abs=1e-8, rel=0)
            np.testing.assert_allclose(result.forces, forces, atol=1e-7, rtol=0)
            np.testing.assert_allclose(result.forces.sum(axis=0), 0, atol=3e-9, rtol=0)
            answers.append(np.asarray(result.forces))
            (row,) = [
                r for r in read_trace(trace) if r["operation"] == "force_response"
            ]
            counts.append(row["counters"])
        for index, budget in enumerate((0, 1e-10, 1e-8, 0)):
            np.testing.assert_allclose(
                answers[index], answers[0], atol=budget + 1e-10, rtol=0
            )
            skipped = counts[index].get("screening_shell_primitive_products_skipped", 0)
            assert (
                counts[index]["shell_primitive_products"] + skipped
                == counts[0]["shell_primitive_products"]
            )
            if budget:
                assert counts[index]["screening_shell_enabled"] == 1
                assert counts[index]["screening_shell_tasks_skipped"] > 0
                assert skipped > 0
            else:
                assert not counts[index].get("screening_shell_enabled", 0)


def test_shell_screening_energy_finite_difference(monkeypatch: Any) -> None:
    """Two displacement steps check force sign and geometry-bound regeneration."""
    assert os.environ.get("SLURM_JOB_ID")
    controls(monkeypatch)
    monkeypatch.setenv("GENERATIVEQC_DF_SHELL_SCREEN_ABS", "1e-10")
    atoms = separated_fragments()
    calc = calculator(basis_representation="spherical")
    actual = calc.singlepoint(atoms)
    for step in (2e-4, 5e-5):
        energies = []
        for sign in (1, -1):
            displaced = [(z, np.array(r, dtype=float)) for z, r in atoms]
            displaced[1][1][0] += sign * step
            energies.append(calc.singlepoint(displaced).energy)
        assert actual.forces[1, 0] == pytest.approx(
            -(energies[0] - energies[1]) / (2 * step), abs=2e-6, rel=0
        )


def test_shell_screening_invalid_control_on_generic(monkeypatch: Any) -> None:
    """Fallback execution still validates the requested scientific policy."""
    assert os.environ.get("SLURM_JOB_ID")
    monkeypatch.setenv("GENERATIVEQC_DF_WEIGHTED_EXECUTION", "generic")
    calc = Calculator(basis="sto-3g", device="cuda", density_fitting="cuda")
    with calc.prepare_batch([[("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]]) as owner:
        for value in ("-1", "nan", "1e309", "invalid"):
            monkeypatch.setenv("GENERATIVEQC_DF_SHELL_SCREEN_ABS", value)
            with pytest.raises(RuntimeError, match="batched item failures"):
                owner.execute(strict=True)
        monkeypatch.setenv("GENERATIVEQC_DF_SHELL_SCREEN_ABS", "off")
        assert owner.execute(strict=True).items[0].converged


@pytest.mark.parametrize("budget,enabled", [(3 << 20, False), (2_000_000, True)])
def test_panel_clipping_and_capacity_fallback(
    budget: int, enabled: bool, monkeypatch: Any, tmp_path: Any
) -> None:
    """A screen may consume spare bytes but must never retile the strict plan."""
    assert os.environ.get("SLURM_JOB_ID")
    controls(monkeypatch)
    monkeypatch.setenv("GENERATIVEQC_DF_RESPONSE_STORAGE", "panel")
    calc = calculator(basis_representation="spherical")
    with calc.prepare_batch([separated_fragments()]) as owner:
        monkeypatch.setenv("GENERATIVEQC_DF_SHELL_SCREEN_ABS", "off")
        owner.execute(strict=True)
        monkeypatch.setenv("GENERATIVEQC_DF_RESPONSE_BUDGET_BYTES", str(budget))
        answers, counters = [], []
        for index, screen in enumerate(("off", "1e-10")):
            monkeypatch.setenv("GENERATIVEQC_DF_SHELL_SCREEN_ABS", screen)
            trace = tmp_path / f"bounded-{index}.jsonl"
            monkeypatch.setenv("GENERATIVEQC_DF_TRACE", str(trace))
            result = owner.execute(strict=True).items[0]
            answers.append(result)
            (row,) = [
                r for r in read_trace(trace) if r["operation"] == "force_response"
            ]
            counters.append(row["counters"])
        np.testing.assert_allclose(
            answers[0].forces, answers[1].forces, atol=2e-10, rtol=0
        )
        assert answers[0].energy == pytest.approx(answers[1].energy, abs=1e-10, rel=0)
        strict, screened = counters
        assert (
            strict["response_auxiliary_blocks"] == screened["response_auxiliary_blocks"]
        )
        assert screened["response_auxiliary_blocks"] > 1
        assert screened["response_probe_total_device_bytes"] <= budget
        assert (
            screened["response_scratch_bytes"]
            == screened["response_probe_total_device_bytes"]
        )
        assert screened.get("screening_shell_enabled", 0) == int(enabled)
        if not enabled:
            assert screened["screening_shell_capacity_fallback"] == 1
            assert (
                screened["response_probe_total_device_bytes"]
                == strict["response_probe_total_device_bytes"]
            )
        assert (
            screened["shell_primitive_products"]
            + screened.get("screening_shell_primitive_products_skipped", 0)
            == strict["shell_primitive_products"]
        )


def test_batch_geometry_screening_matches_strict(
    monkeypatch: Any, tmp_path: Any
) -> None:
    """Each batch geometry must own current norms, even after a coordinate update."""
    assert os.environ.get("SLURM_JOB_ID")
    controls(monkeypatch)
    atoms = separated_fragments()
    moved = [(z, np.array(r, dtype=float)) for z, r in atoms]
    moved[1][1][0] += 0.01
    calc = calculator(basis_representation="spherical")
    with calc.prepare_batch([atoms, moved]) as owner:
        monkeypatch.setenv("GENERATIVEQC_DF_SHELL_SCREEN_ABS", "off")
        owner.execute(strict=True)
        strict = owner.execute(strict=True)
        monkeypatch.setenv("GENERATIVEQC_DF_SHELL_SCREEN_ABS", "1e-10")
        trace = tmp_path / "batch.jsonl"
        monkeypatch.setenv("GENERATIVEQC_DF_TRACE", str(trace))
        screened = owner.execute(strict=True)
        for before, after in zip(strict.items, screened.items, strict=True):
            assert after.energy == pytest.approx(before.energy, abs=1e-10, rel=0)
            np.testing.assert_allclose(after.forces, before.forces, atol=2e-10, rtol=0)
        rows = [r for r in read_trace(trace) if r["operation"] == "force_response"]
        assert len(rows) == 2
        assert all(r["counters"]["screening_shell_enabled"] == 1 for r in rows)
        # A stale far-field norm would omit important AO products after this
        # substantial approach of the remote water to the first fragment.
        near = np.array([r for _, r in atoms], dtype=float)
        near[3:, 0] -= 19
        monkeypatch.delenv("GENERATIVEQC_DF_TRACE")
        screened = owner.execute([near, near], strict=True)
        monkeypatch.setenv("GENERATIVEQC_DF_SHELL_SCREEN_ABS", "off")
        strict = owner.execute([near, near], strict=True)
        for before, after in zip(strict.items, screened.items, strict=True):
            np.testing.assert_allclose(after.forces, before.forces, atol=2e-10, rtol=0)


def test_automatic_shell_budget(monkeypatch: Any, tmp_path: Any) -> None:
    """The qualified automatic shell domain uses the explicit error budget."""
    assert os.environ.get("SLURM_JOB_ID")
    controls(monkeypatch)
    monkeypatch.delenv("GENERATIVEQC_DF_WEIGHTED_EXECUTION")
    monkeypatch.delenv("GENERATIVEQC_DF_SHELL_SCREEN_ABS", raising=False)
    trace = tmp_path / "automatic.jsonl"
    calc = calculator(basis_representation="spherical")
    with calc.prepare_batch([separated_fragments()]) as owner:
        owner.execute(strict=True)
        monkeypatch.setenv("GENERATIVEQC_DF_TRACE", str(trace))
        automatic = owner.execute(strict=True).items[0]
        (row,) = [r for r in read_trace(trace) if r["operation"] == "force_response"]
        assert row["counters"]["response_derivative_profile_promoted"] == 1
        assert row["counters"]["screening_shell_enabled"] == 1
        monkeypatch.delenv("GENERATIVEQC_DF_TRACE")
        monkeypatch.setenv("GENERATIVEQC_DF_SHELL_SCREEN_ABS", "off")
        strict = owner.execute(strict=True).items[0]
        np.testing.assert_allclose(automatic.forces, strict.forces, atol=2e-10, rtol=0)
