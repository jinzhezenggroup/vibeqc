"""Force budgets bound complete CUDA response across public layouts and spins."""

import os
import typing

import numpy as np
import pytest
from vibeqc import Calculator

from benchmarks.df_component_ledger import read_trace

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires a finite Slurm GPU allocation",
)


@pytest.mark.parametrize("pairs", ["full", "symmetric", "packed"])
@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize("method", ["rhf", "uhf"])
def test_screening_force_budget(
    pairs: typing.Any,
    representation: typing.Any,
    method: typing.Any,
    monkeypatch: typing.Any,
    tmp_path: typing.Any,
) -> None:
    """Charge skipped work to one observable budget, retaining exact other classes."""
    from pyscf import gto, scf

    assert os.environ.get("SLURM_JOB_ID")
    atoms = [("O", (0, 0, 0)), ("H", (0, 0, 1.8))]
    if method == "rhf":
        atoms.append(("H", (1.7, 0, -0.6)))
    spin = int(method == "uhf")
    mol = gto.M(
        atom=atoms,
        basis="def2-svp",
        unit="Bohr",
        spin=spin,
        cart=representation == "cartesian",
        verbose=0,
    )
    ref = (scf.UHF if spin else scf.RHF)(mol).density_fit(auxbasis="def2-svp")
    ref.conv_tol, ref.conv_tol_grad, ref.max_cycle = 1e-13, 1e-10, 100
    ref.kernel()
    assert ref.converged
    expected = -ref.nuc_grad_method().kernel()
    for key, value in {
        "VIBEQC_DF_RESPONSE_STORAGE": "jk-scratch",
        "VIBEQC_DF_RESIDENT_EXCHANGE": "full",
        "VIBEQC_DF_RESPONSE_SPACE": "occupied",
        "VIBEQC_DF_EXCHANGE": "occupied",
        "VIBEQC_DF_FINAL_EXCHANGE": "occupied",
        "VIBEQC_DF_WEIGHTED_EXECUTION": "shell",
        "VIBEQC_DF_PRIMITIVE_BUCKETS": "packet",
        "VIBEQC_DF_SHELL_POLICY": "candidate",
        "VIBEQC_DF_DERIVATIVE_PAIRS": pairs,
        "VIBEQC_DF_SHELL_COUNTERS": "1",
    }.items():
        monkeypatch.setenv(key, value)
    calc = Calculator(
        method=method,
        basis="def2-svp",
        basis_representation=representation,
        device="cuda",
        density_fitting="cuda",
        max_iterations=100,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    with calc.prepare_batch([atoms], multiplicities=[spin + 1]) as owner:
        answers = []
        counts = []
        for index, budget in enumerate((0, 1e-8, 1e-6, 1e-4, 0)):
            monkeypatch.setenv("VIBEQC_DF_FORCE_SCREEN_ABS", str(budget))
            trace = tmp_path / f"screen-{index}.jsonl"
            monkeypatch.setenv("VIBEQC_DF_TRACE", str(trace))
            result = owner.execute(strict=True).items[0]
            assert result.energy == pytest.approx(ref.e_tot, abs=1e-9, rel=0)
            np.testing.assert_allclose(
                result.forces, expected, atol=budget + 3e-9, rtol=0
            )
            np.testing.assert_allclose(result.forces.sum(axis=0), 0, atol=3e-9, rtol=0)
            answers.append(np.asarray(result.forces))
            (row,) = [
                r for r in read_trace(trace) if r["operation"] == "force_response"
            ]
            counts.append(row["counters"])
        for index, budget in enumerate((0, 1e-8, 1e-6, 1e-4, 0)):
            np.testing.assert_allclose(
                answers[index], answers[0], atol=budget + 1e-10, rtol=0
            )
            skipped = counts[index].get("screening_000_primitives_skipped", 0)
            assert (
                counts[index]["shell_primitive_products"] + skipped
                == counts[0]["shell_primitive_products"]
            )
            if budget:
                assert counts[index]["screening_000_primitives_considered"] > 0
        assert counts[3]["screening_000_primitives_skipped"] > 0


def test_screened_force_matches_energy_finite_differences(
    monkeypatch: typing.Any,
) -> None:
    """Screening changes force work only; compare two independent energy steps."""
    assert os.environ.get("SLURM_JOB_ID")
    atoms = [("O", (0, 0, 0)), ("H", (0, 0, 1.8)), ("H", (1.7, 0, -0.6))]
    monkeypatch.setenv("VIBEQC_DF_FORCE_SCREEN_ABS", "1e-6")
    monkeypatch.setenv("VIBEQC_DF_WEIGHTED_EXECUTION", "shell")
    calc = Calculator(
        basis="def2-svp",
        device="cuda",
        density_fitting="cuda",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=100,
    )
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


def test_invalid_screening_budget_is_rejected_on_strict_fallback(
    monkeypatch: typing.Any,
) -> None:
    """A generic path cannot silently ignore an invalid requested force policy."""
    assert os.environ.get("SLURM_JOB_ID")
    monkeypatch.setenv("VIBEQC_DF_WEIGHTED_EXECUTION", "generic")
    calc = Calculator(basis="sto-3g", device="cuda", density_fitting="cuda")
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    with calc.prepare_batch([atoms]) as owner:
        expected = owner.execute(strict=True).items[0]
        for value in ("-1", "nan", "1e309", "invalid"):
            monkeypatch.setenv("VIBEQC_DF_FORCE_SCREEN_ABS", value)
            # Batch errors expose the item status; native detail is not part of
            # this public exception contract.
            with pytest.raises(RuntimeError, match="batched item failures"):
                owner.execute(strict=True)
        monkeypatch.setenv("VIBEQC_DF_FORCE_SCREEN_ABS", "off")
        recovered = owner.execute(strict=True).items[0]
        np.testing.assert_allclose(recovered.forces, expected.forces, atol=1e-9, rtol=0)
