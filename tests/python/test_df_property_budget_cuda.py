"""A real 96-AO replay changes value residency while retaining full force response."""

import os

import numpy as np
import pytest
from vibeqc import Calculator, _native

from benchmarks._cases import benchmark_cases
from benchmarks.df_component_ledger import aggregate_host, read_host_trace, read_trace

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.fixture(scope="module")
def tetramer_reference():
    from pyscf import df, gto, scf

    case = benchmark_cases()["water-tetramer-def2-svp-spherical"]
    mol = gto.M(
        atom=case.atoms, basis=case.pyscf_basis, unit="Bohr", cart=False, verbose=0
    )
    mf = scf.RHF(mol).density_fit(auxbasis=case.pyscf_basis)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-10, 100
    energy = mf.kernel()
    assert mf.converged
    eigenvalues = np.linalg.eigvalsh(
        df.addons.make_auxmol(mol, case.pyscf_basis).intor("int2c2e")
    )
    assert np.count_nonzero(eigenvalues > 1e-10 * eigenvalues[-1]) == mol.nao
    # PySCF's DF gradient includes auxiliary-basis response by default.
    forces = -mf.nuc_grad_method().kernel()
    return case, energy, forces


@pytest.mark.parametrize("provider", ["tensor", "generated"])
@pytest.mark.parametrize("budget", [32 << 20, 64 << 20])
def test_property_replay_replans_value_storage_with_complete_forces(
    tetramer_reference, provider, budget, monkeypatch
):
    assert os.environ.get("SLURM_JOB_ID")
    case, energy, forces = tetramer_reference
    monkeypatch.setenv("VIBEQC_ONE_ELECTRON_DERIVATIVES", provider)
    calc = Calculator(
        basis=case.vibeqc_basis,
        basis_representation="spherical",
        device="cuda",
        density_fitting="cuda",
        density_fitting_memory_budget_bytes=budget,
        max_iterations=100,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    # This is a DF-subbudget regression, not acceptance of the larger global
    # inventory. 64 MiB keeps B for both properties with different scratch;
    # 32 MiB retains B only for energy and regenerates it under the force half,
    # with DIIS and all eigensolver workspace floors reserved before tiling.
    with calc.prepare_batch([case.atoms]) as batch:
        for properties in (("energy",), ("energy", "forces"), ("energy",)):
            item = batch.execute(strict=True, properties=properties).items[0]
            assert item.energy == pytest.approx(energy, abs=1e-9, rel=0)
            diagnostics = batch.last_density_fitting_metric_diagnostics()
            assert len(diagnostics) == 1
            assert diagnostics[0].peak_device_bytes <= budget // (
                2 if "forces" in properties else 1
            )
            assert diagnostics[0].streamed == (
                budget == 32 << 20 and "forces" in properties
            )
            if budget == 64 << 20:
                assert (diagnostics[0].auxiliary_tile < 96) == ("forces" in properties)
            if "forces" in properties:
                np.testing.assert_allclose(item.forces, forces, atol=1e-8, rtol=0)


def test_resident_response_budget_replans_without_reference_factors(
    tetramer_reference, monkeypatch, tmp_path
):
    """Changing response scratch preserves full forces and the resident provider."""
    assert os.environ.get("SLURM_JOB_ID")
    case, energy, forces = tetramer_reference
    calc = Calculator(
        basis=case.vibeqc_basis,
        basis_representation="spherical",
        device="cuda",
        density_fitting="cuda",
        max_iterations=100,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    panels = []
    with calc.prepare_batch([case.atoms]) as batch:
        for budget in (4 << 20, 16 << 20, 4 << 20):
            index = len(panels)
            trace = tmp_path / f"response-{index}.jsonl"
            host = tmp_path / f"host-{index}.jsonl"
            monkeypatch.setenv("VIBEQC_DF_RESPONSE_BUDGET_BYTES", str(budget))
            monkeypatch.setenv("VIBEQC_DF_TRACE", str(trace))
            monkeypatch.setenv("VIBEQC_DF_HOST_TRACE", str(host))
            item = batch.execute(strict=True, properties=("energy", "forces")).items[0]
            assert item.energy == pytest.approx(energy, abs=1e-9, rel=0)
            np.testing.assert_allclose(item.forces, forces, atol=1e-8, rtol=0)
            (response,) = [
                r for r in read_trace(trace) if r["operation"] == "force_response"
            ]
            assert not response["source_backed"]
            assert response["counters"]["response_scratch_bytes"] <= budget
            assert aggregate_host(read_host_trace(host))["reference_eigensolves"] == []
            panels.append(response["counters"]["response_auxiliary_blocks"])
    assert panels[0] == panels[2] > panels[1] == 1


@pytest.mark.parametrize(
    ("override", "public_budget"),
    [("0", 0), ("-1", 0), ("1e9", 0), (str(1 << 64), 0), (str(16 << 20), 16 << 20)],
)
def test_response_override_cannot_be_invalid_or_enlarge_public_budget(
    monkeypatch, override, public_budget
):
    """The diagnostic selector cannot turn a bounded caller into an unbounded one."""
    assert os.environ.get("SLURM_JOB_ID")
    monkeypatch.setenv("VIBEQC_DF_RESPONSE_BUDGET_BYTES", override)
    calc = Calculator(
        device="cuda",
        density_fitting="cuda",
        density_fitting_memory_budget_bytes=public_budget,
    )
    with pytest.raises(RuntimeError, match="budget|invalid argument"):
        calc.singlepoint([("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))])


@pytest.mark.parametrize("method", ("rhf", "uhf"))
@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
def test_batch_four_rejects_insufficient_solver_allowance(method, representation):
    """Do not turn the historical 8-MiB provider-test failures into silent retries."""
    assert os.environ.get("SLURM_JOB_ID")
    atoms = [("O", (0, 0, 0)), ("H", (0, 0, 1.8))]
    if method == "rhf":
        atoms.append(("H", (1.7, 0, -0.6)))
    calc = Calculator(
        device="cuda",
        method=method,
        basis="def2-svp",
        basis_representation=representation,
        density_fitting="cuda",
        density_fitting_memory_budget_bytes=8 << 20,
    )
    with calc.prepare_batch(
        [atoms] * 4, multiplicities=[2 if method == "uhf" else 1] * 4
    ) as owner:
        result = owner.execute(strict=False, properties=("energy", "forces"))
        assert all(item.status == _native.STATUS_OUT_OF_MEMORY for item in result.items)
