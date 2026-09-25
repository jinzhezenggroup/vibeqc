"""A real 96-AO replay changes value residency while retaining full force response."""

import os
import typing

import numpy as np
import pytest
from vibeqc import Calculator, _native

from benchmarks._cases import benchmark_cases
from benchmarks.df_component_ledger import aggregate_host, read_host_trace, read_trace
from benchmarks.df_progress_ledger import read_progress

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.fixture(scope="module")
def tetramer_reference() -> typing.Any:
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


@pytest.mark.parametrize("budget", [24 << 20, 32 << 20, 64 << 20])
def test_property_replay_replans_value_storage_with_complete_forces(
    tetramer_reference: typing.Any,
    budget: typing.Any,
    monkeypatch: typing.Any,
    tmp_path: typing.Any,
) -> None:
    assert os.environ.get("SLURM_JOB_ID")
    case, energy, forces = tetramer_reference
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
    # The current policy divides the public allowance by workload, not 50/50.
    # Check the actual value/response caps and their conserved total. At 24 MiB
    # the force replay must regenerate B while the energy replay retains it;
    # larger allowances exercise the resident value path with the same oracle.
    with calc.prepare_batch([case.atoms]) as batch:
        for index, properties in enumerate(
            (("energy",), ("energy", "forces"), ("energy",))
        ):
            progress = tmp_path / f"policy-{index}.jsonl"
            trace = tmp_path / f"device-{index}.jsonl"
            monkeypatch.setenv("VIBEQC_DF_PROGRESS_TRACE", str(progress))
            monkeypatch.setenv("VIBEQC_DF_TRACE", str(trace))
            item = batch.execute(strict=True, properties=properties).items[0]
            assert item.energy == pytest.approx(energy, abs=1e-9, rel=0)
            policies = [
                {value["key"]: value["value"] for value in scope["values"]}
                for scope in read_progress(progress)["scopes"].values()
                if scope["begin"]["name"] == "df_resource_policy"
            ]
            assert policies, "execution did not report its resolved resource policy"
            policy = policies[-1]
            value_budget = policy["resolved_value_budget_bytes"]
            response_budget = policy["resolved_response_budget_bytes"]
            assert policy["resolved_total_budget_bytes"] == budget
            assert value_budget + response_budget == budget
            diagnostics = batch.last_density_fitting_metric_diagnostics()
            assert len(diagnostics) == 1
            assert diagnostics[0].peak_device_bytes <= value_budget
            assert diagnostics[0].streamed == (
                budget == 24 << 20 and "forces" in properties
            )
            if "forces" in properties:
                np.testing.assert_allclose(item.forces, forces, atol=1e-8, rtol=0)
                (response,) = [
                    row
                    for row in read_trace(trace)
                    if row["operation"] == "force_response"
                ]
                assert response["counters"]["response_scratch_bytes"] <= response_budget


def test_resident_response_budget_replans_without_reference_factors(
    tetramer_reference: typing.Any, monkeypatch: typing.Any, tmp_path: typing.Any
) -> None:
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
        # Prime the occupied-response owner before comparing panel budgets.
        # Otherwise the first dense response and subsequent occupied responses
        # have different workspace demands even at the same scratch limit.
        batch.execute(strict=True, properties=("energy", "forces"))
        batch.set_warm_start_updates(False)
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
    # A single occupied panel can fit both allowances. A larger allowance must
    # not require more panels, and returning to the same cap must be stable.
    assert panels[0] == panels[2] >= panels[1] >= 1


@pytest.mark.parametrize(
    ("override", "public_budget"),
    [("0", 0), ("-1", 0), ("1e9", 0), (str(1 << 64), 0), (str(16 << 20), 16 << 20)],
)
def test_response_override_cannot_be_invalid_or_enlarge_public_budget(
    monkeypatch: typing.Any, override: typing.Any, public_budget: typing.Any
) -> None:
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
def test_batch_four_rejects_insufficient_solver_allowance(
    method: typing.Any, representation: typing.Any
) -> None:
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
