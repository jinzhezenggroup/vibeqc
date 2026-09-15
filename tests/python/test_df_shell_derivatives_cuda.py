"""Execution and complete-force qualification of generated shell reuse."""

import os

import numpy as np
import pytest
from vibeqc import Calculator

from benchmarks.df_component_ledger import read_trace

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.mark.parametrize("method", ["rhf", "uhf"])
@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize("schedule", ["warp", "packed", "compact"])
@pytest.mark.parametrize(
    "algebra,staging", [("scalar", "pageable"), ("blas", "pinned-panels")]
)
def test_shell_execution_and_return_to_generic(
    method, representation, schedule, algebra, staging, monkeypatch, tmp_path
):
    """Independent complete gradients and counters exclude a silent generic replay.

    Both orbital p and d shells are present: generated shell and generic
    consumers must partition weights exactly once, with metric/auxiliary
    response still contributing to the same complete atom gradient.
    """
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
        cart=representation == "cartesian",
        spin=spin,
        verbose=0,
    )
    reference = (scf.UHF if spin else scf.RHF)(mol).density_fit(auxbasis="def2-svp")
    reference.conv_tol, reference.conv_tol_grad = 1e-13, 1e-10
    reference.max_cycle = 100
    reference.kernel()
    assert reference.converged
    derivative = reference.nuc_grad_method()
    derivative.auxbasis_response = True
    expected = -derivative.kernel()
    calc = Calculator(
        device="cuda",
        method=method,
        basis="def2-svp",
        basis_representation=representation,
        density_fitting="cuda",
        density_fitting_memory_budget_bytes=16 << 20,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
    )
    monkeypatch.setenv("VIBEQC_DF_WEIGHTED_EXECUTION", "generic")
    monkeypatch.setenv("VIBEQC_DF_SHELL_SCHEDULE", schedule)
    monkeypatch.setenv("VIBEQC_DF_RESPONSE_ALGEBRA", algebra)
    monkeypatch.setenv("VIBEQC_DF_RAW_STAGING", staging)
    with calc.prepare_batch([atoms], multiplicities=[spin + 1]) as owner:
        owner.execute(properties=("energy", "forces"), strict=True)
        routes = [
            ("generic", "full"),
            ("shell-sp", "full"),
            ("shell-sp", "symmetric"),
            ("shell", "full"),
            ("shell", "symmetric"),
            ("generic", "full"),
        ]
        for index, (route, pairs) in enumerate(routes):
            monkeypatch.setenv("VIBEQC_DF_WEIGHTED_EXECUTION", route)
            monkeypatch.setenv("VIBEQC_DF_DERIVATIVE_PAIRS", pairs)
            monkeypatch.setenv("VIBEQC_DF_SHELL_COUNTERS", "1")
            trace = tmp_path / f"response-{index}.jsonl"
            monkeypatch.setenv("VIBEQC_DF_TRACE", str(trace))
            result = owner.execute(properties=("energy", "forces"), strict=True).items[
                0
            ]
            assert result.energy == pytest.approx(reference.e_tot, abs=3e-10)
            np.testing.assert_allclose(result.forces, expected, atol=3e-9, rtol=0)
            np.testing.assert_allclose(result.forces.sum(axis=0), 0, atol=3e-9)
            (record,) = [
                r for r in read_trace(trace) if r["operation"] == "force_response"
            ]
            counters = record["counters"]
            assert (counters.get("response_charge_blas_dots", 0) > 0) == (
                algebra == "blas"
            )
            if not record["source_backed"]:
                assert (counters.get("raw_panel_pinned_host_bytes", 0) > 0) == (
                    staging == "pinned-panels"
                )
            if route != "generic":
                assert counters["three_center_shell_panels"] > 0
                assert (
                    0
                    < counters["shell_triples_nonzero"]
                    <= counters["shell_triples_visited"]
                )
                assert (
                    counters["shell_public_weights_nonzero"]
                    > counters["shell_triples_nonzero"]
                )
                assert (
                    0
                    < counters["shell_primitive_products"]
                    < counters["shell_cartesian_component_products"]
                )
                if route == "shell":
                    ns = mol.nbas
                    logical = ns * ns if pairs == "full" else ns * (ns + 1) // 2
                    assert counters["shell_pairs_logical"] == logical
                    assert counters["shell_triples_visited"] == logical * ns
                    # Slice A folds the two dense loads; it reduces primitive
                    # work while retaining the complete dense weight contract.
                    assert counters["shell_public_weights_consumed"] == mol.nao**3
            else:
                assert "shell_triples_visited" not in counters
            policy = owner._warm_metadata[0]["controls"]["runtime_policy"]
            assert policy["VIBEQC_DF_WEIGHTED_EXECUTION"] == route
