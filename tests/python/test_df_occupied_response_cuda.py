"""Occupied response must consume only its exact verified electronic state."""

import os

import numpy as np
import pytest
from vibeqc import Calculator

from benchmarks._cases import benchmark_cases
from benchmarks.df_component_ledger import read_trace

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.mark.parametrize(
    "method,multiplicity,elements",
    [("rhf", 1, ("H", "H")), ("uhf", 3, ("H", "H")), ("uhf", 2, ("O", "H"))],
)
@pytest.mark.parametrize("batch_size", [1, 2])
def test_occupied_response_replay_and_zero_rank_spin(
    monkeypatch, tmp_path, method, multiplicity, elements, batch_size
):
    """Independent forces, changed geometry, rank-zero beta and dense fallback."""
    from pyscf import gto, scf

    assert os.environ.get("SLURM_JOB_ID")
    monkeypatch.setenv("VIBEQC_DF_EXCHANGE", "occupied")
    monkeypatch.setenv("VIBEQC_DF_RESPONSE_STORAGE", "jk-scratch")
    atoms = [(elements[0], (0.0, 0.0, -0.7)), (elements[1], (0.1, 0.0, 0.7))]
    if elements == ("O", "H"):
        # Reuse the independently qualified open-shell geometry and explicitly
        # match the oracle's spherical basis below (oxygen contains d shells).
        atoms = benchmark_cases()["oh-def2-svp-spherical-uhf"].atoms
    moved = [(element, np.array(position, dtype=float)) for element, position in atoms]
    moved[1][1][0] += 0.001
    expected = []
    for geometry in (atoms, moved):
        mol = gto.M(
            atom=geometry,
            basis="def2-svp",
            unit="Bohr",
            spin=multiplicity - 1,
            verbose=0,
        )
        oracle = (scf.RHF if method == "rhf" else scf.UHF)(mol).density_fit(
            auxbasis="def2-svp"
        )
        oracle.conv_tol, oracle.conv_tol_grad, oracle.max_cycle = 1e-12, 1e-10, 100
        oracle.kernel()
        assert oracle.converged
        expected.append((oracle.e_tot, -oracle.nuc_grad_method().kernel()))
    calc = Calculator(
        method=method,
        basis="def2-svp",
        basis_representation="spherical",
        device="cuda",
        density_fitting="cuda",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=100,
    )
    with calc.prepare_batch(
        [atoms] * batch_size, multiplicities=[multiplicity] * batch_size
    ) as batch:
        batch.execute(strict=True)
        for step, (space, changed, rebuild) in enumerate(
            [
                ("occupied", False, False),
                ("dense", False, False),
                ("auto", False, False),
                ("occupied", True, False),
                ("occupied", True, True),
                ("occupied", True, False),
            ]
        ):
            monkeypatch.setenv("VIBEQC_DF_RESPONSE_SPACE", space)
            monkeypatch.setenv("VIBEQC_DF_FORCE_FINAL_REBUILD", str(int(rebuild)))
            trace = tmp_path / f"response-{step}.jsonl"
            monkeypatch.setenv("VIBEQC_DF_TRACE", str(trace))
            result = batch.execute(
                [np.array([r for _, r in moved])] * batch_size if changed else None,
                strict=True,
            )
            energy, force = expected[int(changed)]
            for item in result.items:
                assert item.energy == pytest.approx(energy, abs=1e-9, rel=0)
                np.testing.assert_allclose(item.forces, force, atol=1e-8, rtol=0)
            records = read_trace(trace)
            responses = [r for r in records if r["operation"] == "force_response"]
            assert len(responses) == batch_size
            for response in responses:
                counters = response["counters"]
                if space == "occupied" and not rebuild:
                    assert counters["response_occupied_projection_products"] > 0
                    n, a = response["nbf"], response["naux"]
                    assert counters["response_pseudo_density_peak_elements"] == (
                        min(a, 64) * n * n
                    )
                    assert counters["response_full_weight_tensor_elements"] == (
                        a * n * n if a <= 64 else 0
                    )
                    assert not counters.get("response_ao_matrix_products", 0)
                else:
                    assert not counters.get("response_occupied_projection_products", 0)
                    assert counters["response_ao_matrix_products"] > 0
