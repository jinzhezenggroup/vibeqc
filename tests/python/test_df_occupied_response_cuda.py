"""Occupied response must consume only its exact verified electronic state."""

import os

import numpy as np
import pytest
from vibeqc import Calculator

from benchmarks._cases import benchmark_cases
from benchmarks.df_component_ledger import read_host_trace, read_trace

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.mark.parametrize(
    "method,multiplicity,elements",
    [("rhf", 1, ("H", "H")), ("uhf", 3, ("H", "H")), ("uhf", 2, ("O", "H"))],
)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("pairs", ["generic", "full", "packed"])
def test_occupied_response_replay_and_zero_rank_spin(
    monkeypatch, tmp_path, method, multiplicity, elements, batch_size, pairs
):
    """Independent forces, changed geometry, rank-zero beta and dense fallback."""
    from pyscf import gto, scf

    assert os.environ.get("SLURM_JOB_ID")
    monkeypatch.setenv("VIBEQC_DF_EXCHANGE", "occupied")
    monkeypatch.setenv("VIBEQC_DF_RESPONSE_STORAGE", "jk-scratch")
    monkeypatch.setenv(
        "VIBEQC_DF_WEIGHTED_EXECUTION", "generic" if pairs == "generic" else "shell"
    )
    monkeypatch.setenv("VIBEQC_DF_SHELL_SCHEDULE", "compact")
    monkeypatch.setenv("VIBEQC_DF_PRIMITIVE_BUCKETS", "packet")
    monkeypatch.setenv(
        "VIBEQC_DF_DERIVATIVE_PAIRS", "full" if pairs == "generic" else pairs
    )
    monkeypatch.setenv("VIBEQC_DF_SHELL_COUNTERS", "1")
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
            host_trace = tmp_path / f"host-response-{step}.jsonl"
            monkeypatch.setenv("VIBEQC_DF_HOST_TRACE", str(host_trace))
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
            regions = [
                region
                for record in read_host_trace(host_trace)
                for region in record["regions"]
            ]
            final_states = {
                region["item"]: region["name"]
                for region in regions
                if region["name"] in ("final_state_reuse", "final_state_corrected")
            }
            assert set(final_states) == set(range(batch_size))
            if step == 0:
                # The unchanged canonical replay must exercise factor reuse.
                assert set(final_states.values()) == {"final_state_reuse"}
            for response in responses:
                counters = response["counters"]
                item = response["system_offset"]
                corrected = final_states[item] == "final_state_corrected"
                if corrected:
                    # Changed OH/UHF geometry can need strict final correction.
                    # Its new determinant has no retained canonical factor
                    # token, so dense response is required. Demand executed
                    # correction evidence instead of accepting arbitrary loss
                    # of occupied reuse; the independent force gate stays fixed.
                    assert any(
                        region["name"] == "strict_final_correction"
                        and region["item"] == item
                        for region in regions
                    )
                automatic = (
                    space == "auto"
                    and method == "rhf"
                    and batch_size == 1
                    and pairs != "generic"
                )
                if (space == "occupied" or automatic) and not (rebuild or corrected):
                    assert counters["response_occupied_projection_products"] > 0
                    n, a = response["nbf"], response["naux"]
                    stride = n * (n + 1) // 2 if pairs == "packed" else n * n
                    assert counters["response_pseudo_density_peak_elements"] == (
                        min(a, 64) * stride
                    )
                    assert counters["response_full_weight_tensor_elements"] == (
                        a * n * n if a <= 64 and pairs != "packed" else 0
                    )
                    assert counters["response_packed_pairs"] == int(pairs == "packed")
                    if pairs == "packed":
                        assert counters["response_dense_weight_panel_elements"] == 0
                        assert counters["shell_public_weights_consumed"] == a * stride
                        assert counters["three_center_derivative_weights"] == a * stride
                        assert (
                            counters["three_center_derivative_weight_bytes"]
                            == a * stride * 8
                        )
                    assert not counters.get("response_ao_matrix_products", 0)
                else:
                    assert not counters.get("response_occupied_projection_products", 0)
                    assert counters["response_ao_matrix_products"] > 0


@pytest.mark.parametrize("batch_products", ["off", "auto"])
def test_packed_response_crosses_ao_blocks_and_auxiliary_panels(
    monkeypatch, tmp_path, batch_products
):
    """A 96-AO oracle covers the ragged second GEMM block and two packed panels.

    Smaller molecular fixtures fit in a single AO block and cannot detect a
    wrong row offset or leading dimension in the direct triangular producer.
    """
    from pyscf import gto, scf

    assert os.environ.get("SLURM_JOB_ID")
    case = benchmark_cases()["water-tetramer-def2-svp-spherical"]
    mol = gto.M(atom=case.atoms, unit="Bohr", basis="def2-svp", verbose=0)
    reference = scf.RHF(mol).density_fit(auxbasis="def2-svp")
    reference.conv_tol, reference.conv_tol_grad, reference.max_cycle = 1e-12, 1e-10, 100
    reference.kernel()
    assert reference.converged
    expected = -reference.nuc_grad_method().kernel()
    monkeypatch.setenv("VIBEQC_DF_EXCHANGE", "occupied")
    monkeypatch.setenv("VIBEQC_DF_RESPONSE_STORAGE", "jk-scratch")
    monkeypatch.setenv("VIBEQC_DF_RESPONSE_SPACE", "occupied")
    monkeypatch.setenv("VIBEQC_DF_RESPONSE_BATCHING", batch_products)
    monkeypatch.setenv("VIBEQC_DF_WEIGHTED_EXECUTION", "shell")
    monkeypatch.setenv("VIBEQC_DF_SHELL_SCHEDULE", "compact")
    monkeypatch.setenv("VIBEQC_DF_PRIMITIVE_BUCKETS", "packet")
    monkeypatch.setenv("VIBEQC_DF_DERIVATIVE_PAIRS", "packed")
    monkeypatch.setenv("VIBEQC_DF_PACKED_AO_BLOCK_ROWS", "64")
    monkeypatch.setenv("VIBEQC_DF_SHELL_COUNTERS", "1")
    calc = Calculator(
        method="rhf",
        basis="def2-svp",
        basis_representation="spherical",
        device="cuda",
        density_fitting="cuda",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=100,
    )
    with calc.prepare_batch([case.atoms]) as batch:
        batch.execute(strict=True)
        trace = tmp_path / "packed-panels.jsonl"
        monkeypatch.setenv("VIBEQC_DF_TRACE", str(trace))
        actual = batch.execute(strict=True).items[0]
        assert actual.energy == pytest.approx(reference.e_tot, abs=1e-9, rel=0)
        np.testing.assert_allclose(actual.forces, expected, atol=1e-8, rtol=0)
    (response,) = [r for r in read_trace(trace) if r["operation"] == "force_response"]
    c = response["counters"]
    assert c["response_packed_pairs"] == 1
    assert c["response_auxiliary_blocks"] == 2
    assert c["response_dense_weight_panel_elements"] == 0
    assert c["response_full_weight_tensor_elements"] == 0
    assert c["response_pseudo_density_peak_elements"] == 64 * (96 * 97 // 2)
    assert c["three_center_derivative_weights"] == 96 * (96 * 97 // 2)
    assert c["shell_public_weights_consumed"] == c["three_center_derivative_weights"]
    assert c["shell_triples_visited"] == (48 * 49 // 2) * 48
    # These include the small unused upper triangles of diagonal blocks.
    assert c["response_pseudo_density_rectangular_elements"] == 96 * (64 * 64 + 32 * 96)
    assert c["response_pseudo_density_block_peak_elements"] == 64 * 64 * (
        64 if batch_products == "auto" else 1
    )
    # Both schedules execute the same exact occupied-space arithmetic. The
    # fused schedule reduces submission count and retains bounded panels.
    assert c["response_occupied_projection_flops"] == 2 * 96 * (
        96 * 96 * 20 + 96 * 20 * 20
    )
    assert c["response_pseudo_density_flops"] == (
        2 * 96 * (96 * 20 * 20 + (64 * 64 + 32 * 96) * 20)
    )
    assert c["response_occupied_projection_blas_calls"] == (
        2 if batch_products == "auto" else 192
    )


@pytest.mark.parametrize("method", ["rhf", "uhf"])
def test_batched_full_response_keeps_rectangular_output(method, monkeypatch, tmp_path):
    """Exercise the full batched expansion with both nonempty UHF spin factors.

    Tiny full-output fixtures exhaust the shared tensor with their weight
    panel and fall back to serial expansion. At 192 AOs, a bounded panel and
    CU staging fit together, so this checks the actual batched full branch.
    """
    from pyscf import gto, scf

    assert os.environ.get("SLURM_JOB_ID")
    case = benchmark_cases()["water-octamer-s4-def2-svp-spherical"]
    mol = gto.M(atom=case.atoms, unit="Bohr", basis="def2-svp", verbose=0)
    reference = (scf.RHF if method == "rhf" else scf.UHF)(mol).density_fit(
        auxbasis="def2-svp"
    )
    reference.conv_tol, reference.conv_tol_grad, reference.max_cycle = 1e-12, 1e-10, 100
    reference.kernel()
    assert reference.converged
    expected = -reference.nuc_grad_method().kernel()
    for name, value in {
        "VIBEQC_DF_EXCHANGE": "occupied",
        "VIBEQC_DF_RESPONSE_STORAGE": "jk-scratch",
        "VIBEQC_DF_RESPONSE_SPACE": "occupied",
        "VIBEQC_DF_WEIGHTED_EXECUTION": "shell",
        "VIBEQC_DF_SHELL_SCHEDULE": "compact",
        "VIBEQC_DF_PRIMITIVE_BUCKETS": "packet",
        "VIBEQC_DF_DERIVATIVE_PAIRS": "full",
    }.items():
        monkeypatch.setenv(name, value)
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
    with calc.prepare_batch([case.atoms]) as batch:
        batch.execute(strict=True)
        batch.set_warm_start_updates(False)
        for policy in ("off", "auto"):
            monkeypatch.setenv("VIBEQC_DF_RESPONSE_BATCHING", policy)
            trace = tmp_path / f"full-{policy}.jsonl"
            monkeypatch.setenv("VIBEQC_DF_TRACE", str(trace))
            actual = batch.execute(strict=True).items[0]
            assert actual.energy == pytest.approx(reference.e_tot, abs=1e-9, rel=0)
            np.testing.assert_allclose(actual.forces, expected, atol=1e-8, rtol=0)
            (response,) = [
                row for row in read_trace(trace) if row["operation"] == "force_response"
            ]
            counters = response["counters"]
            assert counters["response_packed_pairs"] == 0
            assert counters["response_dense_weight_panel_elements"] > 0
            assert counters["response_full_weight_tensor_elements"] == 0
            spins = 2 if method == "uhf" else 1
            assert counters["response_pseudo_density_flops"] == spins * 2 * 192 * (
                192 * 40 * 40 + 192 * 192 * 40
            )
            if policy == "auto":
                assert counters["response_pseudo_density_batched_panels"] > 0
                assert counters["response_pseudo_density_products"] < spins * 2 * 192
                assert counters["response_occupied_projection_blas_calls"] == 2 * spins
