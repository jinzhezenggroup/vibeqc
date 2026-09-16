"""Resident response must lend back J/K scratch without changing full forces."""

import os

import numpy as np
import pytest
from vibeqc import Calculator, Primitive, Shell

from benchmarks._cases import benchmark_cases
from benchmarks.df_component_ledger import read_trace

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


def select_response(monkeypatch, storage):
    """Explicit selectors keep this experiment independent of size promotion."""
    assert os.environ.get("SLURM_JOB_ID")
    monkeypatch.setenv("VIBEQC_DF_RESPONSE_STORAGE", storage)
    monkeypatch.setenv("VIBEQC_DF_RESPONSE_ALGEBRA", "blas")
    monkeypatch.setenv("VIBEQC_DF_WEIGHTED_EXECUTION", "shell")
    monkeypatch.setenv("VIBEQC_DF_SHELL_SCHEDULE", "compact")
    monkeypatch.setenv("VIBEQC_DF_RAW_STAGING", "pinned-panels")
    monkeypatch.setenv("VIBEQC_DF_RESPONSE_BUDGET_BYTES", str(4 << 20))


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("exchange", ["dense", "occupied"])
@pytest.mark.parametrize(
    "case_name", ["water-tetramer-def2-svp-spherical", "oh-def2-svp-spherical-uhf"]
)
def test_jk_scratch_survives_response_property_and_geometry_replays(
    monkeypatch, tmp_path, batch_size, exchange, case_name
):
    """Both spins, batch items and the next SCF reuse the same scratch owners."""
    from pyscf import gto, scf

    monkeypatch.setenv("VIBEQC_DF_EXCHANGE", exchange)
    case = benchmark_cases()[case_name]
    moved = [(z, np.array(r, dtype=float)) for z, r in case.atoms]
    moved[1][1][0] += 0.001
    references = []
    for atoms in (case.atoms, moved):
        molecule = gto.M(
            atom=atoms,
            unit="Bohr",
            basis=case.pyscf_basis,
            cart=False,
            charge=case.charge,
            spin=case.multiplicity - 1,
            verbose=0,
        )
        oracle = (scf.UHF if case.method == "uhf" else scf.RHF)(molecule).density_fit(
            auxbasis=case.pyscf_basis
        )
        oracle.conv_tol, oracle.conv_tol_grad, oracle.max_cycle = 1e-12, 1e-10, 100
        oracle.kernel()
        assert oracle.converged
        references.append((oracle.e_tot, -oracle.nuc_grad_method().kernel()))
    calc = Calculator(
        method=case.method,
        basis=case.vibeqc_basis,
        basis_representation="spherical",
        device="cuda",
        density_fitting="cuda",
        auxiliary_basis=case.vibeqc_basis,
        max_iterations=100,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    coordinates = np.array([r for _, r in moved])
    with calc.prepare_batch(
        [case.atoms] * batch_size,
        charges=[case.charge] * batch_size,
        multiplicities=[case.multiplicity] * batch_size,
    ) as batch:
        for step, (storage, force, changed) in enumerate(
            (
                ("panel", True, False),
                ("jk-scratch", True, False),
                ("jk-scratch", False, False),
                ("jk-scratch", True, False),
                ("panel", True, False),
                ("jk-scratch", True, True),
                ("panel", True, True),
            )
        ):
            select_response(monkeypatch, storage)
            trace = tmp_path / f"stage-{step}.jsonl"
            monkeypatch.setenv("VIBEQC_DF_TRACE", str(trace))
            result = batch.execute(
                [coordinates] * batch_size if changed else None,
                strict=True,
                properties=("energy", "forces") if force else ("energy",),
            )
            expected_energy, expected_force = references[int(changed)]
            for item in result.items:
                assert item.energy == pytest.approx(expected_energy, abs=1e-9, rel=0)
                if force:
                    np.testing.assert_allclose(
                        item.forces, expected_force, atol=1e-8, rtol=0
                    )
            if force:
                responses = [
                    r for r in read_trace(trace) if r["operation"] == "force_response"
                ]
                assert len(responses) == batch_size
                for response in responses:
                    counters = response["counters"]
                    n, a = response["nbf"], response["naux"]
                    assert counters["response_scratch_bytes"] <= 4 << 20
                    if storage == "jk-scratch":
                        assert (
                            counters["response_borrowed_jk_bytes"] == 3 * n * n * a * 8
                        )
                        # One system retains raw device values in former K
                        # scratch; batch scratch keeps its explicit upload.
                        uploads = int(batch_size != 1)
                        assert (
                            counters["raw_value_upload_bytes"]
                            == uploads * n * n * a * 8
                        )
                        assert counters["raw_value_bulk_uploads"] == uploads
                        assert counters.get("raw_value_reused_bytes", 0) == (
                            (1 - uploads) * n * n * a * 8
                        )
                        assert counters["response_ao_matrix_products"] == 2 * a * (
                            2 if case.method == "uhf" else 1
                        )
                        assert counters["response_auxiliary_blocks"] == 1
                        assert not counters.get("raw_panel_host_gather_elements", 0)
                    else:
                        assert not counters.get("response_borrowed_jk_bytes", 0)


@pytest.mark.parametrize("space", ["dense", "occupied"])
@pytest.mark.parametrize("pairs", ["full", "packed"])
def test_jk_scratch_retains_discarded_metric_response(
    monkeypatch, tmp_path, space, pairs
):
    """An unequal near-duplicate auxiliary pair has a finite discarded mode."""
    monkeypatch.setenv("VIBEQC_DF_FINAL_PROJECTION", "reuse")
    monkeypatch.setenv("VIBEQC_DF_FINAL_EXCHANGE", "occupied")
    monkeypatch.setenv("VIBEQC_DF_RESIDENT_EXCHANGE", "full")
    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.1, 0.0, 0.7))]
    basis = [Shell(i, 0, (Primitive(1.0, 1.0),)) for i in range(2)]
    auxiliary = [
        Shell(0, 0, (Primitive(0.6, 1.0),)),
        Shell(0, 0, (Primitive(0.601, 1.0),)),
        Shell(1, 0, (Primitive(0.8, 1.0),)),
    ]
    common = {
        "basis": basis,
        "auxiliary_basis": auxiliary,
        "density_fitting_relative_threshold": 1e-5,
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
        "max_iterations": 100,
    }
    reference = Calculator(device="cpu", density_fitting="cpu", **common).singlepoint(
        atoms
    )
    select_response(monkeypatch, "jk-scratch")
    monkeypatch.setenv("VIBEQC_DF_EXCHANGE", "occupied")
    monkeypatch.setenv("VIBEQC_DF_RESPONSE_SPACE", space)
    monkeypatch.setenv("VIBEQC_DF_DERIVATIVE_PAIRS", pairs)
    trace = tmp_path / "metric-response.jsonl"
    monkeypatch.setenv("VIBEQC_DF_TRACE", str(trace))
    calc = Calculator(device="cuda", density_fitting="cuda", **common)
    with calc.prepare_batch([atoms]) as batch:
        actual = batch.execute(strict=True).items[0]
        (diagnostic,) = batch.last_density_fitting_metric_diagnostics()
        assert diagnostic.effective_rank == 2
        np.testing.assert_allclose(actual.energy, reference.energy, atol=1e-9, rtol=0)
        np.testing.assert_allclose(actual.forces, reference.forces, atol=3e-9, rtol=0)
    responses = [r for r in read_trace(trace) if r["operation"] == "force_response"]
    assert len(responses) == 1
    assert responses[0]["counters"].get("response_final_projection_reused", 0) == 0
    assert (responses[0]["counters"].get("response_occupied_rank", 0) > 0) == (
        space == "occupied"
    )
    assert bool(responses[0]["counters"].get("response_packed_pairs", 0)) == (
        space == "occupied" and pairs == "packed"
    )
    for step in (2e-4, 5e-5):
        energies = []
        for sign in (1, -1):
            displaced = [(z, np.array(r, dtype=float)) for z, r in atoms]
            displaced[1][1][2] += sign * step
            energies.append(calc.singlepoint(displaced).energy)
        assert actual.forces[1, 2] == pytest.approx(
            -(energies[0] - energies[1]) / (2 * step), abs=2e-6, rel=0
        )


def test_jk_scratch_rejects_partial_source_plan(monkeypatch):
    """Retained B alone never authorizes borrowing full-size K buffers."""
    select_response(monkeypatch, "jk-scratch")
    monkeypatch.delenv("VIBEQC_DF_RESPONSE_BUDGET_BYTES")
    case = benchmark_cases()["water-tetramer-def2-svp-spherical"]
    calc = Calculator(
        basis=case.vibeqc_basis,
        basis_representation="spherical",
        device="cuda",
        density_fitting="cuda",
        density_fitting_memory_budget_bytes=32 << 20,
        max_iterations=100,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    with calc.prepare_batch([case.atoms]) as batch:
        with monkeypatch.context() as bounded:
            bounded.setenv("VIBEQC_DF_RESPONSE_STORAGE", "auto")
            expected = batch.execute(strict=True).items[0]
        # generated_df_hf_gradient converts the adapter's INVALID_ARGUMENT to
        # runtime_error; RHF finalization maps that to NUMERICAL_FAILURE, which
        # the batch API renders here. Establish a working source plan first so
        # this assertion cannot pass from a preparation failure.
        with pytest.raises(RuntimeError, match="numerical failure"):
            batch.execute(strict=True)
        monkeypatch.setenv("VIBEQC_DF_RESPONSE_STORAGE", "auto")
        actual = batch.execute(strict=True).items[0]
        assert actual.energy == pytest.approx(expected.energy, abs=1e-9, rel=0)
        np.testing.assert_allclose(actual.forces, expected.forces, atol=1e-8, rtol=0)


@pytest.mark.parametrize(
    ("control", "value"),
    [
        ("VIBEQC_DF_RESPONSE_STORAGE", "invalid"),
        ("VIBEQC_DF_RAW_REUSE", "invalid"),
        ("VIBEQC_DF_RESPONSE_BATCHING", "invalid"),
        ("VIBEQC_DF_DERIVATIVE_PAIRS", "invalid"),
        ("VIBEQC_DF_PACKED_AO_BLOCK_ROWS", "0"),
        ("VIBEQC_DF_PRIMITIVE_BUCKETS", "invalid"),
        ("VIBEQC_DF_RESPONSE_ALGEBRA", "scalar"),
        ("VIBEQC_DF_SERIAL_RESPONSE_DOT", "1"),
        ("VIBEQC_DF_RESPONSE_UPLOAD_PROBE", "packed"),
        ("VIBEQC_DF_RESPONSE_SCATTER_PROBE", "sharded"),
    ],
)
def test_jk_scratch_rejects_incompatible_controls_and_recovers(
    monkeypatch, control, value
):
    """A failed force must drain its borrowed stream before the next SCF replay."""
    atoms = [("H", (0, 0, -0.7)), ("H", (0.1, 0, 0.7))]
    expected = Calculator(device="cpu", density_fitting="cpu").singlepoint(atoms)
    select_response(monkeypatch, "jk-scratch")
    calc = Calculator(device="cuda", density_fitting="cuda")
    with calc.prepare_batch([atoms]) as batch:
        batch.execute(strict=True)
        with monkeypatch.context() as rejected:
            rejected.setenv(control, value)
            # The force callback throws runtime_error even for INVALID_ARGUMENT;
            # batch finalization therefore exposes NUMERICAL_FAILURE here.
            with pytest.raises(RuntimeError, match="numerical failure"):
                batch.execute(strict=True)
        # The storage selector supplies BLAS by default, without requiring an
        # additional environment variable even outside the promoted shape.
        monkeypatch.delenv("VIBEQC_DF_RESPONSE_ALGEBRA")
        actual = batch.execute(strict=True).items[0]
        assert actual.energy == pytest.approx(expected.energy, abs=1e-9, rel=0)
        np.testing.assert_allclose(actual.forces, expected.forces, atol=1e-8, rtol=0)


@pytest.mark.parametrize("model_change", ["orbital", "auxiliary", "metric"])
def test_raw_view_binds_model_and_survives_upload_ablation(
    monkeypatch, tmp_path, model_change
):
    """Equal AO dimensions never authorize reuse across a changed model owner.

    Toggle the upload diagnostic on an unchanged owner, then construct another
    model with the same sizes. Both remain independently checked against CPU
    analytic forces, and only the second model must receive a new identity.
    """
    select_response(monkeypatch, "jk-scratch")
    atoms = [("H", (0, 0, -0.7)), ("H", (0.1, 0, 0.7))]
    owners = []
    for changed in (False, True):
        basis = [
            Shell(
                i,
                0,
                (
                    Primitive(
                        1.03 if changed and model_change == "orbital" else 1.0, 1.0
                    ),
                ),
            )
            for i in range(2)
        ]
        auxiliary = [
            Shell(
                i % 2,
                0,
                (
                    Primitive(
                        e + (0.03 if changed and model_change == "auxiliary" else 0),
                        1.0,
                    ),
                ),
            )
            for i, e in enumerate((0.6, 0.8, 1.1))
        ]
        options = {
            "basis": basis,
            "auxiliary_basis": auxiliary,
            "density_fitting_relative_threshold": 1e-7
            if changed and model_change == "metric"
            else 1e-10,
            "energy_tolerance": 1e-12,
            "density_tolerance": 1e-10,
            "max_iterations": 100,
        }
        expected = Calculator(
            device="cpu", density_fitting="cpu", **options
        ).singlepoint(atoms)
        calc = Calculator(device="cuda", density_fitting="cuda", **options)
        current_owners = []
        with calc.prepare_batch([atoms]) as batch:
            for replay, policy in enumerate(("auto", "off", "auto")):
                monkeypatch.setenv("VIBEQC_DF_RAW_REUSE", policy)
                trace = tmp_path / f"model-{changed}-{replay}.jsonl"
                monkeypatch.setenv("VIBEQC_DF_TRACE", str(trace))
                actual = batch.execute(strict=True).items[0]
                assert actual.energy == pytest.approx(expected.energy, abs=1e-9, rel=0)
                np.testing.assert_allclose(
                    actual.forces, expected.forces, atol=1e-8, rtol=0
                )
                (response,) = [
                    r for r in read_trace(trace) if r["operation"] == "force_response"
                ]
                counters = response["counters"]
                assert counters["raw_value_bulk_uploads"] == int(policy == "off")
                if policy == "auto":
                    current_owners.append(counters["raw_value_owner_identity"])
        assert current_owners[0] == current_owners[1]
        owners.append(current_owners[0])
    assert owners[0] != owners[1]
