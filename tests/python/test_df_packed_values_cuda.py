"""Exact value packing must survive high-level preparation and cache replay."""

import os
import typing

import numpy as np
import pytest
from vibeqc import Calculator, ResourceBudget

from benchmarks._cases import benchmark_cases
from benchmarks.df_component_ledger import read_trace

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.mark.parametrize("response_space", ["auto", "dense", "occupied"])
def test_single_packed_factor_cold_warm_and_force_replay(
    monkeypatch: typing.Any, tmp_path: typing.Any, response_space: str
) -> None:
    """A retained fitted B must reproduce independent energies and forces.

    The raw owner is absent even when response recomputes physical values;
    changing the owner policy must rebuild the prepared value plan.
    """
    from pyscf import gto, scf

    assert os.environ.get("SLURM_JOB_ID")
    atoms = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.8)), ("H", (1.5, 0.0, -0.5))]
    moved = list(atoms)
    moved[1] = ("H", (0.0, 0.0, 1.801))
    references = []
    for geometry in (atoms, moved):
        molecule = gto.M(atom=geometry, unit="Bohr", basis="def2-svp", verbose=0)
        reference = scf.RHF(molecule).density_fit(auxbasis="def2-svp")
        reference.conv_tol = 1e-12
        reference.kernel()
        assert reference.converged
        references.append((reference.e_tot, -reference.nuc_grad_method().kernel()))

    monkeypatch.setenv("VIBEQC_DF_VALUE_STORAGE", "packed-single")
    monkeypatch.setenv("VIBEQC_DF_RESPONSE_SPACE", response_space)
    if response_space == "occupied":
        monkeypatch.setenv("VIBEQC_DF_EXCHANGE", "occupied")
        monkeypatch.setenv("VIBEQC_DF_RESPONSE_BUDGET_BYTES", "135000")
    calculator = Calculator(
        device="cuda",
        method="rhf",
        basis="def2-svp",
        basis_representation="spherical",
        auxiliary_basis="def2-svp",
        density_fitting="cuda",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    occupied_responses = 0
    borrowed_fitted_responses = 0
    with calculator.prepare_batch([atoms], warm_start=True) as batch:
        for index, geometry in enumerate((None, None, moved)):
            trace = tmp_path / f"single-packed-{index}.jsonl"
            monkeypatch.setenv("VIBEQC_DF_TRACE", str(trace))
            result = batch.execute(
                [np.asarray([position for _, position in geometry])]
                if geometry is not None
                else None,
                strict=True,
            )
            item = result.items[0]
            energy, forces = references[int(index == 2)]
            assert item.energy == pytest.approx(energy, abs=1e-8, rel=0)
            np.testing.assert_allclose(item.forces, forces, atol=1e-7, rtol=0)
            records = read_trace(trace)
            occupied_responses += sum(
                record["counters"].get("response_owned_occupied_projection_bytes", 0)
                > 0
                for record in records
                if record["operation"] == "force_response"
            )
            borrowed_fitted_responses += sum(
                record["counters"].get("response_borrowed_whitened_bytes", 0) > 0
                for record in records
                if record["operation"] == "force_response"
            )
            jk = [
                record
                for record in records
                if record["operation"] in ("ri_j", "ri_k", "ri_jk_shared")
            ]
            assert jk
            assert all(
                not record["counters"].get("raw_panel_source_auxiliary_evaluations", 0)
                and not record["counters"].get("shared_raw_source_values", 0)
                for record in jk
            )
            if index != 1:
                owners = [
                    record
                    for record in records
                    if record["operation"] == "resident_three_center_materialization"
                ]
                assert len(owners) == 1
                assert owners[0]["counters"].get("resident_raw_bytes", 0) == 0
                assert owners[0]["counters"].get("resident_transformed_bytes", 0) > 0
        monkeypatch.setenv("VIBEQC_DF_VALUE_STORAGE", "packed")
        trace = tmp_path / "dual-packed-after-single.jsonl"
        monkeypatch.setenv("VIBEQC_DF_TRACE", str(trace))
        result = batch.execute(strict=True).items[0]
        assert result.energy == pytest.approx(references[0][0], abs=1e-8, rel=0)
        np.testing.assert_allclose(result.forces, references[0][1], atol=1e-7, rtol=0)
        owners = [
            record
            for record in read_trace(trace)
            if record["operation"] == "resident_three_center_materialization"
        ]
        assert len(owners) == 1
        assert owners[0]["counters"].get("resident_raw_bytes", 0) > 0
    if response_space == "occupied":
        assert occupied_responses > 0
    elif response_space == "auto":
        assert occupied_responses == 0
        assert borrowed_fitted_responses > 0


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("budget", [0, 256 << 20])
@pytest.mark.parametrize(
    "method,multiplicity,representation,auxiliary,oh",
    [
        ("rhf", 1, "cartesian", "def2-svp", False),
        ("uhf", 3, "spherical", "sto-3g", False),
        ("uhf", 2, "spherical", "def2-svp", True),
    ],
)
def test_packed_preparation_replay_and_representation_replacement(
    monkeypatch: typing.Any,
    tmp_path: typing.Any,
    batch_size: typing.Any,
    budget: typing.Any,
    method: typing.Any,
    multiplicity: typing.Any,
    representation: typing.Any,
    auxiliary: typing.Any,
    oh: typing.Any,
) -> None:
    """Independent forces, unequal auxiliaries, empty spin and changed geometry.

    Reuse one prepared Python owner while changing the value representation.
    Native host metadata and captured device plans must both follow the change,
    including the zero-budget path that formerly built complete host raw A.
    """
    from pyscf import gto, scf

    assert os.environ.get("SLURM_JOB_ID")
    atoms = (
        benchmark_cases()["oh-def2-svp-spherical-uhf"].atoms
        if oh
        else [("H", (0.0, 0.0, -0.7)), ("H", (0.1, 0.0, 0.7))]
    )
    moved = [(symbol, np.array(position, dtype=float)) for symbol, position in atoms]
    moved[1][1][0] += 0.001
    references = []
    for geometry in (atoms, moved):
        mol = gto.M(
            atom=geometry,
            unit="Bohr",
            basis="def2-svp",
            cart=representation == "cartesian",
            spin=multiplicity - 1,
            verbose=0,
        )
        oracle = (scf.RHF if method == "rhf" else scf.UHF)(mol).density_fit(
            auxbasis=auxiliary
        )
        oracle.conv_tol, oracle.conv_tol_grad, oracle.max_cycle = 1e-12, 1e-10, 100
        oracle.kernel()
        assert oracle.converged
        references.append((oracle.e_tot, -oracle.nuc_grad_method().kernel()))
    monkeypatch.setenv("VIBEQC_DF_EXCHANGE", "occupied")
    monkeypatch.setenv("VIBEQC_DF_SEED_EXCHANGE", "dense")
    monkeypatch.setenv("VIBEQC_DF_RESPONSE_STORAGE", "auto")
    monkeypatch.setenv("VIBEQC_DF_RESPONSE_SPACE", "occupied")
    monkeypatch.setenv("VIBEQC_DF_WEIGHTED_EXECUTION", "shell")
    monkeypatch.setenv("VIBEQC_DF_SHELL_SCHEDULE", "compact")
    monkeypatch.setenv("VIBEQC_DF_DERIVATIVE_PAIRS", "packed")
    calc = Calculator(
        method=method,
        basis="def2-svp",
        basis_representation=representation,
        auxiliary_basis=auxiliary,
        device="cuda",
        density_fitting="cuda",
        density_fitting_memory_budget_bytes=budget,
        density_fitting_relative_threshold=1e-12,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        max_iterations=100,
    )
    with calc.prepare_batch(
        [atoms] * batch_size, multiplicities=[multiplicity] * batch_size
    ) as batch:
        for step, (storage, changed) in enumerate(
            [
                ("dense", False),
                ("packed", False),
                ("packed", False),
                ("packed", True),
                ("dense", True),
            ]
        ):
            monkeypatch.setenv("VIBEQC_DF_VALUE_STORAGE", storage)
            trace = tmp_path / f"values-{step}.jsonl"
            monkeypatch.setenv("VIBEQC_DF_TRACE", str(trace))
            result = batch.execute(
                [np.array([r for _, r in moved])] * batch_size if changed else None,
                strict=True,
            )
            energy, force = references[int(changed)]
            for item in result.items:
                assert item.executed_backend == "cuda"
                assert item.energy == pytest.approx(energy, abs=1e-9, rel=0)
                np.testing.assert_allclose(item.forces, force, atol=1e-8, rtol=0)
            records = read_trace(trace)
            packed_setup = [
                r for r in records if r["counters"].get("packed_raw_generation_calls")
            ]
            if storage == "packed":
                if step != 2:
                    assert packed_setup, (
                        "packing was not selected before physical source creation"
                    )
                else:
                    assert not packed_setup, (
                        "unchanged packed geometry rebuilt its source"
                    )
                responses = [r for r in records if r["operation"] == "force_response"]
                assert len(responses) == batch_size
                for response in responses:
                    n, a, counts = (
                        response["nbf"],
                        response["naux"],
                        response["counters"],
                    )
                    assert (
                        counts["raw_packed_value_reused_bytes"]
                        == n * (n + 1) // 2 * a * 8
                    )
                    assert not counts.get("raw_value_upload_bytes", 0)
            else:
                assert not packed_setup, "dense cache replay retained a packed owner"


def test_packed_global_ledger_and_raw_reuse_ablation(
    monkeypatch: typing.Any, tmp_path: typing.Any
) -> None:
    """Charge live packed owners and reject a changed admitted representation.

    Raw-reuse off exercises bounded source regeneration through the same value
    plan. It is a diagnostic control and receives its own resource identity.
    """
    assert os.environ.get("SLURM_JOB_ID")
    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.1, 0.0, 0.7))]
    options = {
        "device": "cuda",
        "density_fitting": "cuda",
        "basis": "def2-svp",
        "auxiliary_basis": "def2-svp",
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
    }
    monkeypatch.setenv("VIBEQC_DF_VALUE_STORAGE", "dense")
    reference = Calculator(**options).singlepoint(atoms)
    monkeypatch.setenv("VIBEQC_DF_VALUE_STORAGE", "packed")
    for raw_reuse in ("auto", "off"):
        monkeypatch.setenv("VIBEQC_DF_RAW_REUSE", raw_reuse)
        probe = Calculator(**options).estimate_resources([atoms] * 2).require_feasible()
        budget = ResourceBudget(
            host_bytes=probe.peak_bytes["host"], device_bytes=probe.peak_bytes["device"]
        )
        calc = Calculator(**options, resource_budget=budget)
        with calc.prepare_batch([atoms] * 2) as batch:
            for replay, properties in enumerate(
                (("energy", "forces"), ("energy",), ("energy", "forces"))
            ):
                trace = tmp_path / f"raw-{raw_reuse}-{replay}.jsonl"
                monkeypatch.setenv("VIBEQC_DF_TRACE", str(trace))
                result = batch.execute(strict=True, properties=properties)
                ledger = batch.resource_diagnostics["observation"]["device_ledger"]
                assert (
                    0
                    < ledger["live_bytes"]
                    <= ledger["peak_bytes"]
                    <= ledger["limit_bytes"]
                )
                assert ledger["rejected_allocations"] == 0
                for item in result.items:
                    assert item.executed_backend == "cuda"
                    assert item.energy == pytest.approx(
                        reference.energy, abs=1e-9, rel=0
                    )
                    if "forces" in properties:
                        np.testing.assert_allclose(
                            item.forces, reference.forces, atol=1e-8, rtol=0
                        )
                if raw_reuse == "off":
                    assert all(
                        not r["counters"].get("raw_packed_value_reused_bytes", 0)
                        for r in read_trace(trace)
                    )
            monkeypatch.setenv("VIBEQC_DF_VALUE_STORAGE", "dense")
            with pytest.raises(ValueError, match="schedule changed"):
                batch.execute(strict=True)
            monkeypatch.setenv("VIBEQC_DF_VALUE_STORAGE", "packed")


@pytest.mark.parametrize("spin", ["restricted", "unrestricted"])
@pytest.mark.parametrize("storage", ["packed", "packed-single"])
def test_packed_composed_fock_keeps_prepared_identity_and_dense_fallback(
    monkeypatch: typing.Any, spin: typing.Any, storage: typing.Any
) -> None:
    """Unknown-rank Fock inputs retain exact bounded K and frozen provenance."""
    from vibeqc.fock import FockBuildSpec, FockPlan
    from vibeqc_compiler.dft import NativeAO

    assert os.environ.get("SLURM_JOB_ID")
    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    spec = FockBuildSpec.hf(spin, coulomb="density_fitted", exchange="density_fitted")
    d = np.array([[0.8, 0.1], [0.1, 0.6]])
    if spin == "unrestricted":
        d = np.stack((d, 0.4 * d))
    monkeypatch.setenv("VIBEQC_DF_VALUE_STORAGE", "dense")
    with (
        NativeAO(atoms) as basis,
        FockPlan(basis, spec, device="cpu") as oracle,
        FockPlan(basis, spec, device="cuda", device_budget_bytes=16 << 20) as dense,
    ):
        monkeypatch.setenv("VIBEQC_DF_VALUE_STORAGE", storage)
        with FockPlan(
            basis, spec, device="cuda", device_budget_bytes=16 << 20
        ) as packed:
            assert packed.identity == dense.identity
            assert packed.execution_identity != dense.execution_identity
            assert dense.diagnostics["df_pair_storage"] == "dense"
            assert packed.diagnostics["df_pair_storage"] == storage
            monkeypatch.setenv("VIBEQC_DF_VALUE_STORAGE", "dense")
            assert packed.diagnostics["df_pair_storage"] == storage
            for symmetric in (True, False):
                density = d.copy()
                if not symmetric:
                    density[..., 0, 1] += 0.17
                expected = oracle.evaluate(density, derivative=symmetric)
                result = packed.evaluate(density, derivative=symmetric)
                assert result.energy == pytest.approx(expected.energy, abs=1e-10, rel=0)
                np.testing.assert_allclose(
                    result.fock, expected.fock, atol=1e-10, rtol=0
                )
                if symmetric:
                    np.testing.assert_allclose(
                        result.gradient, expected.gradient, atol=1e-9, rtol=0
                    )
