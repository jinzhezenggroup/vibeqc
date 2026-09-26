"""Compact DF DIIS keeps independent spin/force accuracy and resets warm history."""

import os
import typing

import numpy as np
import pytest
from vibeqc import Calculator

from benchmarks.df_component_ledger import aggregate_host, read_host_trace
from benchmarks.df_progress_ledger import read_progress, summarize_progress

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.mark.parametrize("method", ("rhf", "uhf"))
@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
@pytest.mark.parametrize("size", (1, 4))
@pytest.mark.parametrize("exchange", ("dense", "occupied"))
@pytest.mark.parametrize("dots", ("serial", "auto"))
def test_compact_diis_independent_forces_and_warm_reset(
    method: typing.Any,
    representation: typing.Any,
    size: typing.Any,
    exchange: typing.Any,
    dots: typing.Any,
    monkeypatch: typing.Any,
    tmp_path: typing.Any,
) -> None:
    from pyscf import gto, scf

    assert os.environ.get("SLURM_JOB_ID")
    atoms = [("O", (0, 0, 0)), ("H", (0, 0, 1.8)), ("H", (1.7, 0, -0.6))]
    if method == "uhf":
        atoms.pop()  # OH doublet has two distinct spin residuals.
    changed = np.array([atom[1] for atom in atoms])
    changed[-1, 0] += 0.02
    changed_atoms = [(a[0], tuple(xyz)) for a, xyz in zip(atoms, changed, strict=True)]
    references = []
    for geometry in (atoms, changed_atoms):
        mol = gto.M(
            atom=geometry,
            basis="def2-svp",
            unit="Bohr",
            cart=representation == "cartesian",
            spin=int(method == "uhf"),
            verbose=0,
        )
        mf = (scf.UHF(mol) if method == "uhf" else scf.RHF(mol)).density_fit(
            auxbasis="def2-svp"
        )
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-10, 100
        mf.kernel()
        assert mf.converged
        gradient = mf.nuc_grad_method()
        gradient.auxbasis_response = True
        references.append((mf.e_tot, -gradient.kernel()))
    monkeypatch.setenv("VIBEQC_DF_EXCHANGE", exchange)
    monkeypatch.setenv("VIBEQC_DF_DIIS_DOTS", dots)
    monkeypatch.delenv("VIBEQC_DF_DISABLE_DEVICE_DIIS", raising=False)
    calc = Calculator(
        method=method,
        basis="def2-svp",
        basis_representation=representation,
        device="cuda",
        density_fitting="cuda",
        # Zero retains the preparation/graph owner on identical-property warm
        # calls. Positive-budget replanning is covered by the resource and
        # property-transition suites; rebuilding cannot test history reset.
        density_fitting_memory_budget_bytes=0,
        diis_history=8,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    with calc.prepare_batch(
        [atoms] * size, multiplicities=[2 if method == "uhf" else 1] * size
    ) as owner:
        for step, (positions, properties) in enumerate(
            (
                (None, ("energy", "forces")),
                (None, ("energy", "forces")),
                (None, ("energy",)),
                (None, ("energy", "forces")),
                ([None] * (size - 1) + [changed], ("energy", "forces")),
            )
        ):
            progress_path = tmp_path / f"progress-{step}.jsonl"
            host_path = tmp_path / f"host-{step}.jsonl"
            monkeypatch.setenv("VIBEQC_DF_PROGRESS_TRACE", str(progress_path))
            monkeypatch.setenv("VIBEQC_DF_HOST_TRACE", str(host_path))
            output = owner.execute(positions, properties=properties, strict=True)
            monkeypatch.delenv("VIBEQC_DF_PROGRESS_TRACE")
            monkeypatch.delenv("VIBEQC_DF_HOST_TRACE")
            for slot, item in enumerate(output.items):
                expected = references[int(step == 4 and slot == size - 1)]
                assert item.converged and 1 <= item.iterations <= 100
                assert abs(item.energy - expected[0]) < 1e-9
                if "forces" in properties:
                    np.testing.assert_allclose(
                        item.forces, expected[1], atol=1e-8, rtol=0
                    )
            progress = summarize_progress(read_progress(progress_path))
            assert any(
                r["key"] == "diis_history" and r["value"] == 8
                for r in progress["observations"]
            )
            if step == 0:
                assert any(
                    r["name"] == "compact_diis_update" for r in progress["observations"]
                )
            if step == 1:
                assert "host:df_plan_setup" not in progress["phases"]
                assert any(
                    r["key"] == "host_graph_replay"
                    or (r["key"] == "warm_seed_reused" and r["value"] == 1)
                    for r in progress["observations"]
                )
            assert "host:diis_retry" not in progress["phases"]
            assert not aggregate_host(read_host_trace(host_path))[
                "eigensolves_by_reason"
            ]


def test_diis_workspace_shortfall_fails_without_numerical_retry(
    monkeypatch: typing.Any, tmp_path: typing.Any
) -> None:
    """A force half-budget cannot borrow response capacity for solver owners."""
    assert os.environ.get("SLURM_JOB_ID")
    path = tmp_path / "shortfall.jsonl"
    monkeypatch.setenv("VIBEQC_DF_PROGRESS_TRACE", str(path))
    calc = Calculator(
        basis="def2-svp",
        device="cuda",
        density_fitting="cuda",
        density_fitting_memory_budget_bytes=4 << 20,
        diis_history=8,
    )
    with calc.prepare_batch([[("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]]) as owner:
        result = owner.execute(properties=("energy", "forces"))
        assert result.items[0].status_message == "out of memory"
    progress = summarize_progress(read_progress(path))
    assert "host:diis_retry" not in progress["phases"]
