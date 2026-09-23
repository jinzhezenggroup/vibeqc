"""Retiring native psss must preserve the explicit bounded registry-gap route."""

import json
import os
from collections.abc import Callable

import numpy as np
import pytest
from vibeqc import Calculator, Primitive, Shell

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)


@pytest.mark.parametrize(
    ("method", "charge", "multiplicity"), (("rhf", 1, 1), ("uhf", 0, 2))
)
def test_bounded_psss_registry_gap_matches_libcint(
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
    method: str,
    charge: int,
    multiplicity: int,
) -> None:
    """Exercise the generic order-one and order-three value drains together."""
    from pyscf import gto, scf

    assert os.environ.get("SLURM_JOB_ID")
    atoms = (("He", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7)))
    # Twenty-one Cartesian AOs exceed the persistent-ERI admission threshold.
    # Keeping both p and f shells catches either side of the merged drain cutoffs.
    basis = tuple(
        Shell(0, angular, (Primitive(exponent, 1.0),))
        for angular, exponent in ((0, 1.5), (1, 0.9), (2, 0.8), (3, 0.6))
    ) + (Shell(1, 0, (Primitive(1.2, 1.0),)),)
    monkeypatch.delenv("VIBEQC_DIRECT_TILE_VALIDATION", raising=False)
    results = []
    ledgers = []
    for bounded, fock_classes in (("none", "all"), ("force", "ssss")):
        monkeypatch.setenv("VIBEQC_BOUNDED_DIRECT_STREAMING", bounded)
        monkeypatch.setenv("VIBEQC_AOT_FOCK_SHELL_CLASSES", fock_classes)
        with Calculator(
            method=method,
            basis=basis,
            device="cuda",
            density_fitting="none",
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
            screening_tolerance=1e-14,
        ).prepare_batch(
            [atoms],
            charges=[charge],
            multiplicities=[multiplicity],
            shell_class_profiling=True,
        ) as prepared:
            results.append(prepared.execute(strict=True).items[0])
            ledgers.append(prepared.last_shell_class_profile())
    assert ledgers[0] == ledgers[1]
    for label in ("psss", "fsss"):
        assert any(row.label == label and row.shell_quartets for row in ledgers[1])

    molecule = gto.M(
        atom=atoms,
        basis={
            "He": [[l, [e, 1.0]] for l, e in ((0, 1.5), (1, 0.9), (2, 0.8), (3, 0.6))],
            "H": [[0, [1.2, 1.0]]],
        },
        unit="Bohr",
        cart=True,
        charge=charge,
        spin=multiplicity - 1,
        verbose=0,
    )
    reference = (scf.RHF if method == "rhf" else scf.UHF)(molecule)
    reference.conv_tol = 1e-13
    reference.conv_tol_grad = 1e-10
    reference.kernel()
    assert reference.converged
    forces = -reference.nuc_grad_method().kernel()
    # Retain independent outputs alongside the strict gate so qualification
    # records can recompute numerical errors without rerunning a GPU job.
    record_property(
        "oracle_comparison",
        json.dumps(
            {
                "reference": {"energy": reference.e_tot, "forces": forces.tolist()},
                "results": [
                    {"energy": result.energy, "forces": result.forces.tolist()}
                    for result in results
                ],
            }
        ),
    )
    for result in results:
        assert result.executed_backend == "cuda"
        assert result.energy == pytest.approx(reference.e_tot, abs=2e-10)
        np.testing.assert_allclose(result.forces, forces, atol=6e-9, rtol=0)
