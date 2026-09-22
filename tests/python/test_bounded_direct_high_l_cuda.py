"""Bounded Direct-HF must retain exact high-angular Fock and force coverage."""

import json
import os
from collections.abc import Callable
from dataclasses import asdict

import numpy as np
import pytest
from vibeqc import Calculator, Primitive, Shell

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires an explicitly Slurm-allocated GPU",
)

ATOMS = (("He", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7)))
BASIS = (
    Shell(0, 0, (Primitive(1.5, 1.0),)),
    Shell(0, 2, (Primitive(0.8, 1.0),)),
    Shell(0, 3, (Primitive(0.6, 1.0),)),
    Shell(1, 0, (Primitive(1.2, 1.0),)),
)


@pytest.mark.parametrize(
    ("method", "charge", "multiplicity"),
    (("rhf", 1, 1), ("uhf", 0, 2)),
)
def test_forced_bounded_high_l_matches_fixed(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    charge: int,
    multiplicity: int,
    record_property: Callable[[str, object], None],
) -> None:
    """Registry gaps use the bounded high-l oracle instead of failing late."""

    assert os.environ.get("SLURM_JOB_ID")
    monkeypatch.delenv("VIBEQC_BOUNDED_DIRECT_STREAMING", raising=False)
    monkeypatch.delenv("VIBEQC_DIRECT_TILE_VALIDATION", raising=False)
    options = {
        "method": method,
        "basis": BASIS,
        "device": "cuda",
        "density_fitting": "none",
        "energy_tolerance": 1.0e-12,
        "density_tolerance": 1.0e-10,
        "screening_tolerance": 1.0e-14,
    }
    # This suite is explicitly admitted by a GPU allocation. Runtime failures
    # must fail the gate rather than masquerading as unavailable-device skips.
    with Calculator(**options).prepare_batch(
        [ATOMS],
        charges=[charge],
        multiplicities=[multiplicity],
        shell_class_profiling=True,
    ) as prepared:
        fixed = prepared.execute(strict=True).items[0]
        fixed_work = prepared.last_shell_class_profile()

    monkeypatch.setenv("VIBEQC_BOUNDED_DIRECT_STREAMING", "force")
    with Calculator(**options).prepare_batch(
        [ATOMS],
        charges=[charge],
        multiplicities=[multiplicity],
        shell_class_profiling=True,
    ) as prepared:
        bounded = prepared.execute(strict=True).items[0]
        bounded_work = prepared.last_shell_class_profile()

    assert fixed.executed_backend == bounded.executed_backend == "cuda"
    assert bounded.energy == pytest.approx(fixed.energy, abs=2.0e-10)
    np.testing.assert_allclose(bounded.forces, fixed.forces, atol=6.0e-9, rtol=0)
    # Generated, native, and fallback routes must own the same final-density
    # shell/AO/primitive domain, with no omitted or double-counted quartets.
    assert fixed_work == bounded_work
    assert any(row.label == "fsss" and row.shell_quartets for row in bounded_work)
    record_property("work_counts", json.dumps([asdict(row) for row in bounded_work]))

    # Both routes share retained recurrence code; their agreement alone cannot
    # establish the mathematics. Recompute the Cartesian reference with libcint.
    from pyscf import gto, scf

    molecule = gto.M(
        atom=ATOMS,
        basis={
            "He": [[0, [1.5, 1.0]], [2, [0.8, 1.0]], [3, [0.6, 1.0]]],
            "H": [[0, [1.2, 1.0]]],
        },
        unit="Bohr",
        cart=True,
        charge=charge,
        spin=multiplicity - 1,
        verbose=0,
    )
    reference = (scf.RHF if method == "rhf" else scf.UHF)(molecule)
    reference.conv_tol = 1.0e-13
    reference.conv_tol_grad = 1.0e-10
    reference.kernel()
    assert reference.converged
    assert bounded.energy == pytest.approx(reference.e_tot, abs=2.0e-10)
    oracle_forces = -reference.nuc_grad_method().kernel()
    np.testing.assert_allclose(bounded.forces, oracle_forces, atol=6.0e-9, rtol=0)
    record_property("oracle_energy_error", abs(bounded.energy - reference.e_tot))
    record_property(
        "oracle_force_error", float(np.max(np.abs(bounded.forces - oracle_forces)))
    )


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize(
    ("method", "charge", "multiplicity"),
    (("rhf", 1, 1), ("uhf", 0, 2)),
)
def test_tile_validation_is_structural_only(
    monkeypatch: pytest.MonkeyPatch,
    bounded: bool,
    method: str,
    charge: int,
    multiplicity: int,
) -> None:
    """A successful descriptor diagnostic cannot return an SCF result."""
    assert os.environ.get("SLURM_JOB_ID")
    monkeypatch.setenv("VIBEQC_DIRECT_TILE_VALIDATION", "validate")
    monkeypatch.setenv(
        "VIBEQC_BOUNDED_DIRECT_STREAMING", "force" if bounded else "none"
    )
    calculator = Calculator(
        method=method, basis=BASIS, device="cuda", density_fitting="none"
    )
    with pytest.raises(NotImplementedError):
        calculator.singlepoint(ATOMS, charge=charge, multiplicity=multiplicity)
