"""Generated value math preserves complete native public/source semantics."""

import os

import numpy as np
import pytest
from vibeqc import Calculator

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1",
    reason="requires a finite Slurm GPU allocation",
)


@pytest.mark.parametrize("method", ["rhf", "uhf"])
@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize("budget", [0, 16 << 20])
def test_value_math_replay_and_geometry(method, representation, budget, monkeypatch):
    """Resident/export and bounded/generated sources use the same independent gates."""
    from pyscf import gto, scf

    assert os.environ.get("SLURM_JOB_ID")
    atoms = [("O", (0, 0, 0)), ("H", (0, 0, 1.8))]
    if method == "rhf":
        atoms.append(("H", (1.7, 0, -0.6)))
    spin = int(method == "uhf")
    geometries = []
    references = []
    for displacement in (0.0, 0.003):
        coordinates = np.array([r for _, r in atoms], dtype=float)
        coordinates[1, 0] += displacement
        geometries.append(coordinates)
        mol = gto.M(
            atom=[(z, r) for (z, _), r in zip(atoms, coordinates)],
            unit="Bohr",
            basis="def2-svp",
            cart=representation == "cartesian",
            spin=spin,
            verbose=0,
        )
        ref = (scf.UHF if spin else scf.RHF)(mol).density_fit(auxbasis="def2-svp")
        ref.conv_tol, ref.conv_tol_grad, ref.max_cycle = 1e-13, 1e-10, 100
        ref.kernel()
        assert ref.converged
        references.append((ref.e_tot, -ref.nuc_grad_method().kernel()))
    for math in ("generic", "polynomial", "rys", "candidate"):
        monkeypatch.setenv("VIBEQC_DF_VALUE_MATH", math)
        monkeypatch.setenv(
            "VIBEQC_DF_VALUE_RAW_MAPPING",
            {
                "generic": "scalar",
                "polynomial": "subgroup",
                "rys": "warp",
                "candidate": "candidate",
            }[math],
        )
        calc = Calculator(
            method=method,
            basis="def2-svp",
            basis_representation=representation,
            device="cuda",
            density_fitting="cuda",
            max_iterations=100,
            density_fitting_memory_budget_bytes=budget,
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
        )
        with calc.prepare_batch([atoms], multiplicities=[spin + 1]) as owner:
            for geometry in (0, 0, 1):
                actual = owner.execute([geometries[geometry]], strict=True).items[0]
                energy, force = references[geometry]
                assert actual.energy == pytest.approx(energy, abs=1e-9, rel=0)
                np.testing.assert_allclose(actual.forces, force, atol=1e-8, rtol=0)
