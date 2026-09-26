"""Independent checks of complete native RHF state and actual force W exports."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np

from benchmarks._cases import benchmark_cases
from benchmarks.issue308_stage_probe import sha256

if TYPE_CHECKING:
    from pathlib import Path


def validate_scf_export(
    fixture: Path,
    arrays_path: Path,
    endpoint: dict,
    *,
    forces: bool,
    checkpoint: Path | None = None,
    reference_forces: Path | None = None,
) -> dict:
    """Compare every component using unchanged energy/force/state tolerances.

    A retained checkpoint isolates the native solver/response from independent
    reference preparation. Its geometry, basis, identity and density must match
    the fixed-D fixture. If absent, converge a fresh independent PySCF state.
    W is read from the native force finalizer, never substituted with a matrix
    reconstructed from the exported orbitals.
    """
    from pyscf import gto, lib, scf

    metadata = json.loads((fixture / "input.json").read_text())
    case = benchmark_cases()[metadata["case"]]
    mol = gto.M(
        atom=case.atoms, basis=case.pyscf_basis, unit="Bohr", cart=False, verbose=0
    )
    mf = scf.RHF(mol).density_fit(auxbasis=case.pyscf_basis)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-12, 1e-10, 100
    if checkpoint:
        expected = metadata.get("checkpoint_sha256")
        if expected and sha256(checkpoint) != expected:
            raise ValueError("independent SCF checkpoint identity mismatch")
        saved_mol = lib.chkfile.load_mol(str(checkpoint))
        assert saved_mol._basis == mol._basis
        np.testing.assert_array_equal(saved_mol.atom_charges(), mol.atom_charges())
        np.testing.assert_allclose(
            saved_mol.atom_coords(), mol.atom_coords(), atol=1e-14, rtol=0
        )
        saved = lib.chkfile.load(str(checkpoint), "scf")
        mf.mo_coeff, mf.mo_occ, mf.mo_energy = (
            saved[k] for k in ("mo_coeff", "mo_occ", "mo_energy")
        )
        mf.e_tot = saved["e_tot"]
    else:
        mf.kernel()
        if not mf.converged:
            raise ValueError("independent reference did not converge")
    n, rank = mol.nao, mol.nelectron // 2
    values = np.fromfile(arrays_path)
    expected_size = 5 * n * n + n + (n * n + 3 * mol.natm if forces else 0)
    if values.size != expected_size or not np.isfinite(values).all():
        raise ValueError("native physical export is incomplete or nonfinite")
    density, overlap, hcore, fock, coeff = values[: 5 * n * n].reshape(5, n, n)
    eps = values[5 * n * n : 5 * n * n + n]
    maximum = lambda value: float(np.max(np.abs(value)))
    with np.load(fixture / "reference.npz") as reference:
        np.testing.assert_allclose(
            mf.make_rdm1(), reference["density"], atol=1e-8, rtol=0
        )
        errors = {
            "density_vs_pyscf": maximum(density - reference["density"]),
            "overlap_vs_pyscf": maximum(overlap - reference["overlap"]),
            "fock_vs_pyscf": maximum(
                fock - (hcore + reference["j"] - 0.5 * reference["k"])
            ),
        }
    errors.update(
        eigen_residual=maximum(fock @ coeff - (overlap @ coeff) * eps),
        orthogonality=maximum(coeff.T @ overlap @ coeff - np.eye(n)),
        electron_trace=abs(float(np.einsum("ij,ji", density, overlap)) - mol.nelectron),
        idempotency=maximum(density @ overlap @ density - 2 * density),
        commutator=maximum(fock @ density @ overlap - overlap @ density @ fock),
        canonical_density=maximum(density - 2 * coeff[:, :rank] @ coeff[:, :rank].T),
        energy_vs_pyscf=abs(endpoint["energy"] - float(mf.e_tot)),
        energy_reconstruction=abs(
            endpoint["energy"]
            - (0.5 * np.einsum("ij,ij", density, hcore + fock) + mol.energy_nuc())
        ),
    )
    net_force = None
    if forces:
        start = 5 * n * n + n
        weighted = values[start : start + n * n].reshape(n, n)
        actual_forces = values[start + n * n :].reshape(mol.natm, 3)
        expected_w = (mf.mo_coeff * (mf.mo_occ * mf.mo_energy)) @ mf.mo_coeff.T
        errors["actual_weighted_density_vs_pyscf"] = maximum(weighted - expected_w)
        errors["actual_weighted_density_vs_exported_orbitals"] = maximum(
            weighted - 2 * (coeff[:, :rank] * eps[:rank]) @ coeff[:, :rank].T
        )
        errors["actual_weighted_density_equation"] = maximum(
            fock @ density - overlap @ weighted
        )
        if reference_forces:
            expected_forces = np.load(reference_forces)
        else:
            gradient = mf.nuc_grad_method()
            gradient.auxbasis_response = True
            expected_forces = -gradient.kernel()
        if (
            expected_forces.shape != actual_forces.shape
            or not np.isfinite(expected_forces).all()
        ):
            raise ValueError("invalid independent force components")
        errors["forces_vs_pyscf"] = maximum(actual_forces - expected_forces)
        net_force = maximum(actual_forces.sum(axis=0))
    return {
        "passed": bool(
            endpoint["converged"]
            and all(np.isfinite(v) and v < 1e-8 for v in errors.values())
            and errors["energy_vs_pyscf"] < 1e-9
        ),
        "errors": errors,
        "maximum_net_force": net_force,
        "arrays_sha256": sha256(arrays_path),
        "checkpoint_sha256": sha256(checkpoint) if checkpoint else None,
        "reference_forces_sha256": sha256(reference_forces)
        if reference_forces
        else None,
        "scope": "Full independent component checks; net force is auxiliary, not a replacement for component comparison.",
    }
