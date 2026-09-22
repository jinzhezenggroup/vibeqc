"""Independent matched-basis/grid PySCF check of native WB97M-V SCF dumps.

Run the native test with a JSONL destination, then pass that file here.
PySCF is an optional qualification oracle, never a production dependency.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def verify(path: Path) -> list[dict]:
    from pyscf import dft, gto, lib
    from pyscf.dft import libxc

    lib.num_threads(1)
    assert libxc.libxc_version() == "7.0.0", (
        "independent fixture requires pinned Libxc 7.0.0"
    )
    rows = []
    for line in path.read_text().splitlines():
        case = json.loads(line)
        names = [f"{gto.mole._symbol(z)}{i}" for i, (z, _) in enumerate(case["atoms"])]
        basis = {name: [] for name in names}
        for atom, angular, primitives in case["shells"]:
            basis[names[atom]].append([angular, *primitives])
        mol = gto.M(
            atom=[(names[i], xyz) for i, (_, xyz) in enumerate(case["atoms"])],
            basis=basis,
            unit="Bohr",
            cart=True,
            charge=case["charge"],
            spin=case["spin"],
            verbose=0,
        )
        mf = dft.UKS(mol) if case["unrestricted"] else dft.RKS(mol)
        mf.xc = "WB97M_V"
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-10
        mf.max_cycle = 200
        mf.small_rho_cutoff = 0.0
        for grids in (mf.grids, mf.nlcgrids):
            grids.coords = np.asarray(case["points"]).reshape(-1, 3)
            grids.weights = np.asarray(case["weights"])
            grids.non0tab = None
        spins = 2 if case["unrestricted"] else 1
        density = np.asarray(case["density"]).reshape(spins, mol.nao, mol.nao)
        native_fock = np.asarray(case["fock"]).reshape(spins, mol.nao, mol.nao)
        dm = density if spins == 2 else density[0]
        veff = mf.get_veff(mol, dm)
        fock = mf.get_hcore() + veff
        native_energy_error = float(
            abs(mf.energy_tot(dm=dm, vhf=veff) - case["energy"])
        )
        fock_error = float(np.max(np.abs(fock - native_fock)))
        # Re-converge independently from a core guess, not from VibeQC's answer.
        energy = float(mf.kernel(dm0=mf.get_init_guess(key="1e")))
        assert mf.converged, case["name"]
        density_error = float(np.max(np.abs(mf.make_rdm1() - dm)))
        assert native_energy_error < 2e-8, (
            case["name"],
            "energy at native density",
            native_energy_error,
        )
        assert fock_error < 2e-7, (case["name"], "complete physical Fock", fock_error)
        assert abs(energy - case["energy"]) < 2e-8, (
            case["name"],
            "independent SCF",
            energy,
            case["energy"],
        )
        assert density_error < 2e-6, (
            case["name"],
            "independent density",
            density_error,
        )
        result = {
            "name": case["name"],
            "energy": energy,
            "energy_error": native_energy_error,
            "fock_error": fock_error,
            "density_error": density_error,
        }
        rows.append(result)
        print(json.dumps(result), flush=True)
    assert len(rows) == 5
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path)
    verify(parser.parse_args().dump)
