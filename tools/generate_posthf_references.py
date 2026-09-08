"""Export independent small AO/MO/MP2 fixtures with pinned PySCF 2.14.0.

PySCF (Apache-2.0) and libcint (BSD-2-Clause) are test-only reference tools.
This original exporter uses their public integral/SCF/MP2 interfaces and an
independent spectral DF reconstruction. No external solver code is copied.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
import numpy as np
from vibeqc import Atom, Primitive, Shell
from vibeqc.calculator import _named_basis_shells
from vibeqc.profiles import canonical_hash

from tools.generate_validation_references import pyscf_molecule


def generate(directory):
    """Save same-C conventional and DF Hamiltonians, not only total energies."""
    import pyscf
    from pyscf import ao2mo, df, gto, mp, scf

    if pyscf.__version__ != "2.14.0":
        raise RuntimeError("reference export requires pinned PySCF 2.14.0")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fixtures = [
        ("h2", [("H", (0, 0, 0)), ("H", (0.1, 0.2, 1.4))], "sto-3g", "cartesian", 0),
        (
            "water",
            [("O", (0, 0, 0)), ("H", (1.4, 0.1, 1.1)), ("H", (-1.2, 0.2, 1.3))],
            "sto-3g",
            "cartesian",
            0,
        ),
        (
            "lih",
            [("Li", (0, 0, 0)), ("H", (0.2, -0.1, 3.0))],
            "sto-3g",
            "real_spherical",
            0,
        ),
        (
            "f_heh",
            [("He", (0, 0, 0)), ("H", (0.3, -0.2, 1.7))],
            (
                Shell(0, 0, (Primitive(1.3, 1),)),
                Shell(0, 3, (Primitive(0.7, 1),)),
                Shell(1, 0, (Primitive(0.6, 1),)),
            ),
            "real_spherical",
            1,
        ),
    ]
    manifests = []
    for name, atoms, basis, representation, charge in fixtures:
        atoms = tuple(Atom.from_value(a) for a in atoms)
        shells = _named_basis_shells(basis, atoms) if isinstance(basis, str) else basis
        inputs = {
            "name": name,
            "atomic_numbers": [a.atomic_number for a in atoms],
            "coordinates": [a.position for a in atoms],
            "shells": [
                {
                    "atom_index": s.atom_index,
                    "angular_momentum": s.angular_momentum,
                    "primitives": [(p.exponent, p.coefficient) for p in s.primitives],
                }
                for s in shells
            ],
            "basis_representation": representation,
            "charge": charge,
            "multiplicity": 1,
        }
        mol, scale, actual = pyscf_molecule(inputs)
        if mol.nao_nr() > 12:
            raise ValueError("dense fixture limit exceeded")
        normal = np.einsum("i,j,k,l->ijkl", scale, scale, scale, scale)
        ao = mol.intor("int2e").reshape((mol.nao_nr(),) * 4)
        # Deliberately use a compact auxiliary set and record fitting error
        # separately. The native regression adds duplicate auxiliary shells
        # in a separate rank/invalidation test.
        auxiliary = gto.M(
            atom=mol.atom,
            basis=mol._basis,
            unit="Bohr",
            cart=mol.cart,
            charge=charge,
            verbose=0,
        )
        aux_scale = 1 / np.sqrt(np.diag(auxiliary.intor("int1e_ovlp")))
        raw = df.incore.aux_e2(mol, auxiliary, aosym="s1").reshape(
            mol.nao_nr(), mol.nao_nr(), auxiliary.nao_nr()
        )
        metric = auxiliary.intor("int2c2e")
        e, u = np.linalg.eigh(metric)
        cutoff = 1e-10 * e[-1]
        keep = e > cutoff
        inverse = (u[:, keep] / e[keep]) @ u[:, keep].T
        df_ao = np.einsum("uvP,PQ,wxQ->uvwx", raw, inverse, raw, optimize=True)
        arrays = {
            "ao": ao * normal,
            "df_ao": df_ao * normal,
            "raw_three_center": raw
            * scale[:, None, None]
            * scale[None, :, None]
            * aux_scale[None, None, :],
            "metric": metric * aux_scale[:, None] * aux_scale[None, :],
        }
        records = {}
        for label, hamiltonian in [("conventional", ao), ("df", df_ao)]:
            mf = scf.RHF(mol)
            mf.conv_tol = 1e-13
            mf.conv_tol_grad = 1e-11
            mf.max_cycle = 200
            mf.direct_scf_tol = 0
            mf._eri = ao2mo.restore(8, hamiltonian, mol.nao_nr())
            mf.kernel()
            if not mf.converged:
                raise RuntimeError(f"failed {name}/{label} SCF")
            c = mf.mo_coeff.copy()
            for column in range(c.shape[1]):
                if c[np.argmax(abs(c[:, column] / scale)), column] < 0:
                    c[:, column] *= -1
            mf.mo_coeff = c
            coupled = mp.MP2(mf, frozen=None)
            energy, t2 = coupled.kernel()
            mo = ao2mo.incore.general(hamiltonian, (c, c, c, c), compact=False).reshape(
                hamiltonian.shape
            )
            d = mf.make_rdm1()
            f = mf.get_fock(dm=d)
            s = mf.get_ovlp()
            for key, value in {
                "C": c / scale[:, None],
                "eps": mf.mo_energy,
                "occ": mf.mo_occ,
                "S": s * scale[:, None] * scale[None, :],
                "h": mf.get_hcore() * scale[:, None] * scale[None, :],
                "F": f * scale[:, None] * scale[None, :],
                "mo": mo,
                "t2": t2,
            }.items():
                arrays[label + "_" + key] = value
            records[label] = {
                "hf_energy": float(mf.e_tot),
                "correlation_energy": float(energy),
                "scf_residual": float(np.max(abs(f @ d @ s - s @ d @ f))),
                "electron_count": mol.nelectron,
            }
        arrays_hash = canonical_hash(
            {
                k: sha256(np.ascontiguousarray(v, dtype="<f8").tobytes()).hexdigest()
                for k, v in arrays.items()
            }
        )
        np.savez_compressed(directory / (name + ".npz"), **arrays)
        configuration = io.StringIO()
        with contextlib.redirect_stdout(configuration):
            np.show_config()
        metadata = {
            "schema": "vibeqc.posthf_reference",
            "version": 1,
            "inputs": inputs,
            "auxiliary_shells": inputs["shells"],
            "actual_angular_momenta": actual,
            "records": records,
            "array_hash": arrays_hash,
            "versions": {
                "pyscf": pyscf.__version__,
                "numpy": np.__version__,
                "numpy_blas_configuration": configuration.getvalue(),
                "python": sys.version,
            },
            "provenance": {
                "pyscf": "Apache-2.0",
                "libcint": "BSD-2-Clause",
                "exporter": "original repository GPL-3.0-or-later",
            },
            "settings": {
                "scf_energy_tolerance": 1e-13,
                "scf_gradient_tolerance": 1e-11,
                "screening": 0,
                "df_relative_threshold": 1e-10,
                "frozen_core": None,
                "mo_phase": "largest unit-normalized AO coefficient positive",
                "eri": "chemists (pq|rs)",
                "t2": "restricted ijab = (ia|jb)/D",
                "units": "Bohr, Hartree",
            },
            "conventional_df_ao_max_difference": float(
                np.max(abs((ao - df_ao) * normal))
            ),
        }
        (directory / (name + ".json")).write_text(json.dumps(metadata, indent=2) + "\n")
        manifests.append({"name": name, "array_hash": arrays_hash, "nmo": mol.nao_nr()})
        print(json.dumps(manifests[-1]), flush=True)
    return manifests


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.output)
