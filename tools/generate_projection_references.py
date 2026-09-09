"""Regenerate independent rectangular-overlap and target-HF fixtures with PySCF."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

import numpy as np

from tools.generate_validation_references import pyscf_molecule


def overlap_inputs(same_geometry):
    """Non-native fixture input values preserve unsorted shell/atom ownership."""
    target_atoms = [["H", [0.1, -0.2, 0.3]], ["H", [0.9, 0.5, -0.4]]]
    source_atoms = (
        target_atoms
        if same_geometry
        else [["H", [0.12, -0.18, 0.31]], ["H", [1.1, 0.3, -0.7]]]
    )
    target_shells = [
        {
            "atom_index": i % 2,
            "angular_momentum": angular_momentum,
            "primitives": [
                [0.17 + 0.2 * angular_momentum, 0.8],
                [1.4 + angular_momentum, -0.13],
            ],
        }
        for i, angular_momentum in enumerate(range(4))
    ]
    source_shells = [
        {
            "atom_index": i % 2,
            "angular_momentum": angular_momentum,
            "primitives": [[0.23 + 0.1 * angular_momentum, 0.75]],
        }
        for i, angular_momentum in enumerate((3, 0, 2, 1, 0))
    ]
    return {
        "target_atoms": target_atoms,
        "source_atoms": source_atoms,
        "target_shells": target_shells,
        "source_shells": source_shells,
    }


def overlap_reference(inputs, target_rep, source_rep):
    """Libcint rectangular integrals with independent public-AO normalization."""
    from pyscf import gto

    def molecule(geometry, shells):
        shells = sorted(shells, key=lambda s: (s["atom_index"], s["angular_momentum"]))
        mol, scales, _ = pyscf_molecule(
            {
                "atomic_numbers": [1, 1],
                "coordinates": [r for _, r in geometry],
                "shells": shells,
                "charge": 0,
                "multiplicity": 1,
                "basis_representation": "cartesian",
            }
        )
        return mol, scales, shells

    tm, ts, to = molecule(inputs["target_atoms"], inputs["target_shells"])
    sm, ss, so = molecule(inputs["source_atoms"], inputs["source_shells"])
    expected = gto.intor_cross("int1e_ovlp_cart", tm, sm) * ts[:, None] * ss[None, :]
    if target_rep == "spherical":
        expected = (tm.cart2sph_coeff() / ts[:, None]).T @ expected
    if source_rep == "spherical":
        expected = expected @ (sm.cart2sph_coeff() / ss[:, None])

    def permutation(original, ordered, representation):
        counts = [
            2 * s["angular_momentum"] + 1
            if representation == "spherical"
            else (s["angular_momentum"] + 1) * (s["angular_momentum"] + 2) // 2
            for s in ordered
        ]
        offsets = np.cumsum([0, *counts])
        return [
            i
            for shell in original
            for i in range(
                offsets[ordered.index(shell)], offsets[ordered.index(shell) + 1]
            )
        ]

    return expected[
        np.ix_(
            permutation(inputs["target_shells"], to, target_rep),
            permutation(inputs["source_shells"], so, source_rep),
        )
    ]


def main():
    import pyscf
    from pyscf import gto, scf

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tests/data/basis_projection_reference.json",
    )
    args = parser.parse_args()
    if pyscf.__version__ != "2.14.0":
        raise RuntimeError("reference regeneration requires pinned PySCF 2.14.0")
    overlap = []
    for same in (False, True):
        inputs = overlap_inputs(same)
        for target_rep in ("cartesian", "spherical"):
            for source_rep in ("cartesian", "spherical"):
                overlap.append(
                    {
                        "inputs": inputs,
                        "same_geometry": same,
                        "target_rep": target_rep,
                        "source_rep": source_rep,
                        "values": overlap_reference(
                            inputs, target_rep, source_rep
                        ).tolist(),
                    }
                )
    atoms = [["H", [0, 0, -0.7]], ["H", [0.1, 0, 0.7]]]
    hf = []
    for method, charge, multiplicity in (("rhf", 0, 1), ("uhf", 1, 2)):
        for fitted in (False, True):
            mol = gto.M(
                atom=atoms,
                unit="Bohr",
                basis="def2-svp",
                charge=charge,
                spin=multiplicity - 1,
                verbose=0,
                cart=True,
            )
            mf = (scf.RHF if method == "rhf" else scf.UHF)(mol)
            if fitted:
                mf = mf.density_fit(auxbasis="def2-svp")
            mf.conv_tol, mf.conv_tol_grad = 1e-13, 1e-10
            mf.kernel()
            if not mf.converged:
                raise RuntimeError("independent target HF did not converge")
            hf.append(
                {
                    "method": method,
                    "charge": charge,
                    "multiplicity": multiplicity,
                    "fitted": fitted,
                    "atoms": atoms,
                    "basis": "def2-svp",
                    "auxiliary_basis": "def2-svp" if fitted else None,
                    "energy": mf.e_tot,
                    "forces": (-mf.nuc_grad_method().kernel()).tolist(),
                    "density": np.asarray(mf.make_rdm1())
                    .reshape(-1, mol.nao_nr(), mol.nao_nr())
                    .tolist(),
                }
            )
    report = {
        "schema": "vibeqc.basis_projection_references",
        "version": 1,
        "provider": "PySCF/libcint",
        "pyscf_version": pyscf.__version__,
        "units": {"coordinates": "bohr", "energy": "hartree", "forces": "hartree/bohr"},
        "overlap": overlap,
        "hf": hf,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
