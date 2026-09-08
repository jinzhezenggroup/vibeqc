"""Generate redistributable local BSE data and independent PySCF integral/HF oracles.

The BSE checkout must match references/manifest.toml. This script imports no
VibeQC basis parser, normalizer, capability table or scientific evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyscf
from pyscf import gto, scf

PIN = "4adaf1372c7101620ca1a9f3130be9ae97fb8f30"
LICENSE_HASH = "14b5a1c21a9e0966e295b9e3d66c0cee9475ffe5931e74d063c3713e2ae9a496"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pyscf_basis(data):
    """Independently map interchange contractions to PySCF's general shell lists."""
    result = {}
    for z, element in data["elements"].items():
        shells = []
        for item in element["electron_shells"]:
            angular, rows = item["angular_momentum"], item["coefficients"]
            if len(angular) == 1:
                shells.append(
                    [
                        angular[0],
                        *[
                            [float(e), *[float(row[i]) for row in rows]]
                            for i, e in enumerate(item["exponents"])
                        ],
                    ]
                )
            else:
                for l, row in zip(angular, rows, strict=True):
                    shells.append(
                        [
                            l,
                            *[
                                [float(e), float(c)]
                                for e, c in zip(item["exponents"], row, strict=True)
                            ],
                        ]
                    )
        result[gto.mole._symbol(int(z))] = shells
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bse-source", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.bse_source), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != PIN or digest(args.bse_source / "LICENSE") != LICENSE_HASH:
        raise RuntimeError("BSE source/license does not match the pinned reference")
    if subprocess.check_output(
        [
            "git",
            "-C",
            str(args.bse_source),
            "diff",
            "HEAD",
            "--",
            "basis_set_exchange/data",
        ],
        text=True,
    ):
        raise RuntimeError("BSE source data must be clean")
    if pyscf.__version__ != "2.14.0":
        raise RuntimeError("reference requires PySCF 2.14.0")
    sys.path.insert(0, str(args.bse_source))
    import basis_set_exchange as bse

    args.output.mkdir(parents=True, exist_ok=True)
    datasets = {
        "sto-3g-ho": bse.get_basis("sto-3g", elements=[1, 8]),
        "cc-pvtz-fe": bse.get_basis("cc-pvtz", elements=[26]),
        "def2-tzvp-au": bse.get_basis("def2-tzvp", elements=[79]),
    }
    synthetic = {
        "molssi_bse_schema": {"schema_type": "complete", "schema_version": "0.1"},
        "name": "vibeqc-synthetic-fe-h",
        "version": "1",
        "function_types": ["gto", "gto_spherical"],
        "elements": {
            "1": datasets["sto-3g-ho"]["elements"]["1"],
            "26": {
                "electron_shells": [
                    {
                        "function_type": "gto",
                        "angular_momentum": [0],
                        "exponents": ["500", "80", "15"],
                        "coefficients": [["0.7", "0.3", "0"], ["0", "-0.2", "1"]],
                    },
                    {
                        "function_type": "gto",
                        "angular_momentum": [1],
                        "exponents": ["8", "2"],
                        "coefficients": [["0.8", "0.2"]],
                    },
                    {
                        "function_type": "gto_spherical",
                        "angular_momentum": [2],
                        "exponents": ["4"],
                        "coefficients": [["1"]],
                    },
                ]
            },
        },
    }
    datasets["synthetic-fe-h"] = synthetic
    for name, data in datasets.items():
        (args.output / (name + ".json")).write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n"
        )
    (args.output / "BSE-LICENSE").write_bytes(
        (args.bse_source / "LICENSE").read_bytes()
    )
    arrays, cases = {}, []
    systems = (
        (
            "water",
            "sto-3g-ho",
            [("O", (0, 0, 0)), ("H", (0, -1.43, 1.1)), ("H", (0, 1.43, 1.1))],
            0,
        ),
        ("fe_ion", "synthetic-fe-h", [("Fe", (0, 0, 0))], 24),
        (
            "fe_h_ion",
            "synthetic-fe-h",
            [("Fe", (0, 0, -0.3)), ("H", (0.2, 0.1, 1.7))],
            25,
        ),
    )
    for name, source, atoms, charge in systems:
        for representation in ("cartesian", "spherical"):
            key = name + "_" + representation
            changed_atoms = [(symbol, list(xyz)) for symbol, xyz in atoms]
            changed_atoms[-1][1][:] = np.add(
                changed_atoms[-1][1], [0.07, -0.02, 0.03]
            ).tolist()
            for suffix, positions in (("", atoms), ("_changed", changed_atoms)):
                oracle_key = key + suffix
                molecule = gto.M(
                    atom=positions,
                    basis=pyscf_basis(datasets[source]),
                    charge=charge,
                    spin=0,
                    unit="Bohr",
                    cart=representation == "cartesian",
                    verbose=0,
                )
                hf = scf.RHF(molecule)
                hf.conv_tol, hf.conv_tol_grad = 1e-13, 1e-10
                hf.max_cycle = 150
                hf.kernel()
                if not hf.converged:
                    raise RuntimeError(f"reference did not converge: {key}")
                overlap = molecule.intor("int1e_ovlp")
                # VibeQC normalizes each Cartesian component, while libcint's
                # Cartesian d/f convention uses common radial shell factors.
                # Apply an independent overlap-derived diagonal basis transform.
                scale = 1 / np.sqrt(np.diag(overlap))
                arrays[oracle_key + "_overlap"] = (
                    scale[:, None] * overlap * scale[None, :]
                )
                arrays[oracle_key + "_kinetic"] = (
                    scale[:, None] * molecule.intor("int1e_kin") * scale[None, :]
                )
                arrays[oracle_key + "_hcore"] = (
                    scale[:, None] * hf.get_hcore() * scale[None, :]
                )
                arrays[oracle_key + "_energy"] = np.array(hf.e_tot)
                arrays[oracle_key + "_forces"] = -hf.nuc_grad_method().kernel()
            cases.append(
                {
                    "name": key,
                    "source": source + ".json",
                    "atoms": atoms,
                    "changed_atoms": changed_atoms,
                    "charge": charge,
                    "multiplicity": 1,
                    "representation": representation,
                    "nao": molecule.nao_nr(),
                    "electrons": molecule.nelec,
                    "ao_labels": molecule.ao_labels(),
                }
            )
    np.savez_compressed(args.output / "oracles.npz", **arrays)
    metadata = {
        "schema": 1,
        "bse_commit": PIN,
        "bse_reported_version": bse.__version__,
        "pyscf": pyscf.__version__,
        "numpy": np.__version__,
        "license": "BSD-3-Clause",
        "synthetic_fe_data": "VibeQC original diagnostic basis; GPL-3.0-or-later. Hydrogen is unchanged BSE STO-3G.",
        "files": {
            p.name: digest(p)
            for p in args.output.iterdir()
            if p.is_file() and p.name != "manifest.json"
        },
        "generator_sha256": digest(__file__),
        "cases": cases,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
