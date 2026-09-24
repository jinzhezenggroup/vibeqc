"""Regenerate the frozen DFT-MP-v1 geometries; never use this at benchmark time.

Requires the repository's Python package and RDKit 2026.03.4. The committed JSON
inputs, rather than this generator or a mutable benchmark catalogue, are the
runtime contract. Re-generation is a read-only audit and must reproduce the
LF-normalized frozen bytes. It never overwrites v1 inputs on a mismatch.
"""

from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from benchmarks._cases import benchmark_cases

ROOT = Path(__file__).resolve().parent
ANGSTROM_TO_BOHR = 1.8897261246257702
SEED = 1185


def _rdkit_case(smiles: str) -> list[list[object]]:
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = SEED
    params.numThreads = 1
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError(f"embedding failed for {smiles}")
    props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    if (
        props is None
        or AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=1000) != 0
    ):
        raise RuntimeError(f"MMFF94s optimization failed for {smiles}")
    conf = mol.GetConformer()
    return [
        [
            atom.GetSymbol(),
            [
                round(float(v) * ANGSTROM_TO_BOHR, 12)
                for v in conf.GetAtomPosition(atom.GetIdx())
            ],
        ]
        for atom in mol.GetAtoms()
    ]


def build() -> dict[str, dict]:
    if rdBase.rdkitVersion != "2026.03.4":
        raise RuntimeError(f"pin RDKit 2026.03.4, found {rdBase.rdkitVersion}")
    prior = benchmark_cases()
    borrowed = {
        "water": "water-def2-svp-spherical",
        "water_dimer": "water-tetramer-def2-svp-spherical",
        "water8": "water-octamer-s4-def2-svp-spherical",
        "water16": "water-hexadecamer-2s4-def2-svp-spherical",
        "water32": "water-32mer-4s4-def2-svp-spherical",
        "oh": "oh-def2-svp-spherical-uhf",
    }
    out = {}
    for key, source in borrowed.items():
        atoms = prior[source].atoms
        if key == "water_dimer":
            atoms = prior[source].atoms[:6]
        out[key] = {
            "atoms": [[element, [float(x) for x in xyz]] for element, xyz in atoms],
            "charge": 0,
            "multiplicity": 2 if key == "oh" else 1,
            "source": {
                "kind": "repository_snapshot",
                "name": source,
                "repository_commit": "6f6a3789c6065adff8e6e603f9b2d3c840852f53",
                "file": "benchmarks/_cases.py",
                "license": "CC-BY-4.0"
                if key in {"water_dimer", "water8", "water16", "water32"}
                else "GPL-3.0-or-later",
                "attribution": "GMTKN55/WATER27, Goerigk et al., PCCP 2017, DOI:10.1039/C7CP04913G"
                if key in {"water_dimer", "water8", "water16", "water32"}
                else "VibeQC repository",
                "source_url": "https://github.com/grimme-lab/GMTKN55"
                if key in {"water_dimer", "water8", "water16", "water32"}
                else "https://github.com/jinzhezenggroup/vibeqc",
                "upstream_geometry_revision": "8d485b37"
                if key in {"water_dimer", "water8", "water16", "water32"}
                else None,
            },
        }
    out["o2"] = {
        "atoms": [
            ["O", [0.0, 0.0, 0.0]],
            ["O", [0.0, 0.0, round(1.208 * ANGSTROM_TO_BOHR, 12)]],
        ],
        "charge": 0,
        "multiplicity": 3,
        "source": {
            "kind": "constructed",
            "bond_length_angstrom": 1.208,
            "license": "GPL-3.0-or-later",
            "attribution": "VibeQC DFT-MP-v1 contract",
        },
    }
    for key, smiles in {
        "benzene": "c1ccccc1",
        "caffeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
        "ace_glygly_nme": "CC(=O)NCC(=O)NCC(=O)NC",
    }.items():
        out[key] = {
            "atoms": _rdkit_case(smiles),
            "charge": 0,
            "multiplicity": 1,
            "source": {
                "kind": "rdkit_etkdg3_mmff94s",
                "smiles": Chem.MolToSmiles(Chem.MolFromSmiles(smiles)),
                "rdkit_version": rdBase.rdkitVersion,
                "seed": SEED,
                "max_iterations": 1000,
                "license": "GPL-3.0-or-later",
                "attribution": "VibeQC DFT-MP-v1 contract; geometry generated with RDKit (BSD-3-Clause)",
            },
        }
    return out


def _render_inputs(generated_cases: dict[str, dict]) -> dict[str, bytes]:
    """Serialize candidate input bytes without touching the frozen files."""
    candidates = {}
    for key, generated in sorted(generated_cases.items()):
        value = {"schema_version": 1, "id": key, "units": "bohr", **generated}
        candidates[f"{key}.json"] = (
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        changed = json.loads(json.dumps(value))
        changed["id"] = f"{key}-changed"
        changed["atoms"][0][1][0] = changed["atoms"][0][1][0] + 0.01
        changed["source"] = {
            "kind": "fixed_displacement",
            "parent": key,
            "atom_index": 0,
            "axis": "x",
            "delta_bohr": 0.01,
            "license": value["source"]["license"],
            "attribution": value["source"]["attribution"],
        }
        candidates[f"{key}-changed.json"] = (
            json.dumps(changed, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
    return candidates


def main() -> None:
    """Audit all frozen inputs before any deliberate versioned regeneration."""
    target = ROOT / "inputs"
    candidates = _render_inputs(build())
    mismatches = []
    for name, candidate in candidates.items():
        try:
            retained = (target / name).read_bytes().replace(b"\r\n", b"\n")
        except OSError as error:
            raise RuntimeError(f"cannot audit frozen input {name}: {error}") from error
        if retained != candidate:
            mismatches.append(name)
    if mismatches:
        raise RuntimeError(
            "regeneration differs from frozen DFT-MP-v1 inputs; no files changed: "
            + ", ".join(mismatches)
        )
    print(f"Verified {len(candidates)} frozen DFT-MP-v1 input files; no files changed.")


if __name__ == "__main__":
    main()
