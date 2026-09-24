"""Audit frozen DFT-MP-v1 geometries; never use this at benchmark time.

The committed JSON inputs are the runtime contract. The RDKit-derived inputs are
reconstructable only by the recorded platform/wheel. Default execution is a
read-only audit. A new version candidate requires an explicit separate output.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from benchmarks._cases import benchmark_cases

ROOT = Path(__file__).resolve().parent
PROVENANCE = ROOT / "generator_provenance.json"
ANGSTROM_TO_BOHR = 1.8897261246257702
SEED = 1185


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def current_generator_provenance() -> dict[str, object]:
    distribution = importlib.metadata.distribution("rdkit")
    record = distribution.read_text("RECORD")
    wheel = distribution.read_text("WHEEL")
    if record is None or wheel is None:
        raise RuntimeError("RDKit distribution metadata is incomplete")
    tags = sorted(
        line.removeprefix("Tag: ")
        for line in wheel.splitlines()
        if line.startswith("Tag: ")
    )
    return {
        "schema_version": 1,
        "implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "python_cache_tag": sys.implementation.cache_tag,
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "rdkit_runtime_version": rdBase.rdkitVersion,
        "rdkit_distribution_version": distribution.version,
        "rdkit_record_sha256": _sha256(record.encode()),
        "rdkit_wheel_sha256": _sha256(wheel.encode()),
        "wheel_tags": tags,
    }


def recorded_generator_provenance() -> dict[str, object]:
    return json.loads(PROVENANCE.read_text(encoding="utf-8"))


def generator_is_qualified() -> bool:
    return current_generator_provenance() == recorded_generator_provenance()


def _require_qualified_generator() -> None:
    current = current_generator_provenance()
    recorded = recorded_generator_provenance()
    if current != recorded:
        fields = sorted(
            key
            for key in set(current) | set(recorded)
            if current.get(key) != recorded.get(key)
        )
        raise RuntimeError(
            "unqualified DFT-MP-v1 generator environment; mismatched fields: "
            + ", ".join(fields)
        )


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
    _require_qualified_generator()
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


def serialized_inputs() -> dict[str, bytes]:
    result = {}
    for key, generated in sorted(build().items()):
        value = {"schema_version": 1, "id": key, "units": "bohr", **generated}
        result[f"{key}.json"] = (
            json.dumps(value, indent=2, sort_keys=True) + "\n"
        ).encode()
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
        result[f"{key}-changed.json"] = (
            json.dumps(changed, indent=2, sort_keys=True) + "\n"
        ).encode()
    return result


def audit_serialized(candidates: dict[str, bytes], frozen: Path) -> None:
    actual_names = {path.name for path in frozen.glob("*.json")}
    mismatches = sorted(actual_names ^ set(candidates))
    for name, candidate in sorted(candidates.items()):
        path = frozen / name
        if path.is_file() and path.read_bytes().replace(b"\r\n", b"\n") != candidate:
            mismatches.append(name)
    if mismatches:
        raise RuntimeError(
            "frozen input audit mismatch: " + ", ".join(sorted(set(mismatches)))
        )


def write_candidate_files(
    candidates: dict[str, bytes], output: Path, *, frozen: Path
) -> None:
    if any(Path(name).name != name for name in candidates):
        raise ValueError("candidate filenames must be flat")
    target = output.resolve()
    frozen = frozen.resolve()
    if target == frozen or target in frozen.parents or frozen in target.parents:
        raise ValueError("candidate generation requires a separate output tree")
    if target.exists() and any(target.iterdir()):
        raise ValueError("candidate output directory must be empty")
    target.mkdir(parents=True, exist_ok=True)
    for name, value in sorted(candidates.items()):
        (target / name).write_bytes(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-output",
        type=Path,
        help="write a deliberate new-version candidate to a separate empty tree",
    )
    arguments = parser.parse_args()
    candidates = serialized_inputs()
    frozen = ROOT / "inputs"
    if arguments.candidate_output is None:
        audit_serialized(candidates, frozen)
        print(f"verified {len(candidates)} frozen DFT-MP-v1 inputs")
    else:
        write_candidate_files(candidates, arguments.candidate_output, frozen=frozen)
        print(
            f"wrote {len(candidates)} candidate inputs to {arguments.candidate_output}"
        )


if __name__ == "__main__":
    main()
