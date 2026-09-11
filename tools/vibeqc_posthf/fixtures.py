"""Load committed independent post-HF fixtures without importing PySCF."""

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np
from vibeqc import Atom, Primitive, Shell
from vibeqc.profiles import canonical_hash

from .reference import ReferenceSnapshot

ROOT = Path(__file__).resolve().parents[2] / "tests/reference_data/posthf"


def load_fixture(name):
    """Check scalar/array provenance before returning independent fixture data."""
    metadata = json.loads((ROOT / (name + ".json")).read_text())
    if (
        metadata["schema"] != "vibeqc.posthf_reference"
        or metadata["version"] != 1
        or metadata["versions"]["pyscf"] != "2.14.0"
    ):
        raise ValueError("unrecognized post-HF reference provenance")
    with np.load(ROOT / (name + ".npz"), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    digest = canonical_hash(
        {
            k: sha256(np.ascontiguousarray(v, dtype="<f8").tobytes()).hexdigest()
            for k, v in arrays.items()
        }
    )
    if digest != metadata["array_hash"]:
        raise ValueError("post-HF reference array hash mismatch")
    return metadata, arrays


def source_arguments(metadata):
    """Recover the original unnormalized physical basis and geometry exactly."""
    inputs = metadata["inputs"]
    atoms = tuple(
        Atom(z, tuple(x))
        for z, x in zip(inputs["atomic_numbers"], inputs["coordinates"])
    )

    def shells(records):
        return tuple(
            Shell(
                s["atom_index"],
                s["angular_momentum"],
                # Match public basis resolution before hashing: JSON integer
                # coefficients (notably the f-shell fixture's 1) and 1.0 are
                # the same physical primitive and become FP64 in NativeSource.
                tuple(Primitive(float(p[0]), float(p[1])) for p in s["primitives"]),
            )
            for s in records
        )

    return {
        "atoms": atoms,
        "basis": shells(inputs["shells"]),
        "auxiliary_basis": shells(metadata["auxiliary_shells"]),
        "charge": inputs["charge"],
        "representation": "spherical"
        if inputs["basis_representation"] == "real_spherical"
        else "cartesian",
    }


def fixture_snapshot(
    metadata, arrays, *, label="conventional", metric=None, generation_id="fixture-1"
):
    """Import identical-C reference values with explicit Hamiltonian identity."""
    args = source_arguments(metadata)
    record = metadata["records"][label]
    geometry = canonical_hash([asdict(a) for a in args["atoms"]])
    basis = canonical_hash(
        {
            "shells": [asdict(s) for s in args["basis"]],
            "representation": metadata["inputs"]["basis_representation"],
        }
    )
    hamiltonian = (
        "conventional-unscreened" if label == "conventional" else metric.hamiltonian_id
    )
    return ReferenceSnapshot(
        *(arrays[label + "_" + k] for k in ("S", "h", "F", "C", "eps", "occ")),
        record["electron_count"],
        record["hf_energy"],
        record["scf_residual"],
        geometry,
        basis,
        generation_id,
        hamiltonian_id=hamiltonian,
        representation=metadata["inputs"]["basis_representation"],
        hf_backend="pyscf-2.14.0-test-reference",
    )
