"""Small explicit CC endpoint inputs; independent reference data, not a solver."""

import json
from pathlib import Path

import numpy as np
from vibeqc import Atom

from tools.vibeqc_posthf import ReferenceSnapshot
from tools.vibeqc_validation.fixtures import calculator_inputs, molecular_inputs
from tools.vibeqc_validation.schema import canonical_hash

ROOT = Path(__file__).resolve().parents[1] / "tests/reference_data/cc/endpoints"


def cases():
    rows = [
        x for x in molecular_inputs() if x["name"] in ("h2", "he", "h2o", "nh3", "ch4")
    ]
    for row in rows:
        row["cc_settings"] = {
            "method": "CCSD",
            "frozen_core": 0,
            "conv_tol": 1e-13,
            "conv_tol_normt": 1e-11,
            "max_cycle": 150,
        }
        if row["name"] == "he":
            # STO-3G alone has no virtual. Add one explicit normalized s shell
            # so the two-electron CCSD=FCI endpoint is nontrivial.
            row["shells"].append(
                {"atom_index": 0, "angular_momentum": 0, "primitives": [[0.25, 1.0]]}
            )
            row["basis_name"] = "bundled STO-3G + explicit s(0.25)"
    return rows


def source_arguments(inputs):
    return {
        "atoms": tuple(
            Atom(z, tuple(x))
            for z, x in zip(inputs["atomic_numbers"], inputs["coordinates"])
        ),
        "basis": calculator_inputs(inputs)["basis"],
        "charge": inputs["charge"],
        "representation": "cartesian",
    }


def array_hash(arrays):
    from hashlib import sha256

    return canonical_hash(
        {
            k: sha256(np.ascontiguousarray(v, dtype="<f8").tobytes()).hexdigest()
            for k, v in arrays.items()
        }
    )


def load(name, root=ROOT):
    meta = json.loads((Path(root) / (name + ".json")).read_text())
    if meta["pyscf"] != "2.14.0" or meta["version"] != 1:
        raise ValueError("CC endpoint reference version mismatch")
    with np.load(Path(root) / (name + ".npz"), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    if (
        array_hash(arrays) != meta["arrays_hash"]
        or canonical_hash(meta["inputs"]) != meta["inputs_hash"]
    ):
        raise ValueError("CC endpoint reference hash mismatch")
    return meta, arrays


def snapshot_from_fixture(source, meta, arrays):
    return ReferenceSnapshot(
        *(arrays[k] for k in ("S", "h", "F", "C", "eps", "occ")),
        source.electron_count,
        meta["hf_energy"],
        meta["scf_residual"],
        source.geometry_hash,
        source.basis_hash,
        "pyscf-ccsd-" + meta["inputs"]["name"],
    )
