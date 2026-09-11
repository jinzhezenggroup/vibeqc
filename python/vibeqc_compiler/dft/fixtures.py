"""Hash-checked independent grids/AO/density fixtures; never import PySCF."""

from __future__ import annotations

import json
from hashlib import sha256

import numpy as np

from vibeqc_compiler.common.paths import source_root
from vibeqc_compiler.common.provenance import canonical_hash

NAMES = ("h2", "water", "f_cartesian", "f_spherical", "diffuse", "tight")


def load_fixture(name):
    """Verify all arrays and mathematical inputs before returning reference data."""
    root = source_root() / "tests/reference_data/grid"
    if name not in NAMES:
        raise ValueError("unknown grid fixture")
    meta = json.loads((root / f"{name}.json").read_text())
    if (
        meta["schema"] != "vibeqc.grid-reference"
        or meta["version"] != 1
        or canonical_hash(meta["inputs"]) != meta["inputs_hash"]
    ):
        raise ValueError("grid fixture identity mismatch")
    with np.load(root / f"{name}.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    for array_name, record in meta["arrays"].items():
        a = arrays[array_name]
        if (
            a.dtype != np.float64
            or list(a.shape) != record["shape"]
            or sha256(a.tobytes()).hexdigest() != record["sha256"]
        ):
            raise ValueError("grid fixture array hash mismatch")
    return meta, arrays


def basis_arguments(meta):
    """Build exactly the original shell coefficients, geometry and representation."""
    from vibeqc import Primitive, Shell

    i = meta["inputs"]
    return {
        "atoms": list(zip(i["atomic_numbers"], i["coordinates"], strict=True)),
        "basis": tuple(
            Shell(
                s["atom_index"],
                s["angular_momentum"],
                tuple(Primitive(*p) for p in s["primitives"]),
            )
            for s in i["shells"]
        ),
        "representation": i["basis_representation"],
        "charge": i["charge"],
        "multiplicity": i["multiplicity"],
    }


def __getattr__(name):
    """Retain the checkout-only ROOT compatibility attribute without eager IO."""
    if name == "ROOT":
        return source_root() / "tests/reference_data/grid"
    raise AttributeError(name)
