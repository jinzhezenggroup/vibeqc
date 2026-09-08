"""Hash-checked fixed-density references; no PySCF import in ordinary tests."""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
from vibeqc.profiles import canonical_hash, file_hash

from tools.vibeqc_dft import ExplicitGrid

ROOT = Path(__file__).resolve().parents[2]
CASES = ("h2", "water", "f_cartesian", "f_spherical")


def load_integration_fixture(name, *, directory=None):
    """Verify input, exporter and every numeric block before use."""
    if name not in CASES:
        raise ValueError("unknown XC integration fixture")
    directory = (
        ROOT / "tests/reference_data/xc_integration"
        if directory is None
        else Path(directory)
    )
    meta = json.loads((directory / f"{name}.json").read_text())
    identity = meta.pop("identity")
    if (
        meta["schema"] != "vibeqc.xc-integration-reference"
        or meta["version"] != 1
        or canonical_hash(meta) != identity
        or canonical_hash(meta["inputs"]) != meta["inputs_hash"]
    ):
        raise ValueError("XC integration fixture identity mismatch")
    for field, path in (
        ("exporter_sha256", "tools/generate_xc_integration_references.py"),
        ("basis_adapter_sha256", "tools/generate_validation_references.py"),
    ):
        if meta["reference"][field] != file_hash(ROOT / path):
            raise ValueError("XC integration reference source mismatch")
    with np.load(directory / f"{name}.npz", allow_pickle=False) as archive:
        arrays = dict(archive)
    if set(arrays) != set(meta["arrays"]):
        raise ValueError("XC integration reference block mismatch")
    for key, value in arrays.items():
        record = meta["arrays"][key]
        if (
            value.dtype != np.float64
            or list(value.shape) != record["shape"]
            or sha256(value.tobytes()).hexdigest() != record["sha256"]
        ):
            raise ValueError("XC integration reference array hash mismatch")
    grid = ExplicitGrid(
        arrays["points"],
        arrays["weights"],
        (0,) * len(arrays["weights"]),
        {"reference": name, "units": "Bohr"},
    )
    return meta, arrays, grid
