"""Load and integrity-check committed one-electron derivative references."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .one_electron_derivatives import OneElectronDerivativeFixture
from .one_electron_reference_io import _ARRAY_FIELDS, _SCHEMA_VERSION, _array_hash

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REFERENCE_DIRECTORY = _ROOT / "tests/reference_data/one_electron_derivatives"


def committed_one_electron_derivative_matrix(
    directory: Path = _DEFAULT_REFERENCE_DIRECTORY,
):
    """Return audited fixtures without importing PySCF/libcint at test time."""
    manifest_path = directory / "manifest.json"
    payload_path = directory / "one_electron_derivatives.npz"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "vibeqc.one-electron-derivative-reference":
        raise ValueError("unexpected one-electron reference schema")
    if manifest.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unsupported one-electron reference schema version")
    rows = manifest.get("fixtures", [])
    if manifest.get("fixture_count") != len(rows):
        raise ValueError("one-electron reference fixture count mismatch")

    fixtures = []
    with np.load(payload_path, allow_pickle=False) as payload:
        expected_names = {
            f"{row['key']}_{field}" for row in rows for field in _ARRAY_FIELDS
        }
        if set(payload.files) != expected_names:
            raise ValueError("one-electron reference payload inventory mismatch")
        for row in rows:
            values = {}
            for field in _ARRAY_FIELDS:
                name = f"{row['key']}_{field}"
                value = np.asarray(payload[name]).copy()
                if _array_hash(value) != row["arrays"][field]:
                    raise ValueError(
                        f"one-electron reference hash mismatch: {row['key']} {field}"
                    )
                values[field] = value
            fixtures.append(
                OneElectronDerivativeFixture(
                    row["inputs"],
                    values["records"],
                    values["weights"],
                    values["reference"],
                    values["spherical_reference"],
                    (values["projection_a"], values["projection_b"]),
                )
            )
    return fixtures
