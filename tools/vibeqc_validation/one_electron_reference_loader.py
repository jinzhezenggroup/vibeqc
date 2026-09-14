"""Load and integrity-check committed one-electron derivative references."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .one_electron_reference_io import _ARRAY_FIELDS, _SCHEMA_VERSION, _array_hash
from .schema import canonical_hash, file_hash

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REFERENCE_DIRECTORY = _ROOT / "tests/reference_data/one_electron_derivatives"


@dataclass
class CommittedOneElectronDerivativeFixture:
    """Serialized one-electron fixture without importing live reference builders."""

    inputs: dict
    records: np.ndarray
    weights: np.ndarray
    reference: np.ndarray
    spherical_reference: np.ndarray
    projections: tuple[np.ndarray, np.ndarray]

    @property
    def input_hash(self):
        return canonical_hash(self.inputs)

    def contract(self, values):
        blocks = values.reshape(*self.weights.shape, 3, 3, 3)
        result = (blocks * self.weights[:, :, None, None, None]).sum(axis=1)
        return result.transpose(1, 2, 3, 0).reshape(self.reference.shape)

    def spherical(self, values):
        a, b = self.projections
        return np.einsum("ia,ocxij,jb->ocxab", a, values, b)


def committed_one_electron_derivative_matrix(
    directory: Path = _DEFAULT_REFERENCE_DIRECTORY,
):
    """Return audited fixtures without importing or evaluating PySCF/libcint."""
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

    provenance = manifest.get("provenance", {})
    expected_sources = {
        "generator_sha256": _ROOT
        / "tools/vibeqc_validation/one_electron_reference_io.py",
        "derivative_fixture_source_sha256": _ROOT
        / "tools/vibeqc_validation/one_electron_derivatives.py",
        "value_fixture_source_sha256": _ROOT
        / "tools/vibeqc_validation/one_electron_values.py",
    }
    for key, source in expected_sources.items():
        if provenance.get(key) != file_hash(source):
            raise ValueError(
                f"one-electron reference provenance is stale for {source.relative_to(_ROOT)}; "
                "regenerate the committed fixture"
            )

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
                CommittedOneElectronDerivativeFixture(
                    row["inputs"],
                    values["records"],
                    values["weights"],
                    values["reference"],
                    values["spherical_reference"],
                    (values["projection_a"], values["projection_b"]),
                )
            )
    return fixtures
