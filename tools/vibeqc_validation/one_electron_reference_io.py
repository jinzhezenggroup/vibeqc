"""Serialization helpers for committed one-electron derivative references."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib import metadata
from pathlib import Path

import numpy as np

from .schema import file_hash

_SCHEMA_VERSION = 1
_ARRAY_FIELDS = (
    "records",
    "weights",
    "reference",
    "spherical_reference",
    "projection_a",
    "projection_b",
)


def _array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def write_derivative_references(fixtures, directory: Path) -> tuple[Path, Path]:
    """Write deterministic metadata plus an NPZ payload from live PySCF fixtures.

    The NPZ container metadata itself is not used as scientific identity because
    ZIP timestamps may vary.  Each numerical array is instead hashed from dtype,
    shape and C-order bytes and those hashes are committed in the manifest.
    """
    directory.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    rows = []
    for index, fixture in enumerate(fixtures):
        key = f"f{index:03d}"
        values = {
            "records": np.asarray(fixture.records),
            "weights": np.asarray(fixture.weights),
            "reference": np.asarray(fixture.reference),
            "spherical_reference": np.asarray(fixture.spherical_reference),
            "projection_a": np.asarray(fixture.projections[0]),
            "projection_b": np.asarray(fixture.projections[1]),
        }
        row_hashes = {}
        for field in _ARRAY_FIELDS:
            name = f"{key}_{field}"
            arrays[name] = values[field]
            row_hashes[field] = _array_hash(values[field])
        rows.append(
            {
                "key": key,
                "inputs": fixture.inputs,
                "arrays": row_hashes,
            }
        )

    payload = directory / "one_electron_derivatives.npz"
    np.savez_compressed(payload, **arrays)

    # Capture enough provenance to regenerate and audit the independent oracle.
    import pyscf

    root = Path(__file__).resolve().parents[2]
    manifest = {
        "schema": "vibeqc.one-electron-derivative-reference",
        "schema_version": _SCHEMA_VERSION,
        "fixture_count": len(rows),
        "fixtures": rows,
        "provenance": {
            "purpose": "independent PySCF/libcint reference generation only",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyscf": pyscf.__version__,
            "scipy": metadata.version("scipy"),
            "generator_sha256": file_hash(__file__),
            "derivative_fixture_source_sha256": file_hash(
                root / "tools/vibeqc_validation/one_electron_derivatives.py"
            ),
            "value_fixture_source_sha256": file_hash(
                root / "tools/vibeqc_validation/one_electron_values.py"
            ),
        },
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return manifest_path, payload
