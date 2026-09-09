"""Generate the extra NUM01 holdout references using optional pinned PySCF."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import platform
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

from tools.generate_validation_references import molecular_data, pyscf_molecule
from tools.vibeqc_numerics.fixtures import extra_inputs
from tools.vibeqc_validation.fixtures import mathematical_hash, validate_fixture
from tools.vibeqc_validation.schema import canonical_hash, file_hash


def generate(destination):
    """Save source/data/library provenance; this function never installs PySCF."""
    import pyscf
    from threadpoolctl import threadpool_info, threadpool_limits

    if pyscf.__version__ != "2.14.0":
        raise ValueError("reference generator requires pinned PySCF 2.14.0")
    root = Path(__file__).resolve().parents[1]
    libcint = Path(pyscf.__file__).parent / "lib/deps/lib/libcint.so"
    pyscf.lib.num_threads(1)
    with threadpool_limits(limits=1):
        config = io.StringIO()
        with contextlib.redirect_stdout(config):
            np.show_config()
        provenance = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "pyscf": pyscf.__version__,
            "numpy": np.__version__,
            "scipy": metadata.version("scipy"),
            "h5py": metadata.version("h5py"),
            "threads": 1,
            "blas": config.getvalue(),
            "loaded_threadpools": threadpool_info(),
            "generator_sha256": file_hash(__file__),
            "shared_generator_sha256": file_hash(
                root / "tools/generate_validation_references.py"
            ),
            "fixture_definition_sha256": file_hash(
                root / "tools/vibeqc_numerics/fixtures.py"
            ),
            "libcint_sha256": file_hash(libcint),
            "basis_pack_sha256": file_hash(root / "python/vibeqc/data/basis_pack.json"),
            "purpose": "independent held-out HF accuracy references",
        }
        destination.mkdir(parents=True, exist_ok=True)
        manifest = {"schema_version": 1, "fixtures": []}
        for inputs in extra_inputs():
            molecule, scale, angular = pyscf_molecule(inputs)
            data = molecular_data(inputs, molecule, scale)
            record = {
                "schema": "vibeqc.reference",
                "schema_version": 1,
                "inputs": inputs,
                "inputs_hash": mathematical_hash(inputs),
                "data": data,
                "data_hash": canonical_hash(data),
                "angular_momenta_loaded": angular,
                "provenance": provenance,
                "provenance_hash": canonical_hash(provenance),
            }
            validate_fixture(record)
            name = inputs["name"] + ".json"
            (destination / name).write_text(
                json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            manifest["fixtures"].append(
                {
                    "file": name,
                    "inputs_hash": record["inputs_hash"],
                    "data_hash": record["data_hash"],
                    "record_hash": canonical_hash(record),
                }
            )
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(json.dumps(generate(arguments.output), indent=2))


if __name__ == "__main__":
    main()
