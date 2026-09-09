"""Generate small SOL01 geometry families with pinned independent PySCF HF.

Run only when explicitly regenerating reference data. Ordinary tests and data
generation use the committed references and do not import/download PySCF.
"""

import argparse
import json
import platform
from copy import deepcopy
from pathlib import Path

import numpy as np
from vibeqc.profiles import canonical_hash, file_hash

from tools.generate_validation_references import molecular_data, pyscf_molecule
from tools.vibeqc_numerics.fixtures import FAMILIES, accuracy_suite


def generate(path):
    import pyscf
    from pyscf import scf
    from threadpoolctl import threadpool_limits

    if pyscf.__version__ != "2.14.0":
        raise ValueError("reference generation requires PySCF 2.14.0")
    pyscf.lib.num_threads(1)
    originals = {r["inputs"]["name"]: r["inputs"] for r in accuracy_suite()}
    rows = []
    with threadpool_limits(limits=1):
        for name in ("h2", "h2-def2-svp", "h2o", "ch4", "hf-plus-uhf"):
            for index, scale_geometry in enumerate((0.98, 1.0, 1.02)):
                inputs = deepcopy(originals[name])
                inputs["cc_settings"] = None
                inputs["name"] = f"{name}-{index}"
                inputs["coordinates"] = (
                    np.asarray(inputs["coordinates"]) * scale_geometry
                ).tolist()
                molecule, scale, _ = pyscf_molecule(inputs)
                data = molecular_data(inputs, molecule, scale)
                alternate = (scf.RHF if inputs["method"] == "rhf" else scf.UHF)(
                    molecule
                )
                alternate.conv_tol = 1e-13
                alternate.conv_tol_grad = 1e-10
                alternate.max_cycle = 300
                alternate.init_guess = "1e"
                alternate.kernel()
                # A competing start and an internal stability calculation are
                # separate evidence; neither is a global minimum certificate.
                check = {
                    "guess": "core_hamiltonian",
                    "converged": bool(alternate.converged),
                    "energy": float(alternate.e_tot),
                    "internal_stability": "not_evaluated",
                    "external_stability": "not_evaluated",
                }
                if alternate.converged:
                    try:
                        _, _, internal, _ = alternate.stability(
                            internal=True, external=False, return_status=True
                        )
                        check["internal_stability"] = (
                            "stable" if internal else "unstable"
                        )
                    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                        check["stability_failure"] = f"{type(exc).__name__}: {exc}"
                family = FAMILIES[name]
                rows.append(
                    {
                        "name": inputs["name"],
                        "trajectory": name,
                        "geometry_index": index,
                        "family": family,
                        "split": "training"
                        if family in ("water", "methane")
                        else "holdout",
                        "inputs": inputs,
                        "reference": data,
                        "competing_solution": check,
                    }
                )
    record = {
        "schema": "vibeqc.scf_reference_families",
        "schema_version": 1,
        "cases": rows,
        "provenance": {
            "pyscf": pyscf.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "threads": 1,
            "generator_sha256": file_hash(Path(__file__)),
            "shared_generator_sha256": file_hash(
                Path(__file__).with_name("generate_validation_references.py")
            ),
            "libcint_sha256": file_hash(
                Path(pyscf.__file__).parent / "lib/deps/lib/libcint.so"
            ),
        },
    }
    record["record_hash"] = canonical_hash(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    return {
        "cases": len(rows),
        "file": str(path),
        "bytes": path.stat().st_size,
        "record_hash": record["record_hash"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(generate(parser.parse_args().output)))
