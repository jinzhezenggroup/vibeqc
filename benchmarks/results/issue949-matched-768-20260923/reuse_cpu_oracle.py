"""Run the DF endpoint with the completed #1049 independent CPU oracle.

The retained 768-AO receipt is checked against the active case and basis before
delegating all native execution and numerical gates to df_policy_endpoint.
This does not reuse the historical GPU timing or native library.
"""

import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
from benchmarks import df_policy_endpoint as endpoint  # noqa: E402


RECEIPT = ROOT / "benchmarks/results/acceptance-closeout-20260922/df-768-panel-ablation.json.gz"


def retained_cpu_reference(case, orbital_basis, auxiliary_basis):
    with gzip.open(RECEIPT, "rt") as stream:
        source = json.load(stream)
    settings = source["scientific_settings"]
    retained = source["cpu_reference"]
    requirements = {
        "aos": source["aos"] == 768,
        "historical_source": source["git_head"] == "0d89ab6fb6219d9af6641d5cfafe0b43f8387206",
        "geometry": json.loads(json.dumps(case.atoms)) == settings["geometries_bohr"],
        "method": case.method == settings["method"] == "rhf",
        "charge_spin": case.charge == 0 and case.multiplicity == 1,
        "representation": case.basis_representation == settings["basis_representation"] == "spherical",
        "orbital_basis": orbital_basis == retained["orbital_basis"] == settings["basis"],
        "auxiliary_basis": auxiliary_basis == retained["auxiliary_basis"] == settings["auxiliary_basis"],
        "basis_size": retained["ao_count"] == retained["auxiliary_count"] == 768,
        "oracle_thresholds": (
            retained["energy_tolerance"] == 1e-13
            and retained["gradient_tolerance"] == 1e-12
            and retained["direct_scf_tolerance"] == 1e-14
            and retained["auxbasis_response"] is True
        ),
        "native_thresholds": (
            settings["metric_relative_threshold"] == 1e-10
            and settings["energy_tolerance"] == 1e-12
            and settings["density_tolerance"] == 1e-10
            and settings["screening_tolerance"] == 1e-12
            and settings["max_iterations"] == 100
            and settings["density_fitting_memory_budget_bytes"] == 0
        ),
    }
    if failed := [name for name, passed in requirements.items() if not passed]:
        raise RuntimeError(f"retained CPU oracle differs: {', '.join(failed)}")
    energy = np.asarray(retained["energies_hartree"])
    force = np.asarray(retained["forces_hartree_per_bohr"])
    if (
        energy.shape != (1,)
        or force.shape != (1, len(case.atoms), 3)
        or not np.isfinite(energy).all()
        or not np.isfinite(force).all()
    ):
        raise RuntimeError("retained CPU oracle arrays are invalid")
    return energy, force, retained


def main():
    endpoint.cpu_reference = retained_cpu_reference
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    try:
        endpoint.main()
    finally:
        if output.exists():
            payload = json.loads(output.read_text())
            payload["reference_reuse"] = {
                "receipt": str(RECEIPT.relative_to(ROOT)),
                "sha256": hashlib.sha256(RECEIPT.read_bytes()).hexdigest(),
                "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "historical_source": "0d89ab6fb6219d9af6641d5cfafe0b43f8387206",
                "scope": "independent CPU numbers only; fresh native library and same-device A/B",
            }
            output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
