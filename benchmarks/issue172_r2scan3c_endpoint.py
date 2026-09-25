"""Record complete r2SCAN-3c energy-plus-force endpoint phases on CUDA.

Run this only in a real GPU allocation after the source-matched independent
oracle and finite-difference tests pass. The JSON preserves each sample rather
than reducing the observed timing to a purported speedup.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import typing
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("VIBEQC_SOURCE_DIR", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "python"))

from vibeqc import Calculator, KsOptions, load_r2scan3c_basis

H2 = [("H", (0.1, -0.1, -0.7)), ("H", (0.0, 0.1, 0.8))]
H3_CATION = [*H2, ("H", (1.6, 0.2, 0.0))]
H2_DIMER = [*H2, ("H", (4.5, 0.2, -0.8)), ("H", (4.4, 0.3, 0.7))]
WATER = [("O", (0.0, 0.0, 0.0)), ("H", (0.75, 0.58, 0.0)), ("H", (-0.75, 0.58, 0.0))]
OH = [("O", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.8))]
CASES = (
    ("h2-rks", H2, 0, 1),
    ("h3-cation-rks", H3_CATION, 1, 1),
    ("h2-dimer-rks", H2_DIMER, 0, 1),
    ("water-rks-spd", WATER, 0, 1),
    ("oh-uks-spd", OH, 0, 2),
    ("water-cation-uks-spd", WATER, 1, 2),
)


def _save(path: Path, report: dict) -> None:
    pending = path.with_suffix(".json.tmp")
    pending.write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    pending.replace(path)


def _record(batch: typing.Any, *, phase: str, coordinates: typing.Any = None) -> dict:
    started = time.perf_counter()
    result = batch.execute(
        coordinates=coordinates, strict=True, properties=("energy", "forces")
    )
    seconds = time.perf_counter() - started
    item = result.items[0]
    assert item.succeeded and item.forces is not None and item.dispersion is not None
    return {
        "phase": phase,
        "complete_energy_force_seconds": seconds,
        "energy_eh": item.energy,
        "forces_eh_per_bohr": item.forces.tolist(),
        "iterations": item.iterations,
        "physical_residual_rms": item.physical_residual_rms,
        "correction_backend": item.dispersion.backend,
        "d4_energy_eh": item.dispersion.d4.energy,
        "gcp_energy_eh": item.dispersion.gcp.energy,
        "resource_diagnostics": batch.resource_diagnostics,
    }


def main() -> int:
    allocation = os.environ.get("SLURM_JOB_ID") or os.environ.get("VIBEQC_QZ_WORKLOAD")
    output = os.environ.get("VIBEQC_R2SCAN3C_BENCH_OUT")
    if not allocation or not output:
        raise SystemExit("real GPU allocation and VIBEQC_R2SCAN3C_BENCH_OUT required")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = os.environ.get("VIBEQC_R2SCAN3C_BENCH_CASES")
    selected_names = set(selected.split(",")) if selected else None
    cases = tuple(
        case for case in CASES if selected_names is None or case[0] in selected_names
    )
    if not cases or (
        selected_names is not None and selected_names != {case[0] for case in cases}
    ):
        raise SystemExit("VIBEQC_R2SCAN3C_BENCH_CASES contains an unknown case")
    library = Path(os.environ["VIBEQC_LIBRARY"])
    with library.open("rb") as stream:
        library_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "schema": "vibeqc.issue172.r2scan3c_complete_endpoint.v1",
        "allocation": allocation,
        "source_tree": os.environ.get("VIBEQC_SOURCE_TREE"),
        "library_sha256": library_hash,
        "basis_identity": load_r2scan3c_basis().identity,
        "grid": "production default KsOptions",
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
        "properties": ["energy", "forces"],
        "case_filter": sorted(selected_names) if selected_names is not None else None,
        "cases": [],
    }
    _save(output_path, report)
    for name, atoms, charge, multiplicity in cases:
        row = {
            "name": name,
            "atom_count": len(atoms),
            "charge": charge,
            "multiplicity": multiplicity,
            "phases": [],
        }
        report["cases"].append(row)
        _save(output_path, report)
        calc = Calculator(
            method="r2scan-3c-rks" if multiplicity == 1 else "r2scan-3c-uks",
            device="cuda",
            ks_options=KsOptions(),
            max_iterations=200,
            energy_tolerance=1e-12,
            density_tolerance=1e-10,
        )
        started = time.perf_counter()
        with calc.prepare_batch(
            [atoms], charges=[charge], multiplicities=[multiplicity]
        ) as batch:
            row["prepare_seconds"] = time.perf_counter() - started
            for phase in ("cold", "warm-1", "warm-2", "warm-3"):
                row["phases"].append(_record(batch, phase=phase))
                if phase == "cold":
                    batch.set_warm_start_updates(False)
                _save(output_path, report)
            positions = np.asarray(
                [position for _, position in atoms], dtype=np.float64
            )
            moved = positions.copy()
            moved[-1] += [0.002, -0.001, 0.003]
            row["phases"].append(_record(batch, phase="changed", coordinates=[moved]))
            _save(output_path, report)
        fresh_atoms = [
            (symbol, tuple(position))
            for (symbol, _), position in zip(atoms, moved, strict=True)
        ]
        started = time.perf_counter()
        fresh = calc.singlepoint(
            fresh_atoms,
            charge=charge,
            multiplicity=multiplicity,
            properties=("energy", "forces"),
        )
        row["fresh_complete_energy_force_seconds"] = time.perf_counter() - started
        row["fresh_energy_eh"] = fresh.energy
        row["fresh_forces_eh_per_bohr"] = fresh.forces.tolist()
        row["fresh_iterations"] = fresh.iterations
        row["fresh_physical_residual_rms"] = fresh.physical_residual_rms
        row["changed_fresh_energy_error_eh"] = abs(
            row["phases"][-1]["energy_eh"] - fresh.energy
        )
        row["changed_fresh_force_max_error_eh_per_bohr"] = float(
            np.max(
                np.abs(
                    np.asarray(row["phases"][-1]["forces_eh_per_bohr"]) - fresh.forces
                )
            )
        )
        row["warm_energy_max_error_eh"] = max(
            abs(phase["energy_eh"] - row["phases"][0]["energy_eh"])
            for phase in row["phases"][1:4]
        )
        row["warm_force_max_error_eh_per_bohr"] = max(
            float(
                np.max(
                    np.abs(
                        np.asarray(phase["forces_eh_per_bohr"])
                        - np.asarray(row["phases"][0]["forces_eh_per_bohr"])
                    )
                )
            )
            for phase in row["phases"][1:4]
        )
        row["passed"] = (
            row["changed_fresh_energy_error_eh"] < 2e-9
            and row["changed_fresh_force_max_error_eh_per_bohr"] < 1e-7
            and row["warm_energy_max_error_eh"] < 2e-9
            and row["warm_force_max_error_eh_per_bohr"] < 1e-7
        )
        _save(output_path, report)
        print(name, "pass" if row["passed"] else "FAIL", flush=True)
        if not row["passed"]:
            report["passed"] = False
            _save(output_path, report)
            return 1
    report["passed"] = True
    _save(output_path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
