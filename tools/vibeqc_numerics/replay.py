"""Re-evaluate recorded densities with the strict operator, without SCF replay."""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

import numpy as np
from vibeqc import Calculator, ResolvedModel
from vibeqc.profiles import canonical_hash, file_hash

from tools.vibeqc_numerics.audit import HFProbe, ProbeControls, StrictHFAudit
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_validation.fixtures import calculator_inputs


def replay(report_path: Path) -> dict:
    """Validate portable state sizes/checksums and reconstruct each physical audit.

    The archive is data-only. Array names/shapes and total uncompressed sizes
    are checked before NumPy allocation. A replay never resumes a solver or
    treats stored convergence/evidence labels as new convergence verification.
    """
    record = json.loads(report_path.read_text())
    digest = record.pop("record_hash")
    if (
        record.get("schema") != "vibeqc.accuracy_experiment"
        or type(record.get("schema_version")) is not int
        or record.get("schema_version") != 1
    ):
        raise ValueError("unsupported accuracy replay schema")
    if canonical_hash(record) != digest:
        raise ValueError("accuracy report checksum mismatch")
    archive_name = record["states"]["file"]
    if Path(archive_name).name != archive_name:
        raise ValueError("replay state must be a sibling data file")
    path = report_path.parent / archive_name
    if (
        not path.is_file()
        or path.stat().st_size != record["states"]["bytes"]
        or path.stat().st_size > 128 << 20
    ):
        raise ValueError("invalid replay archive size")
    if file_hash(path) != record["states"]["sha256"]:
        raise ValueError("replay state checksum mismatch")
    shapes = record["states"]["array_shapes"]
    for shape in shapes.values():
        if (
            not shape
            or len(shape) > 3
            or any(type(n) is not int or not 0 < n <= 4096 for n in shape)
        ):
            raise ValueError("invalid replay array shape")
        if math.prod(shape) > 3 * 4096:
            raise ValueError("replay array exceeds the small-system diagnostic domain")
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if len(entries) != len(shapes) or {e.filename for e in entries} != {
            k + ".npy" for k in shapes
        }:
            raise ValueError("replay archive names differ from the manifest")
        if any(
            e.file_size > 4096 + 8 * math.prod(shapes[e.filename[:-4]]) for e in entries
        ):
            raise ValueError("replay array storage exceeds its declared shape")
        if sum(e.file_size for e in entries) > 128 << 20:
            raise ValueError("replay archive exceeds its uncompressed budget")

    reports = []
    with np.load(path, allow_pickle=False) as arrays:
        for name, shape in shapes.items():
            array = arrays[name]
            if (
                array.shape != tuple(shape)
                or array.dtype.kind != "f"
                or array.dtype.itemsize != 8
                or not np.isfinite(array).all()
            ):
                raise ValueError(
                    "replay array shape/dtype/values differ from the manifest"
                )
        for case in record["cases"]:
            if case["status"] != "pass":
                reports.append(
                    {
                        "case": case["name"],
                        "status": "not_run",
                        "reason": "original strict reference failed",
                    }
                )
                continue
            inputs = case["inputs"]
            atoms = list(
                zip(inputs["atomic_numbers"], inputs["coordinates"], strict=True)
            )
            calc = Calculator(**calculator_inputs(inputs))
            model = ResolvedModel.from_dict(case["model"])
            with NativeSource(atoms, calc._basis, charge=inputs["charge"]) as source:
                audit = StrictHFAudit(source, model)
                strict = {
                    **case["strict"],
                    "label": "strict",
                    "controls": case["strict_controls"],
                    "physical_audit": case["strict_audit"],
                    "status": "converged",
                }
                for row in (strict, *case["rows"]):
                    if row["status"] != "converged":
                        reports.append(
                            {
                                "case": case["name"],
                                "label": row["label"],
                                "status": "not_run",
                                "reason": row["status"],
                            }
                        )
                        continue
                    prefix = case["name"] + "__" + row["label"]
                    density, forces = (
                        arrays[prefix + "__density"],
                        arrays[prefix + "__forces"],
                    )
                    spins = 1 if model.method == "rhf" else 2
                    if density.shape != (
                        spins,
                        source.nbf,
                        source.nbf,
                    ) or forces.shape != (len(atoms), 3):
                        raise ValueError(
                            "replay state does not match its scientific source"
                        )
                    probe = HFProbe(
                        model,
                        ProbeControls(**row["controls"]),
                        record["backend"],
                        row["energy"],
                        row["energy_change"],
                        row["density_rms"],
                        row["iterations"],
                        True,
                        density,
                        forces,
                        0.0,
                        row.get("requested_mixed_fock_threshold"),
                    )
                    current = audit.evaluate(probe)
                    keys = (
                        "energy",
                        "orthonormal_commutator_max",
                        "ao_commutator_max",
                        "electron_trace_error_max",
                        "density_idempotency_max",
                    )
                    errors = {
                        k: abs(current[k] - row["physical_audit"][k]) for k in keys
                    }
                    reports.append(
                        {
                            "case": case["name"],
                            "label": row["label"],
                            "status": "pass"
                            if max(errors.values()) <= 1e-11
                            else "fail",
                            "differences": errors,
                        }
                    )
    return {
        "schema_version": 1,
        "original_record_hash": digest,
        "mode": "fixed_density_operator_replay",
        "scf_executed": False,
        "rows": reports,
        "passed": (
            any(r["status"] == "pass" for r in reports)
            and all(case["status"] == "pass" for case in record["cases"])
            and not any(r["status"] == "fail" for r in reports)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = replay(arguments.report)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.write_text(serialized)
    else:
        print(serialized, end="")
    if not result["passed"]:
        raise SystemExit("fixed-density audit replay failed")


if __name__ == "__main__":
    main()
