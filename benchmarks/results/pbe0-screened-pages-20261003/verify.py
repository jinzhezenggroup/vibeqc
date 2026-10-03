"""Recheck the corrected campaign without relabeling historical timings."""

from __future__ import annotations

import gzip
import hashlib
import json
import lzma
import sys
from pathlib import Path
from statistics import median

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from benchmarks.readme_pbe0 import SCHEMA
from tools.render_omol25_benchmarks import validate


def main() -> None:
    """Validate all 144 corrected endpoints and the original unchanged gates."""
    directory = Path(__file__).resolve().parent
    storage = json.loads((directory / "storage.json").read_text())
    compressed = (directory / "corrected-campaign.json.xz").read_bytes()
    assert hashlib.sha256(compressed).hexdigest() == storage["sha256"]
    raw = lzma.decompress(compressed)
    assert hashlib.sha256(raw).hexdigest() == storage["uncompressed_sha256"]
    members = json.loads(raw)
    assert set(members) == set(storage["members"])
    for name, value in members.items():
        assert hashlib.sha256(value.encode()).hexdigest() == storage["members"][name]
    summary = json.loads(members["summary.json"])
    qualification = json.loads(members["native-qualification.json"])
    assert qualification["status"] == "passed"
    assert qualification["memcheck_errors"] == qualification["initcheck_errors"] == 0
    for key in ("source_identity", "library_sha256"):
        assert storage[key] == summary[key] == qualification[key]
    cases = {case["atoms"]: case for case in summary["cases"]}
    assert set(cases) == {3, 6, 12, 24, 48, 96}
    checked = 0
    for atoms, case in sorted(cases.items()):
        reference_raw = gzip.decompress(
            (
                directory
                / storage["reference_root"]
                / f"water{atoms}-reference.json.gz"
            ).read_bytes()
        )
        reference = json.loads(reference_raw)
        validate(reference, reference, schema=SCHEMA)
        grids = []
        for variant in ("baseline", "candidate"):
            prefix = f"{variant}/{atoms}/"
            raw_record = members[prefix + "native.json"]
            record = json.loads(raw_record)
            campaign = json.loads(members[prefix + "campaign.json"])
            retained = case["variants"][variant]
            assert json.loads(members[prefix + "native.outcome"])["exit_code"] == 0
            assert hashlib.sha256(raw_record.encode()).hexdigest() == retained["sha256"]
            assert (
                record["reference_sha256"]
                == retained["reference_sha256"]
                == hashlib.sha256(reference_raw).hexdigest()
            )
            assert campaign == retained["campaign"]
            assert campaign["claim_consumption_barrier"] is True
            assert campaign["bounded_schwarz_schedule"] == (
                "0" if variant == "baseline" else "1"
            )
            for key in ("source_identity", "library_sha256"):
                assert campaign[key] == summary[key]
            assert record["native_build"]["library_sha256"] == summary["library_sha256"]
            assert (
                record["native_build"]["probe"]["source_identity"]
                == summary["source_identity"]
            )
            assert record["status"] == "measured" and record["stage"] == "complete"
            validate(record, reference, schema=SCHEMA)
            rows = record["records"]
            assert len(rows) == 12
            checked += len(rows)
            errors = {"energy": 0.0, "force": 0.0}
            for row in rows:
                assert row["converged"] and row["status"] == 0 and row["gate"]
                for oracle in reference["records"]:
                    if row["geometry"] != oracle["geometry"]:
                        continue
                    errors["energy"] = max(
                        errors["energy"], abs(row["energy"] - oracle["energy"])
                    )
                    errors["force"] = max(
                        errors["force"],
                        float(
                            np.max(np.abs(np.asarray(row["forces"]) - oracle["forces"]))
                        ),
                    )
            assert errors["energy"] <= 1e-8 and errors["force"] <= 1e-7
            assert (
                errors == retained["maximum_errors_all_same_geometry_reference_pairs"]
            )
            for phase in ("cold", "warm", "moved", "moved-warm"):
                selected = [row for row in rows if row["phase"] == phase]
                expected = retained["phases"][phase]
                seconds = [row["complete_seconds"] for row in selected]
                assert seconds == expected["seconds"]
                assert median(seconds) == expected["median"]
                assert [row["iterations"] for row in selected] == expected["iterations"]
            grid = rows[0]["native_force_components"]["grid_work_plan"]
            assert all(
                row["native_force_components"]["grid_work_plan"] == grid for row in rows
            )
            assert grid["grid_points"] == atoms * 48 * 16 * 32
            assert (
                grid["grid_pair_visits"]
                == (1 + 2 * grid["grid_points"]) * atoms * (atoms - 1) // 2
            )
            grids.append(grid)
        before, after = (
            case["variants"][variant] for variant in ("baseline", "candidate")
        )
        for key in (
            "host",
            "job",
            "cuda_visible_devices",
            "source_identity",
            "library_sha256",
        ):
            assert before["campaign"][key] == after["campaign"][key]
        assert grids[0] == grids[1]
        reduction = (
            1 - after["phases"]["warm"]["median"] / before["phases"]["warm"]["median"]
        )
        assert reduction == case["warm_reduction"]
        print(f"water{atoms}: corrected warm reduction {100 * reduction:.3f}%")
    assert checked == summary["native_endpoints_checked"] == 144
    assert storage["historical"]["commit"] == "0b99c6ce298f2726373f1909ad10a37b5acffc43"
    print("PASS: 144 corrected endpoints; historical pre-fix bytes remain separate")


if __name__ == "__main__":
    main()
