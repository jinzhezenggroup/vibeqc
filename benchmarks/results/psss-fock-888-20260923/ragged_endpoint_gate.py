"""Interleave the exact STO-3G ragged fixture that previously failed PR #888."""

import argparse
import ctypes
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", type=Path, required=True)
parser.add_argument("--candidate", type=Path, required=True)
parser.add_argument("--base-library", type=Path)
parser.add_argument("--candidate-library", type=Path)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--basis", default="sto-3g")
parser.add_argument("--batch", type=int, default=3)
parser.add_argument("--schedules", default="fixed,resident")
parser.add_argument("--cycles", type=int, default=3)
parser.add_argument("--repeats", type=int, default=7)
args = parser.parse_args()
roots = {"base": args.base.resolve(), "candidate": args.candidate.resolve()}
root = roots["candidate"]
libraries = {
    label: (
        getattr(args, label + "_library")
        or path / "build/cuda-release-sm120/libvibeqc.so"
    ).resolve()
    for label, path in roots.items()
}
output = args.output
output.parent.mkdir(parents=True, exist_ok=True)
assert os.environ.get("SLURM_JOB_ID")
sys.path.insert(0, str(roots["candidate"] / "python"))
from vibeqc.autotune import source_identity

identities = {}
for label, path in libraries.items():
    lib = ctypes.CDLL(str(path))
    lib.vibeqc_get_source_identity.restype = ctypes.c_char_p
    identities[label] = lib.vibeqc_get_source_identity().decode()
    assert identities[label] == source_identity(roots[label]), (
        label,
        identities[label],
        source_identity(roots[label]),
    )
payload = {
    "source_identities": identities,
    "slurm_job": os.environ["SLURM_JOB_ID"],
    "completed": False,
    "library_sha256": {
        k: hashlib.sha256(p.read_bytes()).hexdigest() for k, p in libraries.items()
    },
    "source_heads": {
        k: subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=p, text=True
        ).strip()
        for k, p in roots.items()
    },
    "records": [],
}
try:
    for schedule in args.schedules.split(","):
        config = {
            "method": "rhf",
            "basis": args.basis,
            "batch": args.batch,
            "repeats": args.repeats,
            "schedule": schedule,
            "resident_bra": schedule == "resident",
        }
        row = {"config": config, "runs": []}
        payload["records"].append(row)
        for label in ["base", "candidate", "candidate", "base"] * args.cycles:
            env = dict(
                os.environ,
                VIBEQC_LIBRARY=str(libraries[label]),
                VIBEQC_PROFILE="off",
                PYTHONPATH=f"{root}/python:{root}",
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                VIBEQC_PSSS_RESIDENT_BRA="1" if schedule == "resident" else "0",
                VIBEQC_BOUNDED_DIRECT_STREAMING="force"
                if schedule == "paged"
                else "none",
            )
            p = subprocess.run(
                [
                    sys.executable,
                    str(root / "tools/validate_weighted_eri_endpoints.py"),
                    "--worker",
                    json.dumps(config),
                ],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
            if p.returncode:
                raise RuntimeError(p.stderr)
            row["runs"].append({"label": label, "data": json.loads(p.stdout)})
            output.write_text(json.dumps(payload, indent=2) + "\n")
        reference = row["runs"][0]["data"]
        de = df = 0
        row["timing"] = {}
        for phase in ("cold", "warm", "changed_geometry", "changed_warm"):
            times = {"base": [], "candidate": []}
            expected = reference[phase]
            expected = expected[0] if isinstance(expected, list) else expected
            for run in row["runs"]:
                samples = run["data"][phase]
                samples = samples if isinstance(samples, list) else [samples]
                for sample in samples:
                    times[run["label"]].append(sample["milliseconds"])
                    de = max(
                        de,
                        float(
                            np.max(
                                np.abs(
                                    np.array(sample["energies"]) - expected["energies"]
                                )
                            )
                        ),
                    )
                    for actual, ref in zip(
                        sample["forces"], expected["forces"], strict=True
                    ):
                        df = max(df, float(np.max(np.abs(np.array(actual) - ref))))
            medians = {k: statistics.median(v) for k, v in times.items()}
            row["timing"][phase] = {
                **medians,
                "candidate_over_base": medians["candidate"] / medians["base"],
                "samples_ms": times,
            }
        row["work"] = {}
        for label in ("base", "candidate"):
            env = dict(
                os.environ,
                VIBEQC_LIBRARY=str(libraries[label]),
                VIBEQC_PROFILE="off",
                PYTHONPATH=f"{root}/python:{root}",
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
                VIBEQC_PSSS_RESIDENT_BRA="1" if schedule == "resident" else "0",
                VIBEQC_BOUNDED_DIRECT_STREAMING="force"
                if schedule == "paged"
                else "none",
            )
            profiled = subprocess.run(
                [
                    sys.executable,
                    str(root / "tools/validate_weighted_eri_endpoints.py"),
                    "--worker",
                    json.dumps(dict(config, profile_only=True)),
                ],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )
            row["work"][label] = json.loads(profiled.stdout)["work"]
        assert row["work"]["base"] == row["work"]["candidate"]
        row["energy_error"] = de
        row["force_error"] = df
        row["numerical_passed"] = de <= 2e-8 and df <= 2e-7
        row["performance_passed"] = all(
            x["candidate_over_base"] <= 1.02 for x in row["timing"].values()
        )
        print(
            json.dumps(
                {
                    "schedule": schedule,
                    "energy_error": de,
                    "force_error": df,
                    "ratios": {
                        k: v["candidate_over_base"] for k, v in row["timing"].items()
                    },
                }
            ),
            flush=True,
        )
    payload["completed"] = True
    payload["numerical_passed"] = all(r["numerical_passed"] for r in payload["records"])
    payload["performance_passed"] = all(
        r["performance_passed"] for r in payload["records"]
    )
finally:
    output.write_text(json.dumps(payload, indent=2) + "\n")
