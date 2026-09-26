"""Run the existing #206 comparator serially and retain every failed attempt.

Use the promoted default, including the unchanged small-system route.
No tracing, memory sampler, compilation, or independent-reference work overlaps
an endpoint. All-repeat correctness is checked in addition to the comparator's
historical branch-selected gate. Branch differences remain explicit.
"""

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from pathlib import Path

assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")
root = Path.cwd()
out = Path(sys.argv[1]).resolve()
out.mkdir(parents=True, exist_ok=False)
# Release the provider-check process before launching measurements. The runner
# must not keep a second GPU context resident beside either timed engine.
subprocess.run(
    [
        sys.executable,
        "-c",
        "from gpu4pyscf.lib import cutensor; assert cutensor.cutensor is not None and cutensor.contract_engine is None",
    ],
    check=True,
)
manifest = {
    "scope": "Promoted generated-shell default: matched 96/192/384-AO RHF and 19-AO UHF energy/complete-force matrix; full #206 closure remains separate",
    "slurm_job": os.environ["SLURM_JOB_ID"],
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "git_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "git_status": subprocess.check_output(["git", "status", "--porcelain"], text=True),
    "library_sha256": hashlib.sha256(
        Path(os.environ["VIBEQC_LIBRARY"]).read_bytes()
    ).hexdigest(),
    "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "comparator_sha256": hashlib.sha256(
        (root / "benchmarks/compare_gpu4pyscf_batch.py").read_bytes()
    ).hexdigest(),
    "versions": {
        n: importlib.metadata.version(n)
        for n in (
            "gpu4pyscf-cuda12x",
            "cupy-cuda12x",
            "cutensor-cu12",
            "pyscf",
            "numpy",
            "scipy",
        )
    },
    "contract_engine": "cuTENSOR",
    "gpu_before": subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,pstate,clocks.sm,clocks.mem,power.limit",
            "--format=csv",
        ],
        text=True,
    ),
    "controls": {k: v for k, v in os.environ.items() if k.startswith("VIBEQC_")},
    "source_identity": json.loads(
        (Path(__file__).parent / "provenance.json").read_text()
    )["source_identity"],
    "attempts": [],
}


def save():
    tmp = out / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.replace(out / "manifest.json")


save()
qualification = out / "input-qualification.json"
subprocess.run(
    [
        sys.executable,
        str(Path(__file__).with_name("qualify-inputs.py")),
        str(qualification),
    ],
    check=True,
)
manifest["input_qualification_sha256"] = hashlib.sha256(
    qualification.read_bytes()
).hexdigest()
save()
for name, aos, serial in [
    ("water-tetramer-def2-svp-spherical", 96, False),
    ("water-octamer-s4-def2-svp-spherical", 192, False),
    ("oh-def2-svp-spherical-uhf", 19, False),
    ("water-hexadecamer-2s4-def2-svp-spherical", 384, False),
]:
    os.environ["VIBEQC_DF_HOST_RESPONSE_WEIGHTS"] = "0"
    os.environ.pop("VIBEQC_DF_SERIAL_RESPONSE_DOT", None)
    for batch in (1, 4):
        for forces in (True,) if serial else (False, True):
            stem = (
                f"{aos}ao-b{batch}-" + ("forces" if forces else "energy") + "-default"
            )
            result = out / f"{stem}.json"
            command = [
                sys.executable,
                str(root / "benchmarks/compare_gpu4pyscf_batch.py"),
                "--case",
                name,
                "--batch",
                str(batch),
                "--repeats",
                "5",
                "--density-fitting",
                "cuda",
                "--density-fitting-memory-budget-bytes",
                "0",
                "--maximum-energy-error",
                "1e-9",
                "--max-iterations",
                "100",
                "--energy-tolerance",
                "1e-12",
                "--density-tolerance",
                "1e-10",
                "--reference-gradient-tolerance",
                "1e-10",
                "--output",
                str(result),
            ]
            command += (
                ["--maximum-force-error", "1e-8"] if forces else ["--energy-only"]
            )
            row = {
                "case": name,
                "aos": aos,
                "batch": batch,
                "forces": forces,
                "serial_metric_dot": serial,
                "command": command,
                "status": "running",
            }
            manifest["attempts"].append(row)
            save()
            start = time.monotonic()
            with (out / f"{stem}.log").open("w") as log:
                try:
                    run = subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=1200,
                        check=False,
                    )
                    row["returncode"] = run.returncode
                    row["status"] = "completed" if run.returncode == 0 else "failed"
                except subprocess.TimeoutExpired:
                    row["status"] = "timeout"
            row["seconds"] = time.monotonic() - start
            if result.exists():
                row["result_sha256"] = hashlib.sha256(result.read_bytes()).hexdigest()
                payload = json.loads(result.read_text())

                # Search the established result schema without silently assuming
                # that an older/newer comparator has published all repeat rows.
                def pairs(value):
                    if isinstance(value, dict):
                        if "paired_warm_repeats" in value:
                            return value["paired_warm_repeats"]
                        for child in value.values():
                            found = pairs(child)
                            if found is not None:
                                return found
                    return None

                paired = pairs(payload)
                row["all_repeat_accuracy_passed"] = (
                    bool(paired)
                    and len(paired) == 5
                    and all(
                        p["maximum_energy_error_hartree"] <= 1e-9
                        and (
                            not forces
                            or p["maximum_force_error_hartree_per_bohr"] <= 1e-8
                        )
                        for p in paired
                    )
                )
                if not row["all_repeat_accuracy_passed"]:
                    row["status"] = "failed_all_repeat_accuracy"
            elif row["status"] == "completed":
                row["status"] = "failed_missing_result"
            save()
            print(stem, row["status"], flush=True)
manifest["gpu_after"] = subprocess.check_output(
    [
        "nvidia-smi",
        "--query-gpu=name,driver_version,pstate,clocks.sm,clocks.mem,power.limit",
        "--format=csv",
    ],
    text=True,
)
manifest["status"] = (
    "completed"
    if all(r["status"] == "completed" for r in manifest["attempts"])
    else "failed"
)
save()
raise SystemExit(0 if manifest["status"] == "completed" else 1)
