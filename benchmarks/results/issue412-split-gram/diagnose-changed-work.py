"""Changed-work follow-up and intrusive sampled peaks, after clean timing.

Each process restores the same checkpoint. Sampling covers setup through teardown,
so maxima are allocation-wide diagnostic lower bounds, not clean endpoint timings.
"""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

assert os.environ.get("SLURM_JOB_ID"), "requires finite Slurm allocation"
root = Path.cwd()
out = root / ".artifacts/issue412/changed-work-diagnostics"
out.mkdir(exist_ok=False)
env = os.environ.copy()
env.update(
    PYTHONPATH="python:.",
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    LD_LIBRARY_PATH="/group/software/cuda-12.9.1/lib64:"
    + env.get("LD_LIBRARY_PATH", ""),
    VIBEQC_DF_REFERENCE_FINAL_VALIDATION="0",
    VIBEQC_DF_FINAL_PROJECTION="auto",
    VIBEQC_DF_FORCE_SCREEN_ABS="off",
    VIBEQC_DF_EXCHANGE="auto",
    VIBEQC_DF_SEED_EXCHANGE="auto",
    VIBEQC_DF_FINAL_EXCHANGE="auto",
    VIBEQC_DF_SHELL_POLICY="auto",
)
(out / "device.json").write_text(
    json.dumps(
        {
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "query": "name,uuid,driver_version,memory.total,compute_cap",
            "csv": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
                    "--format=csv,noheader",
                ],
                text=True,
            ),
        },
        indent=2,
    )
    + "\n"
)
records = []
for aos, case in [(768, "water-32mer-4s4-def2-svp-spherical")]:
    for observable in ["forces", "energy"]:
        label = f"{aos}-{observable}"
        env["VIBEQC_LIBRARY"] = str(root / ".artifacts/issue412/candidate/libvibeqc.so")
        command = [
            "/tmp/vibeqc-pr-review-env/bin/python",
            "-m",
            "benchmarks.df_policy_endpoint",
            "--aos",
            str(aos),
            "--output",
            str(out / f"{label}.json"),
            "--repeats",
            "1",
            "--control",
            "VIBEQC_DF_RESIDENT_EXCHANGE",
            "--policies",
            "auto",
            "split4",
        ]
        command += [
            "--trace",
            "--journal",
            "--source-patch",
            ".artifacts/issue412/candidate/source.patch",
            "--skip-cold",
            "--warm-checkpoint-in",
            f"/home/jzzeng/codes/vibeqc-issues388-391/.artifacts/issues388-391/final/ablations/{aos}-VIBEQC_DF_DIIS_DOTS.checkpoint",
            "--reference",
            f"benchmarks/results/issue377-379-df/gpu4pyscf/{case}.json",
        ]
        if observable == "energy":
            command += ["--energy-only"]
        with (out / f"{label}.stdout").open("w") as log:
            process = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            readings = []
            while process.poll() is None:
                rss = highwater = 0
                try:
                    for line in (
                        Path(f"/proc/{process.pid}/status").read_text().splitlines()
                    ):
                        if line.startswith("VmRSS:"):
                            rss = int(line.split()[1]) * 1024
                        if line.startswith("VmHWM:"):
                            highwater = int(line.split()[1]) * 1024
                except FileNotFoundError:
                    pass
                gpu_bytes = 0
                sample = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                for line in sample.stdout.splitlines():
                    pid, mib = line.split(",")
                    if int(pid) == process.pid:
                        gpu_bytes = int(mib) * 1024**2
                readings.append([time.monotonic(), rss, highwater, gpu_bytes])
                time.sleep(0.2)
            code = process.wait()
        raw = out / f"{label}.memory-samples.json"
        raw.write_text(json.dumps(readings) + "\n")
        record = {
            "label": label,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "exit_code": code,
            "command": command,
            "scope": "intrusive changed-work diagnostic after fixed-work stop; exclude all times from performance qualification",
            "peak_host_rss_bytes": max(x[1] for x in readings),
            "peak_host_highwater_bytes": max(x[2] for x in readings),
            "peak_process_device_resident_bytes": max(x[3] for x in readings),
            "sample_count": len(readings),
            "samples_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "definition": "Per-process NVIDIA reported device residency at 200 ms plus query cost, and /proc VmRSS/VmHWM from initialization through teardown. Device peak is a sampled lower bound, includes CUDA context/modules, and may miss short allocations; it is not an allocator peak. Each process runs both policies for one observable; the peak is not attributable to either arm separately.",
        }
        records.append(record)
        (out / "resources.json").write_text(json.dumps(records, indent=2) + "\n")
        print(label, code, record["peak_process_device_resident_bytes"], flush=True)
        if code:
            raise SystemExit(code)
