"""Matched Release A/B endpoints; run the parent only in a Slurm GPU job.

Fresh child processes isolate library handles and runtime environment. All
samples retain their numerical results and iteration counts; timing never
filters a sample out of the accuracy gate. Profiling is a separate replay.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PHASES = ("cold", "warm", "changed", "changed_warm")


def worker(config: dict[str, Any]) -> dict[str, Any]:
    import ctypes

    import numpy as np
    from vibeqc import Calculator
    from vibeqc.autotune import source_identity

    from benchmarks._cases import benchmark_cases

    case = benchmark_cases()[config["case"]]
    systems = [case.atoms] * config["batch"]
    extra = {"density_fitting": "cuda"} if config.get("df") else {}
    calc = Calculator(
        method=case.method,
        basis=case.vibeqc_basis,
        device="cuda",
        basis_representation=case.basis_representation,
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
        **extra,
    )
    calc._library.vibeqc_get_source_identity.restype = ctypes.c_char_p
    native_identity = calc._library.vibeqc_get_source_identity().decode()
    if native_identity != source_identity(Path.cwd()):
        raise RuntimeError(
            "native scientific-source identity does not match the checkout"
        )
    moved = [np.array([xyz for _, xyz in case.atoms], dtype=float) for _ in systems]
    for index, xyz in enumerate(moved):
        xyz[-1, 2] += 0.01 + index * 0.001
    prepare_started = time.perf_counter()
    with calc.prepare_batch(
        systems,
        multiplicities=[case.multiplicity] * len(systems),
        warm_start=True,
    ) as prepared:
        preparation = time.perf_counter() - prepare_started

        def sample(coords: Any = None) -> dict[str, Any]:
            started = time.perf_counter()
            result = prepared.execute(coords, strict=True)
            elapsed = time.perf_counter() - started
            assert all(
                x.converged and x.executed_backend == "cuda" for x in result.items
            )
            return {
                "seconds": elapsed,
                "energies": result.energies.tolist(),
                "forces": [x.forces.tolist() for x in result.items],
                "iterations": [x.iterations for x in result.items],
            }

        data = {"preparation_seconds": preparation}
        data["cold"] = [sample()]
        data["warm"] = [sample() for _ in range(config["repeats"])]
        data["changed"] = [sample(moved)]
        data["changed_warm"] = [sample(moved) for _ in range(config["repeats"])]
    if config.get("df"):
        import tempfile

        # Memory polling and event tracing belong only to this intrusive pass.
        # They cannot inflate the clean complete-endpoint timings above.
        with tempfile.TemporaryDirectory(prefix="vibeqc-df-work-") as directory:
            trace = Path(directory) / "trace.jsonl"
            memory = Path(directory) / "memory.csv"
            os.environ["VIBEQC_DF_TRACE"] = str(trace)
            with memory.open("w") as stream:
                monitor = subprocess.Popen(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                        "--loop-ms=100",
                    ],
                    stdout=stream,
                    stderr=subprocess.DEVNULL,
                )
                try:
                    with calc.prepare_batch(
                        systems, multiplicities=[case.multiplicity] * len(systems)
                    ) as profiled:
                        profiled.execute(strict=True)
                        profiled.execute(strict=True)
                finally:
                    monitor.terminate()
                    monitor.wait(timeout=10)
                    os.environ.pop("VIBEQC_DF_TRACE", None)
            data["work"] = [json.loads(line) for line in trace.read_text().splitlines()]
            observed = [
                int(line.strip())
                for line in memory.read_text().splitlines()
                if line.strip().isdigit()
            ]
            data["sampled_device_peak_mib"] = max(observed) if observed else None
    else:
        with calc.prepare_batch(
            systems,
            multiplicities=[case.multiplicity] * len(systems),
            shell_class_profiling=True,
        ) as profiled:
            profiled.execute(strict=True)
            data["work"] = [
                dataclasses.asdict(x)
                for x in profiled.last_shell_class_profile()
                if x.shell_quartets
            ]
    data["head"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    data["source_identity"] = native_identity
    data["library_sha256"] = hashlib.sha256(
        Path(os.environ["VIBEQC_LIBRARY"]).read_bytes()
    ).hexdigest()
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker")
    parser.add_argument("--base", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--base-library", type=Path)
    parser.add_argument("--candidate-library", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--cases", default="water-def2-svp,oh-def2-svp-spherical-uhf")
    parser.add_argument("--schedules", default="fixed,resident,paged")
    parser.add_argument("--batches", default="1,3")
    parser.add_argument("--fallback", action="store_true")
    parser.add_argument("--df", action="store_true")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.loads(args.worker))))
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU qualification requires a Slurm allocation")
    import numpy as np

    payload = {
        "slurm_job": os.environ["SLURM_JOB_ID"],
        "completed": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "fallback": args.fallback,
        "records": [],
        "numerical_passed": True,
        "performance_passed": True,
        "max_regression": 0.02,
        "energy_atol": 1e-9,
        "force_atol": 1e-8,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def publish() -> None:
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    publish()
    try:
        for case in args.cases.split(","):
            for batch in map(int, args.batches.split(",")):
                for schedule in args.schedules.split(","):
                    config = {
                        "case": case,
                        "batch": batch,
                        "schedule": schedule,
                        "repeats": args.repeats,
                        "df": args.df,
                    }
                    row = {"config": config, "runs": []}
                    payload["records"].append(row)
                    for label in [
                        "base",
                        "candidate",
                        "candidate",
                        "base",
                    ] * args.cycles:
                        root = getattr(args, label).resolve()
                        library = (
                            getattr(args, label + "_library")
                            or root / "build/cuda-release-sm120/libvibeqc.so"
                        )
                        env = dict(
                            os.environ,
                            PYTHONPATH=f"{root}/python:{root}",
                            VIBEQC_LIBRARY=str(library.resolve()),
                            VIBEQC_PROFILE="off",
                            OPENBLAS_NUM_THREADS="1",
                            OMP_NUM_THREADS="1",
                            VIBEQC_PSSS_RESIDENT_BRA="1"
                            if schedule == "resident"
                            else "0",
                            VIBEQC_BOUNDED_DIRECT_STREAMING="force"
                            if schedule == "paged"
                            else "none",
                        )
                        if args.fallback:
                            env["VIBEQC_AOT_SHELL_CLASSES"] = "ssss"
                        else:
                            env.pop("VIBEQC_AOT_SHELL_CLASSES", None)
                        result = subprocess.run(
                            [sys.executable, __file__, "--worker", json.dumps(config)],
                            cwd=root,
                            env=env,
                            text=True,
                            capture_output=True,
                            check=False,
                        )
                        if result.returncode:
                            row["failure"] = {
                                "label": label,
                                "stderr": result.stderr,
                                "stdout": result.stdout,
                            }
                            raise RuntimeError(
                                f"{case}/{batch}/{schedule}/{label}: {result.stderr[-2000:]}"
                            )
                        row["runs"].append(
                            {"label": label, "data": json.loads(result.stdout)}
                        )
                        publish()
                    reference = row["runs"][0]["data"]
                    errors_e, errors_f = [], []
                    row["timing"] = {}
                    for phase in PHASES:
                        samples = {
                            label: [
                                s
                                for r in row["runs"]
                                if r["label"] == label
                                for s in r["data"][phase]
                            ]
                            for label in ("base", "candidate")
                        }
                        for label, values in samples.items():
                            for sample in values:
                                expected = reference[phase][0]
                                errors_e.append(
                                    float(
                                        np.max(
                                            np.abs(
                                                np.array(sample["energies"])
                                                - expected["energies"]
                                            )
                                        )
                                    )
                                )
                                errors_f.append(
                                    float(
                                        np.max(
                                            np.abs(
                                                np.array(sample["forces"])
                                                - expected["forces"]
                                            )
                                        )
                                    )
                                )
                        medians = {
                            label: statistics.median(s["seconds"] for s in values)
                            for label, values in samples.items()
                        }
                        ratio = medians["candidate"] / medians["base"]
                        iterations = {
                            label: sorted({tuple(s["iterations"]) for s in values})
                            for label, values in samples.items()
                        }
                        row["timing"][phase] = {
                            **medians,
                            "candidate_over_base": ratio,
                            "iterations": iterations,
                        }
                        payload["performance_passed"] &= (
                            ratio <= 1.02
                            and iterations["base"] == iterations["candidate"]
                        )
                    row["max_energy_error"] = max(errors_e)
                    row["max_force_error"] = max(errors_f)
                    row["prepare_plus_cold_seconds"] = {
                        label: statistics.median(
                            r["data"]["preparation_seconds"]
                            + r["data"]["cold"][0]["seconds"]
                            for r in row["runs"]
                            if r["label"] == label
                        )
                        for label in ("base", "candidate")
                    }
                    payload["numerical_passed"] &= (
                        max(errors_e) <= 1e-9 and max(errors_f) <= 1e-8
                    )
                    row["work_equal"] = all(
                        r["data"]["work"] == reference["work"] for r in row["runs"]
                    )
                    if not args.df:
                        payload["numerical_passed"] &= row["work_equal"]
                    print(
                        json.dumps(
                            {
                                "config": config,
                                "energy": max(errors_e),
                                "force": max(errors_f),
                                "work_equal": row["work_equal"],
                                "ratios": {
                                    p: t["candidate_over_base"]
                                    for p, t in row["timing"].items()
                                },
                            }
                        ),
                        flush=True,
                    )
                    publish()
        payload["completed"] = True
    except Exception as error:
        payload["failure"] = str(error)
        raise
    finally:
        publish()


if __name__ == "__main__":
    main()
