"""Measure #202 same-path endpoints from two explicit production builds.

Run the coordinator inside a finite Slurm GPU job. Each worker imports only
its selected revision and loads that revision's library in a fresh process.
Raw timings and complete numerical endpoints go to an untracked artifact
directory. A comparison never promotes a source or changes a runtime policy.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter


def command(argv, **kwargs):
    """Capture a checked command without invoking a shell."""
    return subprocess.check_output(argv, text=True, **kwargs).strip()


def provenance(root, build):
    """Freeze source/build/environment inputs used by this worker."""
    cache = (build / "CMakeCache.txt").read_text()
    keys = (
        "CMAKE_BUILD_TYPE",
        "CMAKE_CUDA_ARCHITECTURES",
        "CMAKE_CUDA_COMPILER",
        "CMAKE_CXX_COMPILER",
        "VIBEQC_CUDA_FAST_COMPILE",
        "VIBEQC_ENABLE_AOT_SHELLS",
        "VIBEQC_AOT_PROFILE",
        "VIBEQC_CUDA_SPLIT_COMPILE_THREADS",
    )
    entries = dict(
        line.split("=", 1)
        for line in cache.splitlines()
        if "=" in line and not line.startswith(("#", "//"))
    )
    controls = {
        key: next((v for k, v in entries.items() if k.split(":")[0] == key), None)
        for key in keys
    }
    if (
        controls["CMAKE_BUILD_TYPE"] != "Release"
        or controls["VIBEQC_CUDA_FAST_COMPILE"] != "OFF"
    ):
        raise ValueError(
            "matched endpoint evidence requires Release and FAST_COMPILE=OFF"
        )
    return {
        "revision": command(["git", "-C", str(root), "rev-parse", "HEAD"]),
        "dirty": bool(command(["git", "-C", str(root), "status", "--porcelain"])),
        "build": controls,
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "environment": {
            k: v
            for k, v in os.environ.items()
            if k.startswith("VIBEQC_")
            or k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "gpu": command(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "cxx": command([controls["CMAKE_CXX_COMPILER"], "--version"]).splitlines()[0],
        "nvcc": command([controls["CMAKE_CUDA_COMPILER"], "--version"]),
    }


def worker(args):
    """Time synchronous public endpoints without import/setup clock pollution."""
    sys.path.insert(0, str(args.root / "python"))
    import numpy as np
    from vibeqc import Calculator

    record = provenance(args.root, args.build)
    record["filters"] = {
        key: getattr(args, key) for key in ("case", "spin", "approximation", "endpoint")
    }
    record["numpy"] = np.__version__
    library = ctypes.CDLL(str(args.build / "libvibeqc.so"))
    library.vibeqc_get_source_identity.restype = ctypes.c_char_p
    record["native_source_identity"] = library.vibeqc_get_source_identity().decode()
    rows = []
    record["prepared_diagnostics"] = []
    cases = {
        "h2": [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))],
        "water": [
            ("O", (0, 0, 0)),
            ("H", (0, -1.43233673, 1.10715266)),
            ("H", (0, 1.43233673, 1.10715266)),
        ],
    }
    controls = {
        "energy_tolerance": 1e-10,
        "density_tolerance": 1e-8,
        "screening_tolerance": 1e-12,
        "max_iterations": 100,
    }

    def encode(result):
        items = result.items if hasattr(result, "items") else (result,)
        expected = "cuda" if args.device == "cuda" else "cpu_reference"
        if any(
            item.executed_backend != expected or not item.converged for item in items
        ):
            raise RuntimeError("benchmark backend or convergence differs from request")
        return [
            {
                "energy": item.energy,
                "forces": item.forces.tolist(),
                "iterations": item.iterations,
                "energy_change": item.energy_change,
                "density_rms": item.density_rms,
                "warm_start_used": getattr(item, "warm_start_used", False),
                "executed_backend": item.executed_backend,
            }
            for item in items
        ]

    def diagnostic(plan, metadata, count):
        # Outside the timing interval: retain the selected eigensolver and DF
        # allocation/tiling records without turning on scientific profiling.
        def optional(query):
            try:
                return {"entries": [d.to_dict() for d in query()], "reason": None}
            except NotImplementedError:
                # The legacy ABI represents an empty diagnostic set with
                # NOT_IMPLEMENTED (e.g. no CUDA eigensolver in a CPU bucket).
                return {"entries": [], "reason": "no records exposed by this route"}

        record["prepared_diagnostics"].append(
            {
                **metadata,
                "system_count": count,
                "eigensolver": optional(plan.last_eigensolver_diagnostics),
                "density_fitting": optional(
                    plan.last_density_fitting_metric_diagnostics
                ),
            }
        )

    def measure(name, function, metadata):
        if args.endpoint is not None and args.endpoint != name:
            return
        samples, outputs = [], []
        for i in range(args.samples):
            start = perf_counter()
            value = function(i)
            samples.append(perf_counter() - start)
            outputs.append(encode(value))
        rows.append(
            {**metadata, "endpoint": name, "seconds": samples, "outputs": outputs}
        )

    for case, atoms in cases.items():
        if args.case is not None and args.case != case:
            continue
        for spin in ("rhf", "uhf"):
            if args.spin is not None and args.spin != spin:
                continue
            charge, multiplicity = (0, 1) if spin == "rhf" else (1, 2)
            state = {"charge": charge, "multiplicity": multiplicity}
            for approximation in ("exact", "density_fitted"):
                if (
                    args.approximation is not None
                    and args.approximation != approximation
                ):
                    continue
                calc = Calculator(
                    device=args.device,
                    method=spin,
                    basis="sto-3g",
                    **controls,
                    density_fitting=args.device
                    if approximation == "density_fitted"
                    else "none",
                )
                meta = {
                    "case": case,
                    "atoms": atoms,
                    "basis": "sto-3g",
                    "spin": spin,
                    "approximation": approximation,
                    "device": args.device,
                    "state": state,
                    "controls": controls,
                }
                # Initial solve loads kernels/libraries and validates the case.
                calc.singlepoint(atoms, **state)
                measure(
                    "singlepoint",
                    lambda _, calc=calc, atoms=atoms, state=state: calc.singlepoint(
                        atoms, **state
                    ),
                    meta,
                )
                shifted = np.array([atom[1] for atom in atoms], dtype=float)
                shifted[-1, 2] += 1e-3
                original = np.array([atom[1] for atom in atoms], dtype=float)
                with calc.prepare_batch(
                    [atoms], charges=[charge], multiplicities=[multiplicity]
                ) as plan:
                    plan.execute(strict=True)
                    diagnostic(plan, meta, 1)
                    measure("warm_replay", lambda _: plan.execute(strict=True), meta)
                    measure(
                        "changed_geometry",
                        lambda i, shifted=shifted, original=original: plan.execute(
                            [shifted if i % 2 == 0 else original], strict=True
                        ),
                        meta,
                    )
                systems = [
                    [
                        (symbol, (x, y, z + (0.002 * i if a == len(atoms) - 1 else 0)))
                        for a, (symbol, (x, y, z)) in enumerate(atoms)
                    ]
                    for i in range(4)
                ]
                batch_state = {
                    "charges": [charge] * 4,
                    "multiplicities": [multiplicity] * 4,
                }
                measure(
                    "batch_four_cold",
                    lambda _, calc=calc, systems=systems, batch_state=batch_state: (
                        calc.batch_singlepoint(systems, strict=True, **batch_state)
                    ),
                    meta,
                )
                with calc.prepare_batch(systems, **batch_state) as plan:
                    plan.execute(strict=True)
                    diagnostic(plan, meta, 4)
                    measure(
                        "batch_four_warm", lambda _: plan.execute(strict=True), meta
                    )
    record["endpoints"] = rows
    probe = args.output.parent / f"dispatch-{args.label}"
    command(
        [
            record["build"]["CMAKE_CXX_COMPILER"],
            "-std=c++20",
            "-O3",
            "-DNDEBUG",
            "-I",
            str(args.root / "src"),
            "-I",
            str(args.root / "include"),
            str(
                Path(__file__).resolve().parents[1]
                / "benchmarks/fock_dispatch_probe.cpp"
            ),
            "-L",
            str(args.build),
            "-Wl,-rpath," + str(args.build),
            "-lvibeqc",
            "-o",
            str(probe),
        ]
    )
    record["fixed_density"] = json.loads(command([str(probe), args.device]))
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def compare(base, head):
    """Keep accuracy gates quantitative and overhead ratios reviewable."""
    import numpy as np

    rows = []
    for before, after in zip(base["endpoints"], head["endpoints"], strict=True):
        identity = {
            key: before[key]
            for key in ("case", "spin", "approximation", "device", "endpoint")
        }
        if any(after[key] != value for key, value in identity.items()):
            raise ValueError("endpoint identities differ")
        errors = {"energy": 0.0, "forces": 0.0}
        for a, b in zip(before["outputs"], after["outputs"], strict=True):
            for x, y in zip(a, b, strict=True):
                for field, previous in errors.items():
                    errors[field] = max(
                        previous,
                        float(
                            np.max(np.abs(np.asarray(x[field]) - np.asarray(y[field])))
                        ),
                    )
        ratio = statistics.median(after["seconds"]) / statistics.median(
            before["seconds"]
        )
        rows.append(
            {
                **identity,
                "median_ratio": ratio,
                "maximum_absolute_error": errors,
                "accuracy_passed": errors["energy"] <= 2e-9
                and errors["forces"] <= 2e-8,
                "baseline_seconds": before["seconds"],
                "head_seconds": after["seconds"],
            }
        )
    fixed = []
    for a, b in zip(
        base["fixed_density"]["rows"], head["fixed_density"]["rows"], strict=True
    ):
        if (a["nbf"], a["approximation"]) != (b["nbf"], b["approximation"]):
            raise ValueError("fixed-density identities differ")
        error = max(
            float(np.max(np.abs(np.asarray(a[k]) - np.asarray(b[k]))))
            for k in ("coulomb", "exchange")
        )
        fixed.append(
            {
                "nbf": a["nbf"],
                "approximation": a["approximation"],
                "maximum_absolute_error": error,
                "accuracy_passed": error <= 2e-11,
                "median_ratio": statistics.median(b["seconds"])
                / statistics.median(a["seconds"]),
                "baseline_seconds": a["seconds"],
                "head_seconds": b["seconds"],
            }
        )
    return {
        "endpoints": rows,
        "fixed_density": fixed,
        "accuracy_passed": all(r["accuracy_passed"] for r in rows + fixed),
        "performance_decision": "review_required",
        "limits": {
            "energy_hartree": 2e-9,
            "force_hartree_per_bohr": 2e-8,
            "fixed_density_matrix": 2e-11,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--head", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--build-relative", default=".artifacts/overhead-cuda-build")
    parser.add_argument(
        "--output", type=Path, default=Path(".artifacts/fock-strategy-overhead")
    )
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--case", choices=("h2", "water"))
    parser.add_argument("--spin", choices=("rhf", "uhf"))
    parser.add_argument("--approximation", choices=("exact", "density_fitted"))
    parser.add_argument(
        "--endpoint",
        choices=(
            "singlepoint",
            "warm_replay",
            "changed_geometry",
            "batch_four_cold",
            "batch_four_warm",
        ),
        help="repeat a specific endpoint while retaining its normal warmup/setup",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--build", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--device", choices=("cpu", "cuda"), help=argparse.SUPPRESS)
    parser.add_argument("--label", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("run inside a finite Slurm GPU allocation")
    if args.samples < 3:
        parser.error("at least three samples are required")
    if args.worker:
        worker(args)
        return
    if args.baseline is None:
        parser.error("--baseline is required")
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for device in ("cpu", "cuda"):
        results = {}
        for label, root in (
            ("baseline", args.baseline.resolve()),
            ("head", args.head.resolve()),
        ):
            build = root / args.build_relative
            output = (args.output / f"{label}-{device}.json").resolve()
            env = os.environ.copy()
            env.update(
                VIBEQC_LIBRARY=str(build / "libvibeqc.so"),
                VIBEQC_PROFILE="off",
                OMP_NUM_THREADS="1",
                OPENBLAS_NUM_THREADS="1",
            )
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--root",
                    str(root),
                    "--build",
                    str(build),
                    "--device",
                    device,
                    "--label",
                    f"{label}-{device}",
                    "--samples",
                    str(args.samples),
                    "--output",
                    str(output),
                    *[
                        item
                        for key in ("case", "spin", "approximation", "endpoint")
                        if (value := getattr(args, key)) is not None
                        for item in ("--" + key, value)
                    ],
                ],
                env=env,
                check=True,
            )
            results[label] = json.loads(output.read_text())
            print(f"completed {label} {device}", flush=True)
        summaries[device] = compare(results["baseline"], results["head"])
    (args.output / "comparison.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=False) + "\n"
    )
    if not all(row["accuracy_passed"] for row in summaries.values()):
        raise SystemExit("same-approximation accuracy gate failed")


if __name__ == "__main__":
    main()
