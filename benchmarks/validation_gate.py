"""Run shared CPU/HF evidence or wrap existing compilation/GPU tier commands."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from benchmarks._support import environment_metadata
from tools.vibeqc_validation.fixtures import (
    calculator_inputs,
    load_fixtures,
    mathematical_hash,
)
from tools.vibeqc_validation.performance import assess_comparison, measure_interleaved
from tools.vibeqc_validation.schema import (
    TIERS,
    attach_artifact,
    block_error,
    canonical_hash,
    file_hash,
    finite_difference,
    new_evidence,
    outcome,
    write_evidence,
)


def _cuda():
    """Probe only inside Slurm and synchronize the runtime's assigned device.

    No environment rewriting is permitted: the scheduler's local ordinal zero
    is the device used by Calculator. CuPy is not required for this HF example.
    """
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU tier requires a Slurm allocation on this machine")
    library = os.environ.get("VIBEQC_LIBRARY")
    if not library:
        raise RuntimeError("set VIBEQC_LIBRARY to the CUDA build being validated")
    runtime = ctypes.CDLL(library)
    count = ctypes.c_int()
    if runtime.cudaGetDeviceCount(ctypes.byref(count)) != 0 or count.value == 0:
        raise RuntimeError("no scheduler-visible CUDA device")
    if runtime.cudaSetDevice(0) != 0:
        raise RuntimeError("cannot select scheduler-visible CUDA ordinal zero")
    driver = ctypes.CDLL(ctypes.util.find_library("cuda") or "libcuda.so.1")
    name = ctypes.create_string_buffer(256)
    if driver.cuInit(0) != 0 or driver.cuDeviceGetName(name, len(name), 0) != 0:
        raise RuntimeError("cannot identify CUDA device")

    def synchronize():
        if runtime.cudaDeviceSynchronize() != 0:
            raise RuntimeError("CUDA synchronization failed")

    return {
        "name": name.value.decode(),
        "backend": "cuda",
        "ordinal": 0,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
    }, synchronize


def _provenance(record):
    metadata = environment_metadata(distributions={"numpy": ("numpy",)})
    record["revision"] = metadata["git"]["commit"]
    record["environment"] = metadata
    record["toolchain"] = metadata["toolchain"]
    record["hashes"]["source"] = canonical_hash(
        {
            str(p.relative_to(ROOT)): file_hash(p)
            for base in ("src", "include", "python", "tools", "benchmarks")
            for p in sorted((ROOT / base).rglob("*"))
            if p.is_file() and p.suffix in {".py", ".cpp", ".cu", ".hpp", ".h"}
        }
    )
    library = os.environ.get("VIBEQC_LIBRARY")
    record["library_sha256"] = (
        file_hash(library) if library and Path(library).is_file() else None
    )
    record["build"] = None
    if library:
        cache = Path(library).resolve().parent / "CMakeCache.txt"
        if cache.is_file():
            settings = {}
            for line in cache.read_text().splitlines():
                if "=" not in line or line.startswith(("#", "//")):
                    continue
                key, value = line.split("=", 1)
                key = key.split(":", 1)[0]
                if key.startswith(
                    ("VIBEQC_", "CMAKE_CUDA_", "CMAKE_CXX_", "CMAKE_BUILD_TYPE")
                ):
                    settings[key] = value
            record["build"] = {"cache_sha256": file_hash(cache), "settings": settings}


def hf_evidence(*, case="h2", device="cpu", repeats=5, check_fd=False):
    """Demonstrate the full envelope with existing native HF energy/force calls.

    A/B uses two identical native configurations as a protocol control, so no
    speedup is expected. The public executor cannot expose energy-only work,
    complete solver history, or allocator peaks; these remain explicit gaps
    and prohibit performance/production promotion.
    """
    from vibeqc import Calculator

    reference = next(
        r
        for r in load_fixtures()
        if r["inputs"]["name"] == case and r["inputs"]["kind"] == "molecule"
    )
    inputs = reference["inputs"]
    record = new_evidence(
        tier="cpu" if device == "cpu" else "endpoint",
        subject=f"hf/{case}",
        inputs_hash=reference["inputs_hash"],
    )
    _provenance(record)
    record["settings"] = {
        "device": device,
        "repeats": repeats,
        "comparison_kind": "solver",
        "method": inputs["method"],
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-11,
        "screening_tolerance": 1e-14,
        "density_fitting": "none",
        "baseline": "native HF",
        "candidate": "identical native HF protocol control",
    }
    record["reference"] = {
        "record_hash": canonical_hash(reference),
        "data_hash": reference["data_hash"],
        "provenance": reference["provenance"],
        "angular_momenta_loaded": reference["angular_momenta_loaded"],
    }
    record["hashes"]["equation"] = canonical_hash(
        {"method": inputs["method"], "conventions": inputs["conventions"]}
    )
    record["hash_reasons"] = {
        "ir": "handwritten HF executor has no correlated-method IR",
        "schedule": "public HF API does not expose kernel schedule identity",
    }
    record["stages"]["representation"] = outcome("pass")
    record["stages"]["source"] = outcome(
        "pass", source="existing native HF implementation"
    )
    if device == "cuda":
        try:
            record["device"], synchronize = _cuda()
        except (OSError, AttributeError, RuntimeError) as error:
            record["hardware"] = outcome("not-run", str(error))
            return record
    else:
        record["device"] = record["environment"]["host"]
        synchronize = lambda: None  # CPU calls are blocking.
    record["hardware"] = outcome("pass")
    atoms = list(zip(inputs["atomic_numbers"], inputs["coordinates"], strict=True))
    options = calculator_inputs(inputs)
    calculator = Calculator(device=device, **options)
    expected_backend = "cuda" if device == "cuda" else "cpu_reference"

    def diagnostics(item):
        if not item.converged or item.executed_backend != expected_backend:
            raise RuntimeError("HF did not converge on the requested backend")
        return {
            "energy": item.energy,
            "forces": item.forces.tolist(),
            "iterations": item.iterations,
            "density_rms": item.density_rms,
            "energy_change": item.energy_change,
            "backend_selected": item.executed_backend,
        }

    def single(coordinates):
        return calculator.singlepoint(
            list(zip(inputs["atomic_numbers"], coordinates, strict=True)),
            charge=inputs["charge"],
            multiplicity=inputs["multiplicity"],
        )

    try:

        def cold(_selection):
            # Includes native context/plan creation and destruction on every call.
            return diagnostics(single(inputs["coordinates"]))

        record["timings"] += measure_interleaved(
            cold,
            synchronize,
            workload="cold-start",
            inputs_hash=record["inputs_hash"],
            repeats=repeats,
        )
        first = record["timings"][0]["diagnostics"]
        record["backend_selected"] = first["backend_selected"]
        with calculator.prepare_batch(
            [atoms],
            charges=[inputs["charge"]],
            multiplicities=[inputs["multiplicity"]],
            warm_start=True,
        ) as batch:
            batch.execute(strict=True)
            batch.set_warm_start_updates(False)

            def replay(_selection):
                return diagnostics(
                    batch.execute([inputs["coordinates"]], strict=True).items[0]
                )

            for workload in ("unchanged-geometry", "energy-plus-force"):
                record["timings"] += measure_interleaved(
                    replay,
                    synchronize,
                    workload=workload,
                    inputs_hash=record["inputs_hash"],
                    repeats=repeats,
                )
            changed = np.asarray(inputs["coordinates"]).copy()
            changed[-1, 0] += 0.01
            changed_hash = mathematical_hash(
                {**inputs, "coordinates": changed.tolist()}
            )

            def reset_geometry(_selection):
                # Reset outside the measured region so every sample measures
                # one changed-geometry execution from the frozen post-cold dm0.
                batch.execute([inputs["coordinates"]], strict=True)

            def changed_geometry(_selection):
                return diagnostics(batch.execute([changed], strict=True).items[0])

            record["timings"] += measure_interleaved(
                changed_geometry,
                synchronize,
                workload="changed-geometry",
                inputs_hash=changed_hash,
                repeats=repeats,
                prepare=reset_geometry,
            )
        record["workloads"] = {
            "energy-only": outcome(
                "not-run",
                "native HF always computes forces; omitting the output buffer only omits copying",
            )
        }
        record["block_errors"] = {
            "energy": block_error(
                first["energy"], reference["data"]["energy"], atol=1e-10, rtol=0
            ),
            "forces": block_error(
                first["forces"], reference["data"]["forces"], atol=1e-7, rtol=0
            ),
        }
        # Validate every replay, not just an accidentally favorable cold sample.
        changed_reference = diagnostics(single(changed))
        record["changed_geometry_reference"] = {
            "oracle": "native one-shot replay consistency; not independent libcint",
            **changed_reference,
        }
        for index, sample in enumerate(record["timings"]):
            target = (
                changed_reference
                if sample["workload"] == "changed-geometry"
                else reference["data"]
            )
            for key, tolerance in (("energy", 1e-10), ("forces", 1e-7)):
                record["block_errors"][f"sample/{index}/{key}"] = block_error(
                    sample["diagnostics"][key], target[key], atol=tolerance, rtol=0
                )
        record["residuals"] = {
            "density_rms": first["density_rms"],
            "energy_change": first["energy_change"],
            "independent_equation_residual": outcome(
                "not-run", "native HF does not export density/orbitals"
            ),
        }
        passed = all(e["passed"] for e in record["block_errors"].values())
        record["stages"]["numerical"] = outcome(
            "pass" if passed else "fail", None if passed else "HF reference gate failed"
        )
        record["stages"]["endpoint"] = record["stages"]["numerical"].copy()
        record["runtime_comparison"] = assess_comparison(record["timings"])
        record["performance"] = outcome(
            "not-run",
            "protocol control; native allocator/compile cost and full solver history unavailable",
        )
        if check_fd:

            def energy(xyz, policy):
                # Same calculator fixes method, screening, and convergence at
                # all steps; the policy hash documents those settings.
                return single(xyz).energy

            record["finite_difference"] = finite_difference(
                energy,
                inputs["coordinates"],
                reference["data"]["gradient"],
                settings=record["settings"],
            )
    except (RuntimeError, ValueError) as error:
        record["stages"]["numerical"] = outcome("fail", str(error))
    return record


def command_evidence(args):
    """Wrap an existing tier command, retaining original evidence attachments.

    The caller selects the tier; CUDA compilation never probes a GPU. Missing
    hardware is not-run. A GPU command must already be in a Slurm allocation.
    Wrapped commands must fail on numerical failure; their zero exit establishes
    only that command's tier, with no inferred performance/production pass.
    """
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    record = new_evidence(
        tier=args.tier,
        subject=args.subject,
        inputs_hash=canonical_hash({"command": command}),
    )
    _provenance(record)
    record["settings"] = {
        "command": command,
        "timeout_seconds": args.timeout,
        "device": "cuda" if args.tier in {"gpu-numerical", "endpoint"} else "cpu",
    }
    for path in args.attach:
        attach_artifact(record, path, kind="existing-runner")
    if args.unavailable:
        record["hardware"] = outcome("not-run", args.unavailable)
        return record
    if not command:
        raise ValueError(
            "provide a command after -- or an explicit --unavailable reason"
        )
    if args.tier in {"gpu-numerical", "endpoint"}:
        try:
            record["device"], _ = _cuda()
        except (OSError, AttributeError, RuntimeError) as error:
            record["hardware"] = outcome("not-run", str(error))
            return record
    record["hardware"] = (
        outcome("pass")
        if args.tier != "cuda-compile"
        else outcome("not-run", "compilation requires no GPU")
    )
    stage = {
        "cpu": "numerical",
        "cuda-compile": "compilation",
        "gpu-numerical": "numerical",
        "endpoint": "endpoint",
    }[args.tier]
    started = time.perf_counter()
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            # Kill compiler/test descendants too: they can retain an allocation
            # or pipe after the shell itself has exited.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            raise
        result = subprocess.CompletedProcess(
            command, process.returncode, stdout, stderr
        )
        record["command_result"] = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "seconds": time.perf_counter() - started,
        }
        # Without an executor report, a successful process can still have
        # skipped numerical tests. Preserve command success separately.
        record["command_status"] = outcome(
            "pass" if result.returncode == 0 else "fail",
            None if result.returncode == 0 else "command returned nonzero",
        )
        if stage == "compilation":
            record["stages"][stage] = record["command_status"].copy()
            record["compilation"] = {
                "seconds": record["command_result"]["seconds"],
                "reason": None,
            }
        elif result.returncode != 0:
            record["stages"][stage] = record["command_status"].copy()
        else:
            record["stages"][stage] = outcome(
                "not-run",
                "process succeeded; inspect attached executor evidence for actual numerical coverage",
            )
    except FileNotFoundError as error:
        record["stages"][stage] = outcome("not-run", str(error))
    except subprocess.TimeoutExpired:
        record["stages"][stage] = outcome(
            "fail", "tier command exceeded finite timeout"
        )
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    hf = commands.add_parser(
        "hf", help="existing HF protocol control against pinned references"
    )
    hf.add_argument(
        "--case", choices=("h2", "he", "h2o", "nh3", "ch4", "hf-plus-uhf"), default="h2"
    )
    hf.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    hf.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="at least five interleaved A/B samples per workload",
    )
    hf.add_argument(
        "--finite-difference",
        action="store_true",
        help="report gradient errors at three fixed step sizes",
    )
    run = commands.add_parser(
        "run",
        help="wrap existing ctest/pytest, CUDA compile, autotune, or endpoint commands",
    )
    run.add_argument("--tier", choices=TIERS, required=True)
    run.add_argument("--subject", required=True, help="method/shell/case identity")
    run.add_argument(
        "--timeout", type=float, default=600, help="finite command timeout in seconds"
    )
    run.add_argument(
        "--unavailable", help="write not-run with this missing-tool/hardware reason"
    )
    run.add_argument(
        "--attach",
        type=Path,
        action="append",
        default=[],
        help="retain an existing versioned autotune/#135/endpoint artifact by hash",
    )
    run.add_argument("command", nargs=argparse.REMAINDER)
    for child in (hf, run):
        child.add_argument(
            "--output",
            type=Path,
            required=True,
            help="validated JSON evidence destination",
        )
    args = parser.parse_args()
    if args.mode == "hf" and args.repeats < 5:
        parser.error("--repeats must be at least five")
    if args.mode == "run" and (not np.isfinite(args.timeout) or args.timeout <= 0):
        parser.error("--timeout must be positive and finite")
    record = (
        hf_evidence(
            case=args.case,
            device=args.device,
            repeats=args.repeats,
            check_fd=args.finite_difference,
        )
        if args.mode == "hf"
        else command_evidence(args)
    )
    write_evidence(args.output, record)
    print(f"JSON evidence: {args.output}")
    if any(s["status"] == "fail" for s in record["stages"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
