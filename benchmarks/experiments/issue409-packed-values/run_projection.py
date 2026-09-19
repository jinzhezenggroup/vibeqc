"""Plan, run, and independently check the bounded #409 projection experiment.

Use prepare before entering Slurm. A successful fixed-input experiment is only
an implementation choice: the complete molecular Phase A remains mandatory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

_BENCHMARKS_DIR = next(
    parent for parent in Path(__file__).resolve().parents if parent.name == "benchmarks"
)
sys.path.insert(0, str(_BENCHMARKS_DIR))
import numpy as np
from _retention import raw_output_path
from vibeqc.resources import (
    ResourceBudget,
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourceRequest,
    plan_resources,
)

SOURCE = Path(__file__).resolve().parent
MODES = ("full", "unpack32", "unpack128", "unpack256", "direct32x16", "direct64x16")


def digest(path):
    """Hash large binary inputs with bounded host staging."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def resource_plan(fixture, args):
    """Charge every simultaneous trial allocation before launching CUDA."""
    n, a, r = (fixture[key] for key in ("n", "a", "r"))
    m, b, u = n * n * 8, n * n * a * 8, n * r * a * 8
    estimates = tuple(
        ResourceEstimate(name, max(8, size), "device:0", 0, 1)
        for name, size in (
            ("dense control B", b),
            ("lower packed B", n * (n + 1) // 2 * a * 8),
            ("occupied C", n * r * 8),
            ("candidate U", u),
            ("reference U", u),
            ("candidate K", m),
            ("reference K", m),
            ("largest bounded unpack buffer", m * min(a, 256)),
            ("error reductions", 512 * 8),
        )
    ) + (
        ResourceEstimate(
            "cuBLAS workspace allowance",
            64 << 20,
            "device:0",
            0,
            1,
            kind="library",
            accounting="runtime_allowance",
        ),
        ResourceEstimate(
            "bounded host upload and diagnostics", 64 << 20, "pageable", 0, 1
        ),
    )
    identity = ResourceIdentity(
        "exact packed DF projection",
        "bounded FP64 CUDA trial",
        "cuda",
        "fp64",
        json.dumps(
            {key: fixture[key] for key in ("n", "a", "r", "weight", "sha256")},
            sort_keys=True,
        ),
        ("occupied U", "exchange K"),
        "unit-weight lower triangle; full-Q U; unchanged single SYRK",
    )
    request = ResourceRequest(
        fixture["name"],
        identity,
        (ResourceCandidate("common-trial-arena", "resident", estimates),),
        scope_exclusions=(
            "CUDA context and loaded modules",
            "host runtime and file cache",
        ),
    )
    return plan_resources(
        [request], ResourceBudget(device_bytes=args.budget, host_bytes=256 << 20)
    ).require_feasible()


def work(fixture, mode):
    """Logical reads differ from DRAM traffic; opaque BLAS loads are not guessed."""
    n, a, r = (fixture[key] for key in ("n", "a", "r"))
    b, u, packed = n * n * a * 8, n * r * a * 8, n * (n + 1) // 2 * a * 8
    result = {
        "factor_storage_bytes": b if mode == "full" else packed,
        "coefficient_bytes": n * r * 8,
        "U_bytes": u,
        "K_bytes": n * n * 8,
        "projection_flops": 2 * n * n * a * r,
        "gram_flops": a * r * n * (n + 1),
        "unpack_global_write_bytes": 0,
        "unpack_logical_packed_read_bytes": 0,
        "expanded_scratch_bytes": 0,
        "projection_library_calls": 0 if mode.startswith("direct") or not r else 1,
        "projection_cuda_launches": 1 if mode.startswith("direct") and r else 0,
        "projection_synchronizations": 0,
        "gram_library_calls": 1 if r else 0,
        "timed_terminal_event_waits": 1,
        "interpretation": "logical input traffic; no measured hardware DRAM count",
    }
    if mode.startswith("unpack") and r:
        tile = min(a, int(mode.removeprefix("unpack")))
        panels = (a + tile - 1) // tile
        result.update(
            unpack_global_write_bytes=b,
            unpack_logical_packed_read_bytes=b,
            expanded_scratch_bytes=n * n * tile * 8,
            projection_library_calls=panels,
            projection_cuda_launches=panels,
            projection_dense_input_compulsory_bytes=b,
        )
    elif mode.startswith("direct") and r:
        tile = 32 if mode == "direct32x16" else 64
        result.update(
            projection_logical_packed_read_bytes=b * ((r + 15) // 16),
            projection_logical_coefficient_read_bytes=n
            * n
            * r
            * ((a + tile - 1) // tile)
            * 8,
            shared_bytes_per_block=(32 * tile + 33 * 16) * 8,
            # Include predicated output lanes' zero-padded arithmetic too.
            projection_executed_flops=2
            * n
            * ((a + tile - 1) // tile)
            * tile
            * ((r + 15) // 16)
            * 16
            * ((n + 31) // 32)
            * 32,
        )
    elif r:
        result["projection_dense_input_compulsory_bytes"] = b
    result["standalone_candidate_device_bytes_without_library"] = sum(
        result[key]
        for key in (
            "factor_storage_bytes",
            "coefficient_bytes",
            "U_bytes",
            "K_bytes",
            "expanded_scratch_bytes",
        )
    )
    return result


def prepare(args):
    """Freeze hashes, resource reservations and diagnostic input derivations."""
    if args.output.exists():
        raise RuntimeError("refusing to overwrite prepared experiment")
    args.output.mkdir(parents=True)
    fixtures = []
    for size in (384, 768):
        directory = args.captures / str(size)
        bpath = directory / "b.bin"
        full = np.memmap(bpath, dtype="<f8", mode="r", shape=(size, size, size))
        symmetry = 0.0
        for mu in range(size):
            row = np.asarray(full[mu])
            if not np.all(np.isfinite(row)):
                raise RuntimeError("nonfinite input B")
            symmetry = max(symmetry, float(np.max(np.abs(row - full[:, mu, :]))))
        if symmetry > 1e-12:
            raise RuntimeError("captured physical B is not symmetry compressible")
        bhash = digest(bpath)
        for index in range(4):
            prefix = directory / str(index)
            meta = json.loads(prefix.with_suffix(".json").read_text())
            if meta["n"] != size or meta["auxiliary"] != size:
                raise RuntimeError("capture dimensions differ")
            fixtures.append(
                {
                    "name": f"real-{size}-{index}",
                    "n": size,
                    "a": size,
                    "r": meta["rank"],
                    "weight": meta["weight"],
                    "b": str(bpath.resolve()),
                    "c": str(prefix.resolve()) + "-c.bin",
                    "captured_u": str(prefix.resolve()) + "-u.bin",
                    "B_symmetry_max_abs": symmetry,
                    "origin": "eager molecular build; captured timing excluded; see capture trace",
                    "sha256": {"b": bhash},
                }
            )
    for name, n, a, r in (
        ("prefix-192", 192, 192, 40),
        ("unequal-tail", 97, 193, 37),
        ("independent-small", 13, 7, 9),
        ("rank-zero", 13, 7, 0),
    ):
        source = fixtures[0]
        original = np.memmap(source["b"], dtype="<f8", mode="r", shape=(384, 384, 384))
        b = np.array(original[:n, :n, :a], order="C")
        coefficients = np.fromfile(source["c"], dtype="<f8").reshape(384, source["r"])
        c = coefficients[:n, :r].copy()
        bpath, cpath = args.output / f"{name}-b.bin", args.output / f"{name}-c.bin"
        b.tofile(bpath)
        c.tofile(cpath)
        fixtures.append(
            {
                "name": name,
                "n": n,
                "a": a,
                "r": r,
                "weight": source["weight"],
                "b": str(bpath.resolve()),
                "c": str(cpath.resolve()),
                "captured_u": "-",
                "sha256": {"b": digest(bpath)},
                "B_symmetry_max_abs": float(np.max(np.abs(b - b.transpose(1, 0, 2)))),
                "origin": "leading subproblem of real-384-0; synthetic diagnostic, not a molecular endpoint",
            }
        )
    for fixture in fixtures:
        fixture["sha256"]["c"] = digest(fixture["c"])
        if fixture["captured_u"] != "-":
            fixture["sha256"]["captured_u"] = digest(fixture["captured_u"])
        plan = resource_plan(fixture, args)
        fixture["resource_plan_identity"] = plan.identity
        dump(args.output / (fixture["name"] + "-plan.json"), plan.to_dict())
        fixture["work"] = {mode: work(fixture, mode) for mode in MODES}
    dump(
        args.output / "manifest.json",
        {
            "fixtures": fixtures,
            "protocol": json.loads((SOURCE / "protocol.json").read_text()),
            "protocol_sha256": digest(SOURCE / "protocol.json"),
            "executable": str(args.executable.resolve()),
            "executable_sha256": digest(args.executable),
            "source_sha256": digest(SOURCE / "projection.cu"),
            "runner_sha256": digest(__file__),
        },
    )


def run(args):
    """Persist every sample before numerical gates; never weaken a failed gate."""
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("GPU trial requires finite Slurm allocation")
    manifest = json.loads((args.output / "manifest.json").read_text())
    if digest(__file__) != manifest["runner_sha256"]:
        raise RuntimeError("runner changed after preflight")
    if digest(manifest["executable"]) != manifest["executable_sha256"]:
        raise RuntimeError("executable changed after preflight")
    fixtures = manifest["fixtures"]
    # Complete small correctness controls first, then the molecular cases.
    fixtures = fixtures[8:] + fixtures[:8]
    if args.fixture:
        fixtures = [row for row in fixtures if row["name"] == args.fixture]
        if not fixtures:
            raise ValueError("unknown fixture")
    all_results = []
    for fixture in fixtures:
        name = fixture["name"]
        for key, expected in fixture["sha256"].items():
            if digest(fixture[key]) != expected:
                raise RuntimeError(f"{name}: input {key} changed after preflight")
        log = args.output / (name + ".jsonl")
        if log.exists():
            raise RuntimeError("refusing to overwrite samples")
        command = [
            manifest["executable"],
            *(str(fixture[key]) for key in ("n", "a", "r")),
            str(manifest["protocol"]["repeats"]),
            str(fixture["weight"]),
            fixture["b"],
            fixture["c"],
            fixture["captured_u"],
            str(args.output / name),
        ]
        with log.open("w") as out, log.with_suffix(".stderr").open("w") as err:
            process = subprocess.run(command, stdout=out, stderr=err, check=False)
        dump(
            args.output / (name + "-execution.json"),
            {
                "command": command,
                "returncode": process.returncode,
                "slurm_job_id": os.environ["SLURM_JOB_ID"],
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        )
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        if process.returncode or len(rows) != 6 * manifest["protocol"]["repeats"]:
            raise RuntimeError(f"{name}: trial failed; retained {log}")
        b = np.memmap(
            fixture["b"],
            dtype="<f8",
            mode="r",
            shape=(fixture["n"], fixture["n"], fixture["a"]),
        )
        c = np.fromfile(fixture["c"], dtype="<f8").reshape(fixture["n"], fixture["r"])
        maximum = 0.0
        for line in (args.output / (name + "-samples.jsonl")).read_text().splitlines():
            sample = json.loads(line)
            oracle = np.sum(
                b[sample["mu"], :, sample["q"]].astype(np.longdouble)
                * c[:, sample["i"]].astype(np.longdouble),
                dtype=np.longdouble,
            )
            maximum = max(maximum, float(abs(oracle - np.longdouble(sample["value"]))))
        result = {
            "name": name,
            "independent_projection_sample_max_abs": maximum,
            "modes": {},
        }
        for mode in MODES:
            selected = [row for row in rows if row["mode"] == mode]
            result["modes"][mode] = {
                "median_wall_seconds": statistics.median(
                    row["wall_seconds"] for row in selected
                ),
                "median_projection_seconds": statistics.median(
                    row["projection_seconds"] for row in selected
                ),
                "median_gram_mirror_seconds": statistics.median(
                    row["gram_mirror_seconds"] for row in selected
                ),
                "max_U_abs": max(row["U_max_abs"] for row in selected),
                "max_K_abs": max(row["K_max_abs"] for row in selected),
                "passed": all(row["passed"] for row in selected),
            }
        dump(args.output / (name + "-summary.json"), result)
        if (
            maximum > 1e-10
            or not np.isfinite(maximum)
            or not all(row["passed"] for row in rows)
        ):
            raise RuntimeError(f"{name}: numerical qualification failed")
        all_results.append(result)
        print(json.dumps(result), flush=True)
    dump(
        args.output
        / ("summary.json" if not args.fixture else f"summary-{args.fixture}.json"),
        all_results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--captures", type=Path)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument("--budget", type=int, default=12 << 30)
    parser.add_argument("--fixture")
    arguments = parser.parse_args()
    (prepare if arguments.action == "prepare" else run)(arguments)
