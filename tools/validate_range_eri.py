"""Reproduce compact native raw/weighted range-ERI evidence using shared gates.

CUDA execution requires a finite Slurm allocation. This driver keeps the
independent Libcint fixtures and numerical publication machinery in tools;
the installed compiler/runtime has no reference-engine dependency.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "tools")]

import numpy as np
import pyscf
from validate_weighted_eri import make_fixture
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.common.evidence import (
    block_error,
    canonical_hash,
    file_hash,
    new_evidence,
    outcome,
    write_evidence,
)
from vibeqc_compiler.common.paths import source_hashes
from vibeqc_compiler.integral.blocks import WeightTile
from vibeqc_compiler.integral.range_separation import CoulombKernel
from vibeqc_compiler.integral.weighted_eri_execute import (
    PreparedWeightedEri,
    compile_weighted_eri,
)
from vibeqc_compiler.integral.weighted_eri_inputs import prepare_weighted_eri_stream
from vibeqc_validation.publication import publish


def command(*argv):
    """Read finite source/toolchain probes without a shell."""
    return subprocess.check_output(argv, cwd=ROOT, text=True, timeout=30).strip()


def cpu_model(cpuinfo_path=Path("/proc/cpuinfo")):
    """Keep completed evidence publishable when Linux model metadata is absent.

    Missing/unreadable procfs and CPUs with different field names do not
    invalidate the completed native numerical run. The fallback explicitly
    retains an unknown model when the platform cannot identify it either.
    """
    fallback = platform.processor() or "unknown CPU model"
    try:
        lines = cpuinfo_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return fallback
    for line in lines:
        key, separator, value = line.partition(":")
        if separator and key.strip().casefold() == "model name" and value.strip():
            return value.strip()
    return fallback


def run(args):
    """Measure explicit operator endpoints without changing a production profile."""
    if args.backend == "cuda" and not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("range CUDA evidence must run inside Slurm")
    if args.samples < 1:
        raise ValueError("at least one timing sample is required")
    dirty = bool(command("git", "status", "--porcelain"))
    if args.publish is not None and dirty:
        raise ValueError("commit the measured source before publishing range evidence")
    revision = command("git", "rev-parse", "HEAD")
    compiler = (
        CppCompilerAdapter(args.compiler or Path("c++"))
        if args.backend == "cpu"
        else CudaCompilerAdapter(
            args.compiler or Path("nvcc"), cuda_target_info(args.architecture)
        )
    )
    args.output.mkdir(parents=True, exist_ok=False)
    rows, artifacts, resources, errors = [], {}, {}, {}
    full_references = {}
    direction = np.arange(1, 13, dtype=float).reshape(4, 3)
    direction /= np.linalg.norm(direction)
    cases = [
        (name, variant)
        for name in ("psss", "dpsp", "fsss")
        for variant in ("cartesian", "spherical")
    ]
    cases += [("dpsp", "exchange"), ("psss", "coincident")]
    for family in ("long_range", "short_range"):
        radial = CoulombKernel(family, args.omega)
        for name, variant in cases:
            fixture = make_fixture(name, variant, coulomb_kernel=radial)
            indices = (0, fixture["weights"].size // 2, fixture["weights"].size - 1)
            raw_reference = np.array(
                [
                    make_fixture(
                        name, variant, coulomb_kernel=radial, unit_component=i
                    )["reference"]
                    for i in indices
                ]
            )
            artifact = compile_weighted_eri(
                fixture["request"].integral, compiler, args.cache
            )
            key = artifact.program_identity
            # Retain portable scientific identities and measured compiler data.
            # Cache-local filenames remain runtime inputs, not publication paths.
            metadata = artifact.native.metadata
            native_identity = metadata["identity"]
            artifacts[key] = {
                "native_key": metadata["key"],
                "binary_sha256": metadata["binary_sha256"],
                "source_sha256": native_identity["source"],
                "target": native_identity["target"],
                "compile_seconds": metadata["compile_seconds"],
                "compiler_resources": metadata["resources"],
                "operator": radial.to_payload(),
                "angular": list(artifact.integral.signature.angular),
                "component_indices": list(artifact.component_indices),
            }

            def prepare(centers, fixture=fixture):
                return prepare_weighted_eri_stream(
                    fixture["request"],
                    fixture["primitives"],
                    centers,
                    lambda descriptor, _: WeightTile(
                        descriptor.layout, fixture["weights"].ravel()
                    ),
                    projections=fixture["projections"],
                )

            stream = prepare(fixture["centers"])
            started = time.perf_counter()
            plan = PreparedWeightedEri(artifact, record_capacity=7, tile_capacity=3)
            preparation_seconds = time.perf_counter() - started
            with plan:
                cold = plan.contract(stream, profile=True)
                weighted_samples, raw_samples = [], []
                for _ in range(args.samples):
                    weighted = plan.contract(stream, profile=True)
                    raw = plan.raw(
                        fixture["primitives"],
                        fixture["centers"],
                        indices,
                        projections=fixture["projections"],
                        profile=True,
                    )
                    weighted_samples.append(
                        {
                            "wall_seconds": weighted.diagnostics["wall_seconds"],
                            "device": weighted.diagnostics["device_timing"],
                        }
                    )
                    raw_samples.append(
                        {
                            "wall_seconds": raw.diagnostics["wall_seconds"],
                            "device": raw.diagnostics["device_timing"],
                        }
                    )
                    np.testing.assert_allclose(
                        weighted.values[0], fixture["reference"], atol=1e-11, rtol=1e-10
                    )
                    np.testing.assert_allclose(
                        raw.values, raw_reference, atol=1e-11, rtol=1e-10
                    )
                curve = []
                analytic = float(
                    np.sum(weighted.values[0, 1:].reshape(4, 3) * direction)
                )
                for step in (1e-3, 3e-4, 1e-4):
                    plus = plan.contract(
                        prepare(fixture["centers"] + step * direction)
                    ).values[0, 0]
                    minus = plan.contract(
                        prepare(fixture["centers"] - step * direction)
                    ).values[0, 0]
                    derivative = float((plus - minus) / (2 * step))
                    curve.append(
                        {
                            "step_bohr": step,
                            "derivative": derivative,
                            "analytic": analytic,
                        }
                    )
                for result in (weighted, raw):
                    resource = result.diagnostics["resources"]
                    resources[resource["identity"]] = resource
            label = f"{family}/{name}/{variant}"
            errors[label + "/weighted"] = block_error(
                weighted.values[0], fixture["reference"], atol=1e-11, rtol=1e-10
            )
            errors[label + "/raw"] = block_error(
                raw.values, raw_reference, atol=1e-11, rtol=1e-10
            )
            errors[label + "/finite_difference"] = block_error(
                [r["derivative"] for r in curve],
                [analytic] * len(curve),
                atol=1e-6,
                rtol=0,
            )
            errors[label + "/translation"] = block_error(
                raw.values[:, 1:].reshape(3, 4, 3).sum(axis=1),
                np.zeros((3, 3)),
                atol=1e-11,
                rtol=0,
            )
            rows.append(
                {
                    "case": label,
                    "input_hash": fixture["input_hash"],
                    "program_identity": key,
                    "raw_indices": indices,
                    "raw": raw.values.tolist(),
                    "raw_reference": raw_reference.tolist(),
                    "weighted": weighted.values[0].tolist(),
                    "weighted_reference": fixture["reference"].tolist(),
                    "finite_difference": curve,
                    "preparation_seconds": preparation_seconds,
                    "cold_weighted_seconds": cold.diagnostics["wall_seconds"],
                    "weighted_samples": weighted_samples,
                    "raw_samples": raw_samples,
                    "weighted_records": weighted.diagnostics["records"],
                    "raw_records": raw.diagnostics["records"],
                    "weighted_resource": weighted.diagnostics["resources"]["identity"],
                    "raw_resource": raw.diagnostics["resources"]["identity"],
                }
            )
            full_references[name, variant] = make_fixture(name, variant)["reference"]
            print(label, "passed", flush=True)
    for index, (name, variant) in enumerate(cases):
        errors[f"LR+SR/{name}/{variant}"] = block_error(
            np.array(rows[index]["weighted"]) + rows[index + len(cases)]["weighted"],
            full_references[name, variant],
            atol=1e-11,
            rtol=1e-10,
        )
    passed = all(row["passed"] for row in errors.values())
    evidence = new_evidence(
        tier="gpu-numerical" if args.backend == "cuda" else "cpu",
        subject="Experimental generated range ERI raw and weighted shell endpoints",
        inputs_hash=canonical_hash([r["input_hash"] for r in rows]),
    )
    device = (
        command(
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version",
            "--format=csv,noheader",
        )
        if args.backend == "cuda"
        else platform.platform() + " " + cpu_model()
    )
    evidence.update(
        revision=revision,
        backend_selected=args.backend,
        device=device,
        hardware=outcome("pass"),
        block_errors=errors,
    )
    evidence["toolchain"] = {
        "compiler": command(
            str(compiler.cxx if args.backend == "cpu" else compiler.nvcc), "--version"
        ),
        "python": sys.version,
        "numpy": np.__version__,
        "pyscf": pyscf.__version__,
    }
    evidence["settings"] = {
        "device": args.backend,
        "omega_inverse_bohr": args.omega,
        "screening": "disabled",
        "experimental": True,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "samples": args.samples,
    }
    evidence["hashes"] = {
        "equation": canonical_hash(
            {
                "operators": ["erf(omega*r)/r", "erfc(omega*r)/r"],
                "omega": args.omega,
                "derivatives": "all four shell centers at fixed omega and weights",
            }
        ),
        "ir": canonical_hash(sorted(artifacts)),
        "source": canonical_hash(
            source_hashes(
                "common",
                "integral",
                assets=(
                    "src/integrals/range_moments.hpp",
                    "src/integrals/eri_geometry.hpp",
                    "src/scf/weighted_eri_runtime.hpp",
                    "src/scf/cuda_weighted_eri.hpp",
                    "src/tensor/cuda_runtime.cuh",
                    "include/vibeqc/vibeqc.h",
                ),
            )
        ),
        "schedule": canonical_hash(
            {
                "schedule": "bounded_primitive_stream_v2",
                "record_capacity": 7,
                "tile_capacity": 3,
                "artifacts": sorted(artifacts),
            }
        ),
    }
    for stage in ("representation", "source", "compilation", "numerical", "endpoint"):
        evidence["stages"][stage] = outcome(
            "pass" if passed else "fail",
            None if passed else "a declared numerical gate failed",
        )
    evidence["stages"]["production"] = outcome(
        "not-run", "Explicit experimental provider; no production selector is changed."
    )
    evidence["performance"] = outcome(
        "not-run",
        "Raw/weighted timings are diagnostic samples. No matched faster-provider comparison or speedup promotion was attempted.",
    )
    evidence["compilation"] = {
        "seconds": sum(a["compile_seconds"] for a in artifacts.values()),
        "reason": "Sum over distinct compiled programs; cached compiler durations are retained.",
    }
    evidence["memory"]["reason"] = (
        "resources.json retains shared numeric plans verified against native storage. Process-wide peak memory, Python metadata, caller streams and native call stacks/CUDA context are excluded."
    )

    def save(name, value):
        (args.output / name).write_text(
            json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )

    save("samples.json", rows)
    save("resources.json", resources)
    save("artifacts.json", artifacts)
    save(
        "summary.json",
        {
            "revision": revision,
            "dirty": dirty,
            "passed": passed,
            "endpoint_count": len(rows),
            "maximum_raw_or_weighted_error": max(
                v["max_absolute_error"]
                for k, v in errors.items()
                if k.endswith(("/raw", "/weighted"))
            ),
            "scope": "Numerical acceptance of explicit range shell providers only; no RSH functional or performance promotion.",
        },
    )
    evidence["attachments"] = [
        {"path": name, "sha256": file_hash(args.output / name)}
        for name in ("samples.json", "resources.json", "artifacts.json", "summary.json")
    ]
    write_evidence(args.output / "evidence.json", evidence)
    if args.publish is not None:
        argv = [
            "python",
            "tools/validate_range_eri.py",
            "--backend",
            args.backend,
            "--omega",
            str(args.omega),
            "--samples",
            str(args.samples),
            "--architecture",
            args.architecture,
            "--output",
            ".artifacts/range-reproduction",
        ]
        if args.backend == "cuda":
            argv = [
                "srun",
                "--partition=main",
                "--gres=gpu:5090:1",
                "--nodes=1",
                "--ntasks=1",
                "--time=00:10:00",
                *argv,
            ]
        publish(
            args.output,
            {
                "source": {"revision": revision, "dirty": False},
                "reproduction": {"command": argv},
                "decision": {
                    "scope": "numerical",
                    "status": "accepted" if passed else "rejected",
                    "reason": "Independent LR/SR, all center derivatives, raw/weighted, translation and multi-step finite-difference gates are retained. No schedule promotion.",
                },
                "files": [
                    {"path": name, "role": role}
                    for name, role in (
                        ("evidence.json", "evidence"),
                        ("summary.json", "summary"),
                        ("samples.json", "samples"),
                        ("resources.json", "input"),
                        ("artifacts.json", "input"),
                    )
                ],
                "archives": [],
            },
            args.publish,
        )
    return 0 if passed else 1


def main():
    """Select one explicit backend and a new untracked run directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--compiler", type=Path)
    parser.add_argument("--architecture", default="sm_120")
    parser.add_argument("--omega", type=float, default=0.63)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--cache", type=Path, default=ROOT / ".artifacts/range-cache")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish", type=Path)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
