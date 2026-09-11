"""Retain native partial-Hessian/HVP numerical, timing and resource evidence.

Independent Libcint Hessians and three-step first-gradient differences gate
each selected native output. These are diagnostic shell-provider timings;
there is no molecular Hessian or faster-schedule promotion in this driver.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python"), str(ROOT / "tools")]

import numpy as np
import pyscf
from validate_range_eri import command, cpu_model
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
from vibeqc_compiler.integral.second_derivatives import (
    build_eri_second_ir,
    build_one_electron_second_ir,
)
from vibeqc_compiler.integral.second_derivatives_execute import (
    PreparedSecondDerivative,
    SecondPrimitive,
    compile_second_derivative,
)
from vibeqc_validation.publication import publish
from vibeqc_validation.second_derivatives import (
    libcint_eri_hessian,
    libcint_one_electron_hessian,
    libcint_primitive_gradient,
)

POSITIONS = np.array(
    [
        [0.13, -0.31, 0.24],
        [-0.43, 0.27, 0.51],
        [0.68, -0.14, -0.22],
        [-0.21, 0.48, -0.63],
    ]
)
CASES = (
    ("overlap_pd", "overlap", (1, 2), False),
    ("kinetic_pd", "kinetic", (1, 2), False),
    ("attraction_pd", "nuclear_attraction", (1, 2), False),
    ("attraction_fs", "nuclear_attraction", (3, 0), False),
    ("eri_psss", "eri", (1, 0, 0, 0), False),
    ("eri_dpsp", "eri", (2, 1, 0, 1), False),
    ("eri_dpsp_coincident", "eri", (2, 1, 0, 1), True),
    ("eri_fsss", "eri", (3, 0, 0, 0), False),
    ("eri_ffff_minimum_tile", "eri", (3, 3, 3, 3), False),
)


def run(args):
    """Run explicit fixtures and publish only a complete clean-source decision."""
    # Tiny independent oracle blocks use one reference worker; changing the
    # machine's OpenMP default must not make the finite Slurm reproduction hang.
    pyscf.lib.num_threads(1)
    if args.backend == "cuda" and not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("second derivative CUDA evidence requires a Slurm allocation")
    if args.samples < 1:
        raise ValueError("at least one timing sample is required")
    dirty = bool(command("git", "status", "--porcelain"))
    if args.publish is not None and dirty:
        raise ValueError(
            "commit the measured source before publishing second derivative evidence"
        )
    if args.publish is not None and args.case:
        raise ValueError("publication requires the complete declared case inventory")
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

    def save(name, value):
        (args.output / name).write_text(
            json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )

    def execute(integral, components, coordinates, primitives, weights, direction):
        artifact = compile_second_derivative(
            integral,
            compiler,
            args.cache,
            component_indices=components,
            output_indices=coordinates,
        )
        key, metadata = artifact.program_identity, artifact.native.metadata
        artifacts[key] = {
            "native_key": metadata["key"],
            "binary_sha256": metadata["binary_sha256"],
            "source_sha256": metadata["identity"]["source"],
            "target": metadata["identity"]["target"],
            "compile_seconds": metadata["compile_seconds"],
            "compiler_resources": metadata["resources"],
            "angular": list(integral.signature.angular),
            "component_indices": list(components),
            "output_indices": list(coordinates),
            "output": integral.contractions[0].output,
        }
        records = tuple(
            SecondPrimitive(exponents, centers, weights, direction, coefficient)
            for exponents, coefficient, centers in primitives
        )
        started = time.perf_counter()
        plan = PreparedSecondDerivative(artifact, record_capacity=2)
        preparation_seconds = time.perf_counter() - started
        with plan:
            cold = plan.contract(records, profile=True)
            samples = []
            for _ in range(args.samples):
                result = plan.contract(records, profile=True)
                samples.append(
                    {
                        "wall_seconds": result.diagnostics["wall_seconds"],
                        "device": result.diagnostics["device_timing"],
                    }
                )
                np.testing.assert_allclose(
                    result.values, cold.values, atol=5e-13, rtol=5e-13
                )
            resources[key] = plan.resource_plan.to_dict()
        return result.values[0], {
            "program": key,
            "preparation_seconds": preparation_seconds,
            "cold_seconds": cold.diagnostics["wall_seconds"],
            "samples": samples,
            "records": len(records),
            "chunks": result.diagnostics["chunks"],
            "input_sha256": result.diagnostics["input_sha256"],
        }

    for name, family, angular, coincident in CASES:
        if args.case and name not in args.case:
            continue

        def integral(output, family=family, angular=angular):
            return (
                build_eri_second_ir(angular, output=output)
                if family == "eri"
                else build_one_electron_second_ir(
                    family, angular, charge=2.3, output=output
                )
            )

        ir = integral("weighted_hessian")
        count = len(ir.operator.centers)
        dimension = 3 * count
        centers = POSITIONS[:count].copy()
        if coincident:
            centers[1] = centers[0]
        minimum = angular == (3, 3, 3, 3)
        components = (0,) if minimum else (0, min(4, ir.signature.component_count - 1))
        component_weights = (0.73,) if minimum else (0.73, -0.31)
        # The through-ffff endpoint deliberately measures the minimum native
        # tile; other endpoints expose every HVP output coordinate. Independent
        # test fixtures separately validate complete mathematical Hessians.
        coordinates = (
            (0,)
            if minimum
            else tuple(
                i * dimension + (5 * i + 1) % dimension for i in range(dimension)
            )
        )
        hvp_coordinates = (0,) if minimum else tuple(range(dimension))
        direction = np.random.default_rng(178).normal(size=(count, 3))
        direction /= np.linalg.norm(direction)
        base = np.array((0.6, 0.8, 1.1, 0.9)[: len(angular)])
        primitives = tuple(
            (tuple(base * scale), coefficient, tuple(map(tuple, centers)))
            for scale, coefficient in ((1.0, 0.73), (1.2, -0.21), (0.75, 0.08))
        )
        inputs = {
            "family": family,
            "angular": angular,
            "centers": centers.tolist(),
            "charge": 2.3,
            "primitive_exponents_and_scales": [[list(e), c] for e, c, _ in primitives],
            "component_indices": components,
            "weights": component_weights,
            "direction": direction.tolist(),
            "hessian_coordinates": coordinates,
            "hvp_coordinates": hvp_coordinates,
        }
        reference = np.zeros((dimension, dimension, len(components)))
        for exponents, coefficient, _ in primitives:
            block = (
                libcint_eri_hessian(angular, exponents, centers)
                if family == "eri"
                else libcint_one_electron_hessian(
                    family, angular, exponents, centers, 2.3
                )
            )
            reference += (
                coefficient
                * block.reshape(dimension, dimension, -1)[:, :, list(components)]
            )
        raw, raw_timings = [], []
        for packed, index in enumerate(components):
            actual, timing = execute(
                integral("raw_hessian"), (index,), coordinates, primitives, None, None
            )
            expected = reference[:, :, packed].ravel()[list(coordinates)]
            errors[f"{name}/raw/{index}"] = block_error(
                actual, expected, atol=5e-11, rtol=2e-11
            )
            raw.append(actual)
            raw_timings.append(timing)
        weighted, weighted_timing = execute(
            ir, components, coordinates, primitives, component_weights, None
        )
        matrix = np.sum(reference * np.array(component_weights), axis=2)
        errors[f"{name}/weighted"] = block_error(
            weighted, matrix.ravel()[list(coordinates)], atol=5e-11, rtol=2e-11
        )
        errors[f"{name}/raw_weighted"] = block_error(
            weighted,
            np.array(component_weights) @ np.array(raw),
            atol=5e-12,
            rtol=5e-12,
        )
        hvp, hvp_timing = execute(
            integral("weighted_hvp"),
            components,
            hvp_coordinates,
            primitives,
            component_weights,
            tuple(map(tuple, direction)),
        )
        expected_hvp = (matrix @ direction.ravel())[list(hvp_coordinates)]
        errors[f"{name}/hvp"] = block_error(hvp, expected_hvp, atol=5e-11, rtol=2e-11)
        scale = max(1.0, float(np.max(np.abs(matrix))))
        if not minimum:
            errors[f"{name}/hvp_translation"] = block_error(
                hvp.reshape(count, 3).sum(axis=0),
                np.zeros(3),
                atol=5e-12 * scale,
                rtol=0,
            )
        curve = []
        for step in (1e-3, 3e-4, 1e-4):
            values = []
            for sign in (1, -1):
                gradient = np.zeros(dimension)
                for exponents, coefficient, _ in primitives:
                    block = libcint_primitive_gradient(
                        family,
                        angular,
                        exponents,
                        centers + sign * step * direction,
                        2.3,
                    )
                    gradient += coefficient * (
                        block.reshape(dimension, -1)[:, list(components)]
                        @ np.array(component_weights)
                    )
                values.append(gradient)
            finite = ((values[0] - values[1]) / (2 * step))[list(hvp_coordinates)]
            curve.append(
                {
                    "step_bohr": step,
                    "values": finite.tolist(),
                    "max_absolute_error": float(np.max(np.abs(finite - hvp))),
                }
            )
        errors[f"{name}/finite_difference"] = block_error(
            hvp, curve[-1]["values"], atol=2e-7 * scale, rtol=0
        )
        convergence_limit = max(5e-11 * scale, 0.03 * curve[0]["max_absolute_error"])
        errors[f"{name}/finite_difference_convergence"] = block_error(
            [curve[-1]["max_absolute_error"]], [0], atol=convergence_limit, rtol=0
        )
        rows.append(
            {
                "case": name,
                "inputs": inputs,
                "input_hash": canonical_hash(inputs),
                "raw": np.asarray(raw).tolist(),
                "weighted": weighted.tolist(),
                "hvp": hvp.tolist(),
                "reference_weighted": matrix.ravel()[list(coordinates)].tolist(),
                "reference_hvp": expected_hvp.tolist(),
                "finite_difference": curve,
                "raw_timings": raw_timings,
                "weighted_timing": weighted_timing,
                "hvp_timing": hvp_timing,
                "native_translation": "not-run: minimum coordinate tile"
                if minimum
                else "checked",
            }
        )
        # Preserve completed cases even if a later compiler or reference call
        # fails. Only the complete final envelope may be published below.
        save("samples.json", rows)
        save("artifacts.json", artifacts)
        save("resources.json", resources)
        print(
            name,
            "passed"
            if all(v["passed"] for k, v in errors.items() if k.startswith(name + "/"))
            else "FAILED",
            flush=True,
        )
    passed = all(value["passed"] for value in errors.values())
    evidence = new_evidence(
        tier="gpu-numerical" if args.backend == "cuda" else "cpu",
        subject="Experimental native partial-Hessian and fixed-weight HVP shell endpoints",
        inputs_hash=canonical_hash([row["input_hash"] for row in rows]),
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
        "experimental": True,
        "screening": "disabled",
        "electronic_response": "excluded",
        "reference_threads": pyscf.lib.num_threads(),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "samples": args.samples,
        "record_capacity": 2,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "through_ffff_scope": "one explicit coordinate/component tile",
    }
    evidence["hashes"] = {
        "equation": canonical_hash(
            {
                "derivative": "fixed W and v partial integral Hessian/HVP",
                "recovery": "R H R.T; R.T v then R output",
                "precision": "fp64",
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
                "schedule": "bounded_coordinate_tile_v1",
                "record_capacity": 2,
                "artifacts": sorted(artifacts),
            }
        ),
    }
    for stage in ("representation", "source", "compilation", "numerical", "endpoint"):
        evidence["stages"][stage] = outcome(
            "pass" if passed else "fail",
            None if passed else "a declared second-order gate failed",
        )
    evidence["stages"]["production"] = outcome(
        "not-run", "No molecular response assembly or production schedule is enabled."
    )
    evidence["performance"] = outcome(
        "not-run",
        "Useful raw/weighted/HVP timing samples are diagnostic; no matched faster-provider comparison or speedup promotion.",
    )
    evidence["compilation"] = {
        "seconds": sum(value["compile_seconds"] for value in artifacts.values()),
        "reason": "Distinct artifact compiler durations; cache hits retain original durations.",
    }
    evidence["memory"]["reason"] = (
        "Shared numeric plans are verified against native storage; process-wide memory, references, Python metadata, caller inputs and native stacks/CUDA context are excluded. PTXAS resource reports are retained."
    )
    maximum = max(
        value["max_absolute_error"]
        for key, value in errors.items()
        if "/raw/" in key or key.endswith(("/weighted", "/hvp"))
    )
    save(
        "summary.json",
        {
            "revision": revision,
            "dirty": dirty,
            "passed": passed,
            "case_count": len(rows),
            "maximum_analytic_error": maximum,
            "scope": "Selected partial-Hessian/HVP integral endpoints; no molecular Hessian or speedup promotion.",
        },
    )
    names = ("samples.json", "artifacts.json", "resources.json", "summary.json")
    evidence["attachments"] = [
        {"path": name, "sha256": file_hash(args.output / name)} for name in names
    ]
    write_evidence(args.output / "evidence.json", evidence)
    if args.publish is not None:
        argv = [
            "env",
            "OMP_NUM_THREADS=1",
            "OPENBLAS_NUM_THREADS=1",
            "python",
            "tools/validate_second_derivatives.py",
            "--backend",
            args.backend,
            "--samples",
            str(args.samples),
            "--architecture",
            args.architecture,
            "--output",
            f".artifacts/second-reproduction-{args.backend}",
        ]
        if args.backend == "cuda":
            argv = [
                "srun",
                "--partition=main",
                "--gres=gpu:5090:1",
                "--nodes=1",
                "--ntasks=1",
                "--time=00:15:00",
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
                    "reason": "Independent raw/weighted Hessian blocks and HVPs, three-step Libcint first-gradient differences, useful timings and bounded native resources. No schedule promotion.",
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
    """Select an explicit backend and preserve a new transient output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--compiler", type=Path)
    parser.add_argument("--architecture", default="sm_120")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--case", action="append", choices=[row[0] for row in CASES])
    parser.add_argument("--cache", type=Path, default=ROOT / ".artifacts/second-cache")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish", type=Path)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
