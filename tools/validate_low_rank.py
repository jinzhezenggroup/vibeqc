"""Reproduce bounded Cholesky/refinement experiments on tiny independent fixtures.

Whole solves include setup, factorization, host transfers and exact cleanup.
Dense tensors below belong solely to the pinned small reference gates; the
factorization and solver never obtain these reference arrays. DF is a separate
fixed-auxiliary Hamiltonian and is never used as an exact-accuracy comparator.
"""

import argparse
import json
import os
import platform
import sys
import time
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "tools")]

import numpy as np
from validate_range_eri import command, cpu_model
from vibeqc.fock import FockBuildSpec, FockPlan
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
from vibeqc_compiler.common.resources import ResourceBudget
from vibeqc_compiler.dft import NativeAO
from vibeqc_posthf.coulomb_columns import CoulombColumns
from vibeqc_posthf.fixtures import load_fixture, source_arguments
from vibeqc_posthf.low_rank import IncrementalCholesky
from vibeqc_posthf.low_rank_consumers import LowRankProvider
from vibeqc_posthf.low_rank_cuda import CudaIncrementalCholesky, compile_cholesky_cuda
from vibeqc_posthf.low_rank_refinement import RefinementStage, solve_refined_rhf
from vibeqc_posthf.sources import NativeSource
from vibeqc_validation.publication import publish

CASES = ("h2", "water", "lih")
MODES = ("exact", "fixed_rank", "adaptive_rank", "density_fitted")


def run(args):
    """Measure clean source and keep all raw whole-solve samples in one bundle."""
    if args.backend == "cuda" and not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("real GPU evidence requires a finite Slurm allocation")
    if args.samples < 1:
        raise ValueError("at least one sample required")
    dirty = bool(command("git", "status", "--porcelain"))
    if args.publish and dirty:
        raise ValueError("commit measured source before publishing evidence")
    revision = command("git", "rev-parse", "HEAD")
    args.output.mkdir(parents=True, exist_ok=False)
    artifact = None
    compilation_start = time.perf_counter()
    if args.backend == "cuda":
        compiler = CudaCompilerAdapter(args.nvcc, cuda_target_info("sm_120"))
        artifact = compile_cholesky_cuda(compiler, args.output / "cache")
    compilation_seconds = time.perf_counter() - compilation_start
    errors, rows = {}, []
    budget = ResourceBudget(host_bytes=9 << 20, device_bytes=101 << 20)
    for case in CASES:
        meta, arrays = load_fixture(case)
        source_args = source_arguments(meta)
        basis_args = {k: v for k, v in source_args.items() if k != "auxiliary_basis"}
        independent_density = (
            arrays["conventional_C"] * arrays["conventional_occ"]
        ) @ arrays["conventional_C"].T
        baseline = None
        for mode in MODES:
            samples = []
            for sample in range(args.samples):
                started = time.perf_counter()
                with ExitStack() as stack:
                    basis = stack.enter_context(NativeAO(**basis_args))
                    auxiliary = None
                    if mode == "density_fitted":
                        auxiliary = stack.enter_context(
                            NativeAO(
                                **{
                                    **basis_args,
                                    "basis": source_args["auxiliary_basis"],
                                }
                            )
                        )
                    spec = (
                        FockBuildSpec.hf(
                            coulomb="density_fitted", exchange="density_fitted"
                        )
                        if mode == "density_fitted"
                        else FockBuildSpec.hf()
                    )
                    target = stack.enter_context(
                        FockPlan(
                            basis,
                            spec,
                            auxiliary=auxiliary,
                            device=args.backend,
                            screening_tolerance=0,
                        )
                    )
                    record = {
                        "sample": sample,
                        "target_setup_seconds": time.perf_counter() - started,
                    }
                    if mode in ("exact", "density_fitted"):
                        result = target.solve()
                    else:
                        source_start = time.perf_counter()
                        source = stack.enter_context(NativeSource(**source_args))
                        columns = CoulombColumns(source)
                        record["raw_source_setup_seconds"] = (
                            time.perf_counter() - source_start
                        )
                        factor_start = time.perf_counter()
                        options = {
                            "rank_capacity": columns.space.size,
                            "pair_tile": 5,
                            "budget": budget,
                        }
                        factor = stack.enter_context(
                            IncrementalCholesky(columns, **options)
                            if artifact is None
                            else CudaIncrementalCholesky(columns, artifact, **options)
                        )
                        record["factor_setup_seconds"] = (
                            time.perf_counter() - factor_start
                        )
                        stages = (
                            [RefinementStage(1e-5, 6)]
                            if mode == "fixed_rank"
                            else [
                                RefinementStage(0.1, 2, min(3, columns.space.size)),
                                RefinementStage(1e-5, 4),
                            ]
                        )
                        solved = solve_refined_rhf(factor, target, stages)
                        result = solved.exact
                        record.update(
                            stages=solved.stages,
                            approximate_seconds=solved.approximate_seconds,
                            cleanup_seconds=solved.cleanup_seconds,
                            factor=factor.diagnostics(),
                            source_work=dict(columns.statistics),
                            initialization=solved.diagnostics,
                        )
                    # Stop the whole-solve clock before independent dense checks.
                    # Construction and all execution are included; destruction,
                    # native compilation and reference comparisons are separate.
                    record.update(
                        total_solve_seconds=time.perf_counter() - started,
                        energy=result.energy,
                        density=result.density.tolist(),
                        forces=result.forces.tolist(),
                        iterations=result.iterations,
                        energy_change=result.energy_change,
                        density_rms=result.density_rms,
                        fock_builds=result.fock_builds,
                        target_identity=target.identity,
                        target_resources=target.diagnostics,
                    )
                    key = f"{case}/{mode}/{sample}"
                    if mode == "exact" and baseline is None:
                        baseline = result
                    if mode != "density_fitted":
                        errors[key + "/independent_energy"] = block_error(
                            [result.energy],
                            [meta["records"]["conventional"]["hf_energy"]],
                            atol=2e-9,
                            rtol=0,
                        )
                        errors[key + "/independent_density"] = block_error(
                            result.density, independent_density, atol=4e-7, rtol=0
                        )
                        errors[key + "/matched_final_force"] = block_error(
                            result.forces, baseline.forces, atol=3e-7, rtol=0
                        )
                    record["energy_difference_from_exact"] = (
                        result.energy - baseline.energy
                    )
                    record["force_max_difference_from_exact"] = float(
                        np.max(np.abs(result.forces - baseline.forces))
                    )
                    if mode in ("fixed_rank", "adaptive_rank") and sample == 0:
                        # Independent same-approximation checks precede comparison
                        # with the original operator. AO**4 is tiny fixture-only.
                        physical = [
                            factor.space.unpack(factor.factor_tile(i, 1)[0])
                            for i in range(factor.rank)
                        ]
                        approximation = sum(
                            np.einsum("mn,rs->mnrs", b, b) for b in physical
                        )
                        jk = LowRankProvider(factor).jk(independent_density)
                        errors[key + "/same_approximation_J"] = block_error(
                            jk.coulomb,
                            np.einsum(
                                "mnrs,rs->mn", approximation, independent_density
                            ),
                            atol=3e-11,
                            rtol=0,
                        )
                        errors[key + "/same_approximation_K"] = block_error(
                            jk.exchange,
                            np.einsum(
                                "mrns,rs->mn", approximation, independent_density
                            ),
                            atol=3e-11,
                            rtol=0,
                        )
                        residual = np.empty((factor.space.size, factor.space.size))
                        for i in range(factor.space.size):
                            mu, nu = factor.space.pair(i)
                            for j in range(factor.space.size):
                                rho, sigma = factor.space.pair(j)
                                residual[i, j] = (
                                    factor.space.scale(i)
                                    * factor.space.scale(j)
                                    * (
                                        arrays["ao"][mu, nu, rho, sigma]
                                        - approximation[mu, nu, rho, sigma]
                                    )
                                )
                        diag = factor.diagnostics()
                        record["independent_pair_residual"] = {
                            "entry_max": float(np.max(np.abs(residual))),
                            "frobenius_norm": float(np.linalg.norm(residual)),
                            "minimum_eigenvalue": float(
                                np.min(np.linalg.eigvalsh(residual))
                            ),
                            "conditional_entry_bound": diag["conditional_entry_bound"],
                            "conditional_frobenius_bound": diag[
                                "conditional_spectral_and_frobenius_bound"
                            ],
                            "oracle_scope": "tiny pinned dense fixture; excluded from production allocations/timings",
                        }
                        errors[key + "/conditional_entry_bound"] = block_error(
                            [
                                max(
                                    0,
                                    np.max(np.abs(residual))
                                    - diag["conditional_entry_bound"],
                                )
                            ],
                            [0],
                            atol=3e-10,
                            rtol=0,
                        )
                        errors[key + "/conditional_frobenius_bound"] = block_error(
                            [
                                max(
                                    0,
                                    np.linalg.norm(residual)
                                    - diag["conditional_spectral_and_frobenius_bound"],
                                )
                            ],
                            [0],
                            atol=3e-10,
                            rtol=0,
                        )
                    samples.append(record)
            rows.append(
                {
                    "case": case,
                    "mode": mode,
                    "input_hash": meta["array_hash"],
                    "samples": samples,
                    "median_total_seconds": float(
                        np.median([r["total_solve_seconds"] for r in samples])
                    ),
                }
            )
            print(case, mode, rows[-1]["median_total_seconds"], flush=True)
    passed = all(row["passed"] for row in errors.values())
    evidence = new_evidence(
        tier="gpu-numerical" if artifact else "cpu",
        subject="Experimental incremental Coulomb factors and exact-target RHF cleanup",
        inputs_hash=canonical_hash([r["input_hash"] for r in rows]),
    )
    evidence.update(
        revision=revision,
        backend_selected=args.backend,
        device=command(
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version",
            "--format=csv,noheader",
        )
        if artifact
        else platform.platform() + " " + cpu_model(),
        hardware=outcome("pass"),
        block_errors=errors,
    )
    evidence["toolchain"] = {
        "python": sys.version,
        "numpy": np.__version__,
        "native_library": os.environ["VIBEQC_LIBRARY"],
        "native_library_sha256": file_hash(Path(os.environ["VIBEQC_LIBRARY"])),
        "compiler": command(str(args.nvcc), "--version")
        if artifact
        else command("c++", "--version"),
    }
    evidence["settings"] = {
        "samples": args.samples,
        "cases": CASES,
        "modes": MODES,
        "threads": {
            k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "budget": {"host_bytes": 9 << 20, "device_bytes": 101 << 20},
        "gpu_scope": "CPU raw columns/pivot/PSD/diagonalization; CUDA retained projection and J/K; original target CUDA cleanup"
        if artifact
        else "CPU",
        "timing_scope": "whole solve includes setup through exact forces; excludes compilation, destruction and reference checks; section events enabled on CUDA",
        "DF_scope": "separate fixed auxiliary approximation; differences from exact are observations, not matching-accuracy gates",
    }
    scientific_files = sorted((ROOT / "tools/vibeqc_posthf").glob("low_rank*.py")) + [
        ROOT / "tools/vibeqc_posthf/pair_space.py",
        ROOT / "tools/vibeqc_posthf/coulomb_columns.py",
        ROOT / "src/posthf/cuda_low_rank.cu",
    ]
    evidence["hashes"] = {
        "equation": canonical_hash(
            {
                "pair": "sqrt(m_i*m_j) ERI",
                "factor": "L.T L",
                "J": "sum B trace(B D)",
                "K": "sum B D B",
            }
        ),
        "ir": None,
        "source": canonical_hash(
            {str(p.relative_to(ROOT)): file_hash(p) for p in scientific_files}
        ),
        "schedule": canonical_hash(
            {"fixed": [1e-5, 6], "adaptive": [[0.1, 2, 3], [1e-5, 4]], "pair_tile": 5}
        ),
    }
    evidence["hash_reasons"]["ir"] = (
        "Generic pivoted-Cholesky numerical policy and cuBLAS contractions; no new shell/operator IR."
    )
    for stage in ("representation", "source", "compilation", "numerical", "endpoint"):
        evidence["stages"][stage] = outcome(
            "pass" if passed else "fail", None if passed else "declared gate failed"
        )
    evidence["stages"]["production"] = outcome(
        "not-run", "Experimental internal providers; no production selector promotion."
    )
    evidence["performance"] = outcome(
        "not-run",
        "Small-system diagnostic whole-solve samples; no speedup promotion or asymptotic claim.",
    )
    evidence["compilation"] = {
        "seconds": compilation_seconds,
        "reason": "Native low-rank compilation/cache lookup outside whole-solve timing.",
    }
    evidence["memory"]["reason"] = (
        "Per-sample shared factor/initializer plans and observed native arena/provider bytes; target FockPlan reported separately. Reference tensors, driver/context, Python and BLAS host overhead excluded."
    )

    def save(name, value):
        (args.output / name).write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )

    save("samples.json", rows)
    write_evidence(args.output / "evidence.json", evidence)
    summary = {
        "passed": passed,
        "source": {"revision": revision, "dirty": dirty},
        "results": [
            {k: r[k] for k in ("case", "mode", "median_total_seconds")} for r in rows
        ],
        "maximum_gate_error": max(r["max_absolute_error"] for r in errors.values()),
        "decision": "Numerical experimental acceptance only; small-system timings establish no speedup.",
    }
    save("summary.json", summary)
    if args.publish:
        publish(
            args.output,
            {
                "source": summary["source"],
                "reproduction": {
                    "command": [
                        "python",
                        "tools/validate_low_rank.py",
                        "--backend",
                        args.backend,
                        "--samples",
                        str(args.samples),
                        "--output",
                        f".artifacts/low-rank-{args.backend}-reproduction",
                    ]
                },
                "files": [
                    {"path": p, "role": r}
                    for p, r in [
                        ("evidence.json", "evidence"),
                        ("samples.json", "samples"),
                        ("summary.json", "summary"),
                    ]
                ],
                "decision": {
                    "status": "accepted" if passed else "rejected",
                    "scope": "numerical",
                    "reason": summary["decision"],
                },
                "archives": [],
            },
            args.publish,
        )
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--nvcc", type=Path, default=Path("/group/software/cuda-12.9.1/bin/nvcc")
    )
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish", type=Path)
    raise SystemExit(run(parser.parse_args()))
