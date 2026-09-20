"""Compare existing DF derivative routes on explicit equal or practical bases.

This is a qualification probe, not a production selector. Complete clean
endpoints precede a separate intrusive work ledger; no policy is promoted here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
import typing
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np

try:
    from benchmarks._retention import raw_output_path
except ModuleNotFoundError:
    from _retention import raw_output_path

from benchmarks._cases import benchmark_cases
from benchmarks.compare_gpu4pyscf_batch import (
    convergence_payload,
    load_comparison_basis,
    native_build_metadata,
)
from benchmarks.df_component_ledger import aggregate, read_trace

CONTROLS = (
    "WEIGHTED_EXECUTION",
    "DERIVATIVE_PAIRS",
    "SHELL_SCHEDULE",
    "PRIMITIVE_BUCKETS",
)
VARIANTS = {
    "auto": (None, None, None, None),
    "shell": ("shell", "symmetric", "compact", "off"),
    "packet": ("shell", "symmetric", "compact", "packet"),
    "packed": ("shell", "packed", "compact", "packet"),
}
TRACE_CONTROLS = ("TRACE", "HOST_TRACE", "PROGRESS_TRACE", "SHELL_COUNTERS")


@contextmanager
def controls(variant: typing.Any, trace: typing.Any = None) -> typing.Any:
    """Set every arm completely and restore its caller even after a failure."""
    updates = dict(zip(CONTROLS, VARIANTS[variant], strict=True))
    updates.update({name: None for name in TRACE_CONTROLS})
    if trace is not None:
        updates.update(TRACE=str(trace), SHELL_COUNTERS="1")
    updates = {"VIBEQC_DF_" + name: value for name, value in updates.items()}
    old = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def errors(
    result: typing.Any, reference_energy: typing.Any, reference_force: typing.Any
) -> typing.Any:
    """Reject broadcasting/nonfinite output before evaluating physical errors."""
    energy = np.asarray(result.energies)
    force = np.asarray([item.forces for item in result.items])
    if energy.shape != reference_energy.shape or force.shape != reference_force.shape:
        raise RuntimeError("reference/endpoint shapes differ")
    if not all(
        np.all(np.isfinite(x))
        for x in (energy, force, reference_energy, reference_force)
    ):
        raise RuntimeError("nonfinite reference/endpoint")
    return float(np.max(np.abs(energy - reference_energy))), float(
        np.max(np.abs(force - reference_force))
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=benchmark_cases(), required=True)
    parser.add_argument("--orbital-basis-file", type=Path)
    parser.add_argument("--auxiliary-basis-file", type=Path)
    parser.add_argument(
        "--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS)
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Run clean comparisons without a subsequent intrusive pass",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--copies",
        type=int,
        default=1,
        help="Replicate the molecular fixture along x with 8-bohr separation",
    )
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument("--source-patch", type=Path, required=True)
    parser.add_argument("--warm-checkpoint-in", type=Path)
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or args.repeats < 1
        or args.batch < 1
        or args.copies < 1
    ):
        parser.error(
            "requires a finite Slurm allocation and positive repeat/batch counts"
        )
    if len(set(args.variants)) != len(args.variants):
        parser.error("duplicate variants")
    if args.output.exists():
        parser.error("refusing to overwrite evidence")
    patch = args.source_patch.read_bytes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import pyscf
    from pyscf import gto, scf
    from vibeqc import Calculator

    case = benchmark_cases()[args.case]
    if args.copies != 1:
        if not isinstance(case.vibeqc_basis, str) or case.method != "rhf":
            parser.error("replication requires a named-basis closed-shell fixture")
        case = replace(
            case,
            atoms=tuple(
                (element, (xyz[0] + 8.0 * copy, xyz[1], xyz[2]))
                for copy in range(args.copies)
                for element, xyz in case.atoms
            ),
            charge=case.charge * args.copies,
            expected_ao_count=None,
        )
    orbital, cpu_orbital = case.vibeqc_basis, case.pyscf_basis
    if args.orbital_basis_file:
        orbital, cpu_orbital = load_comparison_basis(
            args.orbital_basis_file, case, role="orbital", compute_forces=True
        )
    auxiliary, cpu_auxiliary = orbital, cpu_orbital
    if args.auxiliary_basis_file:
        auxiliary, cpu_auxiliary = load_comparison_basis(
            args.auxiliary_basis_file, case, role="auxiliary", compute_forces=True
        )
    mol = gto.M(
        atom=case.atoms,
        unit="Bohr",
        basis=cpu_orbital,
        cart=case.basis_representation == "cartesian",
        spin=case.multiplicity - 1,
        charge=case.charge,
        verbose=0,
    )
    oracle = (scf.UHF if case.method == "uhf" else scf.RHF)(mol).density_fit(
        auxbasis=cpu_auxiliary
    )
    oracle.conv_tol, oracle.conv_tol_grad, oracle.max_cycle = 1e-13, 1e-12, 200
    oracle.direct_scf_tol = 1e-14
    oracle.kernel()
    if not oracle.converged:
        raise RuntimeError("independent SCF did not converge")
    gradient = oracle.nuc_grad_method()
    gradient.auxbasis_response = True
    ref_energy = np.repeat(oracle.e_tot, args.batch)
    ref_force = np.repeat((-gradient.kernel())[None, :, :], args.batch, axis=0)
    practical = args.auxiliary_basis_file is not None
    energy_gate, force_gate = (3e-11, 3e-11) if practical else (1e-9, 1e-8)
    calc = Calculator(
        method=case.method,
        basis=orbital,
        auxiliary_basis=auxiliary,
        basis_representation=case.basis_representation,
        device="cuda",
        density_fitting="cuda",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-14,
        max_iterations=100,
    )
    payload = {
        "schema": "vibeqc.df_admission_probe.v1",
        "case": args.case,
        "geometries_bohr": [case.atoms] * args.batch,
        "method": case.method,
        "charge": case.charge,
        "multiplicity": case.multiplicity,
        "representation": case.basis_representation,
        "orbital_basis": cpu_orbital,
        "auxiliary_basis": cpu_auxiliary,
        "settings": {
            "energy_tolerance": 1e-12,
            "density_tolerance": 1e-10,
            "screening_tolerance": 1e-14,
            "metric_relative_threshold": 1e-10,
            "max_iterations": 100,
        },
        "gates": {"energy": energy_gate, "force": force_gate},
        "reference": {
            "pyscf_version": pyscf.__version__,
            "energy": ref_energy.tolist(),
            "force": ref_force.tolist(),
            "energy_tolerance": oracle.conv_tol,
            "gradient_tolerance": oracle.conv_tol_grad,
        },
        "native_build": native_build_metadata(calc),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_patch_sha256": hashlib.sha256(patch).hexdigest(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "environment": {
            key: value
            for key, value in os.environ.items()
            if key.startswith("VIBEQC_")
            or key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "variants": {
            key: dict(zip(CONTROLS, VARIANTS[key], strict=True))
            for key in args.variants
        },
        "samples": [],
    }

    def save() -> None:
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")

    save()
    try:
        with controls("auto"):
            start = time.perf_counter()
            owner = calc.prepare_batch(
                [case.atoms] * args.batch,
                charges=[case.charge] * args.batch,
                multiplicities=[case.multiplicity] * args.batch,
            )
            payload["prepare_seconds"] = time.perf_counter() - start
            with owner as batch:

                def execute(
                    phase: typing.Any,
                    variant: typing.Any,
                    repeat: typing.Any,
                    trace: typing.Any = None,
                ) -> None:
                    start = time.perf_counter()
                    result = batch.execute(strict=True)
                    seconds = time.perf_counter() - start
                    energy_error, force_error = errors(result, ref_energy, ref_force)
                    row = {
                        "phase": phase,
                        "variant": variant,
                        "repeat": repeat,
                        "seconds": seconds,
                        "energy_error": energy_error,
                        "force_error": force_error,
                        "energy": result.energies.tolist(),
                        "force": [np.asarray(x.forces).tolist() for x in result.items],
                        "iterations": [x.iterations for x in result.items],
                        "convergence": convergence_payload(result),
                        "metric": [
                            x.to_dict()
                            for x in batch.last_density_fitting_metric_diagnostics()
                        ],
                    }
                    if trace is not None:
                        row["components"] = aggregate(read_trace(trace))
                    payload["samples"].append(row)
                    save()
                    if energy_error > energy_gate or force_error > force_gate:
                        raise RuntimeError(
                            f"{phase}/{variant}: independent energy/force gate failed"
                        )
                    print(
                        phase,
                        variant,
                        repeat,
                        seconds,
                        row["iterations"],
                        force_error,
                        flush=True,
                    )

                execute("cold", "auto", 0)
                if args.warm_checkpoint_in:
                    payload["warm_checkpoint_restore"] = batch.load_checkpoint(
                        args.warm_checkpoint_in, allow_warm=True
                    )
                checkpoint = args.output.with_suffix(".vqckpt")
                if checkpoint.exists():
                    raise RuntimeError("refusing to overwrite checkpoint")
                batch.save_checkpoint(checkpoint)
                from vibeqc.checkpoint import inspect_checkpoint

                payload["frozen_density_sha256"] = {
                    b["name"]: b["sha256"]
                    for b in inspect_checkpoint(checkpoint).blobs
                    if b["name"].startswith("density_")
                }
                batch.set_warm_start_updates(False)
                for repeat in range(args.repeats):
                    for variant in (
                        args.variants if repeat % 2 == 0 else args.variants[::-1]
                    ):
                        with controls(variant):
                            execute("prime", variant, repeat)
                            execute("clean", variant, repeat)
                for variant in () if args.no_diagnostics else args.variants:
                    trace = args.output.with_suffix(f".{variant}.jsonl")
                    with controls(variant):
                        execute("diagnostic-prime", variant, 0)
                    with controls(variant, trace):
                        execute("diagnostic", variant, 0, trace)
        payload["summary"] = {
            variant: statistics.median(
                s["seconds"]
                for s in payload["samples"]
                if s["variant"] == variant and s["phase"] == "clean"
            )
            for variant in args.variants
        }
        save()
    except Exception as exc:
        payload["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        save()
        raise


if __name__ == "__main__":
    main()
