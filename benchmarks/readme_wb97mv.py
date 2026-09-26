"""Matched-grid WB97M-V complete energy/force latency versus GPU4PySCF.

Use the HF README's water/def2-SVP sizes, synchronized interleaved repeats and
fixed post-cold density on each engine. Reference grid generation is adapted to
the explicit common quadrature; GPU4PySCF still owns partitioning, its analytic
grid response, AO/integral/XC/VV10 evaluation, SCF and all force mathematics.
"""

from __future__ import annotations

import argparse
import json
import os
import typing
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

try:
    from _support import raw_output_path
except ModuleNotFoundError:
    from benchmarks._support import raw_output_path


def reference_engine(
    atoms: typing.Any, basis: str, spec: typing.Any, *, spin: int = 0
) -> typing.Any:
    """Build an independent GPU4PySCF engine on the same moving quadrature.

    Supplying atomic rules (not just static final coordinates) is essential:
    GPU4PySCF rebuilds its grids to evaluate analytic partition/grid response.
    Three original-Becke iterations with no radius adjustment match GridSpec.
    """
    import cupy as cp
    from gpu4pyscf.dft import gen_grid, rks, uks
    from pyscf import gto

    if spec.partition_iterations != 3:
        raise ValueError("GPU4PySCF comparator requires three Becke iterations")
    mol = gto.M(atom=atoms, unit="Bohr", basis=basis, spin=spin, cart=False, verbose=0)
    engine = (uks.UKS if spin else rks.RKS)(mol)
    engine.xc = "WB97M_V"
    # Reconstruct the stated quadrature independently from its public spec.
    x, wx = np.polynomial.legendre.leggauss(spec.radial_points)
    t, wt = (x + 1) / 2, wx / 2
    r, wr = t / (1 - t), wt * t**2 / (1 - t) ** 4
    z, wz = np.polynomial.legendre.leggauss(spec.angular_polar)
    phi = np.arange(spec.angular_azimuth) * (2 * np.pi / spec.angular_azimuth)
    zz, pp = np.meshgrid(z, phi, indexing="ij")
    directions = np.stack(
        (np.sqrt(1 - zz**2) * np.cos(pp), np.sqrt(1 - zz**2) * np.sin(pp), zz), axis=-1
    ).reshape(-1, 3)
    wa = np.repeat(wz * 2 * np.pi / len(phi), len(phi))
    table = {}
    radii = dict(spec.element_radii)
    for atom in range(mol.natm):
        symbol, number = mol.atom_symbol(atom), int(mol.atom_charge(atom))
        scale = radii.get(number, 1.0)
        table[symbol] = (
            cp.asarray((r[:, None, None] * directions * scale).reshape(-1, 3)),
            cp.asarray((wr[:, None] * wa * scale**3).ravel()),
        )
    for grids in (engine.grids, engine.nlcgrids):
        grids.radii_adjust = None
        grids.becke_scheme = gen_grid.original_becke
        grids.prune = None
        grids.gen_atomic_grids = lambda mol, *args, **kwargs: table
    engine.small_rho_cutoff = 0
    engine.conv_tol = 1e-11
    engine.conv_tol_grad = 1e-8
    engine.direct_scf_tol = 1e-12
    engine.max_cycle = 180
    return engine


def reference_sample(
    engine: typing.Any, cp: typing.Any, *, density: typing.Any = None
) -> dict[str, typing.Any]:
    """Synchronize the complete fresh SCF plus grid-responsive force endpoint."""
    from benchmarks.compare_gpu4pyscf_batch import (
        GpuCycleTracker,
        gpu_convergence_payload,
    )

    tracker = GpuCycleTracker()
    engine.callback = tracker
    cp.cuda.Stream.null.synchronize()
    started = perf_counter()
    energy = float(engine.kernel(dm0=None if density is None else density.copy()))
    if not engine.converged:
        raise RuntimeError("GPU4PySCF WB97M-V did not converge")
    gradient = engine.nuc_grad_method()
    gradient.grid_response = True
    forces = -gradient.kernel()
    cp.cuda.Stream.null.synchronize()
    elapsed = perf_counter() - started
    convergence = gpu_convergence_payload([engine], [tracker])
    convergence[0]["warm_start"]["used"] = density is not None
    return {
        "seconds": elapsed,
        "convergence": convergence,
        "energies_hartree": [energy],
        "forces_hartree_per_bohr": [cp.asnumpy(forces).tolist()],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atoms", type=int, choices=(3, 6, 12, 24, 48, 96), required=True
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--grid", type=int, nargs=3, default=(48, 16, 32))
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            "run through Slurm main with --gres=gpu:5090:1 and finite --time"
        )
    import cupy as cp
    from vibeqc import Calculator, GridSpec, KsOptions

    from benchmarks._support import cuda_accelerator_metadata, environment_metadata
    from benchmarks.compare_gpu4pyscf_batch import (
        _vibeqc_sample,
        accuracy_gate_summary,
        fixed_warm_start_policy,
        interleaved_engine_order,
        native_build_metadata,
        pair_repeat_accuracy,
    )
    from benchmarks.readme_hf_scaling import scaling_cases

    args.output.parent.mkdir(parents=True, exist_ok=True)
    atoms = scaling_cases()[f"water-{args.atoms}"].atoms
    spec = GridSpec(
        radial_points=args.grid[0],
        angular_polar=args.grid[1],
        angular_azimuth=args.grid[2],
    )
    record = {
        "schema": "vibeqc.readme-wb97mv.v1",
        "status": "running",
        "atoms": args.atoms,
        "aos": args.atoms * 8,
        "basis": "def2-svp spherical",
        "method": "WB97M-V/RKS",
        "endpoint": "SCF energy plus analytic forces",
        "grid": asdict(spec),
        "warm_start_policy": fixed_warm_start_policy(),
        "native_samples": [],
        "reference_samples": [],
    }

    def save(stage: str) -> None:
        record["stage"] = stage
        args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    batch = None
    try:
        record["environment"] = environment_metadata(
            distributions={
                "pyscf": ("pyscf",),
                "gpu4pyscf": ("gpu4pyscf", "gpu4pyscf-cuda12x"),
                "cupy": ("cupy-cuda12x", "cupy"),
                "numpy": ("numpy",),
            }
        )
        record["environment"]["accelerator"] = cuda_accelerator_metadata(cp)
        started = perf_counter()
        reference = reference_engine(atoms, "def2-svp", spec)
        record["reference_prepare_seconds"] = perf_counter() - started
        if not args.reference_only:
            save("native/prepare")
            started = perf_counter()
            calc = Calculator(
                method="wb97m-v",
                basis="def2-svp",
                device="cuda",
                basis_representation="spherical",
                ks_options=KsOptions(grid=spec),
                energy_tolerance=1e-11,
                density_tolerance=1e-9,
                screening_tolerance=1e-12,
                max_iterations=180,
            )
            batch = calc.prepare_batch([atoms], warm_start=True)
            record["native_prepare_seconds"] = perf_counter() - started
            record["native_build"] = native_build_metadata(calc)
            save("native/cold")
            record["native_cold"] = _vibeqc_sample(batch, cp, -2, True)
            batch.set_warm_start_updates(False)
            save("native/priming")
            record["native_priming"] = _vibeqc_sample(batch, cp, -1, True)
        save("reference/cold")
        record["reference_cold"] = reference_sample(reference, cp)
        dm = reference.make_rdm1().copy()
        save("reference/priming")
        record["reference_priming"] = reference_sample(reference, cp, density=dm)
        for index, name in enumerate(interleaved_engine_order(args.repeats)):
            save(f"{name}/repeat/{index}")
            if name == "vibeqc" and batch is not None:
                record["native_samples"].append(_vibeqc_sample(batch, cp, index, True))
            elif name == "gpu4pyscf":
                record["reference_samples"].append(
                    reference_sample(reference, cp, density=dm)
                )
        if batch is not None:
            pairs = pair_repeat_accuracy(
                [
                    record["native_cold"],
                    record["native_priming"],
                    *record["native_samples"],
                ],
                [
                    record["reference_cold"],
                    record["reference_priming"],
                    *record["reference_samples"],
                ],
            )
            record["accuracy"] = accuracy_gate_summary(pairs)
            record["accuracy_pairs"] = pairs
            record["native_force_work"] = (batch.resource_diagnostics or {}).get(
                "generated_force"
            )
            if record["native_force_work"] is None:
                record["native_force_work"] = batch._stationary_cuda_execution.last_work
            record["gates"] = {"energy_hartree": 1e-8, "force_hartree_per_bohr": 1e-7}
            record["accepted"] = (
                record["accuracy"]["maximum_energy_error_hartree"] <= 1e-8
                and record["accuracy"]["maximum_force_error_hartree_per_bohr"] <= 1e-7
            )
        record["timing"] = {
            name: {
                "median_seconds": median(s["seconds"] for s in samples),
                "min_seconds": min(s["seconds"] for s in samples),
                "max_seconds": max(s["seconds"] for s in samples),
            }
            for name, samples in (
                ("vibeqc", record["native_samples"]),
                ("gpu4pyscf", record["reference_samples"]),
            )
            if samples
        }
        record["status"] = "measured"
        save("complete")
        if batch is not None and not record["accepted"]:
            raise RuntimeError("complete WB97M-V energy/force accuracy gate failed")
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        save(record.get("stage", "setup"))
        raise
    finally:
        if batch is not None:
            batch.close()


if __name__ == "__main__":
    main()
