"""Compare complete direct/DF RHF force endpoints under one scientific protocol.

Run reference and native processes separately inside a finite Slurm allocation.
Each approximation has its own independent GPU4PySCF energy/force oracle. DF
ablations share one frozen post-cold (or post-move) density and alternate order;
trace replays are separate from clean wall times. This is an explicit packed-B
qualification; reference timing is measured separately with its own stopping
policy. This does not qualify the default DF planner.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

from benchmarks._cases import benchmark_cases
from benchmarks._retention import raw_output_path
from benchmarks.compare_gpu4pyscf_batch import (
    GpuCycleTracker,
    _configure_reference_scf,
    gpu_convergence_payload,
    load_comparison_basis,
    native_build_metadata,
    require_tuned_native_build,
    scaled_geometries,
)
from benchmarks.df_component_ledger import read_trace
from benchmarks.readme_hf_scaling import scaling_cases

if TYPE_CHECKING:
    from collections.abc import Iterator


CASES = {
    96: "water-tetramer-def2-svp-spherical",
    384: "water-hexadecamer-2s4-def2-svp-spherical",
    768: "water-32mer-4s4-def2-svp-spherical",
}
CONTROLS = {
    "df-baseline": ("dense", "spectral"),
    "df-final-k": ("auto", "spectral"),
    "df": ("auto", "auto"),
}
AUXILIARY = Path(
    "benchmarks/results/issue206-practical-auxiliary/identity/cc-pvdz-jkfit.json"
)


@contextmanager
def reference_work_counter(engine: Any) -> Iterator[dict[str, Any]]:
    """Count actual SCF get_veff calls, including the pre-loop Fock.

    The tiny Python counter stays inside the endpoint; it does not perform
    extra electronic work. Restore descriptors and callbacks even on failure.
    These counts describe SCF only, separately from the gradient's contractions.
    """
    original = engine.get_veff
    owned = "get_veff" in vars(engine)
    override = vars(engine).get("get_veff")
    callback = engine.callback
    tracker = GpuCycleTracker()
    work = {"scf_jk_builds": 0, "tracker": tracker}

    def counted(*args: Any, **kwargs: Any) -> Any:
        work["scf_jk_builds"] += 1
        return original(*args, **kwargs)

    engine.get_veff = counted
    engine.callback = tracker
    try:
        yield work
    finally:
        if owned:
            engine.get_veff = override
        else:
            del engine.get_veff
        engine.callback = callback


def select_control(route: str) -> None:
    """Change only final-K/response arithmetic; retain the same value/SCF owner."""
    if route in CONTROLS:
        final, metric = CONTROLS[route]
        os.environ["VIBEQC_DF_FINAL_EXCHANGE"] = final
        os.environ["VIBEQC_DF_OCCUPIED_METRIC"] = metric


def check_endpoint(item: Any, reference: dict) -> dict:
    """Gate every endpoint, including diagnostic and cold calls, without broadcasting."""
    actual = np.asarray(item.forces)
    expected = np.asarray(reference["forces"])
    valid = (
        item.succeeded
        and item.converged
        and reference["converged"]
        and actual.shape == expected.shape
        and np.isfinite(actual).all()
        and np.isfinite(expected).all()
        and np.isfinite(item.energy)
        and np.isfinite(reference["energy"])
    )
    energy_error = abs(item.energy - reference["energy"]) if valid else None
    force_error = float(np.max(np.abs(actual - expected))) if valid else None
    return {
        "energy_error": energy_error,
        "force_error": force_error,
        "gate": bool(valid and energy_error <= 1e-8 and force_error <= 1e-7),
    }


def main() -> None:
    """Write each completed endpoint before proceeding, preserving failed evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("reference", "native"))
    parser.add_argument(
        "--aos", type=int, choices=(24, 48, 96, 192, 384, 768), required=True
    )
    parser.add_argument(
        "--nested-water",
        action="store_true",
        help="use the README's nested water32mer prefixes at every size",
    )
    parser.add_argument("--route", choices=("direct", *CONTROLS), required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--disable-warm-reuse",
        action="store_true",
        help="same-binary DF control: rebuild the ordinary seed and baseline",
    )
    parser.add_argument(
        "--interleave",
        action="store_true",
        help="alternate all DF controls on one frozen density",
    )
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("GPU work requires a finite Slurm allocation")
    if args.repeats < 1 or (args.interleave and args.route == "direct"):
        parser.error("positive repeats and a DF route are required for interleaving")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    is_df = args.route != "direct"
    if not args.nested_water and args.aos not in CASES:
        parser.error("this size requires --nested-water")
    case_name = f"water-{args.aos // 8}" if args.nested_water else CASES[args.aos]
    case = (
        scaling_cases()[case_name]
        if args.nested_water
        else benchmark_cases()[case_name]
    )
    native_aux, reference_aux = load_comparison_basis(
        AUXILIARY, case, role="auxiliary", compute_forces=True
    )
    original = scaled_geometries(case.atoms, 1)[0]
    moved = [
        (z, tuple(np.asarray(x) + ((0, 0, 0.001) if i == 1 else (0, 0, 0))))
        for i, (z, x) in enumerate(original)
    ]
    geometries = [original, moved]
    protocol = {
        "case": case_name,
        "aos": args.aos,
        "method": "rhf",
        "orbital_basis": case.pyscf_basis,
        "representation": "spherical",
        "geometries_bohr": geometries,
        "properties": ["energy", "forces"],
        "density_fitting": is_df,
        "auxiliary_sha256": hashlib.sha256(AUXILIARY.read_bytes()).hexdigest()
        if is_df
        else None,
        "energy_tolerance": 1e-12,
        "density_tolerance": 1e-10,
        "reference_gradient_tolerance": 1e-10,
        "screening_tolerance": 1e-12,
        "reference_direct_scf_tolerance": 1e-14,
        "max_iterations": 100,
        "energy_gate": 1e-8,
        "force_gate": 1e-7,
        "threads": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    # JSON canonicalization also normalizes the geometry's numpy scalar/tuple layout.
    protocol = json.loads(json.dumps(protocol))
    identity = {
        "protocol": protocol,
        "mode": args.mode,
        "route": args.route,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_status": subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ),
        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pyscf")},
        "cold_definition": "prepare_batch plus first complete execute; calculator/library loading excluded",
        "warm_definition": "fixed engine-local post-cold or post-move density; updates disabled",
        "native_acceptance": {
            "policy": "shared_cuda_hf_fp64_v1",
            "energy": "abs(E - previous_E) < energy_tolerance + 16*epsilon*max(1,abs(E),abs(previous_E))",
            "density": "density_step_rms < density_tolerance",
            "physical": "max_abs(FDS-SDF) <= min(1e-8,density_tolerance), both UHF spins",
            "baseline": "finite previous energy required; direct and qualified singleton occupied DF may reuse an exactly matched validated warm state; other DF seeds rebuild it",
            "final": "strict physical final Fock validation remains required",
        },
        "fock_build_count_contract": "null means the native API does not export a count; DF diagnostic traces separately retain actual SCF/final-K work",
        "ablations": CONTROLS,
    }
    records = []
    reference_samples = []

    def save(row: dict, *, reference_sample: bool = False) -> None:
        (reference_samples if reference_sample else records).append(row)
        (output / "results.json").write_text(
            json.dumps(
                {
                    "identity": identity,
                    "records": records,
                    "reference_samples": reference_samples,
                },
                indent=2,
            )
            + "\n"
        )
        print(
            json.dumps(
                {k: v for k, v in row.items() if k not in ("forces", "metric", "trace")}
            ),
            flush=True,
        )

    if args.mode == "reference":
        import cupy as cp
        import gpu4pyscf
        from pyscf import gto, scf

        identity["packages"]["gpu4pyscf"] = gpu4pyscf.__version__
        identity["reference_timing"] = (
            "separate process; complete synchronized endpoints; actual SCF get_veff calls counted"
        )
        previous_geometry_density = None
        for index, geometry in enumerate(geometries):
            cp.cuda.get_current_stream().synchronize()
            prepare_start = time.perf_counter()
            mol = gto.M(
                atom=geometry,
                basis=case.pyscf_basis,
                unit="Bohr",
                cart=False,
                verbose=0,
            )
            mf = scf.RHF(mol)
            if is_df:
                mf = mf.density_fit(auxbasis=reference_aux)
            mf = mf.to_gpu()
            _configure_reference_scf(
                mf,
                energy_tolerance=1e-12,
                gradient_tolerance=1e-10,
                max_iterations=100,
                full_fock=True,
                density_fitting=is_df,
            )
            prepare_seconds = time.perf_counter() - prepare_start

            def reference_execute(
                phase: str,
                seed: Any,
                repeat: int = 0,
                *,
                mf: Any = mf,
                index: int = index,
                prepare_seconds: float = prepare_seconds,
            ) -> dict:
                # A seed copy is input preparation, just as the native frozen
                # host snapshot exists before execute starts. Fock work stays timed.
                dm0 = None if seed is None else seed.copy()
                cp.cuda.get_current_stream().synchronize()
                with reference_work_counter(mf) as work:
                    start = time.perf_counter()
                    energy = float(mf.kernel(dm0=dm0))
                    scf_builds = work["scf_jk_builds"]
                    gradient = mf.nuc_grad_method()
                    if is_df:
                        gradient.auxbasis_response = True
                    forces = cp.asnumpy(-gradient.kernel())
                    cp.cuda.get_current_stream().synchronize()
                    seconds = time.perf_counter() - start
                    convergence = gpu_convergence_payload([mf], [work["tracker"]])[0]
                if (
                    not mf.converged
                    or not np.isfinite(energy)
                    or not np.isfinite(forces).all()
                ):
                    raise RuntimeError("independent reference failed")
                return {
                    "geometry": index,
                    "phase": phase,
                    "repeat": repeat,
                    "energy": energy,
                    "forces": forces.tolist(),
                    "converged": bool(mf.converged),
                    "iterations": convergence["iterations"],
                    "scf_jk_builds": scf_builds,
                    "convergence": convergence,
                    "seconds": seconds,
                    "complete_seconds": seconds
                    + (prepare_seconds if phase in ("cold", "moved") else 0),
                    "prepare_seconds": prepare_seconds
                    if phase in ("cold", "moved")
                    else 0,
                }

            oracle_row = reference_execute(
                "cold" if index == 0 else "moved", previous_geometry_density
            )
            save(oracle_row)
            frozen = mf.make_rdm1().copy()
            previous_geometry_density = frozen.copy()
            for repeat in range(args.repeats):
                row = reference_execute(
                    "warm" if index == 0 else "moved-warm", frozen, repeat
                )
                row.update(
                    check_endpoint(
                        SimpleNamespace(
                            energy=row["energy"],
                            forces=np.asarray(row["forces"]),
                            converged=row["converged"],
                            succeeded=True,
                        ),
                        oracle_row,
                    )
                )
                save(row, reference_sample=True)
                if not row["gate"]:
                    raise RuntimeError(
                        "reference replay failed its complete endpoint gate"
                    )
            del mf, mol, frozen
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
        return

    from vibeqc import Calculator

    if args.reference is None:
        parser.error("native mode requires --reference results.json")
    oracle = json.loads(args.reference.read_text())
    if (
        oracle["identity"]["protocol"] != protocol
        or oracle["identity"]["mode"] != "reference"
    ):
        raise ValueError("reference protocol/geometry differs")
    identity["reference_sha256"] = hashlib.sha256(
        args.reference.read_bytes()
    ).hexdigest()
    for name in list(os.environ):
        if name.startswith("VIBEQC_DF_"):
            del os.environ[name]
    if is_df:
        for name, value in {
            "VALUE_STORAGE": "packed-single",
            "EXCHANGE": "occupied",
            "RESPONSE_SPACE": "occupied",
            "OCCUPIED_RESPONSE_SOURCE": "fitted",
            "SOURCE_DERIVATIVE_SCHEDULE": "qualify",
            "RESPONSE_ALGEBRA": "blas",
            "RESPONSE_BUDGET_BYTES": str(
                {
                    24: 64 << 20,
                    48: 64 << 20,
                    96: 64 << 20,
                    192: 256 << 20,
                    384: 1_000_000_000,
                    768: 5_000_000_000,
                }[args.aos]
            ),
        }.items():
            os.environ["VIBEQC_DF_" + name] = value
    select_control(args.route)
    if args.disable_warm_reuse:
        os.environ["VIBEQC_DF_WARM_REUSE"] = "0"
    identity["environment"] = {
        k: v for k, v in os.environ.items() if k.startswith("VIBEQC_")
    }
    calc = Calculator(
        device="cuda",
        method="rhf",
        basis=case.vibeqc_basis,
        basis_representation="spherical",
        auxiliary_basis=native_aux if is_df else None,
        density_fitting="cuda" if is_df else "none",
        density_fitting_memory_budget_bytes=0,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        screening_tolerance=1e-12,
        max_iterations=100,
    )
    identity["native_build"] = native_build_metadata(calc)
    require_tuned_native_build(identity["native_build"])
    start = time.perf_counter()
    owner = calc.prepare_batch([original], warm_start=True)
    prepare_seconds = time.perf_counter() - start

    with owner:

        def execute(
            phase: str,
            index: int,
            route: str,
            repeat: int = 0,
            diagnostic: bool = False,
        ) -> None:
            select_control(route)
            coords = None if index == 0 else [np.asarray([x for _, x in moved])]
            trace = output / f"{phase}-{route}-{repeat}.jsonl"
            if diagnostic:
                os.environ["VIBEQC_DF_TRACE"] = str(trace)
            start = time.perf_counter()
            item = owner.execute(coords, strict=False).items[0]
            seconds = time.perf_counter() - start
            os.environ.pop("VIBEQC_DF_TRACE", None)
            row = {
                "phase": phase,
                "route": route,
                "repeat": repeat,
                "diagnostic": diagnostic,
                "seconds": seconds,
                "prepare_seconds": prepare_seconds if phase == "cold" else 0,
                "complete_seconds": seconds
                + (prepare_seconds if phase == "cold" else 0),
                "energy": item.energy,
                "forces": item.forces.tolist() if item.forces is not None else None,
                "converged": item.converged,
                "status": item.status,
                "detail": item.status_message,
                "iterations": item.iterations,
                "energy_change": item.energy_change,
                "density_rms": item.density_rms,
                "warm_start_used": item.warm_start_used,
                "warm_start_fallback": item.warm_start_fallback,
                "fock_builds": item.fock_builds,
                **check_endpoint(item, oracle["records"][index]),
            }
            if is_df:
                row["metric"] = [
                    d.to_dict() for d in owner.last_density_fitting_metric_diagnostics()
                ]
            if diagnostic:
                row["trace"] = read_trace(trace)
            save(row)
            if not row["gate"]:
                raise RuntimeError(f"independent endpoint gate failed: {phase}/{route}")

        routes = list(CONTROLS) if args.interleave else [args.route]
        for index, phase in enumerate(("cold", "moved")):
            owner.set_warm_start_updates(True)
            execute(phase, index, args.route)
            owner.set_warm_start_updates(False)
            for repeat in range(args.repeats):
                for route in routes if repeat % 2 == 0 else routes[::-1]:
                    execute(
                        "warm" if index == 0 else "moved-warm", index, route, repeat
                    )
            if is_df:
                for route in routes:
                    execute(
                        "trace-warm" if index == 0 else "trace-moved-warm",
                        index,
                        route,
                        diagnostic=True,
                    )


if __name__ == "__main__":
    main()
