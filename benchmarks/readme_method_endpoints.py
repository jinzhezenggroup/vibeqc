"""Measure DFT and the internal CUDA CCSD(T) composition for the README.

Run this driver inside a Slurm GPU allocation. Unsupported native combinations remain
explicit records, never a substituted scientific method or backend. DFT uses
identical explicit quadrature in both engines; its timings cover SCF energy,
not forces. CC timings include fresh GPU RHF, host AO/MO work, GPU CCSD and (T).
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vibeqc import Calculator, GridSpec, KsOptions
from vibeqc_compiler.dft.grid import MolecularGrid

from benchmarks._endpoint_progress import EndpointProgress, save_record
from benchmarks._gpu4pyscf_grid import preserve_reference_grid_order
from benchmarks._support import (
    cuda_accelerator_metadata,
    environment_metadata,
    raw_output_path,
)
from benchmarks.compare_gpu4pyscf_batch import (
    _gpu_sample,
    _vibeqc_sample,
    accuracy_gate_summary,
    fixed_warm_start_policy,
    interleaved_engine_order,
    load_comparison_basis,
    native_build_metadata,
    pair_repeat_accuracy,
)
from benchmarks.readme_hf_scaling import scaling_cases

ROOT = Path(__file__).resolve().parents[1]
AUXILIARY = (
    ROOT / "benchmarks/results/issue206-practical-auxiliary/identity/cc-pvdz-jkfit.json"
)


def binary_identity(calculator: Calculator) -> dict:
    """Hash the actual loaded library without probing a GPU for CPU results."""
    library = calculator._library
    path = Path(library._name).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    library.vibeqc_get_source_identity.argtypes = []
    library.vibeqc_get_source_identity.restype = ctypes.c_char_p
    return {
        "library_sha256": digest,
        "library_path": str(path),
        "source_identity": library.vibeqc_get_source_identity().decode(),
    }


def branch_summary(samples: list[dict]) -> dict:
    """Keep branch-conditioned latencies; pooling unstable branches is invalid."""
    grouped: dict[str, list[float]] = {}
    for sample in samples:
        key = ",".join(str(c["iterations"]) for c in sample["convergence"])
        grouped.setdefault(key, []).append(sample["seconds"])
    return {
        branch: {"count": len(values), "seconds": values}
        for branch, values in grouped.items()
    }


def comparison_grid(molecular: MolecularGrid) -> tuple[Any, Any, str, dict]:
    """Export identical quadrature in larger host tiles, outside solve timers.

    The reference export's default 256-point tiles repeat Python partition loops
    excessively for 96 atoms. Larger public grid tiles preserve point order and
    arithmetic, while a geometry/spec-keyed cache shares the export across the
    direct/DF and same-grid functional cases. Neither changes native execution.
    """
    import tempfile

    import numpy as np

    started = time.perf_counter()
    cache = ROOT / ".cache/readme-grids" / (molecular.identity + ".npz")
    hit = cache.exists()
    if hit:
        with np.load(cache, allow_pickle=False) as saved:
            points, weights = saved["points"], saved["weights"]
    else:
        tiles = list(molecular.tiles(tile_points=32768))
        points = np.concatenate([tile.points for tile in tiles])
        weights = np.concatenate([tile.weights for tile in tiles])
        cache.parent.mkdir(parents=True, exist_ok=True)
        # Publish atomically so optional host precomputation cannot expose a
        # partial NPZ to a GPU job starting the same scientific workload.
        with tempfile.TemporaryDirectory(dir=cache.parent) as temporary:
            pending = Path(temporary) / "grid.npz"
            np.savez(pending, points=points, weights=weights)
            pending.replace(cache)
    if points.shape != (molecular.npoint, 3) or weights.shape != (molecular.npoint,):
        raise ValueError("cached quadrature does not match the requested grid")
    digest = hashlib.sha256(points.tobytes() + weights.tobytes()).hexdigest()
    return (
        points,
        weights,
        digest,
        {
            "seconds": time.perf_counter() - started,
            "cache_hit": hit,
            "tile_points": 32768,
            "molecular_grid_identity": molecular.identity,
            "array_identity": "sha256(float64 points bytes followed by weights bytes)",
        },
    )


def run_dft(args: argparse.Namespace, record: dict) -> None:
    """Benchmark the common discrete KS energy and preserve unavailable routes."""
    import cupy as cp
    from pyscf import dft, gto

    case = scaling_cases()[f"water-{args.atoms}"]
    record["environment"]["accelerator"] = cuda_accelerator_metadata(cp)
    method = args.method + "-rks"
    # Hybrid production grid policy is not promoted. An explicit v1 grid is
    # therefore required; PBE and r2SCAN retain their own current API defaults.
    options = KsOptions(grid=GridSpec()) if args.method == "pbe0" else KsOptions()
    resolver = Calculator(
        method=method, basis="def2-svp", device="cpu", ks_options=options
    )
    points, weights, grid_identity, export = comparison_grid(
        MolecularGrid(case.atoms, spec=resolver.ks_options.grid)
    )
    record.update(
        method=method,
        atoms=args.atoms,
        aos=args.atoms * 8,
        basis="def2-SVP spherical",
        mode=args.mode,
        endpoint="SCF energy",
        grid=asdict(resolver.ks_options.grid),
        grid_points=len(weights),
        grid_identity=grid_identity,
        grid_export=export,
        native_build=binary_identity(resolver),
        warm_start_policy=fixed_warm_start_policy(),
    )
    mol = gto.M(atom=case.atoms, unit="Bohr", basis="def2-svp", cart=False, verbose=0)
    engine = dft.RKS(mol)
    engine.xc = {"pbe": "PBE", "pbe0": "PBE0", "r2scan": "R2SCAN"}[args.method]
    native_aux = None
    if args.mode == "df":
        native_aux, reference_aux = load_comparison_basis(
            AUXILIARY, case, role="auxiliary", compute_forces=False
        )
        engine = engine.density_fit(auxbasis=reference_aux)
        record["auxiliary_basis"] = {
            "name": "cc-pVDZ-JKFIT",
            "sha256": hashlib.sha256(AUXILIARY.read_bytes()).hexdigest(),
        }
    engine = engine.to_gpu()
    engine.grids.coords = cp.asarray(points)
    engine.grids.weights = cp.asarray(weights)
    preserve_reference_grid_order(engine)
    record["reference_strict_grid_order"] = True
    record["reference_empty_ao_block_adapter"] = "one identically zero AO row"
    engine.small_rho_cutoff = 0
    engine.conv_tol = 1e-10
    engine.conv_tol_grad = 1e-8
    engine.direct_scf_tol = 1e-14
    engine.direct_scf = args.mode != "df"
    engine.max_cycle = 100
    progress = EndpointProgress(
        args.output, record, args.endpoint_seconds, args.trace_scf
    )
    progress.instrument_reference(engine, cp)
    record["diagnostic_stage_synchronization"] = args.trace_scf
    batch = None
    if args.reference_only:
        record["native_not_measured"] = "reference-only run requested"
    else:
        try:
            started = time.perf_counter()
            calculator = Calculator(
                method=method,
                basis="def2-svp",
                basis_representation="spherical",
                device="cuda",
                ks_options=options,
                density_fitting="cuda" if args.mode == "df" else "none",
                auxiliary_basis=native_aux,
                max_iterations=100,
                energy_tolerance=1e-10,
                density_tolerance=1e-9,
                screening_tolerance=1e-12,
            )
            batch = calculator.prepare_batch([case.atoms], warm_start=True)
            record["native_prepare_seconds"] = time.perf_counter() - started
            record["native_build"].update(native_build_metadata(calculator))
        except (NotImplementedError, RuntimeError) as error:
            hybrid_rejection = (
                args.method == "pbe0"
                and isinstance(error, NotImplementedError)
                and str(error)
                == "VIBEQC error 3: requested capability is not implemented"
            )
            # The C ABI reports only its generic status when native KS admission
            # rejects a global hybrid on CUDA. Keep this exception scoped to PBE0;
            # the same error from a supported PBE/r2SCAN route must remain a failure.
            if not hybrid_rejection and not any(
                message in str(error)
                for message in (
                    "DFT supports conventional Coulomb only",
                    "scaled/global-hybrid KS currently requires CPU",
                )
            ):
                raise
            record["native_unavailable"] = f"{type(error).__name__}: {error}"
            if hybrid_rejection:
                record["native_unavailable"] += (
                    " (CUDA global-hybrid KS is not promoted)"
                )
    try:
        if batch is not None:
            with progress.measure("native/cold"):
                record["native_cold"] = _vibeqc_sample(batch, cp, -2, False)
            batch.set_warm_start_updates(False)
        # Cold reference has no dm0. Subsequent samples use one frozen post-cold
        # density, including an unmeasured priming call on both engines.
        from benchmarks.compare_gpu4pyscf_batch import (
            GpuCycleTracker,
            gpu_convergence_payload,
        )

        tracker = GpuCycleTracker()
        engine.callback = tracker
        with progress.measure("reference/cold"):
            cp.cuda.Stream.null.synchronize()
            started = time.perf_counter()
            energy = float(engine.kernel())
            cp.cuda.Stream.null.synchronize()
            record["reference_cold"] = {
                "seconds": time.perf_counter() - started,
                "convergence": gpu_convergence_payload([engine], [tracker]),
                "energies_hartree": [energy],
                "forces_hartree_per_bohr": None,
            }
            for state in record["reference_cold"]["convergence"]:
                state["warm_start"]["used"] = False
        if not engine.converged:
            raise RuntimeError(
                "reference cold SCF did not converge; warm timing stopped"
            )
        dm0 = engine.make_rdm1().copy()
        record["priming"] = {}
        with progress.measure("reference/priming"):
            record["priming"]["reference"] = _gpu_sample([engine], [dm0], cp, -1, False)
        if batch is not None:
            with progress.measure("native/priming"):
                record["priming"]["native"] = _vibeqc_sample(batch, cp, -1, False)
        native, reference = [], []
        record["native_samples"], record["reference_samples"] = native, reference
        for index, name in enumerate(interleaved_engine_order(args.repeats)):
            if name == "vibeqc" and batch is not None:
                with progress.measure(f"native/repeat/{index}"):
                    native.append(_vibeqc_sample(batch, cp, index, False))
            elif name == "gpu4pyscf":
                with progress.measure(f"reference/repeat/{index}"):
                    reference.append(_gpu_sample([engine], [dm0], cp, index, False))
        record["branches"] = {
            "vibeqc": branch_summary(native),
            "gpu4pyscf": branch_summary(reference),
        }
        if native:
            # Include cold and priming in the gate, in addition to all repeats.
            pairs = pair_repeat_accuracy(
                [record["native_cold"], record["priming"]["native"], *native],
                [record["reference_cold"], record["priming"]["reference"], *reference],
            )
            record["accuracy"] = accuracy_gate_summary(pairs)
            record["accuracy_pairs"] = pairs
        record["status"] = "measured" if native else "reference_only"
    finally:
        if batch is not None:
            batch.close()


def run_cc(args: argparse.Namespace, record: dict) -> None:
    """Time the internal CUDA composition, including its documented host work.

    RHF, resident CCSD and bounded (T) execute on CUDA. The existing small-system
    snapshot bridge canonicalizes on the host; the conventional integral
    provider also owns CPU AO/MO preparation. Both are inside the endpoint timer.
    Independent committed PySCF energies are validation inputs only.
    """
    import shutil

    import cupy as cp
    from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
    from vibeqc_compiler.common.cuda_target import cuda_target_info

    from tools.cc_gradient_fixtures import source_arguments
    from tools.vibeqc_cc import SolverOptions, rccsd_t_energy
    from tools.vibeqc_posthf.export import export_rhf
    from tools.vibeqc_posthf.providers import ConventionalProvider
    from tools.vibeqc_posthf.sources import NativeSource

    fixture = ROOT / f"tests/reference_data/cc/gradients/{args.molecule}.json"
    reference = json.loads(fixture.read_text())
    inputs = reference["inputs"]
    triples_file = ROOT / "tests/reference_data/cc/rccsd-t.json"
    triples_reference = next(
        row["et_ground_truth"]
        for row in json.loads(triples_file.read_text())["molecules"]
        if row["name"] == args.molecule
    )
    expected = reference["total_energy"] + triples_reference
    compiler = CudaCompilerAdapter(
        Path(shutil.which("nvcc") or "nvcc"), cuda_target_info("sm_120")
    )
    cache = Path(os.environ.get("README_CC_CACHE", ".cache/readme-ccsd-t-sm120"))
    options = SolverOptions(
        max_iterations=150, energy_tolerance=1e-12, residual_tolerance=1e-10
    )
    identity_calc = Calculator(method="rhf", device="cuda")
    record.update(
        method="CCSD(T)",
        device="cuda",
        molecule=args.molecule,
        basis="STO-3G",
        atoms=list(zip(inputs["atomic_numbers"], inputs["coordinates"], strict=True)),
        endpoint="internal CUDA RHF + resident CCSD + bounded (T), including CPU canonicalization and AO/MO preparation",
        native_public=False,
        native_build=binary_identity(identity_calc)
        | native_build_metadata(identity_calc),
        input_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
        triples_reference_sha256=hashlib.sha256(triples_file.read_bytes()).hexdigest(),
        reference_energy_hartree=expected,
        reference_triples_hartree=triples_reference,
        warm_start_policy="fresh RHF, provider and amplitudes per call; compiled artifacts retained",
        solver_options=asdict(options),
        triples_virtual_chunk=1,
    )
    record["environment"]["accelerator"] = cuda_accelerator_metadata(cp)

    def native() -> dict:
        cp.cuda.Stream.null.synchronize()
        started = time.perf_counter()
        with NativeSource(**source_arguments(inputs)) as source:
            snapshot, hf = export_rhf(
                source, backend="cuda", max_iterations=200, tolerance=1e-11
            )
            with ConventionalProvider(
                snapshot, source, budget_bytes=64 << 20
            ) as provider:
                result = rccsd_t_energy(
                    snapshot,
                    provider,
                    backend="cuda-resident",
                    options=options,
                    compiler=compiler,
                    cache=cache,
                    vir_chunk_size=1,
                    triples_oracle=False,
                )
                work = dict(provider.statistics)
        cp.cuda.Stream.null.synchronize()
        elapsed = time.perf_counter() - started
        if not result.converged:
            raise RuntimeError(f"CUDA CCSD(T) did not converge: {result.reason}")
        if (
            hf["backend"] != "cuda"
            or result.provenance["ccsd_backend"] != "cuda-resident"
            or result.provenance["triples_backend"] != "cuda-bounded-tiles"
        ):
            raise RuntimeError(
                "CCSD(T) benchmark did not execute all three GPU solver stages"
            )
        return {
            "seconds": elapsed,
            "energy": result.total_energy,
            "converged": True,
            "scf_iterations": hf["iterations"],
            "ccsd_iterations": result.ccsd.iterations,
            "triples_energy": result.triples_energy,
            "nocc": snapshot.nocc,
            "nvir": snapshot.nmo - snapshot.nocc,
            "virtual_triples": result.provenance["virtual_triple_count"],
            "replay_singles_residual_max": result.ccsd.final_r1_max,
            "replay_doubles_residual_max": result.ccsd.final_r2_max,
            "provider_work": work,
            "hf_export": hf,
            "provenance": result.provenance,
        }

    record["native_cold"] = native()
    record["native_samples"] = [native() for _ in range(args.repeats)]
    all_samples = [record["native_cold"], *record["native_samples"]]
    record["maximum_energy_error_hartree"] = max(
        abs(row["energy"] - expected) for row in all_samples
    )
    record["maximum_triples_error_hartree"] = max(
        abs(row["triples_energy"] - triples_reference) for row in all_samples
    )
    record["status"] = "measured"


def validate_record(record: dict) -> None:
    """Reject unconverged or inaccurate endpoints before publishing a pass."""
    samples = []
    for engine in ("native", "reference"):
        if record.get(engine + "_cold"):
            samples += [record[engine + "_cold"], *record[engine + "_samples"]]
    samples += list(record.get("priming", {}).values())
    for sample in samples:
        energy = sample.get("energies_hartree", [sample.get("energy")])
        if not all(value is not None and math.isfinite(value) for value in energy):
            raise RuntimeError("endpoint energy is not finite")
        if not math.isfinite(sample["seconds"]) or sample["seconds"] <= 0:
            raise RuntimeError("endpoint duration must be finite and positive")
        convergence = sample.get("convergence", [sample])
        if not all(item["converged"] for item in convergence):
            raise RuntimeError("endpoint failed the all-sample convergence gate")
    error = record.get("maximum_energy_error_hartree")
    if record.get("accuracy"):
        error = record["accuracy"]["maximum_energy_error_hartree"]
    if record.get("native_samples") and error is None:
        raise RuntimeError("native timing requires an independent energy comparison")
    if error is not None and (
        not math.isfinite(error)
        or error > record["gates"]["maximum_energy_error_hartree"]
    ):
        raise RuntimeError(f"endpoint energy error {error} exceeds the acceptance gate")
    triples_error = record.get("maximum_triples_error_hartree")
    if triples_error is not None and (
        not math.isfinite(triples_error) or triples_error > 1e-10
    ):
        raise RuntimeError("CUDA (T) correction exceeds the independent 1e-10 Eh gate")
    record["convergence_gate_passed"] = True
    # A reference-only series can establish convergence and latency, but cannot
    # certify absent native/reference numerical agreement.
    record["gates_passed"] = True if error is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=("pbe", "pbe0", "r2scan", "ccsd(t)"), required=True
    )
    parser.add_argument("--mode", choices=("direct", "df"), default="direct")
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Measure the GPU reference independently of a stopped or unavailable native route",
    )
    parser.add_argument("--atoms", type=int, choices=(3, 6, 12, 24, 48, 96), default=3)
    parser.add_argument("--molecule", choices=("h2o", "nh3", "ch4"), default="h2o")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--endpoint-seconds",
        type=float,
        default=120,
        help="Hard per-solve deadline, independent of the Slurm whole-job limit",
    )
    parser.add_argument(
        "--trace-scf",
        action="store_true",
        help="Diagnostic cycle/stage trace; stage synchronization affects timing",
    )
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("at least two repeats are required for branch reporting")
    if not math.isfinite(args.endpoint_seconds) or args.endpoint_seconds <= 0:
        parser.error("--endpoint-seconds must be finite and positive")
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("GPU endpoints require a Slurm GPU allocation")
    record: dict[str, Any] = {
        "schema": "vibeqc.readme-endpoint.v1",
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arguments": vars(args) | {"output": str(args.output)},
        "environment": environment_metadata(
            distributions={
                "numpy": ("numpy",),
                "pyscf": ("pyscf",),
                "gpu4pyscf": ("gpu4pyscf-cuda12x",),
                "cupy": ("cupy-cuda12x",),
            }
        ),
        "cpu": platform.processor(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "threads": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "gates": {"maximum_energy_error_hartree": 1e-8, "all_samples_converged": True},
    }
    try:
        (run_cc if args.method == "ccsd(t)" else run_dft)(args, record)
        validate_record(record)
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        save_record(args.output, record)
        print(
            json.dumps(
                {
                    k: record[k]
                    for k in (
                        "status",
                        "error",
                        "accuracy",
                        "native_unavailable",
                        "maximum_energy_error_hartree",
                    )
                    if k in record
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
