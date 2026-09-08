"""Record independent grid/AO gates and complete bounded CPU/CUDA workflow costs.

GPU execution requires the caller's finite Slurm allocation. The evidence
describes grid/AO/density primitives, not DFT energies or nuclear gradients.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "python")]
import numpy as np

from tools.vibeqc_codegen.cuda_adapter import CudaCompilerAdapter
from tools.vibeqc_codegen.cuda_target import cuda_target_info
from tools.vibeqc_dft import GridSpec, MolecularGrid, NativeAO, density_features
from tools.vibeqc_dft.cuda import CudaGrid, compile_cuda
from tools.vibeqc_dft.fixtures import NAMES, basis_arguments, load_fixture
from tools.vibeqc_dft.prepared import PreparedGrid, PreparedGridBatch
from tools.vibeqc_validation.schema import (
    block_error,
    canonical_hash,
    file_hash,
    new_evidence,
    outcome,
    validate_evidence,
)


def error(actual, reference):
    """Apply CG01's elementwise absolute floor and relative scale without averaging."""
    result = block_error(actual, reference, atol=1e-11, rtol=1e-10)
    if not result["passed"]:
        raise AssertionError(result)
    return result


def fixture_gates(meta, arrays, artifact):
    """Check every derivative/invariant entry on the identical independent grid."""
    results = {}
    with NativeAO(**basis_arguments(meta)) as basis:
        jets = basis.evaluate(arrays["points"], 3)
        results["cpu/ao_jets"] = error(jets, arrays["ao_jets"])
        for k, v in density_features(jets, arrays["density"]).items():
            results[f"cpu/{k}"] = error(v, arrays[k])
        if artifact:
            with CudaGrid(basis, artifact, order=3, tile_points=31) as cuda:
                cuda.set_density(arrays["density"])
                all_results = {}
                for begin in range(0, len(arrays["points"]), 31):
                    tile = cuda.evaluate(
                        arrays["points"][begin : begin + 31], download_jets=True
                    )
                    for k, v in tile.items():
                        all_results.setdefault(k, []).append(v)
                for k, parts in all_results.items():
                    results[f"cuda/{k}"] = error(
                        np.concatenate(parts, axis=1), arrays[k]
                    )
                metrics = cuda.metrics()
                assert metrics["owned_device_bytes"] == cuda.plan.allocation_bytes
                assert metrics["provider_retained_bytes"] <= cuda.plan.provider_bytes
    return results


def convergence(meta, arrays):
    """Report raw density integrals approaching Tr(DS); never renormalize weights."""
    name = meta["inputs"]["name"]
    radius = {"tight": 0.04, "diffuse": 10}.get(name, 1.0)
    radii = tuple((z, radius) for z in sorted(set(meta["inputs"]["atomic_numbers"])))
    total = arrays["density"].sum(axis=0)
    expected = float(np.einsum("ij,ji->", total, arrays["overlap"]))
    samples = []
    with NativeAO(**basis_arguments(meta)) as basis:
        for radial, polar in ((16, 8), (32, 12), (64, 20)):
            grid = MolecularGrid(
                basis.atoms,
                GridSpec(radial, polar, 2 * polar, radii),
                multiplicity=meta["inputs"]["multiplicity"],
            )
            value = 0.0
            started = time.perf_counter()
            for tile in grid.tiles(251):
                ao = basis.evaluate(tile.points, 0)[0]
                value += float(np.dot(tile.weights, np.sum(ao * (ao @ total), axis=1)))
            samples.append(
                {
                    "spec": asdict(grid.spec),
                    "points": grid.npoint,
                    "integrated_electrons": value,
                    "absolute_error": abs(value - expected),
                    "seconds": time.perf_counter() - started,
                }
            )
    if (
        samples[-1]["absolute_error"] >= 1e-5
        or samples[-1]["absolute_error"] >= samples[0]["absolute_error"]
    ):
        raise AssertionError((name, samples))
    return {
        "trace_DS": expected,
        "samples": samples,
        "gate": 1e-5,
        "renormalization": False,
    }


def compare_fields(cpu, gpu, density):
    """Check every molecular-grid feature, with bounded independent CPU tiles."""
    errors = {}
    for a, b in zip(
        cpu.iter_features(density), gpu.iter_features(density), strict=True
    ):
        assert a.begin == b.begin
        for k in a.features:
            result = error(b.features[k], a.features[k])
            if (
                k not in errors
                or result["max_scaled_error"] > errors[k]["max_scaled_error"]
            ):
                errors[k] = result
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--nvcc", type=Path, default=Path("/group/software/cuda-12.9.1/bin/nvcc")
    )
    parser.add_argument("--cache", type=Path, default=Path("/tmp/dft160-cuda-cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--cases", nargs="+", choices=NAMES, default=list(NAMES))
    args = parser.parse_args()
    if args.samples < 5:
        parser.error("at least five interleaved samples are required")
    if args.cuda and not os.environ.get("SLURM_JOB_ID"):
        parser.error("CUDA evidence requires a finite Slurm allocation")
    args.output.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            text=True,
        )
    )
    artifact = (
        compile_cuda(
            CudaCompilerAdapter(args.nvcc, cuda_target_info("sm_120")), args.cache
        )
        if args.cuda
        else None
    )
    device = {"kind": "cpu", "name": platform.machine()}
    if artifact:
        device = {
            "kind": "cuda",
            "nvidia_smi": subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,uuid,driver_version",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip(),
        }
    source = canonical_hash(
        {
            str(p.relative_to(ROOT)): file_hash(p)
            for pattern in ("src/dft/*", "tools/vibeqc_dft/*.py")
            for p in ROOT.glob(pattern)
        }
    )
    records = []
    batch_items = []
    batch_densities = []
    for name in args.cases:
        meta, arrays = load_fixture(name)
        independent = fixture_gates(meta, arrays, artifact)
        integration = convergence(meta, arrays)
        radius = {"tight": 0.04, "diffuse": 10}.get(name, 1.0)
        radii = tuple(
            (z, radius) for z in sorted(set(meta["inputs"]["atomic_numbers"]))
        )
        spec = GridSpec(24, 12, 24, radii)
        options = {
            **basis_arguments(meta),
            "spec": spec,
            "tile_points": 251,
            "order": 1,
            "artifact": artifact,
        }
        state_hash = canonical_hash(
            {
                "fixture": meta["inputs_hash"],
                "arrays": meta["arrays"],
                "grid": asdict(spec),
            }
        )
        record = new_evidence(
            tier="endpoint", subject=f"grid/{name}", inputs_hash=state_hash
        )
        record.update(
            revision=revision,
            device=device,
            backend_selected="cuda" if artifact else "cpu",
            toolchain={
                "python": sys.version,
                "numpy": np.__version__,
                **(artifact.metadata["identity"]["toolchain"] if artifact else {}),
            },
            hardware=outcome("pass"),
        )
        record["settings"] = {
            "device": "cuda" if artifact else "cpu",
            "dirty": dirty,
            "fast_compile": False,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "grid": asdict(spec),
            "reference": "PySCF 2.14.0",
            "scope": "grid/AO/features; no DFT method or promotion",
            "placement": "host grid; CUDA AO/BLAS/invariants; explicit output D2H"
            if artifact
            else "CPU",
            "screening": "none",
            "native_library_sha256": file_hash(os.environ["VIBEQC_LIBRARY"]),
        }
        record["hashes"] = {
            "equation": canonical_hash(meta["conventions"]),
            "ir": canonical_hash(asdict(spec)),
            "source": source,
            "schedule": canonical_hash({"tile": 251, "order": 1, "ao_threads": 128}),
        }
        record["block_errors"] = independent
        record["integration_convergence"] = integration
        record["details"] = []
        # Check complete fields at fixed and genuinely changed geometry before
        # using repeated timings. The density is intentionally held constant.
        changed_coordinates = np.array(meta["inputs"]["coordinates"])
        changed_coordinates[-1] += [0.05, -0.03, 0.02]
        with PreparedGrid(**options, backend="cpu") as cpu:
            expected = cpu.integrate(arrays["density"])
            cpu.reconfigure(coordinates=changed_coordinates)
            changed_expected = cpu.integrate(arrays["density"])
        if artifact:
            with (
                PreparedGrid(**options, backend="cpu") as cpu,
                PreparedGrid(**options, backend="cuda") as gpu,
            ):
                record["full_grid_errors"] = compare_fields(cpu, gpu, arrays["density"])
                cpu.reconfigure(coordinates=changed_coordinates)
                gpu.reconfigure(coordinates=changed_coordinates)
                record["changed_grid_errors"] = compare_fields(
                    cpu, gpu, arrays["density"]
                )
        peaks = []
        allocations = []
        for trial in range(args.samples):
            backends = ["cpu", "cuda"] if artifact else ["cpu"]
            if trial % 2:
                backends.reverse()
            for backend in backends:
                started = time.perf_counter()
                with PreparedGrid(**options, backend=backend) as plan:
                    cold = plan.integrate(arrays["density"])
                    cold_seconds = time.perf_counter() - started
                    warm = plan.integrate(arrays["density"])
                    changed_started = time.perf_counter()
                    plan.reconfigure(coordinates=changed_coordinates)
                    changed = plan.integrate(arrays["density"])
                    changed_seconds = time.perf_counter() - changed_started
                    replay = plan.integrate(arrays["density"])
                    for result, reference in (
                        (cold, expected),
                        (warm, expected),
                        (changed, changed_expected),
                        (replay, changed_expected),
                    ):
                        for k in ("electrons", "integrated_tau"):
                            error(result[k], reference[k])
                    details = {
                        "trial": trial,
                        "backend": backend,
                        "cold": cold,
                        "warm": warm,
                        "changed": changed,
                        "changed_replay": replay,
                        "cold_seconds": cold_seconds,
                        "changed_seconds": changed_seconds,
                    }
                    record["details"].append(details)
                    for workload, seconds, identity in (
                        ("cold-start", cold_seconds, state_hash),
                        ("unchanged-geometry", warm["seconds"], state_hash),
                        (
                            "changed-geometry",
                            changed_seconds,
                            canonical_hash(changed_coordinates.tolist()),
                        ),
                    ):
                        record["timings"].append(
                            {
                                "selection": "candidate"
                                if backend == "cuda"
                                else "baseline",
                                "seconds": seconds,
                                "workload": workload,
                                "inputs_hash": identity,
                                "trial": trial,
                            }
                        )
                    diagnostics = plan.diagnostics()
                    peaks.append(
                        max(
                            diagnostics["peak_bytes"],
                            diagnostics["replacement_peak_bytes"],
                        )
                    )
                    if backend == "cuda":
                        metrics = diagnostics["cuda"]
                        allocations.append(metrics["owned_device_bytes"])
                        assert (
                            metrics["owned_device_bytes"] == plan.plan.allocation_bytes
                        )
                        assert (
                            metrics["provider_retained_bytes"]
                            <= plan.plan.provider_bytes
                        )
        for stage in ("representation", "source", "numerical", "endpoint"):
            record["stages"][stage] = outcome("pass")
        record["stages"]["compilation"] = outcome("pass")
        record["compilation"] = (
            {
                "seconds": artifact.metadata["compile_seconds"],
                "reason": None,
                "artifact": artifact.metadata,
            }
            if artifact
            else {"seconds": None, "reason": "CPU native build recorded separately"}
        )
        record["memory"] = {
            "allocated_bytes": max(allocations, default=0),
            "peak_bytes": max(peaks),
            "reason": None,
            "scope": "numeric buffers including transient old/replacement overlap; docs/dft_grid.md",
        }
        record["performance"] = outcome(
            "not-run", "initial interface evidence; no default schedule replacement"
        )
        record["medians"] = {
            backend: {
                workload: statistics.median(
                    t["seconds"]
                    for t in record["timings"]
                    if t["selection"] == selection and t["workload"] == workload
                )
                for workload in ("cold-start", "unchanged-geometry", "changed-geometry")
            }
            for backend, selection in (("cpu", "baseline"), ("cuda", "candidate"))
            if backend == "cpu" or artifact
        }
        validate_evidence(record)
        records.append(record)
        (args.output / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
        batch_items.append({**options, "backend": "cuda" if artifact else "cpu"})
        batch_densities.append(arrays["density"])
        print(name, "pass", record["medians"], flush=True)
    # Keep both small validation fleets live, then interleave replay samples.
    # Each fleet reports its own capacity and their simultaneous sum is explicit.
    from contextlib import ExitStack

    with ExitStack() as stack:
        fleets = {
            backend: stack.enter_context(
                PreparedGridBatch(
                    [{**item, "backend": backend} for item in batch_items]
                )
            )
            for backend in (["cpu", "cuda"] if artifact else ["cpu"])
        }
        expected = fleets["cpu"].execute(batch_densities)
        raw = []
        for trial in range(args.samples):
            order = list(fleets)
            if trial % 2:
                order.reverse()
            for backend in order:
                started = time.perf_counter()
                results = fleets[backend].execute(batch_densities)
                elapsed = time.perf_counter() - started
                assert all(r["status"] == "pass" for r in results)
                for actual, reference in zip(results, expected, strict=True):
                    for key in ("electrons", "integrated_tau"):
                        error(actual["result"][key], reference["result"][key])
                raw.append(
                    {
                        "trial": trial,
                        "backend": backend,
                        "seconds": elapsed,
                        "results": results,
                    }
                )
        report = {
            "revision": revision,
            "point_offsets": fleets["cpu"].point_offsets,
            "ao_offsets": fleets["cpu"].ao_offsets,
            "fleet_peak_bytes": {k: v.peak_bytes for k, v in fleets.items()},
            "simultaneous_validation_peak_bytes": sum(
                v.peak_bytes for v in fleets.values()
            ),
            "raw": raw,
            "scope": "sequential independent private streams; no fused batch claim",
        }
        (args.output / "batch.json").write_text(json.dumps(report, indent=2) + "\n")
    print("all grid evidence passed", flush=True)


if __name__ == "__main__":
    main()
