"""GPU triples-tiles validation for #150 B.
Run on qz (inspire) with access to NVCC and CUDA.

Usage (on qz):
  export SCRATCH=/path/to/scratch/issue-150-b
  source $SCRATCH/venv/bin/activate
  export VIBEQC_TENSOR_ARCH=sm_90
  export VIBEQC_NVCC=/usr/local/cuda/bin/nvcc
  export PYTHONPATH=$SCRATCH/repo/python:$SCRATCH/repo
  python tools/validate_cc_triples_tiles.py --output $SCRATCH/results --cache $SCRATCH/cache \\
      --nvcc /usr/local/cuda/bin/nvcc --architecture sm_90

Options:
  --compile-only   Only compile (no GPU execution needed)
  --gpu-run        Full GPU execution + numerical validation
  --budget N       Memory budget in MiB for boundedness tests (default: 256,512)
"""

import argparse
import json
import os
import platform
import sys as _compiler_sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, Path(os.environ.get("SCRATCH", "")) / "repo"):
    if (candidate / "tools/vibeqc_cc").is_dir():
        ROOT = candidate.resolve()
        break

_compiler_sys.path.insert(0, str(ROOT / "python"))

import numpy as np
from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.integral.cuda_target import cuda_target_info
from vibeqc_compiler.tensor.cuda_execute import tensor_source_identity
from vibeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda
from vibeqc_compiler.tensor.cuda_resident import PreparedResident, compile_resident

from tools.vibeqc_cc.triples import triples_energy
from tools.vibeqc_cc.triples_tiles import (
    TriplesTileEnumerator,
    build_tile_triples_program,
    tile_triples_energy,
    tile_triples_energy_masked,
)
from tools.vibeqc_validation.schema import file_hash

ENDPOINTS_DIR = ROOT / "tests/reference_data/cc/endpoints"
GROUND_TRUTH = {
    "h2": (1, 1, 8.392021714075268e-49),
    "he": (1, 1, 0.0),
    "h2o": (5, 2, -6.731393342463869e-05),
    "nh3": (5, 3, -1.122922812723691e-04),
    "ch4": (5, 4, -1.555665872715297e-04),
}


def load_endpoint(name):
    with np.load(ENDPOINTS_DIR / f"{name}.npz", allow_pickle=False) as data:
        eps = data["eps"]
        occ = data["occ"]
        C = data["C"]
        F = data["F"]
        g = data["g"]
        t1 = data["t1"]
        t2 = data["t2"]
    nocc = int(np.sum(occ > 0))
    nvir = len(eps) - nocc
    fov = (C.T @ F @ C)[:nocc, nocc:]
    return {
        "nocc": nocc,
        "nvir": nvir,
        "ovvv": g[:nocc, nocc:, nocc:, nocc:],
        "ovoo": g[:nocc, nocc:, :nocc, :nocc],
        "ovov": g[:nocc, nocc:, :nocc, nocc:],
        "fov": fov,
        "t1": t1,
        "t2": t2,
        "eps_o": eps[:nocc],
        "eps_v": eps[nocc:],
    }


def run_tiles_gpu(feeds, tile_prog, resident, enumerator):
    """Evaluate all tiles through the resident owner, returning per-tile scalars."""
    nocc = feeds["nocc"]
    nvir = feeds["nvir"]
    ovvv = feeds["ovvv"]
    ovoo = feeds["ovoo"]
    ovov = feeds["ovov"]
    fov = feeds["fov"]
    t1 = feeds["t1"]
    t2 = feeds["t2"]
    eps_o = feeds["eps_o"]
    eps_v = feeds["eps_v"]

    per_tile = []
    per_tile_metrics = []
    et = 0.0

    for tile in enumerator:
        a_end = tile.a_end
        sub_feeds = {
            "ovvv": np.ascontiguousarray(ovvv[:, :a_end, :a_end, :a_end]),
            "ovoo": np.ascontiguousarray(ovoo[:, :a_end, :, :]),
            "ovov": np.ascontiguousarray(ovov[:, :a_end, :, :a_end]),
            "fov": np.ascontiguousarray(fov[:, :a_end]),
            "t1": np.ascontiguousarray(t1[:, :a_end]),
            "t2": np.ascontiguousarray(t2[:, :, :a_end, :a_end]),
            "eps_o": eps_o,
            "eps_v": eps_v[:a_end],
        }
        resident.upload(sub_feeds)
        leases, metrics = resident.run(profile=False)
        et_tile = float(resident.download(leases["triples_energy"])[()])
        per_tile.append(et_tile)
        per_tile_metrics.append(
            {
                "a_start": tile.a_start,
                "a_end": tile.a_end,
                "device_ms": metrics.get("device_ms", 0),
                "host_seconds": metrics.get("host_seconds", 0),
            }
        )
        et += et_tile
    return et, per_tile, per_tile_metrics


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)

    compiler = CudaCompilerAdapter(args.nvcc, cuda_target_info(args.architecture))
    budgets = [int(b) * (1 << 20) for b in args.budget.split(",")]

    manifest = {
        "scope": "#150 B bounded CUDA triples tiles",
        "schema": "vibeqc.ccsd-t.tile-validation/1",
        "tensor_source_identity": tensor_source_identity(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "compiler_target": compiler.target.to_payload(),
        "runtime_device": None,
        "molecules": [],
    }

    for name in ("h2", "he", "h2o", "nh3", "ch4"):
        expected_o, expected_v, expected_et = GROUND_TRUTH[name]
        feeds = load_endpoint(name)
        nocc, nvir = feeds["nocc"], feeds["nvir"]
        assert nocc == expected_o and nvir == expected_v

        # CPU reference
        t0 = time.perf_counter()
        cpu_et = triples_energy(
            nocc,
            nvir,
            feeds["ovvv"],
            feeds["ovoo"],
            feeds["ovov"],
            feeds["fov"],
            feeds["t1"],
            feeds["t2"],
            feeds["eps_o"],
            feeds["eps_v"],
        )
        cpu_time_s = time.perf_counter() - t0

        print(f"\n{name} (o={nocc}, v={nvir})", flush=True)
        print(f"  E_T(CPU) = {cpu_et:.15e}  ({cpu_time_s:.3f}s)", flush=True)
        if name in ("h2", "he"):
            assert abs(cpu_et) < 1e-9, f"{name}: near-zero triples"
        else:
            assert abs(cpu_et - expected_et) <= 1e-9, (
                f"{name}: E_T={cpu_et} vs truth={expected_et}"
            )
            print(f"  |dE_T| vs ground truth = {abs(cpu_et - expected_et):.2e}", flush=True)

        mol_record = {
            "name": name,
            "nocc": nocc,
            "nvir": nvir,
            "cpu_et": cpu_et,
            "cpu_time_s": cpu_time_s,
            "ground_truth_et": expected_et,
            "budgets": [],
        }

        # Tile enumerator for this molecule
        enumerator = TriplesTileEnumerator(nocc, nvir, vir_chunk_size=nvir)

        for max_bytes in budgets:
            budget_mib = max_bytes // (1 << 20)
            print(f"  budget={budget_mib} MiB ...", flush=True, end=" ")

            try:
                # Build and plan the tile program
                tile_prog = build_tile_triples_program(nocc, nvir)
                plan = plan_cuda(
                    tile_prog,
                    compiler.target,
                    max_bytes=max_bytes,
                    schedule=TensorSchedule(),
                )
                print(f"plan_ok peak={plan.peak_bytes//1024} KiB", flush=True, end=" ")

                if args.compile_only:
                    artifact = compile_resident(plan, compiler, cache)
                    print(f"compiled {artifact.metadata['key'][:16]}", flush=True)
                    mol_record["budgets"].append(
                        {
                            "budget_mib": budget_mib,
                            "peak_bytes": plan.peak_bytes,
                            "compiled": True,
                            "artifact_key": artifact.metadata["key"],
                            "gpu_run": None,
                        }
                    )
                    continue

                # Full GPU run
                artifact = compile_resident(plan, compiler, cache)
                if manifest["runtime_device"] is None:
                    manifest["runtime_device"] = {
                        "architecture": args.architecture,
                    }

                t0 = time.perf_counter()
                with PreparedResident(plan, artifact, device=0) as resident:
                    # CPU reference per-tile with masking
                    enumerator_1 = TriplesTileEnumerator(nocc, nvir, vir_chunk_size=nvir)
                    cpu_per_tile = []
                    for t in enumerator_1:
                        cpu_per_tile.append(
                            tile_triples_energy_masked(
                                t, nocc,
                                feeds["ovvv"], feeds["ovoo"], feeds["ovov"],
                                feeds["fov"], feeds["t1"], feeds["t2"],
                                feeds["eps_o"], feeds["eps_v"],
                            )
                        )

                    # Warm-up run
                    gpu_et, gpu_per_tile, tile_metrics = run_tiles_gpu(
                        feeds, tile_prog, resident, enumerator
                    )

                    # Second run for determinism
                    gpu_et2, gpu_per_tile2, _ = run_tiles_gpu(
                        feeds, tile_prog, resident, enumerator
                    )
                gpu_time_s = time.perf_counter() - t0

                # Numerical checks
                de_total = abs(gpu_et - cpu_et)
                de_total_ok = de_total <= 1e-9

                per_tile_diffs = [
                    abs(g - c) for g, c in zip(gpu_per_tile, cpu_per_tile)
                ]
                per_tile_ok = all(d <= 1e-10 for d in per_tile_diffs)
                max_tile_diff = max(per_tile_diffs) if per_tile_diffs else 0

                deterministic = all(
                    abs(a - b) == 0 for a, b in zip(gpu_per_tile, gpu_per_tile2)
                )

                print(
                    f"|dE|={de_total:.2e} "
                    f"max_tile_diff={max_tile_diff:.2e} "
                    f"det={deterministic}",
                    flush=True,
                )

                gpu_run = {
                    "budget_mib": budget_mib,
                    "peak_bytes": plan.peak_bytes,
                    "artifact_key": artifact.metadata["key"],
                    "gpu_et": gpu_et,
                    "gpu_et_run2": gpu_et2,
                    "cpu_et": cpu_et,
                    "de_total": de_total,
                    "de_total_ok": de_total_ok,
                    "per_tile_gpu": gpu_per_tile,
                    "per_tile_cpu_masked": cpu_per_tile,
                    "per_tile_diffs": per_tile_diffs,
                    "max_tile_diff": max_tile_diff,
                    "per_tile_ok": per_tile_ok,
                    "deterministic": deterministic,
                    "gpu_time_s": gpu_time_s,
                    "tile_metrics": tile_metrics,
                    "tile_count": len(gpu_per_tile),
                }

                mol_record["budgets"].append(
                    {
                        "budget_mib": budget_mib,
                        "peak_bytes": plan.peak_bytes,
                        "compiled": True,
                        "artifact_key": artifact.metadata["key"],
                        "gpu_run": gpu_run,
                    }
                )

            except ValueError as e:
                if "infeasible" in str(e):
                    print(f"INFEASIBLE: {e}", flush=True)
                    mol_record["budgets"].append(
                        {
                            "budget_mib": budget_mib,
                            "peak_bytes": None,
                            "compiled": False,
                            "infeasible_reason": str(e),
                            "gpu_run": None,
                        }
                    )
                else:
                    raise

        manifest["molecules"].append(mol_record)

    # Write results
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    print(f"\nResults written to {output}/manifest.json", flush=True)

    # Summary
    all_ok = True
    for mol in manifest["molecules"]:
        for b in mol.get("budgets", []):
            if b.get("gpu_run"):
                r = b["gpu_run"]
                status = (
                    f"de={r['de_total']:.2e}"
                    f" tiles={r['max_tile_diff']:.2e}"
                    f" det={r['deterministic']}"
                )
                if not r["de_total_ok"] or not r["per_tile_ok"] or not r["deterministic"]:
                    all_ok = False
                    print(f"FAIL {mol['name']} {b['budget_mib']}MiB: {status}", flush=True)
                else:
                    print(f"PASS {mol['name']} {b['budget_mib']}MiB: {status}", flush=True)

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU triples-tiles validation for #150 B")
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--cache", required=True, help="Compilation cache directory")
    parser.add_argument("--nvcc", required=True, help="Path to nvcc")
    parser.add_argument("--architecture", required=True, help="CUDA architecture (e.g. sm_90)")
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Only compile, no GPU execution",
    )
    parser.add_argument(
        "--budget",
        default="256,512",
        help="Comma-separated memory budgets in MiB (default: 256,512)",
    )
    args = parser.parse_args()
    run(args)