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

from tools.vibeqc_cc.triples import triples_energy
from tools.vibeqc_cc.triples_cuda import (
    CudaTriplesTiles,
    TriplesTileConfig,
)
from tools.vibeqc_cc.triples_tiles import (
    TriplesTileEnumerator,
    tile_triples_energy_masked,
)

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
    return (
        nocc,
        nvir,
        g[:nocc, nocc:, nocc:, nocc:],
        g[:nocc, nocc:, :nocc, :nocc],
        g[:nocc, nocc:, :nocc, nocc:],
        fov,
        t1,
        t2,
        eps[:nocc],
        eps[nocc:],
    )


def arrays_dict(feeds):
    names = ("ovvv", "ovoo", "ovov", "fov", "t1", "t2", "eps_o", "eps_v")
    return dict(zip(names, feeds[2:]))


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
        nocc, nvir = feeds[0], feeds[1]
        assert nocc == expected_o and nvir == expected_v

        # CPU reference
        t0 = time.perf_counter()
        cpu_et = triples_energy(nocc, nvir, *feeds[2:])
        cpu_time_s = time.perf_counter() - t0

        print(f"\n{name} (o={nocc}, v={nvir})", flush=True)
        print(f"  E_T(CPU) = {cpu_et:.15e}  ({cpu_time_s:.3f}s)", flush=True)
        if name in ("h2", "he"):
            assert abs(cpu_et) < 1e-9, f"{name}: near-zero triples"
        else:
            assert abs(cpu_et - expected_et) <= 1e-9, (
                f"{name}: E_T={cpu_et} vs truth={expected_et}"
            )
            print(
                f"  |dE_T| vs ground truth = {abs(cpu_et - expected_et):.2e}",
                flush=True,
            )

        mol_record = {
            "name": name,
            "nocc": nocc,
            "nvir": nvir,
            "cpu_et": cpu_et,
            "cpu_time_s": cpu_time_s,
            "ground_truth_et": expected_et,
            "tile_results": [],
        }

        arrays = arrays_dict(feeds)

        # Test both single-tile (vir_chunk_size=nvir) and multi-tile (< nvir)
        for vir_chunk_size in (nvir, max(1, nvir // 2 if nvir > 1 else 1)):
            if vir_chunk_size == 0:
                continue
            for max_bytes in budgets:
                budget_mib = max_bytes // (1 << 20)
                label = f"  chunk={vir_chunk_size}, budget={budget_mib}MiB"
                print(f"{label} ...", flush=True, end=" ")

                config = TriplesTileConfig(
                    nocc=nocc,
                    nvir=nvir,
                    vir_chunk_size=vir_chunk_size,
                    max_bytes=max_bytes,
                )

                try:
                    if args.compile_only:
                        # Compile just one tile to verify the plan is feasible
                        from vibeqc_compiler.tensor.cuda_plan import plan_cuda
                        from vibeqc_compiler.tensor.cuda_resident import (
                            compile_resident,
                        )

                        from tools.vibeqc_cc.triples_tiles import (
                            build_tile_triples_program,
                        )

                        enum = TriplesTileEnumerator(
                            nocc, nvir, vir_chunk_size=vir_chunk_size
                        )
                        first_tile = next(iter(enum))
                        tile_prog = build_tile_triples_program(
                            nocc,
                            first_tile.a_end,
                            vir_chunk=(first_tile.a_start, first_tile.a_end),
                        )
                        plan = plan_cuda(
                            tile_prog, compiler.target, max_bytes=max_bytes
                        )
                        artifact = compile_resident(plan, compiler, cache)
                        print(
                            f"compiled peak={plan.peak_bytes // 1024}KiB "
                            f"key={artifact.metadata['key'][:16]}",
                            flush=True,
                        )
                        mol_record["tile_results"].append(
                            {
                                "vir_chunk_size": vir_chunk_size,
                                "budget_mib": budget_mib,
                                "compiled": True,
                                "peak_bytes": plan.peak_bytes,
                                "artifact_key": artifact.metadata["key"],
                                "gpu_run": None,
                            }
                        )
                        continue

                    # Full GPU run via the corrected CudaTriplesTiles
                    with CudaTriplesTiles(config, compiler, cache) as tiles:
                        t0 = time.perf_counter()
                        result = tiles.run_tiles(arrays)
                    gpu_time_s = time.perf_counter() - t0

                    # CPU masked per-tile reference
                    enum = TriplesTileEnumerator(
                        nocc, nvir, vir_chunk_size=vir_chunk_size
                    )
                    cpu_per_tile = [
                        tile_triples_energy_masked(t, nocc, *feeds[2:]) for t in enum
                    ]

                    per_tile_diffs = [
                        abs(g - c) for g, c in zip(result.per_tile, cpu_per_tile)
                    ]
                    max_tile_diff = max(per_tile_diffs) if per_tile_diffs else 0
                    de_total = abs(result.et - cpu_et)

                    de_total_ok = de_total <= 1e-9
                    per_tile_ok = all(d <= 1e-10 for d in per_tile_diffs)

                    print(
                        f"tiles={result.tile_count} "
                        f"|dE|={de_total:.2e} "
                        f"max_tile_diff={max_tile_diff:.2e} "
                        f"peak={result.peak_device_bytes // 1024}KiB "
                        f"time={gpu_time_s:.2f}s",
                        flush=True,
                    )

                    gpu_run = {
                        "vir_chunk_size": vir_chunk_size,
                        "budget_mib": budget_mib,
                        "peak_device_bytes": result.peak_device_bytes,
                        "peak_bytes_per_tile": result.peak_bytes_per_tile,
                        "artifact_keys": result.artifact_keys,
                        "gpu_et": result.et,
                        "cpu_et": cpu_et,
                        "de_total": de_total,
                        "de_total_ok": de_total_ok,
                        "per_tile_gpu": result.per_tile,
                        "per_tile_cpu_masked": cpu_per_tile,
                        "per_tile_diffs": per_tile_diffs,
                        "max_tile_diff": max_tile_diff,
                        "per_tile_ok": per_tile_ok,
                        "gpu_time_s": gpu_time_s,
                        "tile_count": result.tile_count,
                        "timing": result.timing,
                    }

                    mol_record["tile_results"].append(
                        {
                            "vir_chunk_size": vir_chunk_size,
                            "budget_mib": budget_mib,
                            "compiled": True,
                            "peak_bytes": result.peak_device_bytes,
                            "gpu_run": gpu_run,
                        }
                    )

                except ValueError as e:
                    if "infeasible" in str(e):
                        print(f"INFEASIBLE: {e}", flush=True)
                        mol_record["tile_results"].append(
                            {
                                "vir_chunk_size": vir_chunk_size,
                                "budget_mib": budget_mib,
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
        for tr in mol.get("tile_results", []):
            r = tr.get("gpu_run")
            if r is None:
                continue
            status = (
                f"chunk={tr['vir_chunk_size']} "
                f"tiles={r['tile_count']} "
                f"de={r['de_total']:.2e} "
                f"tile_diff={r['max_tile_diff']:.2e} "
                f"time={r['gpu_time_s']:.2f}s"
            )
            if not r["de_total_ok"] or not r["per_tile_ok"]:
                all_ok = False
                print(f"FAIL {mol['name']} {tr['budget_mib']}MiB: {status}", flush=True)
            else:
                print(f"PASS {mol['name']} {tr['budget_mib']}MiB: {status}", flush=True)

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPU triples-tiles validation for #150 B"
    )
    parser.add_argument("--output", required=True, help="Output directory for results")
    parser.add_argument("--cache", required=True, help="Compilation cache directory")
    parser.add_argument("--nvcc", required=True, help="Path to nvcc")
    parser.add_argument(
        "--architecture", required=True, help="CUDA architecture (e.g. sm_90)"
    )
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
