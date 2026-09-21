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
import hashlib
import json
import os
import platform
import subprocess
import sys as _compiler_sys
import time
import typing
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

ENDPOINTS_DIR = ROOT / "tests/reference_data/cc/endpoints"
GROUND_TRUTH = {
    "h2": (1, 1, 8.392021714075268e-49),
    "he": (1, 1, 0.0),
    "h2o": (5, 2, -6.731393342463869e-05),
    "nh3": (5, 3, -1.122922812723691e-04),
    "ch4": (5, 4, -1.555665872715297e-04),
}


QUALIFICATION_SOURCE_PATHS = (
    "tools/vibeqc_cc/triples.py",
    "tools/vibeqc_cc/triples_tiles.py",
    "tools/vibeqc_cc/triples_cuda.py",
    "tools/validate_cc_triples_tiles.py",
)


def _qualification_source_identity() -> typing.Any:
    """Bind retained evidence to the exact git head and orchestration bytes."""
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            text=True,
        ).strip()
    )
    hashes = {}
    for rel in QUALIFICATION_SOURCE_PATHS:
        hashes[rel] = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    return {"git_head": git_head, "worktree_dirty": dirty, "sha256": hashes}


def _bitwise_equal_float64(left: typing.Any, right: typing.Any) -> typing.Any:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    return a.shape == b.shape and np.array_equal(a.view(np.uint64), b.view(np.uint64))


def load_endpoint(name: typing.Any) -> typing.Any:
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


def arrays_dict(feeds: typing.Any) -> typing.Any:
    names = ("ovvv", "ovoo", "ovov", "fov", "t1", "t2", "eps_o", "eps_v")
    return dict(zip(names, feeds[2:]))


def run(args: typing.Any) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)

    compiler = CudaCompilerAdapter(
        Path(args.nvcc),
        cuda_target_info(args.architecture),
        compile_timeout=args.compile_timeout,
    )
    budgets = [int(b) * (1 << 20) for b in args.budget.split(",")]
    selected_molecules = tuple(
        name.strip() for name in args.molecules.split(",") if name.strip()
    )
    unknown_molecules = sorted(set(selected_molecules) - set(GROUND_TRUTH))
    if not selected_molecules or unknown_molecules:
        raise ValueError(
            "molecules must be a nonempty comma-separated subset of "
            f"{sorted(GROUND_TRUTH)}; unknown={unknown_molecules}"
        )

    source_identity = _qualification_source_identity()
    if not args.compile_only and source_identity["worktree_dirty"]:
        raise RuntimeError(
            "real-device qualification requires a clean tracked worktree so "
            "git_head binds the executed sources"
        )

    manifest = {
        "scope": "#150 B bounded CUDA triples tiles",
        "schema": "vibeqc.ccsd-t.tile-validation/1",
        "compile_only": args.compile_only,
        "tensor_source_identity": tensor_source_identity(),
        "qualification_source_identity": source_identity,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "compiler_target": compiler.target.to_payload(),
        "runtime_device": None,
        "selected_molecules": selected_molecules,
        "molecules": [],
    }

    for name in selected_molecules:
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

        # Test both single-tile (vir_chunk_size=nvir) and multi-tile (< nvir).
        # Deduplicate two-electron endpoints where both choices are 1.
        chunk_sizes = tuple(dict.fromkeys((nvir, max(1, nvir // 2 if nvir > 1 else 1))))
        for vir_chunk_size in chunk_sizes:
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
                        # Use the same budget-driven runtime-domain selection as
                        # production, but stop before device allocation/execution.
                        from vibeqc_compiler.tensor.cuda_resident import (
                            compile_resident,
                        )

                        tiles = CudaTriplesTiles(config, compiler, cache)
                        capacity, plan, capacity_attempts = tiles.plan_runtime_domain()
                        artifact = compile_resident(plan, compiler, cache)
                        print(
                            f"compiled capacity={capacity} "
                            f"peak={plan.peak_bytes // 1024}KiB "
                            f"key={artifact.metadata['key'][:16]}",
                            flush=True,
                        )
                        mol_record["tile_results"].append(
                            {
                                "vir_chunk_size": vir_chunk_size,
                                "budget_mib": budget_mib,
                                "compiled": True,
                                "runtime_domain_capacity": capacity,
                                "capacity_selection": list(capacity_attempts),
                                "peak_bytes": plan.peak_bytes,
                                "artifact_key": artifact.metadata["key"],
                                "gpu_run": None,
                            }
                        )
                        continue

                    # Full GPU run plus an independent second GPU run for the
                    # claimed bitwise-determinism gate. Only the first run
                    # enables the CPU oracle.
                    with CudaTriplesTiles(config, compiler, cache) as tiles:
                        t0 = time.perf_counter()
                        result = tiles.run_tiles(arrays, oracle=True)
                        gpu_time_s = time.perf_counter() - t0

                        t0 = time.perf_counter()
                        repeat = tiles.run_tiles(arrays, oracle=False)
                        repeat_gpu_time_s = time.perf_counter() - t0

                    if manifest["runtime_device"] is None:
                        manifest["runtime_device"] = result.runtime_device

                    cpu_per_tile = result.per_tile_masked_cpu
                    assert cpu_per_tile is not None
                    per_tile_bitwise_equal = _bitwise_equal_float64(
                        result.per_tile, repeat.per_tile
                    )
                    total_bitwise_equal = _bitwise_equal_float64(result.et, repeat.et)
                    determinism_ok = per_tile_bitwise_equal and total_bitwise_equal
                    budget_ok = all(
                        peak <= max_bytes for peak in result.peak_bytes_per_tile
                    )

                    per_tile_diffs = [
                        abs(g - c) for g, c in zip(result.per_tile, cpu_per_tile)
                    ]
                    max_tile_diff = max(per_tile_diffs) if per_tile_diffs else 0
                    de_total = abs(result.et - cpu_et)

                    de_total_ok = bool(de_total <= 1e-9)
                    per_tile_ok = bool(all(d <= 1e-10 for d in per_tile_diffs))

                    print(
                        f"tiles={result.tile_count} "
                        f"|dE|={de_total:.2e} "
                        f"max_tile_diff={max_tile_diff:.2e} "
                        f"peak={result.peak_device_bytes // 1024}KiB "
                        f"det={'yes' if determinism_ok else 'NO'} "
                        f"time={gpu_time_s:.2f}s",
                        flush=True,
                    )

                    gpu_run = {
                        "vir_chunk_size": vir_chunk_size,
                        "budget_mib": budget_mib,
                        "peak_device_bytes": result.peak_device_bytes,
                        "peak_bytes_per_tile": result.peak_bytes_per_tile,
                        "artifact_keys": result.artifact_keys,
                        "runtime_device": result.runtime_device,
                        "gpu_et": result.et,
                        "cpu_et": cpu_et,
                        "de_total": de_total,
                        "de_total_ok": de_total_ok,
                        "per_tile_gpu": result.per_tile,
                        "per_tile_cpu_masked": cpu_per_tile,
                        "per_tile_diffs": per_tile_diffs,
                        "max_tile_diff": max_tile_diff,
                        "per_tile_ok": per_tile_ok,
                        "budget_ok": budget_ok,
                        "determinism_ok": determinism_ok,
                        "per_tile_bitwise_equal": per_tile_bitwise_equal,
                        "total_bitwise_equal": total_bitwise_equal,
                        "repeat_gpu_et": repeat.et,
                        "repeat_per_tile_gpu": repeat.per_tile,
                        "repeat_artifact_keys": repeat.artifact_keys,
                        "repeat_runtime_device": repeat.runtime_device,
                        "repeat_gpu_time_s": repeat_gpu_time_s,
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
                # A budget rejection is useful diagnostic evidence, but it
                # cannot qualify a requested shape/budget as a successful run.
                # In particular, rejecting every plan must not exit zero.
                if not tr["compiled"] or not args.compile_only:
                    all_ok = False
                    reason = tr.get("infeasible_reason", "missing GPU execution")
                    print(
                        f"FAIL {mol['name']} {tr['budget_mib']}MiB "
                        f"chunk={tr['vir_chunk_size']}: {reason}",
                        flush=True,
                    )
                continue
            status = (
                f"chunk={tr['vir_chunk_size']} "
                f"tiles={r['tile_count']} "
                f"de={r['de_total']:.2e} "
                f"tile_diff={r['max_tile_diff']:.2e} "
                f"det={r['determinism_ok']} "
                f"time={r['gpu_time_s']:.2f}s"
            )
            if (
                not r["de_total_ok"]
                or not r["per_tile_ok"]
                or not r["budget_ok"]
                or not r["determinism_ok"]
            ):
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
    parser.add_argument(
        "--molecules",
        default="h2,he,h2o,nh3,ch4",
        help=(
            "Comma-separated endpoint subset. The default runs all endpoints; "
            "qualification may scope a retained run to the minimum cases needed "
            "for a stated gate, e.g. h2,h2o."
        ),
    )
    parser.add_argument(
        "--compile-timeout",
        type=float,
        default=1800.0,
        help="NVCC timeout per tile shape in seconds (default: 1800)",
    )
    args = parser.parse_args()
    run(args)
