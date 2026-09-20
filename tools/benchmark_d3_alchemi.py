#!/usr/bin/env python3
"""Compare VibeQC D3(BJ) against NVIDIA ALCHEMI Toolkit-Ops.

The benchmark deliberately reports NVIDIA neighbor-list construction, D3-only,
and combined pipeline latency separately. VibeQC currently has no explicit
neighbor-list stage, so its synchronous warm execute latency is reported as one
endpoint. The output records precision and cutoff semantics to prevent an FP64
VibeQC result from being presented as directly equivalent to ALCHEMI's FP32 D3
outputs.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

BOHR_PER_ANGSTROM = 1.8897261254578281


def _repository_revision() -> str:
    root = Path(__file__).resolve().parents[1]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


@dataclass(frozen=True)
class Workload:
    atoms_per_system: int
    batch_size: int

    @property
    def total_atoms(self) -> int:
        return self.atoms_per_system * self.batch_size


def make_system(atom_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic nonperiodic mixed-element geometry in bohr."""
    if not 1 <= atom_count <= 4096:
        raise ValueError("atoms_per_system must be in [1, 4096]")
    side = int(np.ceil(atom_count ** (1.0 / 3.0)))
    spacing = 3.2
    xyz = np.empty((atom_count, 3), dtype=np.float64)
    for atom in range(atom_count):
        i = atom % side
        j = (atom // side) % side
        k = atom // (side * side)
        xyz[atom] = (i * spacing, j * spacing, k * spacing)
    xyz -= xyz.mean(axis=0, keepdims=True)
    elements = np.asarray((6, 1, 8, 7, 9, 16, 17, 1), dtype=np.int32)
    numbers = np.resize(elements, atom_count).copy()
    return numbers, xyz


def parse_workload(value: str) -> Workload:
    """Parse NATOMSxBATCH, for example 32x64."""
    try:
        atoms, batch = (int(piece) for piece in value.lower().split("x", 1))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("workload must be NATOMSxBATCH") from error
    if not 1 <= atoms <= 4096 or batch < 1:
        raise argparse.ArgumentTypeError("NATOMS must be 1..4096 and BATCH >= 1")
    return Workload(atoms, batch)


def _median_timed(
    function: Callable[[], Any], *, warmup: int, repeats: int, synchronize=None
) -> tuple[float, list[float]]:
    for _ in range(warmup):
        function()
    if synchronize is not None:
        synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        if synchronize is not None:
            synchronize()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), samples


def _cutoff_method(method: str, cutoff_bohr: float):
    from vibeqc_compiler.method import METHOD_CATALOG

    base = METHOD_CATALOG[method]
    if base.dispersion is None:
        raise ValueError(f"{method} does not contain a D3 correction")
    correction = replace(
        base.dispersion,
        cn_cutoff=cutoff_bohr,
        pair_cutoff=cutoff_bohr,
        pair_switch_width=0.0,
    )
    return replace(base, dispersion=correction)


def _damping(method: str) -> dict[str, float]:
    from vibeqc_compiler.method import METHOD_CATALOG

    spec = METHOD_CATALOG[method].dispersion
    if spec is None:
        raise ValueError(f"{method} does not contain a D3 correction")
    return {"s6": spec.s6, "s8": spec.s8, "a1": spec.a1, "a2": spec.a2}


def benchmark_vibeqc(
    workload: Workload,
    *,
    method: str,
    cutoff_bohr: float,
    device: str,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    from vibeqc import D3CorrectionBatch

    numbers, positions = make_system(workload.atoms_per_system)
    systems = [(numbers, positions) for _ in range(workload.batch_size)]
    model = _cutoff_method(method, cutoff_bohr)
    start = time.perf_counter()
    batch = D3CorrectionBatch(model, systems, device=device)
    prepare_seconds = time.perf_counter() - start
    try:
        median, samples = _median_timed(batch.execute, warmup=warmup, repeats=repeats)
        results = batch.execute()
        diagnostic = asdict(batch.diagnostic())
    finally:
        batch.close()
    return {
        "implementation": "vibeqc",
        "precision": "fp64",
        "prepare_seconds": prepare_seconds,
        "execute_median_seconds": median,
        "execute_samples_seconds": samples,
        "atoms_per_second": workload.total_atoms / median,
        "systems_per_second": workload.batch_size / median,
        "energies_hartree": [result.energy for result in results],
        "gradients_hartree_per_bohr": [result.gradient.tolist() for result in results],
        "candidate_undirected_pairs": (
            workload.batch_size
            * workload.atoms_per_system
            * (workload.atoms_per_system - 1)
            // 2
        ),
        "diagnostic": diagnostic,
    }


def _load_alchemi_params(path: Path, torch, device: str):
    if not path.exists():
        raise FileNotFoundError(
            f"ALCHEMI parameter cache not found: {path}. Generate it with the "
            "NVIDIA ALCHEMI Toolkit-Ops dispersion example before benchmarking."
        )
    raw = torch.load(path, map_location="cpu", weights_only=True)
    required = ("rcov", "r4r2", "c6ab", "cn_ref")
    missing = [name for name in required if name not in raw]
    if missing:
        raise ValueError(f"ALCHEMI parameter cache is missing: {missing}")
    return {name: raw[name].to(device=device) for name in required}


def benchmark_alchemi(
    workload: Workload,
    *,
    method: str,
    cutoff_bohr: float,
    device: str,
    warmup: int,
    repeats: int,
    params_path: Path,
    neighbor_method: str | None,
    position_dtype: str,
) -> dict[str, Any]:
    import torch
    from nvalchemiops.torch.interactions.dispersion import dftd3
    from nvalchemiops.torch.neighbors import neighbor_list

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("ALCHEMI requested CUDA but this PyTorch build has no CUDA")
    torch_device = torch.device(device)
    dtype = torch.float64 if position_dtype == "float64" else torch.float32
    numbers_np, positions_np = make_system(workload.atoms_per_system)
    positions = torch.as_tensor(
        np.tile(positions_np, (workload.batch_size, 1)),
        dtype=dtype,
        device=torch_device,
    )
    numbers = torch.as_tensor(
        np.tile(numbers_np, workload.batch_size), dtype=torch.int32, device=torch_device
    )
    batch_idx = torch.repeat_interleave(
        torch.arange(workload.batch_size, dtype=torch.int32, device=torch_device),
        workload.atoms_per_system,
    )
    batch_ptr = (
        torch.arange(workload.batch_size + 1, dtype=torch.int32, device=torch_device)
        * workload.atoms_per_system
    )
    params = _load_alchemi_params(params_path, torch, device)
    damping = _damping(method)

    def build_neighbors():
        kwargs = {
            "positions": positions,
            "cutoff": cutoff_bohr,
            "batch_idx": batch_idx if workload.batch_size > 1 else None,
            "batch_ptr": batch_ptr if workload.batch_size > 1 else None,
            "return_neighbor_list": True,
        }
        if neighbor_method is not None:
            kwargs["method"] = neighbor_method
        return neighbor_list(**kwargs)

    def unpack_neighbors(result):
        if len(result) == 2:
            edge_index, ptr = result
            return edge_index, ptr, None
        if len(result) == 3:
            edge_index, ptr, unit_shifts = result
            return edge_index, ptr, unit_shifts
        raise RuntimeError(
            f"unexpected ALCHEMI neighbor-list result arity: {len(result)}"
        )

    edges, neighbor_ptr, shifts = unpack_neighbors(build_neighbors())

    def evaluate(edge_index, ptr, unit_shifts):
        kwargs = {
            "positions": positions,
            "numbers": numbers,
            "batch_idx": batch_idx if workload.batch_size > 1 else None,
            "neighbor_list": edge_index,
            "neighbor_ptr": ptr,
            "num_systems": workload.batch_size,
            "d3_params": params,
            **damping,
        }
        if unit_shifts is not None:
            kwargs["unit_shifts"] = unit_shifts
        return dftd3(**kwargs)

    def run_d3():
        return evaluate(edges, neighbor_ptr, shifts)

    def run_pipeline():
        new_edges, new_ptr, new_shifts = unpack_neighbors(build_neighbors())
        return evaluate(new_edges, new_ptr, new_shifts)

    synchronize = torch.cuda.synchronize if device == "cuda" else None
    neighbor_median, neighbor_samples = _median_timed(
        build_neighbors, warmup=warmup, repeats=repeats, synchronize=synchronize
    )
    d3_median, d3_samples = _median_timed(
        run_d3, warmup=warmup, repeats=repeats, synchronize=synchronize
    )
    pipeline_median, pipeline_samples = _median_timed(
        run_pipeline, warmup=warmup, repeats=repeats, synchronize=synchronize
    )
    energy, forces, _ = run_d3()
    if synchronize is not None:
        synchronize()
    return {
        "implementation": "nvidia-alchemi-toolkit-ops",
        "version": __import__("nvalchemiops").__version__,
        "positions_precision": position_dtype,
        "outputs_precision": "fp32",
        "neighbor_method": neighbor_method or "auto",
        "neighbor_edges": int(edges.shape[1]),
        "neighbor_median_seconds": neighbor_median,
        "neighbor_samples_seconds": neighbor_samples,
        "d3_only_median_seconds": d3_median,
        "d3_only_samples_seconds": d3_samples,
        "pipeline_median_seconds": pipeline_median,
        "pipeline_samples_seconds": pipeline_samples,
        "d3_only_atoms_per_second": workload.total_atoms / d3_median,
        "pipeline_atoms_per_second": workload.total_atoms / pipeline_median,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device_name": (
            torch.cuda.get_device_name(torch_device) if device == "cuda" else "cpu"
        ),
        "energies_hartree": energy.detach().cpu().double().tolist(),
        "gradients_hartree_per_bohr": (-forces.detach().cpu().double()).tolist(),
    }


def _error_summary(vibeqc: dict[str, Any], alchemi: dict[str, Any]) -> dict[str, float]:
    lhs_e = np.asarray(vibeqc["energies_hartree"], dtype=np.float64)
    rhs_e = np.asarray(alchemi["energies_hartree"], dtype=np.float64)
    lhs_g = np.asarray(vibeqc["gradients_hartree_per_bohr"], dtype=np.float64)
    rhs_g = np.asarray(alchemi["gradients_hartree_per_bohr"], dtype=np.float64)
    if any(
        value.ndim not in (2, 3) or value.shape[-1] != 3 for value in (lhs_g, rhs_g)
    ):
        raise ValueError("gradients require an explicit Cartesian axis of length three")
    lhs_g, rhs_g = lhs_g.reshape(-1, 3), rhs_g.reshape(-1, 3)
    if (
        lhs_e.shape != rhs_e.shape
        or lhs_e.ndim != 1
        or lhs_g.shape != rhs_g.shape
        or not lhs_e.size
        or not lhs_g.size
        or any(not np.isfinite(value).all() for value in (lhs_e, rhs_e, lhs_g, rhs_g))
    ):
        raise ValueError("comparison requires finite outputs with matching shapes")
    return {
        "max_abs_energy_hartree": float(np.max(np.abs(lhs_e - rhs_e))),
        "max_abs_gradient_hartree_per_bohr": float(np.max(np.abs(lhs_g - rhs_g))),
    }


def _compact_scientific_outputs(result: dict[str, Any]) -> None:
    energies = np.asarray(result.pop("energies_hartree"), dtype=np.float64)
    gradients = np.asarray(result.pop("gradients_hartree_per_bohr"), dtype=np.float64)
    result["energy_sum_hartree"] = float(np.sum(energies))
    result["max_abs_gradient_hartree_per_bohr"] = float(np.max(np.abs(gradients)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workload",
        action="append",
        type=parse_workload,
        default=None,
        help="repeatable NATOMSxBATCH workload (default: 32x1, 128x1, 32x32)",
    )
    parser.add_argument(
        "--method", choices=("PBE-D3(BJ)", "PBE0-D3(BJ)"), default="PBE-D3(BJ)"
    )
    parser.add_argument("--cutoff-angstrom", type=float, default=15.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--alchemi-params",
        type=Path,
        default=Path("~/.cache/nvalchemiops/dftd3_parameters.pt").expanduser(),
    )
    parser.add_argument(
        "--alchemi-neighbor-method",
        choices=("auto", "naive", "cell_list", "cluster_tile"),
        default="auto",
    )
    parser.add_argument(
        "--alchemi-position-dtype", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument("--vibeqc-only", action="store_true")
    parser.add_argument("--alchemi-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--vibeqc-library", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.vibeqc_only and args.alchemi_only:
        raise SystemExit("--vibeqc-only and --alchemi-only are mutually exclusive")
    if args.cutoff_angstrom <= 0 or args.warmup < 0 or args.repeats < 1:
        raise SystemExit("cutoff/repeat arguments are invalid")
    if args.vibeqc_library is not None:
        os.environ["VIBEQC_LIBRARY"] = str(args.vibeqc_library.resolve())
    workloads = args.workload or [Workload(32, 1), Workload(128, 1), Workload(32, 32)]
    cutoff_bohr = args.cutoff_angstrom * BOHR_PER_ANGSTROM
    neighbor_method = (
        None if args.alchemi_neighbor_method == "auto" else args.alchemi_neighbor_method
    )
    output: dict[str, Any] = {
        "schema": "vibeqc.d3_alchemi_benchmark.v1",
        "host": platform.node(),
        "python": platform.python_version(),
        "vibeqc_revision": _repository_revision(),
        "vibeqc_library": os.environ.get("VIBEQC_LIBRARY", "auto"),
        "method": args.method,
        "cutoff_angstrom": args.cutoff_angstrom,
        "cutoff_bohr": cutoff_bohr,
        "timing": "synchronized wall-clock warm-call latency",
        "precision_note": (
            "VibeQC production D3 is FP64. ALCHEMI Toolkit-Ops 0.4.x uses FP32 "
            "reference parameters and FP32 energy/force/CN outputs; timing is a "
            "performance reference, not an equal-precision winner claim."
        ),
        "cases": [],
    }
    for workload in workloads:
        case: dict[str, Any] = {"workload": asdict(workload)}
        if not args.alchemi_only:
            case["vibeqc"] = benchmark_vibeqc(
                workload,
                method=args.method,
                cutoff_bohr=cutoff_bohr,
                device=args.device,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        if not args.vibeqc_only:
            case["alchemi"] = benchmark_alchemi(
                workload,
                method=args.method,
                cutoff_bohr=cutoff_bohr,
                device=args.device,
                warmup=args.warmup,
                repeats=args.repeats,
                params_path=args.alchemi_params,
                neighbor_method=neighbor_method,
                position_dtype=args.alchemi_position_dtype,
            )
        if "vibeqc" in case and "alchemi" in case:
            case["numerical_delta"] = _error_summary(case["vibeqc"], case["alchemi"])
        for implementation in ("vibeqc", "alchemi"):
            if implementation in case:
                _compact_scientific_outputs(case[implementation])
        output["cases"].append(case)
        print(json.dumps(case, indent=2))
    rendered = json.dumps(output, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered)


if __name__ == "__main__":
    main()
