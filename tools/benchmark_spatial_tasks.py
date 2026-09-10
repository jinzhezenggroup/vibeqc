"""Measure bounded dense/local AO candidates and available fixed-density XC.

Run each worker in a fresh process. ``--root`` can select a historical dense
checkout; it must match the explicit CPU library. CUDA execution requires a
finite Slurm allocation. Screened arithmetic uses an identical-mask reference;
approximation differences from unscreened collocation are reported separately.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path


def capture(argv):
    """Read checked provenance without invoking a shell."""
    return subprocess.check_output(argv, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--dense-only", action="store_true")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--atoms", type=int, nargs="+", default=[4, 16])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or any(n not in (4, 16) for n in args.atoms):
        raise ValueError("positive sample count and supported atom counts required")
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.root.resolve()
    if capture(["git", "-C", str(root), "status", "--porcelain"]):
        raise ValueError("benchmark requires a clean source checkout")
    if args.backend == "cuda" and not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("real GPU execution requires finite Slurm allocation")
    sys.path[:0] = [str(root / "python"), str(root)]
    os.environ["VIBEQC_LIBRARY"] = str(args.library.resolve())
    import numpy as np
    from vibeqc import Primitive, Shell
    from vibeqc.autotune import source_identity
    from vibeqc_compiler.common.evidence import block_error, canonical_hash
    from vibeqc_compiler.common.provenance import file_hash
    from vibeqc_compiler.dft import ExplicitGrid, NativeAO
    from vibeqc_compiler.dft.features import density_features
    from vibeqc_compiler.dft.plan import plan_tiles
    from vibeqc_compiler.xc import FixedDensityXC, functional
    from vibeqc_compiler.xc.potential import assemble_potential
    from vibeqc_compiler.xc.program import build_program, pack_grid_features

    native = ctypes.CDLL(str(args.library.resolve()))
    native.vibeqc_get_source_identity.restype = ctypes.c_char_p
    identity = source_identity(root)
    if native.vibeqc_get_source_identity().decode() != identity:
        raise ValueError("CPU library and source checkout identity differ")
    artifact = None
    if args.backend == "cuda":
        from vibeqc_compiler.dft.cuda import CudaGrid, compile_cuda
        from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
        from vibeqc_compiler.integral.cuda_target import cuda_target_info

        artifact = compile_cuda(
            CudaCompilerAdapter(
                Path("/group/software/cuda-12.9.1/bin/nvcc"), cuda_target_info("sm_120")
            ),
            args.cache,
        )
    if not args.dense_only:
        from vibeqc_compiler.common.resources import ResourceBudget
        from vibeqc_compiler.dft.spatial import SpatialPolicy
        from vibeqc_compiler.dft.spatial_prepared import PreparedSpatialGrid

    report = {
        "schema": "vibeqc.spatial-task-benchmark.v1",
        "revision": capture(["git", "-C", str(root), "rev-parse", "HEAD"]),
        "dirty": False,
        "source_identity": identity,
        "library_sha256": file_hash(args.library),
        "backend": args.backend,
        "python": sys.version,
        "numpy": np.__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": capture(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version",
                "--format=csv,noheader",
            ]
        )
        if artifact
        else None,
        "artifact": None if artifact is None else artifact.metadata,
        "cases": [],
        "timing_scope": "synchronized host wall time; bounded diagnostic feature outputs; CPU complete fixed-density PBE E/V; GPU scatter is an explicit host-input diagnostic, not GPU XC",
    }
    spec = functional("PBE", spin="polarized")
    consumer, program = FixedDensityXC(spec), build_program(spec, order=1)

    def gate(actual, expected):
        error = block_error(actual, expected, atol=1e-11, rtol=1e-10)
        if not error["passed"]:
            raise AssertionError(error)
        return error

    def reference(basis, grid, density, tiles):
        """Bounded global-index oracle; zero omitted jets before nonlinear XC."""
        features = {
            "rho": np.zeros((2, len(grid.points))),
            "gradient": np.zeros((2, len(grid.points), 3)),
            "sigma": np.zeros((3, len(grid.points))),
            "tau": np.zeros((2, len(grid.points))),
        }
        energy, potential = 0.0, np.zeros_like(density)
        for ids, active in tiles:
            jets = basis.evaluate(grid.points[ids], 1)
            if active is not None:
                omitted = np.ones(basis.nao, dtype=bool)
                omitted[active] = False
                jets = jets.copy()
                jets[:, :, omitted] = 0
            f = density_features(jets, density)
            for key, value in f.items():
                features[key][:, ids] = value
            values = program.unpack(program.evaluate(pack_grid_features(spec, f)))
            energy += float(grid.weights[ids] @ values["energy_density"])
            potential += assemble_potential(
                spec, jets, f["gradient"], values["gradient"], grid.weights[ids]
            )
        return features, energy, potential

    for natom in args.atoms:
        for representation in ("cartesian", "spherical"):
            rng = np.random.default_rng(234000 + natom)
            centers = np.array([[6.0 * i, 0.1 * (i % 2), 0] for i in range(natom)])
            atoms = [("H", tuple(r)) for r in centers]
            shells = tuple(
                Shell(i, l, (Primitive(0.7, 1), Primitive(1.5, -0.1)))
                for i in range(natom)
                for l in (0, 1, 3)
            )
            points = np.concatenate(
                [r + 0.2 * rng.normal(size=(32, 3)) for r in centers]
            )
            grid = ExplicitGrid(
                points,
                rng.uniform(0.001, 0.003, len(points)),
                tuple(np.repeat(np.arange(natom), 32)),
                {},
            )
            with NativeAO(atoms, basis=shells, representation=representation) as basis:
                c = rng.normal(size=(2, basis.nao, 7)) / np.sqrt(basis.nao)
                density = c @ c.swapaxes(1, 2) + 0.1 * np.eye(basis.nao)
                for tile_points, host_budget, device_budget in (
                    (16, 32 << 20, 160 << 20),
                    (64, 64 << 20, 256 << 20),
                ):
                    dense_tiles = [
                        (np.arange(i, min(i + tile_points, len(points))), None)
                        for i in range(0, len(points), tile_points)
                    ]
                    full = reference(basis, grid, density, dense_tiles)
                    modes = (
                        ("dense",)
                        if args.dense_only
                        else ("dense", "local_off", "local_screened")
                    )
                    for mode in modes:
                        row = {
                            "case": f"{natom}/{representation}/tile{tile_points}/{mode}",
                            "natom": natom,
                            "nao": basis.nao,
                            "npoint": len(points),
                            "tile_points": tile_points,
                            "mode": mode,
                            "budgets": {
                                "host_bytes": host_budget,
                                "device_bytes": device_budget,
                            },
                            "inputs": {
                                "seed": 234000 + natom,
                                "basis": basis.identity,
                                "grid": grid.identity,
                                "density": canonical_hash(density.tolist()),
                                "functional": spec.identity,
                            },
                            "samples": [],
                        }
                        expected = full
                        for sample in range(args.samples):
                            started = time.perf_counter()
                            owner = None
                            cuda = None
                            if mode != "dense":
                                screened = mode == "local_screened"
                                policy = SpatialPolicy(
                                    region_points=32,
                                    screening="absolute_ao_jet" if screened else "off",
                                    cutoff=1e-9 if screened else 0,
                                )
                                owner = PreparedSpatialGrid(
                                    basis,
                                    grid,
                                    policy=policy,
                                    backend=args.backend,
                                    tile_points=tile_points,
                                    artifact=artifact,
                                    resource_budget=ResourceBudget(
                                        host_bytes=host_budget,
                                        device_bytes=device_budget,
                                    ),
                                )
                                cuda = owner._cuda
                                row["mask_identity"] = owner.tasks.identity
                                row["active_aos"] = [
                                    len(t.ao_ids) for t in owner.tasks.tasks
                                ]
                                row["metadata_bytes"] = owner.tasks.numeric_bytes
                                row["resource_plan"] = asdict(owner.resource_plan)
                                tiles = [(ids, t.ao_ids) for t, ids in owner._tiles()]
                                row["tile_plan"] = asdict(owner.tile_plan)
                            else:
                                tiles = dense_tiles
                                plan = plan_tiles(
                                    basis,
                                    backend=args.backend,
                                    order=1,
                                    tile_points=tile_points,
                                    budget_bytes=host_budget + device_budget,
                                )
                                row["tile_plan"] = asdict(plan)
                                row["resource_scope"] = (
                                    "Legacy dense aggregate numeric budget; no shared host/device replacement guarantee."
                                )
                                if artifact:
                                    cuda = CudaGrid(
                                        basis,
                                        artifact,
                                        order=1,
                                        tile_points=tile_points,
                                        budget_bytes=host_budget + device_budget,
                                    )
                            construction = time.perf_counter() - started
                            try:
                                # Reference construction is outside every execution clock.
                                if sample == 0 and mode != "dense":
                                    expected = reference(basis, grid, density, tiles)
                                found = {
                                    key: np.empty_like(value)
                                    for key, value in expected[0].items()
                                }
                                ao_seconds = density_seconds = 0.0
                                started = time.perf_counter()
                                if owner is not None:
                                    for tile in owner.iter_features(density):
                                        for key, value in tile.features.items():
                                            found[key][:, tile.point_ids] = value
                                else:
                                    if cuda:
                                        cuda.set_density(density)
                                    for ids, _ in dense_tiles:
                                        if cuda:
                                            f = cuda.evaluate(points[ids])
                                        else:
                                            section = time.perf_counter()
                                            jets = basis.evaluate(points[ids], 1)
                                            ao_seconds += time.perf_counter() - section
                                            section = time.perf_counter()
                                            f = density_features(jets, density)
                                            density_seconds += (
                                                time.perf_counter() - section
                                            )
                                        for key, value in f.items():
                                            found[key][:, ids] = value
                                measured = {
                                    "sample": sample,
                                    "construction_seconds": construction,
                                    "features_seconds": time.perf_counter() - started,
                                    "errors": {
                                        key: gate(value, expected[0][key])
                                        for key, value in found.items()
                                    },
                                }
                                if args.backend == "cpu" and owner is None:
                                    measured["cpu_feature_sections"] = {
                                        "ao_seconds": ao_seconds,
                                        "density_seconds": density_seconds,
                                    }
                                if args.backend == "cpu":
                                    started = time.perf_counter()
                                    xc = (
                                        consumer.integrate(
                                            basis,
                                            grid,
                                            density,
                                            tile_points=tile_points,
                                            spatial=owner,
                                        )
                                        if owner
                                        else consumer.integrate(
                                            basis,
                                            grid,
                                            density,
                                            tile_points=tile_points,
                                        )
                                    )
                                    measured["xc_seconds"] = (
                                        time.perf_counter() - started
                                    )
                                    measured["energy"] = xc.energy
                                    measured["potential_sha256"] = canonical_hash(
                                        xc.potential.tolist()
                                    )
                                    measured["errors"].update(
                                        energy=gate([xc.energy], [expected[1]]),
                                        potential=gate(xc.potential, expected[2]),
                                    )
                                if owner and cuda:
                                    # This tests the native consumer/scatter boundary. Matrices
                                    # are diagnostic input, not fabricated XC device output.
                                    before = cuda.metrics()
                                    scatter_seconds = 0.0
                                    started = time.perf_counter()
                                    want = np.zeros_like(density)
                                    with owner.device_tasks(density) as iterator:
                                        for index, (task, ids, lease) in enumerate(
                                            iterator
                                        ):
                                            active = task.ao_ids
                                            v = (1 + active) / basis.nao
                                            local = (
                                                np.stack(
                                                    (
                                                        np.outer(v, v),
                                                        0.7 * np.outer(v, v),
                                                    )
                                                )
                                                * grid.weights[ids].sum()
                                            )
                                            want[
                                                :, active[:, None], active[None, :]
                                            ] += local
                                            section = time.perf_counter()
                                            got = lease.scatter(
                                                local,
                                                reset=index == 0,
                                                download=index == len(tiles) - 1,
                                            )
                                            scatter_seconds += (
                                                time.perf_counter() - section
                                            )
                                    measured["device_consumer_scatter_seconds"] = (
                                        time.perf_counter() - started
                                    )
                                    measured["scatter_calls_seconds"] = scatter_seconds
                                    measured["errors"]["scatter"] = gate(got, want)
                                    after = cuda.metrics()
                                    measured["consumer_metric_delta"] = {
                                        k: after[k] - before[k]
                                        for k in before
                                        if k.endswith("_ms")
                                    }
                                if cuda:
                                    measured["native_metrics"] = cuda.metrics()
                                if owner:
                                    measured["cpu_sections"] = dict(owner.timings)
                                row["samples"].append(measured)
                            finally:
                                if owner:
                                    owner.close()
                                elif cuda:
                                    cuda.close()
                        row["approximation_difference"] = {
                            key: float(np.max(np.abs(expected[0][key] - value)))
                            for key, value in full[0].items()
                        }
                        row["approximation_difference"].update(
                            energy=abs(expected[1] - full[1]),
                            potential=float(np.max(np.abs(expected[2] - full[2]))),
                        )
                        report["cases"].append(row)
                        print(
                            json.dumps(
                                {
                                    "case": row["case"],
                                    "samples": args.samples,
                                    "construction_seconds": row["samples"][0][
                                        "construction_seconds"
                                    ],
                                    "features_seconds": row["samples"][0][
                                        "features_seconds"
                                    ],
                                }
                            ),
                            flush=True,
                        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    )


if __name__ == "__main__":
    main()
