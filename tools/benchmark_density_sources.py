"""Measure #235 GPU D/C parity and complete available fixed-density XC cost.

Use a clean checkout, fresh cache/output and a finite Slurm GPU allocation.
Saved independent fixtures are hash checked; PySCF is not imported. Test-only
Cholesky factors expose the positive-definite XC fixtures to both algorithms;
production never factorizes an arbitrary D to manufacture the orbital route.
Timings are diagnostic: native CPU XC/potential assembly and transfers remain
part of this endpoint, and this runner does not promote either algorithm.
"""

import argparse
import csv
import ctypes
import json
import os
import platform
import shutil
import subprocess
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from itertools import product
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

import numpy as np
from vibeqc.autotune import source_identity
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.cuda_adapter import resolve_cuda_execution_profile
from vibeqc_compiler.common.evidence import (
    block_error,
    new_evidence,
    outcome,
    write_evidence,
)
from vibeqc_compiler.common.provenance import canonical_hash, file_hash, find_nvcc
from vibeqc_compiler.common.resources import ResourceBudget
from vibeqc_compiler.dft import DensitySource, NativeAO
from vibeqc_compiler.dft.cuda import CudaGrid, compile_cuda
from vibeqc_compiler.dft.fixtures import NAMES, basis_arguments, load_fixture
from vibeqc_compiler.dft.spatial import SpatialPolicy
from vibeqc_compiler.dft.spatial_prepared import PreparedSpatialGrid
from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.integral.cuda_target import cuda_target_info
from vibeqc_compiler.xc import functional
from vibeqc_compiler.xc.contractions import ContractionProgram
from vibeqc_compiler.xc.integration_fixtures import CASES, load_integration_fixture
from vibeqc_compiler.xc.native import NativeContractionProgram
from vibeqc_compiler.xc.prepared import PreparedXCContractions

from tools.vibeqc_validation.hardware import (
    CUDA_BENCHMARK_PROFILES,
    qualify_cuda_device,
)

ROUTES = ("density_matrix", "orbitals")
LAYOUTS = (("total", "unpolarized"), ("total", "polarized"), ("spin", "polarized"))
BUDGETS = (128 << 20, 256 << 20)
FUNCTIONALS = ("LDA_XC_PW", "PBE")


def capture(argv):
    """Collect bounded provenance commands without shell interpolation."""
    return subprocess.check_output(argv, text=True, timeout=60, cwd=ROOT).strip()


def probe_gpu(nvcc, profile):
    """Bind provenance and capability checks to CUDA ordinal zero.

    NVML/nvidia-smi ordinals need not match remapped CUDA ordinals. Resolve
    the PCI bus ID through the selected CUDA runtime, query that device
    explicitly, and qualify it by the requested hardware profile rather than
    by a marketing product name.
    """
    runtime = ctypes.CDLL(str(nvcc.parent.parent / "lib64/libcudart.so"))
    pci_bus = runtime.cudaDeviceGetPCIBusId
    pci_bus.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    pci_bus.restype = ctypes.c_int
    bus = ctypes.create_string_buffer(32)
    status = pci_bus(bus, len(bus), 0)
    if status or not bus.value:
        raise RuntimeError(f"cannot identify the assigned CUDA device: {status}")
    bus_id = bus.value.decode("ascii")

    runtime_version = ctypes.c_int()
    get_runtime_version = runtime.cudaRuntimeGetVersion
    get_runtime_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_runtime_version.restype = ctypes.c_int
    status = get_runtime_version(ctypes.byref(runtime_version))
    if status:
        raise RuntimeError(f"cannot query the assigned CUDA runtime: {status}")

    gpu = capture(
        [
            "nvidia-smi",
            "--id=" + bus_id,
            "--query-gpu=name,uuid,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ]
    )
    rows = list(csv.reader(gpu.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 5:
        raise RuntimeError("cannot record provenance for the assigned CUDA device")
    name, uuid, driver, memory, compute_capability = (
        value.strip() for value in rows[0]
    )
    try:
        capability = tuple(int(value) for value in compute_capability.split("."))
    except ValueError as error:
        raise RuntimeError(
            "assigned CUDA device reported an invalid compute capability"
        ) from error
    qualification = qualify_cuda_device(
        {"name": name, "compute_capability": capability}, profile
    )
    return {
        "gpu": gpu,
        "name": name,
        "uuid": uuid,
        "driver_version": driver,
        "memory_total": memory,
        "compute_capability": list(capability),
        "cuda_runtime_version": runtime_version.value,
        "qualification": qualification,
        "pci_bus_id": bus_id,
        "visible_device_ordinal": 0,
    }


def gate(actual, expected):
    """Fail on any entry outside the existing FP64 gate, including near zero."""
    result = block_error(
        np.atleast_1d(actual), np.atleast_1d(expected), atol=1e-11, rtol=1e-10
    )
    if not result["passed"]:
        raise AssertionError(result)
    return result


def record_worst(errors, key, actual, expected):
    """Retain the worst scaled whole-block gate across all repeated executions."""
    error = gate(actual, expected)
    if (
        key not in errors
        or error["max_scaled_error"] >= errors[key]["max_scaled_error"]
    ):
        errors[key] = error


def checked_metrics(cuda):
    """Compare actual native arena/provider ownership with the numeric plan."""
    metrics = cuda.metrics()
    if (
        metrics["owned_device_bytes"] != cuda.plan.allocation_bytes
        or metrics["provider_retained_bytes"] > cuda.plan.provider_bytes
    ):
        raise AssertionError(
            "native CUDA allocation exceeded or differs from planned regions"
        )
    return metrics


def feature_cases(artifact, errors):
    """Use saved independent AO/rho/gradient/sigma/tau blocks with partial tiles."""
    rows = []
    for name in NAMES:
        meta, data = load_fixture(name)
        with NativeAO(**basis_arguments(meta)) as basis:
            source = DensitySource(data["density"], basis_identity=basis.identity)
            source = source.with_orbitals(
                data["coefficients"], data["occupations"], stamp=source.stamp
            )
            with CudaGrid(
                basis,
                artifact,
                order=3,
                tile_points=7,
                orbital_capacity=tuple(map(len, source.occupations)),
                orbital_tile=3,
            ) as cuda:
                for route in ROUTES:
                    cuda.set_source(source, stamp=source.stamp, route=route)
                    for begin in range(0, len(data["points"]), 7):
                        values = cuda.evaluate(
                            data["points"][begin : begin + 7],
                            stamp=source.stamp,
                            download_jets=True,
                        )
                        for key, value in values.items():
                            record_worst(
                                errors,
                                f"features/{name}/{route}/{key}",
                                value,
                                data[key][:, begin : begin + 7],
                            )
                rows.append(
                    {
                        "case": name,
                        "fixture_hash": canonical_hash(meta),
                        "source": asdict(source.stamp),
                        "factor_identity": source.factor_identity,
                        "plan": asdict(cuda.plan),
                        "native_metrics": checked_metrics(cuda),
                    }
                )
    return rows


@contextmanager
def endpoint_owner(
    basis,
    grid,
    artifact,
    native,
    ingredients,
    budget,
    mode,
    *,
    tile_points=7,
    orbital_tile=3,
    region_points=11,
    orbital_capacity=None,
):
    """Compose either existing dense or fixed-mask spatial CUDA ownership."""
    with ExitStack() as stack:
        settings = {
            "orbital_capacity": (basis.nao,) * 2
            if orbital_capacity is None
            else orbital_capacity,
            "orbital_tile": orbital_tile,
            "ingredients": ingredients,
            "resource_budget": budget,
            "tile_points": tile_points,
        }
        spatial = None
        if mode == "dense":
            cuda = stack.enter_context(
                CudaGrid(basis, artifact, order=native.contract.ao_order, **settings)
            )
            options = {"density_grid": cuda}
        else:
            spatial = stack.enter_context(
                PreparedSpatialGrid(
                    basis,
                    grid,
                    backend="cuda",
                    artifact=artifact,
                    policy=SpatialPolicy(
                        region_points=region_points,
                        screening=mode,
                        cutoff=0 if mode == "off" else 1e-4,
                    ),
                    **settings,
                )
            )
            cuda, options = spatial._cuda, {"spatial": spatial}
        endpoint = stack.enter_context(
            PreparedXCContractions(
                native, basis, grid, resource_budget=budget, **options
            )
        )
        yield cuda, endpoint, spatial


def masked_endpoint(basis, grid, source, spatial, spec, layout):
    """Global-index Python oracle at exactly the prepared fixed AO mask.

    This changes only omitted collocation columns, leaving global D and matrix
    assembly independent of the candidate's local gather/scatter. Screening
    error against the original unmasked fixture is reported separately.
    """
    oracle = ContractionProgram(spec)
    energy = 0.0
    potential = np.zeros((2 if spec.spin == "polarized" else 1, basis.nao, basis.nao))
    for task, ids in spatial._tiles():
        if not len(task.ao_ids):
            continue
        jets = basis.evaluate(grid.points[ids], oracle.contract.ao_order).copy()
        omitted = np.ones(basis.nao, dtype=bool)
        omitted[task.ao_ids] = False
        jets[:, :, omitted] = 0
        value = oracle.evaluate(jets, source.density, grid.weights[ids])
        energy += value["energy"]
        potential += value["potential"]
    return energy, potential.mean(axis=0) if layout == "total" else potential


def endpoint_cases(artifact, programs, samples, errors, timings, *, spatial=False):
    """Run five or more interleaved D/C pairs on each identical discrete model."""
    rows = []
    for case in CASES:
        meta, data, grid = load_integration_fixture(case)
        with NativeAO(**basis_arguments(meta)) as basis:
            for layout, spin in LAYOUTS:
                started = perf_counter()
                source = DensitySource(
                    data[f"density_{layout}"], basis_identity=basis.identity
                )
                source_seconds = perf_counter() - started
                started = perf_counter()
                c = tuple(np.linalg.cholesky(d) for d in source.density)
                fixture_factor_seconds = perf_counter() - started
                started = perf_counter()
                source = source.with_orbitals(
                    c, (np.ones(basis.nao),) * 2, stamp=source.stamp
                )
                validation_seconds = perf_counter() - started
                for name in FUNCTIONALS:
                    native = programs[name, spin]
                    ingredients = (
                        ("rho",)
                        if name == "LDA_XC_PW"
                        else ("rho", "gradient", "sigma")
                    )
                    modes = ("off", "absolute_ao_jet") if spatial else ("dense",)
                    for mode, cap in product(modes, BUDGETS):
                        label = f"{case}/{name}/{layout}/{spin}/{cap}"
                        if spatial:
                            label += "/" + mode
                        budget = ResourceBudget(host_bytes=32 << 20, device_bytes=cap)
                        started = perf_counter()
                        with endpoint_owner(
                            basis, grid, artifact, native, ingredients, budget, mode
                        ) as (cuda, endpoint, spatial_owner):
                            construction_seconds = perf_counter() - started
                            reference = (
                                data[f"{name}_{layout}_energy"][0],
                                data[f"{name}_{layout}_potential"],
                            )
                            mask_record = None
                            if spatial_owner is not None:
                                masked = masked_endpoint(
                                    basis,
                                    grid,
                                    source,
                                    spatial_owner,
                                    native.spec,
                                    layout,
                                )
                                if mode == "off":
                                    for got, want in zip(
                                        masked, reference, strict=True
                                    ):
                                        gate(got, want)
                                mask_record = {
                                    "identity": spatial_owner.tasks.identity,
                                    "generation_id": spatial_owner.tasks.generation_id,
                                    "policy": asdict(spatial_owner.tasks.policy),
                                    "active_ao_counts": [
                                        len(t.ao_ids) for t in spatial_owner.tasks.tasks
                                    ],
                                    "task_point_counts": [
                                        len(t.point_ids)
                                        for t in spatial_owner.tasks.tasks
                                    ],
                                    "screening_error_scope": "difference from saved unmasked fixture; not an arithmetic gate",
                                    "screening_energy_difference": abs(
                                        masked[0] - reference[0]
                                    ),
                                    "screening_potential_max_difference": float(
                                        np.max(np.abs(masked[1] - reference[1]))
                                    ),
                                }
                                reference = masked
                            # Warm both actual routes before alternating timed calls.
                            for route in ROUTES:
                                endpoint.execute(
                                    source, stamp=source.stamp, route=route
                                )
                            inputs_hash = canonical_hash(
                                {
                                    "fixture": canonical_hash(meta),
                                    "grid": grid.identity,
                                    "source": source.stamp.identity,
                                    "factor": source.factor_identity,
                                    "functional": native.contract.identity,
                                    **(
                                        {"spatial_mask": spatial_owner.tasks.identity}
                                        if spatial_owner is not None
                                        else {}
                                    ),
                                }
                            )
                            for repeat in range(samples):
                                for route in (
                                    ROUTES if repeat % 2 == 0 else ROUTES[::-1]
                                ):
                                    started = perf_counter()
                                    value = endpoint.execute(
                                        source, stamp=source.stamp, route=route
                                    )
                                    elapsed = perf_counter() - started
                                    potential = (
                                        value["potential"].mean(axis=0)
                                        if layout == "total"
                                        else value["potential"]
                                    )
                                    record_worst(
                                        errors,
                                        label + "/" + route + "/energy",
                                        value["energy"],
                                        reference[0],
                                    )
                                    record_worst(
                                        errors,
                                        label + "/" + route + "/potential",
                                        potential,
                                        reference[1],
                                    )
                                    statistics = endpoint.statistics
                                    if statistics["source"]["source_kind"] != route:
                                        raise AssertionError(
                                            "measurement silently changed the requested route"
                                        )
                                    timings.append(
                                        {
                                            "case": label,
                                            "repeat": repeat,
                                            "selection": "baseline"
                                            if route == ROUTES[0]
                                            else "candidate",
                                            "workload": "unchanged-geometry",
                                            "inputs_hash": inputs_hash,
                                            "seconds": elapsed,
                                            "diagnostics": {
                                                "route": route,
                                                "cpu_contraction_seconds": statistics[
                                                    "cpu_contraction_seconds"
                                                ],
                                                "factor_packing_seconds": statistics[
                                                    "source"
                                                ]["factor_packing_seconds"],
                                                "source_upload_seconds": statistics[
                                                    "source"
                                                ]["source_upload_seconds"],
                                                "source_upload_bytes": statistics[
                                                    "source"
                                                ]["source_upload_bytes"],
                                                "device_seconds": statistics[
                                                    "device_seconds"
                                                ],
                                                "density_matrix_products": statistics[
                                                    "density_matrix_products"
                                                ],
                                                "orbital_matrix_products": statistics[
                                                    "orbital_matrix_products"
                                                ],
                                                "matrix_assembly_products": statistics[
                                                    "matrix_assembly_products"
                                                ],
                                            },
                                        }
                                    )
                            rows.append(
                                {
                                    "case": label,
                                    "inputs_hash": inputs_hash,
                                    "fixture_hash": canonical_hash(meta),
                                    "source": asdict(source.stamp),
                                    "factor_identity": source.factor_identity,
                                    "npoint": len(grid.weights),
                                    "nao": basis.nao,
                                    "source_construction_seconds": source_seconds,
                                    "fixture_factorization_seconds": fixture_factor_seconds,
                                    "external_factor_validation_seconds": validation_seconds,
                                    "owner_construction_seconds": construction_seconds,
                                    "tile_plan": asdict(cuda.plan),
                                    "resource_plan": endpoint.resource_plan.to_dict(),
                                    "native_metrics": checked_metrics(cuda),
                                    **(
                                        {"spatial": mask_record}
                                        if mask_record is not None
                                        else {}
                                    ),
                                }
                            )
                        print(label, flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--hardware-profile",
        choices=tuple(CUDA_BENCHMARK_PROFILES),
        default="sm120",
        help=(
            "CUDA eligibility profile; rtx5090-reproduction retains the old "
            "exact-device reproduction gate"
        ),
    )
    parser.add_argument(
        "--spatial",
        action="store_true",
        help="measure prepared unscreened and screened local D/C XC",
    )
    parser.add_argument(
        "--workload-matrix",
        type=Path,
        help="hash-checked larger independent SCF-state fixtures",
    )
    args = parser.parse_args()
    hardware_profile = CUDA_BENCHMARK_PROFILES[args.hardware_profile]
    if args.workload_matrix and args.spatial:
        raise ValueError("workload matrix already declares its dense/local modes")
    execution_profile = resolve_cuda_execution_profile(
        default_slurm_time="00:30:00" if args.workload_matrix else "00:10:00"
    )
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    assigned_partition = os.environ.get("SLURM_JOB_PARTITION")
    if not execution_profile.local and (
        not slurm_job_id or not os.environ.get("CUDA_VISIBLE_DEVICES")
    ):
        raise RuntimeError(
            "run through the configured finite CUDA benchmark allocation; "
            "preserve assigned visibility"
        )
    if (
        not execution_profile.local
        and execution_profile.partition is not None
        and assigned_partition
        and assigned_partition != execution_profile.partition
    ):
        raise RuntimeError(
            "assigned Slurm partition does not match the configured benchmark profile"
        )
    if args.samples < 5 or capture(["git", "status", "--porcelain"]):
        raise ValueError("at least five samples and a clean checkout required")
    if args.output.exists() or args.cache.exists():
        raise FileExistsError("use fresh output and compilation cache paths")
    nvcc = find_nvcc()
    if nvcc is None:
        raise RuntimeError("NVCC is required")
    os.environ["VIBEQC_LIBRARY"] = str(args.library.resolve())
    library = ctypes.CDLL(str(args.library.resolve()))
    library.vibeqc_get_source_identity.restype = ctypes.c_char_p
    identity = source_identity(ROOT)
    if library.vibeqc_get_source_identity().decode() != identity:
        raise ValueError("native library/source identity mismatch; rebuild first")
    report = new_evidence(
        tier="endpoint",
        subject=(
            "#235 C3 registered larger D/C workload and replica throughput"
            if args.workload_matrix
            else "#235 C1 prepared spatial D/C density and CPU XC E/V"
            if args.spatial
            else "#235 B bounded GPU D/C density and CPU XC E/V"
        ),
        inputs_hash=canonical_hash(
            {
                **(
                    {
                        "workload_matrix": file_hash(
                            args.workload_matrix / "manifest.json"
                        )
                    }
                    if args.workload_matrix
                    else {}
                ),
                "features": NAMES,
                "endpoints": CASES,
                "functionals": FUNCTIONALS,
                "layouts": LAYOUTS,
                "budgets": BUDGETS,
                **(
                    {"spatial_modes": ("off", "absolute_ao_jet")}
                    if args.spatial
                    else {}
                ),
            }
        ),
    )
    report.update(
        revision=capture(["git", "rev-parse", "HEAD"]),
        dirty=False,
        source_identity=identity,
        library_sha256=file_hash(args.library),
        backend_selected="cuda",
    )
    report["device"] = {
        **probe_gpu(nvcc, hardware_profile),
        "host": platform.platform(),
        "slurm_job_id": slurm_job_id,
        "slurm_job": (
            capture(["scontrol", "show", "job", slurm_job_id, "--oneliner"])
            if slurm_job_id
            else None
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    report["hardware"] = outcome(
        "pass",
        probe="validated assigned CUDA hardware profile",
        profile=hardware_profile.name,
    )
    report["toolchain"] = {
        "python": sys.version,
        "numpy": np.__version__,
        "nvcc": capture([str(nvcc), "--version"]),
        "cxx": capture(["c++", "--version"]),
        "threads": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }
    report["settings"] = {
        "device": "cuda",
        "execution_profile": execution_profile.to_dict(),
        "hardware_profile": hardware_profile.name,
        "fast_compile": False,
        "samples_per_route": args.samples,
        "collocation_backend": "cuda",
        "xc_backend": "native_cpu",
        "tile_points": 256 if args.workload_matrix else 7,
        "orbital_tile": 16 if args.workload_matrix else 3,
        **(
            {
                "workload_matrix": {
                    "manifest_sha256": file_hash(
                        args.workload_matrix / "manifest.json"
                    ),
                    "runner_sha256": file_hash(
                        ROOT / "tools/density_workload_matrix.py"
                    ),
                    "region_points": 512,
                    "local_cases": ["water4_svp", "water8_svp"],
                    "screened_cutoff": 1e-4,
                    "batch_sizes": [1, 4],
                    "batch_execution": "serial independent replica states through one shared prepared owner",
                }
            }
            if args.workload_matrix
            else {}
        ),
        **(
            {
                "spatial_modes": ("off", "absolute_ao_jet"),
                "region_points": 11,
                "screened_cutoff": 1e-4,
            }
            if args.spatial
            else {}
        ),
        "scope": "fixed-density features and E/V; source validation/setup separately measured; no SCF, forces or performance promotion",
        "feature_error_scope": "worst scaled tile per feature and route",
        "endpoint_error_scope": "worst whole-block sample per energy/potential and route",
    }
    started = perf_counter()
    artifact = compile_cuda(
        CudaCompilerAdapter(
            nvcc, cuda_target_info(hardware_profile.target_architecture)
        ),
        args.cache,
    )
    report["cuda_artifact"] = artifact.metadata
    programs = {}
    report["xc_artifacts"] = {}
    for name in FUNCTIONALS:
        for spin in ("unpolarized", "polarized"):
            native = NativeContractionProgram(
                functional(name, spin=spin),
                compiler=CppCompilerAdapter(Path(shutil.which("c++"))),
                cache=args.cache,
            )
            programs[name, spin] = native
            report["xc_artifacts"][name + "/" + spin] = {
                "contract_identity": native.contract.identity,
                "artifact": native.artifact.metadata,
            }
    report["compilation"] = {
        "seconds": perf_counter() - started,
        "reason": "fresh CUDA and four native CPU XC artifacts, including code generation",
    }
    report["hashes"]["source"] = identity
    report["hashes"]["schedule"] = canonical_hash(report["settings"])
    report["hash_reasons"] = {
        "equation": "component identities retained in CUDA/XC artifacts; no combined SCF equation",
        "ir": "component generated identities retained in artifacts; no combined method IR",
    }
    for stage in ("representation", "source", "compilation"):
        report["stages"][stage] = outcome("pass")
    report["feature_cases"] = feature_cases(artifact, report["block_errors"])
    if args.workload_matrix:
        from tools.density_workload_matrix import measure_matrix, validate_matrix_errors

        report["endpoint_cases"], report["batch_cases"] = measure_matrix(
            args.workload_matrix,
            artifact,
            programs,
            args.samples,
            report["block_errors"],
            report["timings"],
        )
        endpoint_count = 36
        if len(report["timings"]) != 2 * (endpoint_count + 12) * args.samples:
            raise AssertionError("incomplete registered workload timing inventory")
        validate_matrix_errors(report["block_errors"])
    else:
        report["endpoint_cases"] = endpoint_cases(
            artifact,
            programs,
            args.samples,
            report["block_errors"],
            report["timings"],
            spatial=args.spatial,
        )
        endpoint_count = 96 if args.spatial else 48
        if (
            len(report["endpoint_cases"]) != endpoint_count
            or len(report["timings"]) != 2 * endpoint_count * args.samples
            or len(report["block_errors"]) != 60 + 4 * endpoint_count
        ):
            raise AssertionError("incomplete acceptance inventory")
    if args.spatial and not any(
        0 < n < row["nao"]
        for row in report["endpoint_cases"]
        for n in row["spatial"]["active_ao_counts"]
    ):
        raise AssertionError("spatial acceptance requires nonempty strict local maps")
    report["stages"]["numerical"] = outcome("pass", independent_feature_fixtures=6)
    report["stages"]["endpoint"] = outcome("pass", identical_grid_cases=endpoint_count)
    report["stages"]["production"] = outcome(
        "not-run",
        "native CPU SCF integration is separately validated in #302; complete forces and #168 selector promotion remain separate",
    )
    report["performance"] = outcome(
        "not-run",
        "fixed-density diagnostic samples do not establish a complete SCF/force endpoint winner",
    )
    report["memory"] = {
        "allocated_bytes": max(
            row["native_metrics"]["owned_device_bytes"]
            for row in report["endpoint_cases"]
        ),
        "peak_bytes": None,
        "reason": "native arena observations and conservative composed numeric capacities are recorded per case; no measured whole-process peak",
    }
    report["reproduction"] = {
        "command": execution_profile.wrap(
            [
                "env",
                "OMP_NUM_THREADS=1",
                "OPENBLAS_NUM_THREADS=1",
                "VIBEQC_NVCC=" + str(nvcc),
                sys.executable,
                "tools/benchmark_density_sources.py",
                *sys.argv[1:],
            ]
        )
    }
    write_evidence(args.output / "evidence.json", report)
    specification = {
        "schema": "vibeqc.benchmark-publication.v1",
        "source": {"revision": report["revision"], "dirty": False},
        "reproduction": report["reproduction"],
        "files": [{"path": "evidence.json", "role": "evidence"}],
        "archives": [],
        "decision": {
            "status": "accepted",
            "scope": "numerical",
            "reason": f"six independent feature fixtures and {endpoint_count} identical-grid/mask E/V cases pass FP64 gates with bounded native ownership; diagnostic timings make no promotion claim",
        },
    }
    (args.output / "publication.json").write_text(
        json.dumps(specification, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
