"""Benchmark the OMol25 functional/basis on the README's nested water clusters.

Separate GPU4PySCF and native processes use one offline spherical basis and
moving quadrature. This matches the HF endpoint/frozen-density protocol, not
OMol25's ORCA implementation, molecular distribution, or integration grid.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import numpy as np
from generativeqc.elements import atomic_number
from generativeqc_compiler.dft.nonlocal_policy import (
    MOLECULAR_VV10_DENSITY_POLICY,
    MOLECULAR_VV10_DENSITY_THRESHOLD,
)

from benchmarks._support import environment_metadata, raw_output_path
from benchmarks.compare_df_direct_endpoint import check_endpoint, reference_work_counter
from benchmarks.compare_gpu4pyscf_batch import (
    gpu_convergence_payload,
    load_comparison_basis,
    native_build_metadata,
    require_tuned_native_build,
)
from benchmarks.readme_hf_scaling import scaling_cases
from benchmarks.readme_wb97mv import reference_engine, reference_vv10_domain

SIZES = (3, 6, 12, 24, 48, 96)
SCHEMA = "generativeqc.readme-omol25.v3"


@dataclass(frozen=True)
class EndpointSpec:
    """Scientific identity for runners sharing the HF complete-endpoint protocol."""

    method: str
    basis: str
    schema: str
    reference_full_fock: bool = False

    @property
    def native_method(self) -> str:
        """Use the registered RKS identifier, not a bare XC functional alias."""
        return f"{self.method}-rks"

    @property
    def has_vv10(self) -> bool:
        """Only the range-separated OMol25 composition needs nonlocal screening."""
        return self.method == "wb97m-v"


OMOL25 = EndpointSpec("wb97m-v", "def2-TZVPD", SCHEMA)


def source_hashes() -> dict[str, str]:
    """Identify dirty consumers and the native schedule behind the binary hash."""
    root = Path(__file__).resolve().parents[1]
    paths = (
        "python/generativeqc/calculator.py",
        "python/generativeqc/ks.py",
        "python/generativeqc/_ks_snapshot.py",
        "python/generativeqc/_snapshot_grid_cache.py",
        "python/generativeqc/resources_ks.py",
        "python/generativeqc/_stationary_cuda.py",
        "python/generativeqc/batch.py",
        "python/generativeqc_compiler/method/stationary_resources.py",
        "python/generativeqc_compiler/method/stationary_composite_resources.py",
        "python/generativeqc/_stationary_composite_cuda.py",
        "benchmarks/readme_omol25.py",
        "benchmarks/readme_pbe0.py",
        "benchmarks/readme_wb97mv.py",
        "benchmarks/compare_df_direct_endpoint.py",
        "src/scf/cuda/direct_jk.cpp",
        "src/scf/cuda/direct_jk_kernels.cu",
        "src/scf/cuda/direct_jk_kernels.hpp",
        "src/scf/cuda/direct_jk_plan.hpp",
        "src/scf/cuda/one_electron_gradient_bridge.cu",
        "src/scf/cuda/basis_transform_kernels.cu",
        "src/scf/cuda/basis_transform_kernels.hpp",
        "python/generativeqc_compiler/integral/direct_source_contraction_cuda.py",
    )
    return {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths
    }


def protocol(
    atoms: int,
    basis: Any,
    spec: Any,
    repeats: int,
    *,
    benchmark: EndpointSpec = OMOL25,
) -> dict[str, Any]:
    """Describe inputs without loading a native library or touching the GPU."""
    original = scaling_cases()[f"water-{atoms}"].atoms
    moved = [
        (
            number,
            tuple(np.asarray(coords) + ((0, 0, 0.001) if index == 1 else (0, 0, 0))),
        )
        for index, (number, coords) in enumerate(original)
    ]
    aos = sum(
        sum(
            (2 * shell.angular_momentum + 1) * len(shell.coefficients)
            for shell in basis.by_element[atomic_number(number)].shells
        )
        for number, _ in original
    )
    return json.loads(
        json.dumps(
            {
                "case": f"water-{atoms}",
                "atoms": atoms,
                "aos": aos,
                "method": f"{benchmark.method.upper()}/RKS",
                "basis": benchmark.basis,
                "basis_identity": basis.identity,
                "representation": "spherical",
                "geometries_bohr": [original, moved],
                "grid": asdict(spec),
                "properties": ["energy", "forces"],
                "force_return": "host_array_in_timed_endpoint",
                "density_fitting": False,
                **(
                    {
                        "nonlocal_density_policy": MOLECULAR_VV10_DENSITY_POLICY,
                        "nonlocal_density_threshold": float(
                            MOLECULAR_VV10_DENSITY_THRESHOLD
                        ),
                        "reference_nonlocal_weight_threshold": 1e-14,
                    }
                    if benchmark.has_vv10
                    else {}
                ),
                "repeats": repeats,
                "energy_tolerance": 1e-12,
                "density_tolerance": 1e-10,
                "reference_gradient_tolerance": 1e-10,
                "screening_tolerance": 1e-12,
                "reference_direct_scf_tolerance": 1e-14,
                "reference_fock_policy": (
                    "full-density-rebuild"
                    if benchmark.reference_full_fock
                    else "incremental"
                ),
                "max_iterations": 100,
                "energy_gate": 1e-8,
                "force_gate": 1e-7,
                "cold_definition": "preparation plus first synchronized energy/force endpoint",
                "warm_definition": f"{repeats} fixed engine-local post-cold/post-move density replays",
                "threads": {
                    name: os.environ.get(name)
                    for name in (
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                    )
                },
            }
        )
    )


def check_record(row: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    """Reuse HF's finite-value, shape, convergence and complete-force gates."""
    return check_endpoint(
        SimpleNamespace(
            energy=row["energy"],
            forces=np.asarray(row["forces"]),
            converged=row["converged"],
            succeeded=row["status"] == 0,
        ),
        oracle,
    )


def require_force_endpoint(capabilities: dict[str, Any]) -> None:
    """Honor the candidate's public capability, without hard-coding an SPD gate.

    Generic integral or internal geometry admission does not qualify the public
    force endpoint. Qualified f-shell candidates must not be rejected here.
    """
    if "forces" not in capabilities["supported_properties"]:
        raise NotImplementedError(
            "the native Calculator does not advertise analytic forces for this basis; "
            "no HF-equivalent energy-plus-force endpoint is available"
        )


def native_fock_observation(item: Any) -> dict[str, Any]:
    """Read the solver's counter, including failed solves, without using iterations.

    KS publishes its physical-build count in its own diagnostic; the legacy
    top-level batch getter may be unavailable. Keep the origin explicit and
    preserve missing measurements as null rather than inferring work.
    """
    diagnostic = getattr(item, "ks_diagnostic", None)
    count = getattr(diagnostic, "fock_builds", None)
    source = "ks_diagnostic.fock_builds"
    if count is None:
        count = getattr(item, "fock_builds", None)
        source = "fock_builds"
    return {
        "fock_builds": count,
        "fock_builds_source": source if count is not None else None,
    }


def reference_xc_backend(engine: Any, *, spin: int = 0) -> dict[str, Any]:
    """Read the same cached XCfun flags used by GPU4PySCF after the timer.

    GPU4PySCF 1.8.1 falls back for the whole semilocal expression if any
    component lacks CUDA LibXC support; a mixed list is not a mixed execution
    backend. Missing flags remain unknown, never evidence of CPU fallback.
    This describes semilocal evaluation, not exact exchange or VV10 kernels.
    """
    record = {
        "xc_code": engine.xc,
        "spin": spin,
        "backend": "unknown",
        "components": [],
    }
    try:
        functions = engine._numint._init_xcfuns(engine.xc, spin)
        for function, coefficient in functions:
            on_gpu = getattr(function, "on_gpu", None)
            record["components"].append(
                {
                    "functional_id": int(function.func_id),
                    "coefficient": float(coefficient),
                    "on_gpu": on_gpu if type(on_gpu) is bool else None,
                }
            )
    except (AttributeError, TypeError, ValueError, NotImplementedError) as error:
        record["unavailable_reason"] = f"{type(error).__name__}: {error}"
        return record
    flags = [component["on_gpu"] for component in record["components"]]
    if not flags:
        record["backend"] = "no-semilocal-xc"
    elif any(flag is False for flag in flags):
        record["backend"] = "pyscf-cpu-libxc"
    elif all(flag is True for flag in flags):
        record["backend"] = "cuda-libxc"
    return record


def main(benchmark: EndpointSpec = OMOL25) -> None:
    """Run one independent engine with shared inputs and no production switches."""
    """Journal every phase before GPU work, including failures and timeouts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=("reference", "native"))
    parser.add_argument("--atoms", type=int, choices=SIZES, required=True)
    parser.add_argument("--basis-file", type=Path, required=True)
    parser.add_argument("--grid", type=int, nargs=3, default=(48, 16, 32))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("GPU work requires finite srun on main with --gres=gpu:5090:1")
    if args.repeats < 1 or min(args.grid) < 1:
        parser.error("repeats and grid dimensions must be positive")
    if args.engine == "native" and args.reference is None:
        parser.error("native execution requires the independent --reference JSON")

    import cupy as cp
    from generativeqc import Calculator, GridSpec, KsOptions

    from benchmarks._support import cuda_accelerator_metadata
    from benchmarks.dft_force_components import normalize_force_work

    case = scaling_cases()[f"water-{args.atoms}"]
    basis, reference_basis = load_comparison_basis(
        args.basis_file, case, role="orbital", compute_forces=True
    )
    if basis.name.lower() != benchmark.basis.lower():
        raise ValueError(
            f"this benchmark requires the complete {benchmark.basis} basis"
        )
    spec = GridSpec(
        radial_points=args.grid[0],
        angular_polar=args.grid[1],
        angular_azimuth=args.grid[2],
    )
    scientific = protocol(args.atoms, basis, spec, args.repeats, benchmark=benchmark)
    oracle = None
    if args.engine == "native" and args.reference.exists():
        oracle = json.loads(args.reference.read_text())
    record: dict[str, Any] = {
        "schema": benchmark.schema,
        "engine": args.engine,
        "status": "running",
        "protocol": scientific,
        "records": [],
        "source_file_sha256": source_hashes(),
        "native_schedule_policy": "automatic-generated-SPD/canonical-through-f"
        if benchmark.has_vv10
        else "automatic-generated-SPD",
        "basis_file_sha256": hashlib.sha256(args.basis_file.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest()
        if args.reference and args.reference.exists()
        else None,
        "environment": environment_metadata(
            distributions={
                "pyscf": ("pyscf",),
                "gpu4pyscf": ("gpu4pyscf-cuda12x",),
                "cupy": ("cupy-cuda12x",),
                "cutensor": ("cutensor-cu12",),
                "numpy": ("numpy",),
            }
        ),
        "scheduler": {
            name: os.environ.get(name)
            for name in ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_GPUS_ON_NODE")
        },
    }
    record["environment"]["accelerator"] = cuda_accelerator_metadata(cp)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(stage: str) -> None:
        record["stage"] = stage
        args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    def retain(row: dict[str, Any], reference: dict[str, Any]) -> None:
        row.update(check_record(row, reference))
        record["records"].append(row)
        save(record["stage"])
        if not row["gate"]:
            raise RuntimeError(f"independent energy/force gate failed: {row['phase']}")

    owner = None
    try:
        save("setup")
        geometries = scientific["geometries_bohr"]
        if args.engine == "reference":
            density = None
            for geometry_index, geometry in enumerate(geometries):
                phase = "cold" if geometry_index == 0 else "moved"
                save(f"{phase}/prepare")
                cp.cuda.Stream.null.synchronize()
                started = perf_counter()
                engine = reference_engine(
                    geometry,
                    reference_basis,
                    spec,
                    xc=benchmark.method.upper().replace("-", "_"),
                    full_fock=benchmark.reference_full_fock,
                )
                from gpu4pyscf.dft import libxc
                from gpu4pyscf.lib import cutensor

                record["reference_tensor_engine"] = (
                    "cuTENSOR" if cutensor.cutensor is not None else "CuPy einsum"
                )
                record["reference_libxc"] = libxc.__version__
                engine.conv_tol = 1e-12
                engine.conv_tol_grad = 1e-10
                engine.direct_scf_tol = 1e-14
                engine.max_cycle = 100
                cp.cuda.Stream.null.synchronize()
                prepare_seconds = perf_counter() - started
                if engine.mol.nao_nr() != scientific["aos"]:
                    raise ValueError("native and reference basis dimensions differ")

                def execute_reference(
                    phase: str,
                    repeat: int,
                    seed: Any,
                    prepare: float = 0,
                    *,
                    engine: Any = engine,
                    geometry_index: int = geometry_index,
                ) -> dict[str, Any]:
                    """Bind the geometry owner; density copies remain inside the timer."""
                    save(f"{phase}/{repeat}")
                    cp.cuda.Stream.null.synchronize()
                    with (
                        reference_work_counter(engine) as work,
                        (
                            reference_vv10_domain(
                                scientific["nonlocal_density_threshold"]
                            )
                            if benchmark.has_vv10
                            else nullcontext()
                        ) as domain,
                    ):
                        started = perf_counter()
                        energy = float(
                            engine.kernel(dm0=None if seed is None else seed.copy())
                        )
                        if not engine.converged:
                            raise RuntimeError("GPU4PySCF SCF did not converge")
                        gradient = engine.nuc_grad_method()
                        gradient.grid_response = True
                        forces = cp.asnumpy(-gradient.kernel())
                        cp.cuda.Stream.null.synchronize()
                        seconds = perf_counter() - started
                        convergence = gpu_convergence_payload(
                            [engine], [work["tracker"]]
                        )[0]
                    return {
                        "phase": phase,
                        "geometry": geometry_index,
                        "repeat": repeat,
                        "energy": energy,
                        "forces": forces.tolist(),
                        "converged": bool(engine.converged),
                        "status": 0,
                        "iterations": convergence["iterations"],
                        "scf_jk_builds": work["scf_jk_builds"],
                        "scf_final_residuals": convergence["final_residuals"],
                        "warm_start_used": seed is not None,
                        "reference_xc_backend": reference_xc_backend(engine),
                        **({"reference_vv10_domain": domain} if domain else {}),
                        "seconds": seconds,
                        "prepare_seconds": prepare,
                        "complete_seconds": seconds + prepare,
                    }

                baseline = execute_reference(phase, 0, density, prepare_seconds)
                retain(baseline, baseline)
                density = engine.make_rdm1().copy()
                record.setdefault("grid_points", []).append(
                    {
                        "semilocal": len(engine.grids.weights),
                        **(
                            {"vv10": len(engine.nlcgrids.weights)}
                            if benchmark.has_vv10
                            else {}
                        ),
                    }
                )
                for repeat in range(args.repeats):
                    retain(
                        execute_reference(
                            "warm" if geometry_index == 0 else "moved-warm",
                            repeat,
                            density,
                        ),
                        baseline,
                    )
        else:
            save("native/calculator")
            calc = Calculator(
                method=benchmark.native_method,
                basis=basis,
                device="cuda",
                basis_representation="spherical",
                ks_options=KsOptions(grid=spec),
                energy_tolerance=1e-12,
                density_tolerance=1e-10,
                screening_tolerance=1e-12,
                max_iterations=100,
            )
            record["native_build"] = native_build_metadata(calc)
            require_tuned_native_build(record["native_build"])
            record["native_capabilities"] = {
                "supported_properties": sorted(calc.capabilities.supported_properties),
                "maximum_basis_angular_momentum": max(
                    shell.angular_momentum
                    for element in basis.elements
                    for shell in element.shells
                ),
            }
            save("native/capability")
            require_force_endpoint(record["native_capabilities"])
            if (
                oracle is None
                or oracle["protocol"] != scientific
                or oracle["status"] != "measured"
            ):
                raise ValueError(
                    "independent reference protocol differs or is incomplete"
                )
            save("cold/prepare")
            cp.cuda.Stream.null.synchronize()
            started = perf_counter()
            owner = calc.prepare_batch([geometries[0]], warm_start=True)
            force_work = None
            original_force = owner._public_dft_cuda_force

            def observed_force(*values: Any) -> Any:
                """Retain already-produced work; default owners have no resource ledger."""
                nonlocal force_work
                result = original_force(*values)
                force_work = result[1]
                return result

            owner._public_dft_cuda_force = observed_force
            cp.cuda.Stream.null.synchronize()
            prepare_seconds = perf_counter() - started
            for geometry_index in range(2):
                baseline = next(
                    row
                    for row in oracle["records"]
                    if row["geometry"] == geometry_index
                    and row["phase"] in ("cold", "moved")
                )
                coords = (
                    None
                    if geometry_index == 0
                    else [np.asarray([xyz for _, xyz in geometries[1]])]
                )
                owner.set_warm_start_updates(True)
                for repeat in range(args.repeats + 1):
                    phase = (
                        ("cold" if geometry_index == 0 else "moved")
                        if repeat == 0
                        else ("warm" if geometry_index == 0 else "moved-warm")
                    )
                    save(f"{phase}/{max(0, repeat - 1)}")
                    cp.cuda.Stream.null.synchronize()
                    started = perf_counter()
                    force_work = None
                    item = owner.execute(
                        coords, strict=False, properties=("energy", "forces")
                    ).items[0]
                    cp.cuda.Stream.null.synchronize()
                    seconds = perf_counter() - started
                    prepare = prepare_seconds if phase == "cold" else 0
                    row = {
                        "phase": phase,
                        "geometry": geometry_index,
                        "repeat": max(0, repeat - 1),
                        "energy": item.energy,
                        "forces": item.forces.tolist()
                        if item.forces is not None
                        else None,
                        "converged": item.converged,
                        "status": item.status,
                        "detail": item.status_message,
                        "iterations": item.iterations,
                        **native_fock_observation(item),
                        "energy_change": item.energy_change,
                        "density_rms": item.density_rms,
                        "physical_residual_rms": item.physical_residual_rms,
                        "warm_start_used": item.warm_start_used,
                        "warm_start_fallback": item.warm_start_fallback,
                        "seconds": seconds,
                        "prepare_seconds": prepare,
                        "complete_seconds": seconds + prepare,
                        "native_force_components": (
                            normalize_force_work(force_work)
                            if force_work is not None
                            else None
                        ),
                    }
                    retain(row, baseline)
                    if repeat == 0:
                        owner.set_warm_start_updates(False)
            owner._public_dft_cuda_force = original_force
            record["native_force_components"] = normalize_force_work(force_work)
        record["status"] = "measured"
        save("complete")
    except NotImplementedError as error:
        record.update(status="unsupported", error=f"{type(error).__name__}: {error}")
        save(record.get("stage", "setup"))
        raise
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        save(record.get("stage", "setup"))
        raise
    finally:
        if owner is not None:
            owner.close()


if __name__ == "__main__":
    main()
