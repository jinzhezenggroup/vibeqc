"""Same-process QCE/GPU4PySCF HF energy+gradient microbenchmarks.

The ``sp8`` case measures tiny-system native-call and warm-plan overhead.  The
``sdf18-direct`` case crosses QCE's current persistent-ERI threshold and thus
also exercises screened direct J/K.  ``he3-sd21-direct`` adds a somewhat larger
direct workload.  ``water-def2-svp`` is the first named-basis molecular case.
The first three use artificial Cartesian bases for which exact common-engine
validation exists.

The UHF cases cover both persistent-ERI and screened-direct spin paths.

Run GPU work through the site scheduler, for example::

    srun --partition=main --gres=gpu:5090:1 --nodes=1 --ntasks=1 \
      env CUDA_PATH=/group/software/cuda-12.9.1 \
      PATH=/group/software/cuda-12.9.1/bin:/usr/bin:/bin \
      LD_LIBRARY_PATH=/group/software/cuda-12.9.1/lib64 \
      QCE_LIBRARY=$PWD/build/libqce.so.0.1.0 PYTHONPATH=python \
      build/gpu4pyscf-venv/bin/python benchmarks/compare_gpu4pyscf.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import statistics
import time
from typing import Any

import cupy as cp
from pyscf import gto, scf
from gpu4pyscf.scf import uhf as gpu_uhf

from qce import Calculator, Primitive, Shell

from _support import environment_metadata, write_result


@dataclass(frozen=True)
class BenchmarkCase:
    """One exact common workload for QCE and PySCF/GPU4PySCF."""

    description: str
    atoms: tuple[tuple[str, tuple[float, float, float]], ...]
    qce_basis: str | tuple[Shell, ...]
    pyscf_basis: str | dict[str, list]
    charge: int = 0
    multiplicity: int = 1
    method: str = "rhf"


def benchmark_cases() -> dict[str, BenchmarkCase]:
    """Return artificial and bundled named-basis validation cases."""

    sp_atoms = (("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7)))
    return {
        "sp8": BenchmarkCase(
            description="H2, 8 Cartesian s/p AOs",
            atoms=sp_atoms,
            qce_basis=(
                Shell(0, 0, (Primitive(1.2, 1.0),)),
                Shell(0, 1, (Primitive(0.7, 1.0),)),
                Shell(1, 0, (Primitive(1.2, 1.0),)),
                Shell(1, 1, (Primitive(0.7, 1.0),)),
            ),
            pyscf_basis={
                "H": [[0, [1.2, 1.0]], [1, [0.7, 1.0]]],
            },
        ),
        "sdf18-direct": BenchmarkCase(
            description="HeH+, 18 Cartesian s/d/f AOs, screened direct J/K",
            atoms=(("He", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))),
            qce_basis=(
                Shell(0, 0, (Primitive(1.5, 1.0),)),
                Shell(0, 2, (Primitive(0.8, 1.0),)),
                Shell(0, 3, (Primitive(0.6, 1.0),)),
                Shell(1, 0, (Primitive(1.2, 1.0),)),
            ),
            pyscf_basis={
                "He": [[0, [1.5, 1.0]], [2, [0.8, 1.0]], [3, [0.6, 1.0]]],
                "H": [[0, [1.2, 1.0]]],
            },
            charge=1,
        ),
        "he3-sd21-direct": BenchmarkCase(
            description="linear He3, 21 Cartesian s/d AOs, direct J/K",
            atoms=(
                ("He", (0.0, 0.0, -2.0)),
                ("He", (0.0, 0.0, 0.0)),
                ("He", (0.0, 0.0, 2.0)),
            ),
            qce_basis=tuple(
                shell
                for atom_index in range(3)
                for shell in (
                    Shell(atom_index, 0, (Primitive(1.5, 1.0),)),
                    Shell(atom_index, 2, (Primitive(0.8, 1.0),)),
                )
            ),
            pyscf_basis={
                "He": [[0, [1.5, 1.0]], [2, [0.8, 1.0]]],
            },
        ),
        "water-def2-svp": BenchmarkCase(
            description="H2O, 25 Cartesian AOs, def2-SVP direct J/K",
            atoms=(
                ("O", (0.0, 0.0, 0.0)),
                ("H", (0.0, -1.43233673, 1.10715266)),
                ("H", (0.0, 1.43233673, 1.10715266)),
            ),
            qce_basis="def2-svp",
            pyscf_basis="def2-svp",
        ),
        "h2plus-uhf2": BenchmarkCase(
            description="H2+, 2 Cartesian s AOs, UHF doublet",
            atoms=sp_atoms,
            qce_basis=(
                Shell(
                    0,
                    0,
                    (
                        Primitive(3.42525091, 0.15432897),
                        Primitive(0.62391373, 0.53532814),
                        Primitive(0.16885540, 0.44463454),
                    ),
                ),
                Shell(
                    1,
                    0,
                    (
                        Primitive(3.42525091, 0.15432897),
                        Primitive(0.62391373, 0.53532814),
                        Primitive(0.16885540, 0.44463454),
                    ),
                ),
            ),
            pyscf_basis={
                "H": [
                    [
                        0,
                        [3.42525091, 0.15432897],
                        [0.62391373, 0.53532814],
                        [0.16885540, 0.44463454],
                    ]
                ],
            },
            charge=1,
            multiplicity=2,
            method="uhf",
        ),
        "heh-sdf18-uhf": BenchmarkCase(
            description="HeH, 18 Cartesian s/d/f AOs, direct UHF doublet",
            atoms=(("He", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))),
            qce_basis=(
                Shell(0, 0, (Primitive(1.5, 1.0),)),
                Shell(0, 2, (Primitive(0.8, 1.0),)),
                Shell(0, 3, (Primitive(0.6, 1.0),)),
                Shell(1, 0, (Primitive(1.2, 1.0),)),
            ),
            pyscf_basis={
                "He": [[0, [1.5, 1.0]], [2, [0.8, 1.0]], [3, [0.6, 1.0]]],
                "H": [[0, [1.2, 1.0]]],
            },
            multiplicity=2,
            method="uhf",
        ),
    }


def accelerator_metadata() -> dict[str, Any]:
    """Return the CUDA device properties relevant to performance comparisons."""

    device_id = cp.cuda.Device().id
    properties = cp.cuda.runtime.getDeviceProperties(device_id)
    name = properties.get("name")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace").rstrip("\x00")
    return {
        "backend": "cuda",
        "device_id": device_id,
        "name": name,
        "compute_capability": [
            properties.get("major"),
            properties.get("minor"),
        ],
        "total_global_memory_bytes": properties.get("totalGlobalMem"),
        "multiprocessor_count": properties.get("multiProcessorCount"),
        "clock_rate_khz": properties.get("clockRate"),
        "driver_version": cp.cuda.runtime.driverGetVersion(),
        "runtime_version": cp.cuda.runtime.runtimeGetVersion(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=9)
    cases = benchmark_cases()
    parser.add_argument("--case", choices=cases, default="sp8")
    parser.add_argument(
        "--output",
        help="optional JSON path for raw timings and reproducibility metadata",
    )
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    case = cases[args.case]

    calculator = Calculator(
        method=case.method,
        basis=case.qce_basis,
        device="cuda",
        energy_tolerance=1.0e-12,
        density_tolerance=1.0e-10,
    )
    with calculator.prepare_batch(
        [case.atoms],
        charges=[case.charge],
        multiplicities=[case.multiplicity],
        warm_start=True,
    ) as batch:
        start = time.perf_counter()
        qce_result = batch.execute(strict=True)
        qce_cold = time.perf_counter() - start
        qce_warm: list[float] = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            qce_result = batch.execute(strict=True)
            qce_warm.append(time.perf_counter() - start)

    molecule = gto.M(
        atom=case.atoms,
        unit="Bohr",
        charge=case.charge,
        spin=case.multiplicity - 1,
        cart=True,
        basis=case.pyscf_basis,
        verbose=0,
    )
    gpu4pyscf = (
        gpu_uhf.UHF(molecule)
        if case.method == "uhf"
        else scf.RHF(molecule).to_gpu()
    )
    gpu4pyscf.conv_tol = 1.0e-12
    gpu4pyscf.conv_tol_grad = 1.0e-10
    gpu4pyscf.direct_scf_tol = 1.0e-14

    cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    gpu4pyscf_energy = gpu4pyscf.kernel()
    gpu4pyscf_gradient = gpu4pyscf.nuc_grad_method().kernel()
    cp.cuda.Stream.null.synchronize()
    gpu4pyscf_cold = time.perf_counter() - start

    gpu4pyscf_warm: list[float] = []
    for _ in range(args.repeats):
        density = gpu4pyscf.make_rdm1()
        cp.cuda.Stream.null.synchronize()
        start = time.perf_counter()
        gpu4pyscf_energy = gpu4pyscf.kernel(dm0=density)
        gpu4pyscf_gradient = gpu4pyscf.nuc_grad_method().kernel()
        cp.cuda.Stream.null.synchronize()
        gpu4pyscf_warm.append(time.perf_counter() - start)

    qce_item = qce_result.items[0]
    print(
        f"scope: {case.description}, {case.method.upper()} energy + "
        "analytic gradient"
    )
    print(f"QCE energy: {qce_item.energy:.15f} Eh")
    print(f"GPU4PySCF energy: {float(gpu4pyscf_energy):.15f} Eh")
    print(f"QCE force z(atom 0): {qce_item.forces[0, 2]:.15f} Eh/bohr")
    print(
        "GPU4PySCF force z(atom 0): "
        f"{-float(gpu4pyscf_gradient[0, 2]):.15f} Eh/bohr"
    )
    print(f"QCE cold plan execution: {qce_cold * 1.0e3:.3f} ms")
    print(
        "QCE warm median/min: "
        f"{statistics.median(qce_warm) * 1.0e3:.3f}/"
        f"{min(qce_warm) * 1.0e3:.3f} ms"
    )
    print(f"GPU4PySCF cold execution: {gpu4pyscf_cold * 1.0e3:.3f} ms")
    print(
        "GPU4PySCF warm median/min: "
        f"{statistics.median(gpu4pyscf_warm) * 1.0e3:.3f}/"
        f"{min(gpu4pyscf_warm) * 1.0e3:.3f} ms"
    )
    print("warning: this microbenchmark is not a realistic basis-set comparison")
    if args.output:
        payload = {
            "schema_version": 1,
            "benchmark": "compare_gpu4pyscf",
            "environment": environment_metadata(
                distributions={
                    "cupy": ("cupy-cuda12x", "cupy"),
                    "gpu4pyscf": ("gpu4pyscf-cuda12x", "gpu4pyscf"),
                    "numpy": ("numpy",),
                    "pyscf": ("pyscf",),
                },
                accelerator=accelerator_metadata(),
            ),
            "workload": {
                "case": args.case,
                "description": case.description,
                "method": case.method,
                "atoms": [
                    {"element": element, "coordinates_bohr": list(coordinates)}
                    for element, coordinates in case.atoms
                ],
                "charge": case.charge,
                "multiplicity": case.multiplicity,
                "cartesian_basis": True,
                "energy_tolerance": 1.0e-12,
                "density_tolerance": 1.0e-10,
                "direct_scf_tolerance": 1.0e-14,
            },
            "settings": {"repeats": args.repeats},
            "qce": {
                "energy_hartree": qce_item.energy,
                "forces_hartree_per_bohr": qce_item.forces.tolist(),
                "cold_seconds": qce_cold,
                "warm_seconds": qce_warm,
                "warm_median_seconds": statistics.median(qce_warm),
                "warm_minimum_seconds": min(qce_warm),
            },
            "gpu4pyscf": {
                "energy_hartree": float(gpu4pyscf_energy),
                "forces_hartree_per_bohr": (-gpu4pyscf_gradient).get().tolist(),
                "cold_seconds": gpu4pyscf_cold,
                "warm_seconds": gpu4pyscf_warm,
                "warm_median_seconds": statistics.median(gpu4pyscf_warm),
                "warm_minimum_seconds": min(gpu4pyscf_warm),
            },
        }
        destination = write_result(args.output, payload)
        print(f"JSON result: {destination}")


if __name__ == "__main__":
    main()
