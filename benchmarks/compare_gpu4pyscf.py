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
import statistics
import time

from qce import Calculator

from _cases import benchmark_cases
from _support import (
    cuda_accelerator_metadata,
    environment_metadata,
    write_result,
)


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
    import cupy as cp
    from pyscf import gto, scf
    from gpu4pyscf.scf import uhf as gpu_uhf

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
                accelerator=cuda_accelerator_metadata(cp),
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
