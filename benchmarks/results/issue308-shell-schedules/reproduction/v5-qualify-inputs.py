"""Record actual stock GPU4PySCF DF factor ranks before clean timing.

The decomposition wrapper only observes the unmodified provider's return.
Independent metric spectra use CPU libcint; no qualification work is timed as
an endpoint. The process exits before the separate comparison processes start.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

assert os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")
import cupy as cp
import numpy as np
from gpu4pyscf.df import df as gpu_df
from pyscf import df, gto, lib, scf

from benchmarks._cases import benchmark_cases
from benchmarks.compare_gpu4pyscf_batch import scaled_geometries

lib.num_threads(1)
output = Path(sys.argv[1])
results = []
original = gpu_df._decompose_j2c
observed = []


def observe(*args, **kwargs):
    result = original(*args, **kwargs)
    observed.append(
        {
            "factorization": result[1],
            "coefficient_shape": list(result[0].shape),
            "dtype": str(result[0].dtype),
            "fallback_absolute_threshold": float(gpu_df.LINEAR_DEP_THR),
        }
    )
    return result


gpu_df._decompose_j2c = observe
for name in (
    "water-tetramer-def2-svp-spherical",
    "water-octamer-s4-def2-svp-spherical",
    "water-hexadecamer-2s4-def2-svp-spherical",
    "oh-def2-svp-spherical-uhf",
):
    case = benchmark_cases()[name]
    for batch in (1, 4):
        for slot, atoms in enumerate(scaled_geometries(case.atoms, batch)):
            mol = gto.M(
                atom=atoms,
                unit="Bohr",
                basis=case.pyscf_basis,
                cart=False,
                charge=case.charge,
                spin=case.multiplicity - 1,
                verbose=0,
            )
            aux = df.addons.make_auxmol(mol, case.pyscf_basis)
            metric = aux.intor("int2c2e")
            eig = np.linalg.eigvalsh(metric)
            start = len(observed)
            engine = (
                (scf.UHF(mol) if case.method == "uhf" else scf.RHF(mol))
                .density_fit(auxbasis=case.pyscf_basis)
                .to_gpu()
            )
            engine.with_df.build()
            cp.cuda.runtime.deviceSynchronize()
            shapes = [list(a.shape) for a in engine.with_df._cderi]
            rank = sum(a[0] for a in shapes)
            row = {
                "case": name,
                "batch": batch,
                "slot": slot,
                "atoms": atoms,
                "unit": "Bohr",
                "representation": "spherical",
                "nbf": mol.nao,
                "naux": aux.nao,
                "orbital_basis": mol._basis,
                "auxiliary_basis": aux._basis,
                "reference_metric_sha256": hashlib.sha256(metric.tobytes()).hexdigest(),
                "reference_metric_extrema": [float(eig[0]), float(eig[-1])],
                "reference_condition": float(eig[-1] / eig[0]),
                "relative_threshold": 1e-10,
                "reference_effective_rank": int(
                    np.count_nonzero(eig > 1e-10 * eig[-1])
                ),
                "gpu4pyscf_factor_blocks": shapes,
                "gpu4pyscf_effective_rank": rank,
                "gpu4pyscf_decomposition": observed[start:],
                "gpu4pyscf_default_initial_guess": engine.init_guess,
            }
            row["matching_full_rank"] = (
                rank == row["reference_effective_rank"] == aux.nao
            )
            results.append(row)
            output.write_text(
                json.dumps(
                    {"slurm_job": os.environ["SLURM_JOB_ID"], "results": results},
                    indent=2,
                )
                + "\n"
            )
            assert row["matching_full_rank"], row
            del engine
            cp.get_default_memory_pool().free_all_blocks()
print(
    "Qualified matched basis/metric ranks for",
    len(results),
    "input geometries",
    flush=True,
)
