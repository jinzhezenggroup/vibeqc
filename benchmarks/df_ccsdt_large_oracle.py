"""Independent PySCF reference for hundreds-AO correlation-only DF RCCSD(T).

Run as validation tooling, never import this module from a production owner.
The retained B factors use the explicit symmetric metric cutoff of the method
contract; injecting their packed AO form into PySCF compares the same fitted
Hamiltonian, independent of PySCF's default Cholesky/metric-cutoff choices.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time

import numpy as np
import pyscf
from pyscf import cc, df, gto, lib, scf

from benchmarks._retention import raw_output_path


def geometry(
    case: str,
) -> tuple[list[tuple[str, tuple[float, float, float]]], str, str]:
    """Fixed ideal molecular geometries in Angstrom, with exact point symmetry."""
    atoms = []
    if case == "ethane230":
        for sign, offset in ((1, 0.0), (-1, math.pi / 3)):
            atoms.append(("C", (0.0, 0.0, sign * 0.77)))
            for index in range(3):
                phi = offset + index * 2 * math.pi / 3
                radius = 1.09 * math.sqrt(8) / 3
                atoms.append(
                    (
                        "H",
                        (
                            radius * math.cos(phi),
                            radius * math.sin(phi),
                            sign * (0.77 + 1.09 / 3),
                        ),
                    )
                )
        return atoms, "aug-cc-pvtz", "aug-cc-pvtz-ri"
    if case == "benzene264":
        for element, radius in (("C", 1.397), ("H", 2.487)):
            for index in range(6):
                phi = index * math.pi / 3
                atoms.append(
                    (element, (radius * math.cos(phi), radius * math.sin(phi), 0.0))
                )
        return atoms, "cc-pvtz", "cc-pvtz-ri"
    raise ValueError(f"unknown large DF case {case}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("ethane230", "benzene264"), required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    if pyscf.__version__ != "2.14.0":
        raise RuntimeError(
            f"reference requires pinned PySCF 2.14.0, found {pyscf.__version__}"
        )
    lib.num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    atoms, basis, auxiliary = geometry(args.case)
    started = time.perf_counter()
    timings = {}
    mol = gto.M(
        atom=atoms,
        basis=basis,
        unit="Angstrom",
        spin=0,
        charge=0,
        verbose=4,
        max_memory=32000,
    )
    reference = scf.RHF(mol)
    reference.conv_tol = 1e-12
    reference.conv_tol_grad = 1e-10
    reference.max_cycle = 150
    reference.kernel()
    if not reference.converged:
        raise RuntimeError("independent conventional RHF did not converge")
    timings["rhf_seconds"] = time.perf_counter() - started
    phase = time.perf_counter()
    auxmol = df.addons.make_auxmol(mol, auxiliary)
    metric = auxmol.intor("int2c2e")
    values, vectors = np.linalg.eigh(metric)
    threshold = 1e-10 * values[-1]
    retained = values > threshold
    if np.min(abs(values - threshold)) < 1e-12 * values[-1]:
        raise RuntimeError("DF metric rank branch is unresolved")
    whitening = (vectors[:, retained] / np.sqrt(values[retained])) @ vectors[
        :, retained
    ].T
    raw = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
    bao = (whitening.T @ raw.reshape(mol.nao_nr() ** 2, auxmol.nao_nr()).T).reshape(
        auxmol.nao_nr(), mol.nao_nr(), mol.nao_nr()
    )
    packed = lib.pack_tril(bao)
    coefficients = reference.mo_coeff
    bmo = np.einsum("Qmn,mp,nq->Qpq", bao, coefficients, coefficients, optimize=True)
    timings["df_source_seconds"] = time.perf_counter() - phase
    # The independent solver gets the specified factors, while Fock/orbitals
    # remain those of conventional RHF. No approximate Fock is substituted.
    solver = cc.CCSD(reference).density_fit(auxbasis=auxiliary)
    solver.with_df._cderi = packed
    solver.conv_tol = 1e-12
    solver.conv_tol_normt = 1e-10
    solver.max_cycle = 150
    solver.max_memory = 32000
    phase = time.perf_counter()
    correlation, t1, t2 = solver.kernel()
    if not solver.converged:
        raise RuntimeError("independent same-Hamiltonian DF RCCSD did not converge")
    timings["ccsd_seconds"] = time.perf_counter() - phase
    phase = time.perf_counter()
    eris = solver.ao2mo()
    next_t1, next_t2 = solver.update_amps(t1, t2, eris)
    o = mol.nelectron // 2
    eps = reference.mo_energy
    d1 = eps[:o, None] - eps[None, o:]
    d2 = d1[:, None, :, None] + d1[None, :, None, :]
    residuals = (
        float(np.max(abs((next_t1 - t1) * d1))),
        float(np.max(abs((next_t2 - t2) * d2))),
    )
    if max(residuals) > 1e-9:
        raise RuntimeError(f"independent DF RCCSD replay residuals failed: {residuals}")
    triples = float(solver.ccsd_t(eris=eris))
    timings["replay_and_triples_seconds"] = time.perf_counter() - phase
    phase = time.perf_counter()
    state = args.output / "state.npz"
    np.savez(
        state,
        coefficients=coefficients,
        orbital_energies=eps,
        overlap=reference.get_ovlp(),
        hcore=reference.get_hcore(),
        fock=reference.get_fock(),
        metric=metric,
        metric_eigenvalues=values,
        metric_whitening=whitening,
        raw_three_center=raw,
        bmo=bmo,
        t1=t1,
        t2=t2,
    )
    timings["state_write_seconds"] = time.perf_counter() - phase
    timings["complete_seconds"] = time.perf_counter() - started
    record = {
        "schema": "generativeqc.df-ccsdt.large-oracle.v1",
        "case": args.case,
        "pyscf_version": pyscf.__version__,
        "numpy_version": np.__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "threads": args.threads,
        "atoms_angstrom": atoms,
        "basis": basis,
        "auxiliary_basis": auxiliary,
        "nbf": mol.nao_nr(),
        "nocc": o,
        "nvir": mol.nao_nr() - o,
        "naux": auxmol.nao_nr(),
        "reference_mode": "conventional-unscreened-rhf",
        "correlation_mode": "density-fitting",
        "metric_policy": "symmetric inverse square root",
        "metric_relative_threshold": 1e-10,
        "metric_absolute_threshold": float(threshold),
        "metric_rank": int(retained.sum()),
        "metric_condition_number": float(values[-1] / values[retained][0]),
        "rhf_energy": float(reference.e_tot),
        "correlation_energy": float(correlation),
        "triples_energy": triples,
        "total_energy": float(reference.e_tot + correlation + triples),
        "replay_r1_max": residuals[0],
        "replay_r2_max": residuals[1],
        "timings": timings,
        "state_bytes": state.stat().st_size,
    }
    with state.open("rb") as stream:
        record["state_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    (args.output / "oracle.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
