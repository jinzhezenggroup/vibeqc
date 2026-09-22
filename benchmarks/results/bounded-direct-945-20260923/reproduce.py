"""Qualify automatic Direct-HF admission with fixed and independent endpoints."""

import argparse
import ctypes
import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from pyscf import gto, scf
from vibeqc import Atom, Calculator
from vibeqc.autotune import source_identity
from vibeqc.calculator import _named_basis_shells

from benchmarks._cases import benchmark_cases

p = argparse.ArgumentParser()
p.add_argument("--case", default="water-tetramer-def2-tzvp-spherical")
p.add_argument("--batch", type=int, default=2)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--repeats", type=int, default=0)
p.add_argument(
    "--uhf", action="store_true", help="Test the singly ionized doublet with UHF"
)
p.add_argument(
    "--oracle", type=Path, help="Replay an independently stabilized PySCF reference"
)
a = p.parse_args()
assert os.environ.get("SLURM_JOB_ID")
for key in (
    "VIBEQC_BOUNDED_DIRECT_STREAMING",
    "VIBEQC_DIRECT_TILE_VALIDATION",
    "VIBEQC_AOT_SHELL_CLASSES",
):
    assert key not in os.environ, key
case = benchmark_cases()[a.case]
if a.uhf:
    case = dataclasses.replace(case, method="uhf", charge=1, multiplicity=2)
record = {
    "completed": False,
    "slurm_job": os.environ["SLURM_JOB_ID"],
    "case": a.case,
    "batch": a.batch,
    "method": case.method,
    "library_sha256": hashlib.sha256(
        Path(os.environ["VIBEQC_LIBRARY"]).read_bytes()
    ).hexdigest(),
}


def publish(stage: str) -> None:
    record["stage"] = stage
    a.output.write_text(json.dumps(record, indent=2) + "\n")
    print(stage, flush=True)


# Count exact Cartesian tiles before screening. This is the runtime's immutable
# fixed-arena admission census, even when the physical basis is spherical.
shells = _named_basis_shells(
    case.vibeqc_basis, [Atom.from_value((s, xyz)) for s, xyz in case.atoms]
)
counts = [(s.angular_momentum + 1) * (s.angular_momentum + 2) // 2 for s in shells]
pairs = np.array(
    [
        ni * nj if i != j else ni * (ni + 1) // 2
        for i, ni in enumerate(counts)
        for j, nj in enumerate(counts[: i + 1])
    ],
    dtype=np.int64,
)
tiles = sum(
    int(np.sum((n * pairs[:i] + 255) // 256))
    + (int(n) * (int(n) + 1) // 2 + 255) // 256
    for i, n in enumerate(pairs)
)
record["admission"] = {
    "shells": len(shells),
    "cartesian_aos": sum(counts),
    "tiles_per_system": tiles,
    "fixed_descriptor_bytes_per_system": tiles * 192,
    "fixed_budget_bytes": 1 << 30,
    "task_bytes": 192,
}
assert tiles * 192 <= 1 << 30, "single-system reference must fit fixed admission"
assert a.batch * tiles * 192 > 1 << 30, (
    "batch must cross production fixed-arena ceiling"
)
publish("oracle_start")
if a.oracle:
    oracle = json.loads(a.oracle.read_text())
else:
    mol = gto.M(
        atom=case.atoms,
        basis=case.pyscf_basis,
        unit="Bohr",
        cart=case.basis_representation == "cartesian",
        charge=case.charge,
        spin=case.multiplicity - 1,
        verbose=0,
    )
    ref = (scf.RHF if case.method == "rhf" else scf.UHF)(mol)
    ref.conv_tol = 1e-12
    ref.conv_tol_grad = 1e-9
    ref.max_cycle = 200
    start = time.perf_counter()
    ref.kernel()
    assert ref.converged
    oracle = {
        "energy": float(ref.e_tot),
        "forces": (-ref.nuc_grad_method().kernel()).tolist(),
        "seconds": time.perf_counter() - start,
    }
record["oracle"] = oracle
publish("fixed_start")
calc = Calculator(
    method=case.method,
    basis=case.vibeqc_basis,
    basis_representation=case.basis_representation,
    device="cuda",
    density_fitting="none",
    max_iterations=160,
    energy_tolerance=1e-12,
    density_tolerance=1e-10,
    screening_tolerance=1e-14,
)
calc._library.vibeqc_get_source_identity.restype = ctypes.c_char_p
record["source_identity"] = calc._library.vibeqc_get_source_identity().decode()
assert record["source_identity"] == source_identity(Path.cwd())
start = time.perf_counter()
with calc.prepare_batch(
    [case.atoms],
    charges=[case.charge],
    multiplicities=[case.multiplicity],
    shell_class_profiling=True,
) as prepared:
    fixed = prepared.execute(strict=True).items[0]
    record["fixed_work"] = [
        dataclasses.asdict(x) for x in prepared.last_shell_class_profile()
    ]
record["fixed"] = {
    "energy": fixed.energy,
    "forces": fixed.forces.tolist(),
    "seconds": time.perf_counter() - start,
    "iterations": fixed.iterations,
}
assert fixed.executed_backend == "cuda"
publish("automatic_start")
start = time.perf_counter()
with calc.prepare_batch(
    [case.atoms] * a.batch,
    charges=[case.charge] * a.batch,
    multiplicities=[case.multiplicity] * a.batch,
    shell_class_profiling=True,
) as prepared:
    record["preparation_seconds"] = time.perf_counter() - start
    record["automatic"] = []
    for repeat in range(a.repeats + 1):
        start = time.perf_counter()
        result = prepared.execute(strict=True)
        elapsed = time.perf_counter() - start
        row = {
            "seconds": elapsed,
            "energies": result.energies.tolist(),
            "iterations": [x.iterations for x in result.items],
            "forces": [x.forces.tolist() for x in result.items],
        }
        row["work"] = [
            dataclasses.asdict(x) for x in prepared.last_shell_class_profile()
        ]
        record["automatic"].append(row)
        publish(f"automatic_repeat_{repeat}_done")
        for fixed_row, auto_row in zip(record["fixed_work"], row["work"], strict=True):
            for key in ("shell_quartets", "tiles", "ao_quartets", "primitive_quartets"):
                assert auto_row[key] == a.batch * fixed_row[key], (
                    auto_row["label"],
                    key,
                    auto_row[key],
                    a.batch * fixed_row[key],
                )
        for item in result.items:
            assert item.converged and item.executed_backend == "cuda"
            assert abs(item.energy - fixed.energy) <= 2e-10
            assert np.max(np.abs(item.forces - fixed.forces)) <= 1e-8
            assert abs(item.energy - oracle["energy"]) <= 2e-8
            assert np.max(np.abs(item.forces - oracle["forces"])) <= 2e-7
record["completed"] = True
publish("passed")
