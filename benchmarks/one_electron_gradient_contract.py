"""Archive arbitrary-weight raw/fused equivalence and standalone staging costs.

The raw oracle is libcint on the host. Timings cover the complete synchronous
CUDA bridge, including system construction and transfers; they are not a
comparison of CPU and GPU integral-kernel speeds.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from vibeqc import Calculator, Primitive, Shell
from vibeqc.autotune import source_identity

from benchmarks._cases import benchmark_cases
from tools.vibeqc_validation.one_electron_gradient import (
    execute_gradient,
    reference_matrices,
)
from tools.vibeqc_validation.schema import canonical_hash, file_hash


def main():
    """Hold nonsymmetric S/T/V weights fixed across every measured mapping."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("sp8", "sdf18-direct"), default="sp8")
    parser.add_argument("--contraction-length", type=int, default=1)
    parser.add_argument("--maximum-bytes", type=int, default=128 << 10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".artifacts/benchmarks/one_electron_gradient_contract.json"),
    )
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("run real-GPU measurements inside a finite Slurm allocation")
    if args.contraction_length < 1:
        parser.error("contraction length must be positive")
    case = benchmark_cases()[args.case]
    basis = tuple(
        Shell(
            s.atom_index,
            s.angular_momentum,
            tuple(
                Primitive(
                    s.primitives[0].exponent * (1 + 0.2 * k),
                    (-1.0 if k % 3 == 2 else 1.0) / (k + 1),
                )
                for k in range(args.contraction_length)
            ),
        )
        for s in case.vibeqc_basis
    )
    inputs = {
        "atomic_numbers": [{"H": 1, "He": 2}[a] for a, _ in case.atoms],
        "coordinates": [list(r) for _, r in case.atoms],
        "charge": case.charge,
        "multiplicity": case.multiplicity,
        "basis_representation": "cartesian",
        "shells": [
            {
                "atom_index": s.atom_index,
                "angular_momentum": s.angular_momentum,
                "primitives": [[p.exponent, p.coefficient] for p in s.primitives],
            }
            for s in basis
        ],
    }
    values, raw = reference_matrices(inputs)
    weights = np.random.default_rng(141).normal(size=values.shape)
    expected = np.einsum("axoij,oij->ax", raw, weights)
    calc = Calculator(device="cuda", basis=basis)
    records = []
    for schedule in (0, 1, 2):

        def execute(schedule=schedule):
            return execute_gradient(
                calc,
                case.atoms,
                weights,
                schedule=schedule,
                maximum_bytes=args.maximum_bytes,
                charge=case.charge,
            )

        execute()  # Exclude first context/module loading from the measured calls.
        times = []
        for _ in range(5):
            start = time.perf_counter()
            actual, resources = execute()
            times.append(time.perf_counter() - start)
            np.testing.assert_allclose(actual, expected, atol=2e-10, rtol=2e-11)
        records.append(
            {
                "schedule": schedule,
                "seconds": times,
                "resources": resources,
                "gradient": actual.tolist(),
                "maximum_absolute_error": float(np.max(np.abs(actual - expected))),
            }
        )
    report = {
        "schema": "vibeqc.one_electron_gradient_contract",
        "version": 1,
        "inputs": inputs,
        "input_hash": canonical_hash(inputs),
        "weight_hash": canonical_hash(weights.tolist()),
        "records": records,
        "raw_oracle": "host libcint full dS/dT/dV",
        "raw_tensor_bytes": raw.nbytes,
        "reference_df_SH_derivative_bytes": 2 * raw.nbytes // 3,
        "caller_weight_bytes": weights.nbytes,
        "maximum_bytes": args.maximum_bytes,
        "source_identity": source_identity(ROOT),
        "binary_hash": file_hash(calc._library._name),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "timing_scope": "standalone synchronous CUDA bridge including metadata/weight uploads and gradient download",
        "host_scope": "numeric bridge capacities; excludes caller weights/system and opaque CUDA allocations",
        "accuracy_passed": True,
        "production_promoted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"case": args.case, "records": records}, indent=2))


if __name__ == "__main__":
    main()
