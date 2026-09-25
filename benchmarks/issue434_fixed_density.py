"""Replay the #434 saved density without rerunning SCF and compare native operators."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.linalg

try:
    from benchmarks._retention import raw_output_path
except ModuleNotFoundError:
    from _retention import raw_output_path


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def maximum(array: np.ndarray) -> float:
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("diagnostic differences must be non-empty and finite")
    return float(np.max(np.abs(array)))


def residuals(
    fock: np.ndarray, overlap: np.ndarray, density: np.ndarray
) -> dict[str, float]:
    eigenvalues, coefficients = scipy.linalg.eigh(fock, overlap, check_finite=True)
    electrons = float(np.einsum("ij,ji", density, overlap))
    occupied_count = round(electrons / 2.0)
    occupied = coefficients[:, :occupied_count]
    projector = 2.0 * occupied @ occupied.T
    delta = projector - density
    return {
        "occupied_count": occupied_count,
        "lowest_eigenvalue": float(eigenvalues[0]),
        "highest_occupied_eigenvalue": float(eigenvalues[occupied_count - 1]),
        "lowest_virtual_eigenvalue": float(eigenvalues[occupied_count]),
        "commutator_maximum": maximum(
            fock @ density @ overlap - overlap @ density @ fock
        ),
        "fixed_point_density_maximum": maximum(delta),
        "fixed_point_density_rms": float(np.sqrt(np.mean(delta * delta))),
        "electron_trace_error": abs(electrons - 2.0 * occupied_count),
        "idempotency_error": maximum(density @ overlap @ density - 2.0 * density),
    }


def chunked_maximum_difference(left: Path, right: Path, chunk: int = 1 << 20) -> float:
    if type(chunk) is not int or chunk <= 0:
        raise ValueError("comparison chunk must be a positive integer")
    lhs = np.load(left, mmap_mode="r", allow_pickle=False)
    if lhs.dtype != np.dtype(np.float64) or lhs.size == 0:
        raise ValueError("reference snapshot must be a non-empty native float64 array")
    expected_bytes = lhs.size * np.dtype(np.float64).itemsize
    if right.stat().st_size != expected_bytes:
        raise ValueError(f"raw snapshot size mismatch for {right.name}")
    rhs = np.memmap(right, dtype=np.float64, mode="r", shape=lhs.shape)
    result = 0.0
    flat_lhs, flat_rhs = lhs.reshape(-1), rhs.reshape(-1)
    for begin in range(0, flat_lhs.size, chunk):
        end = min(flat_lhs.size, begin + chunk)
        result = max(result, maximum(flat_lhs[begin:end] - flat_rhs[begin:end]))
    return result


def coulomb_from_raw(
    raw: np.ndarray, metric: np.ndarray, density: np.ndarray
) -> np.ndarray:
    """Recompose J with one fixed CPU solve so raw/metric perturbations are comparable."""
    matrix = raw.reshape(-1, metric.shape[0])
    charge = matrix.T @ density.reshape(-1)
    fitted = scipy.linalg.solve(metric, charge, assume_a="pos", check_finite=False)
    return (matrix @ fitted).reshape(density.shape)


def coulomb_from_whitened_raw(
    raw: np.ndarray, metric: np.ndarray, density: np.ndarray
) -> np.ndarray:
    """Recompose J after explicit Cholesky whitening of the three-center columns."""
    matrix = raw.reshape(-1, metric.shape[0])
    factor = scipy.linalg.cholesky(metric, lower=True, check_finite=False)
    whitened = scipy.linalg.solve_triangular(
        factor, matrix.T, lower=True, check_finite=False
    ).T
    density_flat = density.reshape(-1)
    return (whitened @ (whitened.T @ density_flat)).reshape(density.shape)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ or "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise SystemExit(
            "#434 fixed-density capture must run inside a Slurm GPU allocation"
        )
    if args.output.exists():
        raise SystemExit(f"refusing to reuse output directory: {args.output}")
    args.output.mkdir(parents=True)

    fixture_report = json.loads((args.fixture / "report.json").read_text())
    expected = fixture_report.get("fixture_sha256", {})
    for name in ("input.txt", "reference.npz", "raw.npy"):
        path = args.fixture / name
        if not path.is_file():
            raise SystemExit(f"missing fixture file: {path}")
        if name in expected and sha256(path) != expected[name]:
            raise SystemExit(f"fixture hash mismatch: {name}")

    env = {
        key: value for key, value in os.environ.items() if not key.startswith("VIBEQC_")
    }
    env.update(
        VIBEQC_DF_EXCHANGE="occupied",
        VIBEQC_DF_VALUE_STORAGE="dense",
        VIBEQC_DF_RESIDENT_EXCHANGE="auto",
    )
    result = subprocess.run(
        [
            str(args.probe.resolve()),
            str((args.fixture / "input.txt").resolve()),
            str(args.output.resolve()),
        ],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=300,
        check=False,
    )
    (args.output / "native.jsonl").write_text(result.stdout)
    if result.returncode:
        print(result.stdout, file=sys.stderr, end="")
        raise SystemExit(result.returncode)
    records = [
        json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
    ]
    identity = next(item for item in records if item.get("operation") == "identity")
    diagnostic = next(
        item for item in records if item.get("operation") == "fixed_density_fock"
    )
    nbf, naux = int(diagnostic["nbf"]), int(diagnostic["naux"])
    if nbf != fixture_report["nbf"] or naux != fixture_report["naux"]:
        raise ValueError("native dimensions differ from the preserved fixture")

    reference_file = args.fixture / "reference.npz"
    with np.load(reference_file, allow_pickle=False) as saved:
        reference = {name: np.asarray(saved[name]) for name in saved.files}
    shapes = {
        "overlap": (nbf, nbf),
        "hcore": (nbf, nbf),
        "coulomb": (nbf, nbf),
        "exchange": (nbf, nbf),
        "fock": (nbf, nbf),
        "metric": (naux, naux),
        "density": (nbf, nbf),
    }
    native: dict[str, np.ndarray] = {}
    differences: dict[str, float] = {}
    for name, shape in shapes.items():
        path = args.output / f"{name}.bin"
        array = np.fromfile(path, dtype=np.float64).reshape(shape)
        native[name] = array
        differences[name] = maximum(array - reference[name])

    raw_error = chunked_maximum_difference(
        args.fixture / "raw.npy", args.output / "raw.bin"
    )
    differences["raw_three_center"] = raw_error
    native_residual = residuals(native["fock"], native["overlap"], native["density"])
    independent_residual = residuals(
        reference["fock"], reference["overlap"], reference["density"]
    )
    cross = {
        "native_fock_independent_overlap": residuals(
            native["fock"], reference["overlap"], reference["density"]
        ),
        "independent_fock_native_overlap": residuals(
            reference["fock"], native["overlap"], reference["density"]
        ),
    }
    fock_recomposition_error = maximum(
        native["fock"]
        - (native["hcore"] + native["coulomb"] - 0.5 * native["exchange"])
    )
    reference_raw = np.load(args.fixture / "raw.npy", mmap_mode="r", allow_pickle=False)
    native_raw = np.memmap(
        args.output / "raw.bin", dtype=np.float64, mode="r", shape=reference_raw.shape
    )
    j_reference_raw_metric = coulomb_from_raw(
        reference_raw, reference["metric"], reference["density"]
    )
    j_native_raw_reference_metric = coulomb_from_raw(
        native_raw, reference["metric"], reference["density"]
    )
    j_reference_raw_native_metric = coulomb_from_raw(
        reference_raw, native["metric"], reference["density"]
    )
    j_native_raw_metric = coulomb_from_raw(
        native_raw, native["metric"], reference["density"]
    )
    raw_effect = j_native_raw_reference_metric - j_reference_raw_metric
    metric_effect = j_reference_raw_native_metric - j_reference_raw_metric
    combined_effect = j_native_raw_metric - j_reference_raw_metric
    coulomb_attribution = {
        "raw_only_effect_maximum": maximum(raw_effect),
        "metric_only_effect_maximum": maximum(metric_effect),
        "combined_effect_maximum": maximum(combined_effect),
        "nonlinear_cross_term_maximum": maximum(
            combined_effect - raw_effect - metric_effect
        ),
        "reference_recomposition_vs_independent_j": maximum(
            j_reference_raw_metric - reference["coulomb"]
        ),
        "native_recomposition_vs_native_j": maximum(
            j_native_raw_metric - native["coulomb"]
        ),
    }

    j_reference_raw_metric_whitened = coulomb_from_whitened_raw(
        reference_raw, reference["metric"], reference["density"]
    )
    j_native_raw_reference_metric_whitened = coulomb_from_whitened_raw(
        native_raw, reference["metric"], reference["density"]
    )
    j_reference_raw_native_metric_whitened = coulomb_from_whitened_raw(
        reference_raw, native["metric"], reference["density"]
    )
    j_native_raw_metric_whitened = coulomb_from_whitened_raw(
        native_raw, native["metric"], reference["density"]
    )
    whitened_raw_effect = (
        j_native_raw_reference_metric_whitened - j_reference_raw_metric_whitened
    )
    whitened_metric_effect = (
        j_reference_raw_native_metric_whitened - j_reference_raw_metric_whitened
    )
    whitened_combined_effect = (
        j_native_raw_metric_whitened - j_reference_raw_metric_whitened
    )
    coulomb_arithmetic = {
        "reference_direct_vs_whitened_maximum": maximum(
            j_reference_raw_metric - j_reference_raw_metric_whitened
        ),
        "native_direct_vs_whitened_maximum": maximum(
            j_native_raw_metric - j_native_raw_metric_whitened
        ),
        "raw_effect_schedule_delta_maximum": maximum(
            raw_effect - whitened_raw_effect
        ),
        "metric_effect_schedule_delta_maximum": maximum(
            metric_effect - whitened_metric_effect
        ),
        "combined_effect_schedule_delta_maximum": maximum(
            combined_effect - whitened_combined_effect
        ),
        "whitened_raw_only_effect_maximum": maximum(whitened_raw_effect),
        "whitened_metric_only_effect_maximum": maximum(whitened_metric_effect),
        "whitened_combined_effect_maximum": maximum(whitened_combined_effect),
        "whitened_nonlinear_cross_term_maximum": maximum(
            whitened_combined_effect - whitened_raw_effect - whitened_metric_effect
        ),
    }
    fock_delta = native["fock"] - reference["fock"]
    maximum_index = np.unravel_index(np.argmax(np.abs(fock_delta)), fock_delta.shape)
    maximum_fock_delta_components = {
        "index": [int(value) for value in maximum_index],
        "fock": float(fock_delta[maximum_index]),
        "hcore": float((native["hcore"] - reference["hcore"])[maximum_index]),
        "coulomb": float((native["coulomb"] - reference["coulomb"])[maximum_index]),
        "minus_half_exchange": float(
            -0.5 * (native["exchange"] - reference["exchange"])[maximum_index]
        ),
    }
    report = {
        "schema": "vibeqc.issue434.fixed-density-diagnosis.v1",
        "scope": "Exact saved-density native/independent operator comparison; no SCF rerun or admission.",
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "probe_sha256": sha256(args.probe),
        "fixture_report_sha256": sha256(args.fixture / "report.json"),
        "native_identity": identity,
        "diagnostic": diagnostic,
        "maximum_absolute_differences": differences,
        "native_residuals": native_residual,
        "independent_residuals": independent_residual,
        "cross_composition_residuals": cross,
        "native_fock_recomposition_error": fock_recomposition_error,
        "coulomb_cross_recomposition": coulomb_attribution,
        "coulomb_arithmetic_schedule": coulomb_arithmetic,
        "maximum_fock_delta_components": maximum_fock_delta_components,
        "strict_1e12": {
            "native_commutator_pass": native_residual["commutator_maximum"] <= 1.0e-12,
            "native_projector_pass": native_residual["fixed_point_density_maximum"]
            <= 1.0e-12,
            "independent_commutator_pass": independent_residual["commutator_maximum"]
            <= 1.0e-12,
            "independent_projector_pass": independent_residual[
                "fixed_point_density_maximum"
            ]
            <= 1.0e-12,
        },
    }
    report["output_sha256"] = {
        path.name: sha256(path) for path in sorted(args.output.glob("*.bin"))
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
