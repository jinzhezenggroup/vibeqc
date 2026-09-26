"""Independent Libxc point acceptance for generated split-hybrid C++/CUDA.

This is validation tooling, never imported by production code generation.
CUDA mode requires NVCC and a real CUDA device; it never falls back to the host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT)]

SEED = 20260926
METHODS = (
    ("M06-2X", "m06_2x_point.cuh", 450, "450,236"),
    ("MN15", "mn15_point.cuh", 268, "268,269"),
)


def point_inputs() -> tuple[NDArray[np.float64], list[str], int]:
    rng = np.random.default_rng(SEED)
    rho = 10.0 ** rng.uniform(-2.0, 0.0, size=(24, 2))
    gradient = rng.normal(size=(24, 2, 3)) * rho[:, :, None] ** (4.0 / 3.0) * 0.2
    aa = np.sum(gradient[:, 0] ** 2, axis=1)
    ab = np.sum(gradient[:, 0] * gradient[:, 1], axis=1)
    bb = np.sum(gradient[:, 1] ** 2, axis=1)
    tau = rho ** (5.0 / 3.0) * rng.uniform(0.5, 2.0, size=(24, 2))
    tau += np.stack((aa, bb), axis=1) / (8.0 * rho)
    rows = np.column_stack((rho, aa, ab, bb, tau)).tolist()
    labels = [f"interior-{i}" for i in range(len(rows))]
    interior_count = len(rows)
    for label, values in (
        ("restricted-embedding", (0.2, 0.2, 0.01, 0.01, 0.01, 0.3, 0.3)),
        ("vacuum", (0, 0, 0, 0, 0, 0, 0)),
        ("empty-beta", (0.2, 0, 0.01, 0, 0, 0.3, 0)),
        ("empty-alpha", (0, 0.2, 0, 0, 0.01, 0, 0.3)),
        ("zero-gradient", (0.2, 0.15, 0, 0, 0, 0.3, 0.2)),
        ("zero-tau", (0.2, 0.15, 0, 0, 0, 0, 0)),
        ("antiparallel-gradient", (0.2, 0.2, 0.04, -0.04, 0.04, 0.4, 0.4)),
        ("fermi-hole-boundary", (0.2, 0.15, 0.01, 0, 0.01, 0.001, 0.001)),
    ):
        rows.append(list(values))
        labels.append(label)
    for density in (1e-30, 1e-20, 1e-16, 1e-15, 1e-14, 1e-12):
        for minority in (0.0, 1e-10, 0.5):
            a, b = density * (1 - minority), density * minority
            rows.append(
                [
                    a,
                    b,
                    a ** (8 / 3) * 0.1,
                    0,
                    b ** (8 / 3) * 0.1,
                    a ** (5 / 3),
                    b ** (5 / 3),
                ]
            )
            labels.append(f"tail-{density:g}-minority-{minority:g}")
    return np.asarray(rows, dtype=np.float64), labels, interior_count


def libxc_density(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Realize the Gram invariants as two gradient vectors for the Libxc API."""
    result = np.zeros((2, 6, len(points)), dtype=np.float64)
    result[:, 0] = points[:, :2].T
    result[:, 5] = points[:, 5:7].T
    alpha = np.sqrt(points[:, 2])
    beta_x = np.divide(points[:, 3], alpha, out=np.zeros_like(alpha), where=alpha > 0)
    remainder = points[:, 4] - beta_x * beta_x
    if np.any(remainder < -1e-12 * np.maximum(points[:, 4], 1e-300)):
        raise ValueError(
            "sigma invariants do not form a positive semidefinite Gram matrix"
        )
    result[0, 1] = alpha
    result[1, 1] = beta_x
    result[1, 2] = np.sqrt(np.maximum(remainder, 0.0))
    return result


def reference_points(code: str, points: NDArray[np.float64]) -> NDArray[np.float64]:
    from pyscf.dft import libxc

    exc, vxc, _, _ = libxc.eval_xc(code, libxc_density(points), spin=1, deriv=1)
    energy = np.asarray(exc) * np.sum(points[:, :2], axis=1)
    # Libxc exposes derivatives of energy density, not derivatives of epsilon.
    return np.column_stack((energy, vxc[0], vxc[1], vxc[3]))


def run_probe(
    executable: Path, code: int, points: NDArray[np.float64], work: Path, stem: str
) -> tuple[NDArray[np.float64], str]:
    source = work / f"{stem}-input.bin"
    target = work / f"{stem}-output.bin"
    np.ascontiguousarray(points, dtype=np.float64).tofile(source)
    completed = subprocess.run(
        [str(executable), str(code), str(len(points)), str(source), str(target)],
        text=True,
        capture_output=True,
        check=True,
        timeout=120,
    )
    values = np.fromfile(target, dtype=np.float64)
    if values.size != len(points) * 8:
        raise RuntimeError("point executable returned a truncated result")
    return values.reshape(-1, 8), completed.stderr.strip()


def comparison(
    actual: NDArray[np.float64], expected: NDArray[np.float64]
) -> dict[str, object]:
    if not np.all(np.isfinite(expected)) or not np.all(np.isfinite(actual)):
        return {"passed": False, "reason": "nonfinite oracle or generated output"}
    absolute = np.abs(actual - expected)
    atol = np.array([2e-10] + [3e-8] * 7)
    rtol = np.array([2e-9] + [3e-7] * 7)
    normalized = absolute / (atol + rtol * np.abs(expected))
    point, field = np.unravel_index(np.argmax(normalized), normalized.shape)
    return {
        "passed": bool(np.max(normalized) <= 1.0),
        "max_absolute_error_per_field": np.max(absolute, axis=0).tolist(),
        "max_normalized_error": float(np.max(normalized)),
        "worst_point": int(point),
        "worst_field": int(field),
        "actual_at_worst": float(actual[point, field]),
        "reference_at_worst": float(expected[point, field]),
        "absolute_tolerances": atol.tolist(),
        "relative_tolerances": rtol.tolist(),
    }


def finite_difference_points(
    points: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    selected = points[:4]
    steps = np.maximum(np.abs(selected), 1e-300) * 1e-4
    steps[:, 3] = np.sqrt(selected[:, 2] * selected[:, 4]) * 1e-5
    trials = []
    for point, h in zip(selected, steps, strict=True):
        for field in range(7):
            plus, minus = point.copy(), point.copy()
            plus[field] += h[field]
            minus[field] -= h[field]
            trials.extend((plus, minus))
    return np.asarray(trials), steps


def empty_spin_neighbors(
    points: NDArray[np.float64], labels: list[str]
) -> tuple[NDArray[np.float64], list[str]]:
    """Include both neighboring majority floats at each exact-empty spin."""
    rows = []
    names = []
    for label, majority in (("empty-alpha", 1), ("empty-beta", 0)):
        original = points[labels.index(label)]
        rows.append(original.copy())
        names.append(label)
        for direction in (-np.inf, np.inf):
            adjacent = original.copy()
            adjacent[majority] = np.nextafter(adjacent[majority], direction)
            rows.append(adjacent)
            names.append(label + ("-ulp-down" if direction < 0 else "-ulp-up"))
    return np.asarray(rows), names


def validate(
    backend: str,
    work: Path,
    cuda_arch: str | None,
    required_libxc: str,
    libxc_archive: Path,
) -> dict[str, object]:
    from pyscf.dft import libxc

    from tools.generate_xc_split_hybrid_cuda import emit_split_hybrid_device
    from tools.split_hybrid_wide_oracle import evaluate as evaluate_wide

    if required_libxc != "7.0.0":
        raise ValueError("the independent wide oracle requires Libxc 7.0.0")
    version = str(libxc.libxc_version())
    if version != required_libxc:
        raise RuntimeError(
            f"expected Libxc {required_libxc}, got {version}; use the pinned oracle"
        )
    compiler = shutil.which("nvcc" if backend == "cuda" else "c++")
    if compiler is None:
        raise RuntimeError(f"{backend} acceptance compiler is unavailable")
    if backend == "cuda" and (
        cuda_arch is None or re.fullmatch(r"sm_[0-9]+[a-z]?", cuda_arch) is None
    ):
        raise ValueError(
            "CUDA acceptance requires --cuda-arch sm_<actual-device-capability>"
        )
    work.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, filename, _, _ in METHODS:
        source = emit_split_hybrid_device(name)
        (work / filename).write_text(source, encoding="utf-8")
        hashes[name] = hashlib.sha256(source.encode()).hexdigest()
    original = ROOT / "tests/native/split_hybrid_point_probe.cpp"
    source = work / ("probe.cu" if backend == "cuda" else "probe.cpp")
    shutil.copyfile(original, source)
    executable = work / "split-hybrid-points"
    flags = (
        ["-O2", "--fmad=false", "-lineinfo", f"-arch={cuda_arch}"]
        if backend == "cuda"
        else ["-O1", "-fno-fast-math", "-ffp-contract=off"]
    )
    command = [
        compiler,
        "-std=c++17",
        f"-DVIBEQC_VALIDATE_CUDA={int(backend == 'cuda')}",
        *flags,
        "-I" + str(work),
        str(source),
        "-o",
        str(executable),
    ]
    subprocess.run(command, check=True, text=True, capture_output=True, timeout=300)
    points, labels, count = point_inputs()
    boundary, boundary_labels = empty_spin_neighbors(points, labels)
    wide_compiler = shutil.which("c++")
    if wide_compiler is None:
        raise RuntimeError("wide Libxc boundary oracle requires a host C++ compiler")
    wide, wide_identity = evaluate_wide(
        libxc_archive, wide_compiler, boundary, work / "wide-oracle"
    )
    fhc_point = points[labels.index("fermi-hole-boundary") :][:1].copy()
    clipped = fhc_point.copy()
    clipped[0, 2] = min(clipped[0, 2], 8.0 * clipped[0, 0] * clipped[0, 5])
    clipped[0, 4] = min(clipped[0, 4], 8.0 * clipped[0, 1] * clipped[0, 6])
    for _, _, _, oracle_code in METHODS:
        if not np.array_equal(
            reference_points(oracle_code, fhc_point),
            reference_points(oracle_code, clipped),
        ):
            raise RuntimeError("pinned Libxc oracle does not enforce global FHC")
    identity = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    compiler_version = subprocess.run(
        [compiler, "--version"], text=True, capture_output=True, check=True
    ).stdout.strip()
    dirty = (
        subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--"], cwd=ROOT, check=False
        ).returncode
        != 0
    )
    report = {
        "backend": backend,
        "libxc_version": version,
        "libxc_global_fhc_verified": True,
        "git_sha": identity,
        "compiler_version": compiler_version,
        "dirty_tracked_sources": dirty,
        "seed": SEED,
        "point_count": len(points),
        "boundary_labels": boundary_labels,
        "wide_oracle": wide_identity,
        "compile_command": command,
        "header_sha256": hashes,
        "methods": {},
        "passed": not dirty,
    }
    for name, _, code, oracle_code in METHODS:
        actual, device = run_probe(executable, code, points, work, name)
        expected = reference_points(oracle_code, points)
        result = comparison(actual, expected)
        binary64_full = result if name == "M06-2X" else None
        if name == "M06-2X":
            # The compiled binary64 reference loses minority precision at
            # exactly empty spins: its gate is disjoint from the 113-bit
            # source gate there. Every removed row is checked by the wide
            # oracle below; all other binary64 rows remain mandatory.
            ordinary = [
                index
                for index, label in enumerate(labels)
                if label not in ("empty-alpha", "empty-beta")
            ]
            result = comparison(actual[ordinary], expected[ordinary])
            result["binary64_full_diagnostic"] = binary64_full
            result["wide_reference_points"] = ["empty-alpha", "empty-beta"]
        result["device"] = device
        if "worst_point" in result:
            result["worst_case"] = (
                labels[ordinary[result["worst_point"]]]
                if name == "M06-2X"
                else labels[result["worst_point"]]
            )
        result["interior"] = comparison(actual[:count], expected[:count])
        if name == "M06-2X":
            generated_boundary, _ = run_probe(
                executable, code, boundary, work, name + "-wide-boundary"
            )
            result["high_precision_boundary"] = comparison(generated_boundary, wide)
            if "worst_point" in result["high_precision_boundary"]:
                result["high_precision_boundary"]["worst_case"] = boundary_labels[
                    result["high_precision_boundary"]["worst_point"]
                ]
            result["passed"] = bool(
                result["passed"] and result["high_precision_boundary"]["passed"]
            )
            np.savez(
                work / f"{name}-wide-boundary-evidence.npz",
                points=boundary,
                actual=generated_boundary,
                oracle=wide,
                labels=np.asarray(boundary_labels),
            )
        trials, steps = finite_difference_points(points)
        evaluated, _ = run_probe(executable, code, trials, work, name + "-fd")
        fd = (
            evaluated[:, 0].reshape(4, 7, 2)[:, :, 0]
            - evaluated[:, 0].reshape(4, 7, 2)[:, :, 1]
        ) / (2 * steps)
        fd_error = np.abs(fd - actual[:4, 1:]) / (2e-6 + 2e-5 * np.abs(actual[:4, 1:]))
        result["finite_difference_passed"] = bool(
            np.all(np.isfinite(fd_error)) and np.max(fd_error) <= 1.0
        )
        result["finite_difference_max_normalized_error"] = (
            float(np.max(fd_error)) if np.all(np.isfinite(fd_error)) else None
        )
        rks_index = labels.index("restricted-embedding")
        rks_rho = np.sum(libxc_density(points[rks_index : rks_index + 1]), axis=0)
        rks_exc, rks_vxc, _, _ = libxc.eval_xc(oracle_code, rks_rho, spin=0, deriv=1)
        rks_expected = np.array(
            [
                float(rks_exc[0] * rks_rho[0, 0]),
                float(rks_vxc[0][0]),
                float(rks_vxc[1][0]),
                float(rks_vxc[3][0]),
            ]
        )
        row = actual[rks_index]
        rks_actual = np.array(
            [
                row[0],
                (row[1] + row[2]) / 2,
                (row[3] + row[4] + row[5]) / 4,
                (row[6] + row[7]) / 2,
            ]
        )
        result["restricted_embedding_passed"] = bool(
            np.all(np.isfinite(rks_actual))
            and np.allclose(rks_actual, rks_expected, rtol=3e-7, atol=3e-8)
        )
        result["passed"] = bool(
            result["passed"]
            and result["finite_difference_passed"]
            and result["restricted_embedding_passed"]
        )
        np.savez(
            work / f"{name}-evidence.npz",
            points=points,
            actual=actual,
            oracle=expected,
            finite_difference=fd,
            labels=np.asarray(labels),
        )
        report["methods"][name] = result
        report["passed"] = report["passed"] and result["passed"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("host", "cuda"), required=True)
    parser.add_argument("--cuda-arch")
    parser.add_argument("--require-libxc", default="7.0.0")
    parser.add_argument("--libxc-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    work = output.parent / ("points-" + args.backend)
    try:
        report = validate(
            args.backend, work, args.cuda_arch, args.require_libxc, args.libxc_archive
        )
    # Fail closed and retain diagnostics for unexpected compiler/oracle errors.
    except Exception as error:  # noqa: BLE001
        report = {
            "backend": args.backend,
            "passed": False,
            "error": f"{type(error).__name__}: {error}",
        }
        if isinstance(error, subprocess.CalledProcessError):
            report["stdout"] = error.stdout
            report["stderr"] = error.stderr
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"passed": report["passed"], "report": str(output)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
