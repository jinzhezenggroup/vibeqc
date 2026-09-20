"""Independent libcint checks of direct packed generation, whitening and J.

Small correctness diagnostics only. The CPU oracle supplies the metric map to
this stand-alone consumer; production must reuse the native cuSOLVER owner.
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import typing
from pathlib import Path

_BENCHMARKS_DIR = next(
    parent for parent in Path(__file__).resolve().parents if parent.name == "benchmarks"
)
sys.path.insert(0, str(_BENCHMARKS_DIR))
import numpy as np
from _retention import raw_output_path
from vibeqc.resources import (
    ResourceBudget,
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourceRequest,
    plan_resources,
)

from tools.validate_df_source import fixture_systems, write_input
from tools.vibeqc_validation.df_gradient import reference_df_matrices
from tools.vibeqc_validation.f_shell_numerics import numerical_error
from tools.vibeqc_validation.schema import canonical_hash, file_hash


def dump(path: typing.Any, value: typing.Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(args: typing.Any) -> None:
    """Preflight the whole diagnostic arena, then persist results before gates."""
    if not os.environ.get("SLURM_JOB_ID") or args.output.exists():
        raise RuntimeError(
            "fresh output directory and finite Slurm allocation required"
        )
    args.output.mkdir(parents=True)
    fixtures = [
        ("cartesian", fixture_systems("cartesian"), 1e-12),
        ("spherical", fixture_systems("spherical"), 1e-12),
        (
            "long-contractions",
            fixture_systems("cartesian", long_contractions=True),
            1e-12,
        ),
        ("truncated", fixture_systems("spherical"), 0.2),
    ]
    mixed = copy.deepcopy(fixtures[0][1])
    for _, auxiliary in mixed:
        auxiliary["basis_representation"] = "spherical"
    fixtures.append(("unequal-mixed-representation", mixed, 1e-12))
    report = {
        "scope": "direct packed raw/whitening/J correctness only; no SCF or latency qualification",
        "executable_sha256": file_hash(args.executable),
        "native_library_sha256": file_hash(args.library),
        "producer_source_sha256": file_hash(Path(__file__).with_name("producer.cpp")),
        "runner_sha256": file_hash(__file__),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gates": {"raw_metric_atol": 2e-10, "B_J_atol": 2e-9, "rtol": 2e-10},
        "runs": [],
    }
    dump(args.output / "report.json", report)
    if args.only:
        unknown = set(args.only) - {fixture[0] for fixture in fixtures}
        if unknown:
            raise ValueError(f"unknown fixtures: {unknown}")
        fixtures = [fixture for fixture in fixtures if fixture[0] in args.only]
    for name, systems, threshold in fixtures:
        directory = args.output / name
        directory.mkdir()
        write_input(directory / "basis.txt", systems)
        xs, ds, expected, ranks = [], [], [], []
        for item, (orbital, auxiliary) in enumerate(systems):
            raw, metric, _, _ = reference_df_matrices(orbital, auxiliary)
            eigenvalues, eigenvectors = np.linalg.eigh(metric)
            retained = eigenvalues > threshold * eigenvalues.max()
            inverse = (
                eigenvectors[:, retained] / np.sqrt(eigenvalues[retained])
            ) @ eigenvectors[:, retained].T
            ranks.append(int(retained.sum()))
            n, _, a = raw.shape
            # Deliberately nonsymmetric D exercises D_mn+D_nm exactly; no
            # physical-density assumption may hide an off-diagonal factor.
            density = np.fromfunction(
                lambda i, j: 0.3 * (i == j) + 0.043 / (1 + i + 2 * j), (n, n)
            )
            density *= item + 1
            b = raw @ inverse
            j = np.einsum("ijp,klp,kl->ij", b, b, density, optimize=True)
            indices = np.tril_indices(n)
            expected.append(
                {
                    "raw": raw[indices],
                    "B": b[indices],
                    "J": j[indices],
                    "metric": metric,
                }
            )
            xs.append(inverse)
            ds.append(density)
        np.asarray(xs, dtype="<f8").tofile(directory / "X.bin")
        np.asarray(ds, dtype="<f8").tofile(directory / "D.bin")
        pairs = n * (n + 1) // 2
        numeric = (2 * pairs * a + a * a + 2 * pairs + a) * 8
        identity = ResourceIdentity(
            "packed DF producer diagnostic",
            "native public-basis source",
            "cuda",
            "fp64",
            json.dumps(
                {
                    "n": n,
                    "a": a,
                    "systems": canonical_hash(systems),
                    "threshold": threshold,
                }
            ),
            ("raw A", "B", "J"),
            "direct lower rows; all-Q whitening; unit-weight packed J",
        )
        estimates = (
            ResourceEstimate(
                "one active item's packed numeric buffers", numeric, "device:0", 0, 1
            ),
            ResourceEstimate(
                "bounded source metadata/setup allowance", 128 << 20, "device:0", 0, 1
            ),
            ResourceEstimate(
                "cuBLAS library allowance",
                64 << 20,
                "device:0",
                0,
                1,
                kind="library",
                accounting="runtime_allowance",
            ),
            ResourceEstimate(
                "bounded oracle and output host arrays", 128 << 20, "pageable", 0, 1
            ),
        )
        request = ResourceRequest(
            name,
            identity,
            (ResourceCandidate("direct-packed", "resident", estimates),),
            scope_exclusions=("CUDA context and loaded modules", "host runtime"),
        )
        plan = plan_resources(
            [request], ResourceBudget(device_bytes=1 << 30, host_bytes=512 << 20)
        ).require_feasible()
        dump(directory / "resource-plan.json", plan.to_dict())
        command = [
            str(args.executable.resolve()),
            str(directory / "basis.txt"),
            str(directory / "X.bin"),
            str(directory / "D.bin"),
            str(directory / "native"),
            str(1 << 30),
        ]
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = (
            str(args.library.resolve().parent) + ":" + env.get("LD_LIBRARY_PATH", "")
        )
        result = subprocess.run(
            command, capture_output=True, text=True, env=env, check=False
        )
        (directory / "native.log").write_text(result.stdout + result.stderr)
        row = {
            "name": name,
            "returncode": result.returncode,
            "command": command,
            "threshold": threshold,
            "ranks": ranks,
            "errors": [],
            "passed": False,
        }
        report["runs"].append(row)
        dump(args.output / "report.json", report)
        if result.returncode:
            raise RuntimeError(f"producer failed: {name}")
        row["runtime"] = [json.loads(line) for line in result.stdout.splitlines()]
        metrics = np.fromfile(directory / "native-metric.bin", dtype="<f8").reshape(
            len(systems), a, a
        )
        for item, reference in enumerate(expected):
            errors = {}
            for key, oracle in reference.items():
                actual = (
                    metrics[item]
                    if key == "metric"
                    else np.fromfile(
                        directory / f"native-{item}-{key}.bin", dtype="<f8"
                    ).reshape(oracle.shape)
                )
                errors[key] = numerical_error(
                    actual,
                    oracle,
                    atol=2e-10 if key in ("raw", "metric") else 2e-9,
                    rtol=2e-10,
                )
            row["errors"].append(errors)
        row["passed"] = all(
            error["passed"] for item in row["errors"] for error in item.values()
        )
        if name == "truncated":
            row["passed"] &= all(rank < a for rank in ranks)
        row["passed"] &= all(
            item["source_device_bytes"] <= 128 << 20 for item in row["runtime"]
        )
        dump(args.output / "report.json", report)
        print(name, row["passed"], flush=True)
        if not row["passed"]:
            raise RuntimeError(f"independent producer gate failed: {name}")
    report["passed"] = True
    dump(args.output / "report.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=raw_output_path, required=True)
    parser.add_argument("--only", nargs="+")
    main(parser.parse_args())
