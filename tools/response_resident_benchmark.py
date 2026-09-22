"""Matched exact-CUDA response endpoints; run only in a Slurm GPU allocation.

Oracle preparation is separate from timing. Both execution modes use the same
committed reference and exact unscreened CUDA J/K plan. Every measured repeat
must satisfy the independent dense equation; no timing sample is filtered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from vibeqc import _native
from vibeqc.profiles import probe_device

from tools.vibeqc_posthf.fixtures import (
    fixture_snapshot,
    load_fixture,
    source_arguments,
)
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_response import (
    CudaDirectJKBackend,
    GMRESOptions,
    RHFResponseOperator,
    explicit_rhf_response_matrix,
    resident_vector_slots,
    solve_many,
)

STRATEGIES = ("sequential", "blocked", "recycled")
COUNTERS = (
    "h2d_bytes",
    "d2h_bytes",
    "synchronizations",
    "operator_actions",
    "blas_calls",
)


def _command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _oracle(reference: object, arrays: dict) -> np.ndarray:
    """Use independent committed Libcint AO integrals, never native J/K."""
    c = reference.coefficients
    mo = np.einsum("up,vq,wr,xs,uvwx->pqrs", c, c, c, c, arrays["ao"], optimize=True)
    # The oracle helper only needs the reference/layout and a MO-block reader.
    from tools.vibeqc_response import DenseAOResponseBackend

    backend = DenseAOResponseBackend(arrays["ao"])
    problem = RHFResponseOperator.build_problem(reference, backend)
    provider = SimpleNamespace(
        snapshot=reference,
        get=lambda block: SimpleNamespace(to_host=lambda: mo[np.ix_(*block.slots)]),
    )
    return explicit_rhf_response_matrix(problem, provider)


def _response_case(name: str, repeats: int) -> dict:
    meta, arrays = load_fixture(name)
    reference = fixture_snapshot(meta, arrays)
    matrix = _oracle(reference, arrays)
    rng = np.random.default_rng(179)
    rhs = rng.normal(size=(matrix.shape[0], 4))
    rhs[:, 3] = rhs[:, 0] + 2 * rhs[:, 1]
    expected = np.linalg.solve(matrix, rhs)
    options = GMRESOptions(rtol=1e-11, max_iterations=40)
    records = []
    for strategy in STRATEGIES:
        for execution in ("host", "cuda-resident"):
            started = time.perf_counter()
            samples = []
            with ExitStack() as stack:
                source = stack.enter_context(NativeSource(**source_arguments(meta)))
                backend = stack.enter_context(CudaDirectJKBackend(source))
                problem = RHFResponseOperator.build_problem(reference, backend)
                operator = RHFResponseOperator(problem, backend)
                owner = None
                setup_counters = None
                slots = 0
                if execution == "cuda-resident":
                    slots = resident_vector_slots(
                        problem.dimension,
                        options,
                        rhs_count=rhs.shape[1],
                        strategy=strategy,
                    )
                    owner = stack.enter_context(
                        backend.resident_response(
                            problem, vector_slots=slots, device_budget_bytes=16 << 20
                        )
                    )
                    operator._krylov_engine = owner
                    setup_counters = owner.diagnostics
                setup_seconds = time.perf_counter() - started
                for repeat in range(repeats):
                    before = owner.diagnostics if owner else dict(backend.statistics)
                    begin = time.perf_counter()
                    answer = solve_many(
                        operator,
                        rhs,
                        strategy=strategy,
                        options=options,
                        collect_basis=False,
                        raise_on_failure=True,
                    )
                    # Include assembly of the caller-visible N-by-RHS solution.
                    solution = answer.solution
                    seconds = time.perf_counter() - begin
                    after = owner.diagnostics if owner else dict(backend.statistics)
                    residual = matrix @ solution - rhs
                    relative = float(
                        np.max(
                            np.linalg.norm(residual, axis=0)
                            / np.linalg.norm(rhs, axis=0)
                        )
                    )
                    error = float(np.max(np.abs(solution - expected)))
                    if relative > 1e-9 or error > 3e-9:
                        raise AssertionError(
                            f"{name}/{strategy}/{execution}: numerical gate failed"
                        )
                    if owner:
                        counters = {key: after[key] - before[key] for key in COUNTERS}
                        counter_source = "native resident diagnostic deltas"
                        assert counters["operator_actions"] == answer.operator_actions
                        assert counters["h2d_bytes"] == rhs.nbytes
                        assert not owner._live and not owner._retained
                    else:
                        calls = after["actions"] - before["actions"]
                        assert calls == answer.operator_actions
                        # Successful restricted exact J/K evaluates one density,
                        # downloads two matrices, and completes one stream fence.
                        # Derived from direct_jk.cpp, not profiler measurements.
                        counters = {
                            "h2d_bytes": calls * source.nbf**2 * 8,
                            "d2h_bytes": calls * source.nbf**2 * 16,
                            "synchronizations": calls,
                            "operator_actions": calls,
                            "blas_calls": None,
                        }
                        counter_source = "observed successful J/K calls times audited restricted API payload; excludes setup/teardown"
                    samples.append(
                        {
                            "repeat": repeat,
                            "solve_endpoint_seconds": seconds,
                            "solver_reported_seconds": answer.seconds,
                            "operator_seconds": answer.operator_seconds,
                            "orthogonalization_seconds": answer.orthogonalization_seconds,
                            "recycling_seconds": answer.recycling_seconds,
                            "counters": counters,
                            "counter_source": counter_source,
                            "peak_workspace_bytes": answer.peak_workspace_bytes,
                            "maximum_relative_residual": relative,
                            "maximum_solution_error": error,
                            "iterations": [item.iterations for item in answer.results],
                        }
                    )
                resources = {
                    "jk_retained_device_bytes": backend.device_resident_bytes,
                    "response_retained_device_bytes": 0
                    if owner is None
                    else owner.workspace_bytes,
                    "vector_slots": slots,
                    "solver_budget_bytes": options.max_workspace_bytes,
                }
                identities = {
                    "source": source.identity,
                    "reference": reference.identity,
                    "operator": operator.identity,
                    "provider": backend.identity,
                    "resident": None if owner is None else owner.identity,
                }
                teardown_started = time.perf_counter()
            teardown_seconds = time.perf_counter() - teardown_started
            records.append(
                {
                    "strategy": strategy,
                    "execution": execution,
                    "setup_seconds": setup_seconds,
                    "teardown_seconds": teardown_seconds,
                    "cold_endpoint_seconds": setup_seconds
                    + samples[0]["solve_endpoint_seconds"]
                    + teardown_seconds,
                    "setup_resident_counters": setup_counters,
                    "resources": resources,
                    "identities": identities,
                    "samples": samples,
                }
            )
    return {
        "case": name,
        "nbf": reference.nmo,
        "dimension": matrix.shape[0],
        "rhs_count": rhs.shape[1],
        "rhs_sha256": hashlib.sha256(rhs.tobytes()).hexdigest(),
        "options": asdict(options),
        "records": records,
    }


def _consumer() -> list[dict]:
    """Gate complete HVPs with PySCF; retain native assembly parity separately."""
    import pyscf

    from tools.vibeqc_hessian import NativeRHFState, analytic_hessian, rhf_hvp_many
    from tools.vibeqc_validation.hessian_fixtures import (
        fixture_inputs,
        oracle_analytic_hessian,
    )

    records = []
    with NativeSource(**fixture_inputs("h2")) as source:
        state = NativeRHFState.from_source(source)
        native_dense = analytic_hessian(state)["total"]
        dense = oracle_analytic_hessian(source)
        directions = np.random.default_rng(1804).normal(size=(3, state.nat, 3))
        directions /= np.linalg.norm(directions.reshape(3, -1), axis=1)[:, None, None]
        expected = np.stack(
            [np.einsum("abxy,by->ax", dense, vector) for vector in directions]
        )
        native_expected = np.stack(
            [np.einsum("abxy,by->ax", native_dense, vector) for vector in directions]
        )
        for strategy in STRATEGIES:
            for execution in ("host", "cuda-resident"):
                begin = time.perf_counter()
                result = rhf_hvp_many(
                    state,
                    directions,
                    strategy=strategy,
                    jk_backend="cuda",
                    response_execution=execution,
                )
                seconds = time.perf_counter() - begin
                error = float(np.max(np.abs(result.values - expected)))
                native_error = float(np.max(np.abs(result.values - native_expected)))
                if not np.isfinite(error) or error > 1e-9:
                    raise AssertionError(
                        "complete HVP independent numerical gate failed"
                    )
                if not np.isfinite(native_error) or native_error > 1e-9:
                    raise AssertionError("complete HVP native assembly parity failed")
                records.append(
                    {
                        "strategy": strategy,
                        "execution": execution,
                        "complete_hvp_seconds": seconds,
                        "maximum_error": error,
                        "native_dense_maximum_error": native_error,
                        "oracle": {
                            "implementation": "pyscf.hessian.rhf.Hessian.kernel",
                            "version": pyscf.__version__,
                            "maximum_error_gate": 1e-9,
                            "basis": "exact source shell primitives; Cartesian/Bohr",
                        },
                        "directions": directions.tolist(),
                        "oracle_hvp": expected.tolist(),
                        "hvp": result.values.tolist(),
                        "diagnostics": result.diagnostics,
                    }
                )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--consumer", action="store_true")
    parser.add_argument(
        "--consumer-only",
        action="store_true",
        help="qualify all six complete HVP endpoints without rerunning response timing",
    )
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("a Slurm GPU allocation is required")
    if args.repeats < 2:
        parser.error("at least one cold and one warm solve are required")
    library = _native.load_library()
    lib = Path(library._name).resolve()
    root = Path(__file__).resolve().parents[1]
    revision = _command("git", "rev-parse", "HEAD")
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=root)
    # Staged additions are included. Untracked files are never silently claimed
    # as reproducible source; the runner itself must be tracked before running.
    _command(
        "git", "ls-files", "--error-unmatch", "tools/response_resident_benchmark.py"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if patch:
        args.output.with_suffix(".patch").write_bytes(patch)
    result = {
        "schema": "vibeqc.response-resident-evidence/v2",
        "source": {
            "revision": revision,
            "dirty": bool(patch),
            "patch_sha256": hashlib.sha256(patch).hexdigest() if patch else None,
        },
        "native_binary": {
            "path": str(lib),
            "sha256": _hash(lib),
            "probe": probe_device(library, 0),
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "slurm_job": os.environ["SLURM_JOB_ID"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": _command(
                "nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"
            ),
            "cuda": _command("nvcc", "--version"),
            "openblas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        },
        "scope": {
            "hamiltonian": "conventional-unscreened exact FP64 CUDA J/K in both modes",
            "preparation": "fixture loading, RHS and independent oracle excluded; source/backend/resident creation included in setup",
            "warm": "same owner, repeated RHS; each recycled solve starts with an empty space and recycles only within its four RHS",
            "cold_endpoint": "plan setup + first solve/publication + final owner teardown; excludes process startup and oracle; the first provider setup may include lazy CUDA context initialization",
            "publication": "one solution per RHS; final Arnoldi basis suppressed",
            "host_boundary": "RHS validation/rank diagnostic, projected scalars/SVD/least squares and convergence remain host controlled",
            "component_timing": "action-only operator; basis/projection/range orthogonalization; retained-space recycling; other costs only in full solve time",
            "memory": "retained J/K + response arena and conservative logical solver reservation; excludes CUDA context/libraries and provider setup temporaries",
            "consumer": "H2 native state, independent PySCF analytic Hessian and native dense parity reference prepared before complete HVP timing; first/second derivatives and final assembly use declared CPU defaults",
            "decision": "qualification only; no production-size, changed-geometry performance, default selection or speedup promotion",
        },
        "response": [],
        "consumer": [],
    }
    for name in () if args.consumer_only else ("h2", "lih", "water"):
        result["response"].append(_response_case(name, args.repeats))
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(f"{name}: all matched response samples passed", flush=True)
    if args.consumer or args.consumer_only:
        result["consumer"] = _consumer()
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
