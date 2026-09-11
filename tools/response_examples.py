"""Run bounded RHF response-solver numerical/resource evidence.

The CPU path exercises the shared matrix-free native shell-tile J/K operator.
``--device cuda`` exercises the streamed native CUDA density-fitting J/K plan
inside the caller's Slurm allocation and records the actual device/resource
diagnostics.  The same GMRES/recycling code is used for both paths.
"""

from __future__ import annotations

# Source-tree CLI bootstrap for transitive compiler clients.
import sys as _compiler_sys
from pathlib import Path as _CompilerPath

_compiler_sys.path.insert(
    0, str(_CompilerPath(__file__).resolve().parents[1] / "python")
)

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from tools.vibeqc_posthf.df import DFProvider, MetricFactor
from tools.vibeqc_posthf.fixtures import (
    fixture_snapshot,
    load_fixture,
    source_arguments,
)
from tools.vibeqc_posthf.providers import ConventionalProvider
from tools.vibeqc_posthf.sources import CudaDFSource, NativeSource
from tools.vibeqc_response import (
    CudaDFJKBackend,
    GMRESOptions,
    NativeJKBackend,
    RHFResponseOperator,
    explicit_rhf_response_matrix,
    finite_rotation_jvp,
    solve_many,
)


def _serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serializable(item) for item in value]
    return value


def _solve_record(result, expected=None):
    record = {
        "strategy": result.strategy,
        "converged": result.converged,
        "seconds": result.seconds,
        "operator_actions": result.operator_actions,
        "peak_workspace_bytes": result.peak_workspace_bytes,
        "rhs_rank": result.rhs_rank,
        "rank_deficient_rhs": result.rank_deficient_rhs,
        "solutions": [
            {
                "converged": item.converged,
                "reason": item.reason,
                "iterations": item.iterations,
                "residual_norm": item.residual_norm,
                "relative_residual": item.relative_residual,
                "operator_actions": item.operator_actions,
                "recycled_vectors": item.recycled_vectors,
                "orthogonalization_seconds": item.orthogonalization_seconds,
                "operator_seconds": item.operator_seconds,
                "workspace_bytes": item.workspace_bytes,
            }
            for item in result.results
        ],
    }
    if expected is not None:
        record["maximum_solution_error"] = float(
            np.max(np.abs(result.solution - expected))
        )
    return record


def _context(args, meta, arrays):
    """Return owned source, backend, snapshot, provider and backend metadata."""
    if args.device == "cuda":
        source = CudaDFSource(
            **source_arguments(meta), tile_capacity=args.df_tile_capacity
        )
        metric = MetricFactor.from_source(source)
        snapshot = fixture_snapshot(meta, arrays, label="df", metric=metric)
        backend = CudaDFJKBackend(
            source,
            device_id=args.device_id,
            metric_threshold=metric.relative_threshold,
            hamiltonian_id=metric.hamiltonian_id,
            metric=metric,
        )
        provider = DFProvider(
            snapshot,
            source,
            metric,
            axis_tile=args.axis_tile,
            auxiliary_tile=args.df_auxiliary_tile,
        )
        return source, backend, snapshot, provider, metric, "native-cuda-df-streamed-jk"
    source = NativeSource(**source_arguments(meta))
    snapshot = fixture_snapshot(meta, arrays)
    backend = NativeJKBackend(source, axis_tile=args.axis_tile)
    provider = ConventionalProvider(snapshot, source, axis_tile=args.axis_tile)
    return source, backend, snapshot, provider, None, "native-cpu-shell-tile-jk"


def run(args):
    meta, arrays = load_fixture(args.case)
    started = time.perf_counter()
    source, backend, snapshot, provider, metric, backend_label = _context(
        args, meta, arrays
    )
    try:
        problem = RHFResponseOperator.build_problem(snapshot, backend)
        operator = RHFResponseOperator(problem, backend)
        with provider:
            explicit = explicit_rhf_response_matrix(problem, provider)
            provider_statistics = dict(getattr(provider, "statistics", {}))
        rng = np.random.default_rng(args.seed)
        rhs = rng.normal(size=(problem.dimension, args.rhs_count))
        options = GMRESOptions(
            rtol=args.rtol,
            atol=args.atol,
            restart=args.restart,
            max_iterations=args.max_iterations,
            max_workspace_bytes=args.max_workspace_bytes,
        )
        expected = np.linalg.solve(explicit, rhs)
        strategies = {}
        for strategy in ("sequential", "blocked", "recycled"):
            result = solve_many(operator, rhs, strategy=strategy, options=options)
            strategies[strategy] = _solve_record(result, expected)
        left = rng.normal(size=problem.dimension)
        right = rng.normal(size=problem.dimension)
        dot_error = operator.dot_identity(left, right)
        finite = finite_rotation_jvp(
            problem, backend, left, step=args.finite_difference_step
        )
        finite_error = float(np.linalg.norm(finite - operator.apply(left)))
        source_metrics = (
            source.source_metrics()
            if hasattr(source, "source_metrics")
            else {"device_bytes": 0, "generation_ms": 0.0, "transfer_ms": 0.0}
        )
        not_run = {}
        if args.device == "cuda":
            not_run["native_gpu_direct_four_center_jk"] = (
                "the real-device path uses the streamed native CUDA DF J/K plan; "
                "a direct four-center GPU response backend is not promoted here"
            )
        return {
            "schema": "vibeqc.response-evidence",
            "version": 1,
            "case": args.case,
            "device": args.device,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "source_identity": source.identity,
            "reference_identity": snapshot.identity,
            "problem_identity": problem.identity,
            "operator_identity": operator.identity,
            "operator_backend": backend.identity,
            "operator_backend_label": backend_label,
            "matrix_free": True,
            "metric_identity": None if metric is None else metric.identity,
            "source_metrics": source_metrics,
            "backend_statistics": dict(getattr(backend, "statistics", {})),
            "provider_statistics": provider_statistics,
            "dimension": problem.dimension,
            "rhs_count": int(rhs.shape[1]),
            "rhs_rank": int(np.linalg.matrix_rank(rhs, tol=1e-12)),
            "dot_identity_error": dot_error,
            "finite_rotation_error": finite_error,
            "strategies": strategies,
            "not_run": not_run,
            "elapsed_seconds": time.perf_counter() - started,
        }
    finally:
        close = getattr(backend, "close", None)
        if close is not None:
            close()
        source.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="water", choices=("h2", "water", "lih"))
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--axis-tile", type=int, default=2)
    parser.add_argument("--df-tile-capacity", type=int, default=64)
    parser.add_argument("--df-auxiliary-tile", type=int, default=3)
    parser.add_argument("--rhs-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=179)
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--restart", type=int, default=20)
    parser.add_argument("--max-iterations", type=int, default=200)
    parser.add_argument("--max-workspace-bytes", type=int, default=64 << 20)
    parser.add_argument("--finite-difference-step", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_serializable(evidence), indent=2) + "\n")


if __name__ == "__main__":
    main()
