"""Audit the experimental #149 helper on real CUDA: bounded GPU solver convergence, failure and replay.

This is a converged molecular endpoint, *not* the #149 A fixed-amplitude kernel
parity (``tools/validate_cc_cuda``). The GPU solver uses the exact #148 control
law and physical equations; only the residual/energy evaluation backend moves
to the #146 executor. Every accepted root is re-checked on the freshly expanded
physical DAG and its state file re-reproduces on the CPU solver.
"""

# Source-tree CLI bootstrap; importing the compiler needs no native runtime.
import sys as _compiler_sys
import typing
from pathlib import Path as _CompilerPath

_compiler_sys.path.insert(
    0, str(_CompilerPath(__file__).resolve().parents[1] / "python")
)

import argparse
import json
import platform
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.integral.cuda_target import cuda_target_info
from vibeqc_compiler.tensor.cuda_execute import compile_cuda, tensor_source_identity

from tools.cc_endpoint_fixtures import load, snapshot_from_fixture
from tools.replay_ccsd import replay
from tools.vibeqc_cc.gpu_solver import solve_gpu
from tools.vibeqc_cc.gpu_state import solver_plans
from tools.vibeqc_cc.solver import SolverOptions
from tools.vibeqc_posthf.providers import BlockResult, ConventionalProvider
from tools.vibeqc_validation.schema import file_hash

ROOT = Path(__file__).resolve().parents[1]


class FixtureProvider(ConventionalProvider):
    """Exact committed MO integrals; exercises the GPU solver without native AO."""

    def __init__(self, snapshot: typing.Any, g: typing.Any) -> None:
        self.snapshot = snapshot
        self.g = g
        self.backend = "cpu"
        self.source = SimpleNamespace(_check_open=lambda: None)

    def get(self, block: typing.Any) -> typing.Any:
        return BlockResult(
            block,
            self.g[np.ix_(*block.slots)],
            self.snapshot.identity,
            self.snapshot.hamiltonian_id,
            {},
        )


def run(
    output: typing.Any,
    compiler: typing.Any,
    cache: typing.Any,
    *,
    compile_only: typing.Any = False,
) -> typing.Any:
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        p.relative_to(ROOT).as_posix(): file_hash(p)
        for p in [
            *sorted((ROOT / "tools/vibeqc_cc").glob("*.py")),
            Path(__file__),
        ]
    }
    target = compiler.target.architecture
    manifest = {
        "scope": "Experimental host-staged CUDA validation helper; #149 B/C remain open",
        "base_dependency": "#148 CPU solver + #146 FP64 CUDA executor",
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
        ),
        "nvcc_version": subprocess.check_output(
            [str(compiler.nvcc), "--version"], text=True
        ).strip(),
        "target_architecture": target,
        "source_files": sources,
        "tensor_source_identity": tensor_source_identity(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "compile_only": compile_only,
        "cases": [],
    }
    for name in ("h2", "he", "h2o", "nh3", "ch4"):
        meta, a = load(name)
        source = SimpleNamespace(
            electron_count=int(a["occ"].sum()),
            geometry_hash="endpoint-" + name,
            basis_hash="endpoint-basis-" + name,
        )
        snapshot = snapshot_from_fixture(source, meta, a)
        provider = FixtureProvider(snapshot, a["g"])
        options = SolverOptions(residual_tolerance=1e-10, energy_tolerance=1e-12)
        if compile_only:
            primary, independent, diagnostic = solver_plans(
                snapshot.nocc, snapshot.nmo - snapshot.nocc, compiler.target, options
            )
            artifacts = [
                compile_cuda(plan, compiler, cache) for plan in (primary, independent)
            ]
            manifest["cases"].append(
                {
                    "name": name,
                    "executed": False,
                    "compiled_binary_sha256": [
                        file_hash(artifact.library) for artifact in artifacts
                    ],
                    "combined_peak_bytes": diagnostic["combined_peak_bytes"],
                }
            )
            print(
                f"compiled {name} (no device execution or numerical acceptance)",
                flush=True,
            )
            continue
        result = solve_gpu(
            snapshot,
            provider,
            compiler=compiler,
            cache=cache,
            options=options,
        )
        record = {
            "name": name,
            "status": result.status,
            "reason": result.reason,
            "converged": result.converged,
            "iterations": len(result.history),
            "correlation_energy": result.correlation_energy,
            "total_energy": result.total_energy,
            "reference_total_energy": meta["total_energy"],
            "energy_error": (
                None
                if result.total_energy is None
                else abs(result.total_energy - meta["total_energy"])
            ),
            "final_independent_r1_max": (
                result.history[-1].get("independent_r1_max") if result.history else None
            ),
            "final_independent_r2_max": (
                result.history[-1].get("independent_r2_max") if result.history else None
            ),
            "backend": result.provenance.get("backend"),
            "device": result.provenance.get("device"),
            "residency": result.provenance.get("residency"),
            "combined_peak_bytes": result.provenance.get("combined_peak_bytes"),
            "primary_peak_bytes": result.provenance.get("primary_peak_bytes"),
            "independent_replay_peak_bytes": result.provenance.get(
                "independent_replay_peak_bytes"
            ),
            "transfer": result.provenance.get("transfer"),
            "integral_hash": result.provenance.get("integral_hash"),
            "equation_hash": result.provenance.get("equation_hash"),
            "iteration_equation_hash": result.provenance.get("iteration_equation_hash"),
        }
        # Failure/replay parity: the state file must re-reproduce on CPU, within
        # cross-backend FP64 rounding (CPU BLAS vs cuBLAS, no exact bit guarantee).
        state_path = output / f"{name}-state.json"
        result.write(state_path)
        reproduced = replay(state_path)
        record["replay_status"] = reproduced.status
        record["replay_t1_max_diff"] = float(np.max(np.abs(reproduced.t1 - result.t1)))
        record["replay_t2_max_diff"] = float(np.max(np.abs(reproduced.t2 - result.t2)))
        record["replay_energy_diff"] = (
            None
            if reproduced.total_energy is None or result.total_energy is None
            else abs(reproduced.total_energy - result.total_energy)
        )
        record["replay_matches"] = (
            reproduced.status == result.status
            and record["replay_t1_max_diff"] <= 1e-10
            and record["replay_t2_max_diff"] <= 1e-10
            and (record["replay_energy_diff"] or 0) <= 1e-10
        )
        record["passed"] = (
            result.converged
            and result.total_energy is not None
            and abs(result.total_energy - meta["total_energy"]) <= 1e-8
            and max(
                result.history[-1]["independent_r1_max"],
                result.history[-1]["independent_r2_max"],
            )
            <= 1e-10
            and record["replay_matches"]
        )
        step = output / f"{name}.json"
        step.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        manifest["cases"].append({"file": f"{name}.json", "passed": record["passed"]})
        print(
            f"{name}: {'converged' if result.converged else result.status} "
            f"{len(result.history)} iters, dE={record['energy_error']:.3e}, "
            f"replay={record['replay_matches']}",
            flush=True,
        )
    manifest["passed"] = bool(
        not compile_only
        and all(c.get("passed") for c in manifest["cases"] if "file" in c)
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    report = run(
        args.output,
        CudaCompilerAdapter(args.nvcc, cuda_target_info(args.architecture)),
        args.cache,
        compile_only=args.compile_only,
    )
    raise SystemExit(0 if args.compile_only or report["passed"] else 1)
