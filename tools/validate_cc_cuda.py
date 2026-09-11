"""Audit #149 A on real CUDA: fixed nonzero T, all nodes, two shapes/budgets.

This is kernel parity evidence, never a converged molecular endpoint. CPU
interpreter and saved independent #148 references use exactly the same feeds.
"""

# Source-tree CLI bootstrap; importing the compiler needs no native runtime.
import sys as _compiler_sys
from pathlib import Path as _CompilerPath

_compiler_sys.path.insert(
    0, str(_CompilerPath(__file__).resolve().parents[1] / "python")
)

import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.integral.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import execute
from vibeqc_compiler.tensor.cuda_execute import compile_cuda, tensor_source_identity
from vibeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda

from tools.validate_cc import load_references
from tools.vibeqc_cc.cuda import PreparedRCCSDResidual, rccsd_program
from tools.vibeqc_cc.oracle import dense_feeds
from tools.vibeqc_validation.schema import block_error, canonical_hash, file_hash

ROOT = Path(__file__).resolve().parents[1]


def run(output, compiler, cache, *, compile_only=False):
    output.mkdir(parents=True, exist_ok=True)
    reference_path = ROOT / "tests/reference_data/cc/rccsd-b.json"
    references = load_references(reference_path)
    sources = {
        p.relative_to(ROOT).as_posix(): file_hash(p)
        for p in [*sorted((ROOT / "tools/vibeqc_cc").glob("*.py")), Path(__file__)]
    }
    manifest = {
        "scope": "#149 A fixed amplitudes only; B/C and molecular endpoints pending",
        "base_dependency": "PR #215 / 5f31c4289db59853e4f64a942a51b4cccc68a3cd",
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_files": sources,
        "tensor_source_identity": tensor_source_identity(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "reference_sha256": file_hash(reference_path),
        "reference_cases_hash": references["cases_hash"],
        "cases": [],
    }
    for case in references["cases"]:
        arrays = [
            np.asarray(case["inputs"][k], dtype=np.float64)
            for k in ("fock", "eri", "t1", "t2")
        ]
        feeds = dense_feeds(*arrays)
        shape = arrays[2].shape
        assert np.max(np.abs(arrays[3])) > 0
        program = rccsd_program(*shape, trace=True)
        cpu = execute(program, feeds).outputs
        external = {
            **case["intermediates"],
            "correlation_energy": case["correlation_energy"],
            "singles_residual": case["updates"][0]["residual"],
            "doubles_residual": case["updates"][0]["residual2"],
        }
        unit = plan_cuda(
            program,
            compiler.target,
            library_bytes=0,
            schedule=TensorSchedule(tile_m=1, tile_n=1, tile_k=1),
        )
        budgets = (("normal", 256 << 20, 4 << 20), ("minimum", unit.peak_bytes, 0))
        for label, budget, workspace in budgets:
            if compile_only:
                plan = plan_cuda(
                    program, compiler.target, max_bytes=budget, library_bytes=workspace
                )
                artifact = compile_cuda(plan, compiler, cache)
                print(
                    f"compiled {case['name']}-{label}: {artifact.metadata['key']}",
                    flush=True,
                )
                continue
            with PreparedRCCSDResidual(
                *shape,
                compiler,
                cache,
                trace=True,
                max_bytes=budget,
                library_bytes=workspace,
            ) as prepared:
                runs = [
                    prepared.execute(feeds, profile=profile)
                    for profile in (False, True, False)
                ]
                checks = {}
                for i, result in enumerate(runs):
                    for origin, values in (("cpu", cpu), ("external", external)):
                        for name, ref in values.items():
                            check = block_error(
                                result.outputs[name],
                                np.asarray(ref),
                                atol=1e-11,
                                rtol=1e-10,
                            )
                            check["passed"] &= check["max_absolute_error"] <= (
                                1e-8 if name == "correlation_energy" else 1e-9
                            )
                            checks[f"run{i}/{origin}/{name}"] = check
                record = {
                    "case": case["name"],
                    "shape": shape,
                    "budget": label,
                    "inputs_hash": canonical_hash(
                        {k: v.tolist() for k, v in feeds.items()}
                    ),
                    "nonzero_t2_max": float(np.max(np.abs(arrays[3]))),
                    "plan": prepared.plan.to_payload(),
                    "artifact": prepared.artifact.metadata,
                    "device": prepared.executor.device,
                    "metrics": [r.metrics for r in runs],
                    "transfers_per_execution": {
                        "h2d_bytes": sum(
                            prepared.plan.steps[i].node.spec.size * 8
                            for i in prepared.plan.inputs
                        ),
                        "d2h_bytes": sum(v.nbytes for v in runs[0].outputs.values())
                        + 4,
                        "scope": "numeric feeds, detached trace outputs and arithmetic error flag",
                    },
                    "checks": checks,
                    "passed": all(c["passed"] for c in checks.values()),
                }
                name = f"{case['name']}-{label}.json"
                (output / name).write_text(json.dumps(record, indent=2) + "\n")
                manifest["cases"].append({"file": name, "passed": record["passed"]})
                print(f"{name}: passed={record['passed']}", flush=True)
        try:
            plan_cuda(
                program, compiler.target, max_bytes=unit.peak_bytes - 1, library_bytes=0
            )
        except ValueError as error:
            manifest.setdefault("budget_failures", []).append(str(error))
        else:
            raise AssertionError("infeasible minimum budget accepted")
    manifest["compiled_only"] = compile_only
    manifest["passed"] = bool(manifest["cases"]) and all(
        c["passed"] for c in manifest["cases"]
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
