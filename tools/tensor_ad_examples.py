"""CUDA compile/numerical evidence for generated TensorIR JVP/VJP programs.

This is slice C of #151.  It builds a fixed, unconverged CC-like scalar
energy fragment, generates demand-driven JVP and VJP programs, and compares
their GPU execution against the independent CPU interpreter and directional
finite differences.  It never claims a complete CC method or solver.

GPU modes must run inside the caller's finite Slurm allocation.
"""

from __future__ import annotations

# Source-tree CLI bootstrap; importing the compiler needs no native runtime.
import sys as _compiler_sys
from pathlib import Path as _CompilerPath

_compiler_sys.path.insert(
    0, str(_CompilerPath(__file__).resolve().parents[1] / "python")
)

import argparse
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vibeqc.profiles import find_nvcc
from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.integral.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    einsum,
    execute,
    input_tensor,
    jvp,
    linearize,
    transpose,
    transpose_program,
    vjp,
)
from vibeqc_compiler.tensor.cuda_execute import (
    PreparedCuda,
    compile_cuda,
    tensor_source_identity,
)
from vibeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda

from tools.vibeqc_validation.schema import (
    GATES,
    block_error,
    canonical_hash,
    new_evidence,
    outcome,
    validate_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def fixture(nocc: int, nvir: int, *, seed: int = 151):
    """Fixed scalar-energy fragment with realistic unconverged amplitudes."""
    occupied = IndexSpace("occupied", "occupied", nocc)
    virtual = IndexSpace("virtual", "virtual", nvir)
    i, j = Index("i", occupied), Index("j", occupied)
    a, b, e = Index("a", virtual), Index("b", virtual), Index("e", virtual)

    def parameter(name, indices):
        return input_tensor(
            name,
            TensorSpec(
                indices,
                role="parameter",
                representation="spin_orbital",
                differentiable=True,
            ),
        )

    t = parameter("t", (i, j, a, b))
    f = parameter("f", (a, e))
    g = parameter("g", (i, j, a, b))
    term = einsum("ae,ijeb->ijab", f, t)
    residual = add(term, transpose(term, (0, 1, 3, 2)), coefficients=(1, -1))
    energy = einsum("ijab,ijab->", g, residual, coefficient="1/4")
    program = Program(
        {"energy": energy},
        provenance={
            "fixture_version": 1,
            "seed": seed,
            "nocc": nocc,
            "nvir": nvir,
            "scope": "fixed scalar CC-like fragment; no CC solver or molecular method",
        },
    )
    rng = np.random.default_rng(seed)
    shape = (nocc, nocc, nvir, nvir)
    feeds = {
        "t": rng.normal(scale=0.02, size=shape),
        "f": rng.normal(scale=0.1, size=(nvir, nvir)),
        "g": rng.normal(scale=0.02, size=shape),
    }
    tangents = {
        "t": rng.normal(scale=0.02, size=shape),
        "f": rng.normal(scale=0.1, size=(nvir, nvir)),
        "g": rng.normal(scale=0.02, size=shape),
    }
    return program, feeds, tangents


def cpu_references(program, feeds, tangents, cotangent):
    """Generate programs and check them against the interpreter/FD references."""
    forward = linearize(program, list(tangents), outputs=["energy"])
    reverse = transpose_program(program, ["energy"], inputs=list(tangents))
    forward_feeds = {
        **feeds,
        **{f"d_{name}": value for name, value in tangents.items()},
    }
    reverse_feeds = {**feeds, "bar_energy": cotangent}
    forward_outputs = execute(forward.program, forward_feeds).outputs
    reverse_outputs = execute(reverse.program, reverse_feeds).outputs
    reference_forward = jvp(program, feeds, tangents, outputs=["energy"])
    reference_reverse = vjp(
        program, feeds, {"energy": cotangent}, inputs=list(tangents)
    )
    np.testing.assert_allclose(
        forward_outputs["d_energy"],
        reference_forward.output_tangents["energy"],
        rtol=1e-12,
        atol=1e-12,
    )
    for name in tangents:
        np.testing.assert_allclose(
            reverse_outputs[f"bar_{name}"],
            reference_reverse.input_cotangents[name],
            rtol=1e-12,
            atol=1e-12,
        )
    finite_differences = []
    for step in (1e-3, 1e-4, 1e-5):
        plus = {name: feeds[name] + step * tangents[name] for name in feeds}
        minus = {name: feeds[name] - step * tangents[name] for name in feeds}
        derivative = (
            execute(program, plus).outputs["energy"]
            - execute(program, minus).outputs["energy"]
        ) / (2 * step)
        finite_differences.append(
            {
                "step": step,
                "error": block_error(
                    forward_outputs["d_energy"],
                    derivative,
                    atol=1e-8,
                    rtol=1e-6,
                ),
            }
        )
    lhs = float(cotangent * forward_outputs["d_energy"])
    rhs = sum(
        float(np.sum(reverse_outputs[f"bar_{name}"] * tangents[name]))
        for name in tangents
    )
    dot_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-30)
    assert dot_error <= 1e-10, (lhs, rhs, dot_error)
    return (
        forward,
        reverse,
        forward_feeds,
        reverse_feeds,
        forward_outputs,
        reverse_outputs,
        finite_differences,
        dot_error,
    )


def _plan(generated, target, *, max_bytes, recompute):
    schedule = TensorSchedule(
        views=True,
        fuse=True,
        recompute=recompute,
        direct_gemm=False,
        tile_m=32,
        tile_n=32,
        tile_k=32,
    )
    return plan_cuda(generated.program, target, schedule=schedule, max_bytes=max_bytes)


def _recomputation_count(plan):
    counts = Counter(step.node for step in plan.steps)
    return sum(count - 1 for count in counts.values() if count > 1)


def run(args):
    nvcc = args.nvcc or find_nvcc()
    if nvcc is None:
        raise ValueError("provide --nvcc or VIBEQC_NVCC")
    compiler = CudaCompilerAdapter(
        nvcc, cuda_target_info(args.architecture), args.compile_timeout
    )
    program, feeds, tangents = fixture(args.nocc, args.nvir, seed=args.seed)
    cotangent = np.asarray(args.cotangent)
    (
        forward,
        reverse,
        forward_feeds,
        reverse_feeds,
        forward_outputs,
        reverse_outputs,
        finite_differences,
        dot_error,
    ) = cpu_references(program, feeds, tangents, cotangent)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "primal.equation.json").write_text(program.dumps())
    (args.output / "jvp.equation.json").write_text(forward.program.dumps())
    (args.output / "vjp.equation.json").write_text(reverse.program.dumps())
    records = []
    if args.mode == "compile":
        for label, generated in (("jvp", forward), ("vjp", reverse)):
            plan = _plan(
                generated, compiler.target, max_bytes=args.max_bytes, recompute=True
            )
            artifact = compile_cuda(plan, compiler, args.cache)
            (args.output / f"{label}.plan.json").write_text(
                json.dumps(plan.to_payload(), indent=2, sort_keys=True) + "\n"
            )
            records.append((label, plan, artifact, None))
    else:
        for label, generated, reference in (
            ("jvp", forward, forward_outputs),
            ("vjp", reverse, reverse_outputs),
        ):
            plan = _plan(
                generated, compiler.target, max_bytes=args.max_bytes, recompute=True
            )
            artifact = compile_cuda(plan, compiler, args.cache)
            with PreparedCuda(plan, artifact, device=args.device) as prepared:
                result = prepared.execute(
                    forward_feeds if label == "jvp" else reverse_feeds
                )
                profile = prepared.execute(
                    forward_feeds if label == "jvp" else reverse_feeds,
                    profile=True,
                )
                record = new_evidence(
                    tier="gpu-numerical",
                    subject=f"TensorIR/#151 generated {label.upper()}",
                    inputs_hash=canonical_hash(
                        {
                            "primal": program.logical_hash,
                            "derivative": generated.program.logical_hash,
                            "seed": args.seed,
                            "nocc": args.nocc,
                            "nvir": args.nvir,
                        }
                    ),
                )
                record.update(
                    revision=revision,
                    backend_selected="cuda",
                    device=prepared.device,
                    hardware=outcome("pass"),
                    toolchain={
                        **artifact.metadata["identity"]["toolchain"],
                        "python": platform.python_version(),
                        "numpy": np.__version__,
                    },
                    settings={
                        "mode": "numerical",
                        "fast_compile": False,
                        "dirty": dirty,
                        "seed": args.seed,
                        "nocc": args.nocc,
                        "nvir": args.nvir,
                        "scope": "generated JVP/VJP for a fixed scalar CC-like fragment",
                        "reference": "CPU TensorIR interpreter plus directional finite differences",
                        "graph_status": prepared.graph_status,
                        "plan": plan.to_payload(),
                        "recomputed_nodes": _recomputation_count(plan),
                        "finite_differences": finite_differences,
                        "dot_relative_error": dot_error,
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    },
                )
                record["hashes"].update(
                    equation=generated.program.logical_hash,
                    ir=canonical_hash(generated.program.to_payload()),
                    source=tensor_source_identity(),
                    schedule=plan.identity,
                )
                record["compilation"] = {
                    "seconds": artifact.metadata["compile_seconds"],
                    "reason": None,
                    "resources": artifact.metadata["resources"],
                    "binary_sha256": artifact.metadata["binary_sha256"],
                }
                for stage in ("representation", "source", "compilation"):
                    record["stages"][stage] = outcome("pass")
                for name, value in reference.items():
                    record["block_errors"][name] = block_error(
                        result.outputs[name], value, **GATES["integral_fp64"]
                    )
                passed = all(
                    error["passed"] for error in record["block_errors"].values()
                )
                record["stages"]["numerical"] = outcome(
                    "pass" if passed else "fail",
                    None if passed else "GPU numerical parity failed",
                )
                record["stages"]["endpoint"] = outcome(
                    "pass" if passed else "fail",
                    None if passed else "GPU endpoint parity failed",
                )
                record["stages"]["production"] = outcome(
                    "not-run", "no production promotion in this evidence run"
                )
                record["performance"] = outcome(
                    "not-run", "no interleaved baseline comparison"
                )
                record["memory"] = {
                    "allocated_bytes": plan.allocation_bytes
                    + profile.metrics["provider_retained_bytes"],
                    "peak_bytes": plan.peak_bytes,
                    "reason": None,
                    "observed_device_delta": profile.metrics["observed_device_delta"],
                    "provider_retained_bytes": profile.metrics[
                        "provider_retained_bytes"
                    ],
                }
                record["settings"]["metrics"] = result.metrics
                record["settings"]["profile_metrics"] = profile.metrics
                validate_evidence(record)
                records.append((label, plan, artifact, record))
    for label, plan, artifact, record in records:
        (args.output / f"{label}.plan.json").write_text(
            json.dumps(plan.to_payload(), indent=2, sort_keys=True) + "\n"
        )
        if record is not None:
            (args.output / f"{label}.evidence.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n"
            )
    (args.output / "evidence.json").write_text(
        json.dumps(
            [record for _, _, _, record in records if record is not None],
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("compile", "numerical"), default="numerical")
    parser.add_argument("--nvcc", type=Path)
    parser.add_argument("--architecture", default="sm_120")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--nocc", type=int, default=3)
    parser.add_argument("--nvir", type=int, default=4)
    parser.add_argument("--seed", type=int, default=151)
    parser.add_argument("--cotangent", type=float, default=1.0)
    parser.add_argument("--compile-timeout", type=float, default=300)
    parser.add_argument("--max-bytes", type=int, default=256 * 1024**2)
    parser.add_argument(
        "--cache", type=Path, default=Path("build/tensor-ad-cuda-cache")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.nocc, args.nvir) < 1:
        parser.error("nocc and nvir must be positive")
    records = run(args)
    if args.mode == "numerical":
        return int(
            any(
                record is not None and record["stages"]["numerical"]["status"] != "pass"
                for _, _, _, record in records
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
