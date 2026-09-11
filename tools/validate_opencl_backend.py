"""Exercise the optional OpenCL compiler/runtime and an existing integral DAG.

Run inside Slurm with the explicitly allocated RTX 5090. This is an expression
and runtime smoke gate, not an OpenCL HF provider or a domestic-GPU claim.
"""

# Source-tree CLI bootstrap; importing the compiler needs no native runtime.
import sys as _compiler_sys
from pathlib import Path as _CompilerPath

_compiler_sys.path.insert(
    0, str(_CompilerPath(__file__).resolve().parents[1] / "python")
)

import argparse
import ctypes as c
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from vibeqc_compiler.integral.artifact_cache import LocalArtifactCache
from vibeqc_compiler.integral.cache import integral_cache_key
from vibeqc_compiler.integral.df_values import (
    _reference_boys,
    build_df_component_kernel,
    build_df_value_ir,
)
from vibeqc_compiler.integral.expr import Graph
from vibeqc_compiler.integral.opencl_lowering import (
    ScalarKernel,
    emit_opencl,
    source_hash,
)
from vibeqc_compiler.integral.opencl_runtime import OpenCLRuntime
from vibeqc_compiler.integral.runtime_backend import (
    CompiledArtifactIdentity,
    ExecutionShape,
)

from tools.vibeqc_validation.df_values import make_df_value_fixture
from tools.vibeqc_validation.schema import block_error, canonical_hash


def integral_fixture():
    """Normalized (p_x|p_x) metric primitives against independent libcint blocks."""
    integral = build_df_value_ir("coulomb_metric", (1, 1))
    kernel = build_df_component_kernel(integral, ("x", "x"))
    graph = kernel.graph
    p, q, weight = (
        graph.variable(name) for name in ("exponent_p", "exponent_q", "weight")
    )
    # The established component IR excludes the radial prefactor. Include it
    # in the shared expression before choosing any source language or schedule.
    value = (
        kernel.value * weight * (2 * math.pi**2.5) / (p * q * graph.power(p + q, 0.5))
    )
    names = tuple(
        sorted(
            {
                str(graph.nodes[i].payload)
                for i in graph.topological_order((value,))
                if graph.nodes[i].operation == "variable"
            }
        )
    )
    records, references, groups, provenance = [], [], [], []
    for variant, lengths in (
        ("asymmetric", (2, 1)),
        ("coincident", (2, 1)),
        ("asymmetric", (9, 7)),
    ):
        fixture = make_df_value_fixture(
            (1, 1), variant=variant, primitive_lengths=lengths
        )
        # First block component is xx; the remaining Cartesian channels are
        # validated by the existing DF suite, not claimed by this smoke kernel.
        primitive_records = fixture.records[: fixture.primitive_count]
        groups.append(len(primitive_records))
        references.append(fixture.reference[0, 0])
        provenance.append(fixture.inputs)
        for primitive in primitive_records:
            a, b = primitive["exponents"][[0, 2]]
            difference = primitive["centers"][0] - primitive["centers"][2]
            rho = a * b / (a + b)
            variables = {
                "exponent_p": a,
                "exponent_q": b,
                "weight": primitive["weight"],
                "inverse_two_p": 0.5 / a,
                "inverse_two_q": 0.5 / b,
                "rho": rho,
                **{f"difference_{axis}": x for axis, x in zip("xyz", difference)},
                **{
                    f"boys_{n}": x
                    for n, x in enumerate(
                        _reference_boys(rho * np.dot(difference, difference), 3)
                    )
                },
            }
            records.append([variables[name] for name in names])
    identity = canonical_hash(
        {
            "integral": integral_cache_key(integral),
            "components": kernel.components,
            "normalization": "normalized contracted Cartesian primitives",
            "radial_prefactor": "2*pi^(5/2)/(p*q*sqrt(p+q))",
        }
    )
    return (
        ScalarKernel(graph, (value,), names, identity),
        np.asarray(records),
        references,
        groups,
        provenance,
    )


def execute_program(runtime, kernel, inputs, directory, *, workgroup=32):
    """Compile and link separately, then retain real event timings and resources."""
    shape = ExecutionShape(workgroup)
    source = emit_opencl(kernel, runtime.capabilities(), shape)
    (directory / f"{kernel.name}-{workgroup}.cl").write_text(source)
    started = time.perf_counter()
    compiled = runtime.compile(source)
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    executable = runtime.link((compiled,))
    link_seconds = time.perf_counter() - started
    compile_log, link_log = runtime.build_log(compiled), runtime.build_log(executable)
    (directory / f"{kernel.name}-{workgroup}-compile.log").write_text(compile_log)
    (directory / f"{kernel.name}-{workgroup}-link.log").write_text(link_log)
    inputs = np.ascontiguousarray(inputs, dtype=np.float64)
    output_shape = (len(inputs), len(kernel.roots))
    source_buffer = runtime.allocate(inputs.nbytes)
    output_buffer = runtime.allocate(math.prod(output_shape) * 8)
    runtime.write(source_buffer, inputs.tobytes())
    nanoseconds = []
    for _ in range(7):
        event = runtime.launch(
            executable,
            kernel.name,
            (source_buffer, output_buffer, c.c_uint64(len(inputs))),
            items=len(inputs),
            shape=shape,
        )
        nanoseconds.append(runtime.elapsed_nanoseconds(event))
        runtime.release(event)
    values = (
        np.frombuffer(
            runtime.read(output_buffer, math.prod(output_shape) * 8), dtype=np.float64
        )
        .copy()
        .reshape(output_shape)
    )
    resource_usage = runtime.resources(executable, kernel.name)
    device = runtime.device
    identity = CompiledArtifactIdentity(
        "opencl",
        device["uuid"] or device["vendor"] + "/" + device["name"],
        device["version"],
        "vendor ICD online compiler " + device["driver"],
        device["driver"],
        device["language"],
        ("-cl-std=CL1.2",),
        kernel.scientific_hash,
        source_hash(source),
        canonical_hash(asdict(shape)),
    )
    binary = runtime.binary(executable)
    cache = LocalArtifactCache(directory / "local-cache")
    cache.install(identity, binary)
    reloaded = runtime.load_binary(cache.load(identity), identity)
    event = runtime.launch(
        reloaded,
        kernel.name,
        (source_buffer, output_buffer, c.c_uint64(len(inputs))),
        items=len(inputs),
        shape=shape,
    )
    runtime.wait(event)
    cached_values = np.frombuffer(
        runtime.read(output_buffer, math.prod(output_shape) * 8), dtype=np.float64
    ).reshape(output_shape)
    np.testing.assert_array_equal(cached_values, values)
    for resource in (
        event,
        reloaded,
        source_buffer,
        output_buffer,
        executable,
        compiled,
    ):
        runtime.release(resource)
    return values, {
        "identity": asdict(identity),
        "binary_bytes": len(binary),
        "cache_roundtrip_passed": True,
        "artifact_key": identity.key,
        "compile_seconds": compile_seconds,
        "link_seconds": link_seconds,
        "kernel_nanoseconds": nanoseconds,
        "resources": resource_usage,
        "source_hash": source_hash(source),
        "inputs_hash": canonical_hash(inputs.tolist()),
        "input_bytes": inputs.nbytes,
        "output_bytes": math.prod(output_shape) * 8,
    }


def main():
    """Record current vendor/runtime provenance without advertising a full backend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--library", help="explicit installed OpenCL ICD loader")
    parser.add_argument("--device-uuid", help="exact queried OpenCL GPU UUID")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error(
            "execute this real-GPU smoke gate inside a finite Slurm allocation"
        )
    args.directory.mkdir(parents=True, exist_ok=True)
    with OpenCLRuntime(library=args.library, device_uuid=args.device_uuid) as runtime:
        # This machine has one GPU. An ICD may ignore Slurm's CUDA mask; a
        # multi-GPU gate needs a scheduler-to-OpenCL UUID mapping before use.
        if len(runtime._devices()) != 1:
            parser.error("this Slurm smoke gate requires a single-GPU OpenCL inventory")
        # First prove the native compiler/queue/memory path with a small DAG.
        graph = Graph()
        x, y = graph.variable("x"), graph.variable("y")
        toy = ScalarKernel(
            graph,
            (2 * x + y,),
            ("x", "y"),
            canonical_hash({"expression": "2*x+y"}),
            "small_expression",
        )
        toy_inputs = np.arange(34, dtype=float).reshape(17, 2) / 7
        toy_values, toy_report = execute_program(
            runtime, toy, toy_inputs, args.directory
        )
        toy_report["numerical"] = block_error(
            toy_values[:, 0],
            2 * toy_inputs[:, 0] + toy_inputs[:, 1],
            atol=1e-14,
            rtol=1e-14,
        )
        kernel, inputs, reference, groups, provenance = integral_fixture()
        runs = []
        for workgroup in (16, 32, 64):
            raw, report = execute_program(
                runtime, kernel, inputs, args.directory, workgroup=workgroup
            )
            offsets = np.cumsum([0, *groups])
            actual = np.array(
                [raw[offsets[i] : offsets[i + 1], 0].sum() for i in range(len(groups))]
            )
            report["numerical"] = block_error(actual, reference, atol=1e-11, rtol=3e-12)
            report["contracted_values"] = actual.tolist()
            runs.append(report)
        report = {
            "schema": "vibeqc.opencl_expression_smoke",
            "version": 1,
            "device": {k: v for k, v in runtime.device.items() if k != "handle"},
            "capabilities": asdict(runtime.capabilities()),
            "target": asdict(runtime.target_info()),
            "provenance": {
                "executed_at_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pyscf": importlib.metadata.version("pyscf"),
                "git_head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=ROOT,
                    text=True,
                ).strip(),
                "worktree_dirty": bool(
                    subprocess.check_output(
                        ["git", "status", "--porcelain"],
                        cwd=ROOT,
                        text=True,
                    ).strip()
                ),
                "source_files": {
                    str(path.relative_to(ROOT)): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in sorted(
                        {
                            Path(__file__).resolve(),
                            *ROOT.glob("python/vibeqc_compiler/integral/*.py"),
                            *ROOT.glob("tools/vibeqc_validation/*.py"),
                        }
                    )
                },
            },
            "toy": toy_report,
            "integral_runs": runs,
            "inputs": provenance,
            "references": reference,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "production_hf_supported": False,
            "domestic_gpu_validation": "not-run",
            "host_boundary": "Boys fixture inputs and contraction of primitive outputs; no production integral provider",
            "passed": toy_report["numerical"]["passed"]
            and all(r["numerical"]["passed"] for r in runs),
        }
    (args.directory / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {"passed": report["passed"], "report": str(args.directory / "report.json")}
        )
    )
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
