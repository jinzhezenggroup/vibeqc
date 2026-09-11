"""Compile two shared weighted-DAG materialization candidates without a GPU.

This resource screen is weaker than the Slurm numerical and molecular endpoint
gates. Each kernel reads identical runtime geometry/weights and writes all
thirteen results, so dead outputs cannot hide register or spill costs.
"""

# Source-tree CLI bootstrap; importing the compiler needs no native runtime.
import sys as _compiler_sys
from pathlib import Path as _CompilerPath

_compiler_sys.path.insert(
    0, str(_CompilerPath(__file__).resolve().parents[1] / "python")
)

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vibeqc_compiler.integral.shell_class import build_weighted_shell_contraction_kernel
from vibeqc_compiler.integral.shell_spec import PSSS_SPEC
from vibeqc_compiler.integral.weighted_eri import (
    build_weighted_eri_ir,
    build_weighted_eri_kernel,
)
from vibeqc_compiler.integral.weighted_eri_cuda import emit_psss_weighted_header

from tools.vibeqc_validation.schema import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "nvcc": subprocess.check_output([str(args.nvcc), "--version"], text=True),
        "candidates": [],
    }
    integral = build_weighted_eri_ir((1, 0, 0, 0))
    report["new_graph_nodes"] = len(build_weighted_eri_kernel(integral).graph.nodes)
    report["previous_component_cloning_nodes"] = len(
        build_weighted_shell_contraction_kernel(PSSS_SPEC).graph.nodes
    )
    for inline in (False, True):
        stem = args.output / ("inline" if inline else "materialized")
        start = time.perf_counter()
        header = stem.with_suffix(".cuh")
        header.write_text(emit_psss_weighted_header(inline_single_use=inline))
        generation_seconds = time.perf_counter() - start
        source = stem.with_suffix(".cu")
        source.write_text(
            f'#include "{header.name}"\n'
            + """
using namespace vibeqc::scf::generated_weighted_eri;
extern "C" __global__ void weighted_screen(const Geometry* geometry,
                                           const double* weights, Gradient* results) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  results[i] = psss(geometry[i], weights + 3*i);
}
"""
        )
        obj = stem.with_suffix(".o")
        command = [
            str(args.nvcc),
            "-O3",
            "-std=c++20",
            "-arch=sm_120",
            "-Xptxas=-v",
            "-c",
            str(source),
            "-o",
            str(obj),
        ]
        start = time.perf_counter()
        result = subprocess.run(command, text=True, capture_output=True, check=True)
        compile_seconds = time.perf_counter() - start
        log = stem.with_suffix(".log")
        log.write_text(result.stdout + result.stderr)
        report["candidates"].append(
            {
                "inline_single_use": inline,
                "source_sha256": file_hash(source),
                "header_sha256": file_hash(header),
                "object_sha256": file_hash(obj),
                "generation_seconds": generation_seconds,
                "compile_seconds": compile_seconds,
                "source_bytes": source.stat().st_size + header.stat().st_size,
                "object_bytes": obj.stat().st_size,
                "command": command,
                "ptxas": result.stdout + result.stderr,
            }
        )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
