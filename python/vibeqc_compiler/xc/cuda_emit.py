"""Consumer-directed XC CUDA lowering through shared scalar CSE and runtime."""

import typing
from dataclasses import dataclass

from vibeqc_compiler.common.paths import LAYOUT_VERSION, asset_path, source_hashes
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.integral.scalar_c import ScalarCEmitter
from vibeqc_compiler.integral.expr import RematerializationPolicy


@dataclass(frozen=True)
class XCSchedule:
    """Explicit local candidate: baseline scalar outputs, fused or bounded groups."""

    variant: str = "baseline"
    threads: int = 128
    group_size: int = 8

    def __post_init__(self) -> None:
        if self.variant not in ("baseline", "fused", "split"):
            raise ValueError("unknown XC CUDA variant")
        if type(self.threads) is not int or self.threads not in (64, 128, 256):
            raise ValueError("XC threads must be 64, 128 or 256")
        if type(self.group_size) is not int or not 1 <= self.group_size <= 36:
            raise ValueError("XC output group must be in [1,36]")


def emit_cuda(program: typing.Any, schedule: typing.Any = None) -> typing.Any:
    """Emit deterministic FP64 source and tuning metadata; no device execution."""
    schedule = schedule or XCSchedule()
    nout = len(program.outputs)
    width = (
        1
        if schedule.variant == "baseline"
        else nout
        if schedule.variant == "fused"
        else schedule.group_size
    )
    groups = [tuple(range(i, min(i + width, nout))) for i in range(0, nout, width)]
    piecewise = any(
        program.graph.nodes[identifier].operation == "select_le"
        for identifier in program.graph.topological_order(program.roots)
    )
    contract = {
        "schema": LAYOUT_VERSION,
        "generator_sources": source_hashes(
            "common",
            "integral",
            "xc",
            "dft",
            assets=(
                "src/tensor/cuda_runtime.cuh",
                "src/runtime/bounded_workspace.hpp",
                "src/runtime/cuda_resources.cuh",
                "src/runtime/resource_cuda.cuh",
                "src/runtime/resource_ledger.hpp",
                "src/tensor/metrics.hpp",
                "src/runtime/allocation_measurement.hpp",
                "src/dft/xc_runtime.cuh",
            ),
        ),
        "expression_hash": program.expression_hash,
        "functional": program.spec.to_payload(),
        "features": program.spec.features,
        "outputs": program.outputs,
        "layout": "FP64 feature/output-major, contiguous point lanes",
        "variant": schedule.variant,
        "groups": groups,
        "threads": schedule.threads,
        "placement": (
            "branch_local_materialized" if piecewise else "inline_single_use"
        ),
        "fp64": "--fmad=false; no fast math",
    }
    identity = canonical_hash(contract)
    lines = [
        "// Generated from audited MPL-2.0 expressions; see upstream/libxc/7.0.0/COPYING.",
        '#include "cuda_runtime.cuh"',
        "#include <cmath>",
        "using namespace vibeqc_tensor;",
        f"constexpr size_t XC_INPUTS = {len(program.spec.features)};",
        f"constexpr size_t XC_OUTPUTS = {nout};",
        f'constexpr const char* XC_IDENTITY = "{identity}";',
    ]
    models = []
    for index, group in enumerate(groups):
        roots = tuple(program.roots[i] for i in group)
        group_piecewise = any(
            program.graph.nodes[identifier].operation == "select_le"
            for identifier in program.graph.topological_order(roots)
        )
        if group_piecewise:
            placement = None
            models.append(
                {
                    "policy": "branch_local_materialized",
                    "ssa_upper_bound": program.graph.analyze_ssa(roots).to_payload(),
                }
            )
        else:
            placement = program.graph.materialization_plan(
                roots, RematerializationPolicy.inline_single_use_values()
            )
            models.append(placement.to_payload())
        emitter = ScalarCEmitter(
            program.graph,
            {
                name: f"input[{i} * npoint + point]"
                for i, name in enumerate(program.spec.features)
            },
            placement,
        )
        emitter.emit(roots)
        lines.extend(
            (
                f"__global__ void xc_group_{index}(const double* input, double* output, size_t npoint, int* error) {{",
                "  for (size_t point = size_t(blockIdx.x) * blockDim.x + threadIdx.x; point < npoint; point += size_t(blockDim.x) * gridDim.x) {",
            )
        )
        if program.order == 0:
            density = (
                "input[point] + input[npoint + point]"
                if program.spec.spin == "polarized"
                else "input[point]"
            )
            lines.append(f"    if ({density} == 0.0) {{")
            lines.extend(f"      output[{i} * npoint + point] = 0.0;" for i in group)
            lines.extend(("      continue;", "    }"))
        lines.extend("  " + line for line in emitter.lines)
        lines.extend(
            f"    output[{i} * npoint + point] = vibeqc_tensor::finite({emitter.reference(program.roots[i])}, error, {i});"
            for i in group
        )
        lines.extend(("  }", "}"))
    lines.append(
        "void xc_launch(const double* input, double* output, size_t npoint, int* error, cudaStream_t stream) {"
    )
    for index in range(len(groups)):
        lines.append(
            f"  xc_group_{index}<<<blocks(npoint, {schedule.threads}), {schedule.threads}, 0, stream>>>(input, output, npoint, error);"
        )
        lines.append("  cuda_check(cudaGetLastError());")
    lines.extend(("}", '#include "xc_runtime.cuh"', ""))
    source = "\n".join(lines)
    return (
        source,
        {**contract, "identity": identity, "static_models": models},
        (
            asset_path("src/tensor/cuda_runtime.cuh"),
            asset_path("src/runtime/bounded_workspace.hpp"),
            asset_path("src/runtime/cuda_resources.cuh"),
            asset_path("src/runtime/resource_cuda.cuh"),
            asset_path("src/runtime/resource_ledger.hpp"),
            asset_path("src/tensor/metrics.hpp"),
            asset_path("src/runtime/allocation_measurement.hpp"),
            asset_path("src/dft/xc_runtime.cuh"),
        ),
    )
