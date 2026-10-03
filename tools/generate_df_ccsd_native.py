"""Emit bounded native DF virtual residual/response actions from TensorIR.

This generator shares the conventional runtime-shape emitter. The compiler
derives the equations and AD; native consumers own auxiliary accumulation,
allocations, execution state, and admission of the complete endpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "python"))

from generativeqc_compiler.cc.df_equations import build_df_virtual_response_programs
from generativeqc_compiler.tensor import Program, prepare_for_backend

from tools.generate_rccsd_native import (
    REPRESENTATIVE,
    _cpu_function,
    _cuda_program,
    _required_function,
)

INPUTS = (
    "bov",
    "bvv",
    "t1",
    "t2",
    "d_t1",
    "d_t2",
    "bar_df_virtual_singles",
    "bar_df_virtual_doubles",
)
OUTPUTS = {
    "virtual": ("VirtualOutputs", ("df_virtual_singles", "df_virtual_doubles")),
    "amplitude_jvp": (
        "VirtualOutputs",
        ("d_df_virtual_singles", "d_df_virtual_doubles"),
    ),
    "amplitude_vjp": ("AmplitudeOutputs", ("bar_t1", "bar_t2")),
    "factor_vjp": ("FactorOutputs", ("bar_bov", "bar_bvv")),
}


def programs(backend: str) -> dict[str, Program]:
    """Prepare the fixed symbolic schedule without changing auxiliary scope."""
    source = build_df_virtual_response_programs(*REPRESENTATIVE)
    result = {}
    for name, program in (
        ("virtual", source.primal),
        ("amplitude_jvp", source.amplitude_jvp.program),
        ("amplitude_vjp", source.amplitude_vjp.program),
        ("factor_vjp", source.factor_vjp.program),
    ):
        prepared = prepare_for_backend(program, backend, preserve_reduction_order=True)
        if any(
            sum(index.space.kind == "virtual" for index in node.spec.indices) > 2
            for node in prepared.live_nodes
        ):
            raise ValueError(
                "native DF action reconstructed an unbounded virtual block"
            )
        result[name] = Program(
            prepared.outputs,
            provenance={
                **prepared.provenance,
                "native_execution_order": "dependencies",
            },
        )
    return result


def cpu_header() -> str:
    """CPU actions and shared exact runtime-shape arena queries."""
    lines = [
        "// Generated DF RCCSD one-auxiliary-slice equations; do not edit.",
        "#pragma once",
        "#include <algorithm>",
        "#include <cmath>",
        "#include <initializer_list>",
        '#include "posthf/capacity.hpp"',
        "namespace generativeqc::cc::generated::df {",
        "using posthf::checked_add;",
        "using posthf::checked_mul;",
        "inline std::size_t checked_product(std::initializer_list<std::size_t> factors) {",
        "  std::size_t count=1; for (auto factor : factors) count=checked_mul(count,factor); return count;",
        "}",
        "struct Inputs {",
        *[f"  const double* {name}{{}};" for name in INPUTS],
        "};",
        "struct VirtualOutputs { const double *singles, *doubles; };",
        "struct AmplitudeOutputs { const double *t1, *t2; };",
        "struct FactorOutputs { const double *bov, *bvv; };",
    ]
    for backend in ("cpu", "cuda"):
        for name, program in programs(backend).items():
            lines.append(
                f'inline constexpr const char* {name}_{backend}_equation_hash = "{program.logical_hash}";'
            )
            lines.append(
                f"inline constexpr std::size_t {name}_{backend}_operation_count = {sum(node.op != 'input' for node in program.live_nodes)};"
            )
            lines.append(
                _required_function(program, f"{name}_{backend}_arena_elements")
            )
            if backend == "cpu":
                output_type, fields = OUTPUTS[name]
                lines.append(
                    _cpu_function(
                        program,
                        f"run_{name}_cpu",
                        output_type,
                        input_overrides={key: f"inputs.{key}" for key in INPUTS},
                        output_fields=fields,
                    )
                )
    lines += ["}  // namespace generativeqc::cc::generated::df", ""]
    return "\n".join(lines)


def cuda_header() -> str:
    """Borrowed device state; allocation/lifetime stay with its native owner."""
    return "\n".join(
        [
            "// Generated DF RCCSD device declarations; do not edit.",
            "#pragma once",
            "#include <cuda_runtime.h>",
            '#include "generated_df_ccsd_cpu.hpp"',
            "namespace generativeqc::cc::generated::df {",
            "struct CudaState : Inputs {",
            "  std::size_t o{}, v{};",
            "  double* response_arena{};",
            "  int* error{};",
            "  cudaStream_t stream{};",
            "};",
            *[
                f"{output_type} run_{name}_cuda(CudaState& state);"
                for name, (output_type, _) in OUTPUTS.items()
            ],
            *[
                f"{output_type} run_{name}_accumulate_cuda(CudaState& state);"
                for name, (output_type, _) in OUTPUTS.items()
                if name != "virtual"
            ],
            "// Composition entry: caller clears error once before all Q slices.",
            "VirtualOutputs run_virtual_accumulate_cuda(CudaState& state);",
            "}  // namespace generativeqc::cc::generated::df",
            "",
        ]
    )


def cuda_source() -> str:
    """Emit CUDA primal and response arithmetic using the common native owner."""
    lines = [
        '#include "generated_df_ccsd_cuda.cuh"',
        '#include "tensor/cuda_runtime.cuh"',
        "namespace generativeqc::cc::generated::df {",
    ]
    for name, program in programs("cuda").items():
        output_type, fields = OUTPUTS[name]
        lines.append(
            _cuda_program(
                program,
                f"df_{name}",
                output_type,
                input_overrides={key: f"s.{key}" for key in INPUTS},
                output_fields=fields,
                reset_error=False,
            )
        )
        if name == "virtual":
            lines.append(
                "VirtualOutputs run_virtual_accumulate_cuda(CudaState& state) { return run_df_virtual(state); }"
            )
            lines.append(
                "VirtualOutputs run_virtual_cuda(CudaState& state) { generativeqc_tensor::cuda_check(cudaMemsetAsync(state.error,0,sizeof(int),state.stream)); return run_df_virtual(state); }"
            )
            continue
        lines.append(
            f"{output_type} run_{name}_accumulate_cuda(CudaState& state) {{ return run_df_{name}(state); }}"
        )
        lines.append(
            f"{output_type} run_{name}_cuda(CudaState& state) {{ generativeqc_tensor::cuda_check(cudaMemsetAsync(state.error,0,sizeof(int),state.stream)); return run_df_{name}(state); }}"
        )
    lines += ["}  // namespace generativeqc::cc::generated::df", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, producer in (
        ("generated_df_ccsd_cpu.hpp", cpu_header),
        ("generated_df_ccsd_cuda.cuh", cuda_header),
        ("generated_df_ccsd_cuda.cu", cuda_source),
    ):
        (args.output_dir / name).write_text(producer())


if __name__ == "__main__":
    main()
