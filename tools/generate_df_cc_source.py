"""Generate DF-CC factor packing and retained-block BLAS traversal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from generativeqc_compiler.cc.df_source import (
    BLOCK_FACTORS,
    FACTOR_NAMES,
    block_program,
    factor_program,
)
from generativeqc_compiler.tensor.cuda_gemm import gemm_contract

from tools.generate_rccsd_native import (
    REPRESENTATIVE,
    _cpu_function,
    _cuda_program,
    _label_dims,
    _required_function,
)


def cpu_header() -> str:
    """Emit packing oracle/capacity and equation-derived matrix products."""
    program = factor_program(*REPRESENTATIVE, 1)
    lines = [
        "// Generated DF-CC source packing/blocks; do not edit.",
        "#pragma once",
        "#include <algorithm>",
        "#include <cmath>",
        "#include <initializer_list>",
        '#include "posthf/capacity.hpp"',
        "namespace generativeqc::cc::generated::df_source {",
        "using posthf::checked_add; using posthf::checked_mul;",
        "inline std::size_t checked_product(std::initializer_list<std::size_t> factors) {",
        "  std::size_t count=1; for(auto x:factors)count=checked_mul(count,x);return count;}",
        "struct FactorOutputs { const double *boo{}, *bov{}, *bvo{}, *bvv{}; };",
        f'inline constexpr const char* factor_equation_hash="{program.logical_hash}";',
        _required_function(program, "factor_arena_elements", batch_dim=True),
        _cpu_function(
            program,
            "pack_cpu",
            "FactorOutputs",
            signature="const double* bmo",
            input_overrides={"bmo": "bmo"},
            batch_dim=True,
            output_fields=FACTOR_NAMES,
        ),
    ]
    for name, (left, right) in BLOCK_FACTORS.items():
        block = block_program(*REPRESENTATIVE, 1, name)
        node = block.outputs[name]
        g = gemm_contract(node)
        if (
            g is None
            or g.batch_labels
            or g.coefficient != 1
            or g.output_labels != g.m_labels + g.n_labels
            or g.a_labels != g.k_labels + g.m_labels
            or g.b_labels != g.k_labels + g.n_labels
        ):
            raise ValueError("retained DF block is not a direct factor Gram product")
        dims = _label_dims(node)

        def extent(labels: tuple[int, ...], dimensions: dict[int, str] = dims) -> str:
            return "checked_product({" + ",".join(dimensions[x] for x in labels) + "})"

        m, n, k = (extent(labels) for labels in (g.m_labels, g.n_labels, g.k_labels))
        lines += [
            f'inline constexpr const char* {name}_equation_hash="{block.logical_hash}";',
            f"inline std::size_t {name}_elements(std::size_t o,std::size_t v) {{ return checked_mul({m},{n}); }}",
            f"inline std::size_t {name}_summands(std::size_t o,std::size_t v,std::size_t q) {{ return checked_product({{{m},{n},{k}}}); }}",
            "template<class Gemm>",
            f"void build_{name}(std::size_t o,std::size_t v,std::size_t q,const FactorOutputs& factors,double* output,Gemm&& gemm) {{",
            # Q-major factors are packed column-major [pair,Q] matrices. Reverse
            # operands so the column-major output is the row-major ijkl block.
            f"  gemm('N','T',{n},{m},{k},factors.{right},factors.{left},output);",
            "}",
        ]
    lines += [
        "// Phase peaks include caller-owned host outputs only in the publication phase.",
        "struct SourceLayout { std::size_t source_values,row_values,matrix_values,packing_values,output_values,largest_block,transform_bytes,packing_bytes,blocks_bytes; };",
        "inline SourceLayout source_layout(std::size_t o,std::size_t v,std::size_t q) {",
        '  if(!o||!v||!q)throw std::invalid_argument("empty DF-CC source shape");',
        "  const auto n=checked_add(o,v);",
        "  SourceLayout p{}; p.matrix_values=checked_mul(n,n);",
        "  p.source_values=checked_mul(p.matrix_values,q); p.row_values=checked_mul(n,q);",
        "  p.packing_values=factor_arena_elements(o,v,q);",
        "  p.output_values=checked_mul(q,checked_add(checked_mul(o,v),checked_mul(v,v)));",
    ]
    for name in BLOCK_FACTORS:
        lines += [
            f"  p.output_values=checked_add(p.output_values,{name}_elements(o,v));",
            f"  p.largest_block=std::max(p.largest_block,{name}_elements(o,v));",
        ]
    lines += [
        "  p.transform_bytes=checked_mul(sizeof(double),checked_add(checked_mul(2,p.source_values),checked_add(p.row_values,p.matrix_values)));",
        "  p.packing_bytes=checked_add(sizeof(int),checked_mul(sizeof(double),checked_add(p.source_values,p.packing_values)));",
        "  p.blocks_bytes=checked_add(sizeof(int),checked_mul(sizeof(double),checked_add(p.packing_values,checked_add(p.largest_block,p.output_values))));",
        "  return p;",
        "}",
    ]
    lines += ["}  // namespace generativeqc::cc::generated::df_source", ""]
    return "\n".join(lines)


def cuda_header() -> str:
    return """// Generated DF-CC source device interface; do not edit.
#pragma once
#include <cuda_runtime.h>
#include "generated_df_cc_source_cpu.hpp"
namespace generativeqc::cc::generated::df_source {
struct CudaState {
  std::size_t o{},v{},q{};
  const double* bmo{};
  double* response_arena{};
  int* error{};
  cudaStream_t stream{};
};
FactorOutputs pack_cuda(CudaState& state);
}
"""


def cuda_source() -> str:
    program = factor_program(*REPRESENTATIVE, 1)
    return "\n".join(
        [
            '#include "generated_df_cc_source_cuda.cuh"',
            '#include "tensor/cuda_runtime.cuh"',
            "namespace generativeqc::cc::generated::df_source {",
            _cuda_program(
                program,
                "source_pack",
                "FactorOutputs",
                input_overrides={"bmo": "s.bmo"},
                batch_dim=True,
                output_fields=FACTOR_NAMES,
            ),
            "FactorOutputs pack_cuda(CudaState& state) { return run_source_pack(state); }",
            "}",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, emit in (
        ("_cpu.hpp", cpu_header),
        ("_cuda.cuh", cuda_header),
        ("_cuda.cu", cuda_source),
    ):
        (args.output_dir / ("generated_df_cc_source" + suffix)).write_text(
            emit(), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
