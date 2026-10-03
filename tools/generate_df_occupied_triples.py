"""Emit bounded DF panels, W GEMMs and a fused occupied-tile (T) epilogue."""

from __future__ import annotations

import argparse
import sys
import typing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path[:0] = [str(ROOT), str(ROOT / "python")]

from generativeqc_compiler.cc.occupied_triples import (
    PERMUTATIONS,
    df_panel_program,
    energy_scalar_program,
    moment_program,
    v_scalar_program,
)
from generativeqc_compiler.cc.triples import _LABELS, VP
from generativeqc_compiler.tensor.cuda_gemm import gemm_contract
from generativeqc_compiler.tensor.scalar_cpp import emit_scalar_cpp

if typing.TYPE_CHECKING:
    from generativeqc_compiler.tensor import Node, Program


def _inputs(program: Program) -> tuple[str, ...]:
    return tuple(sorted(n.attrs["name"] for n in program.live_nodes if n.op == "input"))


def _call(program: Program, name: str, bindings: dict[str, str], output: str) -> str:
    names = _inputs(program)
    if set(names) != set(bindings):
        raise ValueError("scalar boundary bindings do not match the equation")
    return name + "(" + ",".join([*(bindings[key] for key in names), output]) + ");"


def _gemm(
    node: Node, bindings: dict[str, tuple[str, str]], alpha: str, beta: str
) -> str:
    """Derive a column-major call from a TensorIR product and strided views.

    Views supply the address and physical leading dimension of each logical
    row-major input. Operand permutation, transposition, contraction dimensions
    and output layout come exclusively from the shared GEMM contract.
    """
    g = gemm_contract(node)
    if g is None or g.batch_labels or g.coefficient != 1 or g.dtype != "float64":
        raise ValueError("occupied triples require unbatched unit FP64 products")
    a, b, c, m, n, k = (
        g.a_labels,
        g.b_labels,
        g.output_labels,
        g.m_labels,
        g.n_labels,
        g.k_labels,
    )
    aa, bb = (bindings[x.attrs["name"]] for x in node.inputs)
    if c != m + n:
        a, b, m, n, aa, bb = b, a, n, m, bb, aa
    if c != m + n:
        raise ValueError("occupied triples output needs an unqualified packing")
    dims = {
        label: {"occupied": "o", "virtual": "v", "auxiliary": "q"}[index.space.kind]
        for operand, labels in zip(node.inputs, node.attrs["labels"], strict=True)
        for index, label in zip(operand.spec.indices, labels, strict=True)
    }

    def extent(labels: tuple[int, ...]) -> str:
        return "checked_product({" + ",".join(dims[x] for x in labels) + "})"

    def trans(
        actual: tuple[int, ...], rows: tuple[int, ...], columns: tuple[int, ...]
    ) -> str:
        if actual == rows + columns:
            return "N"
        if actual == columns + rows:
            return "T"
        raise ValueError("occupied triples input needs an unqualified packing")

    # Transpose the entire row-major product, reversing the two operands.
    return (
        f"gemm('{trans(b, k, n)}','{trans(a, m, k)}',"
        f"{extent(n)},{extent(m)},{extent(k)},{alpha},"
        f"{bb[0]},{bb[1]},{aa[0]},{aa[1]},{beta},output,{extent(n)});"
    )


def header() -> str:
    panel = df_panel_program(3, 4)
    moments = moment_program(2, 3)
    scalar, v_scalar = energy_scalar_program(), v_scalar_program()
    w = moments.outputs["w"]
    if w.op != "add" or len(w.inputs) != 2:
        raise ValueError("occupied W seed must contain two audited products")
    weights = w.attrs["coefficients"]
    coefficient = lambda pair: f"({pair[0]}.0/{pair[1]}.0)"
    lines = [
        "// Generated from occupied_triples TensorIR; do not edit.",
        "#pragma once",
        "#include <cstddef>",
        "#include <cmath>",
        "#include <initializer_list>",
        '#include "posthf/capacity.hpp"',
        "#ifdef __CUDACC__",
        "#define GQC_DF_TRIPLES_HD __host__ __device__",
        "#else",
        "#define GQC_DF_TRIPLES_HD",
        "#endif",
        "namespace generativeqc::cc::triples::generated_df {",
        "using posthf::checked_add; using posthf::checked_mul;",
        "inline std::size_t checked_product(std::initializer_list<std::size_t> xs) {",
        "  std::size_t n=1; for (auto x:xs) n=checked_mul(n,x); return n; }",
        f'inline constexpr const char* panel_hash="{panel.logical_hash}";',
        f'inline constexpr const char* moment_hash="{moments.logical_hash}";',
        f'inline constexpr const char* epilogue_hash="{scalar.logical_hash}";',
        "inline constexpr unsigned permutations[6][3]={"
        + ",".join("{" + ",".join(map(str, p)) + "}" for p in PERMUTATIONS)
        + "};",
        "struct Inputs { const double *bov{},*bvv{},*ovoo{},*ovov{},*fov{},*t1{},*t2{},*eps_o{},*eps_v{}; };",
        "template<class Gemm> void build_panel(std::size_t o,std::size_t v,std::size_t q,",
        "  std::size_t i,const Inputs& in,double* output,Gemm&& gemm) {",
        _gemm(
            panel.outputs["panel"],
            {"bov_i": ("in.bov+i*v", "o*v"), "bvv": ("in.bvv", "v*v")},
            "1.0",
            "0.0",
        ),
        "}",
        "template<class Gemm> void build_w(std::size_t o,std::size_t v,",
        "  std::size_t i,std::size_t j,std::size_t k,const Inputs& in,",
        "  const double* panel,double* output,Gemm&& gemm) {",
        _gemm(
            w.inputs[0],
            {"panel": ("panel", "v"), "t2_kj": ("in.t2+(k*o+j)*v*v", "v")},
            coefficient(weights[0]),
            "0.0",
        ),
        _gemm(
            w.inputs[1],
            {
                "ovoo_ij": ("in.ovoo+(i*v*o+j)*o", "o*o"),
                "t2_mk": ("in.t2+k*v*v", "o*v*v"),
            },
            coefficient(weights[1]),
            "1.0",
        ),
        "}",
    ]
    for program, name in ((v_scalar, "v_element"), (scalar, "energy_element")):
        lines.append(
            emit_scalar_cpp(
                program,
                function_name=name,
                caller_owned_checks=True,
                ordered_native_sums=True,
            ).replace("inline bool ", "GQC_DF_TRIPLES_HD inline bool ")
        )
    lines += [
        "}  // namespace generativeqc::cc::triples::generated_df",
        "#undef GQC_DF_TRIPLES_HD",
        "",
    ]
    return "\n".join(lines)


def cuda_header() -> str:
    return """// Generated occupied-tile epilogue interface; do not edit.
#pragma once
#include <cuda_runtime.h>
#include "generated_df_occupied_triples.hpp"
namespace generativeqc::cc::triples::generated_df {
void energy_tile(std::size_t o,std::size_t v,std::size_t i,std::size_t j,std::size_t k,
                 double degeneracy,double threshold,const Inputs& in,const double* moments,
                 unsigned blocks,double* partials,int* error,cudaStream_t stream);
}
"""


def cuda_source() -> str:
    scalar, v_scalar = energy_scalar_program(), v_scalar_program()
    bindings = {"denominator": "denominator"}
    lines = [
        '#include "generated_df_occupied_triples_cuda.cuh"',
        '#include "tensor/cuda_runtime.cuh"',
        "namespace generativeqc::cc::triples::generated_df {",
        "__global__ void tile_kernel(std::size_t o,std::size_t v,std::size_t i,",
        "  std::size_t j,std::size_t k,double degeneracy,double threshold,Inputs in,",
        "  const double* moments,double* partials,int* error) {",
        "  const auto v3=v*v*v; const std::size_t occupied[3]={i,j,k};",
        "  double accumulated=0.0;",
        "  for (std::size_t flat=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;",
        "       flat<v3;flat+=std::size_t(blockDim.x)*gridDim.x) {",
        "    const std::size_t a=flat/(v*v),b=(flat/v)%v,c=flat%v;",
        "    const double gap=in.eps_o[i]+in.eps_o[j]+in.eps_o[k]-in.eps_v[a]-in.eps_v[b]-in.eps_v[c];",
        "    const double denominator=gap*degeneracy;",
        "    if (!isfinite(gap) || gap>=0.0 || fabs(gap)<=threshold || !isfinite(denominator)) {",
        "      atomicCAS(error,0,1); continue; }",
    ]
    for index, occ in enumerate(_LABELS):
        p = PERMUTATIONS[index]
        I, J, K = (f"occupied[{axis}]" for axis in p)
        v_name = f"v_{occ}"
        lines += [
            f"    double {v_name}=0.0;",
            "    "
            + _call(
                v_scalar,
                "v_element",
                {
                    "ovov": f"in.ovov[((({I})*v+a)*o+({J}))*v+b]",
                    "t1": f"in.t1[({K})*v+c]",
                    "t2": f"in.t2[((({I})*o+({J}))*v+a)*v+b]",
                    "fov": f"in.fov[({K})*v+c]",
                },
                v_name,
            ),
        ]
        bindings[v_name] = v_name
        for vir in _LABELS:
            x, y, z = ("abc"[axis] for axis in VP[vir])
            bindings[f"w_{occ}_{vir}"] = f"moments[{index}*v3+({x}*v+{y})*v+{z}]"
    lines += [
        "    double value=0.0;",
        "    " + _call(scalar, "energy_element", bindings, "value"),
        "    accumulated+=generativeqc_tensor::finite(value,error,1);",
        "  }",
        "  __shared__ double sums[256]; sums[threadIdx.x]=accumulated; __syncthreads();",
        "  for (unsigned stride=blockDim.x/2;stride;stride/=2) {",
        "    if (threadIdx.x<stride) sums[threadIdx.x]+=sums[threadIdx.x+stride];",
        "    __syncthreads(); }",
        "  if (threadIdx.x==0) partials[blockIdx.x]=generativeqc_tensor::finite(sums[0],error,2);",
        "}",
        "void energy_tile(std::size_t o,std::size_t v,std::size_t i,std::size_t j,std::size_t k,",
        "  double degeneracy,double threshold,const Inputs& in,const double* moments,",
        "  unsigned blocks,double* partials,int* error,cudaStream_t stream) {",
        "  tile_kernel<<<blocks,256,0,stream>>>(o,v,i,j,k,degeneracy,threshold,in,moments,partials,error);",
        "  generativeqc_tensor::cuda_check(cudaGetLastError());",
        "}",
        "}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, emitter in (
        (".hpp", header),
        ("_cuda.cuh", cuda_header),
        ("_cuda.cu", cuda_source),
    ):
        (args.output_dir / ("generated_df_occupied_triples" + suffix)).write_text(
            emitter()
        )


if __name__ == "__main__":
    main()
