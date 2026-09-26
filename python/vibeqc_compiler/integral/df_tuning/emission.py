"""Instantiate the exact native production packet kernel for one typed trial."""

import typing

from .policy import DfDerivativeTrial

API = r"""#pragma once
#include "scf/cuda/df_shell_derivatives.cuh"
namespace vibeqc::df_benchmark {
using scf::DfShellBasisView;
using scf::DfDerivativePairs;
using Launch = cudaError_t(*)(DfShellBasisView,DfShellBasisView,DfShellBasisView,
    const double*,std::size_t,std::size_t,const double*,double*,unsigned long long*,
    DfDerivativePairs,bool,cudaStream_t);
struct Candidate { const char* key; unsigned a,b,c,variant; Launch launch; };
} // namespace vibeqc::df_benchmark
"""


def emit_candidate(trial: DfDerivativeTrial) -> typing.Any:
    """Emit only one packet instantiation, sharing all equations and scheduling."""
    a, b, c = trial.angular
    arguments = f"{a},{b},{c},{trial.variant},{str(trial.lowering == 'rys').lower()}"
    return f"""#include "df_benchmark_api.hpp"
#include "scf/cuda/df_shell_kernel.cuh"
namespace vibeqc::df_benchmark {{
cudaError_t {trial.symbol}(DfShellBasisView first,DfShellBasisView second,
    DfShellBasisView third,const double* positions,std::size_t begin,std::size_t count,
    const double* weights,double* gradient,unsigned long long* counters,
    DfDerivativePairs pairs,bool triangle,cudaStream_t stream) {{
  using Schedule=scf::generated_df_shell::Schedule<{a},{b},{c},{trial.variant}>;
  const auto tasks=(triangle ? first.count[{a}]*(first.count[{a}]+1)/2 :
      first.count[{a}]*second.count[{b}])*third.count[{c}];
  if(!tasks) return cudaSuccess;
  scf::SignaturePacket packet;
  packet.count=1;
  packet.blocks=(tasks+Schedule::groups-1)/Schedule::groups;
  packet.slices[0]={{first.shell_ids,second.shell_ids,third.shell_ids,
      first.begin[{a}],second.begin[{b}],third.begin[{c}],
      first.count[{a}],second.count[{b}],third.count[{c}],
      first.primitives,second.primitives,third.primitives,0,tasks,triangle}};
  scf::shell_packet<{arguments}>
      <<<packet.blocks,Schedule::lanes*Schedule::groups,0,stream>>>(
      first,third,positions,begin,count,weights,gradient,counters,pairs,packet,nullptr);
  return cudaPeekAtLastError();
}}
}} // namespace vibeqc::df_benchmark
"""


def emit_driver(trials: typing.Any) -> typing.Any:
    """Register every runnable object in one executable and one GPU allocation."""
    lines = ['#include "df_benchmark_driver.hpp"', "namespace vibeqc::df_benchmark {"]
    signature = (
        "(DfShellBasisView,DfShellBasisView,DfShellBasisView,const double*,"
        "std::size_t,std::size_t,const double*,double*,unsigned long long*,"
        "DfDerivativePairs,bool,cudaStream_t)"
    )
    lines.extend(f"cudaError_t {trial.symbol}{signature};" for trial in trials)
    lines.extend(
        [
            "}",
            "int main(int argc,char** argv) {",
            "using namespace vibeqc::df_benchmark;",
            "return run(argc,argv,{",
        ]
    )
    lines.extend(
        f'{{"{t.key}",{t.angular[0]},{t.angular[1]},{t.angular[2]},'
        f"{t.variant},{t.symbol}}},"
        for t in trials
    )
    lines.extend(["});", "}"])
    return "\n".join(lines) + "\n"
