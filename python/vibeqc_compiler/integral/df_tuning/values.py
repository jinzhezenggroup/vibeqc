"""Typed raw-value candidates using the same scalar IR as native DF sources."""

import hashlib
import json
import typing
from dataclasses import asdict, dataclass

from ..df_value_candidates import VALUE_CLASSES

VALUE_LOWERINGS = ("generic", "polynomial", "rys")
VALUE_LANES = (1, 4, 32)


@dataclass(frozen=True)
class DfValueTrial:
    """Bind value consumer, class, lowering and primitive reduction schedule."""

    angular: tuple[int, int, int]
    lowering: str
    lanes: int
    consumer: str = "raw_cartesian"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.angular, tuple)
            or any(type(l) is not int for l in self.angular)
            or self.angular not in VALUE_CLASSES
            or self.lowering not in VALUE_LOWERINGS
        ):
            raise ValueError("unsupported value lowering/class")
        if type(self.lanes) is not int or self.lanes not in VALUE_LANES:
            raise ValueError("unsupported primitive reduction schedule")
        if self.consumer != "raw_cartesian":
            raise ValueError("this experiment measures raw Cartesian value work")

    @property
    def key(self) -> typing.Any:
        return f"{self.consumer}:{''.join(map(str, self.angular))}:{self.lowering}:lanes{self.lanes}"

    @property
    def symbol(self) -> typing.Any:
        return "df_value_" + self.key.replace(":", "_")

    @property
    def block_threads(self) -> typing.Any:
        return 128

    def artifact_key(
        self,
        *,
        generator_sha256: typing.Any,
        architecture: typing.Any,
        toolchain: typing.Any,
    ) -> typing.Any:
        """Consumer participates so value/derivative artifacts cannot collide."""
        if not all(
            isinstance(x, str) and x
            for x in (generator_sha256, architecture, toolchain)
        ):
            raise ValueError("complete generator/target/toolchain identity required")
        return hashlib.sha256(
            json.dumps(
                {
                    "schema": 1,
                    "trial": asdict(self),
                    "generator": generator_sha256,
                    "architecture": architecture,
                    "toolchain": toolchain,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()


def enumerate_value_trials() -> typing.Any:
    """Only implemented class-specific math enters the finite batch search."""
    return tuple(
        DfValueTrial((a[0], a[1], a[2]), math, lanes)
        for a in VALUE_CLASSES
        for math in VALUE_LOWERINGS
        for lanes in VALUE_LANES
    )


VALUE_API = r"""#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include "df_values.cuh"
namespace vibeqc::df_value_benchmark {
namespace scalar=scf::generated_df;
struct Input {
  unsigned count; scalar::Angular angular[3]; scalar::Vec3 centers[3];
  double exponents[3],weight;
};
static_assert(sizeof(Input)==144 && offsetof(Input,centers)==40);
using Launch=cudaError_t(*)(const Input*,double*,std::size_t,std::size_t,std::size_t);
struct Candidate { const char* key; unsigned a,b,c; Launch launch; };
}
"""


def emit_value_candidate(trial: typing.Any) -> typing.Any:
    """A bounded contraction schedule; all integral equations remain generated."""
    a, b, c = trial.angular
    evaluator = (
        "scalar::three_center"
        if trial.lowering == "generic"
        else (
            f"scf::generated_df_value_candidates::Value<{a},{b},{c},"
            f"{str(trial.lowering == 'rys').lower()}>::evaluate"
        )
    )
    return f"""#include "df_value_benchmark_api.hpp"
#include "generated_df_value_candidates.cuh"
namespace vibeqc::df_value_benchmark {{
namespace {{
__global__ void value_candidate(const Input* in,double* out,std::size_t tasks,
                               std::size_t primitives,std::size_t replicas) {{
  constexpr unsigned lanes={trial.lanes};
  const auto tid=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  const auto task=tid/lanes; const unsigned lane=threadIdx.x%lanes;
  if(task>=tasks*replicas) return;
  double value=0;
  for(std::size_t p=lane;p<primitives;p+=lanes) {{
    const auto& x=in[(task%tasks)*primitives+p];
    value+=x.weight*{evaluator}(x.exponents[0],x.centers[0],x.angular[0],
      x.exponents[1],x.centers[1],x.angular[1],x.exponents[2],x.centers[2],x.angular[2]);
  }}
  // Every contraction owns a complete power-of-two subgroup, including tails.
  const unsigned mask=__activemask();
  for(unsigned d=lanes/2;d;d/=2) value+=__shfl_down_sync(mask,value,d,lanes);
  if(lane==0) out[task]=value;
}}
}}
cudaError_t {trial.symbol}(const Input* in,double* out,std::size_t tasks,
                          std::size_t primitives,std::size_t replicas) {{
  value_candidate<<<(tasks*replicas*{trial.lanes}+127)/128,128>>>(in,out,tasks,primitives,replicas);
  return cudaPeekAtLastError();
}}
}}
"""


def emit_value_driver(trials: typing.Any) -> typing.Any:
    """Link all eligible value objects into the shared finite benchmark driver."""
    lines = [
        '#include "df_value_benchmark_driver.hpp"',
        "namespace vibeqc::df_value_benchmark {",
    ]
    lines += [
        f"cudaError_t {t.symbol}(const Input*,double*,std::size_t,std::size_t,std::size_t);"
        for t in trials
    ]
    lines += [
        "}",
        "int main(int argc,char** argv) {",
        "using namespace vibeqc::df_value_benchmark;",
        "return run(argc,argv,{",
    ]
    lines += [
        f'{{"{t.key}",{t.angular[0]},{t.angular[1]},{t.angular[2]},{t.symbol}}},'
        for t in trials
    ]
    return "\n".join(lines + ["});", "}", ""])
