"""Compiler-owned directional first-integral matrix contraction plans.

The consumer declares output AO slots and optional fixed external matrix
weights. No HF coefficient, SCF state or device execution policy is inferred.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np

from vibeqc_compiler.common.provenance import canonical_hash

from .first_derivatives_native import (
    emit_first_component_evaluator,
    validate_first_components,
)
from .ir_serialization import integral_to_payload
from .weight_pullback import normalized_cartesian_components


@dataclass(frozen=True)
class DirectionalMatrixTerm:
    """Add coefficient * weight[k,l] * dI(v) into output[slot,i,j].

    Pairs name mathematical AO shell slots, not atom indices. An omitted
    weight_pair means a unit scalar. The external matrix/direction are fixed
    inputs of this primitive; their own responses belong to method assembly.
    """

    output_slot: int
    output_pair: tuple[int, int]
    weight_pair: tuple[int, int] | None = None
    coefficient: float = 1.0

    def __post_init__(self):
        if type(self.output_slot) is not int or not 0 <= self.output_slot < 32:
            raise ValueError("directional output slot must be in [0,32)")
        for name in ("output_pair", "weight_pair"):
            pair = getattr(self, name)
            if pair is None and name == "weight_pair":
                continue
            pair = tuple(pair)
            if len(pair) != 2 or any(
                type(i) is not int or not 0 <= i < 4 for i in pair
            ):
                raise ValueError("directional matrix pairs require two shell slots")
            object.__setattr__(self, name, pair)
        if isinstance(self.coefficient, (bool, complex)) or not math.isfinite(
            self.coefficient
        ):
            raise ValueError("directional coefficient must be finite real")
        object.__setattr__(self, "coefficient", float(self.coefficient))


def validate_directional(integral, indices, terms):
    validate_first_components(integral, indices)
    if any(shell.convention != "cartesian" for shell in integral.signature.shells):
        raise ValueError("directional matrix execution requires Cartesian AO slots")
    if integral.operator.range_separated:
        raise ValueError("directional first execution admits full Coulomb only")
    if not 1 <= len(terms) <= 8 or any(
        not isinstance(t, DirectionalMatrixTerm) for t in terms
    ):
        raise ValueError("declare one to eight directional matrix terms")
    shells = len(integral.signature.shells)
    for term in terms:
        for pair in (term.output_pair, term.weight_pair):
            if pair is not None and any(i >= shells for i in pair):
                raise ValueError("directional matrix term uses a missing shell slot")


def directional_identity(integral, indices, terms):
    validate_directional(integral, indices, terms)
    return canonical_hash(
        {
            "schema": "vibeqc.first-directional-matrix/v1",
            "integral": integral_to_payload(integral),
            "components": tuple(indices),
            "terms": [asdict(t) for t in terms],
            "normalization": "radial-record-times-generated-cartesian-factor",
            "backend": "cuda-fp64",
        }
    )


def emit_directional_matrix(integral, indices, terms, *, runtime_identity):
    """Generate mathematical direction/weight/scatter; reuse primitive DAGs."""
    indices, terms = tuple(indices), tuple(terms)
    identity = directional_identity(integral, indices, terms)
    if len(runtime_identity) != 64 or any(
        c not in "0123456789abcdef" for c in runtime_identity
    ):
        raise ValueError("invalid directional runtime identity")
    source, evaluator = emit_first_component_evaluator(
        integral, indices, backend="cuda"
    )
    nshell, ncenter = len(integral.signature.shells), len(integral.operator.centers)
    shape = integral.signature.component_shape
    coordinates = [np.unravel_index(i, shape) for i in indices]
    components = ",".join(
        "{" + ",".join(str(x) for x in (*row, *((0,) * (4 - nshell)))) + "}"
        for row in coordinates
    )
    scales = [
        scale
        for _, scale in normalized_cartesian_components(
            integral.signature.angular, np.ones(shape)
        )
    ]
    scales = ",".join(float(scales[i]).hex() for i in indices)
    writes = []
    for ordinal, term in enumerate(terms):
        i, j = term.output_pair
        weight = (
            "1.0"
            if term.weight_pair is None
            else f"weights[ao[{term.weight_pair[0]}]*nbf+ao[{term.weight_pair[1]}]]"
        )
        writes += [
            f"    const double term_{ordinal} = {term.coefficient.hex()} * ({weight}) * directional;",
            f"    if (!isfinite(term_{ordinal})) {{ atomicCAS(error,0,1); return; }}",
            f"    atomicAdd(output + {term.output_slot}*nbf*nbf + ao[{i}]*nbf + ao[{j}], term_{ordinal});",
        ]
    out_slots = max(t.output_slot for t in terms) + 1
    source += f"""
#include "integrals/first_directional_runtime.cuh"
namespace {{
using namespace vibeqc::integrals::first_directional;
struct Program {{
  static constexpr unsigned exponents={nshell}, centers={ncenter}, components={len(indices)}, output_slots={out_slots};
  static unsigned extent(unsigned i) {{ const unsigned shape[] = {{{",".join(str(s) for s in shape)}}}; return shape[i]; }}
  __device__ static bool evaluate(const double* input,std::size_t component,double* output) {{
{evaluator}
  }}
  __device__ static void accumulate(const double* record,std::size_t component,Mapping mapping,
      std::size_t nbf,const double* direction,const double* weights,double* output,int* error) {{
    const unsigned indices[][4] = {{{components}}};
    const double normalization[] = {{{scales}}};
    double input[{nshell + 3 * ncenter}]{{}}, gradient[{1 + 3 * ncenter}]{{}};
    for (unsigned s=0;s<exponents;++s) input[s]=record[s];
    for (unsigned c=0;c<3*centers;++c) input[exponents+c]=record[4+c];
    if (!evaluate(input,component,gradient)) {{ atomicCAS(error,0,1); return; }}
    double directional=0;
    for (unsigned c=0;c<centers;++c)
      for (unsigned axis=0;axis<3;++axis)
        directional += gradient[1+3*c+axis]*direction[3*mapping.atoms[c]+axis];
    directional *= record[16]*normalization[component];
    if (!isfinite(directional)) {{ atomicCAS(error,0,1); return; }}
    std::size_t ao[4]{{}};
    for (unsigned s=0;s<exponents;++s) ao[s]=mapping.offsets[s]+indices[component][s];
{chr(10).join(writes)}
  }}
}};
constexpr const char* runtime_identity="{runtime_identity}";
Plan& plan(void* value) {{
  if (!value) throw std::invalid_argument("null directional plan");
  return *static_cast<Plan*>(value);
}}
}}
extern "C" const char* vibeqc_directional_identity_v1() {{ return "{identity}"; }}
extern "C" const char* vibeqc_directional_abi_v1() {{ return runtime_identity; }}
extern "C" int vibeqc_directional_create_v1(int device,int major,int minor,std::size_t nbf,
    std::size_t natoms,std::size_t outputs,std::size_t capacity,std::size_t budget,void** output,
    char* detail,std::size_t size) {{
  if(output) *output=nullptr;
  return boundary([&] {{
    if(!output) throw std::invalid_argument("null directional output handle");
    auto owner=std::make_unique<Plan>(device,major,minor,nbf,natoms,outputs,capacity,budget,runtime_identity);
    *output=owner.release();
  }},detail,size);
}}
extern "C" void vibeqc_directional_destroy_v1(void* value) {{ delete static_cast<Plan*>(value); }}
extern "C" int vibeqc_directional_reset_v1(void* value,const double* weights,std::size_t nw,
    const double* direction,std::size_t nd,char* detail,std::size_t size) {{
  return boundary([&] {{ plan(value).reset(weights,nw,direction,nd,runtime_identity); }},detail,size);
}}
extern "C" int vibeqc_directional_append_v1(void* value,const double* records,std::size_t count,
    const Mapping* mapping,char* detail,std::size_t size) {{
  return boundary([&] {{
    if(!mapping) throw std::invalid_argument("null directional mapping");
    append<Program>(plan(value),records,count,*mapping,runtime_identity);
  }},detail,size);
}}
extern "C" int vibeqc_directional_finish_v1(void* value,double* output,std::size_t count,
    char* detail,std::size_t size) {{
  return boundary([&] {{ plan(value).finish(output,count,runtime_identity); }},detail,size);
}}
"""
    return source
