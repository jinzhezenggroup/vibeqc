"""Compiler-owned weighted first-integral gradient contraction plans."""

from __future__ import annotations

import math
import typing
from dataclasses import asdict, dataclass

import numpy as np

from vibeqc_compiler.common.provenance import canonical_hash

from .first_derivatives_native import (
    emit_first_component_evaluator,
    validate_first_components,
)
from .ir_serialization import integral_to_payload
from .weight_pullback import normalized_cartesian_components

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

    from .ir import IntegralIR


@dataclass(frozen=True)
class FirstGradientWeight:
    """One external AO matrix factor addressed by shell-slot pair."""

    matrix_slot: int
    pair: tuple[int, int]

    def __post_init__(self) -> None:
        if type(self.matrix_slot) is not int or not 0 <= self.matrix_slot < 8:
            raise ValueError("first-gradient matrix slot must be in [0,8)")
        pair = tuple(self.pair)
        if len(pair) != 2 or any(type(i) is not int or not 0 <= i < 4 for i in pair):
            raise ValueError("first-gradient AO pair requires two shell slots")
        object.__setattr__(self, "pair", pair)


@dataclass(frozen=True)
class FirstGradientTerm:
    """Add coefficient * product(weights) * dI/dR to atomic Cartesian output."""

    weights: tuple[FirstGradientWeight, ...]
    coefficient: float = 1.0

    def __post_init__(self) -> None:
        weights = tuple(self.weights)
        if not 1 <= len(weights) <= 2 or any(
            not isinstance(weight, FirstGradientWeight) for weight in weights
        ):
            raise ValueError(
                "first-gradient term requires one or two AO matrix weights"
            )
        if isinstance(self.coefficient, (bool, complex)) or not math.isfinite(
            self.coefficient
        ):
            raise ValueError("first-gradient coefficient must be finite real")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "coefficient", float(self.coefficient))


def validate_first_gradient(
    integral: IntegralIR, indices: Sequence[int], terms: Sequence[FirstGradientTerm]
) -> None:
    validate_first_components(integral, indices)
    if any(shell.convention != "cartesian" for shell in integral.signature.shells):
        raise ValueError("first-gradient execution requires Cartesian AO slots")
    if integral.operator.range_separated:
        raise ValueError("first-gradient execution admits full Coulomb only")
    if not 1 <= len(terms) <= 8 or any(
        not isinstance(t, FirstGradientTerm) for t in terms
    ):
        raise ValueError("declare one to eight first-gradient terms")
    shells = len(integral.signature.shells)
    for term in terms:
        for weight in term.weights:
            if any(i >= shells for i in weight.pair):
                raise ValueError("first-gradient term uses a missing shell slot")


def first_gradient_identity(
    integral: IntegralIR, indices: Sequence[int], terms: Sequence[FirstGradientTerm]
) -> str:
    validate_first_gradient(integral, indices, terms)
    return canonical_hash(
        {
            "schema": "vibeqc.first-weighted-gradient/v1",
            "integral": integral_to_payload(integral),
            "components": tuple(indices),
            "terms": [asdict(t) for t in terms],
            "normalization": "radial-record-times-generated-cartesian-factor",
            "backend": "cuda-fp64",
        }
    )


def emit_first_gradient(
    integral: IntegralIR,
    indices: Sequence[int],
    terms: Sequence[FirstGradientTerm],
    *,
    runtime_identity: str,
) -> str:
    """Generate AO-weight pullback into atomic gradients using primitive DAGs."""
    indices, terms = tuple(indices), tuple(terms)
    identity = first_gradient_identity(integral, indices, terms)
    if len(runtime_identity) != 64 or any(
        c not in "0123456789abcdef" for c in runtime_identity
    ):
        raise ValueError("invalid first-gradient runtime identity")
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
        factors = []
        for weight in term.weights:
            i, j = weight.pair
            factors.append(f"weights[{weight.matrix_slot}*nbf*nbf+ao[{i}]*nbf+ao[{j}]]")
        product = " * ".join(f"({factor})" for factor in factors)
        writes += [
            f"    const double term_{ordinal} = {float(term.coefficient).hex()} * ({product}) * radial;",
            f"    if (!isfinite(term_{ordinal})) {{ atomicCAS(error,0,1); return; }}",
            "    for (unsigned c=0;c<centers;++c) for (unsigned axis=0;axis<3;++axis)",
            f"      atomicAdd(output+3*mapping.atoms[c]+axis, term_{ordinal}*gradient[1+3*c+axis]);",
        ]
    weight_slots = (
        max(weight.matrix_slot for term in terms for weight in term.weights) + 1
    )
    source += f"""
#include "integrals/first_gradient_runtime.cuh"
namespace {{
using namespace vibeqc::integrals::first_gradient;
struct Program {{
  static constexpr unsigned exponents={nshell}, centers={ncenter}, components={len(indices)}, weight_slots={weight_slots};
  static unsigned extent(unsigned i) {{ const unsigned shape[] = {{{",".join(str(s) for s in shape)}}}; return shape[i]; }}
  __device__ static bool evaluate(const double* input,std::size_t component,double* output) {{
{evaluator}
  }}
  __device__ static void accumulate(const double* record,std::size_t component,Mapping mapping,
      std::size_t nbf,const double* weights,double* output,int* error) {{
    const unsigned indices[][4] = {{{components}}};
    const double normalization[] = {{{scales}}};
    double input[{nshell + 3 * ncenter}]{{}}, gradient[{1 + 3 * ncenter}]{{}};
    for (unsigned s=0;s<exponents;++s) input[s]=record[s];
    for (unsigned c=0;c<3*centers;++c) input[exponents+c]=record[4+c];
    if (!evaluate(input,component,gradient)) {{ atomicCAS(error,0,1); return; }}
    const double radial=record[16]*normalization[component];
    if (!isfinite(radial)) {{ atomicCAS(error,0,1); return; }}
    std::size_t ao[4]{{}};
    for (unsigned s=0;s<exponents;++s) ao[s]=mapping.offsets[s]+indices[component][s];
{chr(10).join(writes)}
  }}
}};
constexpr const char* runtime_identity="{runtime_identity}";
Plan& plan(void* value) {{
  if (!value) throw std::invalid_argument("null first-gradient plan");
  return *static_cast<Plan*>(value);
}}
}}
extern "C" const char* vibeqc_first_gradient_identity_v1() {{ return "{identity}"; }}
extern "C" const char* vibeqc_first_gradient_abi_v1() {{ return runtime_identity; }}
extern "C" int vibeqc_first_gradient_create_v1(int device,int major,int minor,std::size_t nbf,
    std::size_t natoms,std::size_t weight_slots,std::size_t capacity,std::size_t budget,void** output,
    char* detail,std::size_t size) {{
  if(output) *output=nullptr;
  return boundary([&] {{
    if(!output) throw std::invalid_argument("null first-gradient output handle");
    auto owner=std::make_unique<Plan>(device,major,minor,nbf,natoms,weight_slots,capacity,budget,runtime_identity);
    *output=owner.release();
  }},detail,size);
}}
extern "C" void vibeqc_first_gradient_destroy_v1(void* value) {{ delete static_cast<Plan*>(value); }}
extern "C" int vibeqc_first_gradient_reset_v1(void* value,const double* weights,std::size_t count,
    char* detail,std::size_t size) {{
  return boundary([&] {{ plan(value).reset(weights,count,runtime_identity); }},detail,size);
}}
extern "C" int vibeqc_first_gradient_append_v1(void* value,const double* records,std::size_t count,
    const Mapping* mapping,char* detail,std::size_t size) {{
  return boundary([&] {{
    if(!mapping) throw std::invalid_argument("null first-gradient mapping");
    append<Program>(plan(value),records,count,*mapping,runtime_identity);
  }},detail,size);
}}
extern "C" int vibeqc_first_gradient_finish_v1(void* value,double* output,std::size_t count,
    char* detail,std::size_t size) {{
  return boundary([&] {{ plan(value).finish(output,count,runtime_identity); }},detail,size);
}}
"""
    return source
