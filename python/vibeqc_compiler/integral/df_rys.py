"""One-root Rys quadrature for the bounded 000 first-derivative lowering.

Nodes are t², with weight integrating exp(-T*t²) on [0,1]. Only moments zero
and one are required. Higher root counts deliberately remain outside this slice.
The independent incomplete-gamma reference lives in the qualification tests.
"""

import math

ASYMPTOTIC_ARGUMENT = 40.0

TAYLOR_COEFFICIENTS = tuple(
    tuple(1.0 / (math.factorial(k) * (2 * k + 2 * order + 1)) for k in range(25))
    for order in (0, 1)
)


def rys_roots(argument, nroots=1):
    """Evaluate one FP64 node/weight; this host aid is never a runtime dependency."""
    if not math.isfinite(argument) or argument < 0 or nroots != 1:
        raise ValueError("000 Rys requires finite T >= 0 and exactly one root")
    if argument >= ASYMPTOTIC_ARGUMENT:
        return (0.5 / argument,), (0.5 * math.sqrt(math.pi) / math.sqrt(argument),)
    if argument < 0.5:
        moments = []
        for coefficients in TAYLOR_COEFFICIENTS:
            value = 0.0
            for coefficient in reversed(coefficients):
                value = -argument * value + coefficient
            moments.append(value)
        return (moments[1] / moments[0],), (moments[0],)
    weight = (
        0.5 * math.sqrt(math.pi) * math.erf(math.sqrt(argument)) / math.sqrt(argument)
    )
    return ((0.5 / argument) * (1.0 - math.exp(-argument) / weight),), (weight,)


def emit_df_rys_cuda():
    """Emit independently owned analytic quadrature, stable at T=0 and large T.

    The small-T branch avoids cancellation in F1. Its fixed Taylor polynomial
    has an independently tested remainder; the ordinary branch uses erf/exp.
    Above T=40, the omitted F1 tail is below 4e-17 relative (F0 is smaller),
    beneath FP64 epsilon. This avoids unnecessary transcendental work in the
    asymptotic domain. Dividing before multiplying preserves extreme finite T.
    """
    coefficients = ",\n".join(
        "  {" + ",".join(map(repr, row)) + "}" for row in TAYLOR_COEFFICIENTS
    )
    return (
        r"""// Generated one-root DF Rys quadrature; nodes are t^2.
#pragma once
#include <cmath>
namespace vibeqc::scf::generated_df_rys {
static __device__ const double taylor_coefficients[2][25] = {
"""
        + coefficients
        + r"""
};
template<unsigned N>
__device__ __forceinline__ void roots(double argument,double* nodes,double* weights) {
  static_assert(N==1, "Only the 000 one-root slice is qualified here");
  if(argument>=ASYMPTOTIC_ARGUMENT) {
    nodes[0]=0.5/argument;
    weights[0]=0.88622692545275801365/sqrt(argument);
  } else if(argument<0.5) {
    double moments[2]{};
#pragma unroll
    for(unsigned order=0;order<2;++order) {
      double value=0;
#pragma unroll
      for(int k=24;k>=0;--k) value=fma(-argument,value,taylor_coefficients[order][k]);
      moments[order]=value;
    }
    nodes[0]=moments[1]/moments[0];weights[0]=moments[0];
  } else {
    const double square_root=sqrt(argument);
    weights[0]=0.88622692545275801365*erf(square_root)/square_root;
    nodes[0]=(0.5/argument)*(1-exp(-argument)/weights[0]);
  }
}
} // namespace vibeqc::scf::generated_df_rys
"""
    ).replace("ASYMPTOTIC_ARGUMENT", repr(ASYMPTOTIC_ARGUMENT))
