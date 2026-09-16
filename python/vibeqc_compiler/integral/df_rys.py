"""Shared one/two-root quadrature for bounded full-range DF derivatives.

Nodes are t², with weight integrating exp(-T*t²) on [0,1]. The two-root
path uses independently generated Chebyshev polynomials and an analytic large-T
limit; it never inverts a floating-point moment matrix at runtime.
The independent incomplete-gamma reference lives in the qualification tests.
"""

import math

from .df_rys2_data import DF_RYS2_COEFFICIENTS

ASYMPTOTIC_ARGUMENT = 40.0

TAYLOR_COEFFICIENTS = tuple(
    tuple(1.0 / (math.factorial(k) * (2 * k + 2 * order + 1)) for k in range(25))
    for order in (0, 1)
)


def rys_roots(argument, nroots=1):
    """Evaluate ordered FP64 t² nodes; this host aid is not a runtime dependency."""
    if not math.isfinite(argument) or argument < 0 or nroots not in (1, 2):
        raise ValueError("DF Rys requires finite T >= 0 and one or two roots")
    if nroots == 2:
        if argument >= 48.0:
            x0, x1 = (3 - math.sqrt(6)) / 2, (3 + math.sqrt(6)) / 2
            w1 = (math.sqrt(6) - 2) / (2 * math.sqrt(6))
            scale = 0.5 * math.sqrt(math.pi) / math.sqrt(argument)
            return (x0 / argument, x1 / argument), ((1 - w1) * scale, w1 * scale)
        interval = int(argument / 2)
        x = argument - (2 * interval + 1)
        values = []
        for coefficients in DF_RYS2_COEFFICIENTS[4 * interval : 4 * interval + 4]:
            b1 = b2 = 0.0
            for coefficient in coefficients[:0:-1]:
                b0 = 2 * x * b1 - b2 + coefficient
                b2, b1 = b1, b0
            values.append(x * b1 - b2 + coefficients[0])
        return tuple(values[:2]), tuple(values[2:])
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
    two_root_coefficients = ",\n".join(
        "  {" + ",".join(map(repr, row)) + "}" for row in DF_RYS2_COEFFICIENTS
    )
    return (
        (
            r"""// Generated low-order DF Rys quadrature; nodes are t^2.
#pragma once
#include <cmath>
namespace vibeqc::scf::generated_df_rys {
static __device__ const double two_root_coefficients[96][18] = {
__TWO_ROOT_COEFFICIENTS__
};
static __device__ const double taylor_coefficients[2][25] = {
"""
            + coefficients
            + r"""
};
template<unsigned N>
__device__ __forceinline__ void roots(double argument,double* nodes,double* weights) {
  static_assert(N==1 || N==2, "Only bounded full-range one/two-root DF is available");
  if constexpr(N==2) {
    if(argument>=48.0) {
      // Generalized Laguerre (alpha=-1/2) limit. At T=48 the omitted
      // relative F3 tail is below 8e-18; lower moments have smaller tails.
      nodes[0]=0.27525512860841095090/argument;
      nodes[1]=2.7247448713915890491/argument;
      const double scale=0.88622692545275801365/sqrt(argument);
      weights[0]=0.90824829046386301637*scale;
      weights[1]=0.09175170953613698363*scale;
    } else {
      const unsigned interval=static_cast<unsigned>(argument*0.5);
      const double x=argument-(2*interval+1);
      double values[4];
#pragma unroll
      for(unsigned series=0;series<4;++series) {
        const double* c=two_root_coefficients[4*interval+series];
        double b1=0,b2=0;
#pragma unroll
        for(int k=17;k>0;--k) {
          const double b0=fma(2*x,b1,c[k]-b2);
          b2=b1;b1=b0;
        }
        values[series]=fma(x,b1,c[0]-b2);
      }
      nodes[0]=values[0];nodes[1]=values[1];
      weights[0]=values[2];weights[1]=values[3];
    }
    return;
  }
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
        )
        .replace("ASYMPTOTIC_ARGUMENT", repr(ASYMPTOTIC_ARGUMENT))
        .replace("__TWO_ROOT_COEFFICIENTS__", two_root_coefficients)
    )
