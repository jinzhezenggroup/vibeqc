"""Typed DF policies connecting shared scientific lowering to basis traversal.

Rank is the number of real Gaussian factors: two for M and three for A.
The native traversal knows no integral family or Gaussian derivative identity.
Raw coordinate projections and externally weighted consumers share these exact
center channels, including the translation-derived auxiliary contribution.
"""


def emit_df_policy_cuda():
    """Emit value and first-response policies with one common runtime contract."""
    return r"""// Generated DF value/response policies; runtime owns basis traversal.
#ifndef VIBEQC_GENERATED_DF_POLICY_CUH
#define VIBEQC_GENERATED_DF_POLICY_CUH
#include "df_values.cuh"
#include "generated_df_derivatives.cuh"
namespace vibeqc::scf::generated_df_policy {
struct Value {
  using Vec3 = generated_df::Vec3;
  using Angular = generated_df::Angular;
  using Accumulator = double;
  template <unsigned Rank>
  __device__ static void accumulate(double& out, const double* e, const Vec3* r,
                                    const Angular* a, double weight) {
    static_assert(Rank == 2 || Rank == 3);
    if constexpr (Rank == 2)
      out += weight * generated_df::metric(e[0],r[0],a[0],e[1],r[1],a[1]);
    else
      out += weight * generated_df::three_center(e[0],r[0],a[0],e[1],r[1],a[1],e[2],r[2],a[2]);
  }
};
struct Derivative {
  using Vec3 = generated_df_derivatives::Vec3;
  using Angular = generated_df_derivatives::Angular;
  struct Accumulator { double gradient[3][3]{}; };
  template <unsigned Rank>
  __device__ static void accumulate(Accumulator& out, const double* e, const Vec3* r,
                                    const Angular* a, double weight) {
    static_assert(Rank == 2 || Rank == 3);
    generated_df_derivatives::Response result;
    if constexpr (Rank == 2)
      result = generated_df_derivatives::metric(e[0],r[0],a[0],e[1],r[1],a[1]);
    else
      result = generated_df_derivatives::three_center(e[0],r[0],a[0],e[1],r[1],a[1],e[2],r[2],a[2]);
    const Vec3 channels[3]{result.first, Rank == 2 ? result.third : result.second, result.third};
#pragma unroll
    for (unsigned slot=0;slot<Rank;++slot) {
      out.gradient[slot][0] += weight*channels[slot].x;
      out.gradient[slot][1] += weight*channels[slot].y;
      out.gradient[slot][2] += weight*channels[slot].z;
    }
  }
};
} // namespace vibeqc::scf::generated_df_policy
#endif
"""
