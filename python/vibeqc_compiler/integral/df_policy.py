"""Typed DF policies connecting shared scientific lowering to basis traversal.

Rank is the number of real Gaussian factors: two for M and three for A.
The native traversal knows no integral family or Gaussian derivative identity.
Raw coordinate projections and externally weighted consumers share these exact
center channels, including the translation-derived auxiliary contribution.
"""


def emit_df_policy_cuda(*, derivatives=False):
    """Emit one consumer's policy without registering unused device tables.

    CUDA emits host registration symbols even for device definitions. Keep
    values separate so a derivative-only TU carries no unused Rys value data;
    scalar headers themselves use internal linkage for safe multi-TU reuse.
    """
    guard = (
        "VIBEQC_GENERATED_DF_"
        + ("DERIVATIVE" if derivatives else "VALUE")
        + "_POLICY_CUH"
    )
    header = "generated_df_derivatives.cuh" if derivatives else "df_values.cuh"
    value = r"""struct Value {
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
"""
    derivative = r"""struct Derivative {
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
"""
    return (
        "// Generated DF policy; runtime owns normalized basis traversal.\n"
        f"#ifndef {guard}\n#define {guard}\n"
        f'#include "{header}"\n'
        "namespace vibeqc::scf::generated_df_policy {\n"
        + (derivative if derivatives else value)
        + "} // namespace vibeqc::scf::generated_df_policy\n#endif\n"
    )
