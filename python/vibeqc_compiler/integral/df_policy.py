"""Typed DF policies connecting shared scientific lowering to basis traversal.

Rank is the number of real Gaussian factors: two for M and three for A.
The native traversal knows no integral family or Gaussian derivative identity.
Raw coordinate projections and externally weighted consumers share these exact
center channels, including the translation-derived auxiliary contribution.
"""


def emit_df_derivative_schedule_cuda():
    """Emit weighted-response scheduling independently of scalar mathematics.

    Four lanes split long primitive products while keeping eight independent
    outputs per warp. A separate artifact lets measured scheduling changes
    leave raw value/derivative consumers and their mathematical policy intact.
    """
    return """// Generated weighted DF schedule; generic runtime performs the reduction.
#ifndef VIBEQC_GENERATED_DF_DERIVATIVE_SCHEDULE_CUH
#define VIBEQC_GENERATED_DF_DERIVATIVE_SCHEDULE_CUH
namespace vibeqc::scf::generated_df_policy {
struct WeightedSchedule {
  static constexpr unsigned block_threads = 32;
  static constexpr unsigned lanes_per_element = 4;
};
} // namespace vibeqc::scf::generated_df_policy
#endif
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
    value = r"""// Split a transformed output's warp across auxiliary source terms and
// primitive products. Four lanes retain contraction parallelism while eight
// source terms progress independently, including short and long contractions.
struct ValueSourceSchedule {
  static constexpr unsigned primitive_lanes = 4;
};
struct Value {
  using Vec3 = generated_df::Vec3;
  using Angular = generated_df::Angular;
  using Accumulator = double;
  template <unsigned Rank>
  __device__ static void accumulate(double& out, const double* e, const Vec3* r,
                                    const Angular* a, double weight) {
    static_assert(Rank == 2 || Rank == 3);
    // Scalar evaluators take references. Isolate their small arguments so an
    // escaped reference does not force the complete traversal state to local
    // memory; the runtime's arrays can then be scalarized into registers.
    const Vec3 first=r[0], second=r[1];
    const Angular first_angular=a[0], second_angular=a[1];
    if constexpr (Rank == 2)
      out += weight * generated_df::metric(e[0],first,first_angular,e[1],second,second_angular);
    else {
      const Vec3 third=r[2]; const Angular third_angular=a[2];
      out += weight * generated_df::three_center(e[0],first,first_angular,e[1],second,second_angular,e[2],third,third_angular);
    }
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
    // Keep callee reference arguments independent of the traversal aggregate.
    const Vec3 first=r[0], second=r[1];
    const Angular first_angular=a[0], second_angular=a[1];
    if constexpr (Rank == 2)
      result = generated_df_derivatives::metric(e[0],first,first_angular,e[1],second,second_angular);
    else {
      const Vec3 third=r[2]; const Angular third_angular=a[2];
      result = generated_df_derivatives::three_center(e[0],first,first_angular,e[1],second,second_angular,e[2],third,third_angular);
    }
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
