#ifndef VIBEQC_INTEGRALS_RANGE_MOMENTS_HPP
#define VIBEQC_INTEGRALS_RANGE_MOMENTS_HPP

#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define VIBEQC_RANGE_HD __host__ __device__
#else
#define VIBEQC_RANGE_HD
#endif

namespace vibeqc::integrals {

/** Radial kernels in atomic units. Omega is finite, nonnegative, inverse Bohr.
 * Nuclear derivatives hold omega fixed. No omega derivatives or screening
 * bounds are provided by this primitive moment evaluator.
 */
enum class CoulombRange : std::uint32_t { Full = 0, Long = 1, Short = 2 };

/** Positive interval quadrature with an explicit compile-time moment bound.
 *
 * LR integrates u^(2n) exp(-T u^2) over [0, omega/hypot(omega,sqrt(rho))];
 * SR integrates over the complementary interval. In particular, SR never
 * subtracts nearly equal full/LR values. Its width uses a rational expression
 * that remains accurate when the lower endpoint rounds to one.
 *
 * This intentionally conservative experimental evaluator uses a 64-point
 * Gauss-Legendre rule, with positive weights and factored powers/decay. A tail
 * beyond an additional exponent of 90+2*nmax is negligible for these orders;
 * clipping it resolves arbitrarily narrow peaks at large T. The full-range
 * branch is for standalone generated consumers, not a replacement for the
 * established ordinary-Coulomb evaluators.
 *
 * On invalid inputs, returns false without modifying output. The caller owns
 * maximum_order+1 doubles. Geometry derivatives consume the next moment via
 * dF_n/dT = -F_(n+1), rather than differentiating quadrature or clipping branches.
 */
template <unsigned MaximumOrder>
VIBEQC_RANGE_HD inline bool bounded_range_moments(unsigned maximum_order, double argument,
                                                  double rho, CoulombRange range, double omega,
                                                  double* output) {
  static_assert(MaximumOrder <= 14, "the validated quadrature domain ends at order 14");
  if (!output || maximum_order > MaximumOrder || !std::isfinite(argument) || argument < 0 ||
      !std::isfinite(rho) || rho <= 0 || !std::isfinite(omega) || omega < 0 ||
      static_cast<std::uint32_t>(range) > 2 || (range == CoulombRange::Full && omega != 0))
    return false;

  const double root_rho = std::sqrt(rho);
  const double radius = std::hypot(omega, root_rho);
  const double boundary = omega / radius;
  double lower = 0, width = 1;
  if (range == CoulombRange::Long) width = boundary;
  if (range == CoulombRange::Short) {
    lower = boundary;
    const double ratio = root_rho / radius;
    width = ratio * (ratio / (1 + boundary));
  }
  for (unsigned n = 0; n <= maximum_order; ++n) output[n] = 0;
  if (width == 0) return true;
  const double cutoff = 90.0 + 2 * maximum_order;
  if (argument * width * (2 * lower + width) > cutoff) {
    const double scaled = cutoff / argument;
    width = scaled / (std::hypot(lower, std::sqrt(scaled)) + lower);
  }
  const double decay = std::exp(-argument * lower * lower);
  if (decay == 0) return true;
  const double upper = lower + width;
  // Positive roots and weights of P_64 on [-1,1], stored as exact FP64
  // hexadecimal literals. Reflecting each node supplies the full rule.
  constexpr double rule[32][2] = {
      {0x1.8ef487a8cbc32p-6, 0x1.8ee0567ee2e5dp-5}, {0x1.2afad5ee95ad0p-4, 0x1.8dee238192cdcp-5},
      {0x1.f182ff48e8a26p-4, 0x1.8c0a5097676c0p-5}, {0x1.5b6e88ad5c00fp-3, 0x1.89360387fe3b9p-5},
      {0x1.bd489b79ec83bp-3, 0x1.8572f41fbb53dp-5}, {0x1.0f0a26c56e49cp-2, 0x1.80c36b24bdd21p-5},
      {0x1.3ecb6c46c76cbp-2, 0x1.7b2a40f3ccde2p-5}, {0x1.6dcb1f0620fffp-2, 0x1.74aadbc614fb9p-5},
      {0x1.9becb55272c9dp-2, 0x1.6d492da0c2510p-5}, {0x1.c9142c5898fc5p-2, 0x1.6509b1efb8dfcp-5},
      {0x1.f52619257c3a1p-2, 0x1.5bf16accdf431p-5}, {0x1.1003dca600f34p-1, 0x1.5205ddf5a36e2p-5},
      {0x1.24cf81925487fp-1, 0x1.474d117092830p-5}, {0x1.38e95ace7b3c3p-1, 0x1.3bcd87e50de1ep-5},
      {0x1.4c4533c68b412p-1, 0x1.2f8e3ca7574e3p-5}, {0x1.5ed74b4532f83p-1, 0x1.22969f7b5c8bdp-5},
      {0x1.70945a96f12c4p-1, 0x1.14ee9010d92e3p-5}, {0x1.81719c62ec68ep-1, 0x1.069e593b92368p-5},
      {0x1.9164d335425e2p-1, 0x1.ef5d57d53b4b8p-6}, {0x1.a0644fb6d8db8p-1, 0x1.d05133c3af946p-6},
      {0x1.ae66f68eedbc6p-1, 0x1.b02b2071c0c76p-6}, {0x1.bb6445eadae2cp-1, 0x1.8efea34684612p-6},
      {0x1.c7545aa8c0dadp-1, 0x1.6cdfe10bba36ep-6}, {0x1.d22ff5221288ap-1, 0x1.49e391bd2143cp-6},
      {0x1.dbf07d935a5afp-1, 0x1.261ef40a7a2d6p-6}, {0x1.e490081f2891bp-1, 0x1.01a7c0a5c98d9p-6},
      {0x1.ec09586b58faap-1, 0x1.b9283b35df8d9p-7}, {0x1.f257e4db5aabcp-1, 0x1.6df524de84deap-7},
      {0x1.f777d976cfadap-1, 0x1.21e400109d33cp-7}, {0x1.fb661ac8c85a9p-1, 0x1.aa46b24145aa3p-8},
      {0x1.fe204ab274eccp-1, 0x1.0fc7ac3ac3b55p-8}, {0x1.ffa4e911f7533p-1, 0x1.d379f1845dadfp-10},
  };
  for (unsigned k = 0; k < 32; ++k) {
    for (int sign = -1; sign <= 1; sign += 2) {
      const double delta = width * (0.5 + sign * 0.5 * rule[k][0]);
      const double unit = (lower + delta) / upper;
      const double squared = unit * unit;
      double term = 0.5 * rule[k][1] * std::exp(-argument * delta * (2 * lower + delta));
      for (unsigned n = 0; n <= maximum_order; ++n) {
        output[n] += term;
        term *= squared;
      }
    }
  }
  double factor = width * decay;
  for (unsigned n = 0; n <= maximum_order; ++n) {
    output[n] *= factor;
    factor *= upper * upper;
  }
  return true;
}

/** Preserve the first-derivative moment boundary and caller storage contract.
 * Second-order geometry must opt into bounded_range_moments<14> with fifteen
 * owned doubles. Calling the established API with order fourteen still fails
 * before touching its output, protecting existing fourteen-element buffers.
 */
VIBEQC_RANGE_HD inline bool range_moments(unsigned maximum_order, double argument, double rho,
                                          CoulombRange range, double omega, double* output) {
  return bounded_range_moments<13>(maximum_order, argument, rho, range, omega, output);
}
}  // namespace vibeqc::integrals
#undef VIBEQC_RANGE_HD
#endif
