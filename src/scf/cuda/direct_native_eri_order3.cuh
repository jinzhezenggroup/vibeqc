#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/boys_table.cuh"
#include "scf/cuda/cartesian_angular.cuh"
#include "scf/cuda/direct_native_pair_order2.cuh"
#include "scf/cuda/direct_native_pair_order3.cuh"
#include "scf/cuda/gaussian_geometry.cuh"
#include "scf/cuda/packed_basis.hpp"
#include "scf/cuda/scalar_math.cuh"

// Retained direct integral arithmetic for eri order3.
// Shared definitions use ordinary inline linkage; host plans and queue policy
// remain outside this numerical owner.
namespace vibeqc::scf::cuda_execution {

/** Value and P-Q chain derivatives of a weighted order-two Hermite DAG. */
struct WeightedOrder2Coulomb {
  double c0;
  double cx;
  double cy;
  double cz;
  double value;
  double chain[3];
};

/**
 * Contract the ten order-zero-through-two Hermite coefficients with exactly
 * twenty unique Cartesian Coulomb states. The returned chain is only the
 * derivative through P-Q; shell-class helpers add their explicit PA/PB/QC
 * coefficient derivatives and Gaussian-pair decay separately.
 */
__device__ __forceinline__ WeightedOrder2Coulomb contract_weighted_order2_coulomb(
    double rho, double x, double y, double z, const double (&boys)[4], double h0, double hx,
    double hy, double hz, double hxx, double hxy, double hxz, double hyy, double hyz, double hzz) {
  const double twice_rho = 2.0 * rho;
  const double twice_rho_squared = twice_rho * twice_rho;
  const double twice_rho_cubed = twice_rho_squared * twice_rho;
  const double c0 = boys[0];
  const double cx = -twice_rho * x * boys[1];
  const double cy = -twice_rho * y * boys[1];
  const double cz = -twice_rho * z * boys[1];
  const double cxx = twice_rho_squared * x * x * boys[2] - twice_rho * boys[1];
  const double cxy = twice_rho_squared * x * y * boys[2];
  const double cxz = twice_rho_squared * x * z * boys[2];
  const double cyy = twice_rho_squared * y * y * boys[2] - twice_rho * boys[1];
  const double cyz = twice_rho_squared * y * z * boys[2];
  const double czz = twice_rho_squared * z * z * boys[2] - twice_rho * boys[1];
  const double cxxx =
      -twice_rho_cubed * x * x * x * boys[3] + 3.0 * twice_rho_squared * x * boys[2];
  const double cxxy = -twice_rho_cubed * x * x * y * boys[3] + twice_rho_squared * y * boys[2];
  const double cxxz = -twice_rho_cubed * x * x * z * boys[3] + twice_rho_squared * z * boys[2];
  const double cxyy = -twice_rho_cubed * x * y * y * boys[3] + twice_rho_squared * x * boys[2];
  const double cxyz = -twice_rho_cubed * x * y * z * boys[3];
  const double cxzz = -twice_rho_cubed * x * z * z * boys[3] + twice_rho_squared * x * boys[2];
  const double cyyy =
      -twice_rho_cubed * y * y * y * boys[3] + 3.0 * twice_rho_squared * y * boys[2];
  const double cyyz = -twice_rho_cubed * y * y * z * boys[3] + twice_rho_squared * z * boys[2];
  const double cyzz = -twice_rho_cubed * y * z * z * boys[3] + twice_rho_squared * y * boys[2];
  const double czzz =
      -twice_rho_cubed * z * z * z * boys[3] + 3.0 * twice_rho_squared * z * boys[2];

  WeightedOrder2Coulomb result{};
  result.c0 = c0;
  result.cx = cx;
  result.cy = cy;
  result.cz = cz;
  result.value = h0 * c0 + hx * cx + hy * cy + hz * cz + hxx * cxx + hxy * cxy + hxz * cxz +
                 hyy * cyy + hyz * cyz + hzz * czz;
  result.chain[0] = h0 * cx + hx * cxx + hy * cxy + hz * cxz + hxx * cxxx + hxy * cxxy +
                    hxz * cxxz + hyy * cxyy + hyz * cxyz + hzz * cxzz;
  result.chain[1] = h0 * cy + hx * cxy + hy * cyy + hz * cyz + hxx * cxxy + hxy * cxyy +
                    hxz * cxyz + hyy * cyyy + hyz * cyyz + hzz * cyzz;
  result.chain[2] = h0 * cz + hx * cxz + hy * cyz + hz * czz + hxx * cxxz + hxy * cxyz +
                    hxz * cxzz + hyy * cyyz + hyz * cyzz + hzz * czzz;
  return result;
}

/**
 * Closed order-3 contraction for canonical (f s|s s), (d p|s s),
 * (d s|p s), and (p p|p s) primitive quartets.
 */
template <unsigned FirstShellAngular, unsigned SecondShellAngular, unsigned ThirdShellAngular,
          unsigned FourthShellAngular, typename Scalar>
__device__ inline Scalar primitive_eri_order3(
    double alpha, const Vec3<Scalar>& first, const Angular& angular_first, double beta,
    const Vec3<Scalar>& second, const Angular& angular_second, double gamma,
    const Vec3<Scalar>& third, const Angular& angular_third, double delta,
    const Vec3<Scalar>& fourth, const Angular& angular_fourth) {
  constexpr unsigned FirstPairOrder = FirstShellAngular + SecondShellAngular;
  constexpr unsigned SecondPairOrder = ThirdShellAngular + FourthShellAngular;
  static_assert(FirstPairOrder + SecondPairOrder == 3);
  static_assert(FirstPairOrder <= 3 && SecondPairOrder <= 3);
  constexpr unsigned FirstTermCount = 1U << FirstPairOrder;
  constexpr unsigned SecondTermCount = 1U << SecondPairOrder;

  const double p = alpha + beta;
  const double q = gamma + delta;
  const double mu = alpha * beta / p;
  const double nu = gamma * delta / q;
  const double rho = p * q / (p + q);
  const Vec3<Scalar> product_p = product_center(alpha, first, beta, second);
  const Vec3<Scalar> product_q = product_center(gamma, third, delta, fourth);
  const ThirdOrderPairExpansion<FirstPairOrder, Scalar> first_expansion =
      make_third_order_pair_expansion<FirstShellAngular, SecondShellAngular>(
          p, product_p, first, angular_first, second, angular_second);
  const ThirdOrderPairExpansion<SecondPairOrder, Scalar> second_expansion =
      make_third_order_pair_expansion<ThirdShellAngular, FourthShellAngular>(
          q, product_q, third, angular_third, fourth, angular_fourth);
  Scalar boys[4];
  boys_values<3>(rho * distance_squared(product_p, product_q), boys);
  const Vec3<Scalar> product_difference{
      product_p.x - product_q.x,
      product_p.y - product_q.y,
      product_p.z - product_q.z,
  };

  Scalar value = scalar<Scalar>(0.0);
  for (unsigned first_term = 0; first_term < FirstTermCount; ++first_term) {
    for (unsigned second_term = 0; second_term < SecondTermCount; ++second_term) {
      const LowOrderHermiteTerm<Scalar>& first_item = first_expansion.terms[first_term];
      const LowOrderHermiteTerm<Scalar>& second_item = second_expansion.terms[second_term];
      const double sign =
          (low_order_derivative_total(second_item.derivative_state) & 1U) == 0 ? 1.0 : -1.0;
      value = value +
              sign * first_item.coefficient * second_item.coefficient *
                  third_order_coulomb(first_item.derivative_state + second_item.derivative_state,
                                      rho, product_difference, boys);
    }
  }

  const Scalar pair_decay =
      qexp(-mu * distance_squared(first, second) - nu * distance_squared(third, fourth));
  return 2.0 * pow(kPi, 2.5) / (p * q * sqrt(p + q)) * pair_decay * value;
}

}  // namespace vibeqc::scf::cuda_execution
