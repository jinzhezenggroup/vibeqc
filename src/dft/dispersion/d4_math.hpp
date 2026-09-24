
#pragma once

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "dft/dispersion/d4_types.hpp"

#if defined(__CUDACC__)
#define VIBEQC_D4_MATH_HD __host__ __device__
#else
#define VIBEQC_D4_MATH_HD
#endif

namespace vibeqc::dft::dispersion::math {

inline constexpr int kMaximumReferences = 7;
inline constexpr double kReferenceWeightFactor = 6.0;
inline constexpr double kMinimumWeightNorm = 1.4916681462400413e-154;
inline constexpr double kCoordinationSteepness = 7.5;
inline constexpr double kElectronegativityScale = 4.10451;
inline constexpr double kElectronegativityShift = 19.08857;
inline constexpr double kElectronegativityWidth = 11.28174;
inline constexpr double kInverseSqrtPi = 0.5641895835477562869480794515607726;

struct ChargeScale {
  double value;
  double derivative;
};

template <bool NeedDerivative>
VIBEQC_D4_MATH_HD inline ChargeScale charge_scale_impl(double a, double c, double qref,
                                                       double qmod) {
  ChargeScale result{exp(a), 0.0};
  if (qmod <= 0.0) return result;
  const double inner = exp(c * (1.0 - qref / qmod));
  result.value = exp(a * (1.0 - inner));
  if constexpr (NeedDerivative) {
    if (inner != 0.0 && result.value != 0.0)
      result.derivative = -a * c * inner * result.value * (qref / qmod) / qmod;
  }
  return result;
}

VIBEQC_D4_MATH_HD inline ChargeScale charge_scale(double a, double c, double qref, double qmod) {
  return charge_scale_impl<true>(a, c, qref, qmod);
}

VIBEQC_D4_MATH_HD inline double charge_scale_value(double a, double c, double qref, double qmod) {
  return charge_scale_impl<false>(a, c, qref, qmod).value;
}

struct CoordinationPair {
  double value;
  double derivative;
};

struct CoordinationParameters {
  double radius;
  double electronegativity_factor;
};

template <class FirstElement, class SecondElement>
VIBEQC_D4_MATH_HD inline CoordinationParameters coordination_parameters(
    const FirstElement& first, const SecondElement& second) {
  const double radius = first.covalent_radius + second.covalent_radius;
  const double den =
      fabs(first.electronegativity - second.electronegativity) + kElectronegativityShift;
  const double en = kElectronegativityScale *
                    exp(-(den * den) / (2.0 * kElectronegativityWidth * kElectronegativityWidth));
  return {radius, en};
}

VIBEQC_D4_MATH_HD inline CoordinationPair coordination_pair(CoordinationParameters parameters,
                                                            double distance) {
  const double x = kCoordinationSteepness * (distance - parameters.radius) / parameters.radius;
  return {0.5 * parameters.electronegativity_factor * (1.0 + erf(-x)),
          -parameters.electronegativity_factor * kCoordinationSteepness * exp(-x * x) *
              kInverseSqrtPi / parameters.radius};
}

template <class FirstElement, class SecondElement>
VIBEQC_D4_MATH_HD inline CoordinationPair coordination_pair(const FirstElement& first,
                                                            const SecondElement& second,
                                                            double distance) {
  return coordination_pair(coordination_parameters(first, second), distance);
}

template <class FirstElement, class SecondElement>
VIBEQC_D4_MATH_HD inline double pair_rr(const FirstElement& first, const SecondElement& second) {
  return 3.0 * first.r4r2 * second.r4r2;
}

template <class FirstElement, class SecondElement>
VIBEQC_D4_MATH_HD inline double damping_radius(const FirstElement& first,
                                               const SecondElement& second, double a1, double a2) {
  return a1 * sqrt(pair_rr(first, second)) + a2;
}

struct PairDamping {
  double value;
  double derivative;
};

VIBEQC_D4_MATH_HD inline PairDamping pair_damping(double distance_squared, double rr, double r0,
                                                  double s6, double s8) {
  const double r2_squared = distance_squared * distance_squared;
  const double r2_cubed = r2_squared * distance_squared;
  const double r0_squared = r0 * r0;
  const double r0_fourth = r0_squared * r0_squared;
  const double r0_sixth = r0_fourth * r0_squared;
  const double t6 = 1.0 / (r2_cubed + r0_sixth);
  const double t8 = 1.0 / (r2_squared * r2_squared + r0_fourth * r0_fourth);
  return {s6 * t6 + s8 * rr * t8,
          -6.0 * s6 * r2_squared * t6 * t6 - 8.0 * s8 * rr * r2_cubed * t8 * t8};
}

template <class FirstElement, class SecondElement>
VIBEQC_D4_MATH_HD inline PairDamping pair_damping(const FirstElement& first,
                                                  const SecondElement& second,
                                                  double distance_squared, double s6, double s8,
                                                  double a1, double a2) {
  const double rr = pair_rr(first, second);
  return pair_damping(distance_squared, rr, damping_radius(first, second, a1, a2), s6, s8);
}

template <class ElementData, class ReferenceData>
VIBEQC_D4_MATH_HD inline void atom_weights(const ElementData& element,
                                           const ReferenceData* references, double coordination,
                                           double charge, bool zero_charge, double ga, double gc,
                                           double* weights, double* cn_derivatives,
                                           double* charge_derivatives) {
  for (int local = 0; local < kMaximumReferences; ++local) {
    weights[local] = 0.0;
    if (cn_derivatives != nullptr) cn_derivatives[local] = 0.0;
    if (charge_derivatives != nullptr) charge_derivatives[local] = 0.0;
  }

  double normalization = 0.0;
  double normalization_derivative = 0.0;
  double maximum_reference_cn = -DBL_MAX;
  for (int local = 0; local < element.reference_count; ++local) {
    const auto reference = references[element.reference_offset + local];
    maximum_reference_cn = fmax(maximum_reference_cn, reference.coordination_number);
    const double delta = coordination - reference.coordination_number;
    for (int gaussian = 1; gaussian <= reference.gaussian_count; ++gaussian) {
      const double factor = static_cast<double>(gaussian) * kReferenceWeightFactor;
      const double value = exp(-factor * delta * delta);
      normalization += value;
      if (cn_derivatives != nullptr) normalization_derivative -= 2.0 * factor * delta * value;
    }
  }

  const double inverse_normalization =
      normalization > kMinimumWeightNorm ? 1.0 / normalization : 0.0;
  const double qmod = (zero_charge ? 0.0 : charge) + element.effective_charge;
  for (int local = 0; local < element.reference_count; ++local) {
    const auto reference = references[element.reference_offset + local];
    const double delta = coordination - reference.coordination_number;
    double numerator = 0.0;
    double numerator_derivative = 0.0;
    for (int gaussian = 1; gaussian <= reference.gaussian_count; ++gaussian) {
      const double factor = static_cast<double>(gaussian) * kReferenceWeightFactor;
      const double value = exp(-factor * delta * delta);
      numerator += value;
      if (cn_derivatives != nullptr) numerator_derivative -= 2.0 * factor * delta * value;
    }
    const double cn_weight =
        inverse_normalization != 0.0
            ? numerator * inverse_normalization
            : (fabs(maximum_reference_cn - reference.coordination_number) < 1.0e-12 ? 1.0 : 0.0);
    const double cn_derivative =
        cn_derivatives != nullptr && inverse_normalization != 0.0
            ? inverse_normalization * (numerator_derivative -
                                       numerator * normalization_derivative * inverse_normalization)
            : 0.0;
    const double charge_steepness = gc * element.hardness;
    const double reference_charge = reference.charge + element.effective_charge;
    if (charge_derivatives != nullptr) {
      const auto scale = charge_scale(ga, charge_steepness, reference_charge, qmod);
      weights[local] = cn_weight * scale.value;
      if (cn_derivatives != nullptr) cn_derivatives[local] = cn_derivative * scale.value;
      charge_derivatives[local] = zero_charge ? 0.0 : cn_weight * scale.derivative;
    } else {
      const double scale = charge_scale_value(ga, charge_steepness, reference_charge, qmod);
      weights[local] = cn_weight * scale;
      if (cn_derivatives != nullptr) cn_derivatives[local] = cn_derivative * scale;
    }
  }
}

struct Coefficient {
  double c6;
  double first_cn;
  double second_cn;
  double first_charge;
  double second_charge;
};

struct PackedReferenceC6 {
  const double* values;
  VIBEQC_D4_MATH_HD inline double operator()(int first, int second) const {
    const int high = first > second ? first : second;
    const int low = first > second ? second : first;
    return values[high * (high + 1) / 2 + low];
  }
};

struct DenseReferenceC6 {
  const double* values;
  std::size_t stride;
  VIBEQC_D4_MATH_HD inline double operator()(int first, int second) const {
    return values[static_cast<std::size_t>(first) * stride + static_cast<std::size_t>(second)];
  }
};

template <bool NeedC6, bool NeedCn, bool NeedCharge, class FirstElement, class SecondElement,
          class ReferenceC6>
VIBEQC_D4_MATH_HD inline Coefficient coefficient_selected(
    const FirstElement& first_element, const SecondElement& second_element,
    ReferenceC6 reference_c6, const double* first_weights, const double* first_cn_derivatives,
    const double* first_charge_derivatives, const double* second_weights,
    const double* second_cn_derivatives, const double* second_charge_derivatives) {
  Coefficient result{};
  for (int first_ref = 0; first_ref < first_element.reference_count; ++first_ref) {
    const int global_first = first_element.reference_offset + first_ref;
    const double first_value = first_weights[first_ref];
    const double first_cn =
        NeedCn && first_cn_derivatives != nullptr ? first_cn_derivatives[first_ref] : 0.0;
    const double first_charge = NeedCharge && first_charge_derivatives != nullptr
                                    ? first_charge_derivatives[first_ref]
                                    : 0.0;
    for (int second_ref = 0; second_ref < second_element.reference_count; ++second_ref) {
      const int global_second = second_element.reference_offset + second_ref;
      const double reference = reference_c6(global_first, global_second);
      const double second_value = second_weights[second_ref];
      if constexpr (NeedC6) result.c6 += first_value * second_value * reference;
      if constexpr (NeedCn) {
        const double second_cn =
            second_cn_derivatives == nullptr ? 0.0 : second_cn_derivatives[second_ref];
        result.first_cn += first_cn * second_value * reference;
        result.second_cn += first_value * second_cn * reference;
      }
      if constexpr (NeedCharge) {
        const double second_charge =
            second_charge_derivatives == nullptr ? 0.0 : second_charge_derivatives[second_ref];
        result.first_charge += first_charge * second_value * reference;
        result.second_charge += first_value * second_charge * reference;
      }
    }
  }
  return result;
}

template <class FirstElement, class SecondElement, class ReferenceC6>
VIBEQC_D4_MATH_HD inline Coefficient coefficient(
    const FirstElement& first_element, const SecondElement& second_element,
    ReferenceC6 reference_c6, const double* first_weights, const double* first_cn_derivatives,
    const double* first_charge_derivatives, const double* second_weights,
    const double* second_cn_derivatives, const double* second_charge_derivatives) {
  return coefficient_selected<true, true, true>(
      first_element, second_element, reference_c6, first_weights, first_cn_derivatives,
      first_charge_derivatives, second_weights, second_cn_derivatives, second_charge_derivatives);
}

VIBEQC_D4_MATH_HD inline double atm_radial(double target, double other_first, double other_second,
                                           double r5_product, double damping, double angle,
                                           double damping_derivative, double c9) {
  const double angle_derivative =
      -0.375 *
      (target * target * target + target * target * (other_first + other_second) +
       target * (3.0 * other_first * other_first + 2.0 * other_first * other_second +
                 3.0 * other_second * other_second) -
       5.0 * (other_first - other_second) * (other_first - other_second) *
           (other_first + other_second)) /
      r5_product;
  return c9 * (-angle_derivative * damping + angle * damping_derivative) / target;
}

}  // namespace vibeqc::dft::dispersion::math

#undef VIBEQC_D4_MATH_HD
