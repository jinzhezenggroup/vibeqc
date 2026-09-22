#pragma once

#include <cstddef>
#include <cstdint>

#include "dft/dispersion/d3_atm.hpp"
#include "dft/dispersion/d3_zero.hpp"

#if defined(__CUDACC__)
#define VIBEQC_D3_MODEL_HD __host__ __device__
#else
#define VIBEQC_D3_MODEL_HD
#endif

namespace vibeqc::dft::dispersion {

enum class D3Damping : std::int32_t { bj = 1, zero = 2 };

struct D3ModelParameters {
  D3Damping damping{D3Damping::bj};
  D3Parameters bj{};
  D3ZeroParameters zero{};
  D3ATMParameters atm{0.0, 0.0, 0.0, 0.0};
  bool atm_enabled{false};
};

VIBEQC_D3_MODEL_HD inline bool valid_d3_model(const D3ModelParameters& model) {
  if (model.damping == D3Damping::bj) {
    if (!d3_detail::valid_parameters(model.bj)) return false;
  } else if (model.damping == D3Damping::zero) {
    if (!d3_zero_detail::valid_parameters(model.zero)) return false;
  } else {
    return false;
  }
  if (!model.atm_enabled) return model.atm.s9 == 0.0;
  return model.damping == D3Damping::bj && model.atm.s9 > 0.0 &&
         d3_atm_detail::valid_parameters(model.atm);
}

struct D3PairTerm {
  bool included{false};
  double damping{};
  // d(phi * switch) / dr divided by r; multiply by -C6 and (ri-rj)
  // for the direct Cartesian pair-gradient contribution.
  double radial_derivative_over_distance{};
};

VIBEQC_D3_MODEL_HD inline bool d3_pair_term(std::size_t first, std::size_t second,
                                            const std::int32_t* z, double r2,
                                            const D3ModelParameters& model, D3Tables tables,
                                            D3PairTerm& out) {
  using namespace d3_detail;
  out = {};
  const double r = sqrt(r2);
  const double rr = 3.0 * tables.elements[z[first] - 1].r4r2 * tables.elements[z[second] - 1].r4r2;
  if (!(rr > 0.0) || !finite(rr)) return false;

  double phi = 0.0;
  double derivative_over_distance = 0.0;
  double pair_cutoff = 0.0;
  double pair_switch_width = 0.0;
  if (model.damping == D3Damping::bj) {
    const auto& p = model.bj;
    pair_cutoff = p.pair_cutoff;
    pair_switch_width = p.pair_switch_width;
    if (pair_cutoff > 0.0 && r2 > pair_cutoff * pair_cutoff) return true;
    const double rd = p.a1 * sqrt(rr) + p.a2;
    if (!(rd > 0.0) || !finite(rd)) return false;
    const double r4 = r2 * r2, r6 = r4 * r2, r8 = r4 * r4;
    const double rd2 = rd * rd, rd4 = rd2 * rd2, rd6 = rd4 * rd2, rd8 = rd4 * rd4;
    const double t6 = 1.0 / (r6 + rd6), t8 = 1.0 / (r8 + rd8);
    phi = p.s6 * t6 + p.s8 * rr * t8;
    derivative_over_distance = p.s6 * (-6.0 * r4 * t6 * t6) + p.s8 * rr * (-8.0 * r6 * t8 * t8);
  } else if (model.damping == D3Damping::zero) {
    const auto& p = model.zero;
    pair_cutoff = p.pair_cutoff;
    pair_switch_width = p.pair_switch_width;
    if (pair_cutoff > 0.0 && r2 > pair_cutoff * pair_cutoff) return true;
    const double r0 = tables.pairs[pair_index(z[first], z[second])].vdw_radius;
    if (!(r0 > 0.0) || !finite(r0)) return false;
    const auto term6 = d3_zero_detail::damped_inverse_power(r, r2, p.rs6 * r0, p.alpha6, 6);
    const auto term8 = d3_zero_detail::damped_inverse_power(r, r2, p.rs8 * r0, p.alpha6 + 2.0, 8);
    if (!finite(term6.value) || !finite(term8.value) || !finite(term6.derivative_over_distance) ||
        !finite(term8.derivative_over_distance))
      return false;
    phi = p.s6 * term6.value + p.s8 * rr * term8.value;
    derivative_over_distance =
        p.s6 * term6.derivative_over_distance + p.s8 * rr * term8.derivative_over_distance;
  } else {
    return false;
  }

  double cutoff_derivative = 0.0;
  const double cutoff = smooth_cutoff(r, pair_cutoff, pair_switch_width, cutoff_derivative);
  out.included = cutoff != 0.0;
  out.damping = cutoff * phi;
  out.radial_derivative_over_distance =
      cutoff * derivative_over_distance + cutoff_derivative * phi / r;
  return finite(out.damping) && finite(out.radial_derivative_over_distance);
}

VIBEQC_D3_MODEL_HD inline D3Status evaluate_d3_model(std::size_t n, const std::int32_t* z,
                                                     const double* xyz,
                                                     const D3ModelParameters& model,
                                                     D3Tables tables, double* workspace,
                                                     std::size_t workspace_elements, double* energy,
                                                     double* gradient) {
  if (!valid_d3_model(model)) return D3Status::invalid_argument;
  D3Status status = D3Status::unsupported;
  if (model.damping == D3Damping::bj) {
    status = evaluate_d3_bj(n, z, xyz, model.bj, tables, workspace, workspace_elements, energy,
                            gradient);
  } else if (model.damping == D3Damping::zero) {
    status = evaluate_d3_zero(n, z, xyz, model.zero, tables, workspace, workspace_elements, energy,
                              gradient);
  }
  if (status != D3Status::success || !model.atm_enabled) return status;
  return evaluate_d3_bj_atm(n, z, xyz, model.atm, tables, workspace, workspace_elements, energy,
                            gradient, true);
}

VIBEQC_D3_MODEL_HD inline const char* d3_variant_identity(const D3ModelParameters& model) {
  if (model.damping == D3Damping::bj) return model.atm_enabled ? "d3.bj-atm" : "d3.bj-two-body";
  if (model.damping == D3Damping::zero && !model.atm_enabled) return "d3.zero-two-body";
  return "d3.unsupported";
}

}  // namespace vibeqc::dft::dispersion

#undef VIBEQC_D3_MODEL_HD
