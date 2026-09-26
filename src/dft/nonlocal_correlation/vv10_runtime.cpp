#include "dft/nonlocal_correlation/vv10_runtime.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <new>
#include <numbers>
#include <stdexcept>

namespace vibeqc::dft::nlc {
namespace {

constexpr double kFourPiOverThree = 4.0 * std::numbers::pi_v<double> / 3.0;

bool finite_positive(double value) { return std::isfinite(value) && value > 0.0; }

std::uint64_t checked_array_bytes(std::uint32_t points, std::uint64_t arrays) {
  constexpr std::uint64_t bytes = sizeof(double);
  const auto n = static_cast<std::uint64_t>(points);
  if (n > std::numeric_limits<std::uint64_t>::max() / arrays / bytes)
    throw std::overflow_error("VV10 workspace extent overflow");
  return arrays * bytes * n;
}

Vv10ResourceUsage resource_usage(vibeqc_backend backend, std::uint32_t point_count,
                                 std::uint32_t tile_points, std::uint64_t maximum_bytes) {
  const auto n = static_cast<std::uint64_t>(point_count);
  const auto tile = static_cast<std::uint64_t>(std::min(tile_points, point_count));
  const auto host = checked_array_bytes(point_count, backend == VIBEQC_BACKEND_CUDA ? 7u : 12u);
  std::uint64_t device = 0;
  if (backend == VIBEQC_BACKEND_CUDA) {
    device = checked_array_bytes(point_count, 21u);
    if (device > std::numeric_limits<std::uint64_t>::max() - sizeof(double))
      throw std::overflow_error("VV10 CUDA failure-flag extent overflow");
    device += sizeof(double);
  }
  if (host > std::numeric_limits<std::uint64_t>::max() - device)
    throw std::overflow_error("VV10 total workspace extent overflow");
  const auto workspace = host + device;
  return Vv10ResourceUsage{workspace,   host,
                           device,      maximum_bytes,
                           n * n,       (n + tile - 1u) / tile,
                           point_count, static_cast<std::uint32_t>(tile)};
}

struct PairValues {
  double phi{};
  double dphi_dr2{};
  double dphi_domega_i{};
  double dphi_dkappa_i{};
};

PairValues pair_values(double r2, double omega_i, double omega_j, double kappa_i, double kappa_j,
                       Vv10Variant variant) {
  PairValues result;
  if (variant == Vv10Variant::rvv10) {
    const auto ai = omega_i / kappa_i;
    const auto aj = omega_j / kappa_j;
    const auto zi = 1.0 + ai * r2;
    const auto zj = 1.0 + aj * r2;
    const auto kappa_product = kappa_i * kappa_j;
    const auto denominator = std::pow(kappa_product, 1.5) * zi * zj * (zi + zj);
    result.phi = -1.5 / denominator;
    const auto factor_z = 1.0 / zi + 1.0 / (zi + zj);
    result.dphi_domega_i = -result.phi * r2 / kappa_i * factor_z;
    result.dphi_dkappa_i = result.phi / kappa_i * (-1.5 + (zi - 1.0) * factor_z);
    const auto logarithmic = ai / zi + aj / zj + (ai + aj) / (zi + zj);
    result.dphi_dr2 = -result.phi * logarithmic;
  } else {
    const auto gi = omega_i * r2 + kappa_i;
    const auto gj = omega_j * r2 + kappa_j;
    result.phi = -1.5 / (gi * gj * (gi + gj));
    const auto dphi_dgi = -result.phi * (1.0 / gi + 1.0 / (gi + gj));
    result.dphi_domega_i = dphi_dgi * r2;
    result.dphi_dkappa_i = dphi_dgi;
    const auto logarithmic = omega_i / gi + omega_j / gj + (omega_i + omega_j) / (gi + gj);
    result.dphi_dr2 = -result.phi * logarithmic;
  }
  return result;
}

bool finite_pair(const PairValues& value) {
  return std::isfinite(value.phi) && std::isfinite(value.dphi_dr2) &&
         std::isfinite(value.dphi_domega_i) && std::isfinite(value.dphi_dkappa_i);
}

}  // namespace

std::unique_ptr<Vv10Plan> Vv10Plan::prepare(vibeqc_backend backend, int device_id,
                                            std::uint32_t point_count, std::uint32_t tile_points,
                                            Vv10Parameters parameters, std::uint64_t maximum_bytes,
                                            std::string& detail, vibeqc_status& status) {
  status = VIBEQC_STATUS_INVALID_ARGUMENT;
  detail.clear();
  if (backend != VIBEQC_BACKEND_CPU_REFERENCE && backend != VIBEQC_BACKEND_CUDA) {
    detail = "VV10 production pair execution requires CPU_REFERENCE or CUDA";
    status = VIBEQC_STATUS_NOT_IMPLEMENTED;
    return nullptr;
  }
#if !VIBEQC_HAS_CUDA
  if (backend == VIBEQC_BACKEND_CUDA) {
    detail = "VV10 CUDA lowerer is unavailable in this build";
    status = VIBEQC_STATUS_NOT_IMPLEMENTED;
    return nullptr;
  }
#endif
  if (backend == VIBEQC_BACKEND_CUDA && device_id < 0) {
    detail = "VV10 CUDA execution requires a nonnegative device id";
    return nullptr;
  }
  if (!point_count || !tile_points) {
    detail = "VV10 plan requires nonzero point_count and tile_points";
    return nullptr;
  }
  if (point_count > std::numeric_limits<std::uint32_t>::max() / 3u) {
    detail = "VV10 point_count exceeds the public flattened-coordinate count domain";
    return nullptr;
  }
  if (parameters.variant != Vv10Variant::vv10 && parameters.variant != Vv10Variant::rvv10) {
    detail = "VV10 plan received an unsupported kernel variant";
    return nullptr;
  }
  if (!finite_positive(parameters.b) || !finite_positive(parameters.c) ||
      !finite_positive(parameters.coefficient)) {
    detail = "VV10 parameters b, C and coefficient must be finite and positive";
    return nullptr;
  }
  if (!maximum_bytes) {
    detail = "VV10 plan requires a positive maximum_bytes budget";
    return nullptr;
  }

  try {
    const auto resources = resource_usage(backend, point_count, tile_points, maximum_bytes);
    if (resources.workspace_bytes > maximum_bytes) {
      detail = "VV10 provider workspace exceeds maximum_bytes before execution";
      status = VIBEQC_STATUS_OUT_OF_MEMORY;
      return nullptr;
    }
    auto plan = std::unique_ptr<Vv10Plan>(new Vv10Plan(backend, device_id, parameters, resources));
    if (backend == VIBEQC_BACKEND_CPU_REFERENCE) {
      plan->omega_.resize(point_count);
      plan->kappa_.resize(point_count);
      plan->weighted_density_.resize(point_count);
      plan->domega_drho_.resize(point_count);
      plan->domega_dsigma_.resize(point_count);
      plan->dkappa_drho_.resize(point_count);
    }
    status = VIBEQC_STATUS_SUCCESS;
    return plan;
  } catch (const std::bad_alloc&) {
    detail = "VV10 provider workspace allocation failed";
    status = VIBEQC_STATUS_OUT_OF_MEMORY;
    return nullptr;
  } catch (const std::overflow_error& error) {
    detail = error.what();
    status = VIBEQC_STATUS_OUT_OF_MEMORY;
    return nullptr;
  }
}

vibeqc_status Vv10Plan::execute(std::span<const double> coordinates,
                                std::span<const double> weights, std::span<const double> density,
                                std::span<const double> density_gradient, double& energy,
                                std::span<double> vrho, std::span<double> vsigma,
                                std::span<double> point_derivative,
                                std::span<double> weight_derivative, std::string& detail) {
  detail.clear();
  const auto n = static_cast<std::size_t>(resources_.point_count);
  if (coordinates.size() != 3 * n || weights.size() != n || density.size() != n ||
      density_gradient.size() != 3 * n) {
    detail = "VV10 fixed-grid input shape does not match the prepared point count";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const bool want_features = !vrho.empty() || !vsigma.empty();
  if (want_features && (vrho.size() != n || vsigma.size() != n)) {
    detail = "VV10 feature derivatives require matching vrho/vsigma output arrays";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const bool want_geometry = !point_derivative.empty() || !weight_derivative.empty();
  if (want_geometry && (point_derivative.size() != 3 * n || weight_derivative.size() != n)) {
    detail = "VV10 geometry derivatives require matching point/weight output arrays";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  for (std::size_t i = 0; i < n; ++i) {
    if (!finite_positive(density[i]) || !std::isfinite(weights[i]) ||
        !std::isfinite(density_gradient[3 * i]) || !std::isfinite(density_gradient[3 * i + 1]) ||
        !std::isfinite(density_gradient[3 * i + 2]) || !std::isfinite(coordinates[3 * i]) ||
        !std::isfinite(coordinates[3 * i + 1]) || !std::isfinite(coordinates[3 * i + 2])) {
      detail = "VV10 fixed-grid inputs must be finite with strictly positive density";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
  }

#if VIBEQC_HAS_CUDA
  if (backend_ == VIBEQC_BACKEND_CUDA) {
    try {
      execute_vv10_cuda(coordinates.data(), weights.data(), density.data(), density_gradient.data(),
                        n, resources_.tile_points, parameters_, device_id_, energy, vrho.data(),
                        vsigma.data(), want_geometry ? point_derivative.data() : nullptr,
                        want_geometry ? weight_derivative.data() : nullptr);
      return VIBEQC_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) {
      detail = "VV10 CUDA workspace allocation failed";
      return VIBEQC_STATUS_OUT_OF_MEMORY;
    } catch (const std::overflow_error& error) {
      detail = error.what();
      return VIBEQC_STATUS_NUMERICAL_FAILURE;
    } catch (const std::exception& error) {
      detail = error.what();
      return VIBEQC_STATUS_CUDA_ERROR;
    }
  }
#endif

  const auto b = parameters_.b;
  const auto c = parameters_.c;
  const auto coefficient = parameters_.coefficient;
  const auto beta = std::pow(3.0 / (b * b), 0.75) / 32.0;
  if (!std::isfinite(beta)) {
    detail = "VV10 beta is nonfinite";
    return VIBEQC_STATUS_NUMERICAL_FAILURE;
  }

  for (std::size_t i = 0; i < n; ++i) {
    const auto rho = density[i];
    const auto x = density_gradient[3 * i];
    const auto y = density_gradient[3 * i + 1];
    const auto z = density_gradient[3 * i + 2];
    const auto sigma = x * x + y * y + z * z;
    const auto rho2 = rho * rho;
    const auto rho4 = rho2 * rho2;
    const auto sigma2 = sigma * sigma;
    const auto omega2 = c * sigma2 / rho4 + kFourPiOverThree * rho;
    const auto omega = std::sqrt(omega2);
    const auto kappa = b * 1.5 * std::numbers::pi_v<double> *
                       std::pow(rho / (9.0 * std::numbers::pi_v<double>), 1.0 / 6.0);
    double domega_drho = 0.0, domega_dsigma = 0.0, dkappa_drho = 0.0;
    if (want_features) {
      const auto rho5 = rho4 * rho;
      domega_drho = (kFourPiOverThree - 4.0 * c * sigma2 / rho5) / (2.0 * omega);
      domega_dsigma = c * sigma / (omega * rho4);
      dkappa_drho = kappa / (6.0 * rho);
    }
    if (!finite_positive(omega) || !finite_positive(kappa) || !std::isfinite(domega_drho) ||
        !std::isfinite(domega_dsigma) || !std::isfinite(dkappa_drho)) {
      detail = "VV10 local scales are nonfinite";
      return VIBEQC_STATUS_NUMERICAL_FAILURE;
    }
    omega_[i] = omega;
    kappa_[i] = kappa;
    weighted_density_[i] = weights[i] * rho;
    domega_drho_[i] = domega_drho;
    domega_dsigma_[i] = domega_dsigma;
    dkappa_drho_[i] = dkappa_drho;
  }

  double total_energy = 0.0;
  const auto tile = static_cast<std::size_t>(resources_.tile_points);
  for (std::size_t i = 0; i < n; ++i) {
    double sum_phi = 0.0;
    double sum_rho = 0.0;
    double sum_sigma = 0.0;
    std::array<double, 3> coordinate_sum{};
    for (std::size_t begin = 0; begin < n; begin += tile) {
      const auto end = std::min(begin + tile, n);
      for (std::size_t j = begin; j < end; ++j) {
        const auto dx = coordinates[3 * i] - coordinates[3 * j];
        const auto dy = coordinates[3 * i + 1] - coordinates[3 * j + 1];
        const auto dz = coordinates[3 * i + 2] - coordinates[3 * j + 2];
        const auto r2 = dx * dx + dy * dy + dz * dz;
        const auto pair =
            pair_values(r2, omega_[i], omega_[j], kappa_[i], kappa_[j], parameters_.variant);
        if (!finite_pair(pair)) {
          detail = "VV10 pair kernel produced a nonfinite value";
          return VIBEQC_STATUS_NUMERICAL_FAILURE;
        }
        const auto partner = weighted_density_[j];
        sum_phi += partner * pair.phi;
        if (want_features) {
          const auto dphi_drho =
              pair.dphi_domega_i * domega_drho_[i] + pair.dphi_dkappa_i * dkappa_drho_[i];
          const auto dphi_dsigma = pair.dphi_domega_i * domega_dsigma_[i];
          sum_rho += partner * dphi_drho;
          sum_sigma += partner * dphi_dsigma;
        }
        if (want_geometry) {
          const auto factor = 2.0 * partner * pair.dphi_dr2;
          coordinate_sum[0] += factor * dx;
          coordinate_sum[1] += factor * dy;
          coordinate_sum[2] += factor * dz;
        }
      }
    }
    total_energy += weighted_density_[i] * (beta + 0.5 * sum_phi);
    if (want_features) {
      vrho[i] = coefficient * (beta + sum_phi + density[i] * sum_rho);
      vsigma[i] = coefficient * density[i] * sum_sigma;
    }
    if (want_geometry) {
      const auto prefactor = coefficient * weighted_density_[i];
      point_derivative[3 * i] = prefactor * coordinate_sum[0];
      point_derivative[3 * i + 1] = prefactor * coordinate_sum[1];
      point_derivative[3 * i + 2] = prefactor * coordinate_sum[2];
      weight_derivative[i] = coefficient * density[i] * (beta + sum_phi);
    }
  }
  energy = coefficient * total_energy;
  if (!std::isfinite(energy) ||
      (want_features &&
       (!std::all_of(vrho.begin(), vrho.end(), [](double x) { return std::isfinite(x); }) ||
        !std::all_of(vsigma.begin(), vsigma.end(), [](double x) { return std::isfinite(x); }))) ||
      (want_geometry && (!std::all_of(point_derivative.begin(), point_derivative.end(),
                                      [](double x) { return std::isfinite(x); }) ||
                         !std::all_of(weight_derivative.begin(), weight_derivative.end(),
                                      [](double x) { return std::isfinite(x); })))) {
    detail = "VV10 execution produced nonfinite output";
    return VIBEQC_STATUS_NUMERICAL_FAILURE;
  }
  return VIBEQC_STATUS_SUCCESS;
}

}  // namespace vibeqc::dft::nlc
