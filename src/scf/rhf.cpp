#include "scf/rhf.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "posthf/capacity.hpp"
#include "posthf/raw_source.hpp"
#include "runtime/resource_ledger.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/cuda/rhf_policy.hpp"
#include "scf/cuda_density_fitting.hpp"
#include "scf/cuda_density_fitting_integrals.hpp"
#include "scf/cuda_df_gradient.hpp"
#include "scf/cuda_fock_provider.hpp"
#include "scf/cuda_one_electron_gradient.hpp"
#include "scf/density_fitting.hpp"
#include "scf/fock_build.hpp"
#include "scf/fock_prepared.hpp"
#include "scf/fock_provider.hpp"
#include "scf/initial_guess/density.hpp"
#include "scf/reference/mean_field.hpp"
#include "scf/solver/diis.hpp"
#include "scf/solver/mean_field_driver.hpp"
#include "scf/solver/proposal_control.hpp"

namespace vibeqc::scf {
namespace {

using initial_guess::prepare_initial_density;
using initial_guess::prepare_initial_uhf_density;
using initial_guess::spin_occupations;
using reference::commutator_residual;
using reference::concatenate;
using reference::density_from_orbitals;
using reference::density_rms;
using reference::EigenResult;
using reference::electronic_energy;
using reference::energy_weighted_density;
using reference::generalized_eigen;
using reference::index;
using reference::Matrix;
using reference::multiply;
using reference::residual_rms;
using reference::split_spin_matrices;
using reference::symmetric_orthogonalizer;
using reference::transpose;
using reference::uhf_electronic_energy;
using solver::Diis;
using solver::run_rhf_host_plan;
using solver::run_uhf_host_plan;
using solver::validate_seed;

/** Assemble immutable DF state from already-evaluated one- and three-center data. */
DensityFittingScfData assemble_density_fitting_data(integrals::IntegralData one_electron,
                                                    integrals::DensityFittingIntegralData raw,
                                                    double relative_threshold) {
  if (!(relative_threshold > 0.0) || !(relative_threshold < 1.0) ||
      !std::isfinite(relative_threshold)) {
    throw std::invalid_argument(
        "DF metric relative threshold must lie strictly between zero and one");
  }
  DensityFittingScfData data;
  data.one_electron = std::move(one_electron);
  data.raw = std::move(raw);
  data.metric_relative_threshold = relative_threshold;
  const DensityFittingMetricFactor factor =
      factor_density_fitting_metric(data.raw.metric, data.raw.naux, relative_threshold);
  data.three_center =
      orthonormalize_density_fitting_three_center(data.raw.three_center, data.raw.nbf, factor);
  if (data.raw.nbf != data.one_electron.nbf) {
    throw std::runtime_error("DF orbital and one-electron AO dimensions are inconsistent");
  }
  return data;
}

/**
 * Release tensor-sized host storage after a source-backed CUDA plan has taken
 * ownership of tile regeneration.  Dimensions and compact metric metadata
 * remain available to finalization, while the source plan supplies all
 * three-center and derivative values on demand under the caller's budget.
 */
[[maybe_unused]] void discard_density_fitting_tensor_storage(DensityFittingScfData& data) {
  std::vector<double>().swap(data.raw.three_center);
  std::vector<double>().swap(data.raw.three_center_derivative);
  std::vector<double>().swap(data.raw.metric_derivative);
  std::vector<double>().swap(data.three_center.values);
}

/** Build a budgeted record without materializing transformed DF tensors. */
[[maybe_unused]] DensityFittingScfData assemble_density_fitting_metadata(
    integrals::IntegralData one_electron, integrals::DensityFittingIntegralData raw,
    double relative_threshold) {
  DensityFittingScfData data;
  data.one_electron = std::move(one_electron);
  data.metric_relative_threshold = relative_threshold;
  data.raw.nbf = raw.nbf;
  data.raw.naux = raw.naux;
  data.raw.ncoord = raw.ncoord;
  data.three_center.nbf = raw.nbf;
  data.three_center.naux = raw.naux;
  return data;
}

/** Bind a per-geometry fused response only when AO derivative tensors were omitted. */
void bind_generated_one_electron(DensityFittingScfData& data, const core::System& system,
                                 int device_id, std::size_t budget) {
#if VIBEQC_HAS_CUDA
  if (device_id >= 0 && data.one_electron.overlap_derivative.empty()) {
    data.one_electron_gradient_system = system;
    data.one_electron_gradient_device = device_id;
    data.one_electron_gradient_mapping = cuda_policy::one_electron_derivative_mapping_requested();
    data.one_electron_gradient_budget = budget ? budget : 128U * 1024U * 1024U;
  }
#else
  (void)data;
  (void)system;
  (void)device_id;
  (void)budget;
#endif
}

/** Reserve half a constrained DF request for the generated force bridge. */
[[maybe_unused]] std::size_t df_response_budget(std::size_t requested) {
  return requested ? requested / 2 : 128U * 1024U * 1024U;
}
void bind_generated_df(DensityFittingScfData& data, const core::System& orbital,
                       const core::System& auxiliary, int device, std::size_t budget) {
#if VIBEQC_HAS_CUDA
  if (device >= 0) {
    data.df_gradient_orbital = orbital;
    data.df_gradient_auxiliary = auxiliary;
    data.df_gradient_mapping = cuda_policy::df_derivative_mapping_requested();
    data.df_gradient_budget = df_response_budget(budget);
  }
#else
  (void)data;
  (void)orbital;
  (void)auxiliary;
  (void)device;
  (void)budget;
#endif
}
/** Energy-only and CPU caches have no bound CUDA response to invalidate. */
[[maybe_unused]] bool df_response_policy_matches(const DensityFittingScfData& data,
                                                 std::size_t budget, double relative_threshold,
                                                 bool needs_cuda_response) {
#if VIBEQC_HAS_CUDA
  const bool generated = needs_cuda_response;
  return data.metric_relative_threshold == relative_threshold &&
         data.df_gradient_orbital.has_value() == generated &&
         (!generated ||
          (data.df_gradient_mapping == cuda_policy::df_derivative_mapping_requested() &&
           data.df_gradient_budget == df_response_budget(budget)));
#else
  (void)data;
  (void)budget;
  (void)relative_threshold;
  (void)needs_cuda_response;
  return true;
#endif
}

/** Cached DF response state must follow policy and budget changes on replay. */
[[maybe_unused]] bool one_electron_response_policy_matches(const DensityFittingScfData& data,
                                                           std::size_t requested_budget,
                                                           bool needs_cuda_response) {
#if VIBEQC_HAS_CUDA
  const bool generated =
      needs_cuda_response && cuda_policy::generated_one_electron_derivatives_requested();
  const auto effective_budget = requested_budget ? requested_budget : 128U * 1024U * 1024U;
  return data.one_electron_gradient_system.has_value() == generated &&
         (!generated || (data.one_electron_gradient_mapping ==
                             cuda_policy::one_electron_derivative_mapping_requested() &&
                         data.one_electron_gradient_budget == effective_budget));
#else
  (void)data;
  (void)requested_budget;
  (void)needs_cuda_response;
  return true;
#endif
}

[[maybe_unused]] DensityFittingScfData prepare_density_fitting_data(
    const core::System& system, const core::System& auxiliary_system, double relative_threshold,
    int cuda_device_id = -1, std::size_t output_budget_bytes = 0U,
    bool include_derivatives = true) {
  // A non-negative device selects the CUDA Cartesian evaluator for the raw
  // metric/three-center tensors.  The default keeps CPU-reference callers
  // entirely on the existing oracle path.
  if (!(relative_threshold > 0.0) || !(relative_threshold < 1.0) ||
      !std::isfinite(relative_threshold)) {
    throw std::invalid_argument(
        "DF metric relative threshold must lie strictly between zero and one");
  }
  DensityFittingScfData data;
#if !VIBEQC_HAS_CUDA
  (void)output_budget_bytes;
#endif
#if VIBEQC_HAS_CUDA
  if (cuda_device_id >= 0) {
    integrals::IntegralData cartesian_one_electron;
    std::string one_electron_detail;
    const vibeqc_status one_electron_status = build_cuda_one_electron_integrals(
        cuda_device_id, system, cartesian_one_electron, one_electron_detail,
        include_derivatives && !cuda_policy::generated_one_electron_derivatives_requested(),
        include_derivatives);
    if (one_electron_status != VIBEQC_STATUS_SUCCESS) {
      throw std::runtime_error(one_electron_detail.empty()
                                   ? "CUDA one-electron integral generation failed"
                                   : one_electron_detail);
    }
    data.one_electron = integrals::transform_integrals(cartesian_one_electron, system);

    // A positive budget uses the source-backed plan, which regenerates all DF
    // values and derivatives from compact device metadata. Do not build the
    // complete raw metric/three-center tensors just to discard them before
    // plan creation; retaining only dimensions and one-electron response data
    // keeps the setup peak bounded by the caller's request.
    if (output_budget_bytes != 0U) {
      integrals::DensityFittingIntegralData metadata;
      metadata.nbf = molecule::ao_count(system);
      metadata.naux = molecule::ao_count(auxiliary_system);
      metadata.ncoord = include_derivatives ? system.atoms.size() * 3U : 0U;
      data = assemble_density_fitting_metadata(std::move(data.one_electron), std::move(metadata),
                                               relative_threshold);
      if (include_derivatives) {
        bind_generated_one_electron(data, system, cuda_device_id, output_budget_bytes);
        bind_generated_df(data, system, auxiliary_system, cuda_device_id, output_budget_bytes);
      }
      return data;
    }

    integrals::DensityFittingIntegralData cartesian;
    std::string detail;
    const vibeqc_status status = build_cuda_density_fitting_integrals(
        cuda_device_id, system, auxiliary_system, cartesian, detail, false);
    if (status != VIBEQC_STATUS_SUCCESS) {
      throw std::runtime_error(detail.empty() ? "CUDA density-fitting integral generation failed"
                                              : detail);
    }
    // Device recurrence operates on normalized Cartesian source AOs.  Apply
    // the same independently tested public spherical transform used by the
    // host oracle after the device values and derivatives are downloaded.
    data.raw = integrals::transform_density_fitting_integrals(cartesian, system, auxiliary_system);
  } else {
    data.one_electron = integrals::build_integrals(system, include_derivatives);
    data.raw =
        integrals::build_density_fitting_integrals(system, auxiliary_system, include_derivatives);
  }
#else
  (void)cuda_device_id;
  data.one_electron = integrals::build_integrals(system, include_derivatives);
  data.raw =
      integrals::build_density_fitting_integrals(system, auxiliary_system, include_derivatives);
#endif
  data = assemble_density_fitting_data(std::move(data.one_electron), std::move(data.raw),
                                       relative_threshold);
  if (include_derivatives) {
    bind_generated_one_electron(data, system, cuda_device_id, output_budget_bytes);
    bind_generated_df(data, system, auxiliary_system, cuda_device_id, output_budget_bytes);
  }
  return data;
}

Matrix build_density_fitting_rhf_fock(const Matrix& hcore,
                                      const DensityFittingThreeCenter& three_center,
                                      const Matrix& density) {
  const DensityFittingRhfJk jk = build_density_fitting_rhf_jk(three_center, density);
  Matrix fock = hcore;
  for (std::size_t element = 0; element < fock.size(); ++element) {
    fock[element] += jk.coulomb[element] - 0.5 * jk.exchange[element];
  }
  return fock;
}

std::pair<Matrix, Matrix> build_density_fitting_uhf_focks(
    const Matrix& hcore, const DensityFittingThreeCenter& three_center, const Matrix& alpha_density,
    const Matrix& beta_density) {
  const DensityFittingUhfJk jk =
      build_density_fitting_uhf_jk(three_center, alpha_density, beta_density);
  Matrix alpha_fock = hcore;
  Matrix beta_fock = hcore;
  for (std::size_t element = 0; element < hcore.size(); ++element) {
    alpha_fock[element] += jk.coulomb[element] - jk.alpha_exchange[element];
    beta_fock[element] += jk.coulomb[element] - jk.beta_exchange[element];
  }
  return {std::move(alpha_fock), std::move(beta_fock)};
}

/** Adapt stationary RHF/UHF weights to the shared external-weight derivative.
 * D and energy-weighted D already contain their proper occupation factors.
 * UHF sums both spin channels; no additional factor of two belongs here.
 */
Matrix generated_one_electron_hf_gradient(const DensityFittingScfData& data, const Matrix& density,
                                          const Matrix& weighted,
                                          const Matrix* beta_density = nullptr,
                                          const Matrix* beta_weighted = nullptr) {
  Matrix gradient;
#if VIBEQC_HAS_CUDA
  if (!data.one_electron_gradient_system) return gradient;
  Matrix total_density, total_weighted;
  std::span<const double> d = density, w = weighted;
  if (beta_density) {
    total_density = density;
    total_weighted = weighted;
    for (std::size_t i = 0; i < density.size(); ++i) {
      total_density[i] += (*beta_density)[i];
      total_weighted[i] += (*beta_weighted)[i];
    }
    d = total_density;
    w = total_weighted;
  }
  std::string detail;
  const auto status = execute_cuda_one_electron_gradient(
      data.one_electron_gradient_device, *data.one_electron_gradient_system, w, d, d,
      data.one_electron_gradient_mapping, data.one_electron_gradient_budget, gradient, detail,
      nullptr, -1.0);
  if (status == VIBEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
  if (status != VIBEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
#else
  (void)data;
  (void)density;
  (void)weighted;
  (void)beta_density;
  (void)beta_weighted;
#endif
  return gradient;
}

/** Form DF weights and contract before entering any legacy fallback guard. */
Matrix generated_df_hf_gradient(const DensityFittingScfData& data, CudaDensityFittingJkPlan* plan,
                                std::size_t system, const Matrix& density,
                                const Matrix* beta = nullptr) {
  Matrix gradient;
#if VIBEQC_HAS_CUDA
  if (!data.df_gradient_orbital) return gradient;
  if (!plan || !data.df_gradient_auxiliary)
    throw std::runtime_error("generated DF response has no matching CUDA plan or geometry");
  const auto spin_staging_bytes = beta ? density.size() * sizeof(double) : 0U;
  if (spin_staging_bytes >= data.df_gradient_budget) throw std::bad_alloc();
  Matrix total;
  std::vector<DensityFittingDensityResponse> terms;
  if (beta) {
    total.resize(density.size());
    for (std::size_t i = 0; i < density.size(); ++i) total[i] = density[i] + (*beta)[i];
    terms = {{total, 1.0, 0.0}, {density, 0.0, 0.5}, {*beta, 0.0, 0.5}};
  } else {
    terms = {{density, 1.0, 0.25}};
  }
  std::string detail;
  const auto status = execute_cuda_density_fitting_generated_force_response(
      plan, system, *data.df_gradient_orbital, *data.df_gradient_auxiliary, data.raw.three_center,
      data.raw.metric, terms, data.df_gradient_mapping,
      data.df_gradient_budget - spin_staging_bytes, 0, gradient, detail);
  if (status == VIBEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
  if (status != VIBEQC_STATUS_SUCCESS || gradient.size() != data.raw.ncoord)
    throw std::runtime_error(detail.empty() ? "generated DF response failed" : detail);
#else
  (void)data;
  (void)plan;
  (void)system;
  (void)density;
  (void)beta;
#endif
  return gradient;
}

[[maybe_unused]] void finalize_density_fitting_rhf(const DensityFittingScfData& data,
                                                   const Matrix& orthogonalizer,
                                                   std::size_t occupied, Matrix& density,
                                                   const ScfOptions& options, ScfResult& result,
                                                   CudaDensityFittingJkPlan* cuda_plan = nullptr,
                                                   std::size_t cuda_system = 0) {
#if !VIBEQC_HAS_CUDA
  (void)cuda_plan;
#endif
  const std::size_t n = data.one_electron.nbf;
  Matrix final_fock;
  const auto execute_item_rhf_jk =
      [&](const Matrix& item_density, std::vector<double>& item_coulomb,
          std::vector<double>& item_exchange, std::string& item_detail) -> vibeqc_status {
    const std::size_t batch =
        cuda_plan == nullptr ? 0U : cuda_density_fitting_jk_plan_batch_size(cuda_plan);
    if (batch <= 1U) {
      return execute_cuda_density_fitting_rhf_jk(cuda_plan, item_density, item_coulomb,
                                                 item_exchange, item_detail);
    }
    if (cuda_system >= batch || item_density.empty() ||
        item_density.size() > std::numeric_limits<std::size_t>::max() / batch) {
      item_detail = "CUDA DF bucket item index or density dimensions are invalid";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    return execute_cuda_density_fitting_rhf_jk_item(cuda_plan, cuda_system, item_density,
                                                    item_coulomb, item_exchange, item_detail);
  };
  if (cuda_plan != nullptr) {
    std::vector<double> coulomb;
    std::vector<double> exchange;
    std::string detail;
    const vibeqc_status jk_status = execute_item_rhf_jk(density, coulomb, exchange, detail);
    if (jk_status == VIBEQC_STATUS_SUCCESS && coulomb.size() == density.size() &&
        exchange.size() == density.size()) {
      final_fock = data.one_electron.hcore;
      for (std::size_t item = 0; item < final_fock.size(); ++item) {
        final_fock[item] += coulomb[item] - 0.5 * exchange[item];
      }
    }
  }
  if (final_fock.empty()) {
    if (data.three_center.values.empty()) {
      throw std::runtime_error(
          "CUDA DF source-backed J/K failed after tensor storage was released");
    }
    final_fock =
        build_density_fitting_rhf_fock(data.one_electron.hcore, data.three_center, density);
  }
  EigenResult orbitals = generalized_eigen(final_fock, orthogonalizer, n);
  density = density_from_orbitals(orbitals.vectors, n, occupied);
  if (cuda_plan != nullptr) {
    std::vector<double> coulomb;
    std::vector<double> exchange;
    std::string detail;
    const vibeqc_status jk_status = execute_item_rhf_jk(density, coulomb, exchange, detail);
    if (jk_status == VIBEQC_STATUS_SUCCESS && coulomb.size() == density.size() &&
        exchange.size() == density.size()) {
      final_fock = data.one_electron.hcore;
      for (std::size_t item = 0; item < final_fock.size(); ++item) {
        final_fock[item] += coulomb[item] - 0.5 * exchange[item];
      }
    } else {
      if (data.three_center.values.empty()) {
        throw std::runtime_error(
            "CUDA DF source-backed J/K failed after tensor storage was released");
      }
      final_fock =
          build_density_fitting_rhf_fock(data.one_electron.hcore, data.three_center, density);
    }
  } else {
    final_fock =
        build_density_fitting_rhf_fock(data.one_electron.hcore, data.three_center, density);
  }
  result.energy = electronic_energy(density, data.one_electron.hcore, final_fock) +
                  data.one_electron.nuclear_repulsion;
  if (!options.compute_forces) {
    result.density = density;
    return;
  }

  const Matrix weighted = energy_weighted_density(orbitals.vectors, orbitals.values, n, occupied);
  // The CUDA response is the sole device path; failures propagate before force
  // assembly. The independent CPU calculation below serves CPU callers only.
  const Matrix generated_one_electron = generated_one_electron_hf_gradient(data, density, weighted);
  const Matrix generated_df = generated_df_hf_gradient(data, cuda_plan, cuda_system, density);
  if (cuda_plan != nullptr) {
    if (generated_df.size() != data.raw.ncoord)
      throw std::runtime_error("generated DF response has invalid coordinate dimensions");
    result.forces.assign(data.raw.ncoord, 0.0);
    const std::size_t matrix_elements = data.raw.nbf * data.raw.nbf;
    for (std::size_t coordinate = 0; coordinate < data.raw.ncoord; ++coordinate) {
      if (!generated_one_electron.empty()) {
        result.forces[coordinate] =
            -(generated_one_electron[coordinate] + generated_df[coordinate] +
              data.one_electron.nuclear_repulsion_derivative[coordinate]);
        continue;
      }
      const double* overlap_derivative =
          data.one_electron.overlap_derivative.data() + coordinate * matrix_elements;
      const double* hcore_derivative =
          data.one_electron.hcore_derivative.data() + coordinate * matrix_elements;
      double derivative =
          generated_df[coordinate] + data.one_electron.nuclear_repulsion_derivative[coordinate];
      for (std::size_t item = 0; item < matrix_elements; ++item) {
        derivative +=
            density[item] * hcore_derivative[item] - weighted[item] * overlap_derivative[item];
      }
      result.forces[coordinate] = -derivative;
    }
  } else {
    if (!generated_one_electron.empty()) {
      throw std::runtime_error(
          "CUDA DF response failed with generated one-electron gradients selected");
    }
    if (data.raw.three_center.empty()) {
      throw std::runtime_error(
          "CUDA DF source-backed force response failed after tensor storage was released");
    }
    result.forces = build_density_fitting_rhf_forces(data.one_electron, data.raw, density, weighted,
                                                     options.density_fitting_relative_threshold);
  }
  result.density = density;
}

[[maybe_unused]] void finalize_density_fitting_uhf(
    const DensityFittingScfData& data, const Matrix& orthogonalizer, std::size_t alpha_occupied,
    std::size_t beta_occupied, Matrix& alpha_density, Matrix& beta_density,
    const ScfOptions& options, ScfResult& result, CudaDensityFittingJkPlan* cuda_plan = nullptr,
    std::size_t cuda_system = 0) {
#if !VIBEQC_HAS_CUDA
  (void)cuda_plan;
#endif
  const std::size_t n = data.one_electron.nbf;
  Matrix alpha_fock;
  Matrix beta_fock;
  const auto execute_item_uhf_jk =
      [&](const Matrix& item_alpha, const Matrix& item_beta, std::vector<double>& item_coulomb,
          std::vector<double>& item_alpha_exchange, std::vector<double>& item_beta_exchange,
          std::string& item_detail) -> vibeqc_status {
    const std::size_t batch =
        cuda_plan == nullptr ? 0U : cuda_density_fitting_jk_plan_batch_size(cuda_plan);
    if (batch <= 1U) {
      return execute_cuda_density_fitting_uhf_jk(cuda_plan, item_alpha, item_beta, item_coulomb,
                                                 item_alpha_exchange, item_beta_exchange,
                                                 item_detail);
    }
    if (cuda_system >= batch || item_alpha.size() != item_beta.size() || item_alpha.empty() ||
        item_alpha.size() > std::numeric_limits<std::size_t>::max() / batch) {
      item_detail = "CUDA DF bucket item index or spin-density dimensions are invalid";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    return execute_cuda_density_fitting_uhf_jk_item(cuda_plan, cuda_system, item_alpha, item_beta,
                                                    item_coulomb, item_alpha_exchange,
                                                    item_beta_exchange, item_detail);
  };
  if (cuda_plan != nullptr) {
    std::vector<double> coulomb;
    std::vector<double> alpha_exchange;
    std::vector<double> beta_exchange;
    std::string detail;
    if (execute_item_uhf_jk(alpha_density, beta_density, coulomb, alpha_exchange, beta_exchange,
                            detail) == VIBEQC_STATUS_SUCCESS &&
        coulomb.size() == alpha_density.size() && alpha_exchange.size() == alpha_density.size() &&
        beta_exchange.size() == beta_density.size()) {
      alpha_fock = data.one_electron.hcore;
      beta_fock = data.one_electron.hcore;
      for (std::size_t item = 0; item < alpha_fock.size(); ++item) {
        alpha_fock[item] += coulomb[item] - alpha_exchange[item];
        beta_fock[item] += coulomb[item] - beta_exchange[item];
      }
    }
  }
  if (alpha_fock.empty() || beta_fock.empty()) {
    if (data.three_center.values.empty()) {
      throw std::runtime_error(
          "CUDA DF source-backed J/K failed after tensor storage was released");
    }
    std::tie(alpha_fock, beta_fock) = build_density_fitting_uhf_focks(
        data.one_electron.hcore, data.three_center, alpha_density, beta_density);
  }
  EigenResult alpha_orbitals = generalized_eigen(alpha_fock, orthogonalizer, n);
  EigenResult beta_orbitals = generalized_eigen(beta_fock, orthogonalizer, n);
  alpha_density = density_from_orbitals(alpha_orbitals.vectors, n, alpha_occupied, 1.0);
  beta_density = density_from_orbitals(beta_orbitals.vectors, n, beta_occupied, 1.0);
  if (cuda_plan != nullptr) {
    std::vector<double> coulomb;
    std::vector<double> alpha_exchange;
    std::vector<double> beta_exchange;
    std::string detail;
    if (execute_item_uhf_jk(alpha_density, beta_density, coulomb, alpha_exchange, beta_exchange,
                            detail) == VIBEQC_STATUS_SUCCESS &&
        coulomb.size() == alpha_density.size() && alpha_exchange.size() == alpha_density.size() &&
        beta_exchange.size() == beta_density.size()) {
      alpha_fock = data.one_electron.hcore;
      beta_fock = data.one_electron.hcore;
      for (std::size_t item = 0; item < alpha_fock.size(); ++item) {
        alpha_fock[item] += coulomb[item] - alpha_exchange[item];
        beta_fock[item] += coulomb[item] - beta_exchange[item];
      }
    } else {
      if (data.three_center.values.empty()) {
        throw std::runtime_error(
            "CUDA DF source-backed J/K failed after tensor storage was released");
      }
      std::tie(alpha_fock, beta_fock) = build_density_fitting_uhf_focks(
          data.one_electron.hcore, data.three_center, alpha_density, beta_density);
    }
  } else {
    std::tie(alpha_fock, beta_fock) = build_density_fitting_uhf_focks(
        data.one_electron.hcore, data.three_center, alpha_density, beta_density);
  }
  result.energy = uhf_electronic_energy(alpha_density, beta_density, data.one_electron.hcore,
                                        alpha_fock, beta_fock) +
                  data.one_electron.nuclear_repulsion;
  if (!options.compute_forces) {
    result.density = concatenate(alpha_density, beta_density);
    return;
  }

  const Matrix alpha_weighted = energy_weighted_density(
      alpha_orbitals.vectors, alpha_orbitals.values, n, alpha_occupied, 1.0);
  const Matrix beta_weighted =
      energy_weighted_density(beta_orbitals.vectors, beta_orbitals.values, n, beta_occupied, 1.0);
  const Matrix generated_one_electron = generated_one_electron_hf_gradient(
      data, alpha_density, alpha_weighted, &beta_density, &beta_weighted);
  const Matrix generated_df =
      generated_df_hf_gradient(data, cuda_plan, cuda_system, alpha_density, &beta_density);
  if (cuda_plan != nullptr) {
    if (generated_df.size() != data.raw.ncoord)
      throw std::runtime_error("generated DF response has invalid coordinate dimensions");
    result.forces.assign(data.raw.ncoord, 0.0);
    const std::size_t matrix_elements = data.raw.nbf * data.raw.nbf;
    for (std::size_t coordinate = 0; coordinate < data.raw.ncoord; ++coordinate) {
      if (!generated_one_electron.empty()) {
        result.forces[coordinate] =
            -(generated_one_electron[coordinate] + generated_df[coordinate] +
              data.one_electron.nuclear_repulsion_derivative[coordinate]);
        continue;
      }
      const double* overlap_derivative =
          data.one_electron.overlap_derivative.data() + coordinate * matrix_elements;
      const double* hcore_derivative =
          data.one_electron.hcore_derivative.data() + coordinate * matrix_elements;
      double derivative =
          generated_df[coordinate] + data.one_electron.nuclear_repulsion_derivative[coordinate];
      for (std::size_t item = 0; item < matrix_elements; ++item) {
        const double total_density = alpha_density[item] + beta_density[item];
        const double total_weighted = alpha_weighted[item] + beta_weighted[item];
        derivative +=
            total_density * hcore_derivative[item] - total_weighted * overlap_derivative[item];
      }
      result.forces[coordinate] = -derivative;
    }
  } else {
    if (!generated_one_electron.empty()) {
      throw std::runtime_error(
          "CUDA DF response failed with generated one-electron gradients selected");
    }
    if (data.raw.three_center.empty()) {
      throw std::runtime_error(
          "CUDA DF source-backed force response failed after tensor storage was released");
    }
    result.forces = build_density_fitting_uhf_forces(data.one_electron, data.raw, alpha_density,
                                                     beta_density, alpha_weighted, beta_weighted,
                                                     options.density_fitting_relative_threshold);
  }
  result.density = concatenate(alpha_density, beta_density);
}

}  // namespace

void validate_physical_reference(PhysicalReference& ref) {
  const auto n = ref.nbf;
  if (!n || !ref.nocc || ref.nocc >= n || n > SIZE_MAX / n || ref.orbital_energies.size() != n)
    throw std::invalid_argument("invalid physical reference dimensions/occupations");
  for (const auto* a : {&ref.overlap, &ref.hcore, &ref.fock, &ref.coefficients, &ref.density})
    if (a->size() != n * n) throw std::invalid_argument("invalid physical reference matrix shape");
  for (const auto* a : {&ref.overlap, &ref.hcore, &ref.fock, &ref.coefficients, &ref.density,
                        &ref.orbital_energies})
    if (!std::all_of(a->begin(), a->end(), [](double x) { return std::isfinite(x); }))
      throw std::runtime_error("nonfinite physical RHF reference");
  const auto canonical_density = density_from_orbitals(ref.coefficients, n, ref.nocc);
  const auto residual = commutator_residual(ref.fock, ref.density, ref.overlap, n);
  const auto fc = multiply(ref.fock, ref.coefficients, n);
  const auto sc = multiply(ref.overlap, ref.coefficients, n);
  const auto ct = transpose(ref.coefficients, n);
  const auto csc = multiply(ct, sc, n);
  const auto cfc = multiply(ct, fc, n);
  // Finite inputs can still overflow during validation. NaN comparisons below
  // must never allow an invalid reference to reach the correlation consumer.
  for (const auto* a : {&canonical_density, &residual, &fc, &sc, &csc, &cfc})
    if (!std::all_of(a->begin(), a->end(), [](double x) { return std::isfinite(x); }))
      throw std::runtime_error("nonfinite physical reference validation");
  if (!std::isfinite(ref.energy)) throw std::runtime_error("nonfinite physical reference energy");
  ref.commutator_residual = ref.canonical_density_drift = ref.eigen_residual = 0.0;
  double orthogonality = 0.0;
  double canonical = 0.0;
  for (std::size_t mu = 0; mu < n; ++mu) {
    for (std::size_t p = 0; p < n; ++p) {
      const auto k = index(mu, p, n);
      ref.commutator_residual = std::max(ref.commutator_residual, std::abs(residual[k]));
      ref.canonical_density_drift =
          std::max(ref.canonical_density_drift, std::abs(canonical_density[k] - ref.density[k]));
      ref.eigen_residual =
          std::max(ref.eigen_residual, std::abs(fc[k] - sc[k] * ref.orbital_energies[p]));
      orthogonality = std::max(orthogonality, std::abs(csc[k] - (mu == p ? 1.0 : 0.0)));
      canonical = std::max(canonical, std::abs(cfc[k] - (mu == p ? ref.orbital_energies[p] : 0.0)));
    }
  }
  if (std::max({ref.commutator_residual, ref.canonical_density_drift, ref.eigen_residual,
                orthogonality, canonical}) > 1e-8)
    throw std::runtime_error("invalid physical RHF reference: residual/canonicality drift");
}

void validate_hf_warm_density(const core::System& source, vibeqc_method method,
                              const std::vector<double>& density) {
  posthf::RawSource raw(source);
  const auto n = raw.nbf();
  Matrix overlap(n * n);
  raw.read(posthf::RawSource::Operator::overlap, {0, 0, 0, 0}, {n, n, 1, 1}, overlap.data(),
           overlap.size());
  if (method == VIBEQC_METHOD_UHF) {
    const auto [alpha, beta] = spin_occupations(source);
    validate_seed(overlap, density, n, {static_cast<unsigned>(alpha), static_cast<unsigned>(beta)},
                  1.0);
  } else {
    if (source.electron_count <= 0 || source.electron_count % 2 || source.multiplicity != 1)
      throw std::invalid_argument("invalid checkpoint RHF electron/spin counts");
    validate_seed(overlap, density, n, {static_cast<unsigned>(source.electron_count)}, 2.0);
  }
}

ScfResult run_prepared_fock_strategy(const PreparedFockPlan& plan, const ScfOptions& options,
                                     const std::vector<double>* initial_density) {
  const auto strategy = fock_strategy_for_execution(options);
  // A source prepared with first derivatives also owns all value data. An
  // energy-only replay may reuse it without evaluating a response, but no
  // other semantic or execution change can reuse this immutable owner.
  auto required = plan.strategy().spec;
  required.derivative_order = strategy.spec.derivative_order;
  if (strategy.spec.derivative_order > plan.strategy().spec.derivative_order ||
      strategy != resolve_fock_build(required, plan.strategy().backend,
                                     plan.strategy().screening_tolerance,
                                     plan.strategy().metric_relative_threshold))
    throw std::invalid_argument("prepared Fock execution controls changed");
  const auto& system = plan.system();
  if (strategy.spec.spin == FockSpin::Restricted &&
      (system.electron_count <= 0 || system.electron_count % 2 || system.multiplicity != 1))
    throw std::invalid_argument("restricted Fock SCF requires a closed-shell electron count");
  ScfResult result =
      strategy.spec.spin == FockSpin::Unrestricted
          ? run_uhf_host_plan(system, options, plan.one_electron(), plan, initial_density)
          : run_rhf_host_plan(system, options, plan.one_electron(), plan, initial_density);
  // The host (value) Fock build is always FP64; report the requested policy so
  // provenance distinguishes "asked fp64" from "asked auto, collapsed to FP64".
  result.precision.requested_mode = options.precision_mode.value_or(VIBEQC_PRECISION_FP64);
  if (options.export_physical_reference && strategy.spec.spin == FockSpin::Restricted) {
    if (!result.converged) return result;
    if (strategy.spec.coulomb.approximation != FockApproximation::Exact ||
        strategy.spec.exchange.approximation != FockApproximation::Exact ||
        options.screening_tolerance != 0.0)
      throw std::invalid_argument("physical reference requires unscreened conventional integrals");
    const auto n = molecule::ao_count(system);
    const auto occupied = static_cast<std::size_t>(system.electron_count / 2);
    if (occupied == 0 || occupied >= n)
      throw std::invalid_argument("physical RHF reference requires a nonempty virtual space");
    const auto capacity = posthf::rhf_reference_capacity(system, options.diis_history,
                                                         strategy.backend == FockBackend::Cpu);
    if (options.reference_memory_budget_bytes != 0 &&
        capacity > options.reference_memory_budget_bytes)
      throw std::length_error("bounded RHF reference exceeds numeric memory budget");
    const auto jk = plan.build(result.density);
    const auto matrices = assemble_fock(strategy, plan.one_electron().hcore, jk);
    auto ref = std::make_shared<PhysicalReference>();
    ref->nbf = n;
    ref->nocc = occupied;
    ref->overlap = plan.one_electron().overlap;
    ref->hcore = plan.one_electron().hcore;
    ref->fock = matrices.alpha;
    ref->density = result.density;
    const auto orthogonalizer = symmetric_orthogonalizer(ref->overlap, n);
    auto canonical = generalized_eigen(ref->fock, orthogonalizer, n);
    ref->coefficients = std::move(canonical.vectors);
    ref->orbital_energies = std::move(canonical.values);
    ref->energy = electronic_energy(ref->density, ref->hcore, ref->fock) +
                  plan.one_electron().nuclear_repulsion;
    ref->numeric_capacity_bytes = capacity;
    validate_physical_reference(*ref);
    result.energy = ref->energy;
    result.reference = std::move(ref);
  }
  return result;
}

ScfResult run_cpu_fock_strategy(const core::System& system, const core::System* auxiliary,
                                const ScfOptions& options,
                                const std::vector<double>* initial_density) {
  const auto strategy = fock_strategy_for_execution(options);
  if (strategy.backend != FockBackend::Cpu)
    throw std::invalid_argument("CPU Fock entry requires a CPU strategy");
  if (options.export_physical_reference && strategy.spec.spin == FockSpin::Restricted &&
      options.reference_memory_budget_bytes != 0) {
    const auto capacity = posthf::rhf_reference_capacity(system, options.diis_history, true);
    if (capacity > options.reference_memory_budget_bytes)
      throw std::length_error("bounded RHF reference exceeds numeric memory budget");
  }
  const PreparedFockPlan plan(system, auxiliary, strategy);
  return run_prepared_fock_strategy(plan, options, initial_density);
}

ScfResult run_rhf(const core::System& system, const ScfOptions& options,
                  const std::vector<double>* initial_density) {
  ScfOptions execution = options;
  // Export is an internal energy-only consumer; suppress derivative preparation
  // as well as the final force calculation even with default ScfOptions.
  if (execution.export_physical_reference) execution.compute_forces = false;
  if (!execution.resolved_fock_build)
    execution.resolved_fock_build = resolve_fock_build(
        make_hf_fock_spec(FockSpin::Restricted), FockBackend::Cpu, options.screening_tolerance);
  require_exact_direct_strategy(*execution.resolved_fock_build, FockSpin::Restricted,
                                FockBackend::Cpu);
  return run_cpu_fock_strategy(system, nullptr, execution, initial_density);
}

ScfResult run_rhf_density_fitting(const core::System& system, const core::System& auxiliary_system,
                                  const ScfOptions& options,
                                  const std::vector<double>* initial_density) {
  ScfOptions execution = options;
  const auto expected = resolve_fock_build(
      make_hf_fock_spec(FockSpin::Restricted, FockApproximation::DensityFitted), FockBackend::Cpu,
      options.screening_tolerance, options.density_fitting_relative_threshold);
  if (execution.resolved_fock_build && *execution.resolved_fock_build != expected)
    throw std::invalid_argument("legacy DF HF entry requires its resolved standard HF strategy");
  execution.resolved_fock_build = expected;
  return run_cpu_fock_strategy(system, &auxiliary_system, execution, initial_density);
}

ScfResult run_uhf(const core::System& system, const ScfOptions& options,
                  const std::vector<double>* initial_density) {
  ScfOptions execution = options;
  if (!execution.resolved_fock_build)
    execution.resolved_fock_build = resolve_fock_build(
        make_hf_fock_spec(FockSpin::Unrestricted), FockBackend::Cpu, options.screening_tolerance);
  require_exact_direct_strategy(*execution.resolved_fock_build, FockSpin::Unrestricted,
                                FockBackend::Cpu);
  return run_cpu_fock_strategy(system, nullptr, execution, initial_density);
}

ScfResult run_uhf_density_fitting(const core::System& system, const core::System& auxiliary_system,
                                  const ScfOptions& options,
                                  const std::vector<double>* initial_density) {
  ScfOptions execution = options;
  const auto expected = resolve_fock_build(
      make_hf_fock_spec(FockSpin::Unrestricted, FockApproximation::DensityFitted), FockBackend::Cpu,
      options.screening_tolerance, options.density_fitting_relative_threshold);
  if (execution.resolved_fock_build && *execution.resolved_fock_build != expected)
    throw std::invalid_argument("legacy DF HF entry requires its resolved standard HF strategy");
  execution.resolved_fock_build = expected;
  return run_cpu_fock_strategy(system, &auxiliary_system, execution, initial_density);
}

#if VIBEQC_HAS_CUDA

using CudaDensityFittingPlanPtr =
    std::unique_ptr<CudaDensityFittingJkPlan, decltype(&destroy_cuda_density_fitting_jk_plan)>;

CudaDensityFittingPlanPtr make_cuda_density_fitting_plan(
    const DensityFittingScfData& data, const ScfOptions& options, int device_id,
    std::size_t occupied,
    std::vector<CudaDensityFittingMetricDiagnostic>* output_diagnostics = nullptr,
    const core::System* orbital_system = nullptr, const core::System* auxiliary_system = nullptr) {
  const auto planning_budget = options.density_fitting_memory_budget_bytes
                                   ? options.density_fitting_memory_budget_bytes / 2
                                   : options.density_fitting_memory_budget_bytes;

  CudaDensityFittingJkPlan* raw_plan = nullptr;
  std::vector<CudaDensityFittingMetricDiagnostic> diagnostics;
  std::string detail;
  std::size_t auxiliary_tile = 0;
  std::size_t ao_pair_tile = 0;
  if (planning_budget != 0) {
    const DensityFittingTilePlan tile_plan = plan_density_fitting_tiles(
        1, data.raw.nbf, data.raw.naux, std::max<std::size_t>(occupied, 1), planning_budget);
    auxiliary_tile = tile_plan.auxiliary_tile;
    ao_pair_tile = tile_plan.ao_pair_tile;
  }
  if (planning_budget != 0 && orbital_system != nullptr && auxiliary_system != nullptr) {
    CudaDensityFittingIntegralSource* source = nullptr;
    std::vector<double> source_metrics;
    std::size_t source_nbf = 0;
    std::size_t source_naux = 0;
    const vibeqc_status source_status = create_cuda_density_fitting_integral_source(
        device_id, {*orbital_system}, {*auxiliary_system}, &source, source_metrics, source_nbf,
        source_naux, detail);
    if (source_status == VIBEQC_STATUS_OUT_OF_MEMORY && runtime::active_device_resource_ledger)
      throw std::bad_alloc();
    if (source_status != VIBEQC_STATUS_SUCCESS) {
      throw std::runtime_error(detail.empty() ? "CUDA DF source preparation failed" : detail);
    }
    DensityFittingTilePlan source_tile_plan;
    try {
      source_tile_plan = plan_density_fitting_tiles(
          1, source_nbf, source_naux, std::max<std::size_t>(occupied, 1), planning_budget,
          cuda_density_fitting_integral_source_device_bytes(source));
    } catch (...) {
      destroy_cuda_density_fitting_integral_source(source);
      throw;
    }
    auxiliary_tile = source_tile_plan.auxiliary_tile;
    ao_pair_tile = source_tile_plan.ao_pair_tile;
    const vibeqc_status plan_status = create_cuda_density_fitting_jk_plan_from_source(
        device_id, &source, 1, source_nbf, source_naux, source_metrics,
        options.density_fitting_relative_threshold, auxiliary_tile, ao_pair_tile, &raw_plan,
        diagnostics, detail);
    destroy_cuda_density_fitting_integral_source(source);
    if (plan_status == VIBEQC_STATUS_OUT_OF_MEMORY && runtime::active_device_resource_ledger)
      throw std::bad_alloc();
    if (plan_status != VIBEQC_STATUS_SUCCESS) {
      throw std::runtime_error(detail.empty() ? "CUDA density-fitting source plan creation failed"
                                              : detail);
    }
  } else {
    const vibeqc_status status =
        planning_budget != 0
            ? create_cuda_density_fitting_jk_plan_tiled(
                  device_id, 1, data.raw.nbf, data.raw.naux, data.raw.metric, data.raw.three_center,
                  options.density_fitting_relative_threshold, auxiliary_tile, ao_pair_tile,
                  &raw_plan, diagnostics, detail)
            : create_cuda_density_fitting_jk_plan(
                  device_id, 1, data.raw.nbf, data.raw.naux, data.raw.metric, data.raw.three_center,
                  options.density_fitting_relative_threshold, 0, &raw_plan, diagnostics, detail);
    if (status == VIBEQC_STATUS_OUT_OF_MEMORY && runtime::active_device_resource_ledger)
      throw std::bad_alloc();
    if (status != VIBEQC_STATUS_SUCCESS) {
      throw std::runtime_error(detail.empty() ? "CUDA density-fitting plan creation failed"
                                              : detail);
    }
  }
  // Keep the raw plan owned while copying optional diagnostics; an allocation
  // failure in that copy must still release all CUDA resources.
  CudaDensityFittingPlanPtr owned_plan(raw_plan, &destroy_cuda_density_fitting_jk_plan);
  if (output_diagnostics != nullptr) {
    *output_diagnostics = diagnostics;
  }
  return owned_plan;
}

ScfResult run_cuda_independent_fock_strategy(const core::System& system,
                                             const core::System* auxiliary,
                                             const ScfOptions& options, int device_id,
                                             const std::vector<double>* initial_density) {
  const auto strategy = fock_strategy_for_execution(options);
  if (strategy.backend != FockBackend::Cuda)
    throw std::invalid_argument("CUDA Fock entry requires a CUDA strategy");
  const PreparedFockPlan plan(system, auxiliary, strategy, device_id,
                              options.density_fitting_memory_budget_bytes);
  return run_prepared_fock_strategy(plan, options, initial_density);
}

CudaDensityFittingPlanPtr make_cuda_density_fitting_batch_plan(
    const std::vector<DensityFittingScfData>& data, const ScfOptions& options, int device_id,
    std::size_t occupied,
    std::vector<CudaDensityFittingMetricDiagnostic>* output_diagnostics = nullptr,
    const std::vector<core::System>* orbital_systems = nullptr,
    const std::vector<core::System>* auxiliary_systems = nullptr) {
  const auto planning_budget = options.density_fitting_memory_budget_bytes
                                   ? options.density_fitting_memory_budget_bytes / 2
                                   : options.density_fitting_memory_budget_bytes;

  if (data.empty()) {
    throw std::invalid_argument("CUDA density-fitting batch cannot be empty");
  }
  const std::size_t nbf = data.front().raw.nbf;
  const std::size_t naux = data.front().raw.naux;
  std::vector<double> metrics;
  std::vector<double> three_center;
  std::vector<CudaDensityFittingMetricDiagnostic> diagnostics;
  if (planning_budget != 0 && orbital_systems != nullptr && auxiliary_systems != nullptr &&
      orbital_systems->size() == data.size() && auxiliary_systems->size() == data.size()) {
    CudaDensityFittingIntegralSource* source = nullptr;
    std::size_t source_nbf = 0;
    std::size_t source_naux = 0;
    std::string source_detail;
    const vibeqc_status source_status = create_cuda_density_fitting_integral_source(
        device_id, *orbital_systems, *auxiliary_systems, &source, metrics, source_nbf, source_naux,
        source_detail);
    if (source_status == VIBEQC_STATUS_OUT_OF_MEMORY && runtime::active_device_resource_ledger)
      throw std::bad_alloc();
    if (source_status != VIBEQC_STATUS_SUCCESS) {
      throw std::runtime_error(source_detail.empty() ? "CUDA DF source preparation failed"
                                                     : source_detail);
    }
    DensityFittingTilePlan tile_plan;
    try {
      tile_plan = plan_density_fitting_tiles(
          data.size(), nbf, naux, std::max<std::size_t>(occupied, 1), planning_budget,
          cuda_density_fitting_integral_source_device_bytes(source));
    } catch (...) {
      destroy_cuda_density_fitting_integral_source(source);
      throw;
    }
    CudaDensityFittingJkPlan* raw_plan = nullptr;
    std::string detail;
    const vibeqc_status plan_status = create_cuda_density_fitting_jk_plan_from_source(
        device_id, &source, data.size(), source_nbf, source_naux, metrics,
        options.density_fitting_relative_threshold, tile_plan.auxiliary_tile,
        tile_plan.ao_pair_tile, &raw_plan, diagnostics, detail);
    destroy_cuda_density_fitting_integral_source(source);
    if (plan_status == VIBEQC_STATUS_OUT_OF_MEMORY && runtime::active_device_resource_ledger)
      throw std::bad_alloc();
    if (plan_status != VIBEQC_STATUS_SUCCESS) {
      throw std::runtime_error(detail.empty() ? "CUDA DF source plan creation failed" : detail);
    }
    if (output_diagnostics != nullptr) *output_diagnostics = diagnostics;
    return CudaDensityFittingPlanPtr(raw_plan, &destroy_cuda_density_fitting_jk_plan);
  }
  std::size_t metric_size = naux * naux;
  std::size_t tensor_size = nbf * nbf * naux;
  metrics.reserve(data.size() * metric_size);
  three_center.reserve(data.size() * tensor_size);
  for (const DensityFittingScfData& item : data) {
    if (item.raw.nbf != nbf || item.raw.naux != naux || item.raw.metric.size() != metric_size ||
        item.raw.three_center.size() != tensor_size) {
      throw std::invalid_argument(
          "CUDA density-fitting bucket has incompatible auxiliary dimensions");
    }
    metrics.insert(metrics.end(), item.raw.metric.begin(), item.raw.metric.end());
    three_center.insert(three_center.end(), item.raw.three_center.begin(),
                        item.raw.three_center.end());
  }
  CudaDensityFittingJkPlan* raw_plan = nullptr;
  std::string detail;
  std::size_t auxiliary_tile = 0;
  std::size_t ao_pair_tile = 0;
  if (planning_budget != 0) {
    const DensityFittingTilePlan tile_plan = plan_density_fitting_tiles(
        data.size(), nbf, naux, std::max<std::size_t>(occupied, 1), planning_budget);
    auxiliary_tile = tile_plan.auxiliary_tile;
    ao_pair_tile = tile_plan.ao_pair_tile;
  }
  const vibeqc_status status =
      planning_budget != 0
          ? create_cuda_density_fitting_jk_plan_tiled(
                device_id, data.size(), nbf, naux, metrics, three_center,
                options.density_fitting_relative_threshold, auxiliary_tile, ao_pair_tile, &raw_plan,
                diagnostics, detail)
          : create_cuda_density_fitting_jk_plan(
                device_id, data.size(), nbf, naux, metrics, three_center,
                options.density_fitting_relative_threshold, 0, &raw_plan, diagnostics, detail);
  if (status == VIBEQC_STATUS_OUT_OF_MEMORY && runtime::active_device_resource_ledger)
    throw std::bad_alloc();
  if (status != VIBEQC_STATUS_SUCCESS) {
    throw std::runtime_error(detail.empty() ? "CUDA density-fitting batch plan creation failed"
                                            : detail);
  }
  // Keep the raw plan owned while copying optional diagnostics; an allocation
  // failure in that copy must still release all CUDA resources.
  CudaDensityFittingPlanPtr owned_plan(raw_plan, &destroy_cuda_density_fitting_jk_plan);
  if (output_diagnostics != nullptr) {
    *output_diagnostics = diagnostics;
  }
  return owned_plan;
}

core::System density_fitting_auxiliary_for_geometry(
    const std::optional<core::System>& auxiliary_template, const core::System& system) {
  if (!auxiliary_template.has_value()) return system;
  core::System auxiliary = *auxiliary_template;
  auxiliary.atoms = system.atoms;
  auxiliary.charge = system.charge;
  auxiliary.multiplicity = system.multiplicity;
  auxiliary.electron_count = system.electron_count;
  return auxiliary;
}

/**
 * Prepare CUDA DF data for a fleet while isolating failures to individual
 * systems. Raw metric/three-center tensors are generated in homogeneous
 * batches; one-electron tensors remain per-system because their public AO
 * representations may differ even when Cartesian dimensions match.
 */
std::vector<std::optional<DensityFittingScfData>> prepare_cuda_density_fitting_batch(
    const std::vector<core::System>& systems, const std::optional<core::System>& auxiliary_template,
    double relative_threshold, std::size_t output_budget_bytes, int device_id,
    std::vector<vibeqc_status>& statuses) {
  const std::size_t count = systems.size();
  statuses.assign(count, VIBEQC_STATUS_INTERNAL_ERROR);
  std::vector<std::optional<DensityFittingScfData>> prepared(count);
  std::vector<core::System> auxiliaries(count);
  std::vector<bool> auxiliary_valid(count, false);

  for (std::size_t source = 0; source < count; ++source) {
    try {
      auxiliaries[source] =
          density_fitting_auxiliary_for_geometry(auxiliary_template, systems[source]);
      auxiliary_valid[source] = true;
    } catch (const std::bad_alloc&) {
      statuses[source] = VIBEQC_STATUS_OUT_OF_MEMORY;
    } catch (const std::invalid_argument&) {
      statuses[source] = VIBEQC_STATUS_INVALID_ARGUMENT;
    } catch (...) {
      statuses[source] = VIBEQC_STATUS_NUMERICAL_FAILURE;
    }
  }

  // Partition by Cartesian dimensions and atom count. This keeps the batch
  // kernel's packed strides valid while allowing ragged fleets to proceed.
  std::vector<std::vector<std::size_t>> groups;
  for (std::size_t source = 0; source < count; ++source) {
    if (!auxiliary_valid[source]) continue;
    const std::size_t orbital_count = molecule::cartesian_ao_count(systems[source]);
    const std::size_t auxiliary_count = molecule::cartesian_ao_count(auxiliaries[source]);
    const std::size_t atom_count = systems[source].atoms.size();
    bool placed = false;
    for (auto& group : groups) {
      const std::size_t representative = group.front();
      if (molecule::cartesian_ao_count(systems[representative]) == orbital_count &&
          molecule::cartesian_ao_count(auxiliaries[representative]) == auxiliary_count &&
          systems[representative].atoms.size() == atom_count) {
        group.push_back(source);
        placed = true;
        break;
      }
    }
    if (!placed) groups.push_back({source});
  }

  for (const auto& group : groups) {
    // Process homogeneous systems in bounded chunks. Build only the current
    // chunk vectors; retaining a second full copy of the entire group here
    // would consume the same host budget that the chunking is meant to bound.
    const std::size_t representative = group.front();
    const std::size_t nbf_cart = molecule::cartesian_ao_count(systems[representative]);
    const std::size_t naux_cart = molecule::cartesian_ao_count(auxiliaries[representative]);
    const std::size_t coordinates = systems[representative].atoms.size() * 3U;
    // Account for every host vector that can coexist while a chunk is being
    // prepared: possible DF metric/three-center values and derivatives,
    // one-electron values and derivatives, and the packed one-electron
    // staging vectors used by the CUDA bridge.  The positive-budget source
    // path skips the DF generator entirely, but retaining that component here
    // keeps the estimate safe for the non-source fallback.
    const long double matrix_elements = static_cast<long double>(nbf_cart) * nbf_cart;
    const long double metric_elements = static_cast<long double>(naux_cart) * naux_cart;
    const long double three_center_elements = matrix_elements * naux_cart;
    const long double values_and_derivatives = static_cast<long double>(coordinates) + 1.0L;
    const long double density_fitting_output =
        output_budget_bytes == 0U
            ? (metric_elements + three_center_elements) * values_and_derivatives
            : 0.0L;
    // IntegralData retains overlap/hcore plus coordinate-major derivatives;
    // nuclear-repulsion derivatives contribute one scalar per coordinate.
    const long double one_electron_output =
        2.0L * matrix_elements * values_and_derivatives + coordinates;
    // build_cuda_one_electron_integrals_batch keeps packed overlap, hcore,
    // and nuclear vectors alive while copying each item into IntegralData.
    const long double one_electron_staging = 2.0L * matrix_elements + 1.0L;
    // pack_host_batch also retains basis/atom metadata, shell-pair indices,
    // queue descriptors, and Cartesian system copies while the CUDA bridge is
    // active.  Those vectors are irregular (and private to the CUDA TU), so
    // use a deliberately conservative descriptor-count bound rather than
    // pretending the numeric output estimate covers them.
    const long double atoms = static_cast<long double>(systems[representative].atoms.size());
    const long double shells = static_cast<long double>(systems[representative].shells.size());
    long double primitives = 0.0L;
    for (const core::Shell& shell : systems[representative].shells) {
      primitives += static_cast<long double>(shell.primitives.size());
    }
    const long double shell_pairs = shells * (shells + 1.0L) / 2.0L;
    const long double shell_quartets = shell_pairs * (shell_pairs + 1.0L) / 2.0L;
    const long double metadata_doubles_per_system =
        64.0L * (1.0L + atoms + shells + primitives + static_cast<long double>(nbf_cart) +
                 static_cast<long double>(molecule::ao_count(systems[representative])) +
                 shell_pairs + shell_quartets);
    // The one-electron bridge also keeps a Cartesian warm-density matrix and
    // packed pair-index vectors even when no warm density is supplied.
    const long double quadratic_staging_doubles = 2.0L * matrix_elements;
    const long double per_system_output_estimate =
        (density_fitting_output + one_electron_output + one_electron_staging +
         metadata_doubles_per_system + quadratic_staging_doubles) *
        sizeof(double);
    const std::size_t per_system_output =
        per_system_output_estimate >=
                static_cast<long double>(std::numeric_limits<std::size_t>::max())
            ? std::numeric_limits<std::size_t>::max()
            : static_cast<std::size_t>(per_system_output_estimate);
    const std::size_t chunk_size =
        output_budget_bytes == 0U
            ? group.size()
            : std::max<std::size_t>(
                  1U, std::min(group.size(),
                               output_budget_bytes / std::max<std::size_t>(per_system_output, 1U)));
    if (output_budget_bytes != 0U && per_system_output > output_budget_bytes) {
      for (const std::size_t source : group) {
        statuses[source] = VIBEQC_STATUS_OUT_OF_MEMORY;
      }
      continue;
    }
    for (std::size_t chunk_begin = 0; chunk_begin < group.size(); chunk_begin += chunk_size) {
      const std::size_t chunk_end = std::min(group.size(), chunk_begin + chunk_size);
      std::vector<core::System> orbital_chunk;
      std::vector<core::System> auxiliary_chunk;
      orbital_chunk.reserve(chunk_end - chunk_begin);
      if (output_budget_bytes == 0U) {
        auxiliary_chunk.reserve(chunk_end - chunk_begin);
      }
      for (std::size_t slot = chunk_begin; slot < chunk_end; ++slot) {
        orbital_chunk.push_back(systems[group[slot]]);
        if (output_budget_bytes == 0U) {
          auxiliary_chunk.push_back(auxiliaries[group[slot]]);
        }
      }
      std::vector<integrals::DensityFittingIntegralData> raw_batch;
      std::vector<integrals::IntegralData> one_electron_batch;
      std::string detail;
      // Source-backed positive-budget plans regenerate DF values on demand;
      // generating a complete raw batch here only to discard it would defeat
      // the budget. Keep this batch limited to one-electron response data.
      const vibeqc_status batch_status = output_budget_bytes == 0U
                                             ? build_cuda_density_fitting_integrals_batch(
                                                   device_id, orbital_chunk, auxiliary_chunk,
                                                   raw_batch, detail, output_budget_bytes, false)
                                             : VIBEQC_STATUS_SUCCESS;
      const vibeqc_status one_electron_batch_status =
          batch_status == VIBEQC_STATUS_SUCCESS
              ? build_cuda_one_electron_integrals_batch(
                    device_id, orbital_chunk, one_electron_batch, detail,
                    !cuda_policy::generated_one_electron_derivatives_requested())
              : batch_status;
      if (batch_status == VIBEQC_STATUS_SUCCESS &&
          one_electron_batch_status == VIBEQC_STATUS_SUCCESS &&
          (output_budget_bytes != 0U || raw_batch.size() == orbital_chunk.size()) &&
          one_electron_batch.size() == orbital_chunk.size()) {
        for (std::size_t slot = chunk_begin; slot < chunk_end; ++slot) {
          const std::size_t local = slot - chunk_begin;
          const std::size_t source = group[slot];
          try {
            integrals::IntegralData one_electron =
                integrals::transform_integrals(one_electron_batch[local], systems[source]);
            integrals::DensityFittingIntegralData raw;
            if (output_budget_bytes != 0U) {
              raw.nbf = molecule::ao_count(systems[source]);
              raw.naux = molecule::ao_count(auxiliaries[source]);
              raw.ncoord = systems[source].atoms.size() * 3U;
            } else {
              raw = integrals::transform_density_fitting_integrals(
                  raw_batch[local], systems[source], auxiliaries[source]);
            }
            prepared[source] =
                output_budget_bytes != 0U
                    ? assemble_density_fitting_metadata(std::move(one_electron), std::move(raw),
                                                        relative_threshold)
                    : assemble_density_fitting_data(std::move(one_electron), std::move(raw),
                                                    relative_threshold);
          } catch (const std::bad_alloc&) {
            statuses[source] = VIBEQC_STATUS_OUT_OF_MEMORY;
          } catch (const std::invalid_argument&) {
            statuses[source] = VIBEQC_STATUS_INVALID_ARGUMENT;
          } catch (...) {
            statuses[source] = VIBEQC_STATUS_NUMERICAL_FAILURE;
          }
        }
        continue;
      }

      // A chunk-level launch can fail for resource or topology reasons. Retry
      // each item independently so one bad system never poisons its neighbors.
      for (std::size_t slot = chunk_begin; slot < chunk_end; ++slot) {
        const std::size_t source = group[slot];
        if (output_budget_bytes != 0U) {
          // Retry through the bounded batch API with one item.  The legacy
          // single-system entry point may allocate an unbounded derivative
          // result, so it cannot be used here; a one-item batch preserves the
          // output-budget contract while isolating a bad neighbor.
          try {
            std::vector<core::System> single_orbital{systems[source]};
            std::vector<core::System> single_auxiliary{auxiliaries[source]};
            std::vector<integrals::DensityFittingIntegralData> single_raw;
            std::vector<integrals::IntegralData> single_one_electron;
            std::string retry_detail;
            const vibeqc_status retry_raw_status =
                output_budget_bytes == 0U
                    ? build_cuda_density_fitting_integrals_batch(
                          device_id, single_orbital, single_auxiliary, single_raw, retry_detail,
                          output_budget_bytes, false)
                    : VIBEQC_STATUS_SUCCESS;
            const vibeqc_status retry_one_electron_status =
                retry_raw_status == VIBEQC_STATUS_SUCCESS
                    ? build_cuda_one_electron_integrals_batch(
                          device_id, single_orbital, single_one_electron, retry_detail,
                          !cuda_policy::generated_one_electron_derivatives_requested())
                    : retry_raw_status;
            if (retry_raw_status == VIBEQC_STATUS_SUCCESS &&
                retry_one_electron_status == VIBEQC_STATUS_SUCCESS &&
                (output_budget_bytes != 0U || single_raw.size() == 1U) &&
                single_one_electron.size() == 1U) {
              integrals::IntegralData one_electron =
                  integrals::transform_integrals(single_one_electron.front(), systems[source]);
              integrals::DensityFittingIntegralData raw;
              if (output_budget_bytes != 0U) {
                raw.nbf = molecule::ao_count(systems[source]);
                raw.naux = molecule::ao_count(auxiliaries[source]);
                raw.ncoord = systems[source].atoms.size() * 3U;
              } else {
                raw = integrals::transform_density_fitting_integrals(
                    single_raw.front(), systems[source], auxiliaries[source]);
              }
              prepared[source] =
                  output_budget_bytes != 0U
                      ? assemble_density_fitting_metadata(std::move(one_electron), std::move(raw),
                                                          relative_threshold)
                      : assemble_density_fitting_data(std::move(one_electron), std::move(raw),
                                                      relative_threshold);
              continue;
            }
            statuses[source] = retry_raw_status != VIBEQC_STATUS_SUCCESS
                                   ? retry_raw_status
                                   : retry_one_electron_status;
          } catch (const std::bad_alloc&) {
            statuses[source] = VIBEQC_STATUS_OUT_OF_MEMORY;
          } catch (const std::invalid_argument&) {
            statuses[source] = VIBEQC_STATUS_INVALID_ARGUMENT;
          } catch (...) {
            statuses[source] = VIBEQC_STATUS_NUMERICAL_FAILURE;
          }
          continue;
        }
        try {
          integrals::DensityFittingIntegralData cartesian;
          std::string item_detail;
          const vibeqc_status item_status = build_cuda_density_fitting_integrals(
              device_id, systems[source], auxiliaries[source], cartesian, item_detail, false);
          if (item_status != VIBEQC_STATUS_SUCCESS) {
            statuses[source] = item_status;
            continue;
          }
          integrals::IntegralData cartesian_one_electron;
          const vibeqc_status one_electron_status = build_cuda_one_electron_integrals(
              device_id, systems[source], cartesian_one_electron, item_detail,
              !cuda_policy::generated_one_electron_derivatives_requested());
          if (one_electron_status != VIBEQC_STATUS_SUCCESS) {
            statuses[source] = one_electron_status;
            continue;
          }
          integrals::IntegralData one_electron =
              integrals::transform_integrals(cartesian_one_electron, systems[source]);
          integrals::DensityFittingIntegralData raw =
              integrals::transform_density_fitting_integrals(cartesian, systems[source],
                                                             auxiliaries[source]);
          prepared[source] = output_budget_bytes != 0U
                                 ? assemble_density_fitting_metadata(
                                       std::move(one_electron), std::move(raw), relative_threshold)
                                 : assemble_density_fitting_data(
                                       std::move(one_electron), std::move(raw), relative_threshold);
        } catch (const std::bad_alloc&) {
          statuses[source] = VIBEQC_STATUS_OUT_OF_MEMORY;
        } catch (const std::invalid_argument&) {
          statuses[source] = VIBEQC_STATUS_INVALID_ARGUMENT;
        } catch (...) {
          statuses[source] = VIBEQC_STATUS_NUMERICAL_FAILURE;
        }
      }
    }
  }
  for (std::size_t source = 0; source < count; ++source) {
    if (!prepared[source]) continue;
    try {
      bind_generated_one_electron(*prepared[source], systems[source], device_id,
                                  output_budget_bytes);
      bind_generated_df(*prepared[source], systems[source], auxiliaries[source], device_id,
                        output_budget_bytes);
    } catch (const std::bad_alloc&) {
      prepared[source].reset();
      statuses[source] = VIBEQC_STATUS_OUT_OF_MEMORY;
    }
  }
  return prepared;
}

ScfResult run_rhf_density_fitting_cuda_impl(const core::System& system,
                                            const core::System& auxiliary_system,
                                            const ScfOptions& options, int device_id,
                                            const std::vector<double>* initial_density) {
  if (options.hooks || options.strict_initial_density)
    throw std::invalid_argument("SCF proposal callbacks require the CPU reference backend");

  DensityFittingScfData data = prepare_density_fitting_data(
      system, auxiliary_system, options.density_fitting_relative_threshold, device_id,
      options.density_fitting_memory_budget_bytes, options.compute_forces);
  const std::size_t n = data.one_electron.nbf;
  const std::size_t occupied = static_cast<std::size_t>(system.electron_count / 2);
  if (occupied > n) {
    throw std::runtime_error("basis has fewer orbitals than occupied electron pairs");
  }
  const Matrix orthogonalizer = symmetric_orthogonalizer(data.one_electron.overlap, n);
  EigenResult orbitals;
  Matrix density = prepare_initial_density(system, data.one_electron, orthogonalizer, occupied,
                                           initial_density, orbitals);
  ScfResult result;
  result.initial_density_used = initial_density != nullptr;
  const CudaDensityFittingPlanPtr plan = make_cuda_density_fitting_plan(
      data, options, device_id, occupied, nullptr, &system, &auxiliary_system);
  if (options.density_fitting_memory_budget_bytes != 0) {
    discard_density_fitting_tensor_storage(data);
  }

  // Prefer the fully device-resident SCF loop.  It keeps the DF density,
  // Fock assembly, eigensolve, and convergence reductions on the plan stream;
  // the legacy host-orchestrated loop below remains a correctness-preserving
  // fallback for provider/workspace limitations or slow non-convergence.
  {
    std::vector<double> device_final_density;
    std::vector<CudaDensityFittingDeviceScfItem> device_records;
    std::string detail;
    const vibeqc_status device_status = run_cuda_density_fitting_rhf_device_scf(
        plan.get(), data.one_electron.hcore, orthogonalizer, density,
        {static_cast<std::int32_t>(occupied)}, {data.one_electron.nuclear_repulsion},
        options.max_iterations, options.energy_tolerance, options.density_tolerance,
        device_final_density, device_records, detail);
    // A resource rejection must not trigger an undisclosed host SCF retry.
    if (device_status == VIBEQC_STATUS_OUT_OF_MEMORY && runtime::active_device_resource_ledger)
      throw std::bad_alloc();
    if (device_status == VIBEQC_STATUS_SUCCESS && device_records.size() == 1 &&
        device_records.front().converged) {
      density = std::move(device_final_density);
      result.iterations = device_records.front().iterations;
      result.energy = device_records.front().energy;
      result.energy_change = device_records.front().energy_change;
      result.density_rms = device_records.front().density_rms;
      result.converged = true;
      finalize_density_fitting_rhf(data, orthogonalizer, occupied, density, options, result,
                                   plan.get());
      return result;
    }
  }
  Diis diis(options.diis_history);
  double previous_energy = std::numeric_limits<double>::infinity();
  for (unsigned iteration = 1; iteration <= options.max_iterations; ++iteration) {
    std::vector<double> coulomb;
    std::vector<double> exchange;
    std::string detail;
    const vibeqc_status status =
        execute_cuda_density_fitting_rhf_jk(plan.get(), density, coulomb, exchange, detail);
    if (status != VIBEQC_STATUS_SUCCESS) {
      throw std::runtime_error(detail.empty() ? "CUDA density-fitting RHF J/K failed" : detail);
    }
    Matrix fock = data.one_electron.hcore;
    for (std::size_t element = 0; element < fock.size(); ++element) {
      fock[element] += coulomb[element] - 0.5 * exchange[element];
    }
    const double energy = electronic_energy(density, data.one_electron.hcore, fock) +
                          data.one_electron.nuclear_repulsion;
    const Matrix residual = commutator_residual(fock, density, data.one_electron.overlap, n);
    const Matrix effective_fock = diis.update(fock, residual);
    orbitals = generalized_eigen(effective_fock, orthogonalizer, n);
    Matrix next_density = density_from_orbitals(orbitals.vectors, n, occupied);
    result.iterations = iteration;
    result.energy = energy;
    result.energy_change = std::isfinite(previous_energy) ? std::abs(energy - previous_energy)
                                                          : std::numeric_limits<double>::infinity();
    result.density_rms = density_rms(next_density, density);
    if (iteration > 1 && result.energy_change < options.energy_tolerance &&
        result.density_rms < options.density_tolerance) {
      density = std::move(next_density);
      result.converged = true;
      break;
    }
    previous_energy = energy;
    density = std::move(next_density);
  }
  if (!result.converged) return result;
  // The final diagonalization is intentionally rebuilt from the same CPU
  // oracle tensor used for force response. The SCF iterations above exercise
  // the CUDA DF contractions, while this last step keeps the existing
  // variational weighted-density convention exact.
  finalize_density_fitting_rhf(data, orthogonalizer, occupied, density, options, result,
                               plan.get());
  return result;
}

ScfResult run_uhf_density_fitting_cuda_impl(const core::System& system,
                                            const core::System& auxiliary_system,
                                            const ScfOptions& options, int device_id,
                                            const std::vector<double>* initial_density) {
  if (options.hooks || options.strict_initial_density)
    throw std::invalid_argument("SCF proposal callbacks require the CPU reference backend");

  DensityFittingScfData data = prepare_density_fitting_data(
      system, auxiliary_system, options.density_fitting_relative_threshold, device_id,
      options.density_fitting_memory_budget_bytes, options.compute_forces);
  const std::size_t n = data.one_electron.nbf;
  const auto [alpha_occupied, beta_occupied] = spin_occupations(system);
  if (alpha_occupied > n || beta_occupied > n) {
    throw std::runtime_error("basis has fewer orbitals than required UHF spin occupations");
  }
  const Matrix orthogonalizer = symmetric_orthogonalizer(data.one_electron.overlap, n);
  EigenResult alpha_orbitals;
  EigenResult beta_orbitals;
  auto [alpha_density, beta_density] =
      prepare_initial_uhf_density(data.one_electron, orthogonalizer, alpha_occupied, beta_occupied,
                                  initial_density, alpha_orbitals, beta_orbitals);
  ScfResult result;
  result.initial_density_used = initial_density != nullptr;
  const CudaDensityFittingPlanPtr plan = make_cuda_density_fitting_plan(
      data, options, device_id, std::max(alpha_occupied, beta_occupied), nullptr, &system,
      &auxiliary_system);
  if (options.density_fitting_memory_budget_bytes != 0) {
    discard_density_fitting_tensor_storage(data);
  }
  {
    std::vector<double> device_final_alpha;
    std::vector<double> device_final_beta;
    std::vector<CudaDensityFittingDeviceScfItem> device_records;
    std::string detail;
    const vibeqc_status device_status = run_cuda_density_fitting_uhf_device_scf(
        plan.get(), data.one_electron.hcore, orthogonalizer, alpha_density, beta_density,
        {static_cast<std::int32_t>(alpha_occupied)}, {static_cast<std::int32_t>(beta_occupied)},
        {data.one_electron.nuclear_repulsion}, options.max_iterations, options.energy_tolerance,
        options.density_tolerance, device_final_alpha, device_final_beta, device_records, detail);
    // A resource rejection must not trigger an undisclosed host SCF retry.
    if (device_status == VIBEQC_STATUS_OUT_OF_MEMORY && runtime::active_device_resource_ledger)
      throw std::bad_alloc();
    if (device_status == VIBEQC_STATUS_SUCCESS && device_records.size() == 1 &&
        device_records.front().converged) {
      alpha_density = std::move(device_final_alpha);
      beta_density = std::move(device_final_beta);
      result.iterations = device_records.front().iterations;
      result.energy = device_records.front().energy;
      result.energy_change = device_records.front().energy_change;
      result.density_rms = device_records.front().density_rms;
      result.converged = true;
      finalize_density_fitting_uhf(data, orthogonalizer, alpha_occupied, beta_occupied,
                                   alpha_density, beta_density, options, result, plan.get());
      return result;
    }
  }
  Diis diis(options.diis_history);
  double previous_energy = std::numeric_limits<double>::infinity();
  for (unsigned iteration = 1; iteration <= options.max_iterations; ++iteration) {
    std::vector<double> coulomb;
    std::vector<double> alpha_exchange;
    std::vector<double> beta_exchange;
    std::string detail;
    const vibeqc_status status = execute_cuda_density_fitting_uhf_jk(
        plan.get(), alpha_density, beta_density, coulomb, alpha_exchange, beta_exchange, detail);
    if (status != VIBEQC_STATUS_SUCCESS) {
      throw std::runtime_error(detail.empty() ? "CUDA density-fitting UHF J/K failed" : detail);
    }
    Matrix alpha_fock = data.one_electron.hcore;
    Matrix beta_fock = data.one_electron.hcore;
    for (std::size_t element = 0; element < alpha_fock.size(); ++element) {
      alpha_fock[element] += coulomb[element] - alpha_exchange[element];
      beta_fock[element] += coulomb[element] - beta_exchange[element];
    }
    const double energy = uhf_electronic_energy(alpha_density, beta_density,
                                                data.one_electron.hcore, alpha_fock, beta_fock) +
                          data.one_electron.nuclear_repulsion;
    const Matrix alpha_residual =
        commutator_residual(alpha_fock, alpha_density, data.one_electron.overlap, n);
    const Matrix beta_residual =
        commutator_residual(beta_fock, beta_density, data.one_electron.overlap, n);
    const Matrix effective_joined =
        diis.update(concatenate(alpha_fock, beta_fock), concatenate(alpha_residual, beta_residual));
    std::tie(alpha_fock, beta_fock) = split_spin_matrices(effective_joined, n * n);
    alpha_orbitals = generalized_eigen(alpha_fock, orthogonalizer, n);
    beta_orbitals = generalized_eigen(beta_fock, orthogonalizer, n);
    Matrix next_alpha = density_from_orbitals(alpha_orbitals.vectors, n, alpha_occupied, 1.0);
    Matrix next_beta = density_from_orbitals(beta_orbitals.vectors, n, beta_occupied, 1.0);
    result.iterations = iteration;
    result.energy = energy;
    result.energy_change = std::isfinite(previous_energy) ? std::abs(energy - previous_energy)
                                                          : std::numeric_limits<double>::infinity();
    result.density_rms =
        density_rms(concatenate(next_alpha, next_beta), concatenate(alpha_density, beta_density));
    if (iteration > 1 && result.energy_change < options.energy_tolerance &&
        result.density_rms < options.density_tolerance) {
      alpha_density = std::move(next_alpha);
      beta_density = std::move(next_beta);
      result.converged = true;
      break;
    }
    previous_energy = energy;
    alpha_density = std::move(next_alpha);
    beta_density = std::move(next_beta);
  }
  if (!result.converged) return result;
  finalize_density_fitting_uhf(data, orthogonalizer, alpha_occupied, beta_occupied, alpha_density,
                               beta_density, options, result, plan.get());
  return result;
}

std::vector<RhfBucketItem> run_rhf_density_fitting_cuda_bucket_impl(
    const std::vector<core::System>& systems, const std::optional<core::System>& auxiliary_template,
    const ScfOptions& options, const std::vector<const std::vector<double>*>& initial_densities,
    int device_id, std::vector<CudaDensityFittingMetricDiagnostic>* output_diagnostics,
    CudaDensityFittingJkPlan** cached_plan,
    std::vector<std::optional<DensityFittingScfData>>* prepared_cache) {
  if (options.hooks || options.strict_initial_density)
    throw std::invalid_argument("SCF proposal callbacks require the CPU reference backend");

  if (systems.size() != initial_densities.size()) {
    throw std::invalid_argument("CUDA density-fitting RHF bucket density count mismatch");
  }
  std::vector<RhfBucketItem> outputs(systems.size());
  if (systems.empty()) return outputs;

  // Preparation failures are recorded per item.  The remaining compatible
  // systems still share one CUDA plan, preserving fleet failure isolation.
  std::vector<std::size_t> source_indices;
  std::vector<DensityFittingScfData> data;
  std::vector<Matrix> orthogonalizers;
  std::vector<Matrix> densities;
  std::vector<EigenResult> orbitals;
  std::vector<Diis> diis;
  std::vector<double> previous_energies;
  if (prepared_cache != nullptr && prepared_cache->size() != systems.size()) {
    prepared_cache->assign(systems.size(), std::nullopt);
  }
  struct PreparedCacheGuard {
    std::vector<std::optional<DensityFittingScfData>>* cache{};
    std::vector<DensityFittingScfData>* data{};
    ~PreparedCacheGuard() {
      if (cache == nullptr || data == nullptr) return;
      if (cache->size() != data->size()) {
        cache->clear();
        return;
      }
      try {
        for (std::size_t slot = 0; slot < data->size(); ++slot) {
          (*cache)[slot] = std::move((*data)[slot]);
        }
      } catch (...) {
        // Cache restoration is an optimization; never let an allocation
        // failure during unwinding terminate an otherwise valid SCF result.
        cache->clear();
      }
    }
  } cache_guard{prepared_cache, &data};
  source_indices.reserve(systems.size());
  data.reserve(systems.size());
  orthogonalizers.reserve(systems.size());
  densities.reserve(systems.size());
  orbitals.reserve(systems.size());
  diis.reserve(systems.size());
  previous_energies.reserve(systems.size());

  std::vector<vibeqc_status> preparation_status;
  std::vector<std::optional<DensityFittingScfData>> batched_prepared;
  const bool cached_data_complete =
      prepared_cache != nullptr && prepared_cache->size() == systems.size() &&
      std::all_of(
          prepared_cache->begin(), prepared_cache->end(), [&options, device_id](const auto& item) {
            return item.has_value() &&
                   one_electron_response_policy_matches(*item,
                                                        options.density_fitting_memory_budget_bytes,
                                                        options.compute_forces && device_id >= 0) &&
                   df_response_policy_matches(*item, options.density_fitting_memory_budget_bytes,
                                              options.density_fitting_relative_threshold,
                                              options.compute_forces && device_id >= 0);
          });
  if (prepared_cache != nullptr && !cached_data_complete && cached_plan != nullptr &&
      *cached_plan != nullptr) {
    // A budget change can switch resident tensors to a source-backed plan.
    // Rebuild both cache layers together so the new response data and plan
    // share the same storage contract and allocation limit.
    destroy_cuda_density_fitting_jk_plan(*cached_plan);
    *cached_plan = nullptr;
  }
  if (device_id >= 0 && !cached_data_complete) {
    batched_prepared = prepare_cuda_density_fitting_batch(
        systems, auxiliary_template, options.density_fitting_relative_threshold,
        options.density_fitting_memory_budget_bytes, device_id, preparation_status);
  }

  std::size_t nbf = 0;
  std::size_t naux = 0;
  for (std::size_t source = 0; source < systems.size(); ++source) {
    const std::size_t slot_before = data.size();
    try {
      DensityFittingScfData prepared;
      if (device_id >= 0) {
        if (cached_data_complete) {
          prepared = std::move(*(*prepared_cache)[source]);
        } else if (source >= batched_prepared.size() || !batched_prepared[source].has_value()) {
          outputs[source].status = preparation_status[source];
          continue;
        }
        if (!cached_data_complete) prepared = std::move(*batched_prepared[source]);
      } else {
        const core::System auxiliary =
            density_fitting_auxiliary_for_geometry(auxiliary_template, systems[source]);
        prepared = prepare_density_fitting_data(
            systems[source], auxiliary, options.density_fitting_relative_threshold, device_id);
      }
      if (source_indices.empty()) {
        nbf = prepared.raw.nbf;
        naux = prepared.raw.naux;
      } else if (prepared.raw.nbf != nbf || prepared.raw.naux != naux) {
        outputs[source].status = VIBEQC_STATUS_INVALID_ARGUMENT;
        continue;
      }
      const std::size_t occupied = static_cast<std::size_t>(systems[source].electron_count / 2);
      const Matrix orthogonalizer =
          symmetric_orthogonalizer(prepared.one_electron.overlap, prepared.one_electron.nbf);
      EigenResult initial_orbitals;
      Matrix density =
          prepare_initial_density(systems[source], prepared.one_electron, orthogonalizer, occupied,
                                  initial_densities[source], initial_orbitals);
      source_indices.push_back(source);
      data.push_back(std::move(prepared));
      orthogonalizers.push_back(orthogonalizer);
      densities.push_back(std::move(density));
      orbitals.push_back(std::move(initial_orbitals));
      diis.emplace_back(options.diis_history);
      previous_energies.push_back(std::numeric_limits<double>::infinity());
      outputs[source].scf.initial_density_used = initial_densities[source] != nullptr;
    } catch (const std::bad_alloc&) {
      source_indices.resize(slot_before);
      data.resize(slot_before);
      orthogonalizers.resize(slot_before);
      densities.resize(slot_before);
      orbitals.resize(slot_before);
      while (diis.size() > slot_before) diis.pop_back();
      previous_energies.resize(slot_before);
      outputs[source].status = VIBEQC_STATUS_OUT_OF_MEMORY;
    } catch (const std::invalid_argument&) {
      source_indices.resize(slot_before);
      data.resize(slot_before);
      orthogonalizers.resize(slot_before);
      densities.resize(slot_before);
      orbitals.resize(slot_before);
      while (diis.size() > slot_before) diis.pop_back();
      previous_energies.resize(slot_before);
      outputs[source].status = VIBEQC_STATUS_INVALID_ARGUMENT;
    } catch (...) {
      source_indices.resize(slot_before);
      data.resize(slot_before);
      orthogonalizers.resize(slot_before);
      densities.resize(slot_before);
      orbitals.resize(slot_before);
      while (diis.size() > slot_before) diis.pop_back();
      previous_energies.resize(slot_before);
      outputs[source].status = VIBEQC_STATUS_NUMERICAL_FAILURE;
    }
  }
  if (data.empty()) return outputs;

  CudaDensityFittingPlanPtr owned_plan(nullptr, &destroy_cuda_density_fitting_jk_plan);
  CudaDensityFittingJkPlan* plan = cached_plan == nullptr ? nullptr : *cached_plan;
  if (plan != nullptr && cuda_density_fitting_jk_plan_batch_size(plan) != data.size()) {
    // Item-level preparation may shrink a runnable subset after a warm cache
    // was created. Never submit vectors with a different fixed batch stride to
    // the old plan; discard it and rebuild for the surviving systems.
    destroy_cuda_density_fitting_jk_plan(plan);
    plan = nullptr;
    if (cached_plan != nullptr) *cached_plan = nullptr;
  }
  std::vector<CudaDensityFittingMetricDiagnostic> metric_diagnostics;
  try {
    if (plan == nullptr) {
      const std::size_t occupied =
          static_cast<std::size_t>(systems[source_indices.front()].electron_count / 2);
      std::vector<core::System> orbital_systems;
      std::vector<core::System> auxiliary_systems;
      orbital_systems.reserve(source_indices.size());
      auxiliary_systems.reserve(source_indices.size());
      for (const std::size_t source : source_indices) {
        orbital_systems.push_back(systems[source]);
        auxiliary_systems.push_back(
            density_fitting_auxiliary_for_geometry(auxiliary_template, systems[source]));
      }
      owned_plan = make_cuda_density_fitting_batch_plan(data, options, device_id, occupied,
                                                        &metric_diagnostics, &orbital_systems,
                                                        &auxiliary_systems);
      plan = owned_plan.get();
      if (cached_plan != nullptr) {
        *cached_plan = owned_plan.release();
      }
    }
  } catch (const std::bad_alloc&) {
    for (const std::size_t source : source_indices) {
      outputs[source].status = VIBEQC_STATUS_OUT_OF_MEMORY;
    }
    return outputs;
  } catch (const std::invalid_argument&) {
    for (const std::size_t source : source_indices) {
      outputs[source].status = VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    return outputs;
  } catch (...) {
    for (const std::size_t source : source_indices) {
      outputs[source].status = VIBEQC_STATUS_CUDA_ERROR;
    }
    return outputs;
  }
  if (output_diagnostics != nullptr) {
    for (std::size_t slot = 0; slot < metric_diagnostics.size(); ++slot) {
      metric_diagnostics[slot].system_index = source_indices[slot];
    }
    *output_diagnostics = metric_diagnostics;
  }
  if (options.density_fitting_memory_budget_bytes != 0) {
    for (DensityFittingScfData& item : data) {
      discard_density_fitting_tensor_storage(item);
    }
  }

  // A compatible bucket can advance every density without host staging.  The
  // device driver returns only scalar convergence records and the final
  // densities needed by the existing analytic-force oracle.  If a provider
  // rejects the batched eigensolve or does not converge all items, retain the
  // independently isolated host-orchestrated path below.
  {
    const std::size_t matrix_size = nbf * nbf;
    std::vector<double> hcore(data.size() * matrix_size);
    std::vector<double> orthogonalizer(data.size() * matrix_size);
    std::vector<double> initial_density(data.size() * matrix_size);
    std::vector<double> nuclear(data.size());
    std::vector<std::int32_t> occupied(data.size());
    for (std::size_t slot = 0; slot < data.size(); ++slot) {
      std::copy(data[slot].one_electron.hcore.begin(), data[slot].one_electron.hcore.end(),
                hcore.begin() + slot * matrix_size);
      std::copy(orthogonalizers[slot].begin(), orthogonalizers[slot].end(),
                orthogonalizer.begin() + slot * matrix_size);
      std::copy(densities[slot].begin(), densities[slot].end(),
                initial_density.begin() + slot * matrix_size);
      nuclear[slot] = data[slot].one_electron.nuclear_repulsion;
      occupied[slot] = static_cast<std::int32_t>(systems[source_indices[slot]].electron_count / 2);
    }
    std::vector<double> device_final_density;
    std::vector<CudaDensityFittingDeviceScfItem> device_records;
    std::string device_detail;
    const vibeqc_status device_status = run_cuda_density_fitting_rhf_device_scf(
        plan, hcore, orthogonalizer, initial_density, occupied, nuclear, options.max_iterations,
        options.energy_tolerance, options.density_tolerance, device_final_density, device_records,
        device_detail);
    if (device_status == VIBEQC_STATUS_OUT_OF_MEMORY && runtime::active_device_resource_ledger) {
      for (const auto source : source_indices) outputs[source].status = VIBEQC_STATUS_OUT_OF_MEMORY;
      return outputs;
    }
    const bool device_converged =
        device_status == VIBEQC_STATUS_SUCCESS && device_records.size() == data.size() &&
        std::all_of(device_records.begin(), device_records.end(),
                    [](const CudaDensityFittingDeviceScfItem& item) { return item.converged; });
    if (device_converged) {
      for (std::size_t slot = 0; slot < data.size(); ++slot) {
        const std::size_t source = source_indices[slot];
        densities[slot].assign(device_final_density.begin() + slot * matrix_size,
                               device_final_density.begin() + (slot + 1) * matrix_size);
        ScfResult& result = outputs[source].scf;
        result.iterations = device_records[slot].iterations;
        result.energy = device_records[slot].energy;
        result.energy_change = device_records[slot].energy_change;
        result.density_rms = device_records[slot].density_rms;
        result.converged = true;
        try {
          finalize_density_fitting_rhf(data[slot], orthogonalizers[slot],
                                       static_cast<std::size_t>(occupied[slot]), densities[slot],
                                       options, result, plan, slot);
          outputs[source].status = VIBEQC_STATUS_SUCCESS;
        } catch (const std::bad_alloc&) {
          outputs[source].status = VIBEQC_STATUS_OUT_OF_MEMORY;
        } catch (const std::invalid_argument&) {
          outputs[source].status = VIBEQC_STATUS_INVALID_ARGUMENT;
        } catch (...) {
          outputs[source].status = VIBEQC_STATUS_NUMERICAL_FAILURE;
        }
      }
      return outputs;
    }
  }

  const std::size_t matrix_size = nbf * nbf;
  std::vector<double> batch_density(data.size() * matrix_size);
  std::vector<bool> active(data.size(), true);
  std::size_t active_count = data.size();
  for (unsigned iteration = 1; iteration <= options.max_iterations && active_count != 0;
       ++iteration) {
    for (std::size_t slot = 0; slot < densities.size(); ++slot) {
      std::copy(densities[slot].begin(), densities[slot].end(),
                batch_density.begin() + slot * matrix_size);
    }
    std::vector<double> coulomb;
    std::vector<double> exchange;
    std::string detail;
    const vibeqc_status jk_status =
        execute_cuda_density_fitting_rhf_jk(plan, batch_density, coulomb, exchange, detail);
    if (jk_status != VIBEQC_STATUS_SUCCESS) {
      for (std::size_t slot = 0; slot < source_indices.size(); ++slot) {
        if (active[slot]) outputs[source_indices[slot]].status = jk_status;
        active[slot] = false;
      }
      break;
    }

    for (std::size_t slot = 0; slot < data.size(); ++slot) {
      if (!active[slot]) continue;
      const std::size_t source = source_indices[slot];
      try {
        Matrix fock = data[slot].one_electron.hcore;
        const double* j = coulomb.data() + slot * matrix_size;
        const double* k = exchange.data() + slot * matrix_size;
        for (std::size_t element = 0; element < matrix_size; ++element) {
          fock[element] += j[element] - 0.5 * k[element];
        }
        const double energy =
            electronic_energy(densities[slot], data[slot].one_electron.hcore, fock) +
            data[slot].one_electron.nuclear_repulsion;
        const Matrix residual =
            commutator_residual(fock, densities[slot], data[slot].one_electron.overlap, nbf);
        const Matrix effective_fock = diis[slot].update(fock, residual);
        orbitals[slot] = generalized_eigen(effective_fock, orthogonalizers[slot], nbf);
        Matrix next_density =
            density_from_orbitals(orbitals[slot].vectors, nbf,
                                  static_cast<std::size_t>(systems[source].electron_count / 2));
        ScfResult& result = outputs[source].scf;
        result.iterations = iteration;
        result.energy = energy;
        result.energy_change = std::isfinite(previous_energies[slot])
                                   ? std::abs(energy - previous_energies[slot])
                                   : std::numeric_limits<double>::infinity();
        result.density_rms = density_rms(next_density, densities[slot]);
        if (iteration > 1 && result.energy_change < options.energy_tolerance &&
            result.density_rms < options.density_tolerance) {
          densities[slot] = std::move(next_density);
          result.converged = true;
          active[slot] = false;
          --active_count;
        } else {
          previous_energies[slot] = energy;
          densities[slot] = std::move(next_density);
        }
      } catch (const std::bad_alloc&) {
        outputs[source].status = VIBEQC_STATUS_OUT_OF_MEMORY;
        active[slot] = false;
        --active_count;
      } catch (...) {
        outputs[source].status = VIBEQC_STATUS_NUMERICAL_FAILURE;
        active[slot] = false;
        --active_count;
      }
    }
  }

  for (std::size_t slot = 0; slot < source_indices.size(); ++slot) {
    const std::size_t source = source_indices[slot];
    ScfResult& result = outputs[source].scf;
    if (outputs[source].status != VIBEQC_STATUS_INTERNAL_ERROR) {
      continue;
    }
    if (!result.converged) {
      outputs[source].status = VIBEQC_STATUS_SCF_NOT_CONVERGED;
      continue;
    }
    try {
      finalize_density_fitting_rhf(data[slot], orthogonalizers[slot],
                                   static_cast<std::size_t>(systems[source].electron_count / 2),
                                   densities[slot], options, result, plan, slot);
      outputs[source].status = VIBEQC_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) {
      outputs[source].status = VIBEQC_STATUS_OUT_OF_MEMORY;
    } catch (const std::invalid_argument&) {
      outputs[source].status = VIBEQC_STATUS_INVALID_ARGUMENT;
    } catch (...) {
      outputs[source].status = VIBEQC_STATUS_NUMERICAL_FAILURE;
    }
  }
  return outputs;
}

std::vector<RhfBucketItem> run_uhf_density_fitting_cuda_bucket_impl(
    const std::vector<core::System>& systems, const std::optional<core::System>& auxiliary_template,
    const ScfOptions& options, const std::vector<const std::vector<double>*>& initial_densities,
    int device_id, std::vector<CudaDensityFittingMetricDiagnostic>* output_diagnostics,
    CudaDensityFittingJkPlan** cached_plan,
    std::vector<std::optional<DensityFittingScfData>>* prepared_cache) {
  if (options.hooks || options.strict_initial_density)
    throw std::invalid_argument("SCF proposal callbacks require the CPU reference backend");

  if (systems.size() != initial_densities.size()) {
    throw std::invalid_argument("CUDA density-fitting UHF bucket density count mismatch");
  }
  std::vector<RhfBucketItem> outputs(systems.size());
  if (systems.empty()) return outputs;

  std::vector<std::size_t> source_indices;
  std::vector<DensityFittingScfData> data;
  std::vector<Matrix> orthogonalizers;
  std::vector<Matrix> alpha_densities;
  std::vector<Matrix> beta_densities;
  std::vector<EigenResult> alpha_orbitals;
  std::vector<EigenResult> beta_orbitals;
  std::vector<Diis> diis;
  std::vector<double> previous_energies;
  if (prepared_cache != nullptr && prepared_cache->size() != systems.size()) {
    prepared_cache->assign(systems.size(), std::nullopt);
  }
  struct PreparedCacheGuard {
    std::vector<std::optional<DensityFittingScfData>>* cache{};
    std::vector<DensityFittingScfData>* data{};
    ~PreparedCacheGuard() {
      if (cache == nullptr || data == nullptr) return;
      if (cache->size() != data->size()) {
        cache->clear();
        return;
      }
      try {
        for (std::size_t slot = 0; slot < data->size(); ++slot) {
          (*cache)[slot] = std::move((*data)[slot]);
        }
      } catch (...) {
        cache->clear();
      }
    }
  } cache_guard{prepared_cache, &data};
  std::vector<vibeqc_status> preparation_status;
  std::vector<std::optional<DensityFittingScfData>> batched_prepared;
  const bool cached_data_complete =
      prepared_cache != nullptr && prepared_cache->size() == systems.size() &&
      std::all_of(
          prepared_cache->begin(), prepared_cache->end(), [&options, device_id](const auto& item) {
            return item.has_value() &&
                   one_electron_response_policy_matches(*item,
                                                        options.density_fitting_memory_budget_bytes,
                                                        options.compute_forces && device_id >= 0) &&
                   df_response_policy_matches(*item, options.density_fitting_memory_budget_bytes,
                                              options.density_fitting_relative_threshold,
                                              options.compute_forces && device_id >= 0);
          });
  if (prepared_cache != nullptr && !cached_data_complete && cached_plan != nullptr &&
      *cached_plan != nullptr) {
    // A budget change can switch resident tensors to a source-backed plan.
    // Rebuild both cache layers together so the new response data and plan
    // share the same storage contract and allocation limit.
    destroy_cuda_density_fitting_jk_plan(*cached_plan);
    *cached_plan = nullptr;
  }
  if (device_id >= 0 && !cached_data_complete) {
    batched_prepared = prepare_cuda_density_fitting_batch(
        systems, auxiliary_template, options.density_fitting_relative_threshold,
        options.density_fitting_memory_budget_bytes, device_id, preparation_status);
  }
  std::size_t nbf = 0;
  std::size_t naux = 0;
  for (std::size_t source = 0; source < systems.size(); ++source) {
    const std::size_t slot_before = data.size();
    try {
      DensityFittingScfData prepared;
      if (device_id >= 0) {
        if (cached_data_complete) {
          prepared = std::move(*(*prepared_cache)[source]);
        } else if (source >= batched_prepared.size() || !batched_prepared[source].has_value()) {
          outputs[source].status = preparation_status[source];
          continue;
        }
        if (!cached_data_complete) prepared = std::move(*batched_prepared[source]);
      } else {
        const core::System auxiliary =
            density_fitting_auxiliary_for_geometry(auxiliary_template, systems[source]);
        prepared = prepare_density_fitting_data(
            systems[source], auxiliary, options.density_fitting_relative_threshold, device_id);
      }
      if (source_indices.empty()) {
        nbf = prepared.raw.nbf;
        naux = prepared.raw.naux;
      } else if (prepared.raw.nbf != nbf || prepared.raw.naux != naux) {
        outputs[source].status = VIBEQC_STATUS_INVALID_ARGUMENT;
        continue;
      }
      const auto [alpha_occupied, beta_occupied] = spin_occupations(systems[source]);
      if (alpha_occupied > nbf || beta_occupied > nbf) {
        outputs[source].status = VIBEQC_STATUS_INVALID_ARGUMENT;
        continue;
      }
      const Matrix orthogonalizer =
          symmetric_orthogonalizer(prepared.one_electron.overlap, prepared.one_electron.nbf);
      EigenResult initial_alpha_orbitals;
      EigenResult initial_beta_orbitals;
      auto [alpha_density, beta_density] = prepare_initial_uhf_density(
          prepared.one_electron, orthogonalizer, alpha_occupied, beta_occupied,
          initial_densities[source], initial_alpha_orbitals, initial_beta_orbitals);
      source_indices.push_back(source);
      data.push_back(std::move(prepared));
      orthogonalizers.push_back(orthogonalizer);
      alpha_densities.push_back(std::move(alpha_density));
      beta_densities.push_back(std::move(beta_density));
      alpha_orbitals.push_back(std::move(initial_alpha_orbitals));
      beta_orbitals.push_back(std::move(initial_beta_orbitals));
      diis.emplace_back(options.diis_history);
      previous_energies.push_back(std::numeric_limits<double>::infinity());
      outputs[source].scf.initial_density_used = initial_densities[source] != nullptr;
    } catch (const std::bad_alloc&) {
      source_indices.resize(slot_before);
      data.resize(slot_before);
      orthogonalizers.resize(slot_before);
      alpha_densities.resize(slot_before);
      beta_densities.resize(slot_before);
      alpha_orbitals.resize(slot_before);
      beta_orbitals.resize(slot_before);
      while (diis.size() > slot_before) diis.pop_back();
      previous_energies.resize(slot_before);
      outputs[source].status = VIBEQC_STATUS_OUT_OF_MEMORY;
    } catch (const std::invalid_argument&) {
      source_indices.resize(slot_before);
      data.resize(slot_before);
      orthogonalizers.resize(slot_before);
      alpha_densities.resize(slot_before);
      beta_densities.resize(slot_before);
      alpha_orbitals.resize(slot_before);
      beta_orbitals.resize(slot_before);
      while (diis.size() > slot_before) diis.pop_back();
      previous_energies.resize(slot_before);
      outputs[source].status = VIBEQC_STATUS_INVALID_ARGUMENT;
    } catch (...) {
      source_indices.resize(slot_before);
      data.resize(slot_before);
      orthogonalizers.resize(slot_before);
      alpha_densities.resize(slot_before);
      beta_densities.resize(slot_before);
      alpha_orbitals.resize(slot_before);
      beta_orbitals.resize(slot_before);
      while (diis.size() > slot_before) diis.pop_back();
      previous_energies.resize(slot_before);
      outputs[source].status = VIBEQC_STATUS_NUMERICAL_FAILURE;
    }
  }
  if (data.empty()) return outputs;

  CudaDensityFittingPlanPtr owned_plan(nullptr, &destroy_cuda_density_fitting_jk_plan);
  CudaDensityFittingJkPlan* plan = cached_plan == nullptr ? nullptr : *cached_plan;
  if (plan != nullptr && cuda_density_fitting_jk_plan_batch_size(plan) != data.size()) {
    destroy_cuda_density_fitting_jk_plan(plan);
    plan = nullptr;
    if (cached_plan != nullptr) *cached_plan = nullptr;
  }
  std::vector<CudaDensityFittingMetricDiagnostic> metric_diagnostics;
  try {
    if (plan == nullptr) {
      const auto [alpha_occupied, beta_occupied] =
          spin_occupations(systems[source_indices.front()]);
      std::vector<core::System> orbital_systems;
      std::vector<core::System> auxiliary_systems;
      orbital_systems.reserve(source_indices.size());
      auxiliary_systems.reserve(source_indices.size());
      for (const std::size_t source : source_indices) {
        orbital_systems.push_back(systems[source]);
        auxiliary_systems.push_back(
            density_fitting_auxiliary_for_geometry(auxiliary_template, systems[source]));
      }
      owned_plan = make_cuda_density_fitting_batch_plan(
          data, options, device_id, std::max(alpha_occupied, beta_occupied), &metric_diagnostics,
          &orbital_systems, &auxiliary_systems);
      plan = owned_plan.get();
      if (cached_plan != nullptr) {
        *cached_plan = owned_plan.release();
      }
    }
  } catch (const std::bad_alloc&) {
    for (const std::size_t source : source_indices) {
      outputs[source].status = VIBEQC_STATUS_OUT_OF_MEMORY;
    }
    return outputs;
  } catch (const std::invalid_argument&) {
    for (const std::size_t source : source_indices) {
      outputs[source].status = VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    return outputs;
  } catch (...) {
    for (const std::size_t source : source_indices) {
      outputs[source].status = VIBEQC_STATUS_CUDA_ERROR;
    }
    return outputs;
  }
  if (output_diagnostics != nullptr) {
    for (std::size_t slot = 0; slot < metric_diagnostics.size(); ++slot) {
      metric_diagnostics[slot].system_index = source_indices[slot];
    }
    *output_diagnostics = metric_diagnostics;
  }

  if (options.density_fitting_memory_budget_bytes != 0) {
    for (DensityFittingScfData& item : data) {
      discard_density_fitting_tensor_storage(item);
    }
  }

  {
    const std::size_t matrix_size = nbf * nbf;
    std::vector<double> hcore(data.size() * matrix_size);
    std::vector<double> orthogonalizer(data.size() * matrix_size);
    std::vector<double> initial_alpha(data.size() * matrix_size);
    std::vector<double> initial_beta(data.size() * matrix_size);
    std::vector<double> nuclear(data.size());
    std::vector<std::int32_t> alpha_occupied(data.size());
    std::vector<std::int32_t> beta_occupied(data.size());
    for (std::size_t slot = 0; slot < data.size(); ++slot) {
      std::copy(data[slot].one_electron.hcore.begin(), data[slot].one_electron.hcore.end(),
                hcore.begin() + slot * matrix_size);
      std::copy(orthogonalizers[slot].begin(), orthogonalizers[slot].end(),
                orthogonalizer.begin() + slot * matrix_size);
      std::copy(alpha_densities[slot].begin(), alpha_densities[slot].end(),
                initial_alpha.begin() + slot * matrix_size);
      std::copy(beta_densities[slot].begin(), beta_densities[slot].end(),
                initial_beta.begin() + slot * matrix_size);
      nuclear[slot] = data[slot].one_electron.nuclear_repulsion;
      const auto occupations = spin_occupations(systems[source_indices[slot]]);
      alpha_occupied[slot] = static_cast<std::int32_t>(occupations.first);
      beta_occupied[slot] = static_cast<std::int32_t>(occupations.second);
    }
    std::vector<double> device_final_alpha;
    std::vector<double> device_final_beta;
    std::vector<CudaDensityFittingDeviceScfItem> device_records;
    std::string device_detail;
    const vibeqc_status device_status = run_cuda_density_fitting_uhf_device_scf(
        plan, hcore, orthogonalizer, initial_alpha, initial_beta, alpha_occupied, beta_occupied,
        nuclear, options.max_iterations, options.energy_tolerance, options.density_tolerance,
        device_final_alpha, device_final_beta, device_records, device_detail);
    if (device_status == VIBEQC_STATUS_OUT_OF_MEMORY && runtime::active_device_resource_ledger) {
      for (const auto source : source_indices) outputs[source].status = VIBEQC_STATUS_OUT_OF_MEMORY;
      return outputs;
    }
    const bool device_converged =
        device_status == VIBEQC_STATUS_SUCCESS && device_records.size() == data.size() &&
        std::all_of(device_records.begin(), device_records.end(),
                    [](const CudaDensityFittingDeviceScfItem& item) { return item.converged; });
    if (device_converged) {
      for (std::size_t slot = 0; slot < data.size(); ++slot) {
        const std::size_t source = source_indices[slot];
        alpha_densities[slot].assign(device_final_alpha.begin() + slot * matrix_size,
                                     device_final_alpha.begin() + (slot + 1) * matrix_size);
        beta_densities[slot].assign(device_final_beta.begin() + slot * matrix_size,
                                    device_final_beta.begin() + (slot + 1) * matrix_size);
        ScfResult& result = outputs[source].scf;
        result.iterations = device_records[slot].iterations;
        result.energy = device_records[slot].energy;
        result.energy_change = device_records[slot].energy_change;
        result.density_rms = device_records[slot].density_rms;
        result.converged = true;
        try {
          const auto occupations = spin_occupations(systems[source]);
          finalize_density_fitting_uhf(data[slot], orthogonalizers[slot], occupations.first,
                                       occupations.second, alpha_densities[slot],
                                       beta_densities[slot], options, result, plan, slot);
          outputs[source].status = VIBEQC_STATUS_SUCCESS;
        } catch (const std::bad_alloc&) {
          outputs[source].status = VIBEQC_STATUS_OUT_OF_MEMORY;
        } catch (const std::invalid_argument&) {
          outputs[source].status = VIBEQC_STATUS_INVALID_ARGUMENT;
        } catch (...) {
          outputs[source].status = VIBEQC_STATUS_NUMERICAL_FAILURE;
        }
      }
      return outputs;
    }
  }

  const std::size_t matrix_size = nbf * nbf;
  std::vector<double> batch_alpha(data.size() * matrix_size);
  std::vector<double> batch_beta(data.size() * matrix_size);
  std::vector<bool> active(data.size(), true);
  std::size_t active_count = data.size();
  for (unsigned iteration = 1; iteration <= options.max_iterations && active_count != 0;
       ++iteration) {
    for (std::size_t slot = 0; slot < alpha_densities.size(); ++slot) {
      std::copy(alpha_densities[slot].begin(), alpha_densities[slot].end(),
                batch_alpha.begin() + slot * matrix_size);
      std::copy(beta_densities[slot].begin(), beta_densities[slot].end(),
                batch_beta.begin() + slot * matrix_size);
    }
    std::vector<double> coulomb;
    std::vector<double> alpha_exchange;
    std::vector<double> beta_exchange;
    std::string detail;
    const vibeqc_status jk_status = execute_cuda_density_fitting_uhf_jk(
        plan, batch_alpha, batch_beta, coulomb, alpha_exchange, beta_exchange, detail);
    if (jk_status != VIBEQC_STATUS_SUCCESS) {
      for (std::size_t slot = 0; slot < source_indices.size(); ++slot) {
        if (active[slot]) outputs[source_indices[slot]].status = jk_status;
        active[slot] = false;
      }
      break;
    }
    for (std::size_t slot = 0; slot < data.size(); ++slot) {
      if (!active[slot]) continue;
      const std::size_t source = source_indices[slot];
      try {
        const auto [alpha_occupied, beta_occupied] = spin_occupations(systems[source]);
        Matrix alpha_fock = data[slot].one_electron.hcore;
        Matrix beta_fock = data[slot].one_electron.hcore;
        const double* j = coulomb.data() + slot * matrix_size;
        const double* ak = alpha_exchange.data() + slot * matrix_size;
        const double* bk = beta_exchange.data() + slot * matrix_size;
        for (std::size_t element = 0; element < matrix_size; ++element) {
          alpha_fock[element] += j[element] - ak[element];
          beta_fock[element] += j[element] - bk[element];
        }
        const double energy =
            uhf_electronic_energy(alpha_densities[slot], beta_densities[slot],
                                  data[slot].one_electron.hcore, alpha_fock, beta_fock) +
            data[slot].one_electron.nuclear_repulsion;
        const Matrix alpha_residual = commutator_residual(alpha_fock, alpha_densities[slot],
                                                          data[slot].one_electron.overlap, nbf);
        const Matrix beta_residual = commutator_residual(beta_fock, beta_densities[slot],
                                                         data[slot].one_electron.overlap, nbf);
        const Matrix effective_joined = diis[slot].update(
            concatenate(alpha_fock, beta_fock), concatenate(alpha_residual, beta_residual));
        std::tie(alpha_fock, beta_fock) = split_spin_matrices(effective_joined, matrix_size);
        alpha_orbitals[slot] = generalized_eigen(alpha_fock, orthogonalizers[slot], nbf);
        beta_orbitals[slot] = generalized_eigen(beta_fock, orthogonalizers[slot], nbf);
        Matrix next_alpha =
            density_from_orbitals(alpha_orbitals[slot].vectors, nbf, alpha_occupied, 1.0);
        Matrix next_beta =
            density_from_orbitals(beta_orbitals[slot].vectors, nbf, beta_occupied, 1.0);
        ScfResult& result = outputs[source].scf;
        result.iterations = iteration;
        result.energy = energy;
        result.energy_change = std::isfinite(previous_energies[slot])
                                   ? std::abs(energy - previous_energies[slot])
                                   : std::numeric_limits<double>::infinity();
        result.density_rms = density_rms(concatenate(next_alpha, next_beta),
                                         concatenate(alpha_densities[slot], beta_densities[slot]));
        if (iteration > 1 && result.energy_change < options.energy_tolerance &&
            result.density_rms < options.density_tolerance) {
          alpha_densities[slot] = std::move(next_alpha);
          beta_densities[slot] = std::move(next_beta);
          result.converged = true;
          active[slot] = false;
          --active_count;
        } else {
          previous_energies[slot] = energy;
          alpha_densities[slot] = std::move(next_alpha);
          beta_densities[slot] = std::move(next_beta);
        }
      } catch (const std::bad_alloc&) {
        outputs[source].status = VIBEQC_STATUS_OUT_OF_MEMORY;
        active[slot] = false;
        --active_count;
      } catch (...) {
        outputs[source].status = VIBEQC_STATUS_NUMERICAL_FAILURE;
        active[slot] = false;
        --active_count;
      }
    }
  }

  for (std::size_t slot = 0; slot < source_indices.size(); ++slot) {
    const std::size_t source = source_indices[slot];
    ScfResult& result = outputs[source].scf;
    if (outputs[source].status != VIBEQC_STATUS_INTERNAL_ERROR) {
      continue;
    }
    if (!result.converged) {
      outputs[source].status = VIBEQC_STATUS_SCF_NOT_CONVERGED;
      continue;
    }
    try {
      const auto [alpha_occupied, beta_occupied] = spin_occupations(systems[source]);
      finalize_density_fitting_uhf(data[slot], orthogonalizers[slot], alpha_occupied, beta_occupied,
                                   alpha_densities[slot], beta_densities[slot], options, result,
                                   plan, slot);
      outputs[source].status = VIBEQC_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) {
      outputs[source].status = VIBEQC_STATUS_OUT_OF_MEMORY;
    } catch (...) {
      outputs[source].status = VIBEQC_STATUS_NUMERICAL_FAILURE;
    }
  }
  return outputs;
}

#endif  // VIBEQC_HAS_CUDA

#if VIBEQC_HAS_CUDA

ScfResult run_rhf_density_fitting_cuda(const core::System& system,
                                       const core::System& auxiliary_system,
                                       const ScfOptions& options, int device_id,
                                       const std::vector<double>* initial_density) {
  return run_rhf_density_fitting_cuda_impl(system, auxiliary_system, options, device_id,
                                           initial_density);
}

ScfResult run_uhf_density_fitting_cuda(const core::System& system,
                                       const core::System& auxiliary_system,
                                       const ScfOptions& options, int device_id,
                                       const std::vector<double>* initial_density) {
  return run_uhf_density_fitting_cuda_impl(system, auxiliary_system, options, device_id,
                                           initial_density);
}

std::vector<RhfBucketItem> run_rhf_density_fitting_cuda_bucket(
    const std::vector<core::System>& systems, const std::optional<core::System>& auxiliary_template,
    const ScfOptions& options, const std::vector<const std::vector<double>*>& initial_densities,
    int device_id, std::vector<CudaDensityFittingMetricDiagnostic>* diagnostics) {
  return run_rhf_density_fitting_cuda_bucket_impl(systems, auxiliary_template, options,
                                                  initial_densities, device_id, diagnostics,
                                                  nullptr, nullptr);
}

std::vector<RhfBucketItem> run_rhf_density_fitting_cuda_bucket_cached(
    CudaDensityFittingJkPlan** plan, const std::vector<core::System>& systems,
    const std::optional<core::System>& auxiliary_template, const ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities, int device_id,
    std::vector<CudaDensityFittingMetricDiagnostic>* diagnostics,
    std::vector<std::optional<DensityFittingScfData>>* prepared_cache) {
  return run_rhf_density_fitting_cuda_bucket_impl(systems, auxiliary_template, options,
                                                  initial_densities, device_id, diagnostics, plan,
                                                  prepared_cache);
}

std::vector<RhfBucketItem> run_uhf_density_fitting_cuda_bucket(
    const std::vector<core::System>& systems, const std::optional<core::System>& auxiliary_template,
    const ScfOptions& options, const std::vector<const std::vector<double>*>& initial_densities,
    int device_id, std::vector<CudaDensityFittingMetricDiagnostic>* diagnostics) {
  return run_uhf_density_fitting_cuda_bucket_impl(systems, auxiliary_template, options,
                                                  initial_densities, device_id, diagnostics,
                                                  nullptr, nullptr);
}

std::vector<RhfBucketItem> run_uhf_density_fitting_cuda_bucket_cached(
    CudaDensityFittingJkPlan** plan, const std::vector<core::System>& systems,
    const std::optional<core::System>& auxiliary_template, const ScfOptions& options,
    const std::vector<const std::vector<double>*>& initial_densities, int device_id,
    std::vector<CudaDensityFittingMetricDiagnostic>* diagnostics,
    std::vector<std::optional<DensityFittingScfData>>* prepared_cache) {
  return run_uhf_density_fitting_cuda_bucket_impl(systems, auxiliary_template, options,
                                                  initial_densities, device_id, diagnostics, plan,
                                                  prepared_cache);
}

#endif  // VIBEQC_HAS_CUDA

#if !VIBEQC_HAS_CUDA
ScfResult run_cuda_independent_fock_strategy(const core::System&, const core::System*,
                                             const ScfOptions&, int, const std::vector<double>*) {
  throw std::runtime_error("CUDA Fock providers are unavailable in this build");
}

// Keep diagnostics identical to the CUDA backend's small persistent-ERI
// policy without exposing an implementation tuning threshold through the ABI.
constexpr std::size_t kDiagnosticPersistentEriAoLimit = 16;

CudaRhfBasisLayoutStats inspect_rhf_cuda_basis_layout(const std::vector<core::System>& systems) {
  if (systems.empty()) {
    throw std::invalid_argument("a CUDA RHF basis layout requires systems");
  }
  const std::size_t nbf = molecule::ao_count(systems.front());
  const std::size_t direct_nbf = molecule::cartesian_ao_count(systems.front());
  std::size_t shell_count = 0;
  std::size_t shell_pair_count = 0;
  std::size_t shell_quartet_count = 0;
  std::size_t unique_primitives = 0;
  std::size_t expanded_primitives = 0;
  for (const core::System& system : systems) {
    if (molecule::ao_count(system) != nbf || molecule::cartesian_ao_count(system) != direct_nbf) {
      throw std::invalid_argument("systems do not belong to one CUDA RHF bucket");
    }
    shell_count += system.shells.size();
    const std::size_t system_shell_pairs = system.shells.size() * (system.shells.size() + 1) / 2;
    shell_pair_count += system_shell_pairs;
    shell_quartet_count += system_shell_pairs * (system_shell_pairs + 1) / 2;
    for (const core::Shell& shell : system.shells) {
      unique_primitives += shell.primitives.size();
      expanded_primitives +=
          molecule::cartesian_count(shell.angular_momentum) * shell.primitives.size();
    }
  }
  const std::size_t ao_count = systems.size() * nbf;
  const std::size_t direct_ao_count = systems.size() * direct_nbf;
  const std::size_t device_basis_bytes =
      (systems.size() + 1) * sizeof(std::int64_t) +
      shell_count * (sizeof(std::int32_t) + sizeof(std::uint8_t)) +
      3 * (shell_count + 1) * sizeof(std::int64_t) +
      2 * (systems.size() + 1) * sizeof(std::int64_t) +
      shell_pair_count * 3 * sizeof(std::int32_t) +
      ao_count * (sizeof(std::int32_t) + sizeof(std::uint8_t) +
                  3 * molecule::kMaximumAoExpansionTerms * sizeof(std::uint8_t) +
                  molecule::kMaximumAoExpansionTerms * sizeof(double)) +
      direct_ao_count * (sizeof(std::int32_t) + 3 * sizeof(std::uint8_t) + sizeof(double)) +
      (direct_nbf == nbf || nbf <= kDiagnosticPersistentEriAoLimit
           ? 0
           : systems.size() * nbf * direct_nbf * sizeof(double)) +
      unique_primitives * 2 * sizeof(double);
  return {systems.size(),
          shell_count,
          shell_pair_count,
          shell_quartet_count,
          ao_count,
          unique_primitives,
          expanded_primitives,
          device_basis_bytes,
          detail::direct_topology_requires_bounded_streaming(shell_quartet_count),
          detail::direct_topology_requires_bounded_streaming(shell_quartet_count)
              ? detail::kBoundedDirectQueueCapacity
              : 0};
}

ScfResult run_rhf_cuda(const core::System&, const ScfOptions&, int, const std::vector<double>*) {
  throw std::runtime_error("the library was built without CUDA support");
}

std::size_t hf_cuda_owned_device_bytes(const CudaRhfBucketPlan*) noexcept { return 0; }

ScfResult run_uhf_cuda(const core::System&, const ScfOptions&, int, const std::vector<double>*) {
  throw std::runtime_error("the library was built without CUDA support");
}

ScfResult run_rhf_density_fitting_cuda(const core::System&, const core::System&, const ScfOptions&,
                                       int, const std::vector<double>*) {
  throw std::runtime_error("the library was built without CUDA support");
}

ScfResult run_uhf_density_fitting_cuda(const core::System&, const core::System&, const ScfOptions&,
                                       int, const std::vector<double>*) {
  throw std::runtime_error("the library was built without CUDA support");
}

std::vector<RhfBucketItem> run_rhf_density_fitting_cuda_bucket(
    const std::vector<core::System>& systems, const std::optional<core::System>&, const ScfOptions&,
    const std::vector<const std::vector<double>*>&, int,
    std::vector<CudaDensityFittingMetricDiagnostic>* diagnostics) {
  if (diagnostics != nullptr) diagnostics->clear();
  std::vector<RhfBucketItem> outputs(systems.size());
  for (RhfBucketItem& output : outputs) {
    output.status = VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
  return outputs;
}

std::vector<RhfBucketItem> run_rhf_density_fitting_cuda_bucket_cached(
    CudaDensityFittingJkPlan**, const std::vector<core::System>& systems,
    const std::optional<core::System>&, const ScfOptions&,
    const std::vector<const std::vector<double>*>&, int,
    std::vector<CudaDensityFittingMetricDiagnostic>* diagnostics,
    std::vector<std::optional<DensityFittingScfData>>*) {
  if (diagnostics != nullptr) diagnostics->clear();
  std::vector<RhfBucketItem> outputs(systems.size());
  for (RhfBucketItem& output : outputs) {
    output.status = VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
  return outputs;
}

std::vector<RhfBucketItem> run_uhf_density_fitting_cuda_bucket(
    const std::vector<core::System>& systems, const std::optional<core::System>&, const ScfOptions&,
    const std::vector<const std::vector<double>*>&, int,
    std::vector<CudaDensityFittingMetricDiagnostic>* diagnostics) {
  if (diagnostics != nullptr) diagnostics->clear();
  std::vector<RhfBucketItem> outputs(systems.size());
  for (RhfBucketItem& output : outputs) {
    output.status = VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
  return outputs;
}

std::vector<RhfBucketItem> run_uhf_density_fitting_cuda_bucket_cached(
    CudaDensityFittingJkPlan**, const std::vector<core::System>& systems,
    const std::optional<core::System>&, const ScfOptions&,
    const std::vector<const std::vector<double>*>&, int,
    std::vector<CudaDensityFittingMetricDiagnostic>* diagnostics,
    std::vector<std::optional<DensityFittingScfData>>*) {
  if (diagnostics != nullptr) diagnostics->clear();
  std::vector<RhfBucketItem> outputs(systems.size());
  for (RhfBucketItem& output : outputs) {
    output.status = VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
  return outputs;
}

std::vector<RhfBucketItem> run_rhf_cuda_bucket(const std::vector<core::System>& systems,
                                               const ScfOptions&,
                                               const std::vector<const std::vector<double>*>&, int,
                                               bool, bool) {
  std::vector<RhfBucketItem> outputs(systems.size());
  for (RhfBucketItem& output : outputs) {
    output.status = VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
  return outputs;
}

std::vector<RhfBucketItem> run_rhf_cuda_bucket_cached(
    CudaRhfBucketPlan**, const std::vector<core::System>& systems, const ScfOptions&,
    const std::vector<const std::vector<double>*>&, int, bool, bool) {
  std::vector<RhfBucketItem> outputs(systems.size());
  for (RhfBucketItem& output : outputs) {
    output.status = VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
  return outputs;
}

std::vector<RhfBucketItem> run_uhf_cuda_bucket(const std::vector<core::System>& systems,
                                               const ScfOptions&,
                                               const std::vector<const std::vector<double>*>&, int,
                                               bool, bool) {
  std::vector<RhfBucketItem> outputs(systems.size());
  for (RhfBucketItem& output : outputs) {
    output.status = VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
  return outputs;
}

std::vector<RhfBucketItem> run_uhf_cuda_bucket_cached(
    CudaRhfBucketPlan**, const std::vector<core::System>& systems, const ScfOptions&,
    const std::vector<const std::vector<double>*>&, int, bool, bool) {
  std::vector<RhfBucketItem> outputs(systems.size());
  for (RhfBucketItem& output : outputs) {
    output.status = VIBEQC_STATUS_NOT_IMPLEMENTED;
  }
  return outputs;
}

void destroy_rhf_cuda_bucket_plan(CudaRhfBucketPlan*) noexcept {}

void set_rhf_cuda_bucket_warm_start_updates(CudaRhfBucketPlan*, bool) noexcept {}

void clear_rhf_cuda_bucket_warm_starts(CudaRhfBucketPlan*) noexcept {}

bool get_rhf_cuda_shell_class_profile(const CudaRhfBucketPlan*,
                                      CudaRhfShellClassProfile&) noexcept {
  return false;
}

bool get_rhf_cuda_ppps_queue_profile(const CudaRhfBucketPlan*, CudaPppsQueueProfile&) noexcept {
  return false;
}

bool get_rhf_cuda_eigensolver_diagnostic(const CudaRhfBucketPlan*,
                                         CudaEigensolverDiagnostic&) noexcept {
  return false;
}

bool get_rhf_cuda_inactive_eigensolver_profile(const CudaRhfBucketPlan*,
                                               CudaInactiveEigensolverProfile&) noexcept {
  return false;
}
#endif

}  // namespace vibeqc::scf
