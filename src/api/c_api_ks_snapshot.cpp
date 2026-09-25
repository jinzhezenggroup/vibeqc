#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "api/ks_snapshot.hpp"
#include "dft/xc.hpp"
#include "dft/xc_point.hpp"
#include "dft/xc_point_response.hpp"
#include "integrals/ecp.hpp"
#include "integrals/ecp_cuda.hpp"
#include "methods/dft_method.hpp"
#if VIBEQC_HAS_CUDA
#include "dft/cuda_xc.hpp"
#include "runtime/cuda_resources.cuh"
#endif

struct vibeqc_ks_snapshot {
  std::size_t index{};
  vibeqc::dft::CudaKsFinalStateToken token;
  std::vector<double> values;
  double energy{};
  bool all_electron{};
};

struct vibeqc_ks_xc_response {
#if VIBEQC_HAS_CUDA
  std::size_t index{}, count{}, device_bytes{};
  int device{};
  std::uint64_t generation{}, input_bytes{}, extra_synchronizations{1};
  std::uint64_t export_bytes{}, export_reads{}, export_synchronizations{};
  vibeqc::dft::CudaKsFinalStateToken token;
  // Reverse destruction drains the plan before freeing its borrowed arena;
  // every buffer is also tied to the still-live stream on exceptional exit.
  vibeqc::runtime::OwnedCudaStream stream;
  vibeqc::runtime::OwnedCudaBuffer<std::byte> arena;
  vibeqc::runtime::OwnedCudaBuffer<double> density, direction;
  std::unique_ptr<vibeqc::dft::CudaXcPlan> plan;
#endif
};

namespace {
vibeqc_status check_current(const vibeqc_batch& batch, std::size_t index,
                            const vibeqc::dft::CudaKsFinalStateToken& token) {
  vibeqc::dft::CudaKsFinalStateToken current;
  std::string detail;
  const auto status =
      vibeqc::methods::detail::dft_final_state_token(*batch.plan, index, current, detail);
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  return current == token ? VIBEQC_STATUS_SUCCESS : VIBEQC_STATUS_INVALID_ARGUMENT;
}
vibeqc_status check_current(const vibeqc_batch& batch, const vibeqc_ks_snapshot& snapshot) {
  return check_current(batch, snapshot.index, snapshot.token);
}
}  // namespace

extern "C" {
vibeqc_status vibeqc_ks_snapshot_create_v1(vibeqc_batch* batch, std::size_t index,
                                           vibeqc_ks_snapshot** output, std::uint64_t* metadata,
                                           std::size_t metadata_count) {
  if (output) *output = nullptr;
  if (!batch || !output || !metadata || metadata_count != 16) return VIBEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  try {
    auto result = std::make_unique<vibeqc_ks_snapshot>();
    result->index = index;
    std::string detail;
    auto status =
        vibeqc::methods::detail::dft_final_state_token(*batch->plan, index, result->token, detail);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    vibeqc::methods::detail::KsDerivativeSnapshot source;
    status = vibeqc::methods::detail::read_dft_derivative_state(*batch->plan, index, result->token,
                                                                source, detail);
    if (status != VIBEQC_STATUS_SUCCESS) {
      batch->context->last_detail = detail;
      return status;
    }
    const auto& state = source.state;
    result->all_electron = source.system.ecp_terms.empty() &&
                           std::all_of(source.system.atoms.begin(), source.system.atoms.end(),
                                       [](const auto& atom) { return atom.ecp_core == 0; });
    result->energy = state.components.total();
    const auto& identity = state.identity;
    const auto n = state.orbitals.at(0).values.size();
    auto& values = result->values;
    const auto append = [&](const auto& array) {
      values.insert(values.end(), array.begin(), array.end());
    };
    // Fixed private wire layout: residual/charge, atoms, D/F/C/epsilon/f/W,
    // then the actual metric, packed AO basis and explicit quadrature.
    values = {state.diagnostic.physical_residual, static_cast<double>(source.system.charge)};
    for (const auto& atom : source.system.atoms) {
      values.push_back(atom.atomic_number);
      append(atom.position);
    }
    for (const auto& matrix : state.density) append(matrix);
    for (const auto& matrix : state.fock) append(matrix);
    for (const auto& frame : state.orbitals) append(frame.vectors);
    for (const auto& frame : state.orbitals) append(frame.values);
    for (const auto occupied : identity.determinant.occupied)
      for (std::size_t orbital = 0; orbital < n; ++orbital)
        values.push_back(orbital < occupied ? (identity.model.spins == 1 ? 2.0 : 1.0) : 0.0);
    for (const auto& matrix : state.weighted_density) append(matrix);
    append(source.overlap);
    append(source.packed_basis);
    append(source.points);
    append(source.weights);
    append(source.grid_owners);
    const bool cpu = identity.determinant.model.backend == vibeqc::scf::FockBackend::Cpu;
    const auto& exchange = identity.determinant.model.spec.exchange;
    const bool composition = identity.model.semilocal_exchange_scale != 1 ||
                             identity.model.semilocal_correlation_scale != 1 || exchange.present;
    // CPU v2 is unchanged. CUDA v3 adds the same actual prescription/measures
    // suffix while retaining its visible device ordinal. Legacy CUDA v1 reads
    // remain supported by Python, but cannot qualify a weight derivative.
    {
      const auto& grid = identity.model.grid;
      for (double value :
           {static_cast<double>(grid.version), static_cast<double>(grid.radial_points),
            static_cast<double>(grid.angular_polar), static_cast<double>(grid.angular_azimuth),
            static_cast<double>(grid.partition_iterations), grid.coincident_tolerance})
        values.push_back(value);
      append(grid.element_radii);
      append(source.atomic_weights);
      if (!cpu) {
        values.push_back(static_cast<double>(source.export_d2h_bytes));
        values.push_back(static_cast<double>(source.export_reads));
        values.push_back(static_cast<double>(source.export_synchronizations));
      }
    }
    // ECP v4 (CPU) / v5 (CUDA) bind the live Hamiltonian. Hybrid
    // composition uses CPU v6, or v7 when combined with ECP. CUDA hybrids
    // remain fail-closed before snapshot publication.
    const bool ecp = !source.system.ecp_terms.empty();
    if (ecp) {
      for (const auto& atom : source.system.atoms) values.push_back(atom.ecp_core);
      values.push_back(static_cast<double>(source.system.ecp_terms.size()));
      for (const auto& term : source.system.ecp_terms)
        for (double value :
             {static_cast<double>(term.atom_index), static_cast<double>(term.channel),
              static_cast<double>(term.power), term.exponent, term.coefficient})
          values.push_back(value);
    }
    if (composition) {
      values.push_back(identity.model.semilocal_exchange_scale);
      values.push_back(identity.model.semilocal_correlation_scale);
      values.push_back(exchange.present ? exchange.coefficient : 0.0);
    }
    const auto wire_version =
        cpu ? (composition ? (ecp ? 7U : 6U) : (ecp ? 4U : 2U)) : (ecp ? 5U : 3U);
    const std::array<std::uint64_t, 16> info{
        wire_version,
        n,
        identity.model.spins,
        source.system.atoms.size(),
        source.packed_basis.size(),
        source.weights.size(),
        identity.model.functional,
        identity.model.scf_domain_version,
        identity.model.owner,
        identity.determinant.solve_epoch,
        identity.determinant.factor.density_generation,
        identity.determinant.factor.orbital_generation,
        static_cast<std::uint64_t>(identity.model.device),
        static_cast<std::uint64_t>(source.system.basis_representation),
        source.system.multiplicity,
        values.size()};
    // Even publication uses the same token gate as a later derivative read.
    status = check_current(*batch, *result);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    std::copy(info.begin(), info.end(), metadata);
    *output = result.release();
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&batch->context->last_detail);
  }
}

vibeqc_status vibeqc_ks_snapshot_wb97mv_model_v1(const vibeqc_batch* batch,
                                                 const vibeqc_ks_snapshot* snapshot, double* values,
                                                 std::size_t count) {
  if (!batch || !snapshot || !values || count != 9) return VIBEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  try {
    const auto status = check_current(*batch, *snapshot);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    const auto& identity = snapshot->token.identity;
    const auto& model = identity.model;
    const auto& primary = identity.determinant.model;
    if (!snapshot->all_electron || primary.backend != vibeqc::scf::FockBackend::Cpu ||
        model.functional != 4 || !model.range_correction || !model.nonlocal_correlation)
      return VIBEQC_STATUS_NOT_IMPLEMENTED;
    const double spin_factor = model.spins == 1 ? -0.5 : -1.0;
    const double short_exchange = primary.spec.exchange.coefficient / spin_factor;
    const auto& range = *model.range_correction;
    const auto& nonlocal = *model.nonlocal_correlation;
    const std::array<double, 9> proof{
        short_exchange,
        short_exchange + range.spec.exchange.coefficient / spin_factor,
        range.spec.exchange.omega,
        static_cast<double>(nonlocal.variant),
        nonlocal.b,
        nonlocal.c,
        nonlocal.coefficient,
        static_cast<double>(model.nonlocal_density_domain),
        primary.screening_tolerance};
    std::copy(proof.begin(), proof.end(), values);
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&batch->context->last_detail);
  }
}

vibeqc_status vibeqc_ks_snapshot_hamiltonian_v1(const vibeqc_batch* batch,
                                                const vibeqc_ks_snapshot* snapshot,
                                                std::uint32_t* kind) {
  if (!batch || !snapshot || !kind) return VIBEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  try {
    const auto status = check_current(*batch, *snapshot);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    *kind = snapshot->all_electron ? 0U : 1U;
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&batch->context->last_detail);
  }
}

vibeqc_status vibeqc_ks_xc_response_create_v1(vibeqc_batch* batch,
                                              const vibeqc_ks_snapshot* snapshot,
                                              std::size_t tile_points, std::size_t budget_bytes,
                                              vibeqc_ks_xc_response** output) {
  if (output) *output = nullptr;
  if (!batch || !snapshot || !output || !budget_bytes) return VIBEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
#if VIBEQC_HAS_CUDA
  try {
    auto status = check_current(*batch, *snapshot);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    const auto& identity = snapshot->token.identity;
    if (identity.determinant.model.backend != vibeqc::scf::FockBackend::Cuda ||
        identity.model.functional > 1U || !snapshot->all_electron)
      return VIBEQC_STATUS_NOT_IMPLEMENTED;
    vibeqc::methods::detail::KsDerivativeSnapshot source;
    std::string detail;
    status = vibeqc::methods::detail::read_dft_derivative_state(*batch->plan, snapshot->index,
                                                                snapshot->token, source, detail);
    if (status != VIBEQC_STATUS_SUCCESS) {
      batch->context->last_detail = detail;
      return status;
    }
    const vibeqc::dft::AoBasis basis(source.system);
    if (basis.packed != source.packed_basis)
      throw std::invalid_argument("CUDA response packed basis differs from native state");
    const auto layout = vibeqc::dft::cuda_xc_layout_shape(
        basis.natom, basis.nprimitive, basis.nao, source.weights.size(), identity.model.functional,
        identity.model.spins == 2, tile_points, true);
    using vibeqc::runtime::size_mul;
    const auto count = size_mul(size_mul(layout.spins, layout.nao, "XC response overflow"),
                                layout.nao, "XC response overflow");
    const auto bytes = vibeqc::runtime::size_add(
        layout.device_bytes, size_mul(count, 2 * sizeof(double), "XC response overflow"),
        "XC response overflow");
    if (bytes > budget_bytes) throw std::bad_alloc();
    std::vector<double> density;
    for (const auto& spin : source.state.density)
      density.insert(density.end(), spin.begin(), spin.end());
    if (density.size() != count) throw std::invalid_argument("CUDA response spin shape mismatch");
    auto result = std::make_unique<vibeqc_ks_xc_response>();
    result->index = snapshot->index;
    result->token = snapshot->token;
    result->device = identity.model.device;
    result->count = count;
    result->device_bytes = bytes;
    result->export_bytes = source.export_d2h_bytes;
    result->export_reads = source.export_reads;
    result->export_synchronizations = source.export_synchronizations;
    vibeqc::runtime::CudaDeviceScope device(result->device);
    result->stream.create(result->device);
    const auto stream = result->stream.get();
    result->arena.allocate(result->device, layout.device_bytes, stream);
    result->density.allocate(result->device, count, stream);
    result->direction.allocate(result->device, count, stream);
    result->plan = std::make_unique<vibeqc::dft::CudaXcPlan>(
        layout, source.packed_basis, source.points, source.weights, result->arena.get(),
        layout.device_bytes, stream);
    vibeqc::runtime::cuda_resource_check(cudaMemcpyAsync(result->density.get(), density.data(),
                                                         count * sizeof(double),
                                                         cudaMemcpyHostToDevice, stream));
    result->stream.synchronize();
    status = check_current(*batch, *snapshot);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    *output = result.release();
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&batch->context->last_detail);
  }
#else
  (void)tile_points;
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

vibeqc_status vibeqc_ks_xc_response_apply_v1(vibeqc_batch* batch, vibeqc_ks_xc_response* response,
                                             const double* direction, std::size_t count,
                                             double* output, std::size_t output_count) {
  if (!batch || !response || !direction || !output) return VIBEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
#if VIBEQC_HAS_CUDA
  try {
    auto status = check_current(*batch, response->index, response->token);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    if (count != response->count || output_count != count) return VIBEQC_STATUS_INVALID_ARGUMENT;
    vibeqc::runtime::CudaDeviceScope device(response->device);
    const auto stream = response->stream.get();
    // On every failure the Python caller discards its output. Publication is
    // staged below until numerical validity and the live token both hold.
    vibeqc::runtime::cuda_resource_check(cudaMemcpyAsync(response->direction.get(), direction,
                                                         count * sizeof(double),
                                                         cudaMemcpyHostToDevice, stream));
    response->input_bytes += count * sizeof(double);
    response->plan->enqueue_response(response->density.get(), response->direction.get(), count,
                                     ++response->generation);
    const auto scalars = response->plan->read_scalars(response->generation);
    if (scalars.error) throw std::invalid_argument("invalid CUDA XC response direction/domain");
    const auto result = response->plan->download_potential(response->generation);
    status = check_current(*batch, response->index, response->token);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    std::copy(result.begin(), result.end(), output);
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    // A failed upload/launch must not outlive a borrowed host input buffer.
    ++response->extra_synchronizations;
    try {
      response->stream.synchronize();
    } catch (...) {
    }
    return vibeqc::api::map_exception(&batch->context->last_detail);
  }
#else
  (void)count;
  (void)output_count;
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

vibeqc_status vibeqc_ks_xc_response_diagnostic_v1(const vibeqc_ks_xc_response* response,
                                                  std::uint64_t* values, std::size_t count) {
  if (!response || !values || count != 12) return VIBEQC_STATUS_INVALID_ARGUMENT;
#if VIBEQC_HAS_CUDA
  const auto& t = response->plan->transfers();
  const auto& l = response->plan->layout();
  const std::array<std::uint64_t, 12> result{response->device_bytes,
                                             t.setup_h2d_bytes + response->count * sizeof(double),
                                             response->input_bytes,
                                             t.output_d2h_bytes,
                                             t.synchronizations + response->extra_synchronizations,
                                             t.evaluations,
                                             l.spins,
                                             l.nao,
                                             l.npoint,
                                             response->export_bytes,
                                             response->export_reads,
                                             response->export_synchronizations};
  std::copy(result.begin(), result.end(), values);
  return VIBEQC_STATUS_SUCCESS;
#else
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
#endif
}

void vibeqc_ks_xc_response_destroy_v1(vibeqc_ks_xc_response* response) { delete response; }

vibeqc_status vibeqc_ks_snapshot_check_v1(const vibeqc_batch* batch,
                                          const vibeqc_ks_snapshot* snapshot) {
  if (!batch || !snapshot) return VIBEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  try {
    return check_current(*batch, *snapshot);
  } catch (...) {
    return vibeqc::api::map_exception(&batch->context->last_detail);
  }
}

vibeqc_status vibeqc_ks_snapshot_copy_v1(const vibeqc_batch* batch,
                                         const vibeqc_ks_snapshot* snapshot, double* values,
                                         std::size_t count) {
  if (!batch || !snapshot || !values || count != snapshot->values.size())
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  try {
    const auto status = check_current(*batch, *snapshot);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    std::copy(snapshot->values.begin(), snapshot->values.end(), values);
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&batch->context->last_detail);
  }
}

// Diagnostic-only reuse of the backend-specific ECP provider. Return separate
// local/projector all-center derivatives; generated TensorIR owns their D weights.
// Re-read the actual owner under its token instead of accepting a caller's ECP.
vibeqc_status vibeqc_ks_snapshot_ecp_derivatives_v1(vibeqc_batch* batch,
                                                    const vibeqc_ks_snapshot* snapshot,
                                                    double* values, std::size_t count) {
  if (!batch || !snapshot || !values) return VIBEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  try {
    auto status = check_current(*batch, *snapshot);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    vibeqc::methods::detail::KsDerivativeSnapshot source;
    std::string detail;
    status = vibeqc::methods::detail::read_dft_derivative_state(*batch->plan, snapshot->index,
                                                                snapshot->token, source, detail);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    if (source.system.ecp_terms.empty()) return VIBEQC_STATUS_INVALID_ARGUMENT;
    const auto n = source.state.orbitals.at(0).values.size();
    const auto atoms = source.system.atoms.size();
    if (!n || atoms > std::numeric_limits<std::size_t>::max() / 6 / n / n ||
        count != 6 * atoms * n * n)
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    vibeqc::integrals::EcpData ecp;
    if (snapshot->token.identity.determinant.model.backend == vibeqc::scf::FockBackend::Cpu) {
      ecp = vibeqc::integrals::checked_ecp_integrals(source.system, true);
    } else {
      // No CPU derivative fallback. The native two-grid convergence gate is
      // the same one used by the energy owner and direct-HF force consumer.
      status = vibeqc::integrals::ecp_integrals_cuda(snapshot->token.identity.model.device,
                                                     source.system, 0, 0, true, ecp, detail, true);
      if (status != VIBEQC_STATUS_SUCCESS) {
        batch->context->last_detail = detail;
        return status;
      }
    }
    if (ecp.local_derivative.size() != count / 2 || ecp.nonlocal_derivative.size() != count / 2)
      return VIBEQC_STATUS_INTERNAL_ERROR;
    status = check_current(*batch, *snapshot);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    std::copy(ecp.local_derivative.begin(), ecp.local_derivative.end(), values);
    std::copy(ecp.nonlocal_derivative.begin(), ecp.nonlocal_derivative.end(), values + count / 2);
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&batch->context->last_detail);
  }
}

vibeqc_status vibeqc_ks_snapshot_energy_v1(const vibeqc_batch* batch,
                                           const vibeqc_ks_snapshot* snapshot, double* energy) {
  if (!batch || !snapshot || !energy) return VIBEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(batch->context->mutex);
  const auto status = check_current(*batch, *snapshot);
  if (status == VIBEQC_STATUS_SUCCESS) *energy = snapshot->energy;
  return status;
}

/** Private restricted SCF-domain response bridge. Rows contain the directional
 * derivative of (v_rho, v_grad[3]) for total density, before AO assembly. */
vibeqc_status vibeqc_xc_rks_response_batch_v1(std::uint32_t pbe, const double* rho,
                                              const double* gradient, const double* delta_rho,
                                              const double* delta_gradient, std::size_t point_count,
                                              double* values, std::size_t value_count) {
  constexpr std::size_t stride = 4;
  if (pbe > 1 || !rho || !gradient || !delta_rho || !delta_gradient || !values ||
      point_count == 0 || point_count > std::numeric_limits<std::size_t>::max() / stride ||
      value_count != stride * point_count)
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  for (std::size_t point = 0; point < point_count; ++point) {
    const auto xc = vibeqc::dft::point::restricted_response(
        pbe != 0, rho[point], gradient + 3 * point, delta_rho[point], delta_gradient + 3 * point);
    if (!xc.valid) return VIBEQC_STATUS_NUMERICAL_FAILURE;
    values[stride * point] = xc.rho[0];
    for (unsigned k = 0; k < 3; ++k) values[stride * point + 1 + k] = xc.gradient[0][k];
  }
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status vibeqc_xc_uks_response_batch_v1(std::uint32_t pbe, const double* rho,
                                              const double* gradient, const double* delta_rho,
                                              const double* delta_gradient, std::size_t point_count,
                                              double* values, std::size_t value_count) {
  constexpr std::size_t stride = 8;
  if (pbe > 1 || !rho || !gradient || !delta_rho || !delta_gradient || !values ||
      point_count == 0 || point_count > std::numeric_limits<std::size_t>::max() / stride ||
      value_count != stride * point_count)
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  for (std::size_t point = 0; point < point_count; ++point) {
    double local_rho[2], local_delta_rho[2], local_gradient[2][3], local_delta_gradient[2][3];
    for (unsigned s = 0; s < 2; ++s) {
      local_rho[s] = rho[s * point_count + point];
      local_delta_rho[s] = delta_rho[s * point_count + point];
      for (unsigned k = 0; k < 3; ++k) {
        local_gradient[s][k] = gradient[(s * point_count + point) * 3 + k];
        local_delta_gradient[s][k] = delta_gradient[(s * point_count + point) * 3 + k];
      }
    }
    const auto xc = vibeqc::dft::point::unrestricted_response(
        pbe != 0, local_rho, local_gradient, local_delta_rho, local_delta_gradient);
    if (!xc.valid) return VIBEQC_STATUS_NUMERICAL_FAILURE;
    for (unsigned s = 0; s < 2; ++s) {
      values[stride * point + s] = xc.rho[s];
      for (unsigned k = 0; k < 3; ++k) values[stride * point + 2 + 3 * s + k] = xc.gradient[s][k];
    }
  }
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status vibeqc_xc_point_batch_v1(std::uint32_t pbe, const double* rho, const double* gradient,
                                       std::size_t point_count, double* values,
                                       std::size_t value_count) {
  constexpr std::size_t stride = 9;
  if (pbe > 1 || !rho || !gradient || !values || point_count == 0 ||
      point_count > std::numeric_limits<std::size_t>::max() / stride ||
      value_count != stride * point_count)
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  for (std::size_t point = 0; point < point_count; ++point) {
    double local_rho[2]{rho[point], rho[point_count + point]};
    double local_gradient[2][3]{};
    for (std::size_t spin = 0; spin < 2; ++spin)
      for (std::size_t axis = 0; axis < 3; ++axis)
        local_gradient[spin][axis] = gradient[(spin * point_count + point) * 3 + axis];
    const auto xc = vibeqc::dft::point::evaluate(pbe != 0, local_rho, local_gradient);
    if (!xc.valid) return VIBEQC_STATUS_NUMERICAL_FAILURE;
    double* output = values + stride * point;
    output[0] = xc.energy;
    output[1] = xc.rho[0];
    output[2] = xc.rho[1];
    for (std::size_t spin = 0; spin < 2; ++spin)
      for (std::size_t axis = 0; axis < 3; ++axis)
        output[3 + spin * 3 + axis] = xc.gradient[spin][axis];
  }
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status vibeqc_xc_point_batch_v3(std::uint32_t functional, double exchange_scale,
                                       double correlation_scale, const double* rho,
                                       const double* gradient, const double* tau,
                                       std::size_t point_count, double* values,
                                       std::size_t value_count) {
  constexpr std::size_t stride = 11;
  if (!std::isfinite(exchange_scale) || !std::isfinite(correlation_scale) || exchange_scale < 0 ||
      correlation_scale < 0 ||
      (functional != 1 && (exchange_scale != 1.0 || correlation_scale != 1.0)) || functional > 4 ||
      !rho || !gradient || !tau || !values || point_count == 0 ||
      point_count > std::numeric_limits<std::size_t>::max() / stride ||
      value_count != stride * point_count)
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  try {
    for (std::size_t point = 0; point < point_count; ++point) {
      double local_rho[2]{rho[point], rho[point_count + point]};
      double local_gradient[2][3]{};
      double local_tau[2]{tau[point], tau[point_count + point]};
      for (std::size_t spin = 0; spin < 2; ++spin)
        for (std::size_t axis = 0; axis < 3; ++axis)
          local_gradient[spin][axis] = gradient[(spin * point_count + point) * 3 + axis];
      double* output = values + stride * point;
      if (functional < 2) {
        const auto xc = vibeqc::dft::point::evaluate(functional == 1, local_rho, local_gradient,
                                                     exchange_scale, correlation_scale);
        if (!xc.valid) return VIBEQC_STATUS_NUMERICAL_FAILURE;
        output[0] = xc.energy;
        output[1] = xc.rho[0];
        output[2] = xc.rho[1];
        for (std::size_t spin = 0; spin < 2; ++spin)
          for (std::size_t axis = 0; axis < 3; ++axis)
            output[3 + spin * 3 + axis] = xc.gradient[spin][axis];
        output[9] = output[10] = 0.0;
      } else if (functional == 2 || functional == 4) {
        const auto xc =
            functional == 4
                ? vibeqc::dft::evaluate_wb97mv_point(local_rho, local_gradient, local_tau)
                : vibeqc::dft::evaluate_r2scan_point(local_rho, local_gradient, local_tau);
        output[0] = xc.energy;
        output[1] = xc.rho[0];
        output[2] = xc.rho[1];
        for (std::size_t spin = 0; spin < 2; ++spin)
          for (std::size_t axis = 0; axis < 3; ++axis)
            output[3 + spin * 3 + axis] = xc.gradient[spin][axis];
        output[9] = xc.kinetic[0];
        output[10] = xc.kinetic[1];
      } else {
        const auto xc = vibeqc::dft::evaluate_b3lyp_point(local_rho, local_gradient);
        output[0] = xc.energy;
        output[1] = xc.rho[0];
        output[2] = xc.rho[1];
        for (std::size_t spin = 0; spin < 2; ++spin)
          for (std::size_t axis = 0; axis < 3; ++axis)
            output[3 + spin * 3 + axis] = xc.gradient[spin][axis];
        output[9] = output[10] = 0.0;
      }
    }
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return VIBEQC_STATUS_NUMERICAL_FAILURE;
  }
}

vibeqc_status vibeqc_xc_point_batch_v2(std::uint32_t functional, const double* rho,
                                       const double* gradient, const double* tau,
                                       std::size_t point_count, double* values,
                                       std::size_t value_count) {
  return vibeqc_xc_point_batch_v3(functional, 1.0, 1.0, rho, gradient, tau, point_count, values,
                                  value_count);
}

void vibeqc_ks_snapshot_destroy_v1(vibeqc_ks_snapshot* snapshot) { delete snapshot; }
}  // extern "C"
