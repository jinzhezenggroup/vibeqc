#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "api/ks_snapshot.hpp"
#include "dft/xc_point.hpp"
#include "methods/dft_method.hpp"

struct vibeqc_ks_snapshot {
  std::size_t index{};
  vibeqc::dft::CudaKsFinalStateToken token;
  std::vector<double> values;
};

namespace {
vibeqc_status check_current(const vibeqc_batch& batch, const vibeqc_ks_snapshot& snapshot) {
  vibeqc::dft::CudaKsFinalStateToken current;
  std::string detail;
  const auto status =
      vibeqc::methods::detail::dft_final_state_token(*batch.plan, snapshot.index, current, detail);
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  return current == snapshot.token ? VIBEQC_STATUS_SUCCESS : VIBEQC_STATUS_INVALID_ARGUMENT;
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
    const std::array<std::uint64_t, 16> info{
        1,
        n,
        identity.model.spins,
        source.system.atoms.size(),
        source.packed_basis.size(),
        source.weights.size(),
        identity.model.pbe,
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

void vibeqc_ks_snapshot_destroy_v1(vibeqc_ks_snapshot* snapshot) { delete snapshot; }
}  // extern "C"
