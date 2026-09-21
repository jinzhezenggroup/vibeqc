#include "dft/dispersion/d4_runtime.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>

#include "dft/dispersion/d4_eeq_data.hpp"
#include "dft/dispersion/d4_eeq_r2scan3c_c6.hpp"
#include "generated_d4_derivative.hpp"
#if VIBEQC_HAS_CUDA
#include "dft/dispersion/d4_cuda.hpp"
#endif

namespace vibeqc::dft::dispersion {
namespace {

bool add_bytes(std::uint64_t& total, std::uint64_t value) {
  if (value > std::numeric_limits<std::uint64_t>::max() - total) return false;
  total += value;
  return true;
}

std::uint64_t static_table_bytes() {
  return sizeof(eeq_data::kElements) + sizeof(eeq_data::kReferences) +
         sizeof(eeq_data::kReferenceC6Standard) + sizeof(eeq_data::kChargeElements);
}

D4ResourceUsage resources_for(vibeqc_backend backend, std::size_t systems, std::size_t atoms,
                              std::size_t maximum_atoms, std::uint64_t maximum_bytes,
                              std::uint32_t workers, bool& overflow) {
  D4ResourceUsage r{};
  r.maximum_bytes = maximum_bytes;
  r.total_atoms = atoms;
  r.system_count = static_cast<std::uint32_t>(systems);
  r.maximum_atoms = static_cast<std::uint32_t>(maximum_atoms);
  r.worker_blocks = backend == VIBEQC_BACKEND_CUDA ? workers : 1u;
  r.workspace_slots = r.worker_blocks;
  r.table_bytes = static_table_bytes();
  r.plan_host_bytes = (systems + 1) * sizeof(std::uint32_t) + atoms * sizeof(std::int32_t) +
                      systems * sizeof(double) + 6 * atoms * sizeof(double);
  r.execution_host_bytes = 3 * atoms * sizeof(double) + 2 * systems * sizeof(std::uint8_t) +
                           systems * sizeof(vibeqc_status) + systems * sizeof(D4Status) +
                           2 * systems * sizeof(double) + 3 * atoms * sizeof(double) +
                           atoms * sizeof(double);

  const auto cpu_workspace_per_system =
      complete_d4_eeq_workspace_elements(static_cast<int>(maximum_atoms)) * sizeof(double);
  if (backend == VIBEQC_BACKEND_CPU_REFERENCE) {
    r.workspace_bytes = cpu_workspace_per_system;
    r.execution_host_bytes += cpu_workspace_per_system + (4 * maximum_atoms + 2) * sizeof(double);
  } else if (backend == VIBEQC_BACKEND_CUDA) {
    const auto eeq_worker_elements = eeq2019_workspace_elements(static_cast<int>(maximum_atoms)) +
                                     3 * maximum_atoms * maximum_atoms;
    const auto eeq_worker_bytes = eeq_worker_elements * sizeof(double);
#if VIBEQC_HAS_CUDA
    const auto fixed_workspace_bytes = d4_cuda_workspace_elements(atoms) * sizeof(double);
#else
    const auto fixed_workspace_bytes = 27u * atoms * sizeof(double);
#endif
    if (workers && eeq_worker_bytes > std::numeric_limits<std::uint64_t>::max() / workers) {
      overflow = true;
      return r;
    }
    r.workspace_bytes = eeq_worker_bytes * workers;
    if (!add_bytes(r.workspace_bytes, fixed_workspace_bytes)) {
      overflow = true;
      return r;
    }
    r.device_bytes =
        (systems + 1) * sizeof(std::uint32_t) + atoms * sizeof(std::int32_t) +
        systems * sizeof(double) + 3 * atoms * sizeof(double) + 3 * systems * sizeof(std::uint8_t) +
        2 * systems * sizeof(D4Status) + 2 * systems * sizeof(double) + 3 * atoms * sizeof(double) +
        2 * atoms * sizeof(double) + sizeof(std::uint32_t) + r.workspace_bytes + r.table_bytes;
  }
  std::uint64_t total = 0;
  overflow = !add_bytes(total, r.plan_host_bytes) || !add_bytes(total, r.execution_host_bytes) ||
             !add_bytes(total, r.device_bytes);
  if (!overflow && total > maximum_bytes) overflow = true;
  return r;
}

bool valid_profile_parameters(const D4Parameters& p, D4EEQProfile profile) {
  if (!d4_detail::valid_parameters(p) || p.reference_model != D4ReferenceModel::eeq) return false;
  if (profile == D4EEQProfile::standard)
    return std::fabs(p.ga - 3.0) <= 1.0e-15 && std::fabs(p.gc - 2.0) <= 1.0e-15;
  if (profile == D4EEQProfile::r2scan3c)
    return std::fabs(p.ga - 2.0) <= 1.0e-15 && std::fabs(p.gc - 1.0) <= 1.0e-15;
  return false;
}

}  // namespace

std::unique_ptr<D4Plan> D4Plan::prepare(
    vibeqc_backend backend, int device_id, std::vector<std::uint32_t> offsets,
    std::vector<std::int32_t> atomic_numbers, std::vector<double> total_charges,
    std::vector<double> default_coordinates, D4Parameters parameters, D4EEQProfile profile,
    std::uint64_t maximum_bytes, std::string& detail, vibeqc_status& status) {
  status = VIBEQC_STATUS_INVALID_ARGUMENT;
  if ((backend != VIBEQC_BACKEND_CPU_REFERENCE && backend != VIBEQC_BACKEND_CUDA) ||
      maximum_bytes == 0 || offsets.size() < 2 || offsets.front() != 0 ||
      offsets.back() != atomic_numbers.size() || total_charges.size() + 1 != offsets.size() ||
      default_coordinates.size() != 3 * atomic_numbers.size() ||
      !valid_profile_parameters(parameters, profile)) {
    detail = "invalid D4(BJ)-EEQ production plan descriptor";
    return nullptr;
  }
  std::size_t maximum_atoms = 0;
  for (std::size_t i = 0; i + 1 < offsets.size(); ++i) {
    if (offsets[i + 1] <= offsets[i]) {
      detail = "D4 ragged systems must each contain at least one atom";
      return nullptr;
    }
    const auto count = static_cast<std::size_t>(offsets[i + 1] - offsets[i]);
    maximum_atoms = std::max(maximum_atoms, count);
    if (count > static_cast<std::size_t>(kD4MaximumAtoms)) {
      status = VIBEQC_STATUS_NOT_IMPLEMENTED;
      detail = "D4 system exceeds the production per-system atom bound";
      return nullptr;
    }
  }
  for (auto z : atomic_numbers) {
    if (z < 1 || z > static_cast<int>(eeq_data::kElementCount)) {
      status = VIBEQC_STATUS_NOT_IMPLEMENTED;
      detail = "D4 EEQ production data cover atomic numbers 1 through 86";
      return nullptr;
    }
  }
  for (double value : default_coordinates)
    if (!eeq_detail::finite(value)) {
      detail = "D4 prepared coordinates must be finite";
      return nullptr;
    }
  for (double value : total_charges)
    if (!eeq_detail::finite(value)) {
      detail = "D4 total charges must be finite";
      return nullptr;
    }

  const auto systems = offsets.size() - 1;
  std::uint32_t workers = backend == VIBEQC_BACKEND_CUDA
                              ? static_cast<std::uint32_t>(std::min<std::size_t>(systems, 32))
                              : 1u;
  D4ResourceUsage resources{};
  bool budget_failure = true;
  while (workers) {
    resources = resources_for(backend, systems, atomic_numbers.size(), maximum_atoms, maximum_bytes,
                              workers, budget_failure);
    if (!budget_failure) break;
    if (backend != VIBEQC_BACKEND_CUDA || workers == 1) break;
    --workers;
  }
  if (budget_failure) {
    status = VIBEQC_STATUS_OUT_OF_MEMORY;
    detail = "D4 production plan exceeds maximum_bytes even at the minimum workspace schedule";
    return nullptr;
  }

  try {
    auto result = std::unique_ptr<D4Plan>(new D4Plan(
        backend, device_id, std::move(offsets), std::move(atomic_numbers), std::move(total_charges),
        std::move(default_coordinates), parameters, profile, resources));
    if (backend == VIBEQC_BACKEND_CUDA) {
      result->cuda_ = create_d4_cuda_owner(device_id, result->offsets_, result->atomic_numbers_,
                                           result->total_charges_, result->default_coordinates_,
                                           profile, resources, detail, status);
      if (!result->cuda_) return nullptr;
    }
    status = VIBEQC_STATUS_SUCCESS;
    detail.clear();
    return result;
  } catch (const std::bad_alloc&) {
    status = VIBEQC_STATUS_OUT_OF_MEMORY;
    detail = "D4 plan host allocation failed";
    return nullptr;
  } catch (...) {
    status = VIBEQC_STATUS_INTERNAL_ERROR;
    detail = "unexpected D4 plan construction failure";
    return nullptr;
  }
}

D4Plan::~D4Plan() { destroy_d4_cuda_owner(cuda_); }

vibeqc_status D4Plan::execute(std::span<const double> packed_coordinates,
                              std::span<const std::uint8_t> active,
                              std::span<const std::uint8_t> want_gradient,
                              std::vector<D4Status>& statuses,
                              std::vector<double>& energy_components,
                              std::vector<double>& packed_gradients,
                              std::vector<double>& packed_charges, std::string& detail) {
  const auto systems = system_count();
  const auto atoms = atomic_numbers_.size();
  if (packed_coordinates.size() != 3 * atoms || active.size() != systems ||
      want_gradient.size() != systems) {
    detail = "D4 execute received a shape-incompatible ragged replay";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const bool same_coordinates = std::equal(packed_coordinates.begin(), packed_coordinates.end(),
                                           last_coordinates_.begin(), last_coordinates_.end());
  const bool changed = backend_ == VIBEQC_BACKEND_CUDA
                           ? !cuda_coordinates_valid_ || !same_coordinates
                           : !same_coordinates;
  ++counters_.execution_count;
  if (changed)
    ++counters_.changed_geometry_replays;
  else
    ++counters_.unchanged_geometry_replays;

  try {
    statuses.assign(systems, D4Status::success);
    energy_components.assign(2 * systems, 0.0);
    packed_gradients.assign(3 * atoms, 0.0);
    packed_charges.assign(atoms, 0.0);
    if (backend_ == VIBEQC_BACKEND_CUDA) {
      const auto status = execute_d4_cuda(cuda_, parameters_, profile_, packed_coordinates, changed,
                                          active, want_gradient, statuses, energy_components,
                                          packed_gradients, packed_charges, counters_, detail);
      if (status == VIBEQC_STATUS_SUCCESS) {
        if (changed) last_coordinates_.assign(packed_coordinates.begin(), packed_coordinates.end());
        cuda_coordinates_valid_ = true;
      } else if (changed) {
        // A changed replay may already have replaced the resident coordinates
        // before a later CUDA operation failed.  Never let a subsequent replay
        // trust the host-side last-successful geometry until it is re-uploaded.
        cuda_coordinates_valid_ = false;
      }
      return status;
    }

    std::vector<double> workspace(
        complete_d4_eeq_workspace_elements(static_cast<int>(resources_.maximum_atoms)));
    std::vector<double> candidate_gradient(3 * resources_.maximum_atoms);
    std::vector<double> candidate_charges(resources_.maximum_atoms);
    const auto d4_tables = eeq_d4_host_tables(profile_);
    const auto eeq_tables = eeq2019_host_tables();
    for (std::uint32_t system = 0; system < systems; ++system) {
      if (!active[system]) continue;
      const std::size_t begin = offsets_[system];
      const int n = static_cast<int>(offsets_[system + 1] - offsets_[system]);
      double candidate_energy[2] = {0.0, 0.0};
      const auto item_status = ::vibeqc::generated::d4::evaluate_production_d4_eeq(
          n, atomic_numbers_.data() + begin, packed_coordinates.data() + 3 * begin,
          total_charges_[system], parameters_, profile_, d4_tables, eeq_tables, workspace.data(),
          workspace.size(), candidate_energy, candidate_gradient.data(), candidate_charges.data());
      statuses[system] = item_status;
      if (item_status != D4Status::success) continue;
      energy_components[2 * system] = candidate_energy[0];
      energy_components[2 * system + 1] = candidate_energy[1];
      std::copy_n(candidate_charges.data(), n, packed_charges.data() + begin);
      if (want_gradient[system])
        std::copy_n(candidate_gradient.data(), 3 * n, packed_gradients.data() + 3 * begin);
    }
    if (changed) last_coordinates_.assign(packed_coordinates.begin(), packed_coordinates.end());
    detail.clear();
    return VIBEQC_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    detail = "D4 execution host allocation failed";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    detail = "unexpected D4 execution failure";
    return VIBEQC_STATUS_INTERNAL_ERROR;
  }
}

#if !VIBEQC_HAS_CUDA
D4CudaOwner* create_d4_cuda_owner(int, std::span<const std::uint32_t>,
                                  std::span<const std::int32_t>, std::span<const double>,
                                  std::span<const double>, D4EEQProfile, const D4ResourceUsage&,
                                  std::string& detail, vibeqc_status& status) {
  detail = "D4 CUDA production execution is unavailable in this build";
  status = VIBEQC_STATUS_NOT_IMPLEMENTED;
  return nullptr;
}
void destroy_d4_cuda_owner(D4CudaOwner*) noexcept {}
vibeqc_status execute_d4_cuda(D4CudaOwner*, const D4Parameters&, D4EEQProfile,
                              std::span<const double>, bool, std::span<const std::uint8_t>,
                              std::span<const std::uint8_t>, std::vector<D4Status>&,
                              std::vector<double>&, std::vector<double>&, std::vector<double>&,
                              D4RuntimeCounters&, std::string& detail) {
  detail = "D4 CUDA production execution is unavailable in this build";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
}
#endif

}  // namespace vibeqc::dft::dispersion
