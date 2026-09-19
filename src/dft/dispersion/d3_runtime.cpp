#include "dft/dispersion/d3_runtime.hpp"

#include <algorithm>
#include <limits>
#include <new>

#include "d3_data.hpp"

namespace vibeqc::dft::dispersion {
namespace {

bool add_bytes(std::uint64_t& total, std::uint64_t value) {
  if (value > std::numeric_limits<std::uint64_t>::max() - total) return false;
  total += value;
  return true;
}

std::uint64_t static_table_bytes() {
  return sizeof(d3_data::kElements) + sizeof(d3_data::kPairs) + sizeof(d3_data::kReferenceCn) +
         sizeof(d3_data::kReferenceC6);
}

D3ResourceUsage resources_for(vibeqc_backend backend, std::size_t systems, std::size_t atoms,
                              std::size_t maximum_atoms, std::uint64_t maximum_bytes,
                              bool& overflow) {
  D3ResourceUsage r{};
  r.maximum_bytes = maximum_bytes;
  r.total_atoms = atoms;
  r.system_count = static_cast<std::uint32_t>(systems);
  r.maximum_atoms = static_cast<std::uint32_t>(maximum_atoms);
  r.table_bytes = static_table_bytes();
  r.plan_host_bytes = (systems + 1) * sizeof(std::uint32_t) + atoms * sizeof(std::int32_t) +
                      3 * atoms * sizeof(double);
  r.execution_host_bytes = 3 * atoms * sizeof(double) + 2 * systems * sizeof(std::uint8_t) +
                           systems * sizeof(D3Status) + systems * sizeof(double) +
                           3 * atoms * sizeof(double);
  if (backend == VIBEQC_BACKEND_CPU_REFERENCE) {
    r.workspace_bytes = d3_workspace_elements(maximum_atoms) * sizeof(double);
    r.execution_host_bytes += r.workspace_bytes + 3 * maximum_atoms * sizeof(double);
  } else if (backend == VIBEQC_BACKEND_CUDA) {
    r.workspace_bytes = d3_ragged_workspace_elements(atoms) * sizeof(double);
    r.device_bytes = (systems + 1) * sizeof(std::uint32_t) + atoms * sizeof(std::int32_t) +
                     3 * atoms * sizeof(double) + 2 * systems * sizeof(std::uint8_t) +
                     systems * sizeof(D3Status) + systems * sizeof(double) +
                     3 * atoms * sizeof(double) + r.workspace_bytes + r.table_bytes;
  }
  std::uint64_t total = 0;
  overflow = !add_bytes(total, r.plan_host_bytes) || !add_bytes(total, r.execution_host_bytes) ||
             !add_bytes(total, r.device_bytes);
  if (!overflow && total > maximum_bytes) overflow = true;
  return r;
}

}  // namespace

std::unique_ptr<D3Plan> D3Plan::prepare(vibeqc_backend backend, int device_id,
                                        std::vector<std::uint32_t> offsets,
                                        std::vector<std::int32_t> atomic_numbers,
                                        std::vector<double> default_coordinates,
                                        D3Parameters parameters, std::uint64_t maximum_bytes,
                                        std::string& detail, vibeqc_status& status) {
  status = VIBEQC_STATUS_INVALID_ARGUMENT;
  if ((backend != VIBEQC_BACKEND_CPU_REFERENCE && backend != VIBEQC_BACKEND_CUDA) ||
      maximum_bytes == 0 || offsets.size() < 2 || offsets.front() != 0 ||
      offsets.back() != atomic_numbers.size() ||
      default_coordinates.size() != 3 * atomic_numbers.size() ||
      !d3_detail::valid_parameters(parameters)) {
    detail = "invalid D3(BJ) production plan descriptor";
    return nullptr;
  }
  std::size_t maximum_atoms = 0;
  for (std::size_t i = 0; i + 1 < offsets.size(); ++i) {
    if (offsets[i + 1] <= offsets[i]) {
      detail = "D3 ragged systems must each contain at least one atom";
      return nullptr;
    }
    const auto count = static_cast<std::size_t>(offsets[i + 1] - offsets[i]);
    maximum_atoms = std::max(maximum_atoms, count);
    if (count > kD3MaximumAtomsPerSystem) {
      status = VIBEQC_STATUS_NOT_IMPLEMENTED;
      detail = "D3 system exceeds the production per-system atom bound";
      return nullptr;
    }
  }
  for (auto z : atomic_numbers) {
    if (z < 1 || z > 86) {
      status = VIBEQC_STATUS_NOT_IMPLEMENTED;
      detail = "D3 production data cover atomic numbers 1 through 86";
      return nullptr;
    }
  }
  for (double value : default_coordinates) {
    if (!d3_detail::finite(value)) {
      detail = "D3 prepared coordinates must be finite";
      return nullptr;
    }
  }

  bool budget_failure = false;
  auto resources = resources_for(backend, offsets.size() - 1, atomic_numbers.size(), maximum_atoms,
                                 maximum_bytes, budget_failure);
  if (budget_failure) {
    status = VIBEQC_STATUS_OUT_OF_MEMORY;
    detail = "D3 production plan exceeds maximum_bytes";
    return nullptr;
  }

  try {
    auto result = std::unique_ptr<D3Plan>(
        new D3Plan(backend, device_id, std::move(offsets), std::move(atomic_numbers),
                   std::move(default_coordinates), parameters, resources));
    if (backend == VIBEQC_BACKEND_CUDA) {
      result->cuda_ = create_d3_cuda_owner(device_id, result->offsets_, result->atomic_numbers_,
                                           resources, detail, status);
      if (!result->cuda_) return nullptr;
    }
    status = VIBEQC_STATUS_SUCCESS;
    return result;
  } catch (const std::bad_alloc&) {
    status = VIBEQC_STATUS_OUT_OF_MEMORY;
    detail = "D3 plan host allocation failed";
    return nullptr;
  } catch (...) {
    status = VIBEQC_STATUS_INTERNAL_ERROR;
    detail = "unexpected D3 plan construction failure";
    return nullptr;
  }
}

D3Plan::~D3Plan() { destroy_d3_cuda_owner(cuda_); }

vibeqc_status D3Plan::execute(std::span<const double> packed_coordinates,
                              std::span<const std::uint8_t> active,
                              std::span<const std::uint8_t> want_gradient,
                              std::vector<D3Status>& statuses, std::vector<double>& energies,
                              std::vector<double>& packed_gradients, std::string& detail) {
  const auto systems = system_count();
  const auto atoms = atomic_numbers_.size();
  if (packed_coordinates.size() != 3 * atoms || active.size() != systems ||
      want_gradient.size() != systems) {
    detail = "D3 execute received a shape-incompatible ragged replay";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  try {
    statuses.assign(systems, D3Status::success);
    energies.assign(systems, 0.0);
    packed_gradients.assign(3 * atoms, 0.0);
    if (backend_ == VIBEQC_BACKEND_CUDA)
      return execute_d3_cuda(cuda_, parameters_, packed_coordinates, active, want_gradient,
                             statuses, energies, packed_gradients, detail);

    std::vector<double> workspace(d3_workspace_elements(resources_.maximum_atoms));
    std::vector<double> candidate_gradient(3 * resources_.maximum_atoms);
    const auto tables = d3_host_tables();
    for (std::uint32_t system = 0; system < systems; ++system) {
      if (!active[system]) continue;
      const std::size_t begin = offsets_[system];
      const std::size_t n = offsets_[system + 1] - offsets_[system];
      double candidate_energy = 0.0;
      double* gradient = want_gradient[system] ? candidate_gradient.data() : nullptr;
      const auto item_status = evaluate_d3_bj(
          n, atomic_numbers_.data() + begin, packed_coordinates.data() + 3 * begin, parameters_,
          tables, workspace.data(), workspace.size(), &candidate_energy, gradient);
      statuses[system] = item_status;
      if (item_status != D3Status::success) continue;
      energies[system] = candidate_energy;
      if (gradient)
        std::copy_n(candidate_gradient.data(), 3 * n, packed_gradients.data() + 3 * begin);
    }
    return VIBEQC_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    detail = "D3 execution host allocation failed";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  } catch (...) {
    detail = "unexpected D3 execution failure";
    return VIBEQC_STATUS_INTERNAL_ERROR;
  }
}

#if !VIBEQC_HAS_CUDA
D3CudaOwner* create_d3_cuda_owner(int, std::span<const std::uint32_t>,
                                  std::span<const std::int32_t>, const D3ResourceUsage&,
                                  std::string& detail, vibeqc_status& status) {
  detail = "D3 CUDA production execution is unavailable in this build";
  status = VIBEQC_STATUS_NOT_IMPLEMENTED;
  return nullptr;
}
void destroy_d3_cuda_owner(D3CudaOwner*) noexcept {}
vibeqc_status execute_d3_cuda(D3CudaOwner*, const D3Parameters&, std::span<const double>,
                              std::span<const std::uint8_t>, std::span<const std::uint8_t>,
                              std::vector<D3Status>&, std::vector<double>&, std::vector<double>&,
                              std::string& detail) {
  detail = "D3 CUDA production execution is unavailable in this build";
  return VIBEQC_STATUS_NOT_IMPLEMENTED;
}
#endif

}  // namespace vibeqc::dft::dispersion
