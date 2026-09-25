#include "methods/gfn2_runtime_bridge.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "runtime/gfn2_cpu_execution.hpp"
#if defined(VIBEQC_HAS_GFN2_CUDA)
#include "runtime/gfn2_cuda_execution.hpp"
#endif

namespace vibeqc::methods::detail {
namespace {

template <typename T>
vibeqc_xtb_const_buffer_t input_buffer(std::span<const T> values) {
  return {values.empty() ? nullptr : values.data(), values.size_bytes(), VIBEQC_XTB_MEMORY_HOST,
          0u};
}

template <typename T>
vibeqc_xtb_buffer_t output_buffer(std::vector<T>& values) {
  return {values.empty() ? nullptr : values.data(), values.size() * sizeof(T),
          VIBEQC_XTB_MEMORY_HOST, 0u};
}

Gfn2RuntimeStatus map_status(vibeqc_xtb_status_t status) noexcept {
  switch (status) {
    case VIBEQC_XTB_STATUS_SUCCESS:
      return Gfn2RuntimeStatus::kSuccess;
    case VIBEQC_XTB_STATUS_INVALID_ARGUMENT:
      return Gfn2RuntimeStatus::kInvalidArgument;
    case VIBEQC_XTB_STATUS_BACKEND_UNAVAILABLE:
      return Gfn2RuntimeStatus::kBackendUnavailable;
    case VIBEQC_XTB_STATUS_NOT_SUPPORTED:
      return Gfn2RuntimeStatus::kNotSupported;
    case VIBEQC_XTB_STATUS_NOT_IMPLEMENTED:
      return Gfn2RuntimeStatus::kNotImplemented;
    case VIBEQC_XTB_STATUS_ALLOCATION_FAILED:
      return Gfn2RuntimeStatus::kAllocationFailed;
    case VIBEQC_XTB_STATUS_SCC_NOT_CONVERGED:
      return Gfn2RuntimeStatus::kNotConverged;
    case VIBEQC_XTB_STATUS_EIGENSOLVER_FAILED:
      return Gfn2RuntimeStatus::kEigensolverFailed;
    default:
      return Gfn2RuntimeStatus::kInternalError;
  }
}

double gfn2_radial_to_vibeqc(unsigned angular_momentum) {
  double odd_double_factorial = 1.0;
  for (unsigned factor = 1; factor < 2u * angular_momentum; factor += 2u)
    odd_double_factorial *= static_cast<double>(factor);
  return std::sqrt(odd_double_factorial);
}

bool convert_cpu_orbitals(const Gfn2RuntimeRequest& request,
                          const vibeqc::xtb::detail::Gfn2CpuOrbitalSnapshot& snapshot,
                          Gfn2RuntimeOrbitals& result, std::string& error) {
  if (snapshot.orbital_count <= 0 || static_cast<std::uint64_t>(snapshot.orbital_count) >
                                         std::numeric_limits<std::size_t>::max()) {
    error = "GFN2 orbital snapshot has an invalid AO extent";
    return false;
  }
  const std::size_t n = static_cast<std::size_t>(snapshot.orbital_count);
  const std::size_t shell_count = snapshot.angular_momenta.size();
  if (n > std::numeric_limits<std::size_t>::max() / n || snapshot.overlap.size() != n * n ||
      snapshot.coefficients.size() != n * n || snapshot.occupations.size() != 2u * n ||
      snapshot.shell_orbital_offsets.size() != shell_count + 1u ||
      snapshot.shell_primitive_offsets.size() != shell_count + 1u ||
      snapshot.shell_to_atom.size() != shell_count) {
    error = "GFN2 orbital snapshot has inconsistent basis dimensions";
    return false;
  }

  Gfn2RuntimeOrbitals converted;
  auto& source = converted.source_system;
  source.charge = request.charge;
  source.multiplicity = request.multiplicity;
  source.basis_representation = VIBEQC_BASIS_SPHERICAL;
  source.atoms.reserve(request.atomic_numbers.size());
  std::int64_t electrons = -static_cast<std::int64_t>(request.charge);
  for (std::size_t atom = 0; atom < request.atomic_numbers.size(); ++atom) {
    const auto atomic_number = request.atomic_numbers[atom];
    if (atomic_number <= 0 ||
        electrons > std::numeric_limits<std::int64_t>::max() - atomic_number) {
      error = "GFN2 orbital source has an invalid nuclear charge";
      return false;
    }
    electrons += atomic_number;
    source.atoms.push_back({atomic_number,
                            {request.positions[3u * atom], request.positions[3u * atom + 1u],
                             request.positions[3u * atom + 2u]}});
  }
  if (electrons <= 0 || electrons > std::numeric_limits<int>::max()) {
    error = "GFN2 orbital source has an invalid electron count";
    return false;
  }
  source.electron_count = static_cast<int>(electrons);

  source.shells.reserve(shell_count);
  std::vector<std::size_t> new_to_old(n, n);
  for (std::size_t shell_index = 0; shell_index < shell_count; ++shell_index) {
    const auto atom64 = snapshot.shell_to_atom[shell_index];
    const auto orbital_begin64 = snapshot.shell_orbital_offsets[shell_index];
    const auto orbital_end64 = snapshot.shell_orbital_offsets[shell_index + 1u];
    const auto primitive_begin64 = snapshot.shell_primitive_offsets[shell_index];
    const auto primitive_end64 = snapshot.shell_primitive_offsets[shell_index + 1u];
    const unsigned angular = snapshot.angular_momenta[shell_index];
    if (angular > 2u || atom64 < 0 || static_cast<std::uint64_t>(atom64) >= source.atoms.size() ||
        orbital_begin64 < 0 || orbital_end64 < orbital_begin64 || primitive_begin64 < 0 ||
        primitive_end64 <= primitive_begin64 || static_cast<std::uint64_t>(orbital_end64) > n ||
        static_cast<std::uint64_t>(primitive_end64) > snapshot.primitive_exponents.size() ||
        snapshot.primitive_exponents.size() != snapshot.primitive_coefficients.size() ||
        orbital_end64 - orbital_begin64 != static_cast<std::int64_t>(2u * angular + 1u)) {
      error = "GFN2 orbital source shell metadata is inconsistent";
      return false;
    }

    core::Shell shell;
    shell.atom_index = static_cast<std::uint32_t>(atom64);
    shell.angular_momentum = angular;
    const double radial_scale = gfn2_radial_to_vibeqc(angular);
    for (std::int64_t primitive = primitive_begin64; primitive < primitive_end64; ++primitive) {
      const std::size_t p = static_cast<std::size_t>(primitive);
      shell.primitives.push_back(
          {snapshot.primitive_exponents[p], snapshot.primitive_coefficients[p] * radial_scale});
    }
    source.shells.push_back(std::move(shell));

    const std::size_t begin = static_cast<std::size_t>(orbital_begin64);
    if (angular == 1u) {
      // Native GFN2 spherical p order is (y,z,x); VibeQC's public p order is
      // Cartesian (x,y,z). d uses the same m=-2..2 order on both sides.
      new_to_old[begin] = begin + 2u;
      new_to_old[begin + 1u] = begin;
      new_to_old[begin + 2u] = begin + 1u;
    } else {
      for (std::size_t ao = begin; ao < static_cast<std::size_t>(orbital_end64); ++ao)
        new_to_old[ao] = ao;
    }
  }
  if (std::any_of(new_to_old.begin(), new_to_old.end(),
                  [n](std::size_t value) { return value >= n; })) {
    error = "GFN2 orbital source AO permutation is incomplete";
    return false;
  }

  converted.overlap.resize(n * n);
  converted.coefficients.resize(n * n);
  converted.occupations = snapshot.occupations;
  for (std::size_t row = 0; row < n; ++row) {
    const std::size_t old_row = new_to_old[row];
    for (std::size_t column = 0; column < n; ++column) {
      converted.overlap[row * n + column] = snapshot.overlap[old_row * n + new_to_old[column]];
      converted.coefficients[row * n + column] = snapshot.coefficients[old_row * n + column];
    }
  }

  result = std::move(converted);
  error.clear();
  return true;
}

}  // namespace

struct Gfn2RuntimeBridge::Impl {
  Impl(Gfn2RuntimeBackend requested_backend, int device_id) : backend(requested_backend) {
#if defined(VIBEQC_HAS_GFN2_CUDA)
    if (backend == Gfn2RuntimeBackend::kCuda)
      cuda_cache =
          std::make_unique<vibeqc::xtb::detail::Gfn2CudaExecutionCache>(device_id, nullptr);
#else
    (void)device_id;
#endif
  }

  Gfn2RuntimeBackend backend;
  vibeqc::xtb::detail::Gfn2CpuExecutionCache cpu_cache;
#if defined(VIBEQC_HAS_GFN2_CUDA)
  std::unique_ptr<vibeqc::xtb::detail::Gfn2CudaExecutionCache> cuda_cache;
#endif
};

Gfn2RuntimeBridge::Gfn2RuntimeBridge(Gfn2RuntimeBackend backend, int device_id)
    : impl_(std::make_unique<Impl>(backend, device_id)) {}

Gfn2RuntimeBridge::~Gfn2RuntimeBridge() = default;
Gfn2RuntimeBridge::Gfn2RuntimeBridge(Gfn2RuntimeBridge&&) noexcept = default;
Gfn2RuntimeBridge& Gfn2RuntimeBridge::operator=(Gfn2RuntimeBridge&&) noexcept = default;

Gfn2RuntimeResult Gfn2RuntimeBridge::execute(const Gfn2RuntimeRequest& request) {
  Gfn2RuntimeResult result;
  if (request.atomic_numbers.empty() ||
      request.positions.size() != 3u * request.atomic_numbers.size()) {
    result.status = Gfn2RuntimeStatus::kInvalidArgument;
    result.detail = "GFN2 runtime request has inconsistent atom/position extents";
    return result;
  }
  if (request.multiplicity == 0u) {
    result.status = Gfn2RuntimeStatus::kInvalidArgument;
    result.detail = "GFN2 runtime multiplicity must be positive";
    return result;
  }
  if (request.compute_orbitals && impl_->backend != Gfn2RuntimeBackend::kCpu) {
    result.status = Gfn2RuntimeStatus::kNotSupported;
    result.detail = "GFN2 orbital export is currently qualified only for the CPU runtime";
    return result;
  }

  const auto atom_count = static_cast<std::int64_t>(request.atomic_numbers.size());
  std::vector<std::int64_t> atom_offsets{0, atom_count};
  std::vector<double> molecular_charges{static_cast<double>(request.charge)};
  std::vector<std::int32_t> unpaired_electrons{
      static_cast<std::int32_t>(request.multiplicity - 1u)};
  // Preserve the public GFN2 restricted/shared-orbital open-shell contract.
  // Multiplicity sets the unpaired-electron count; unrestricted GFN2 is a
  // separate capability and must not be inferred at this adapter boundary.
  std::vector<std::int32_t> spin_channels{1};

  vibeqc_xtb_batch_t batch{};
  vibeqc_xtb_compute_options_t options{};
  vibeqc_xtb_batch_result_t output{};
  batch.struct_size = sizeof(batch);
  batch.api_version = VIBEQC_XTB_API_VERSION;
  options.struct_size = sizeof(options);
  options.api_version = VIBEQC_XTB_API_VERSION;
  output.struct_size = sizeof(output);
  output.api_version = VIBEQC_XTB_API_VERSION;

  batch.batch_size = 1;
  batch.total_atoms = atom_count;
  batch.atom_offsets = input_buffer<std::int64_t>(atom_offsets);
  batch.atomic_numbers = input_buffer<std::int32_t>(request.atomic_numbers);
  batch.positions = input_buffer<double>(request.positions);
  batch.molecular_charges = input_buffer<double>(molecular_charges);
  batch.unpaired_electrons = input_buffer<std::int32_t>(unpaired_electrons);
  batch.spin_channels = input_buffer<std::int32_t>(spin_channels);

  options.model = VIBEQC_XTB_MODEL_GFN2_XTB;
  options.flags =
      static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_ENERGY) |
      (request.compute_forces ? static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_FORCES) : 0u) |
      (request.compute_atomic_charges
           ? static_cast<std::uint32_t>(VIBEQC_XTB_COMPUTE_ATOMIC_CHARGES)
           : 0u);
  options.max_scc_iterations = request.maximum_iterations;
  options.charge_tolerance = request.charge_tolerance;
  options.energy_tolerance = request.energy_tolerance;
  options.electronic_temperature = VIBEQC_XTB_DEFAULT_ELECTRONIC_TEMPERATURE;
  options.scc_start_mode = VIBEQC_XTB_SCC_START_FRESH;
  options.scc_mixer = VIBEQC_XTB_SCC_MIXER_MODIFIED_BROYDEN;
  options.scc_mixer_history = request.mixer_history;
  options.scc_mixer_damping = 0.4;

  std::vector<double> energies(1u);
  std::vector<double> forces(request.compute_forces ? 3u * request.atomic_numbers.size() : 0u);
  std::vector<double> atomic_charges(request.compute_atomic_charges ? request.atomic_numbers.size()
                                                                    : 0u);
  std::vector<std::int32_t> iterations(1u);
  std::vector<std::uint8_t> converged(1u);
  std::vector<std::int32_t> statuses(1u);
  output.energies = output_buffer(energies);
  output.forces = output_buffer(forces);
  output.atomic_charges = output_buffer(atomic_charges);
  output.scc_iterations = output_buffer(iterations);
  output.scc_converged = output_buffer(converged);
  output.per_system_status = output_buffer(statuses);

  std::string execution_error;
  vibeqc_xtb_status_t execution_status = VIBEQC_XTB_STATUS_NOT_IMPLEMENTED;
  if (impl_->backend == Gfn2RuntimeBackend::kCpu) {
    execution_status = vibeqc::xtb::detail::execute_restricted_gfn2_cpu(
        impl_->cpu_cache, batch, options, output, execution_error);
#if defined(VIBEQC_HAS_GFN2_CUDA)
  } else if (impl_->backend == Gfn2RuntimeBackend::kCuda && impl_->cuda_cache) {
    execution_status = vibeqc::xtb::detail::execute_restricted_gfn2_cuda(
        *impl_->cuda_cache, batch, options, output, execution_error);
#endif
  }
  if (execution_status != VIBEQC_XTB_STATUS_SUCCESS) {
    result.status = map_status(execution_status);
    result.detail = std::move(execution_error);
    return result;
  }

  const auto system_status = static_cast<vibeqc_xtb_status_t>(statuses[0]);
  if (system_status == VIBEQC_XTB_STATUS_EIGENSOLVER_FAILED) {
    result.status = Gfn2RuntimeStatus::kEigensolverFailed;
    result.detail = "generalized eigensolver failed";
    return result;
  }
  if (system_status != VIBEQC_XTB_STATUS_SUCCESS &&
      system_status != VIBEQC_XTB_STATUS_SCC_NOT_CONVERGED) {
    // Match the former public-method boundary exactly: only the eigensolver
    // status has a dedicated public mapping; every other unexpected terminal
    // per-system status remains an internal runtime failure.
    result.status = Gfn2RuntimeStatus::kInternalError;
    result.detail = "unexpected per-system runtime status";
    return result;
  }

  result.status = Gfn2RuntimeStatus::kSuccess;
  result.energy = energies[0];
  result.forces = std::move(forces);
  result.atomic_charges = std::move(atomic_charges);
  result.iterations = iterations[0] < 0 ? 0u : static_cast<unsigned>(iterations[0]);
  result.converged = system_status == VIBEQC_XTB_STATUS_SUCCESS && converged[0] != 0u;
  if (request.compute_orbitals && result.converged) {
    vibeqc::xtb::detail::Gfn2CpuOrbitalSnapshot snapshot;
    std::string snapshot_error;
    const auto snapshot_status = vibeqc::xtb::detail::copy_restricted_gfn2_orbital_snapshot_cpu(
        impl_->cpu_cache, snapshot, snapshot_error);
    if (snapshot_status != VIBEQC_XTB_STATUS_SUCCESS) {
      result.status = map_status(snapshot_status);
      result.detail = std::move(snapshot_error);
      return result;
    }
    Gfn2RuntimeOrbitals orbitals;
    if (!convert_cpu_orbitals(request, snapshot, orbitals, snapshot_error)) {
      result.status = Gfn2RuntimeStatus::kInternalError;
      result.detail = std::move(snapshot_error);
      return result;
    }
    result.orbitals = std::move(orbitals);
  }
  return result;
}

const char* gfn2_runtime_status_name(Gfn2RuntimeStatus status) noexcept {
  switch (status) {
    case Gfn2RuntimeStatus::kSuccess:
      return "success";
    case Gfn2RuntimeStatus::kInvalidArgument:
      return "invalid argument";
    case Gfn2RuntimeStatus::kBackendUnavailable:
      return "backend unavailable";
    case Gfn2RuntimeStatus::kNotSupported:
      return "not supported";
    case Gfn2RuntimeStatus::kNotImplemented:
      return "not implemented";
    case Gfn2RuntimeStatus::kAllocationFailed:
      return "allocation failed";
    case Gfn2RuntimeStatus::kNotConverged:
      return "SCC not converged";
    case Gfn2RuntimeStatus::kEigensolverFailed:
      return "eigensolver failed";
    case Gfn2RuntimeStatus::kInternalError:
      return "internal error";
  }
  return "internal error";
}

}  // namespace vibeqc::methods::detail
