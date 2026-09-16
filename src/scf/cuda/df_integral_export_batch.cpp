#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>

#include "molecule/basis.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/cuda/checked_layout.hpp"
#include "scf/cuda/df_source_internal.hpp"
#include "scf/cuda/df_source_kernels.hpp"
#include "scf/cuda/rhf_policy.hpp"
#include "scf/cuda/topology.hpp"
#include "scf/cuda_density_fitting_integrals.hpp"

namespace vibeqc::scf {

/** Host-owned Cartesian DF tensor export for the explicit raw-integral API. This compatibility
 * output is separate from device-resident source replay. */
namespace {
using namespace cuda_execution;

vibeqc_status cuda_status(cudaError_t status) {
  if (status == cudaSuccess) return VIBEQC_STATUS_SUCCESS;
  return status == cudaErrorMemoryAllocation ? VIBEQC_STATUS_OUT_OF_MEMORY
                                             : VIBEQC_STATUS_CUDA_ERROR;
}

vibeqc_status build_cuda_density_fitting_integrals_batch_impl(
    int device_id, const std::vector<core::System>& orbital_systems,
    const std::vector<core::System>& auxiliary_systems,
    std::vector<integrals::DensityFittingIntegralData>& outputs, std::string& detail,
    std::size_t output_budget_bytes, bool include_derivatives) {
  outputs.clear();
  unsigned value_math = 0, value_lanes = 1;
  if (!cuda_policy::df_value_raw_lanes_requested(value_lanes)) {
    detail = "VIBEQC_DF_VALUE_RAW_MAPPING must be auto, scalar, subgroup, warp or candidate";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (!cuda_policy::df_value_math_requested(value_math)) {
    detail = "VIBEQC_DF_VALUE_MATH must be auto, generic, polynomial rys or candidate";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (device_id < 0 || orbital_systems.empty() ||
      orbital_systems.size() != auxiliary_systems.size()) {
    detail = "CUDA DF integral batch dimensions are invalid";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t batch_size = orbital_systems.size();
  const std::size_t orbital_count = molecule::cartesian_ao_count(orbital_systems.front());
  const std::size_t auxiliary_count = molecule::cartesian_ao_count(auxiliary_systems.front());
  const std::size_t atom_count = orbital_systems.front().atoms.size();
  if (orbital_count == 0U || auxiliary_count == 0U || atom_count == 0U) {
    detail = "CUDA DF integral batch contains an empty basis or geometry";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (batch_size > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    detail = "CUDA DF integral batch exceeds the supported system count";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  std::vector<core::System> combined;
  combined.reserve(batch_size);
  for (std::size_t system = 0; system < batch_size; ++system) {
    const core::System& orbital = orbital_systems[system];
    const core::System& auxiliary = auxiliary_systems[system];
    if (orbital.atoms.size() != atom_count || auxiliary.atoms.size() != atom_count ||
        molecule::cartesian_ao_count(orbital) != orbital_count ||
        molecule::cartesian_ao_count(auxiliary) != auxiliary_count) {
      detail = "CUDA DF integral batch requires homogeneous AO dimensions";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    for (std::size_t atom = 0; atom < atom_count; ++atom) {
      if (orbital.atoms[atom].atomic_number != auxiliary.atoms[atom].atomic_number ||
          orbital.atoms[atom].position != auxiliary.atoms[atom].position) {
        detail = "orbital and auxiliary systems must share geometry";
        return VIBEQC_STATUS_INVALID_ARGUMENT;
      }
    }
    core::System item;
    item.atoms = orbital.atoms;
    item.shells = orbital.shells;
    item.shells.insert(item.shells.end(), auxiliary.shells.begin(), auxiliary.shells.end());
    item.shells.push_back({0, 0, {{0.0, 1.0}}});
    item.charge = orbital.charge;
    item.multiplicity = 1;
    item.electron_count = 2;
    item.basis_representation = VIBEQC_BASIS_CARTESIAN;
    combined.push_back(std::move(item));
  }
  HostBatch host;
  std::vector<const std::vector<double>*> no_warm(batch_size, nullptr);
  if (!pack_host_batch(combined, no_warm, host, false) || host.nbf == 0U) {
    detail = "combined Cartesian DF batch cannot be represented by CUDA";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (orbital_count > std::numeric_limits<std::size_t>::max() - auxiliary_count ||
      orbital_count + auxiliary_count == std::numeric_limits<std::size_t>::max()) {
    detail = "CUDA DF integral batch dimensions overflowed";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t expected_nbf = orbital_count + auxiliary_count + 1U;
  if (host.nbf != expected_nbf ||
      expected_nbf > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    detail = "combined Cartesian DF batch cannot be represented by CUDA";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (auxiliary_count > std::numeric_limits<std::size_t>::max() / auxiliary_count ||
      orbital_count > std::numeric_limits<std::size_t>::max() / orbital_count) {
    detail = "CUDA DF integral batch dimensions overflowed";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t metric_elements = auxiliary_count * auxiliary_count;
  const std::size_t orbital_pair_count = orbital_count * orbital_count;
  if (orbital_pair_count > std::numeric_limits<std::size_t>::max() / auxiliary_count) {
    detail = "CUDA DF integral batch dimensions overflowed";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t three_center_elements = orbital_pair_count * auxiliary_count;
  if (orbital_count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
      auxiliary_count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
      orbital_count > std::numeric_limits<std::size_t>::max() - auxiliary_count) {
    detail = "CUDA DF integral batch exceeds CUDA index limits";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t dummy_index = orbital_count + auxiliary_count;
  if (dummy_index > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
      dummy_index == std::numeric_limits<std::size_t>::max()) {
    detail = "CUDA DF integral batch exceeds CUDA index limits";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t per_system = metric_elements + three_center_elements;
  if (per_system < metric_elements ||
      batch_size > std::numeric_limits<std::size_t>::max() / per_system) {
    detail = "CUDA DF integral batch dimensions overflowed";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  std::size_t coordinate_count = 0;
  if (!checked_multiply(atom_count, 3U, coordinate_count) ||
      coordinate_count == std::numeric_limits<std::size_t>::max()) {
    detail = "CUDA DF integral batch coordinate dimensions overflowed";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
  std::size_t output_elements_per_system = 0;
  if (!checked_multiply(per_system, (include_derivatives ? coordinate_count : 0U) + 1U,
                        output_elements_per_system)) {
    detail = "CUDA DF integral batch output dimensions overflowed";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
  std::size_t per_system_bytes = 0;
  if (!checked_multiply(output_elements_per_system, sizeof(double), per_system_bytes)) {
    detail = "CUDA DF integral batch output bytes overflowed";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
  if (output_budget_bytes != 0U && output_budget_bytes < per_system_bytes) {
    detail = "CUDA DF integral batch budget cannot hold one system output";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
  constexpr unsigned threads = 128U;
  constexpr std::size_t kDefaultOutputChunkBytes = 64U * 1024U * 1024U;
  const std::size_t output_chunk_bytes =
      output_budget_bytes == 0U ? kDefaultOutputChunkBytes : output_budget_bytes;
  const std::size_t chunk_systems =
      std::min(batch_size, std::max<std::size_t>(1U, output_chunk_bytes / per_system_bytes));
  const std::size_t chunk_elements = chunk_systems * per_system;
  if (chunk_elements > std::numeric_limits<unsigned>::max() * static_cast<std::size_t>(threads) ||
      chunk_elements > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
    detail = "CUDA DF integral batch launch dimensions are too large";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }

  cudaError_t cuda_error = cudaSetDevice(device_id);
  if (cuda_error != cudaSuccess) {
    detail = "CUDA device selection failed while generating DF batch";
    return cuda_status(cuda_error);
  }
  cudaStream_t stream = nullptr;
  cuda_error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (cuda_error != cudaSuccess) {
    detail = "CUDA stream creation failed while generating DF batch";
    return cuda_status(cuda_error);
  }
  std::vector<void*> allocations;
  auto release = [&]() {
    for (void* pointer : allocations) (void)runtime::resource_cuda_free(pointer);
    allocations.clear();
    if (stream != nullptr) {
      (void)cudaStreamDestroy(stream);
      stream = nullptr;
    }
  };
  runtime::ResourceScopeExit upload_scope{release};
  auto upload = [&](const void* source, std::size_t bytes) -> void* {
    if (bytes == 0U) return nullptr;
    void* destination = nullptr;
    if (runtime::resource_cuda_malloc(&destination, bytes) != cudaSuccess) return nullptr;
    if (source != nullptr &&
        cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
      (void)runtime::resource_cuda_free(destination);
      return nullptr;
    }
    try {
      allocations.push_back(destination);
    } catch (const std::bad_alloc&) {
      (void)runtime::resource_cuda_free(destination);
      throw;
    }
    return destination;
  };
  auto upload_vector = [&](const auto& values) -> void* {
    return upload(values.data(), values.size() * sizeof(values[0]));
  };

  DeviceBatch device_batch{};
  device_batch.batch_size = static_cast<std::int32_t>(batch_size);
  device_batch.nbf = static_cast<std::int32_t>(host.nbf);
  device_batch.direct_nbf = static_cast<std::int32_t>(host.direct_nbf);
  device_batch.total_atoms = static_cast<std::int64_t>(host.atomic_numbers.size());
  device_batch.total_shells = static_cast<std::int64_t>(host.shell_atoms.size());
  device_batch.atom_offsets = static_cast<const std::int64_t*>(upload_vector(host.atom_offsets));
  device_batch.atom_systems = static_cast<const std::int32_t*>(upload_vector(host.atom_systems));
  device_batch.atomic_numbers =
      static_cast<const std::int32_t*>(upload_vector(host.atomic_numbers));
  device_batch.positions = static_cast<const double*>(upload_vector(host.positions));
  device_batch.shell_atoms = static_cast<const std::int32_t*>(upload_vector(host.shell_atoms));
  device_batch.shell_angular = static_cast<const std::uint8_t*>(upload_vector(host.shell_angular));
  device_batch.shell_ao_offsets =
      static_cast<const std::int64_t*>(upload_vector(host.shell_ao_offsets));
  device_batch.shell_direct_ao_offsets =
      static_cast<const std::int64_t*>(upload_vector(host.shell_direct_ao_offsets));
  device_batch.shell_primitive_offsets =
      static_cast<const std::int64_t*>(upload_vector(host.shell_primitive_offsets));
  device_batch.ao_shells = static_cast<const std::int32_t*>(upload_vector(host.ao_shells));
  device_batch.ao_term_counts =
      static_cast<const std::uint8_t*>(upload_vector(host.ao_term_counts));
  device_batch.ao_term_angular =
      static_cast<const std::uint8_t*>(upload_vector(host.ao_term_angular));
  device_batch.ao_term_coefficients =
      static_cast<const double*>(upload_vector(host.ao_term_coefficients));
  device_batch.direct_ao_shells =
      static_cast<const std::int32_t*>(upload_vector(host.direct_ao_shells));
  device_batch.direct_ao_angular =
      static_cast<const std::uint8_t*>(upload_vector(host.direct_ao_angular));
  device_batch.direct_ao_coefficients =
      static_cast<const double*>(upload_vector(host.direct_ao_coefficients));
  device_batch.primitive_exponents =
      static_cast<const double*>(upload_vector(host.primitive_exponents));
  device_batch.primitive_coefficients =
      static_cast<const double*>(upload_vector(host.primitive_coefficients));
  const std::array<const void*, 18> metadata{device_batch.atom_offsets,
                                             device_batch.atom_systems,
                                             device_batch.atomic_numbers,
                                             device_batch.positions,
                                             device_batch.shell_atoms,
                                             device_batch.shell_angular,
                                             device_batch.shell_ao_offsets,
                                             device_batch.shell_direct_ao_offsets,
                                             device_batch.shell_primitive_offsets,
                                             device_batch.ao_shells,
                                             device_batch.ao_term_counts,
                                             device_batch.ao_term_angular,
                                             device_batch.ao_term_coefficients,
                                             device_batch.direct_ao_shells,
                                             device_batch.direct_ao_angular,
                                             device_batch.direct_ao_coefficients,
                                             device_batch.primitive_exponents,
                                             device_batch.primitive_coefficients};
  for (const void* pointer : metadata) {
    if (pointer == nullptr) {
      detail = "CUDA allocation failed while staging DF batch metadata";
      release();
      return VIBEQC_STATUS_OUT_OF_MEMORY;
    }
  }
  double* device_metric = static_cast<double*>(upload(nullptr, chunk_elements * sizeof(double)));
  if (device_metric == nullptr) {
    detail = "CUDA allocation failed for DF batch output";
    release();
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
  // Keep the packed three-center region adjacent to the metric region so one
  // allocation serves both tensors and derivative launches.
  // Keep the packed three-center region adjacent to the metric region. The
  // allocation is sized only for one bounded system chunk; host output
  // vectors retain the complete batch without requiring a full device copy.
  double* device_three_center = device_metric + chunk_systems * metric_elements;
  try {
    outputs.resize(batch_size);
    for (std::size_t system = 0; system < batch_size; ++system) {
      outputs[system].nbf = orbital_count;
      outputs[system].naux = auxiliary_count;
      outputs[system].ncoord = atom_count * 3U;
      outputs[system].metric.resize(metric_elements);
      outputs[system].three_center.resize(three_center_elements);
      if (include_derivatives) {
        outputs[system].metric_derivative.resize(outputs[system].ncoord * metric_elements);
        outputs[system].three_center_derivative.resize(outputs[system].ncoord *
                                                       three_center_elements);
      }
    }
  } catch (const std::bad_alloc&) {
    detail = "host allocation failed for CUDA DF batch output";
    release();
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }

  std::vector<double> chunk_metric(chunk_systems * metric_elements);
  std::vector<double> chunk_three_center(chunk_systems * three_center_elements);
  auto launch_chunk = [&](std::size_t system_base, std::size_t systems_in_chunk,
                          std::int64_t coordinate) -> cudaError_t {
    const std::size_t launch_elements = systems_in_chunk * per_system;
    const unsigned blocks = static_cast<unsigned>((launch_elements + threads - 1U) / threads);
    if (coordinate < 0) {
      launch_build_cuda_df_integrals_kernel(
          false, blocks, threads, 0, stream, device_batch, orbital_count, auxiliary_count,
          dummy_index, metric_elements, three_center_elements, system_base, systems_in_chunk,
          coordinate, device_metric, device_three_center, value_math, value_lanes);
    } else {
      launch_build_cuda_df_integrals_kernel(
          true, blocks, threads, 0, stream, device_batch, orbital_count, auxiliary_count,
          dummy_index, metric_elements, three_center_elements, system_base, systems_in_chunk,
          coordinate, device_metric, device_three_center, value_math, value_lanes);
    }
    cudaError_t launch_error = cudaGetLastError();
    if (launch_error == cudaSuccess) {
      launch_error = cudaStreamSynchronize(stream);
    }
    return launch_error;
  };

  for (std::size_t system_base = 0; cuda_error == cudaSuccess && system_base < batch_size;
       system_base += chunk_systems) {
    const std::size_t systems_in_chunk = std::min(chunk_systems, batch_size - system_base);
    cuda_error = launch_chunk(system_base, systems_in_chunk, -1);
    if (cuda_error != cudaSuccess) break;
    cuda_error =
        cudaMemcpy(chunk_metric.data(), device_metric,
                   systems_in_chunk * metric_elements * sizeof(double), cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess) break;
    cuda_error = cudaMemcpy(chunk_three_center.data(), device_three_center,
                            systems_in_chunk * three_center_elements * sizeof(double),
                            cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess) break;
    for (std::size_t local = 0; local < systems_in_chunk; ++local) {
      const std::size_t system = system_base + local;
      std::copy(chunk_metric.begin() + local * metric_elements,
                chunk_metric.begin() + (local + 1U) * metric_elements,
                outputs[system].metric.begin());
      std::copy(chunk_three_center.begin() + local * three_center_elements,
                chunk_three_center.begin() + (local + 1U) * three_center_elements,
                outputs[system].three_center.begin());
    }
  }

  for (std::size_t coordinate = 0;
       include_derivatives && cuda_error == cudaSuccess && coordinate < atom_count * 3U;
       ++coordinate) {
    for (std::size_t system_base = 0; cuda_error == cudaSuccess && system_base < batch_size;
         system_base += chunk_systems) {
      const std::size_t systems_in_chunk = std::min(chunk_systems, batch_size - system_base);
      cuda_error =
          launch_chunk(system_base, systems_in_chunk, static_cast<std::int64_t>(coordinate));
      if (cuda_error != cudaSuccess) break;
      cuda_error =
          cudaMemcpy(chunk_metric.data(), device_metric,
                     systems_in_chunk * metric_elements * sizeof(double), cudaMemcpyDeviceToHost);
      if (cuda_error != cudaSuccess) break;
      cuda_error = cudaMemcpy(chunk_three_center.data(), device_three_center,
                              systems_in_chunk * three_center_elements * sizeof(double),
                              cudaMemcpyDeviceToHost);
      if (cuda_error != cudaSuccess) break;
      for (std::size_t local = 0; local < systems_in_chunk; ++local) {
        const std::size_t system = system_base + local;
        std::copy(chunk_metric.begin() + local * metric_elements,
                  chunk_metric.begin() + (local + 1U) * metric_elements,
                  outputs[system].metric_derivative.begin() + coordinate * metric_elements);
        std::copy(
            chunk_three_center.begin() + local * three_center_elements,
            chunk_three_center.begin() + (local + 1U) * three_center_elements,
            outputs[system].three_center_derivative.begin() + coordinate * three_center_elements);
      }
    }
  }
  release();
  if (cuda_error != cudaSuccess) {
    outputs.clear();
    detail = "CUDA kernel failed while generating DF batch derivatives";
    return cuda_status(cuda_error);
  }
  return VIBEQC_STATUS_SUCCESS;
}

}  // namespace

vibeqc_status build_cuda_density_fitting_integrals_batch(
    int device_id, const std::vector<core::System>& orbital_systems,
    const std::vector<core::System>& auxiliary_systems,
    std::vector<integrals::DensityFittingIntegralData>& outputs, std::string& detail,
    std::size_t output_budget_bytes, bool include_derivatives) {
  for (const auto& system : auxiliary_systems) {
    if (!cuda_df_shell_domain(system, "auxiliary", detail)) return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  for (const auto& system : orbital_systems) {
    if (!cuda_df_shell_domain(system, "orbital", detail)) return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  return build_cuda_density_fitting_integrals_batch_impl(device_id, orbital_systems,
                                                         auxiliary_systems, outputs, detail,
                                                         output_budget_bytes, include_derivatives);
}

}  // namespace vibeqc::scf
