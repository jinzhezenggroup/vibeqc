#include <algorithm>
#include <array>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>

#include "molecule/basis.hpp"
#include "runtime/bounded_workspace.hpp"
#include "runtime/resource_usage.hpp"
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

/** Generate Cartesian DF tensors without constructing the full four-center ERI. */
vibeqc_status build_cuda_density_fitting_integrals_impl(
    int device_id, const core::System& orbital_system, const core::System& auxiliary_system,
    integrals::DensityFittingIntegralData& output, std::string& detail, bool include_derivatives) {
  unsigned value_math = 0, value_lanes = 1;
  if (!cuda_policy::df_value_raw_lanes_requested(value_lanes)) {
    detail = "VIBEQC_DF_VALUE_RAW_MAPPING must be auto, scalar, subgroup, warp or candidate";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (!cuda_policy::df_value_math_requested(value_math)) {
    detail = "VIBEQC_DF_VALUE_MATH must be auto, generic, polynomial rys or candidate";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (device_id < 0) {
    detail = "CUDA density-fitting integral generation received an invalid device";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (orbital_system.atoms.size() != auxiliary_system.atoms.size()) {
    detail = "orbital and auxiliary systems must share geometry";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  for (std::size_t atom = 0; atom < orbital_system.atoms.size(); ++atom) {
    if (orbital_system.atoms[atom].atomic_number != auxiliary_system.atoms[atom].atomic_number ||
        orbital_system.atoms[atom].position != auxiliary_system.atoms[atom].position) {
      detail = "orbital and auxiliary systems must share geometry";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
  }

  // Keep both bases Cartesian here.  The public spherical transform is a
  // separate, shared reference operation applied by the SCF preparation path.
  core::System combined;
  combined.atoms = orbital_system.atoms;
  combined.shells = orbital_system.shells;
  combined.shells.insert(combined.shells.end(), auxiliary_system.shells.begin(),
                         auxiliary_system.shells.end());
  // A zero-exponent s shell represents the implicit fourth center in a
  // three-/two-center Coulomb integral.  Its center is algebraically absent
  // from the result, but assigning atom zero keeps DeviceBatch well-formed.
  combined.shells.push_back({0, 0, {{0.0, 1.0}}});
  combined.charge = orbital_system.charge;
  combined.multiplicity = 1;
  combined.electron_count = 2;
  combined.basis_representation = VIBEQC_BASIS_CARTESIAN;

  HostBatch host;
  std::vector<const std::vector<double>*> no_warm(1, nullptr);
  if (!pack_host_batch({combined}, no_warm, host, false)) {
    detail = "combined Cartesian DF basis cannot be represented by CUDA";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t orbital_count = molecule::cartesian_ao_count(orbital_system);
  const std::size_t auxiliary_count = molecule::cartesian_ao_count(auxiliary_system);
  const std::size_t dummy_index = orbital_count + auxiliary_count;
  if (host.nbf != dummy_index + 1U || orbital_count == 0U || auxiliary_count == 0U) {
    detail = "Cartesian DF basis dimensions are inconsistent";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (auxiliary_count > std::numeric_limits<std::int32_t>::max() ||
      dummy_index > std::numeric_limits<std::int32_t>::max() ||
      host.nbf > std::numeric_limits<std::int32_t>::max()) {
    detail = "Cartesian DF basis exceeds CUDA index limits";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (orbital_count > std::numeric_limits<std::size_t>::max() / orbital_count ||
      orbital_count * orbital_count > std::numeric_limits<std::size_t>::max() / auxiliary_count) {
    detail = "CUDA DF tensor dimensions overflowed";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t metric_elements = auxiliary_count * auxiliary_count;
  const std::size_t three_center_elements = orbital_count * orbital_count * auxiliary_count;
  const std::size_t total_elements = metric_elements + three_center_elements;
  if (total_elements < metric_elements ||
      total_elements > std::numeric_limits<unsigned>::max() * static_cast<std::size_t>(128U)) {
    detail = "CUDA DF integral launch dimensions are too large";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }

  cudaError_t cuda_error = cudaSetDevice(device_id);
  if (cuda_error != cudaSuccess) {
    detail = "CUDA device selection failed while generating DF integrals";
    return cuda_status(cuda_error);
  }
  cudaStream_t stream = nullptr;
  cuda_error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (cuda_error != cudaSuccess) {
    detail = "CUDA stream creation failed while generating DF integrals";
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
  device_batch.batch_size = 1;
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
      detail = "CUDA allocation failed while staging DF basis metadata";
      release();
      return VIBEQC_STATUS_OUT_OF_MEMORY;
    }
  }

  double* device_metric = static_cast<double*>(upload(nullptr, metric_elements * sizeof(double)));
  double* device_three_center =
      static_cast<double*>(upload(nullptr, three_center_elements * sizeof(double)));
  if (device_metric == nullptr || device_three_center == nullptr) {
    detail = "CUDA allocation failed for DF integral output";
    release();
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }

  output = {};
  output.nbf = orbital_count;
  output.naux = auxiliary_count;
  output.ncoord = orbital_system.atoms.size() * 3U;
  output.metric.resize(metric_elements);
  output.three_center.resize(three_center_elements);
  if (include_derivatives) {
    output.metric_derivative.resize(output.ncoord * metric_elements);
    output.three_center_derivative.resize(output.ncoord * three_center_elements);
  }
  constexpr unsigned threads = 128U;
  const unsigned blocks = static_cast<unsigned>((total_elements + threads - 1U) / threads);
  launch_build_cuda_df_integrals_kernel(
      false, blocks, threads, 0, stream, device_batch, orbital_count, auxiliary_count, dummy_index,
      metric_elements, three_center_elements, 0, 1, -1, device_metric, device_three_center,
      value_math, value_lanes);
  cuda_error = cudaGetLastError();
  if (cuda_error == cudaSuccess) cuda_error = cudaStreamSynchronize(stream);
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpy(output.metric.data(), device_metric, metric_elements * sizeof(double),
                            cudaMemcpyDeviceToHost);
  }
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpy(output.three_center.data(), device_three_center,
                            three_center_elements * sizeof(double), cudaMemcpyDeviceToHost);
  }
  for (std::size_t coordinate = 0;
       include_derivatives && cuda_error == cudaSuccess && coordinate < output.ncoord;
       ++coordinate) {
    launch_build_cuda_df_integrals_kernel(
        true, blocks, threads, 0, stream, device_batch, orbital_count, auxiliary_count, dummy_index,
        metric_elements, three_center_elements, 0, 1, static_cast<std::int64_t>(coordinate),
        device_metric, device_three_center, value_math, value_lanes);
    cuda_error = cudaGetLastError();
    if (cuda_error == cudaSuccess) cuda_error = cudaStreamSynchronize(stream);
    if (cuda_error == cudaSuccess) {
      cuda_error =
          cudaMemcpy(output.metric_derivative.data() + coordinate * metric_elements, device_metric,
                     metric_elements * sizeof(double), cudaMemcpyDeviceToHost);
    }
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpy(
          output.three_center_derivative.data() + coordinate * three_center_elements,
          device_three_center, three_center_elements * sizeof(double), cudaMemcpyDeviceToHost);
    }
  }
  if (cuda_error != cudaSuccess) {
    detail = "CUDA kernel failed while generating DF integrals";
    release();
    return cuda_status(cuda_error);
  }
  release();
  return VIBEQC_STATUS_SUCCESS;
}

}  // namespace

vibeqc_status build_cuda_density_fitting_integrals(int device_id,
                                                   const core::System& orbital_system,
                                                   const core::System& auxiliary_system,
                                                   integrals::DensityFittingIntegralData& output,
                                                   std::string& detail, bool include_derivatives) {
  if (!cuda_df_shell_domain(auxiliary_system, "auxiliary", detail) ||
      !cuda_df_shell_domain(orbital_system, "orbital", detail)) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  return build_cuda_density_fitting_integrals_impl(device_id, orbital_system, auxiliary_system,
                                                   output, detail, include_derivatives);
}

}  // namespace vibeqc::scf
