#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#include "integrals/ecp_cuda.hpp"
#include "molecule/basis.hpp"
#include "runtime/cuda_component_trace.hpp"
#include "runtime/resource_cuda.cuh"
#include "scf/cuda/one_electron_export_kernels.hpp"
#include "scf/cuda/one_electron_view.hpp"
#include "scf/cuda/rhf_policy.hpp"
#include "scf/cuda/runtime_support.hpp"
#include "scf/cuda/topology.hpp"
#include "scf/cuda_density_fitting_integrals.hpp"

namespace vibeqc::scf {
namespace {
using namespace cuda_execution;

/** Stage explicit host tensors while retaining coordinate/spin and failure semantics. */
vibeqc_status build_cuda_one_electron_integrals_batch_impl(
    int device_id, const std::vector<core::System>& systems,
    std::vector<integrals::IntegralData>& outputs, std::string& detail, bool include_derivatives) {
  outputs.clear();
  if (device_id < 0 || systems.empty()) {
    detail = "CUDA one-electron integral batch dimensions are invalid";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t batch_size = systems.size();
  if (batch_size > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    detail = "CUDA one-electron batch exceeds the supported system count";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  std::vector<core::System> cartesian_systems;
  try {
    cartesian_systems.reserve(batch_size);
    for (const core::System& system : systems) {
      core::System cartesian = system;
      cartesian.basis_representation = VIBEQC_BASIS_CARTESIAN;
      cartesian_systems.push_back(std::move(cartesian));
    }
  } catch (const std::bad_alloc&) {
    detail = "host allocation failed while staging one-electron batch";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }

  HostBatch host;
  std::vector<const std::vector<double>*> no_warm(batch_size, nullptr);
  // Match the spin-independent single-system evaluator for open-shell fleets.
  if (!pack_host_batch(cartesian_systems, no_warm, host, true) || host.nbf == 0U) {
    detail = "Cartesian one-electron batch cannot be represented by CUDA";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t nbf = host.nbf;
  for (const core::System& system : cartesian_systems) {
    if (molecule::cartesian_ao_count(system) != nbf ||
        system.atoms.size() != systems.front().atoms.size()) {
      detail = "CUDA one-electron batch requires homogeneous AO dimensions";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
  }

  if (nbf > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
      nbf > std::numeric_limits<std::size_t>::max() / nbf) {
    detail = "Cartesian one-electron batch dimensions are invalid";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t matrix_elements = nbf * nbf;
  const std::size_t pair_count = nbf * (nbf + 1U) / 2U;
  if (batch_size > std::numeric_limits<std::size_t>::max() / pair_count ||
      batch_size > std::numeric_limits<std::size_t>::max() / matrix_elements) {
    detail = "CUDA one-electron batch dimensions overflowed";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t pair_launch_elements = batch_size * pair_count;
  const std::size_t matrix_batch_elements = batch_size * matrix_elements;
  if (pair_launch_elements >
          std::numeric_limits<unsigned>::max() * static_cast<std::size_t>(128U) ||
      matrix_batch_elements > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
    detail = "CUDA one-electron batch launch dimensions are too large";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }

  std::vector<std::int32_t> pair_first;
  std::vector<std::int32_t> pair_second;
  try {
    pair_first.reserve(pair_count);
    pair_second.reserve(pair_count);
    for (std::size_t row = 0; row < nbf; ++row) {
      for (std::size_t column = 0; column <= row; ++column) {
        pair_first.push_back(static_cast<std::int32_t>(row));
        pair_second.push_back(static_cast<std::int32_t>(column));
      }
    }
  } catch (const std::bad_alloc&) {
    detail = "host allocation failed for one-electron batch pair indices";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }

  cudaError_t cuda_error = cudaSetDevice(device_id);
  if (cuda_error != cudaSuccess) {
    detail = "CUDA device selection failed while generating one-electron batch";
    return cuda_status(cuda_error);
  }
  cudaStream_t stream = nullptr;
  cuda_error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (cuda_error != cudaSuccess) {
    detail = "CUDA stream creation failed while generating one-electron batch";
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
  device_batch.total_shell_pairs = static_cast<std::int64_t>(host.shell_pair_first.size());
  device_batch.shell_pair_first =
      static_cast<const std::int32_t*>(upload_vector(host.shell_pair_first));
  device_batch.shell_pair_second =
      static_cast<const std::int32_t*>(upload_vector(host.shell_pair_second));
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
  const std::array<const void*, 20> metadata{device_batch.shell_pair_first,
                                             device_batch.shell_pair_second,
                                             device_batch.atom_offsets,
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
      detail = "CUDA allocation failed while staging one-electron batch metadata";
      release();
      return VIBEQC_STATUS_OUT_OF_MEMORY;
    }
  }
  const auto* device_pair_first = static_cast<const std::int32_t*>(upload_vector(pair_first));
  const auto* device_pair_second = static_cast<const std::int32_t*>(upload_vector(pair_second));
  double* device_overlap =
      static_cast<double*>(upload(nullptr, matrix_batch_elements * sizeof(double)));
  double* device_hcore =
      static_cast<double*>(upload(nullptr, matrix_batch_elements * sizeof(double)));
  double* device_nuclear = static_cast<double*>(upload(nullptr, batch_size * sizeof(double)));
  if (device_pair_first == nullptr || device_pair_second == nullptr || device_overlap == nullptr ||
      device_hcore == nullptr || device_nuclear == nullptr) {
    detail = "CUDA allocation failed for one-electron batch output";
    release();
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }

  try {
    outputs.resize(batch_size);
    for (integrals::IntegralData& output : outputs) {
      output.nbf = nbf;
      output.ncoord = systems.front().atoms.size() * 3U;
      output.overlap.resize(matrix_elements);
      output.hcore.resize(matrix_elements);
      if (include_derivatives) output.overlap_derivative.resize(output.ncoord * matrix_elements);
      if (include_derivatives) output.hcore_derivative.resize(output.ncoord * matrix_elements);
      output.nuclear_repulsion_derivative.resize(output.ncoord);
    }
  } catch (const std::bad_alloc&) {
    detail = "host allocation failed for one-electron batch output";
    release();
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }

  std::vector<double> packed_overlap;
  std::vector<double> packed_hcore;
  std::vector<double> packed_nuclear;
  try {
    packed_overlap.resize(matrix_batch_elements);
    packed_hcore.resize(matrix_batch_elements);
    packed_nuclear.resize(batch_size);
  } catch (const std::bad_alloc&) {
    detail = "host allocation failed for one-electron batch staging";
    release();
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }

  constexpr unsigned threads = 128U;
  const unsigned blocks = static_cast<unsigned>((pair_launch_elements + threads - 1U) / threads);
  cuda_error = launch_generated_one_electron_values(
      one_electron_view(device_batch), device_pair_first, device_pair_second, pair_count,
      cuda_policy::one_electron_value_mapping_requested(), device_overlap, device_hcore, stream);
  if (cuda_error != cudaSuccess) {
    detail = "generated CUDA one-electron value launch failed";
    release();
    return cuda_status(cuda_error);
  }
  launch_build_cuda_nuclear_repulsion_kernel(
      false, static_cast<unsigned>((batch_size + threads - 1U) / threads), threads, 0, stream,
      device_batch, -1, device_nuclear);
  cuda_error = cudaGetLastError();
  if (cuda_error == cudaSuccess) cuda_error = cudaStreamSynchronize(stream);
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpy(packed_overlap.data(), device_overlap,
                            matrix_batch_elements * sizeof(double), cudaMemcpyDeviceToHost);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpy(packed_hcore.data(), device_hcore,
                              matrix_batch_elements * sizeof(double), cudaMemcpyDeviceToHost);
    }
    if (cuda_error == cudaSuccess) {
      for (std::size_t system = 0; system < batch_size; ++system) {
        std::copy(packed_overlap.begin() + system * matrix_elements,
                  packed_overlap.begin() + (system + 1U) * matrix_elements,
                  outputs[system].overlap.begin());
        std::copy(packed_hcore.begin() + system * matrix_elements,
                  packed_hcore.begin() + (system + 1U) * matrix_elements,
                  outputs[system].hcore.begin());
      }
    }
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpy(packed_nuclear.data(), device_nuclear, batch_size * sizeof(double),
                              cudaMemcpyDeviceToHost);
      if (cuda_error == cudaSuccess) {
        for (std::size_t system = 0; system < batch_size; ++system) {
          outputs[system].nuclear_repulsion = packed_nuclear[system];
        }
      }
    }
  }
  if (cuda_error == cudaSuccess) {
    runtime::cuda_trace::TraceOperation trace(
        include_derivatives ? "one_electron_derivative_export" : "nuclear_derivative_export",
        stream, {batch_size, nbf, 0, false, false});
    for (std::size_t coordinate = 0; coordinate < systems.front().atoms.size() * 3U; ++coordinate) {
      runtime::cuda_trace::TraceRegion generation("one_electron_and_nuclear_derivative_generation",
                                                  stream);
      if (include_derivatives)
        launch_build_cuda_one_electron_derivatives_kernel(
            blocks, threads, 0, stream, device_batch, device_pair_first, device_pair_second,
            pair_count, static_cast<std::int64_t>(coordinate), device_overlap, device_hcore);
      launch_build_cuda_nuclear_repulsion_kernel(
          true, static_cast<unsigned>((batch_size + threads - 1U) / threads), threads, 0, stream,
          device_batch, static_cast<std::int64_t>(coordinate), device_nuclear);
      generation.finish();
      runtime::cuda_trace::trace_counter("atom_coordinates", batch_size);
      runtime::cuda_trace::trace_counter(
          "device_to_host_bytes",
          ((include_derivatives ? 2 * matrix_batch_elements : 0) + batch_size) * sizeof(double));
      runtime::cuda_trace::TraceRegion transfer(
          "one_electron_derivative_output_and_synchronization", stream);
      cuda_error = cudaGetLastError();
      if (cuda_error == cudaSuccess) cuda_error = cudaStreamSynchronize(stream);
      if (cuda_error != cudaSuccess) break;
      if (include_derivatives) {
        cuda_error = cudaMemcpy(packed_overlap.data(), device_overlap,
                                matrix_batch_elements * sizeof(double), cudaMemcpyDeviceToHost);
        if (cuda_error == cudaSuccess) {
          cuda_error = cudaMemcpy(packed_hcore.data(), device_hcore,
                                  matrix_batch_elements * sizeof(double), cudaMemcpyDeviceToHost);
        }
      }
      if (cuda_error == cudaSuccess) {
        cuda_error = cudaMemcpy(packed_nuclear.data(), device_nuclear, batch_size * sizeof(double),
                                cudaMemcpyDeviceToHost);
      }
      if (cuda_error != cudaSuccess) break;
      for (std::size_t system = 0; system < batch_size; ++system) {
        if (include_derivatives) {
          std::copy(packed_overlap.begin() + system * matrix_elements,
                    packed_overlap.begin() + (system + 1U) * matrix_elements,
                    outputs[system].overlap_derivative.begin() + coordinate * matrix_elements);
          std::copy(packed_hcore.begin() + system * matrix_elements,
                    packed_hcore.begin() + (system + 1U) * matrix_elements,
                    outputs[system].hcore_derivative.begin() + coordinate * matrix_elements);
        }
        outputs[system].nuclear_repulsion_derivative[coordinate] = packed_nuclear[system];
      }
    }
  }
  release();
  if (cuda_error != cudaSuccess) {
    outputs.clear();
    detail = "CUDA kernel failed while generating one-electron batch";
    return cuda_status(cuda_error);
  }
  for (std::size_t i = 0; i < systems.size(); ++i) {
    if (systems[i].ecp_terms.empty()) continue;
    integrals::EcpData ecp;
    const auto status = integrals::ecp_integrals_cuda(device_id, cartesian_systems[i], 160, 32,
                                                      include_derivatives, ecp, detail, true);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    integrals::add_ecp(ecp, outputs[i].hcore, outputs[i].hcore_derivative);
  }
  return VIBEQC_STATUS_SUCCESS;
}

}  // namespace

vibeqc_status build_cuda_one_electron_integrals_batch(int device_id,
                                                      const std::vector<core::System>& systems,
                                                      std::vector<integrals::IntegralData>& outputs,
                                                      std::string& detail,
                                                      bool include_derivatives) {
  return build_cuda_one_electron_integrals_batch_impl(device_id, systems, outputs, detail,
                                                      include_derivatives);
}

}  // namespace vibeqc::scf
