// Included inside cuda_rhf.cu's implementation namespace after its staging helpers.
// Keep single-system DF one-electron setup beside its value/response controls;
// this also bounds the size of the main CUDA translation unit's source file.
#ifndef VIBEQC_SCF_CUDA_ONE_ELECTRON_INTEGRALS_CUH
#define VIBEQC_SCF_CUDA_ONE_ELECTRON_INTEGRALS_CUH

vibeqc_status build_cuda_one_electron_integrals_impl(int device_id, const core::System& system,
                                                     integrals::IntegralData& output,
                                                     std::string& detail, bool include_derivatives,
                                                     bool include_nuclear_derivatives) {
  if (device_id < 0) {
    detail = "CUDA one-electron integral generation received an invalid device";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  core::System cartesian_system = system;
  cartesian_system.basis_representation = VIBEQC_BASIS_CARTESIAN;
  HostBatch host;
  std::vector<const std::vector<double>*> no_warm(1, nullptr);
  // One-electron integrals are spin independent. General spin packing accepts
  // both RHF and UHF systems; closed-shell packing incorrectly rejects the
  // odd-electron orbital metadata needed by an open-shell DF endpoint.
  if (!pack_host_batch({cartesian_system}, no_warm, host, true) || host.nbf == 0U) {
    detail = "Cartesian one-electron basis cannot be represented by CUDA";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  if (host.nbf > std::numeric_limits<std::int32_t>::max() ||
      host.nbf > std::numeric_limits<std::size_t>::max() / host.nbf) {
    detail = "Cartesian one-electron basis dimensions are invalid";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t matrix_elements = host.nbf * host.nbf;
  const std::size_t pair_count = host.nbf * (host.nbf + 1U) / 2U;
  if (pair_count > std::numeric_limits<unsigned>::max() * static_cast<std::size_t>(128U)) {
    detail = "CUDA one-electron launch dimensions are too large";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  std::vector<std::int32_t> pair_first;
  std::vector<std::int32_t> pair_second;
  pair_first.reserve(pair_count);
  pair_second.reserve(pair_count);
  for (std::size_t row = 0; row < host.nbf; ++row) {
    for (std::size_t column = 0; column <= row; ++column) {
      pair_first.push_back(static_cast<std::int32_t>(row));
      pair_second.push_back(static_cast<std::int32_t>(column));
    }
  }

  cudaError_t cuda_error = cudaSetDevice(device_id);
  if (cuda_error != cudaSuccess) {
    detail = "CUDA device selection failed while generating one-electron integrals";
    return cuda_status(cuda_error);
  }
  cudaStream_t stream = nullptr;
  cuda_error = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (cuda_error != cudaSuccess) {
    detail = "CUDA stream creation failed while generating one-electron integrals";
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
      detail = "CUDA allocation failed while staging one-electron metadata";
      release();
      return VIBEQC_STATUS_OUT_OF_MEMORY;
    }
  }
  const auto* device_pair_first = static_cast<const std::int32_t*>(upload_vector(pair_first));
  const auto* device_pair_second = static_cast<const std::int32_t*>(upload_vector(pair_second));
  double* device_overlap = static_cast<double*>(upload(nullptr, matrix_elements * sizeof(double)));
  double* device_hcore = static_cast<double*>(upload(nullptr, matrix_elements * sizeof(double)));
  double* device_nuclear = static_cast<double*>(upload(nullptr, sizeof(double)));
  if (device_pair_first == nullptr || device_pair_second == nullptr || device_overlap == nullptr ||
      device_hcore == nullptr || device_nuclear == nullptr) {
    detail = "CUDA allocation failed for one-electron integral output";
    release();
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }

  output = {};
  output.nbf = host.nbf;
  output.ncoord = system.atoms.size() * 3U;
  output.overlap.resize(matrix_elements);
  output.hcore.resize(matrix_elements);
  if (include_derivatives) output.overlap_derivative.resize(output.ncoord * matrix_elements);
  if (include_derivatives) output.hcore_derivative.resize(output.ncoord * matrix_elements);
  if (include_nuclear_derivatives) output.nuclear_repulsion_derivative.resize(output.ncoord);
  constexpr unsigned threads = 128U;
  const unsigned pair_blocks = static_cast<unsigned>((pair_count + threads - 1U) / threads);
  cuda_error = launch_generated_one_electron_values(
      one_electron_view(device_batch), device_pair_first, device_pair_second, pair_count,
      cuda_policy::one_electron_value_mapping_requested(), device_overlap, device_hcore, stream);
  if (cuda_error != cudaSuccess) {
    detail = "generated CUDA one-electron value launch failed";
    release();
    return cuda_status(cuda_error);
  }
  build_cuda_nuclear_repulsion_kernel<false><<<1, 1, 0, stream>>>(device_batch, -1, device_nuclear);
  cuda_error = cudaGetLastError();
  if (cuda_error == cudaSuccess) cuda_error = cudaStreamSynchronize(stream);
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpy(output.overlap.data(), device_overlap, matrix_elements * sizeof(double),
                            cudaMemcpyDeviceToHost);
  }
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpy(output.hcore.data(), device_hcore, matrix_elements * sizeof(double),
                            cudaMemcpyDeviceToHost);
  }
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpy(&output.nuclear_repulsion, device_nuclear, sizeof(double),
                            cudaMemcpyDeviceToHost);
  }
  for (std::size_t coordinate = 0; (include_derivatives || include_nuclear_derivatives) &&
                                   cuda_error == cudaSuccess && coordinate < output.ncoord;
       ++coordinate) {
    if (include_derivatives)
      build_cuda_one_electron_derivatives_kernel<<<pair_blocks, threads, 0, stream>>>(
          device_batch, device_pair_first, device_pair_second, pair_count,
          static_cast<std::int64_t>(coordinate), device_overlap, device_hcore);
    if (include_nuclear_derivatives)
      build_cuda_nuclear_repulsion_kernel<true><<<1, 1, 0, stream>>>(
          device_batch, static_cast<std::int64_t>(coordinate), device_nuclear);
    cuda_error = cudaGetLastError();
    if (cuda_error == cudaSuccess) cuda_error = cudaStreamSynchronize(stream);
    if (cuda_error == cudaSuccess && include_derivatives) {
      cuda_error =
          cudaMemcpy(output.overlap_derivative.data() + coordinate * matrix_elements,
                     device_overlap, matrix_elements * sizeof(double), cudaMemcpyDeviceToHost);
    }
    if (cuda_error == cudaSuccess && include_derivatives) {
      cuda_error =
          cudaMemcpy(output.hcore_derivative.data() + coordinate * matrix_elements, device_hcore,
                     matrix_elements * sizeof(double), cudaMemcpyDeviceToHost);
    }
    if (cuda_error == cudaSuccess && include_nuclear_derivatives) {
      cuda_error = cudaMemcpy(output.nuclear_repulsion_derivative.data() + coordinate,
                              device_nuclear, sizeof(double), cudaMemcpyDeviceToHost);
    }
  }
  if (cuda_error != cudaSuccess) {
    detail = "CUDA kernel failed while generating one-electron integrals";
    release();
    return cuda_status(cuda_error);
  }
  release();
  return VIBEQC_STATUS_SUCCESS;
}

#endif
