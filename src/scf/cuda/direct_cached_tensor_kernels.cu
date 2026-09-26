#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "scf/cuda/direct_cached_tensor_kernels.hpp"
#include "scf/cuda/direct_native_contraction.cuh"
#include "scf/cuda/eri_tensor_index.cuh"
#include "scf/cuda/matrix_index.cuh"
#include "scf/cuda/packed_basis.hpp"

namespace vibeqc::scf::cuda_execution {

__global__ void build_eri_kernel(DeviceBatch batch, double* eri) {
  const std::size_t n = static_cast<std::size_t>(batch.nbf);
  const std::size_t eri_size = n * n * n * n;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch.batch_size) * eri_size) return;
  const std::int32_t system = static_cast<std::int32_t>(element / eri_size);
  std::size_t local = element % eri_size;
  const std::int32_t l = static_cast<std::int32_t>(local % n);
  local /= n;
  const std::int32_t k = static_cast<std::int32_t>(local % n);
  local /= n;
  const std::int32_t j = static_cast<std::int32_t>(local % n);
  const std::int32_t i = static_cast<std::int32_t>(local / n);
  eri[element] = contracted_eri<double>(batch, system, i, j, k, l, -1);
}

__global__ void build_fock_kernel(std::int32_t batch_size, std::int32_t nbf, const double* hcore,
                                  const double* eri, const double* density,
                                  const std::uint8_t* active, double* fock) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t eri_size = matrix_size * matrix_size;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch_size) * matrix_size) return;
  const std::int32_t system = static_cast<std::int32_t>(element / matrix_size);
  if (active != nullptr && active[system] == 0) return;
  const std::size_t local = element % matrix_size;
  const std::size_t i = local % n;
  const std::size_t j = local / n;
  const std::size_t matrix_offset = static_cast<std::size_t>(system) * matrix_size;
  const std::size_t eri_offset = static_cast<std::size_t>(system) * eri_size;
  double coulomb = 0.0;
  double exchange = 0.0;
  for (std::size_t k = 0; k < n; ++k) {
    for (std::size_t l = 0; l < n; ++l) {
      const double pkl = density[matrix_offset + matrix_index(k, l, n)];
      coulomb += pkl * eri[eri_offset + eri_index(i, j, k, l, n)];
      exchange += pkl * eri[eri_offset + eri_index(i, k, j, l, n)];
    }
  }
  fock[element] = hcore[element] + coulomb - 0.5 * exchange;
}

__global__ void build_uhf_fock_kernel(std::int32_t batch_size, std::int32_t nbf,
                                      const double* hcore, const double* eri, const double* density,
                                      const std::uint8_t* active, double* fock) {
  const std::size_t n = static_cast<std::size_t>(nbf);
  const std::size_t matrix_size = n * n;
  const std::size_t eri_size = matrix_size * matrix_size;
  const std::size_t element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= static_cast<std::size_t>(batch_size) * 2 * matrix_size) return;
  const std::size_t state = element / matrix_size;
  const std::size_t system = state / 2;
  if (active != nullptr && active[system] == 0) return;
  const std::size_t spin = state % 2;
  const std::size_t local = element % matrix_size;
  const std::size_t i = local % n;
  const std::size_t j = local / n;
  const std::size_t physical_matrix_offset = system * matrix_size;
  const std::size_t eri_offset = system * eri_size;
  const std::size_t alpha_offset = system * 2 * matrix_size;
  const std::size_t beta_offset = alpha_offset + matrix_size;
  const std::size_t spin_offset = alpha_offset + spin * matrix_size;
  double coulomb = 0.0;
  double exchange = 0.0;
  for (std::size_t k = 0; k < n; ++k) {
    for (std::size_t l = 0; l < n; ++l) {
      const std::size_t kl = matrix_index(k, l, n);
      const double total = density[alpha_offset + kl] + density[beta_offset + kl];
      coulomb += total * eri[eri_offset + eri_index(i, j, k, l, n)];
      exchange += density[spin_offset + kl] * eri[eri_offset + eri_index(i, k, j, l, n)];
    }
  }
  fock[element] = hcore[physical_matrix_offset + local] + coulomb - exchange;
}

void launch_build_eri_kernel(dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
                             DeviceBatch batch, double* eri) {
  build_eri_kernel<<<grid, block, shared_bytes, stream>>>(batch, eri);
}

void launch_build_fock_kernel(dim3 grid, dim3 block, std::size_t shared_bytes, cudaStream_t stream,
                              std::int32_t batch_size, std::int32_t nbf, const double* hcore,
                              const double* eri, const double* density, const std::uint8_t* active,
                              double* fock) {
  build_fock_kernel<<<grid, block, shared_bytes, stream>>>(batch_size, nbf, hcore, eri, density,
                                                           active, fock);
}

void launch_build_uhf_fock_kernel(dim3 grid, dim3 block, std::size_t shared_bytes,
                                  cudaStream_t stream, std::int32_t batch_size, std::int32_t nbf,
                                  const double* hcore, const double* eri, const double* density,
                                  const std::uint8_t* active, double* fock) {
  build_uhf_fock_kernel<<<grid, block, shared_bytes, stream>>>(batch_size, nbf, hcore, eri, density,
                                                               active, fock);
}

}  // namespace vibeqc::scf::cuda_execution
