#include "scf/cuda_fock.hpp"

#include <cuda_runtime_api.h>

#include <limits>
#include <new>

namespace qce::scf {
namespace {

__global__ void rhf_fock_bucket_kernel(std::size_t batch_size,
                                       std::size_t n,
                                       const double* hcore,
                                       const double* eri,
                                       const double* density,
                                       double* fock) {
  const std::size_t matrix_size = n * n;
  const std::size_t eri_size = matrix_size * matrix_size;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= batch_size * matrix_size) return;
  const std::size_t system = element / matrix_size;
  const std::size_t local = element % matrix_size;
  const std::size_t i = local / n;
  const std::size_t j = local % n;
  const double* system_density = density + system * matrix_size;
  const double* system_eri = eri + system * eri_size;
  double coulomb = 0.0;
  double exchange = 0.0;
  for (std::size_t k = 0; k < n; ++k) {
    for (std::size_t l = 0; l < n; ++l) {
      const double pkl = system_density[k * n + l];
      const std::size_t coulomb_index = ((i * n + j) * n + k) * n + l;
      const std::size_t exchange_index = ((i * n + k) * n + j) * n + l;
      coulomb += pkl * system_eri[coulomb_index];
      exchange += pkl * system_eri[exchange_index];
    }
  }
  fock[element] = hcore[element] + coulomb - 0.5 * exchange;
}

}  // namespace

struct CudaFockBucketHandle {
  int device_id{};
  std::size_t batch_size{};
  std::size_t nbf{};
  cudaStream_t stream{};
  double* hcore{};
  double* eri{};
  double* density{};
  double* fock{};
};

namespace {

qce_status cuda_failure(cudaError_t error, std::string& detail) {
  detail = cudaGetErrorString(error);
  return error == cudaErrorMemoryAllocation ? QCE_STATUS_OUT_OF_MEMORY
                                             : QCE_STATUS_CUDA_ERROR;
}

void release(CudaFockBucketHandle& handle) {
  cudaSetDevice(handle.device_id);
  cudaFree(handle.hcore);
  cudaFree(handle.eri);
  cudaFree(handle.density);
  cudaFree(handle.fock);
  if (handle.stream != nullptr) cudaStreamDestroy(handle.stream);
  handle = {};
}

}  // namespace

qce_status create_cuda_fock_bucket(int device_id,
                                   std::size_t batch_size,
                                   std::size_t nbf,
                                   const std::vector<double>& hcore,
                                   const std::vector<double>& eri,
                                   CudaFockBucketHandle** handle,
                                   std::string& detail) {
  if (handle == nullptr) return QCE_STATUS_INVALID_ARGUMENT;
  *handle = nullptr;
  if (batch_size == 0 || nbf == 0) return QCE_STATUS_INVALID_ARGUMENT;
  const std::size_t matrix_size = nbf * nbf;
  const std::size_t eri_size = matrix_size * matrix_size;
  if (hcore.size() != batch_size * matrix_size ||
      eri.size() != batch_size * eri_size) {
    detail = "CUDA Fock bucket host buffers have incompatible dimensions";
    return QCE_STATUS_INVALID_ARGUMENT;
  }
  if (batch_size > std::numeric_limits<std::size_t>::max() / matrix_size) {
    return QCE_STATUS_OUT_OF_MEMORY;
  }
  cudaError_t error = cudaSetDevice(device_id);
  if (error != cudaSuccess) return cuda_failure(error, detail);

  CudaFockBucketHandle* candidate = new (std::nothrow) CudaFockBucketHandle{};
  if (candidate == nullptr) return QCE_STATUS_OUT_OF_MEMORY;
  candidate->device_id = device_id;
  candidate->batch_size = batch_size;
  candidate->nbf = nbf;
  const std::size_t matrix_bytes = batch_size * matrix_size * sizeof(double);
  const std::size_t eri_bytes = batch_size * eri_size * sizeof(double);
  if ((error = cudaStreamCreateWithFlags(&candidate->stream,
                                         cudaStreamNonBlocking)) != cudaSuccess ||
      (error = cudaMalloc(reinterpret_cast<void**>(&candidate->hcore),
                          matrix_bytes)) != cudaSuccess ||
      (error = cudaMalloc(reinterpret_cast<void**>(&candidate->eri),
                          eri_bytes)) != cudaSuccess ||
      (error = cudaMalloc(reinterpret_cast<void**>(&candidate->density),
                          matrix_bytes)) != cudaSuccess ||
      (error = cudaMalloc(reinterpret_cast<void**>(&candidate->fock),
                          matrix_bytes)) != cudaSuccess) {
    release(*candidate);
    delete candidate;
    return cuda_failure(error, detail);
  }
  if ((error = cudaMemcpyAsync(candidate->hcore, hcore.data(), matrix_bytes,
                               cudaMemcpyHostToDevice,
                               candidate->stream)) != cudaSuccess ||
      (error = cudaMemcpyAsync(candidate->eri, eri.data(), eri_bytes,
                               cudaMemcpyHostToDevice,
                               candidate->stream)) != cudaSuccess ||
      (error = cudaStreamSynchronize(candidate->stream)) != cudaSuccess) {
    release(*candidate);
    delete candidate;
    return cuda_failure(error, detail);
  }
  *handle = candidate;
  return QCE_STATUS_SUCCESS;
}

qce_status execute_cuda_fock_bucket(CudaFockBucketHandle* handle,
                                    const std::vector<double>& density,
                                    std::vector<double>& fock,
                                    std::string& detail) {
  if (handle == nullptr) return QCE_STATUS_INVALID_ARGUMENT;
  const std::size_t matrix_size = handle->nbf * handle->nbf;
  const std::size_t elements = handle->batch_size * matrix_size;
  if (density.size() != elements || fock.size() != elements) {
    detail = "CUDA Fock bucket iteration buffers have incompatible dimensions";
    return QCE_STATUS_INVALID_ARGUMENT;
  }
  cudaError_t error = cudaSetDevice(handle->device_id);
  if (error != cudaSuccess) return cuda_failure(error, detail);
  const std::size_t matrix_bytes = elements * sizeof(double);
  if ((error = cudaMemcpyAsync(handle->density, density.data(), matrix_bytes,
                               cudaMemcpyHostToDevice,
                               handle->stream)) != cudaSuccess) {
    return cuda_failure(error, detail);
  }

  constexpr unsigned threads = 256;
  const unsigned blocks = static_cast<unsigned>((elements + threads - 1) / threads);
  rhf_fock_bucket_kernel<<<blocks, threads, 0, handle->stream>>>(
      handle->batch_size, handle->nbf, handle->hcore, handle->eri,
      handle->density, handle->fock);
  if ((error = cudaGetLastError()) != cudaSuccess ||
      (error = cudaMemcpyAsync(fock.data(), handle->fock, matrix_bytes,
                               cudaMemcpyDeviceToHost,
                               handle->stream)) != cudaSuccess ||
      (error = cudaStreamSynchronize(handle->stream)) != cudaSuccess) {
    return cuda_failure(error, detail);
  }
  return QCE_STATUS_SUCCESS;
}

void destroy_cuda_fock_bucket(CudaFockBucketHandle* handle) {
  if (handle == nullptr) return;
  release(*handle);
  delete handle;
}

}  // namespace qce::scf
