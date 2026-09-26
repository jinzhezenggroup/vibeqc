#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>

#include "runtime/cuda_resources.cuh"

namespace vibeqc::integrals::hvp_assembly {

struct NumericalFailure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline std::size_t mul(std::size_t a, std::size_t b) {
  if (a && b > std::numeric_limits<std::size_t>::max() / a)
    throw std::invalid_argument("HVP assembly size overflow");
  return a * b;
}

inline unsigned blocks(std::size_t count, unsigned threads) {
  return static_cast<unsigned>(std::min<std::size_t>((count + threads - 1) / threads, 65535));
}

inline void validate_target(int device, int major, int minor) {
  vibeqc::runtime::CudaDeviceScope scope(device);
  cudaDeviceProp properties{};
  vibeqc::runtime::cuda_resource_check(cudaGetDeviceProperties(&properties, device));
  if (properties.major != major || properties.minor != minor)
    throw std::invalid_argument("HVP assembly CUDA target/device mismatch");
}

inline void detail_text(char* detail, std::size_t size, const char* message) {
  if (detail && size) std::snprintf(detail, size, "%s", message ? message : "");
}

struct Mapping {
  std::uint32_t output_indices[12]{};
  std::uint32_t center_atoms[4]{};
  std::uint32_t source_count{};
  std::uint32_t center_count{};
};

__global__ void nuclear_kernel(const double* coords, const double* charges, const double* direction,
                               std::size_t natoms, double* output, int* error) {
  for (std::size_t a = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; a < natoms;
       a += std::size_t(blockDim.x) * gridDim.x) {
    for (std::size_t b = a + 1; b < natoms; ++b) {
      double r[3]{};
      double dv[3]{};
      double d2 = 0.0;
      for (unsigned axis = 0; axis < 3; ++axis) {
        r[axis] = coords[3 * a + axis] - coords[3 * b + axis];
        dv[axis] = direction[3 * a + axis] - direction[3 * b + axis];
        d2 += r[axis] * r[axis];
      }
      if (!(d2 > 0.0) || !isfinite(d2)) {
        atomicCAS(error, 0, 1);
        continue;
      }
      const double distance = sqrt(d2);
      const double prefactor = charges[a] * charges[b] / (d2 * distance);
      double projection = 0.0;
      for (unsigned axis = 0; axis < 3; ++axis) projection += (r[axis] / distance) * dv[axis];
      for (unsigned axis = 0; axis < 3; ++axis) {
        const double value = prefactor * (3.0 * (r[axis] / distance) * projection - dv[axis]);
        if (!isfinite(value)) {
          atomicCAS(error, 0, 1);
          continue;
        }
        atomicAdd(output + 3 * a + axis, value);
        atomicAdd(output + 3 * b + axis, -value);
      }
    }
  }
}

__global__ void add_dense_kernel(const double* source, std::size_t count, double coefficient,
                                 double* output, int* error) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += std::size_t(blockDim.x) * gridDim.x) {
    const double value = coefficient * source[i];
    if (!isfinite(value)) {
      atomicCAS(error, 0, 1);
      continue;
    }
    atomicAdd(output + i, value);
  }
}

__global__ void scatter_kernel(const double* source, Mapping mapping, double coefficient,
                               double* output, int* error) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < mapping.source_count;
       i += std::size_t(blockDim.x) * gridDim.x) {
    const auto coordinate = mapping.output_indices[i];
    const auto center = coordinate / 3;
    const auto axis = coordinate % 3;
    if (center >= mapping.center_count) {
      atomicCAS(error, 0, 1);
      continue;
    }
    const auto atom = mapping.center_atoms[center];
    const double value = coefficient * source[i];
    if (!isfinite(value)) {
      atomicCAS(error, 0, 1);
      continue;
    }
    atomicAdd(output + 3 * atom + axis, value);
  }
}

__global__ void validate_kernel(const double* values, std::size_t count, int* error) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += std::size_t(blockDim.x) * gridDim.x)
    if (!isfinite(values[i])) atomicCAS(error, 0, 1);
}

struct Owner {
  std::size_t natoms{}, coordinates{}, result_bytes{}, coord_bytes{}, charge_bytes{};
  std::size_t direction_offset{}, charge_offset{}, error_offset{}, owned_bytes{};
  std::uint64_t h2d_bytes{}, d2h_bytes{}, synchronizations{}, dense_adds{}, scatters{};
  int device_id{-1};
  std::mutex mutex;
  vibeqc::runtime::OwnedCudaStream stream;
  vibeqc::runtime::OwnedCudaBuffer<unsigned char> arena;

  Owner(int device, int major, int minor, std::size_t atoms, std::size_t budget)
      : natoms(atoms),
        coordinates(mul(atoms, 3)),
        result_bytes(mul(coordinates, sizeof(double))),
        coord_bytes(result_bytes),
        charge_bytes(mul(atoms, sizeof(double))) {
    if (!atoms || atoms > 4096) throw std::invalid_argument("invalid HVP atom count");
    direction_offset = result_bytes + coord_bytes;
    charge_offset = direction_offset + result_bytes;
    error_offset = charge_offset + charge_bytes;
    owned_bytes = error_offset + sizeof(int);
    if (owned_bytes > budget) throw std::bad_alloc();
    validate_target(device, major, minor);
    device_id = device;
    stream.create(device);
    arena.allocate(device, owned_bytes, stream.get());
  }

  unsigned char* base() { return arena.get(); }
  int* error() { return reinterpret_cast<int*>(base() + error_offset); }
  double* result() { return reinterpret_cast<double*>(base()); }
  double* coords() { return reinterpret_cast<double*>(base() + result_bytes); }
  double* direction() { return reinterpret_cast<double*>(base() + direction_offset); }
  double* charges() { return reinterpret_cast<double*>(base() + charge_offset); }

  void check_error(const char* detail) {
    int numerical_error = 0;
    vibeqc::runtime::cuda_resource_check(cudaMemcpyAsync(&numerical_error, error(), sizeof(int),
                                                         cudaMemcpyDeviceToHost, stream.get()));
    vibeqc::runtime::cuda_resource_check(cudaStreamSynchronize(stream.get()));
    ++synchronizations;
    if (numerical_error) throw NumericalFailure(detail);
  }

  void reset_nuclear(const double* host_coords, const double* host_charges,
                     const double* host_direction, std::size_t atoms) {
    std::lock_guard<std::mutex> lock(mutex);
    vibeqc::runtime::CudaDeviceScope device_scope(device_id);
    if (!host_coords || !host_charges || !host_direction || atoms != natoms)
      throw std::invalid_argument("HVP nuclear input shape mismatch");
    for (std::size_t i = 0; i < coordinates; ++i)
      if (!std::isfinite(host_coords[i]) || !std::isfinite(host_direction[i]))
        throw std::invalid_argument("HVP nuclear coordinates/direction must be finite");
    for (std::size_t i = 0; i < natoms; ++i)
      if (!std::isfinite(host_charges[i]))
        throw std::invalid_argument("HVP nuclear charges must be finite");
    vibeqc::runtime::cuda_resource_check(cudaMemsetAsync(result(), 0, result_bytes, stream.get()));
    vibeqc::runtime::cuda_resource_check(cudaMemsetAsync(error(), 0, sizeof(int), stream.get()));
    vibeqc::runtime::cuda_resource_check(
        cudaMemcpyAsync(coords(), host_coords, coord_bytes, cudaMemcpyHostToDevice, stream.get()));
    vibeqc::runtime::cuda_resource_check(cudaMemcpyAsync(direction(), host_direction, result_bytes,
                                                         cudaMemcpyHostToDevice, stream.get()));
    vibeqc::runtime::cuda_resource_check(cudaMemcpyAsync(charges(), host_charges, charge_bytes,
                                                         cudaMemcpyHostToDevice, stream.get()));
    h2d_bytes += coord_bytes + result_bytes + charge_bytes;
    nuclear_kernel<<<blocks(natoms, 64), 64, 0, stream.get()>>>(coords(), charges(), direction(),
                                                                natoms, result(), error());
    vibeqc::runtime::cuda_resource_check(cudaGetLastError());
    check_error("nonfinite CUDA nuclear HVP");
  }

  void add_dense(const double* source, std::size_t count, double coefficient) {
    std::lock_guard<std::mutex> lock(mutex);
    vibeqc::runtime::CudaDeviceScope device_scope(device_id);
    if (!source || count != coordinates || !std::isfinite(coefficient))
      throw std::invalid_argument("invalid dense HVP device contribution");
    vibeqc::runtime::cuda_resource_check(cudaMemsetAsync(error(), 0, sizeof(int), stream.get()));
    add_dense_kernel<<<blocks(count, 128), 128, 0, stream.get()>>>(source, count, coefficient,
                                                                   result(), error());
    vibeqc::runtime::cuda_resource_check(cudaGetLastError());
    ++dense_adds;
    check_error("nonfinite dense CUDA HVP contribution");
  }

  void scatter(const double* source, const std::uint32_t* output_indices, std::size_t source_count,
               const std::uint32_t* center_atoms, std::size_t center_count, double coefficient) {
    std::lock_guard<std::mutex> lock(mutex);
    vibeqc::runtime::CudaDeviceScope device_scope(device_id);
    if (!source || !output_indices || !center_atoms || !source_count || source_count > 12 ||
        !center_count || center_count > 4 || !std::isfinite(coefficient))
      throw std::invalid_argument("invalid compact HVP device contribution");
    Mapping mapping{};
    mapping.source_count = static_cast<std::uint32_t>(source_count);
    mapping.center_count = static_cast<std::uint32_t>(center_count);
    for (std::size_t i = 0; i < source_count; ++i) {
      if (output_indices[i] >= 3 * center_count)
        throw std::invalid_argument("compact HVP coordinate index out of bounds");
      mapping.output_indices[i] = output_indices[i];
    }
    for (std::size_t i = 0; i < center_count; ++i) {
      if (center_atoms[i] >= natoms)
        throw std::invalid_argument("compact HVP atom index out of bounds");
      mapping.center_atoms[i] = center_atoms[i];
    }
    vibeqc::runtime::cuda_resource_check(cudaMemsetAsync(error(), 0, sizeof(int), stream.get()));
    scatter_kernel<<<1, 32, 0, stream.get()>>>(source, mapping, coefficient, result(), error());
    vibeqc::runtime::cuda_resource_check(cudaGetLastError());
    ++scatters;
    check_error("nonfinite compact CUDA HVP contribution");
  }

  const double* output_device() {
    std::lock_guard<std::mutex> lock(mutex);
    vibeqc::runtime::CudaDeviceScope device_scope(device_id);
    vibeqc::runtime::cuda_resource_check(cudaMemsetAsync(error(), 0, sizeof(int), stream.get()));
    validate_kernel<<<blocks(coordinates, 128), 128, 0, stream.get()>>>(result(), coordinates,
                                                                        error());
    vibeqc::runtime::cuda_resource_check(cudaGetLastError());
    check_error("nonfinite final CUDA HVP");
    return result();
  }

  void download(double* output, std::size_t count) {
    std::lock_guard<std::mutex> lock(mutex);
    vibeqc::runtime::CudaDeviceScope device_scope(device_id);
    if (!output || count != coordinates)
      throw std::invalid_argument("HVP result download shape mismatch");
    vibeqc::runtime::cuda_resource_check(cudaMemsetAsync(error(), 0, sizeof(int), stream.get()));
    validate_kernel<<<blocks(coordinates, 128), 128, 0, stream.get()>>>(result(), coordinates,
                                                                        error());
    vibeqc::runtime::cuda_resource_check(cudaGetLastError());
    vibeqc::runtime::cuda_resource_check(
        cudaMemcpyAsync(output, result(), result_bytes, cudaMemcpyDeviceToHost, stream.get()));
    d2h_bytes += result_bytes;
    check_error("nonfinite final CUDA HVP");
  }
};

struct Diagnostics {
  std::uint64_t owned_device_bytes;
  std::uint64_t h2d_bytes;
  std::uint64_t d2h_bytes;
  std::uint64_t synchronizations;
  std::uint64_t dense_adds;
  std::uint64_t scatters;
  std::uint64_t natoms;
};

struct MatrixDiagnostics {
  std::uint64_t owned_device_bytes;
  std::uint64_t d2h_bytes;
  std::uint64_t synchronizations;
  std::uint64_t column_copies;
  std::uint64_t rows;
  std::uint64_t columns;
};

struct MatrixOwner {
  std::size_t rows{}, columns{}, values{}, matrix_bytes{}, error_offset{}, owned_bytes{};
  std::uint64_t d2h_bytes{}, synchronizations{}, column_copies{};
  int device_id{-1};
  std::mutex mutex;
  vibeqc::runtime::OwnedCudaStream stream;
  vibeqc::runtime::OwnedCudaBuffer<unsigned char> arena;

  MatrixOwner(int device, int major, int minor, std::size_t row_count, std::size_t column_count,
              std::size_t budget)
      : rows(row_count),
        columns(column_count),
        values(mul(row_count, column_count)),
        matrix_bytes(mul(values, sizeof(double))),
        error_offset(matrix_bytes),
        owned_bytes(matrix_bytes + sizeof(int)) {
    if (!rows || !columns || rows > 12288 || columns > 12288)
      throw std::invalid_argument("invalid Hessian matrix dimensions");
    if (owned_bytes > budget) throw std::bad_alloc();
    validate_target(device, major, minor);
    device_id = device;
    stream.create(device);
    arena.allocate(device, owned_bytes, stream.get());
  }

  unsigned char* base() { return arena.get(); }
  int* error() { return reinterpret_cast<int*>(base() + error_offset); }
  double* matrix() { return reinterpret_cast<double*>(base()); }

  void reset() {
    std::lock_guard<std::mutex> lock(mutex);
    vibeqc::runtime::CudaDeviceScope device_scope(device_id);
    vibeqc::runtime::cuda_resource_check(cudaMemsetAsync(matrix(), 0, matrix_bytes, stream.get()));
    vibeqc::runtime::cuda_resource_check(cudaStreamSynchronize(stream.get()));
    ++synchronizations;
  }

  void copy_column(const double* source, std::size_t count, std::size_t column) {
    std::lock_guard<std::mutex> lock(mutex);
    vibeqc::runtime::CudaDeviceScope device_scope(device_id);
    if (!source || count != rows || column >= columns)
      throw std::invalid_argument("invalid Hessian device column");
    vibeqc::runtime::cuda_resource_check(cudaMemcpyAsync(matrix() + column * rows, source,
                                                         rows * sizeof(double),
                                                         cudaMemcpyDeviceToDevice, stream.get()));
    vibeqc::runtime::cuda_resource_check(cudaMemsetAsync(error(), 0, sizeof(int), stream.get()));
    validate_kernel<<<blocks(rows, 128), 128, 0, stream.get()>>>(matrix() + column * rows, rows,
                                                                 error());
    vibeqc::runtime::cuda_resource_check(cudaGetLastError());
    int numerical_error = 0;
    vibeqc::runtime::cuda_resource_check(cudaMemcpyAsync(&numerical_error, error(), sizeof(int),
                                                         cudaMemcpyDeviceToHost, stream.get()));
    vibeqc::runtime::cuda_resource_check(cudaStreamSynchronize(stream.get()));
    ++synchronizations;
    if (numerical_error) throw NumericalFailure("nonfinite CUDA Hessian column");
    ++column_copies;
  }

  void download(double* output, std::size_t count) {
    std::lock_guard<std::mutex> lock(mutex);
    vibeqc::runtime::CudaDeviceScope device_scope(device_id);
    if (!output || count != values)
      throw std::invalid_argument("Hessian result download shape mismatch");
    vibeqc::runtime::cuda_resource_check(cudaMemsetAsync(error(), 0, sizeof(int), stream.get()));
    validate_kernel<<<blocks(values, 128), 128, 0, stream.get()>>>(matrix(), values, error());
    vibeqc::runtime::cuda_resource_check(cudaGetLastError());
    vibeqc::runtime::cuda_resource_check(
        cudaMemcpyAsync(output, matrix(), matrix_bytes, cudaMemcpyDeviceToHost, stream.get()));
    d2h_bytes += matrix_bytes;
    int numerical_error = 0;
    vibeqc::runtime::cuda_resource_check(cudaMemcpyAsync(&numerical_error, error(), sizeof(int),
                                                         cudaMemcpyDeviceToHost, stream.get()));
    vibeqc::runtime::cuda_resource_check(cudaStreamSynchronize(stream.get()));
    ++synchronizations;
    if (numerical_error) throw NumericalFailure("nonfinite final CUDA Hessian");
  }
};

template <class Operation>
int boundary(Operation operation, char* detail, std::size_t size) {
  if (detail && size) detail[0] = 0;
  try {
    operation();
    return 0;
  } catch (const NumericalFailure& error) {
    detail_text(detail, size, error.what());
    return 5;
  } catch (const std::bad_alloc& error) {
    detail_text(detail, size, error.what());
    return 7;
  } catch (const std::invalid_argument& error) {
    detail_text(detail, size, error.what());
    return 1;
  } catch (const std::exception& error) {
    detail_text(detail, size, error.what());
    return 8;
  }
}

}  // namespace vibeqc::integrals::hvp_assembly

extern "C" int vibeqc_hvp_assembly_create_v1(int device, int major, int minor, std::size_t natoms,
                                             std::size_t budget, void** output, char* detail,
                                             std::size_t size) {
  using namespace vibeqc::integrals::hvp_assembly;
  if (output) *output = nullptr;
  return boundary(
      [&] {
        if (!output) throw std::invalid_argument("null HVP assembly output");
        auto owner = std::make_unique<Owner>(device, major, minor, natoms, budget);
        *output = owner.release();
      },
      detail, size);
}

extern "C" void vibeqc_hvp_assembly_destroy_v1(void* handle) {
  delete static_cast<vibeqc::integrals::hvp_assembly::Owner*>(handle);
}

extern "C" int vibeqc_hvp_assembly_output_device_v1(void* handle, const double** output,
                                                    std::size_t* count, char* detail,
                                                    std::size_t size) {
  using namespace vibeqc::integrals::hvp_assembly;
  if (output) *output = nullptr;
  if (count) *count = 0;
  return boundary(
      [&] {
        if (!handle || !output || !count)
          throw std::invalid_argument("null HVP assembly device output");
        auto* owner = static_cast<Owner*>(handle);
        *output = owner->output_device();
        *count = owner->coordinates;
      },
      detail, size);
}

extern "C" int vibeqc_hvp_assembly_reset_nuclear_v1(void* handle, const double* coords,
                                                    const double* charges, const double* direction,
                                                    std::size_t natoms, char* detail,
                                                    std::size_t size) {
  using namespace vibeqc::integrals::hvp_assembly;
  return boundary(
      [&] {
        if (!handle) throw std::invalid_argument("null HVP assembly handle");
        static_cast<Owner*>(handle)->reset_nuclear(coords, charges, direction, natoms);
      },
      detail, size);
}

extern "C" int vibeqc_hvp_assembly_add_dense_v1(void* handle, const double* source,
                                                std::size_t count, double coefficient, char* detail,
                                                std::size_t size) {
  using namespace vibeqc::integrals::hvp_assembly;
  return boundary(
      [&] {
        if (!handle) throw std::invalid_argument("null HVP assembly handle");
        static_cast<Owner*>(handle)->add_dense(source, count, coefficient);
      },
      detail, size);
}

extern "C" int vibeqc_hvp_assembly_scatter_v1(void* handle, const double* source,
                                              const std::uint32_t* output_indices,
                                              std::size_t source_count,
                                              const std::uint32_t* center_atoms,
                                              std::size_t center_count, double coefficient,
                                              char* detail, std::size_t size) {
  using namespace vibeqc::integrals::hvp_assembly;
  return boundary(
      [&] {
        if (!handle) throw std::invalid_argument("null HVP assembly handle");
        static_cast<Owner*>(handle)->scatter(source, output_indices, source_count, center_atoms,
                                             center_count, coefficient);
      },
      detail, size);
}

extern "C" int vibeqc_hvp_assembly_download_v1(void* handle, double* output, std::size_t count,
                                               char* detail, std::size_t size) {
  using namespace vibeqc::integrals::hvp_assembly;
  return boundary(
      [&] {
        if (!handle) throw std::invalid_argument("null HVP assembly handle");
        static_cast<Owner*>(handle)->download(output, count);
      },
      detail, size);
}

extern "C" int vibeqc_hvp_assembly_diagnostics_v1(
    void* handle, vibeqc::integrals::hvp_assembly::Diagnostics* output, char* detail,
    std::size_t size) {
  using namespace vibeqc::integrals::hvp_assembly;
  return boundary(
      [&] {
        if (!handle || !output) throw std::invalid_argument("null HVP assembly diagnostics");
        auto* owner = static_cast<Owner*>(handle);
        *output = Diagnostics{
            owner->owned_bytes, owner->h2d_bytes, owner->d2h_bytes, owner->synchronizations,
            owner->dense_adds,  owner->scatters,  owner->natoms};
      },
      detail, size);
}

extern "C" int vibeqc_hessian_assembly_create_v1(int device, int major, int minor, std::size_t rows,
                                                 std::size_t columns, std::size_t budget,
                                                 void** output, char* detail, std::size_t size) {
  using namespace vibeqc::integrals::hvp_assembly;
  if (output) *output = nullptr;
  return boundary(
      [&] {
        if (!output) throw std::invalid_argument("null Hessian assembly output");
        auto owner = std::make_unique<MatrixOwner>(device, major, minor, rows, columns, budget);
        *output = owner.release();
      },
      detail, size);
}

extern "C" void vibeqc_hessian_assembly_destroy_v1(void* handle) {
  delete static_cast<vibeqc::integrals::hvp_assembly::MatrixOwner*>(handle);
}

extern "C" int vibeqc_hessian_assembly_reset_v1(void* handle, char* detail, std::size_t size) {
  using namespace vibeqc::integrals::hvp_assembly;
  return boundary(
      [&] {
        if (!handle) throw std::invalid_argument("null Hessian assembly handle");
        static_cast<MatrixOwner*>(handle)->reset();
      },
      detail, size);
}

extern "C" int vibeqc_hessian_assembly_copy_column_v1(void* handle, const double* source,
                                                      std::size_t count, std::size_t column,
                                                      char* detail, std::size_t size) {
  using namespace vibeqc::integrals::hvp_assembly;
  return boundary(
      [&] {
        if (!handle) throw std::invalid_argument("null Hessian assembly handle");
        static_cast<MatrixOwner*>(handle)->copy_column(source, count, column);
      },
      detail, size);
}

extern "C" int vibeqc_hessian_assembly_download_v1(void* handle, double* output, std::size_t count,
                                                   char* detail, std::size_t size) {
  using namespace vibeqc::integrals::hvp_assembly;
  return boundary(
      [&] {
        if (!handle) throw std::invalid_argument("null Hessian assembly handle");
        static_cast<MatrixOwner*>(handle)->download(output, count);
      },
      detail, size);
}

extern "C" int vibeqc_hessian_assembly_diagnostics_v1(
    void* handle, vibeqc::integrals::hvp_assembly::MatrixDiagnostics* output, char* detail,
    std::size_t size) {
  using namespace vibeqc::integrals::hvp_assembly;
  return boundary(
      [&] {
        if (!handle || !output) throw std::invalid_argument("null Hessian assembly diagnostics");
        auto* owner = static_cast<MatrixOwner*>(handle);
        *output = MatrixDiagnostics{owner->owned_bytes,   owner->d2h_bytes, owner->synchronizations,
                                    owner->column_copies, owner->rows,      owner->columns};
      },
      detail, size);
}
