#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "dft/ao_grid.hpp"
#include "dft/cuda_cosx.hpp"
#include "dft/grid_task_view.cuh"
#include "generated_one_electron_derivatives.cuh"
#include "generated_one_electron_values.cuh"
#include "molecule/basis.hpp"
#include "vibeqc/vibeqc.h"

extern "C" {
int grid_cuda_create_v2(int device, int major, int minor, const size_t* dimensions,
                        const double* basis, size_t capacity, unsigned order, size_t expected_bytes,
                        size_t active_capacity, void** output, char* error, size_t size);
void grid_cuda_destroy_v1(void* pointer);
int grid_cuda_run_selected_v1(void* pointer, const double* points, size_t npoint, int features,
                              const size_t* ao_ids, size_t active, double* feature_output,
                              double* jet_output, char* error, size_t size);
int grid_cuda_view_v1(void* pointer, vibeqc::dft::GridTaskView* output, char* error, size_t size);
int grid_cuda_basis_v1(void* pointer, vibeqc::dft::GridBasisView* output, char* error, size_t size);
}

namespace vibeqc::dft {
namespace {

void check(cudaError_t status) {
  if (status == cudaErrorMemoryAllocation) throw std::bad_alloc();
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

void checked_status(int status, const char* detail) {
  if (status == VIBEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
  if (status != 0) throw std::runtime_error(detail);
}

std::size_t add(std::size_t a, std::size_t b) {
  if (a > std::numeric_limits<std::size_t>::max() - b)
    throw std::overflow_error("CUDA COSX derivative resource size overflow");
  return a + b;
}

std::size_t mul(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::overflow_error("CUDA COSX derivative resource size overflow");
  return a * b;
}

std::size_t derivative_grid_bytes(const AoBasis& basis, std::size_t capacity) {
  constexpr std::size_t jets = 4;
  const std::size_t packed =
      add(add(mul(3, basis.natom), mul(2, basis.nprimitive)), mul(16, basis.nao));
  const std::size_t matrices = mul(2, mul(basis.nao, basis.nao));
  const std::size_t tile = mul(capacity, basis.nao);
  std::size_t elements = add(add(packed, matrices), add(mul(16, capacity), mul(jets + 8, tile)));
  elements = add(elements, add(matrices, add(mul(4, mul(basis.nao, basis.nao)), basis.nao)));
  const std::size_t numeric = mul(sizeof(double), elements);
  const std::size_t error_offset = mul(add(numeric, 255) / 256, 256);
  return add(add(error_offset, 256), 4U << 20);
}

class DeviceGuard {
 public:
  explicit DeviceGuard(int device) {
    check(cudaGetDevice(&previous_));
    check(cudaSetDevice(device));
  }
  ~DeviceGuard() { cudaSetDevice(previous_); }

 private:
  int previous_{};
};

template <class T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() { release(); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void reset(std::size_t count, int device) {
    release();
    device_ = device;
    if (count) check(cudaMalloc(&pointer_, mul(count, sizeof(T))));
  }
  T* get() noexcept { return pointer_; }
  const T* get() const noexcept { return pointer_; }

 private:
  void release() noexcept {
    if (!pointer_) return;
    int previous = 0;
    if (cudaGetDevice(&previous) == cudaSuccess) {
      cudaSetDevice(device_);
      cudaFree(pointer_);
      cudaSetDevice(previous);
    }
    pointer_ = nullptr;
  }

  T* pointer_{};
  int device_{};
};

__device__ double finite_or_flag(double value, int* error) {
  if (!isfinite(value)) {
    atomicCAS(error, 0, 1);
    return 0.0;
  }
  return value;
}

__global__ void esp_probe_derivative_kernel(const double* basis, std::size_t natom,
                                            std::size_t nprimitive, std::size_t nao,
                                            const double* points, std::size_t npoint, double* esp,
                                            double* esp_derivative, int* error) {
  namespace one = vibeqc::scf::generated_one_electron;
  namespace derivative = vibeqc::scf::generated_one_electron_derivatives;
  const double* primitives = basis + 3 * natom;
  const double* records = primitives + 2 * nprimitive;
  const std::size_t matrix = nao * nao;
  const std::size_t total = npoint * matrix;
  for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
       index += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t point = index / matrix;
    const std::size_t row = index / nao % nao;
    const std::size_t column = index % nao;
    if (row < column) continue;

    const double* first = records + 16 * row;
    const double* second = records + 16 * column;
    const auto atom_first = static_cast<std::size_t>(first[0]);
    const auto atom_second = static_cast<std::size_t>(second[0]);
    const double* a = basis + 3 * atom_first;
    const double* b = basis + 3 * atom_second;
    const double* c = points + 3 * point;
    const auto first_begin = static_cast<std::size_t>(first[1]);
    const auto second_begin = static_cast<std::size_t>(second[1]);
    const auto first_count = static_cast<std::size_t>(first[2]);
    const auto second_count = static_cast<std::size_t>(second[2]);
    const auto first_terms = static_cast<unsigned>(first[3]);
    const auto second_terms = static_cast<unsigned>(second[3]);

    double value = 0.0;
    double gradient[3]{};
    for (std::size_t pa = 0; pa < first_count; ++pa) {
      const std::size_t ia = first_begin + pa;
      for (std::size_t pb = 0; pb < second_count; ++pb) {
        const std::size_t ib = second_begin + pb;
        const auto value_pair = one::make_pair(primitives[2 * ia], primitives[2 * ib], a[0], a[1],
                                               a[2], b[0], b[1], b[2]);
        const auto derivative_pair = derivative::make_pair(primitives[2 * ia], primitives[2 * ib],
                                                           a[0], a[1], a[2], b[0], b[1], b[2]);
        const double primitive_weight = primitives[2 * ia + 1] * primitives[2 * ib + 1];
        for (unsigned ti = 0; ti < first_terms; ++ti) {
          const unsigned first_component = one::component_index(
              static_cast<unsigned>(first[4 + 4 * ti]), static_cast<unsigned>(first[5 + 4 * ti]),
              static_cast<unsigned>(first[6 + 4 * ti]));
          for (unsigned tj = 0; tj < second_terms; ++tj) {
            const unsigned second_component =
                one::component_index(static_cast<unsigned>(second[4 + 4 * tj]),
                                     static_cast<unsigned>(second[5 + 4 * tj]),
                                     static_cast<unsigned>(second[6 + 4 * tj]));
            const double factor = primitive_weight * first[7 + 4 * ti] * second[7 + 4 * tj];
            value -= factor * one::attraction(value_pair, first_component, second_component, c[0],
                                              c[1], c[2]);
            const auto v = derivative::attraction_gradient(derivative_pair, first_component,
                                                           second_component, c[0], c[1], c[2]);
            for (unsigned axis = 0; axis < 3; ++axis) {
              // Positive ESP is -V. Translational invariance gives
              // d(-V)/dC = dV/dA + dV/dB.
              gradient[axis] += factor * (v.first[axis] + v.second[axis]);
            }
          }
        }
      }
    }

    const std::size_t transpose = (point * nao + column) * nao + row;
    const double checked_value = finite_or_flag(value, error);
    esp[index] = checked_value;
    esp[transpose] = checked_value;
    for (unsigned axis = 0; axis < 3; ++axis) {
      const double checked_gradient = finite_or_flag(gradient[axis], error);
      const std::size_t offset = (point * 3 + axis) * matrix;
      esp_derivative[offset + row * nao + column] = checked_gradient;
      esp_derivative[offset + column * nao + row] = checked_gradient;
    }
  }
}

__global__ void project_density_kernel(const double* ao, const double* density, std::size_t npoint,
                                       std::size_t nbf, double* projected, int* error) {
  const std::size_t total = npoint * nbf;
  for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
       index += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t point = index / nbf;
    const std::size_t column = index % nbf;
    double value = 0.0;
    for (std::size_t row = 0; row < nbf; ++row)
      value += ao[point * nbf + row] * density[row * nbf + column];
    projected[index] = finite_or_flag(value, error);
  }
}

__global__ void project_density_derivative_kernel(const double* ao, const double* density,
                                                  std::size_t npoint, std::size_t nbf,
                                                  double* projected_derivative, int* error) {
  const std::size_t total = 3 * npoint * nbf;
  const std::size_t jet_stride = npoint * nbf;
  for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
       index += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t axis = index / (npoint * nbf);
    const std::size_t point = index / nbf % npoint;
    const std::size_t column = index % nbf;
    const double* derivative = ao + (axis + 1) * jet_stride + point * nbf;
    double value = 0.0;
    for (std::size_t row = 0; row < nbf; ++row)
      value += derivative[row] * density[row * nbf + column];
    projected_derivative[index] = finite_or_flag(value, error);
  }
}

__global__ void apply_esp_kernel(const double* esp, const double* projected, const double* weights,
                                 std::size_t npoint, std::size_t nbf, double* potential,
                                 int* error) {
  const std::size_t total = npoint * nbf;
  for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
       index += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t point = index / nbf;
    const std::size_t row = index % nbf;
    const double* matrix = esp + point * nbf * nbf;
    double value = 0.0;
    for (std::size_t column = 0; column < nbf; ++column)
      value += matrix[row * nbf + column] * projected[point * nbf + column];
    potential[index] = finite_or_flag(weights[point] * value, error);
  }
}

__global__ void apply_esp_derivative_kernel(const double* esp, const double* esp_derivative,
                                            const double* projected,
                                            const double* projected_derivative,
                                            const double* weights, std::size_t npoint,
                                            std::size_t nbf, double* potential_derivative,
                                            int* error) {
  const std::size_t total = 3 * npoint * nbf;
  const std::size_t matrix = nbf * nbf;
  for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
       index += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t axis = index / (npoint * nbf);
    const std::size_t point = index / nbf % npoint;
    const std::size_t row = index % nbf;
    const double* matrix_value = esp + point * matrix;
    const double* matrix_derivative = esp_derivative + (point * 3 + axis) * matrix;
    const double* projected_value = projected + point * nbf;
    const double* projected_axis = projected_derivative + (axis * npoint + point) * nbf;
    double value = 0.0;
    for (std::size_t column = 0; column < nbf; ++column) {
      value += matrix_derivative[row * nbf + column] * projected_value[column] +
               matrix_value[row * nbf + column] * projected_axis[column];
    }
    potential_derivative[index] = finite_or_flag(weights[point] * value, error);
  }
}

__global__ void contract_point_derivative_kernel(const double* ao, const double* density,
                                                 const double* potential,
                                                 const double* potential_derivative,
                                                 std::size_t npoint, std::size_t nbf,
                                                 double energy_factor, double* point_gradient,
                                                 int* error) {
  const std::size_t total = 3 * npoint;
  const std::size_t jet_stride = npoint * nbf;
  for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
       index += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t point = index / 3;
    const std::size_t axis = index % 3;
    const double* phi = ao + point * nbf;
    const double* phi_derivative = ao + (axis + 1) * jet_stride + point * nbf;
    const double* potential_value = potential + point * nbf;
    const double* potential_axis = potential_derivative + (axis * npoint + point) * nbf;
    double contraction = 0.0;
    for (std::size_t row = 0; row < nbf; ++row) {
      for (std::size_t column = 0; column < nbf; ++column) {
        const double raw_rc =
            phi_derivative[row] * potential_value[column] + phi[row] * potential_axis[column];
        const double raw_cr =
            phi_derivative[column] * potential_value[row] + phi[column] * potential_axis[row];
        contraction += density[row * nbf + column] * 0.5 * (raw_rc + raw_cr);
      }
    }
    point_gradient[index] = finite_or_flag(energy_factor * contraction, error);
  }
}

unsigned blocks(std::size_t work) {
  return static_cast<unsigned>(std::min<std::size_t>((work + 127) / 128, 65535));
}

}  // namespace

std::vector<double> cuda_cosx_point_derivative_reference(const core::System& system,
                                                         std::span<const double> points_xyz,
                                                         std::span<const double> weights,
                                                         std::span<const double> host_density,
                                                         CosxDensityConvention convention,
                                                         std::size_t requested_tile, int device) {
  if (points_xyz.empty() || points_xyz.size() % 3 || weights.size() != points_xyz.size() / 3 ||
      !requested_tile || device < 0)
    throw std::invalid_argument("invalid CUDA COSX point-derivative shape");
  for (double value : points_xyz)
    if (!std::isfinite(value)) throw std::invalid_argument("nonfinite CUDA COSX point");
  for (double value : weights)
    if (!std::isfinite(value)) throw std::invalid_argument("nonfinite CUDA COSX weight");
  if (convention != CosxDensityConvention::spin_resolved &&
      convention != CosxDensityConvention::rhf_spin_summed)
    throw std::invalid_argument("unknown CUDA COSX density convention");

  const AoBasis basis(system);
  const std::size_t n = basis.nao;
  const std::size_t matrix = mul(n, n);
  const std::size_t npoint = weights.size();
  if (host_density.size() != matrix)
    throw std::invalid_argument("CUDA COSX density does not match the AO basis");
  for (double value : host_density)
    if (!std::isfinite(value)) throw std::invalid_argument("nonfinite CUDA COSX density");

  const std::size_t tile_points = std::min(npoint, requested_tile);
  const std::size_t dimensions[]{basis.natom, basis.nprimitive, basis.nao};
  std::vector<std::size_t> ao_ids(n);
  std::iota(ao_ids.begin(), ao_ids.end(), 0);
  std::vector<double> result(mul(3, npoint));

  DeviceGuard guard(device);
  cudaDeviceProp properties{};
  check(cudaGetDeviceProperties(&properties, device));
  void* grid = nullptr;
  cudaStream_t stream = nullptr;
  char message[512]{};
  const auto expected_grid_bytes = derivative_grid_bytes(basis, tile_points);
  const int create_status = grid_cuda_create_v2(
      device, properties.major, properties.minor, dimensions, basis.packed.data(), tile_points, 1,
      expected_grid_bytes, n, &grid, message, sizeof(message));
  if (create_status == VIBEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
  if (create_status != 0 || !grid)
    throw std::runtime_error(message[0] ? message : "CUDA COSX derivative grid preparation failed");

  DeviceBuffer<double> density, esp, esp_derivative, device_weights, projected,
      projected_derivative, potential, potential_derivative, point_gradient;
  DeviceBuffer<int> error;
  try {
    GridBasisView device_basis{};
    checked_status(grid_cuda_basis_v1(grid, &device_basis, message, sizeof(message)),
                   message[0] ? message : "CUDA COSX derivative packed-basis view failed");
    if (device_basis.version != 1 || device_basis.natom != basis.natom ||
        device_basis.nprimitive != basis.nprimitive || device_basis.nao != basis.nao ||
        !device_basis.basis || !device_basis.stream)
      throw std::runtime_error("CUDA COSX derivative received an incompatible packed-basis view");
    stream = device_basis.stream;

    density.reset(matrix, device);
    esp.reset(mul(tile_points, matrix), device);
    esp_derivative.reset(mul(mul(3, tile_points), matrix), device);
    device_weights.reset(tile_points, device);
    projected.reset(mul(tile_points, n), device);
    projected_derivative.reset(mul(mul(3, tile_points), n), device);
    potential.reset(mul(tile_points, n), device);
    potential_derivative.reset(mul(mul(3, tile_points), n), device);
    point_gradient.reset(mul(3, tile_points), device);
    error.reset(1, device);

    check(cudaMemcpyAsync(density.get(), host_density.data(), matrix * sizeof(double),
                          cudaMemcpyHostToDevice, stream));
    check(cudaMemsetAsync(error.get(), 0, sizeof(int), stream));

    for (std::size_t begin = 0; begin < npoint; begin += tile_points) {
      const std::size_t count = std::min(tile_points, npoint - begin);
      checked_status(
          grid_cuda_run_selected_v1(grid, points_xyz.data() + 3 * begin, count, 0, ao_ids.data(), n,
                                    nullptr, nullptr, message, sizeof(message)),
          message[0] ? message : "CUDA COSX derivative AO tile failed");
      GridTaskView view{};
      checked_status(grid_cuda_view_v1(grid, &view, message, sizeof(message)),
                     message[0] ? message : "CUDA COSX derivative AO view failed");
      if (view.version != 1 || view.npoint != count || view.nao != n || view.nactive != n ||
          view.jets != 4 || !view.ao || !view.stream)
        throw std::runtime_error("CUDA COSX derivative received an incompatible AO task view");

      esp_probe_derivative_kernel<<<blocks(count * matrix), 128, 0, view.stream>>>(
          device_basis.basis, device_basis.natom, device_basis.nprimitive, n, view.points, count,
          esp.get(), esp_derivative.get(), error.get());
      check(cudaGetLastError());
      check(cudaMemcpyAsync(device_weights.get(), weights.data() + begin, count * sizeof(double),
                            cudaMemcpyHostToDevice, view.stream));
      project_density_kernel<<<blocks(count * n), 128, 0, view.stream>>>(
          view.ao, density.get(), count, n, projected.get(), error.get());
      check(cudaGetLastError());
      project_density_derivative_kernel<<<blocks(3 * count * n), 128, 0, view.stream>>>(
          view.ao, density.get(), count, n, projected_derivative.get(), error.get());
      check(cudaGetLastError());
      apply_esp_kernel<<<blocks(count * n), 128, 0, view.stream>>>(
          esp.get(), projected.get(), device_weights.get(), count, n, potential.get(), error.get());
      check(cudaGetLastError());
      apply_esp_derivative_kernel<<<blocks(3 * count * n), 128, 0, view.stream>>>(
          esp.get(), esp_derivative.get(), projected.get(), projected_derivative.get(),
          device_weights.get(), count, n, potential_derivative.get(), error.get());
      check(cudaGetLastError());
      const double energy_factor =
          convention == CosxDensityConvention::rhf_spin_summed ? -0.25 : -0.5;
      contract_point_derivative_kernel<<<blocks(3 * count), 128, 0, view.stream>>>(
          view.ao, density.get(), potential.get(), potential_derivative.get(), count, n,
          energy_factor, point_gradient.get(), error.get());
      check(cudaGetLastError());
      check(cudaMemcpyAsync(result.data() + 3 * begin, point_gradient.get(),
                            3 * count * sizeof(double), cudaMemcpyDeviceToHost, view.stream));
    }

    int failure = 0;
    check(cudaMemcpyAsync(&failure, error.get(), sizeof(int), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    if (failure) throw std::runtime_error("nonfinite CUDA COSX point derivative");
    grid_cuda_destroy_v1(grid);
    return result;
  } catch (...) {
    if (stream) (void)cudaStreamSynchronize(stream);
    if (grid) grid_cuda_destroy_v1(grid);
    throw;
  }
}

}  // namespace vibeqc::dft
