#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/ao_grid.hpp"
#include "dft/cuda_cosx.hpp"
#include "dft/grid_task_view.cuh"
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
    throw std::overflow_error("CUDA COSX resource size overflow");
  return a + b;
}
std::size_t mul(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::overflow_error("CUDA COSX resource size overflow");
  return a * b;
}

std::size_t grid_bytes(const AoBasis& basis, std::size_t capacity) {
  const std::size_t packed =
      add(add(mul(3, basis.natom), mul(2, basis.nprimitive)), mul(16, basis.nao));
  const std::size_t matrices = mul(2, mul(basis.nao, basis.nao));
  const std::size_t tile = mul(capacity, basis.nao);
  std::size_t elements = add(add(packed, matrices), add(mul(16, capacity), mul(9, tile)));
  elements = add(elements, add(matrices, add(mul(4, mul(basis.nao, basis.nao)), basis.nao)));
  const std::size_t numeric = mul(sizeof(double), elements);
  const std::size_t error_offset = mul(add(numeric, 255) / 256, 256);
  const std::size_t workspace = add(error_offset, 256);
  return add(workspace, 4U << 20);
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
    count_ = count;
    if (count_) check(cudaMalloc(&pointer_, mul(count_, sizeof(T))));
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
    count_ = 0;
  }

  T* pointer_{};
  std::size_t count_{};
  int device_{};
};

__device__ double finite_or_flag(double value, int* error) {
  if (!isfinite(value)) {
    atomicCAS(error, 0, 1);
    return 0.0;
  }
  return value;
}

__global__ void esp_integrals_kernel(const double* basis, std::size_t natom, std::size_t nprimitive,
                                     std::size_t nao, const double* points, std::size_t npoint,
                                     double* esp, int* error) {
  namespace one = vibeqc::scf::generated_one_electron;
  const double* primitives = basis + 3 * natom;
  const double* records = primitives + 2 * nprimitive;
  const std::size_t total = npoint * nao * nao;
  for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
       index += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t point = index / (nao * nao);
    const std::size_t row = index / nao % nao, column = index % nao;
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
    for (std::size_t pa = 0; pa < first_count; ++pa) {
      const std::size_t ia = first_begin + pa;
      for (std::size_t pb = 0; pb < second_count; ++pb) {
        const std::size_t ib = second_begin + pb;
        const auto pair = one::make_pair(primitives[2 * ia], primitives[2 * ib], a[0], a[1], a[2],
                                         b[0], b[1], b[2]);
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
            // generated attraction is the signed V for unit positive charge.
            // ESP v1 is the positive <mu|1/|r-C||nu> operator.
            const double unit_esp =
                -one::attraction(pair, first_component, second_component, c[0], c[1], c[2]);
            value += primitive_weight * first[7 + 4 * ti] * second[7 + 4 * tj] * unit_esp;
          }
        }
      }
    }
    const double checked = finite_or_flag(value, error);
    esp[(point * nao + row) * nao + column] = checked;
    esp[(point * nao + column) * nao + row] = checked;
  }
}

__global__ void project_density_kernel(const double* ao, const double* density, std::size_t npoint,
                                       std::size_t nbf, double* projected, int* error) {
  const std::size_t total = npoint * nbf;
  for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
       index += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t point = index / nbf, column = index % nbf;
    double value = 0.0;
    for (std::size_t row = 0; row < nbf; ++row)
      value += ao[point * nbf + row] * density[row * nbf + column];
    projected[index] = finite_or_flag(value, error);
  }
}

__global__ void apply_esp_kernel(const double* esp, const double* projected, const double* weights,
                                 std::size_t npoint, std::size_t nbf, double* potential,
                                 int* error) {
  const std::size_t total = npoint * nbf;
  for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
       index += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t point = index / nbf, row = index % nbf;
    const double* matrix = esp + point * nbf * nbf;
    double value = 0.0;
    for (std::size_t column = 0; column < nbf; ++column)
      value += matrix[row * nbf + column] * projected[point * nbf + column];
    potential[index] = finite_or_flag(weights[point] * value, error);
  }
}

__global__ void accumulate_exchange_kernel(const double* ao, const double* potential,
                                           std::size_t npoint, std::size_t nbf, double* raw,
                                           int* error) {
  const std::size_t total = nbf * nbf;
  for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
       index += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t row = index / nbf, column = index % nbf;
    double value = raw[index];
    for (std::size_t point = 0; point < npoint; ++point)
      value += ao[point * nbf + row] * potential[point * nbf + column];
    raw[index] = finite_or_flag(value, error);
  }
}

__global__ void symmetrize_exchange_kernel(const double* raw, std::size_t nbf, double* output,
                                           int* error) {
  const std::size_t total = nbf * nbf;
  for (std::size_t index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; index < total;
       index += std::size_t(blockDim.x) * gridDim.x) {
    const std::size_t row = index / nbf, column = index % nbf;
    output[index] = finite_or_flag(0.5 * (raw[index] + raw[column * nbf + row]), error);
  }
}

unsigned blocks(std::size_t work) {
  return static_cast<unsigned>(std::min<std::size_t>((work + 127) / 128, 65535));
}

}  // namespace

struct CudaCosxStagingPlan::Impl {
  core::System system;
  AoBasis basis;
  std::vector<double> points, weights;
  std::vector<std::size_t> ao_ids;
  std::size_t tile_points{};
  int device{};
  void* grid{};
  CudaCosxStagingDiagnostic diagnostic;
  GridBasisView device_basis{};

  DeviceBuffer<double> density;
  DeviceBuffer<double> esp;
  DeviceBuffer<double> device_weights;
  DeviceBuffer<double> projected;
  DeviceBuffer<double> potential;
  DeviceBuffer<double> raw;
  DeviceBuffer<double> exchange;
  DeviceBuffer<int> error;

  Impl(const core::System& input, std::span<const double> points_xyz,
       std::span<const double> input_weights, std::size_t requested_tile, int selected_device)
      : system(input),
        basis(system),
        points(points_xyz.begin(), points_xyz.end()),
        weights(input_weights.begin(), input_weights.end()),
        tile_points(requested_tile),
        device(selected_device) {
    if (points.empty() || points.size() % 3 || weights.size() != points.size() / 3 ||
        !tile_points || device < 0)
      throw std::invalid_argument("invalid bounded CUDA COSX staging shape");
    for (double value : points)
      if (!std::isfinite(value)) throw std::invalid_argument("nonfinite CUDA COSX point");
    for (double value : weights)
      if (!std::isfinite(value)) throw std::invalid_argument("nonfinite CUDA COSX weight");
    tile_points = std::min(tile_points, weights.size());
    ao_ids.resize(basis.nao);
    std::iota(ao_ids.begin(), ao_ids.end(), 0);

    DeviceGuard guard(device);
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device));
    const std::size_t dimensions[]{basis.natom, basis.nprimitive, basis.nao};
    char message[512]{};
    const auto expected_grid_bytes = grid_bytes(basis, tile_points);
    const int status = grid_cuda_create_v2(device, properties.major, properties.minor, dimensions,
                                           basis.packed.data(), tile_points, 0, expected_grid_bytes,
                                           basis.nao, &grid, message, sizeof(message));
    if (status == VIBEQC_STATUS_OUT_OF_MEMORY) throw std::bad_alloc();
    if (status != 0 || !grid)
      throw std::runtime_error(message[0] ? message : "CUDA COSX grid preparation failed");
    try {
      checked_status(grid_cuda_basis_v1(grid, &device_basis, message, sizeof(message)),
                     message[0] ? message : "CUDA COSX packed-basis view failed");
      if (device_basis.version != 1 || device_basis.natom != basis.natom ||
          device_basis.nprimitive != basis.nprimitive || device_basis.nao != basis.nao ||
          !device_basis.basis || !device_basis.stream)
        throw std::runtime_error("CUDA COSX received an incompatible packed-basis view");

      const std::size_t matrix = mul(basis.nao, basis.nao);
      density.reset(matrix, device);
      esp.reset(mul(tile_points, matrix), device);
      device_weights.reset(tile_points, device);
      projected.reset(mul(tile_points, basis.nao), device);
      potential.reset(mul(tile_points, basis.nao), device);
      raw.reset(matrix, device);
      exchange.reset(matrix, device);
      error.reset(1, device);

      const std::size_t cosx_doubles = add(
          add(add(matrix, mul(tile_points, matrix)), tile_points),
          add(add(mul(tile_points, basis.nao), mul(tile_points, basis.nao)), add(matrix, matrix)));
      diagnostic = {basis.nao,
                    weights.size(),
                    tile_points,
                    expected_grid_bytes,
                    add(mul(cosx_doubles, sizeof(double)), sizeof(int)),
                    0,
                    mul(tile_points, matrix),
                    mul(tile_points, basis.nao),
                    true,
                    true,
                    true};
      diagnostic.device_bytes = add(diagnostic.grid_device_bytes, diagnostic.cosx_device_bytes);
    } catch (...) {
      grid_cuda_destroy_v1(grid);
      grid = nullptr;
      throw;
    }
  }

  ~Impl() {
    // Grid Context and each DeviceBuffer already select their owning device
    // during nonthrowing teardown; never construct a throwing guard here.
    if (grid) grid_cuda_destroy_v1(grid);
  }

  CosxReferenceResult build(std::span<const double> host_density,
                            CosxDensityConvention convention) {
    const std::size_t n = basis.nao, matrix = mul(n, n);
    if (host_density.size() != matrix)
      throw std::invalid_argument("CUDA COSX density does not match the AO basis");
    for (double value : host_density)
      if (!std::isfinite(value)) throw std::invalid_argument("nonfinite CUDA COSX density");
    if (convention != CosxDensityConvention::spin_resolved &&
        convention != CosxDensityConvention::rhf_spin_summed)
      throw std::invalid_argument("unknown CUDA COSX density convention");

    DeviceGuard guard(device);
    bool initialized = false;
    char message[512]{};
    for (std::size_t begin = 0; begin < weights.size(); begin += tile_points) {
      const std::size_t count = std::min(tile_points, weights.size() - begin);
      checked_status(
          grid_cuda_run_selected_v1(grid, points.data() + 3 * begin, count, 0, ao_ids.data(), n,
                                    nullptr, nullptr, message, sizeof(message)),
          message[0] ? message : "CUDA COSX AO tile failed");
      GridTaskView view{};
      checked_status(grid_cuda_view_v1(grid, &view, message, sizeof(message)),
                     message[0] ? message : "CUDA COSX AO view failed");
      if (view.version != 1 || view.npoint != count || view.nao != n || view.nactive != n ||
          view.jets != 1 || !view.ao || !view.stream)
        throw std::runtime_error("CUDA COSX received an incompatible AO task view");

      if (!initialized) {
        check(cudaMemcpyAsync(density.get(), host_density.data(), matrix * sizeof(double),
                              cudaMemcpyHostToDevice, view.stream));
        check(cudaMemsetAsync(raw.get(), 0, matrix * sizeof(double), view.stream));
        check(cudaMemsetAsync(error.get(), 0, sizeof(int), view.stream));
        initialized = true;
      }
      esp_integrals_kernel<<<blocks(count * matrix), 128, 0, view.stream>>>(
          device_basis.basis, device_basis.natom, device_basis.nprimitive, n, view.points, count,
          esp.get(), error.get());
      check(cudaGetLastError());
      check(cudaMemcpyAsync(device_weights.get(), weights.data() + begin, count * sizeof(double),
                            cudaMemcpyHostToDevice, view.stream));

      project_density_kernel<<<blocks(count * n), 128, 0, view.stream>>>(
          view.ao, density.get(), count, n, projected.get(), error.get());
      check(cudaGetLastError());
      apply_esp_kernel<<<blocks(count * n), 128, 0, view.stream>>>(
          esp.get(), projected.get(), device_weights.get(), count, n, potential.get(), error.get());
      check(cudaGetLastError());
      accumulate_exchange_kernel<<<blocks(matrix), 128, 0, view.stream>>>(
          view.ao, potential.get(), count, n, raw.get(), error.get());
      check(cudaGetLastError());
    }

    GridTaskView final_view{};
    checked_status(grid_cuda_view_v1(grid, &final_view, message, sizeof(message)),
                   message[0] ? message : "CUDA COSX final AO view failed");
    symmetrize_exchange_kernel<<<blocks(matrix), 128, 0, final_view.stream>>>(
        raw.get(), n, exchange.get(), error.get());
    check(cudaGetLastError());

    CosxReferenceResult result;
    result.nbf = n;
    result.npoint = weights.size();
    result.spec = {};
    result.convention = convention;
    result.raw_exchange.resize(matrix);
    result.exchange.resize(matrix);
    int failure = 0;
    try {
      check(cudaMemcpyAsync(result.raw_exchange.data(), raw.get(), matrix * sizeof(double),
                            cudaMemcpyDeviceToHost, final_view.stream));
      check(cudaMemcpyAsync(result.exchange.data(), exchange.get(), matrix * sizeof(double),
                            cudaMemcpyDeviceToHost, final_view.stream));
      check(cudaMemcpyAsync(&failure, error.get(), sizeof(int), cudaMemcpyDeviceToHost,
                            final_view.stream));
      check(cudaStreamSynchronize(final_view.stream));
    } catch (...) {
      // A later enqueue may fail while an earlier download still targets these
      // local buffers. Drain before unwinding their lifetime; retain the cause.
      (void)cudaStreamSynchronize(final_view.stream);
      throw;
    }
    if (failure) throw std::runtime_error("nonfinite CUDA COSX staging result");

    double contraction = 0.0;
    for (std::size_t element = 0; element < matrix; ++element)
      contraction += host_density[element] * result.exchange[element];
    result.exchange_energy =
        (convention == CosxDensityConvention::rhf_spin_summed ? -0.25 : -0.5) * contraction;
    if (!std::isfinite(result.exchange_energy))
      throw std::runtime_error("nonfinite CUDA COSX exchange energy");
    return result;
  }
};

CudaCosxStagingPlan::CudaCosxStagingPlan(const core::System& system,
                                         std::span<const double> points_xyz,
                                         std::span<const double> weights, std::size_t tile_points,
                                         int device)
    : impl_(std::make_unique<Impl>(system, points_xyz, weights, tile_points, device)) {}
CudaCosxStagingPlan::~CudaCosxStagingPlan() = default;

CosxReferenceResult CudaCosxStagingPlan::build(std::span<const double> density,
                                               CosxDensityConvention convention) {
  return impl_->build(density, convention);
}

const CudaCosxStagingDiagnostic& CudaCosxStagingPlan::diagnostic() const noexcept {
  return impl_->diagnostic;
}

}  // namespace vibeqc::dft
