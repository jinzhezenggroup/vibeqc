#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <string>
#include <vector>

#include "scf/cuda_density_fitting.hpp"

namespace vibeqc::scf {
namespace {

constexpr unsigned kThreads = 256;

bool checked_multiply(std::size_t first, std::size_t second,
                      std::size_t& product) {
  if (first != 0 && second > std::numeric_limits<std::size_t>::max() / first) {
    return false;
  }
  product = first * second;
  return true;
}

bool checked_bytes(std::size_t elements, std::size_t& bytes) {
  return checked_multiply(elements, sizeof(double), bytes);
}

bool finite_values(const std::vector<double>& values) {
  return std::all_of(values.begin(), values.end(),
                     [](double value) { return std::isfinite(value); });
}

unsigned blocks_for(std::size_t elements) {
  return static_cast<unsigned>((elements + kThreads - 1) / kThreads);
}

vibeqc_status cuda_failure(cudaError_t error, const char* operation,
                           std::string& detail) {
  detail = std::string(operation) + ": " + cudaGetErrorString(error);
  return error == cudaErrorMemoryAllocation ? VIBEQC_STATUS_OUT_OF_MEMORY
                                            : VIBEQC_STATUS_CUDA_ERROR;
}

vibeqc_status blas_failure(cublasStatus_t status, const char* operation,
                           std::string& detail) {
  detail = std::string(operation) + " failed with cuBLAS status " +
           std::to_string(static_cast<int>(status));
  return status == CUBLAS_STATUS_ALLOC_FAILED ? VIBEQC_STATUS_OUT_OF_MEMORY
                                              : VIBEQC_STATUS_CUDA_ERROR;
}

vibeqc_status solver_failure(cusolverStatus_t status, const char* operation,
                             std::string& detail) {
  detail = std::string(operation) + " failed with cuSOLVER status " +
           std::to_string(static_cast<int>(status));
  return status == CUSOLVER_STATUS_ALLOC_FAILED ? VIBEQC_STATUS_OUT_OF_MEMORY
                                                : VIBEQC_STATUS_CUDA_ERROR;
}

__global__ void symmetrize_metrics_kernel(std::size_t dimension,
                                          double* metrics) {
  const std::size_t column =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t row =
      static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
  const std::size_t system = blockIdx.z;
  if (row >= dimension || column >= dimension || row > column) return;
  const std::size_t offset = system * dimension * dimension;
  const std::size_t first = offset + row * dimension + column;
  const std::size_t second = offset + column * dimension + row;
  const double symmetric = 0.5 * (metrics[first] + metrics[second]);
  metrics[first] = symmetric;
  metrics[second] = symmetric;
}

__global__ void scale_eigenvectors_kernel(std::size_t matrix_elements,
                                          std::size_t dimension,
                                          const double* eigenvectors,
                                          const double* scales,
                                          double* scaled_eigenvectors) {
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= matrix_elements) return;
  const std::size_t local = element % (dimension * dimension);
  const std::size_t system = element / (dimension * dimension);
  const std::size_t column = local / dimension;
  scaled_eigenvectors[element] =
      eigenvectors[element] * scales[system * dimension + column];
}

__global__ void sum_spin_density_kernel(std::size_t elements,
                                        const double* alpha, const double* beta,
                                        double* total) {
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= elements) return;
  total[element] = alpha[element] + beta[element];
}

/** Convert the host row-major AO density into cuBLAS column-major storage. */
__global__ void transpose_density_kernel(std::size_t dimension,
                                          const double* row_major,
                                          double* column_major) {
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t elements = dimension * dimension;
  if (element >= elements) return;
  const std::size_t row = element / dimension;
  const std::size_t column = element % dimension;
  column_major[row + column * dimension] = row_major[element];
}

/** Gather raw ((mu nu)|P) values into one column-major matrix per auxiliary. */
__global__ void gather_force_auxiliary_matrices_kernel(
    std::size_t nbf, std::size_t naux, const double* raw, double* matrices) {
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t matrix_elements = nbf * nbf;
  const std::size_t elements = matrix_elements * naux;
  if (element >= elements) return;
  const std::size_t pair = element / naux;
  const std::size_t auxiliary = element % naux;
  const std::size_t row = pair / nbf;
  const std::size_t column = pair % nbf;
  // The source is pair-major with auxiliary contiguous; the destination is
  // auxiliary-major column-major for strided-batched GEMMs.
  matrices[auxiliary * matrix_elements + row + column * nbf] =
      raw[pair * naux + auxiliary];
}

/** Reduce Coulomb and exchange quadratic responses for one coordinate. */
__global__ void reduce_force_response_kernel(
    std::size_t naux, const double* charge, const double* derivative_charge,
    const double* inverse, const double* inverse_derivative,
    const double* exchange_quadratic,
    const double* derivative_exchange_quadratic, double coulomb_coefficient,
    double exchange_coefficient, double* output) {
  const std::size_t thread =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t threads =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  double value = 0.0;
  for (std::size_t auxiliary = thread; auxiliary < naux;
       auxiliary += threads) {
    double potential = 0.0;
    for (std::size_t source = 0; source < naux; ++source) {
      potential += inverse[auxiliary * naux + source] * charge[source];
    }
    value += coulomb_coefficient * derivative_charge[auxiliary] * potential;
  }
  for (std::size_t item = thread; item < naux * naux; item += threads) {
    const std::size_t row = item / naux;
    const std::size_t column = item % naux;
    const std::size_t column_major_item = column * naux + row;
    value += coulomb_coefficient *
             (0.5 * charge[row] * inverse_derivative[item] * charge[column]);
    value -= exchange_coefficient *
             (derivative_exchange_quadratic[column_major_item] *
                  inverse[item] +
              exchange_quadratic[column_major_item] *
                  inverse_derivative[item]);
  }
  __shared__ double partial[kThreads];
  partial[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride != 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(output, partial[0]);
}

__global__ void gather_auxiliary_tile_kernel(
    std::size_t matrix_elements, std::size_t naux, std::size_t system,
    std::size_t auxiliary_begin, std::size_t auxiliary_count,
    const double* three_center, double* tile) {
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t tile_elements = auxiliary_count * matrix_elements;
  if (element >= tile_elements) return;
  const std::size_t auxiliary = element / matrix_elements;
  const std::size_t pair = element % matrix_elements;
  tile[element] = three_center[(system * matrix_elements + pair) * naux +
                               auxiliary_begin + auxiliary];
}

/** Contract one streamed AO-pair tile into the active auxiliary charges. */
__global__ void accumulate_streamed_auxiliary_density_kernel(
    std::size_t pair_count, std::size_t auxiliary_count,
    const double* tile, const double* density, double* auxiliary_density) {
  const std::size_t auxiliary =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (auxiliary >= auxiliary_count) return;
  double value = 0.0;
  for (std::size_t pair = 0; pair < pair_count; ++pair) {
    value += tile[pair * auxiliary_count + auxiliary] * density[pair];
  }
  auxiliary_density[auxiliary] += value;
}

/** Form one streamed AO-pair segment from the active auxiliary charges. */
__global__ void build_streamed_coulomb_tile_kernel(
    std::size_t pair_count, std::size_t auxiliary_count,
    const double* tile, const double* auxiliary_density, double* coulomb) {
  const std::size_t pair =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pair >= pair_count) return;
  double value = 0.0;
  for (std::size_t auxiliary = 0; auxiliary < auxiliary_count; ++auxiliary) {
    value += tile[pair * auxiliary_count + auxiliary] *
             auxiliary_density[auxiliary];
  }
  // Auxiliary tiles are visited sequentially; retain the partial sum from
  // earlier tiles while each AO-pair segment remains disjoint.
  coulomb[pair] += value;
}

__global__ void reduce_exchange_tile_kernel(std::size_t matrix_elements,
                                            std::size_t auxiliary_count,
                                            std::size_t system,
                                            const double* contributions,
                                            double* exchange) {
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= matrix_elements) return;
  double value = 0.0;
  for (std::size_t auxiliary = 0; auxiliary < auxiliary_count; ++auxiliary) {
    value += contributions[auxiliary * matrix_elements + element];
  }
  exchange[system * matrix_elements + element] += value;
}

/** Reduce one auxiliary tile and scatter a row/column exchange subtile. */
__global__ void reduce_exchange_row_tile_kernel(
    std::size_t nbf, std::size_t row_begin, std::size_t row_count,
    std::size_t column_begin, std::size_t column_count,
    std::size_t auxiliary_count, std::size_t system,
    const double* contributions, double* exchange) {
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t tile_elements = row_count * column_count;
  if (element >= tile_elements) return;
  const std::size_t row = element / column_count;
  const std::size_t column = element % column_count;
  double value = 0.0;
  for (std::size_t auxiliary = 0; auxiliary < auxiliary_count;
       ++auxiliary) {
    // cuBLAS writes the row_count x column_count result column-major.  Read
    // it as [column][row] while scattering the public row-major [row][column]
    // tile, preserving the orientation for non-symmetric test densities.
    value += contributions[auxiliary * tile_elements + column * row_count +
                           row];
  }
  exchange[system * nbf * nbf + (row_begin + row) * nbf +
           column_begin + column] += value;
}

/** Assemble the closed-shell Fock matrix from the device-resident DF J/K. */
__global__ void assemble_rhf_fock_kernel(std::size_t elements,
                                         const double* hcore,
                                         const double* coulomb,
                                         const double* exchange,
                                         double* fock) {
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= elements) return;
  fock[element] = hcore[element] + coulomb[element] - 0.5 * exchange[element];
}

/** Assemble both unrestricted spin Fock matrices on device. */
__global__ void assemble_uhf_fock_kernel(std::size_t elements,
                                         const double* hcore,
                                         const double* coulomb,
                                         const double* alpha_exchange,
                                         const double* beta_exchange,
                                         double* alpha_fock,
                                         double* beta_fock) {
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= elements) return;
  alpha_fock[element] = hcore[element] + coulomb[element] -
                        alpha_exchange[element];
  beta_fock[element] = hcore[element] + coulomb[element] -
                       beta_exchange[element];
}

/** Build a density from the occupied columns of device-resident orbitals. */
__global__ void build_device_density_kernel(
    std::size_t batch_size, std::size_t nbf, const std::int32_t* occupied,
    const double* coefficients, double occupation_weight, double* density) {
  const std::size_t matrix_elements = nbf * nbf;
  const std::size_t element =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= batch_size * matrix_elements) return;
  const std::size_t system = element / matrix_elements;
  const std::size_t local = element % matrix_elements;
  const std::size_t row = local % nbf;
  const std::size_t column = local / nbf;
  const std::size_t offset = system * matrix_elements;
  double value = 0.0;
  for (std::int32_t orbital = 0; orbital < occupied[system]; ++orbital) {
    value += occupation_weight *
             coefficients[offset + row + orbital * nbf] *
             coefficients[offset + column + orbital * nbf];
  }
  density[element] = value;
}

/** Device energy reduction; one warp owns each physical system. */
__global__ void compute_device_energy_kernel(
    std::size_t batch_size, std::size_t nbf, const double* density,
    const double* hcore, const double* fock, const double* nuclear_repulsion,
    double* energy) {
  const std::size_t system = blockIdx.x;
  if (system >= batch_size) return;
  const std::size_t matrix_elements = nbf * nbf;
  const std::size_t offset = system * matrix_elements;
  double value = 0.0;
  for (std::size_t element = threadIdx.x; element < matrix_elements;
       element += blockDim.x) {
    value += 0.5 * density[offset + element] *
             (hcore[offset + element] + fock[offset + element]);
  }
  for (unsigned delta = warpSize / 2; delta != 0; delta >>= 1) {
    value += __shfl_down_sync(0xffffffffU, value, delta);
  }
  if (threadIdx.x == 0) energy[system] = value + nuclear_repulsion[system];
}

/** Device UHF energy reduction over alpha and beta spin densities. */
__global__ void compute_device_uhf_energy_kernel(
    std::size_t batch_size, std::size_t nbf, const double* alpha_density,
    const double* beta_density, const double* hcore, const double* alpha_fock,
    const double* beta_fock, const double* nuclear_repulsion,
    double* energy) {
  const std::size_t system = blockIdx.x;
  if (system >= batch_size) return;
  const std::size_t matrix_elements = nbf * nbf;
  const std::size_t physical_offset = system * matrix_elements;
  const std::size_t spin_offset = system * matrix_elements;
  double value = 0.0;
  for (std::size_t element = threadIdx.x; element < matrix_elements;
       element += blockDim.x) {
    value += 0.5 * alpha_density[spin_offset + element] *
             (hcore[physical_offset + element] +
              alpha_fock[spin_offset + element]);
    value += 0.5 * beta_density[spin_offset + element] *
             (hcore[physical_offset + element] +
              beta_fock[spin_offset + element]);
  }
  for (unsigned delta = warpSize / 2; delta != 0; delta >>= 1) {
    value += __shfl_down_sync(0xffffffffU, value, delta);
  }
  if (threadIdx.x == 0) energy[system] = value + nuclear_repulsion[system];
}

/** Update convergence state and advance the resident density in one kernel. */
__global__ void update_device_convergence_kernel(
    std::size_t batch_size, std::size_t nbf, double energy_tolerance,
    double density_tolerance, const double* energy, double* previous_energy,
    const double* next_density, double* density, std::uint8_t* active,
    std::uint8_t* converged, std::uint32_t* iterations,
    double* energy_change, double* density_rms) {
  const std::size_t system = blockIdx.x;
  if (system >= batch_size || active[system] == 0) return;
  const std::size_t matrix_elements = nbf * nbf;
  const std::size_t offset = system * matrix_elements;
  double square = 0.0;
  for (std::size_t element = threadIdx.x; element < matrix_elements;
       element += blockDim.x) {
    const double delta = next_density[offset + element] - density[offset + element];
    square += delta * delta;
  }
  for (unsigned delta = warpSize / 2; delta != 0; delta >>= 1) {
    square += __shfl_down_sync(0xffffffffU, square, delta);
  }
  if (threadIdx.x == 0) {
    const std::uint32_t iteration = iterations[system] + 1;
    const bool has_baseline = isfinite(previous_energy[system]);
    const double change = has_baseline
        ? fabs(energy[system] - previous_energy[system]) : CUDART_INF;
    const double rms = sqrt(square / static_cast<double>(matrix_elements));
    iterations[system] = iteration;
    energy_change[system] = change;
    density_rms[system] = rms;
    if ((iteration > 1 || has_baseline) && change < energy_tolerance &&
        rms < density_tolerance) {
      converged[system] = 1;
      active[system] = 0;
    } else {
      previous_energy[system] = energy[system];
    }
  }
  __syncwarp();
  // Every iteration advances the resident density, including the converged
  // one.  This makes the final host copy the density associated with the
  // convergence test and avoids a second device-to-device staging pass.
  for (std::size_t element = threadIdx.x; element < matrix_elements;
       element += blockDim.x) {
    density[offset + element] = next_density[offset + element];
  }
}

/** UHF convergence update using both spin density differences. */
__global__ void update_device_uhf_convergence_kernel(
    std::size_t batch_size, std::size_t nbf, double energy_tolerance,
    double density_tolerance, const double* energy, double* previous_energy,
    const double* next_alpha, const double* next_beta, double* alpha_density,
    double* beta_density, std::uint8_t* active, std::uint8_t* converged,
    std::uint32_t* iterations, double* energy_change, double* density_rms) {
  const std::size_t system = blockIdx.x;
  if (system >= batch_size || active[system] == 0) return;
  const std::size_t matrix_elements = nbf * nbf;
  const std::size_t offset = system * matrix_elements;
  double square = 0.0;
  for (std::size_t element = threadIdx.x; element < matrix_elements;
       element += blockDim.x) {
    const double da = next_alpha[offset + element] - alpha_density[offset + element];
    const double db = next_beta[offset + element] - beta_density[offset + element];
    square += da * da + db * db;
  }
  for (unsigned delta = warpSize / 2; delta != 0; delta >>= 1) {
    square += __shfl_down_sync(0xffffffffU, square, delta);
  }
  if (threadIdx.x == 0) {
    const std::uint32_t iteration = iterations[system] + 1;
    const bool has_baseline = isfinite(previous_energy[system]);
    const double change = has_baseline
        ? fabs(energy[system] - previous_energy[system]) : CUDART_INF;
    const double rms = sqrt(square / static_cast<double>(2 * matrix_elements));
    iterations[system] = iteration;
    energy_change[system] = change;
    density_rms[system] = rms;
    if ((iteration > 1 || has_baseline) && change < energy_tolerance &&
        rms < density_tolerance) {
      converged[system] = 1;
      active[system] = 0;
    } else {
      previous_energy[system] = energy[system];
    }
  }
  __syncwarp();
  for (std::size_t element = threadIdx.x; element < matrix_elements;
       element += blockDim.x) {
    alpha_density[offset + element] = next_alpha[offset + element];
    beta_density[offset + element] = next_beta[offset + element];
  }
}

struct SetupBuffers {
  double* metrics{};
  double* eigenvalues{};
  double* scales{};
  double* scaled_eigenvectors{};
  double* inverse_square_roots{};
  double* raw_three_center{};
  void* solver_workspace{};
  std::vector<unsigned char> solver_host_workspace;
  int* solver_info{};

  ~SetupBuffers() {
    (void)cudaFree(metrics);
    (void)cudaFree(eigenvalues);
    (void)cudaFree(scales);
    (void)cudaFree(scaled_eigenvectors);
    (void)cudaFree(inverse_square_roots);
    (void)cudaFree(raw_three_center);
    (void)cudaFree(solver_workspace);
    (void)cudaFree(solver_info);
  }
};

}  // namespace

struct CudaDensityFittingJkPlan {
  int device_id{-1};
  std::size_t batch_size{};
  std::size_t nbf{};
  std::size_t naux{};
  std::size_t matrix_elements{};
  std::size_t tensor_elements_per_system{};
  std::size_t auxiliary_tile{};
  std::size_t ao_pair_tile{};
  std::size_t row_tile{};
  cudaStream_t stream{};
  cublasHandle_t blas{};
  cusolverDnHandle_t solver{};
  cusolverDnParams_t solver_parameters{};
  double* three_center{};
  double* primary_density{};
  double* secondary_density{};
  double* total_density{};
  double* auxiliary_density{};
  double* coulomb{};
  double* alpha_exchange{};
  double* beta_exchange{};
  double* auxiliary_tile_values{};
  double* exchange_intermediate{};
  double* exchange_contributions{};
  double* exchange_tile_output{};
  double* exchange_density_column_major{};
  // Partial auxiliary tiles use a host-side raw tensor and inverse metric;
  // only the current transformed tile is resident on the device.
  bool streamed{};
  std::vector<double> streamed_raw_three_center;
  std::vector<double> streamed_inverse_square_roots;
  // Opaque persistent SCF state.  The definition lives with the device
  // solver helpers below so this public plan layout never exposes CUDA graph
  // or cuSOLVER implementation types to callers.
  void* persistent_scf_state{};
};

namespace {

void destroy_persistent_scf_state(void*& state) noexcept;

void release(CudaDensityFittingJkPlan& plan) noexcept {
  if (plan.device_id >= 0) (void)cudaSetDevice(plan.device_id);
  destroy_persistent_scf_state(plan.persistent_scf_state);
  (void)cudaFree(plan.three_center);
  (void)cudaFree(plan.primary_density);
  (void)cudaFree(plan.secondary_density);
  (void)cudaFree(plan.total_density);
  (void)cudaFree(plan.auxiliary_density);
  (void)cudaFree(plan.coulomb);
  (void)cudaFree(plan.alpha_exchange);
  (void)cudaFree(plan.beta_exchange);
  (void)cudaFree(plan.auxiliary_tile_values);
  (void)cudaFree(plan.exchange_intermediate);
  (void)cudaFree(plan.exchange_contributions);
  (void)cudaFree(plan.exchange_tile_output);
  (void)cudaFree(plan.exchange_density_column_major);
  if (plan.solver_parameters != nullptr) {
    (void)cusolverDnDestroyParams(plan.solver_parameters);
  }
  if (plan.solver != nullptr) (void)cusolverDnDestroy(plan.solver);
  if (plan.blas != nullptr) (void)cublasDestroy(plan.blas);
  if (plan.stream != nullptr) (void)cudaStreamDestroy(plan.stream);
  plan = {};
}

vibeqc_status fail_plan(CudaDensityFittingJkPlan* plan, vibeqc_status status) {
  if (plan != nullptr) {
    release(*plan);
    delete plan;
  }
  return status;
}

vibeqc_status allocate_device(void** pointer, std::size_t bytes,
                              const char* description, std::string& detail) {
  const cudaError_t error = cudaMalloc(pointer, bytes);
  return error == cudaSuccess ? VIBEQC_STATUS_SUCCESS
                              : cuda_failure(error, description, detail);
}

vibeqc_status build_coulomb(CudaDensityFittingJkPlan& plan,
                            const double* density, std::string& detail) {
  if (plan.streamed) {
    cudaError_t cuda_error = cudaMemsetAsync(
        plan.auxiliary_density, 0,
        plan.batch_size * plan.naux * sizeof(double), plan.stream);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemsetAsync(
          plan.coulomb, 0,
          plan.batch_size * plan.matrix_elements * sizeof(double),
          plan.stream);
    }
    if (cuda_error != cudaSuccess) {
      return cuda_failure(cuda_error, "zero streamed DF Coulomb buffers",
                          detail);
    }
    // The staged device tile is sized from row_tile * nbf.  The planner's
    // AO-pair tile is a logical budget and may not be divisible by nbf, so
    // process at most the physically allocated row-major capacity per pass.
    const std::size_t pair_tile_capacity = plan.row_tile * plan.nbf;
    const std::size_t pair_tile =
        std::min(plan.ao_pair_tile, pair_tile_capacity);
    std::vector<double> host_tile;
    try {
      host_tile.resize(plan.auxiliary_tile * pair_tile_capacity);
    } catch (const std::bad_alloc&) {
      detail = "host allocation for streamed CUDA DF Coulomb tile failed";
      return VIBEQC_STATUS_OUT_OF_MEMORY;
    }
    for (std::size_t system = 0; system < plan.batch_size; ++system) {
      const double* raw = plan.streamed_raw_three_center.data() +
                          system * plan.tensor_elements_per_system;
      const double* inverse = plan.streamed_inverse_square_roots.data() +
                              system * plan.naux * plan.naux;
      const double* system_density = density + system * plan.matrix_elements;
      for (std::size_t begin = 0; begin < plan.naux;
           begin += plan.auxiliary_tile) {
        const std::size_t count =
            std::min(plan.auxiliary_tile, plan.naux - begin);
        // First accumulate the auxiliary density over bounded AO-pair tiles.
        for (std::size_t pair_begin = 0; pair_begin < plan.matrix_elements;
             pair_begin += pair_tile) {
          const std::size_t pair_count =
              std::min(pair_tile, plan.matrix_elements - pair_begin);
          for (std::size_t pair = 0; pair < pair_count; ++pair) {
            for (std::size_t auxiliary = 0; auxiliary < count; ++auxiliary) {
              double value = 0.0;
              for (std::size_t source = 0; source < plan.naux; ++source) {
                value += raw[(pair_begin + pair) * plan.naux + source] *
                         inverse[(begin + auxiliary) * plan.naux + source];
              }
              // Keep auxiliary contiguous within each pair so the same
              // staging layout is usable by the streamed contraction kernel.
              host_tile[pair * count + auxiliary] = value;
            }
          }
          const std::size_t tile_bytes = pair_count * count * sizeof(double);
          cuda_error = cudaMemcpyAsync(plan.auxiliary_tile_values,
                                       host_tile.data(), tile_bytes,
                                       cudaMemcpyHostToDevice, plan.stream);
          if (cuda_error != cudaSuccess) {
            return cuda_failure(cuda_error,
                                "upload streamed DF Coulomb pair tile",
                                detail);
          }
          accumulate_streamed_auxiliary_density_kernel<<<
              blocks_for(count), kThreads, 0, plan.stream>>>(
              pair_count, count, plan.auxiliary_tile_values,
              system_density + pair_begin,
              plan.auxiliary_density + system * plan.naux + begin);
          cuda_error = cudaPeekAtLastError();
          if (cuda_error != cudaSuccess) {
            return cuda_failure(
                cuda_error, "streamed DF auxiliary-density pair tile", detail);
          }
          cuda_error = cudaStreamSynchronize(plan.stream);
          if (cuda_error != cudaSuccess) {
            return cuda_failure(cuda_error,
                                "finish streamed DF Coulomb pair tile",
                                detail);
          }
        }

        // Revisit the same bounded tiles to form the AO Coulomb output.  AO
        // segments are disjoint, while auxiliary tiles accumulate into the
        // same output through the single stream (so no atomics are needed).
        for (std::size_t pair_begin = 0; pair_begin < plan.matrix_elements;
             pair_begin += pair_tile) {
          const std::size_t pair_count =
              std::min(pair_tile, plan.matrix_elements - pair_begin);
          for (std::size_t pair = 0; pair < pair_count; ++pair) {
            for (std::size_t auxiliary = 0; auxiliary < count; ++auxiliary) {
              double value = 0.0;
              for (std::size_t source = 0; source < plan.naux; ++source) {
                value += raw[(pair_begin + pair) * plan.naux + source] *
                         inverse[(begin + auxiliary) * plan.naux + source];
              }
              host_tile[pair * count + auxiliary] = value;
            }
          }
          const std::size_t tile_bytes = pair_count * count * sizeof(double);
          cuda_error = cudaMemcpyAsync(plan.auxiliary_tile_values,
                                       host_tile.data(), tile_bytes,
                                       cudaMemcpyHostToDevice, plan.stream);
          if (cuda_error != cudaSuccess) {
            return cuda_failure(cuda_error,
                                "upload streamed DF Coulomb output tile",
                                detail);
          }
          build_streamed_coulomb_tile_kernel<<<
              blocks_for(pair_count), kThreads, 0, plan.stream>>>(
              pair_count, count, plan.auxiliary_tile_values,
              plan.auxiliary_density + system * plan.naux + begin,
              plan.coulomb + system * plan.matrix_elements + pair_begin);
          cuda_error = cudaPeekAtLastError();
          if (cuda_error != cudaSuccess) {
            return cuda_failure(cuda_error,
                                "streamed DF Coulomb output tile", detail);
          }
          cuda_error = cudaStreamSynchronize(plan.stream);
          if (cuda_error != cudaSuccess) {
            return cuda_failure(cuda_error,
                                "finish streamed DF Coulomb output tile",
                                detail);
          }
        }
      }
    }
    return VIBEQC_STATUS_SUCCESS;
  }

  const int batch_size = static_cast<int>(plan.batch_size);
  const int matrix_elements = static_cast<int>(plan.matrix_elements);
  const int naux = static_cast<int>(plan.naux);
  const long long tensor_stride =
      static_cast<long long>(plan.tensor_elements_per_system);
  const long long matrix_stride = static_cast<long long>(plan.matrix_elements);
  const long long auxiliary_stride = static_cast<long long>(plan.naux);
  const double one = 1.0;
  const double zero = 0.0;
  cublasStatus_t blas_status = cublasDgemmStridedBatched(
      plan.blas, CUBLAS_OP_N, CUBLAS_OP_N, naux, 1, matrix_elements, &one,
      plan.three_center, naux, tensor_stride, density, matrix_elements,
      matrix_stride, &zero, plan.auxiliary_density, naux, auxiliary_stride,
      batch_size);
  if (blas_status != CUBLAS_STATUS_SUCCESS) {
    return blas_failure(blas_status, "DF auxiliary-density contraction",
                        detail);
  }
  blas_status = cublasDgemmStridedBatched(
      plan.blas, CUBLAS_OP_T, CUBLAS_OP_N, matrix_elements, 1, naux, &one,
      plan.three_center, naux, tensor_stride, plan.auxiliary_density, naux,
      auxiliary_stride, &zero, plan.coulomb, matrix_elements, matrix_stride,
      batch_size);
  return blas_status == CUBLAS_STATUS_SUCCESS
             ? VIBEQC_STATUS_SUCCESS
             : blas_failure(blas_status, "DF Coulomb contraction", detail);
}

vibeqc_status build_exchange(CudaDensityFittingJkPlan& plan,
                             const double* density, double* exchange,
                             std::string& detail,
                             bool density_is_column_major = false) {
  const std::size_t output_elements = plan.batch_size * plan.matrix_elements;
  cudaError_t cuda_error = cudaMemsetAsync(
      exchange, 0, output_elements * sizeof(double), plan.stream);
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "zero DF exchange output", detail);
  }

  const double one = 1.0;
  const double zero = 0.0;

  // The resident path keeps the original full-matrix batched GEMMs.  The
  // streamed path below uses bounded AO row blocks, so every device tile is
  // at most row_tile * nbf rather than auxiliary_tile * nbf^2.
  if (plan.streamed) {
    const std::size_t pair_capacity = plan.row_tile * plan.nbf;
    std::vector<double> host_tile;
    try {
      host_tile.resize(plan.auxiliary_tile * pair_capacity);
    } catch (const std::bad_alloc&) {
      detail = "host allocation for streamed CUDA DF exchange tile failed";
      return VIBEQC_STATUS_OUT_OF_MEMORY;
    }

    // cuBLAS consumes column-major matrices while public host densities and
    // streamed raw tensors are row-major.  Transpose host inputs explicitly;
    // device-resident SCF callers can opt out when their density is already
    // in cuBLAS layout.  This keeps rectangular row-block GEMMs independent
    // of density-symmetry assumptions.
    for (std::size_t system = 0; system < plan.batch_size; ++system) {
      const double* system_density = density + system * plan.matrix_elements;
      const double* density_column_major = system_density;
      if (!density_is_column_major) {
        density_column_major =
            plan.exchange_density_column_major + system * plan.matrix_elements;
        transpose_density_kernel<<<blocks_for(plan.matrix_elements), kThreads,
                                   0, plan.stream>>>(
            plan.nbf, system_density,
            plan.exchange_density_column_major + system * plan.matrix_elements);
        cudaError_t transpose_error = cudaPeekAtLastError();
        if (transpose_error != cudaSuccess) {
          return cuda_failure(transpose_error,
                              "transpose streamed DF exchange density", detail);
        }
      }

      const double* raw = plan.streamed_raw_three_center.data() +
                          system * plan.tensor_elements_per_system;
      const double* inverse = plan.streamed_inverse_square_roots.data() +
                              system * plan.naux * plan.naux;
      for (std::size_t auxiliary_begin = 0; auxiliary_begin < plan.naux;
           auxiliary_begin += plan.auxiliary_tile) {
        const std::size_t auxiliary_count =
            std::min(plan.auxiliary_tile, plan.naux - auxiliary_begin);

        for (std::size_t row_begin = 0; row_begin < plan.nbf;
             row_begin += plan.row_tile) {
          const std::size_t row_count =
              std::min(plan.row_tile, plan.nbf - row_begin);
          const std::size_t pair_count = row_count * plan.nbf;
          // Use the compact stride for this (possibly partial) row block.
          // The staging buffer is allocated for pair_capacity, but copying a
          // partial final block with that larger stride would include gaps
          // between auxiliary tiles and truncate the later tiles on device.
          const std::size_t row_stride = pair_count;

          // Transform this AO-row block for the active auxiliary tile.  The
          // host staging layout is [auxiliary][row][column] row-major; cuBLAS
          // interprets each block as the column-major transpose B_A^T.
          for (std::size_t auxiliary = 0; auxiliary < auxiliary_count;
               ++auxiliary) {
            for (std::size_t row = 0; row < row_count; ++row) {
              for (std::size_t column = 0; column < plan.nbf; ++column) {
                const std::size_t pair = (row_begin + row) * plan.nbf + column;
                double value = 0.0;
                for (std::size_t source = 0; source < plan.naux; ++source) {
                  value += raw[pair * plan.naux + source] *
                           inverse[(auxiliary_begin + auxiliary) * plan.naux +
                                   source];
                }
                host_tile[auxiliary * row_stride + row * plan.nbf +
                          column] = value;
              }
            }
          }
          cuda_error = cudaMemcpyAsync(
              plan.auxiliary_tile_values, host_tile.data(),
              auxiliary_count * row_stride * sizeof(double),
              cudaMemcpyHostToDevice, plan.stream);
          if (cuda_error != cudaSuccess) {
            return cuda_failure(cuda_error,
                                "upload streamed DF exchange row tile", detail);
          }

          // T_A = D^T * B_A^T, stored as nbf x row_count column-major.  The
          // transpose is B_A * D, which is the left factor of K_AB.
          cublasStatus_t blas_status = cublasDgemmStridedBatched(
              plan.blas, CUBLAS_OP_T, CUBLAS_OP_N,
              static_cast<int>(plan.nbf), static_cast<int>(row_count),
              static_cast<int>(plan.nbf), &one,
              density_column_major, static_cast<int>(plan.nbf),
              0, plan.auxiliary_tile_values, static_cast<int>(plan.nbf),
              static_cast<long long>(row_stride), &zero,
              plan.exchange_contributions, static_cast<int>(plan.nbf),
              static_cast<long long>(row_stride),
              static_cast<int>(auxiliary_count));
          if (blas_status != CUBLAS_STATUS_SUCCESS) {
            return blas_failure(blas_status,
                                "streamed DF exchange row GEMM", detail);
          }

          for (std::size_t column_begin = 0; column_begin < plan.nbf;
               column_begin += plan.row_tile) {
            const std::size_t column_count =
                std::min(plan.row_tile, plan.nbf - column_begin);
            const std::size_t column_pair_count = column_count * plan.nbf;
            const std::size_t column_stride = column_pair_count;

            for (std::size_t auxiliary = 0; auxiliary < auxiliary_count;
                 ++auxiliary) {
              for (std::size_t row = 0; row < column_count; ++row) {
                for (std::size_t column = 0; column < plan.nbf; ++column) {
                  const std::size_t pair =
                      (column_begin + row) * plan.nbf + column;
                  double value = 0.0;
                  for (std::size_t source = 0; source < plan.naux; ++source) {
                    value += raw[pair * plan.naux + source] *
                             inverse[(auxiliary_begin + auxiliary) *
                                          plan.naux + source];
                  }
                  host_tile[auxiliary * column_stride + row * plan.nbf +
                            column] = value;
                }
              }
            }
            cuda_error = cudaMemcpyAsync(
                plan.exchange_intermediate, host_tile.data(),
                auxiliary_count * column_stride * sizeof(double),
                cudaMemcpyHostToDevice, plan.stream);
            if (cuda_error != cudaSuccess) {
              return cuda_failure(
                  cuda_error, "upload streamed DF exchange column tile",
                  detail);
            }

            // K_AB = (B_A D) B_B^T.  T_A is stored as D^T B_A^T, so OP_T on
            // it gives B_A D; B_B is stored as B_B^T.  The result is emitted
            // as a column-major [row_count x column_count] tile and the
            // reduction kernel transposes that view while scattering.
            const std::size_t output_stride = row_count * column_count;
            blas_status = cublasDgemmStridedBatched(
                plan.blas, CUBLAS_OP_T, CUBLAS_OP_N,
                static_cast<int>(row_count), static_cast<int>(column_count),
                static_cast<int>(plan.nbf), &one,
                plan.exchange_contributions, static_cast<int>(plan.nbf),
                static_cast<long long>(row_stride),
                plan.exchange_intermediate, static_cast<int>(plan.nbf),
                static_cast<long long>(column_stride),
                &zero, plan.exchange_tile_output, static_cast<int>(row_count),
                static_cast<long long>(output_stride),
                static_cast<int>(auxiliary_count));
            if (blas_status != CUBLAS_STATUS_SUCCESS) {
              return blas_failure(blas_status,
                                  "streamed DF exchange column GEMM", detail);
            }
            reduce_exchange_row_tile_kernel<<<
                blocks_for(output_stride), kThreads, 0, plan.stream>>>(
                plan.nbf, row_begin, row_count, column_begin, column_count,
                auxiliary_count, system, plan.exchange_tile_output, exchange);
            cuda_error = cudaPeekAtLastError();
            if (cuda_error != cudaSuccess) {
              return cuda_failure(cuda_error,
                                  "reduce streamed DF exchange tile", detail);
            }

            // Host staging and the two reusable device input buffers are
            // overwritten on the next column block; fence this block before
            // reusing them.  This is intentionally conservative and keeps
            // correctness independent of pageable-host-copy behavior.
            cuda_error = cudaStreamSynchronize(plan.stream);
            if (cuda_error != cudaSuccess) {
              return cuda_failure(cuda_error,
                                  "finish streamed DF exchange tile", detail);
            }
          }
        }
      }
    }
    return VIBEQC_STATUS_SUCCESS;
  }

  const int nbf = static_cast<int>(plan.nbf);
  const long long matrix_stride = static_cast<long long>(plan.matrix_elements);
  for (std::size_t system = 0; system < plan.batch_size; ++system) {
    const double* system_density = density + system * plan.matrix_elements;
    double* transposed_density =
        plan.exchange_density_column_major + system * plan.matrix_elements;
    const double* density_column_major = system_density;
    if (!density_is_column_major) {
      density_column_major = transposed_density;
      transpose_density_kernel<<<blocks_for(plan.matrix_elements), kThreads, 0,
                                 plan.stream>>>(
          plan.nbf, system_density, transposed_density);
      cuda_error = cudaPeekAtLastError();
      if (cuda_error != cudaSuccess) {
        return cuda_failure(cuda_error, "transpose CUDA DF exchange density",
                            detail);
      }
    }
    for (std::size_t auxiliary_begin = 0; auxiliary_begin < plan.naux;
         auxiliary_begin += plan.auxiliary_tile) {
      const std::size_t auxiliary_count =
          std::min(plan.auxiliary_tile, plan.naux - auxiliary_begin);
      const std::size_t tile_elements = auxiliary_count * plan.matrix_elements;
      gather_auxiliary_tile_kernel<<<blocks_for(tile_elements), kThreads, 0,
                                     plan.stream>>>(
          plan.matrix_elements, plan.naux, system, auxiliary_begin,
          auxiliary_count, plan.three_center, plan.auxiliary_tile_values);
      cuda_error = cudaPeekAtLastError();
      if (cuda_error != cudaSuccess) {
        return cuda_failure(cuda_error, "gather DF exchange tile", detail);
      }

      const int tile_count = static_cast<int>(auxiliary_count);
      cublasStatus_t blas_status = cublasDgemmStridedBatched(
          // The gathered AO-pair tile is B^T in cuBLAS layout.  Use D^T as
          // the first factor; the reduction below maps the column-major
          // result back to row-major public storage, yielding B D B^T.
          plan.blas, CUBLAS_OP_T, CUBLAS_OP_N, nbf, nbf, nbf, &one,
          density_column_major, nbf, 0, plan.auxiliary_tile_values, nbf,
          matrix_stride, &zero, plan.exchange_intermediate, nbf, matrix_stride,
          tile_count);
      if (blas_status != CUBLAS_STATUS_SUCCESS) {
        return blas_failure(blas_status, "DF exchange first GEMM", detail);
      }
      blas_status = cublasDgemmStridedBatched(
          plan.blas, CUBLAS_OP_T, CUBLAS_OP_N, nbf, nbf, nbf, &one,
          plan.auxiliary_tile_values, nbf, matrix_stride,
          plan.exchange_intermediate, nbf, matrix_stride, &zero,
          plan.exchange_contributions, nbf, matrix_stride, tile_count);
      if (blas_status != CUBLAS_STATUS_SUCCESS) {
        return blas_failure(blas_status, "DF exchange second GEMM", detail);
      }
      reduce_exchange_tile_kernel<<<blocks_for(plan.matrix_elements), kThreads,
                                    0, plan.stream>>>(
          plan.matrix_elements, auxiliary_count, system,
          plan.exchange_contributions, exchange);
      cuda_error = cudaPeekAtLastError();
      if (cuda_error != cudaSuccess) {
        return cuda_failure(cuda_error, "reduce DF exchange tile", detail);
      }
    }
  }
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status prepare_outputs(std::size_t elements, std::vector<double>& first,
                              std::vector<double>& second,
                              std::string& detail) {
  try {
    first.assign(elements, 0.0);
    second.assign(elements, 0.0);
  } catch (const std::bad_alloc&) {
    detail = "host allocation for CUDA DF J/K output failed";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
  return VIBEQC_STATUS_SUCCESS;
}

bool validate_execution_input(const CudaDensityFittingJkPlan* plan,
                              const std::vector<double>& density,
                              std::string& detail) {
  if (plan == nullptr) {
    detail = "CUDA DF J/K plan is null";
    return false;
  }
  const std::size_t expected = plan->batch_size * plan->matrix_elements;
  if (density.size() != expected || !finite_values(density)) {
    detail = "CUDA DF density dimensions or values are invalid";
    return false;
  }
  return true;
}

}  // namespace

vibeqc_status create_cuda_density_fitting_jk_plan_tiled(
    int device_id, std::size_t batch_size, std::size_t nbf, std::size_t naux,
    const std::vector<double>& metrics, const std::vector<double>& three_center,
    double relative_threshold, std::size_t auxiliary_tile,
    std::size_t ao_pair_tile,
    CudaDensityFittingJkPlan** plan,
    std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics,
    std::string& detail) {
  detail.clear();
  diagnostics.clear();
  if (plan == nullptr) return VIBEQC_STATUS_INVALID_ARGUMENT;
  *plan = nullptr;
  if (device_id < 0 || batch_size == 0 || nbf == 0 || naux == 0 ||
      !(relative_threshold > 0.0) ||
      !(relative_threshold < 1.0) ||
      batch_size > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      batch_size > 65535 ||
      nbf > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      naux > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    detail = "CUDA DF plan dimensions or metric threshold are invalid";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }

  std::size_t matrix_elements = 0;
  std::size_t metric_elements = 0;
  std::size_t tensor_elements_per_system = 0;
  std::size_t all_matrix_elements = 0;
  std::size_t all_metric_elements = 0;
  std::size_t all_tensor_elements = 0;
  if (!checked_multiply(nbf, nbf, matrix_elements) ||
      !checked_multiply(naux, naux, metric_elements) ||
      !checked_multiply(matrix_elements, naux, tensor_elements_per_system) ||
      !checked_multiply(batch_size, matrix_elements, all_matrix_elements) ||
      !checked_multiply(batch_size, metric_elements, all_metric_elements) ||
      !checked_multiply(batch_size, tensor_elements_per_system,
                        all_tensor_elements) ||
      matrix_elements >
          static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
      metrics.size() != all_metric_elements ||
      three_center.size() != all_tensor_elements || !finite_values(metrics) ||
      !finite_values(three_center)) {
    detail = "CUDA DF plan buffers have invalid dimensions or values";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  auxiliary_tile =
      auxiliary_tile == 0 ? std::min<std::size_t>(naux, 32) : auxiliary_tile;
  if (auxiliary_tile > naux ||
      auxiliary_tile >
          static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    detail = "CUDA DF auxiliary tile is invalid";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  ao_pair_tile = ao_pair_tile == 0 ? matrix_elements : ao_pair_tile;
  if (ao_pair_tile > matrix_elements || ao_pair_tile == 0) {
    detail = "CUDA DF AO-pair tile is invalid";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }

  std::size_t matrix_bytes = 0;
  std::size_t metric_bytes = 0;
  std::size_t tensor_bytes = 0;
  std::size_t auxiliary_bytes = 0;
  std::size_t tile_elements = 0;
  std::size_t tile_bytes = 0;
  // Stream whenever either planner dimension is smaller than the full
  // transformed tensor.  This matters for small auxiliary bases where all
  // Q directions fit but the AO-pair budget still requires row staging.
  const bool streamed =
      auxiliary_tile < naux || ao_pair_tile < matrix_elements;
  const std::size_t staged_row_tile =
      streamed
          ? std::min<std::size_t>(
                nbf, std::max<std::size_t>(1, ao_pair_tile / nbf))
          : nbf;
  std::size_t staged_pair_capacity = 0;
  std::size_t auxiliary_vector_elements = 0;
  std::size_t auxiliary_vector_bytes = 0;
  std::size_t solver_info_bytes = 0;
  if (!checked_bytes(all_matrix_elements, matrix_bytes) ||
      !checked_bytes(all_metric_elements, metric_bytes) ||
      !checked_bytes(all_tensor_elements, tensor_bytes) ||
      !checked_multiply(batch_size, naux, tile_elements) ||
      !checked_bytes(tile_elements, auxiliary_bytes) ||
      !checked_multiply(batch_size, naux, auxiliary_vector_elements) ||
      !checked_bytes(auxiliary_vector_elements, auxiliary_vector_bytes) ||
      !checked_multiply(batch_size, sizeof(int), solver_info_bytes) ||
      !checked_multiply(staged_row_tile, nbf, staged_pair_capacity) ||
      !checked_multiply(auxiliary_tile, staged_pair_capacity, tile_elements) ||
      !checked_bytes(tile_elements, tile_bytes)) {
    detail = "CUDA DF plan storage overflows size_t";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }

  cudaError_t cuda_error = cudaSetDevice(device_id);
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "select CUDA DF device", detail);
  }
  auto* candidate = new (std::nothrow) CudaDensityFittingJkPlan{};
  if (candidate == nullptr) return VIBEQC_STATUS_OUT_OF_MEMORY;
  candidate->device_id = device_id;
  candidate->batch_size = batch_size;
  candidate->nbf = nbf;
  candidate->naux = naux;
  candidate->matrix_elements = matrix_elements;
  candidate->tensor_elements_per_system = tensor_elements_per_system;
  candidate->auxiliary_tile = auxiliary_tile;
  candidate->ao_pair_tile = ao_pair_tile;
  candidate->row_tile = staged_row_tile;
  candidate->streamed = streamed;
  if (candidate->streamed) {
    try {
      candidate->streamed_raw_three_center = three_center;
    } catch (const std::bad_alloc&) {
      delete candidate;
      return VIBEQC_STATUS_OUT_OF_MEMORY;
    }
  }

  cuda_error =
      cudaStreamCreateWithFlags(&candidate->stream, cudaStreamNonBlocking);
  if (cuda_error != cudaSuccess) {
    return fail_plan(candidate,
                     cuda_failure(cuda_error, "create CUDA DF stream", detail));
  }
  cublasStatus_t blas_status = cublasCreate(&candidate->blas);
  if (blas_status == CUBLAS_STATUS_SUCCESS) {
    blas_status = cublasSetStream(candidate->blas, candidate->stream);
  }
  if (blas_status == CUBLAS_STATUS_SUCCESS) {
    blas_status =
        cublasSetPointerMode(candidate->blas, CUBLAS_POINTER_MODE_HOST);
  }
  if (blas_status != CUBLAS_STATUS_SUCCESS) {
    return fail_plan(
        candidate,
        blas_failure(blas_status, "initialize CUDA DF cuBLAS", detail));
  }
  cusolverStatus_t solver_status = cusolverDnCreate(&candidate->solver);
  if (solver_status == CUSOLVER_STATUS_SUCCESS) {
    solver_status = cusolverDnSetStream(candidate->solver, candidate->stream);
  }
  if (solver_status == CUSOLVER_STATUS_SUCCESS) {
    solver_status = cusolverDnCreateParams(&candidate->solver_parameters);
  }
  if (solver_status != CUSOLVER_STATUS_SUCCESS) {
    return fail_plan(
        candidate,
        solver_failure(solver_status, "initialize CUDA DF cuSOLVER", detail));
  }

  auto allocate_permanent = [&](double** pointer, std::size_t bytes,
                                const char* description) {
    return allocate_device(reinterpret_cast<void**>(pointer), bytes,
                           description, detail);
  };
  vibeqc_status status = VIBEQC_STATUS_SUCCESS;
  if (!candidate->streamed) {
    status = allocate_permanent(&candidate->three_center, tensor_bytes,
                                "allocate transformed CUDA DF tensor");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->primary_density, matrix_bytes,
                                "allocate primary CUDA DF density");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->secondary_density, matrix_bytes,
                                "allocate secondary CUDA DF density");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->total_density, matrix_bytes,
                                "allocate total CUDA DF density");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->auxiliary_density, auxiliary_bytes,
                                "allocate CUDA DF auxiliary density");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->coulomb, matrix_bytes,
                                "allocate CUDA DF Coulomb matrices");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->alpha_exchange, matrix_bytes,
                                "allocate CUDA DF alpha exchange matrices");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->beta_exchange, matrix_bytes,
                                "allocate CUDA DF beta exchange matrices");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->auxiliary_tile_values, tile_bytes,
                                "allocate CUDA DF auxiliary tile");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->exchange_intermediate, tile_bytes,
                                "allocate CUDA DF exchange intermediate");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(&candidate->exchange_contributions, tile_bytes,
                                "allocate CUDA DF exchange contributions");
  }
  if (status == VIBEQC_STATUS_SUCCESS && candidate->streamed) {
    status = allocate_permanent(&candidate->exchange_tile_output, tile_bytes,
                                "allocate CUDA DF exchange tile output");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_permanent(
        &candidate->exchange_density_column_major, matrix_bytes,
        "allocate CUDA DF exchange density transpose");
  }
  if (status != VIBEQC_STATUS_SUCCESS) return fail_plan(candidate, status);

  SetupBuffers setup;
  auto allocate_setup = [&](void** pointer, std::size_t bytes,
                            const char* description) {
    return allocate_device(pointer, bytes, description, detail);
  };
  status = allocate_setup(reinterpret_cast<void**>(&setup.metrics),
                          metric_bytes, "allocate CUDA DF metric eigensystem");
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_setup(reinterpret_cast<void**>(&setup.eigenvalues),
                            auxiliary_vector_bytes,
                            "allocate CUDA DF metric eigenvalues");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_setup(reinterpret_cast<void**>(&setup.scales),
                            auxiliary_vector_bytes,
                            "allocate CUDA DF metric eigenvalue scales");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_setup(
        reinterpret_cast<void**>(&setup.scaled_eigenvectors), metric_bytes,
        "allocate scaled CUDA DF metric eigenvectors");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_setup(
        reinterpret_cast<void**>(&setup.inverse_square_roots), metric_bytes,
        "allocate CUDA DF metric inverse square roots");
  }
  if (status == VIBEQC_STATUS_SUCCESS && !candidate->streamed) {
    status = allocate_setup(reinterpret_cast<void**>(&setup.raw_three_center),
                            tensor_bytes,
                            "allocate raw CUDA DF three-center tensor");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate_setup(reinterpret_cast<void**>(&setup.solver_info),
                            solver_info_bytes,
                            "allocate CUDA DF solver status");
  }
  if (status != VIBEQC_STATUS_SUCCESS) return fail_plan(candidate, status);

  cuda_error = cudaMemcpyAsync(setup.metrics, metrics.data(), metric_bytes,
                               cudaMemcpyHostToDevice, candidate->stream);
  if (cuda_error == cudaSuccess && !candidate->streamed) {
    cuda_error = cudaMemcpyAsync(setup.raw_three_center, three_center.data(),
                                 tensor_bytes, cudaMemcpyHostToDevice,
                                 candidate->stream);
  }
  if (cuda_error != cudaSuccess) {
    return fail_plan(
        candidate,
        cuda_failure(cuda_error, "upload CUDA DF setup tensors", detail));
  }
  const dim3 symmetric_threads(16, 16, 1);
  const dim3 symmetric_blocks(
      static_cast<unsigned>((naux + symmetric_threads.x - 1) /
                            symmetric_threads.x),
      static_cast<unsigned>((naux + symmetric_threads.y - 1) /
                            symmetric_threads.y),
      static_cast<unsigned>(batch_size));
  symmetrize_metrics_kernel<<<symmetric_blocks, symmetric_threads, 0,
                              candidate->stream>>>(naux, setup.metrics);
  cuda_error = cudaPeekAtLastError();
  if (cuda_error != cudaSuccess) {
    return fail_plan(
        candidate,
        cuda_failure(cuda_error, "symmetrize CUDA DF metrics", detail));
  }

  std::size_t solver_device_workspace_bytes = 0;
  std::size_t solver_host_workspace_bytes = 0;
  solver_status = cusolverDnXsyevd_bufferSize(
      candidate->solver, candidate->solver_parameters,
      CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
      static_cast<std::int64_t>(naux), CUDA_R_64F, setup.metrics,
      static_cast<std::int64_t>(naux), CUDA_R_64F, setup.eigenvalues,
      CUDA_R_64F, &solver_device_workspace_bytes,
      &solver_host_workspace_bytes);
  if (solver_status != CUSOLVER_STATUS_SUCCESS) {
    return fail_plan(candidate,
                     solver_failure(solver_status,
                                    "size CUDA DF metric eigensolver", detail));
  }
  if (solver_device_workspace_bytes != 0) {
    status = allocate_setup(&setup.solver_workspace,
                            solver_device_workspace_bytes,
                            "allocate CUDA DF metric solver workspace");
    if (status != VIBEQC_STATUS_SUCCESS) return fail_plan(candidate, status);
  }
  try {
    setup.solver_host_workspace.resize(solver_host_workspace_bytes);
  } catch (const std::bad_alloc&) {
    detail = "host allocation for CUDA DF metric solver workspace failed";
    return fail_plan(candidate, VIBEQC_STATUS_OUT_OF_MEMORY);
  }
  for (std::size_t system = 0; system < batch_size; ++system) {
    solver_status = cusolverDnXsyevd(
        candidate->solver, candidate->solver_parameters,
        CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        static_cast<std::int64_t>(naux), CUDA_R_64F,
        setup.metrics + system * metric_elements,
        static_cast<std::int64_t>(naux), CUDA_R_64F,
        setup.eigenvalues + system * naux, CUDA_R_64F,
        setup.solver_workspace, solver_device_workspace_bytes,
        setup.solver_host_workspace.empty()
            ? nullptr
            : setup.solver_host_workspace.data(),
        solver_host_workspace_bytes, setup.solver_info + system);
    if (solver_status != CUSOLVER_STATUS_SUCCESS) {
      return fail_plan(
          candidate,
          solver_failure(solver_status, "diagonalize CUDA DF metric", detail));
    }
  }

  std::vector<double> eigenvalues;
  std::vector<double> scales;
  std::vector<int> solver_info;
  try {
    eigenvalues.resize(batch_size * naux);
    scales.assign(batch_size * naux, 0.0);
    solver_info.resize(batch_size);
    diagnostics.resize(batch_size);
  } catch (const std::bad_alloc&) {
    detail = "host allocation for CUDA DF metric diagnostics failed";
    return fail_plan(candidate, VIBEQC_STATUS_OUT_OF_MEMORY);
  }
  cuda_error = cudaMemcpyAsync(eigenvalues.data(), setup.eigenvalues,
                               eigenvalues.size() * sizeof(double),
                               cudaMemcpyDeviceToHost, candidate->stream);
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpyAsync(solver_info.data(), setup.solver_info,
                                 solver_info.size() * sizeof(int),
                                 cudaMemcpyDeviceToHost, candidate->stream);
  }
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaStreamSynchronize(candidate->stream);
  }
  if (cuda_error != cudaSuccess) {
    return fail_plan(
        candidate,
        cuda_failure(cuda_error, "read CUDA DF metric eigensystem", detail));
  }
  for (std::size_t system = 0; system < batch_size; ++system) {
    if (solver_info[system] != 0) {
      detail = "CUDA DF metric eigensolver did not converge for system " +
               std::to_string(system);
      return fail_plan(candidate, VIBEQC_STATUS_CUDA_ERROR);
    }
    const std::size_t offset = system * naux;
    const double largest = eigenvalues[offset + naux - 1];
    if (!(largest > 0.0) || !std::isfinite(largest)) {
      detail = "CUDA DF metric has no finite positive eigenspace for system " +
               std::to_string(system);
      return fail_plan(candidate, VIBEQC_STATUS_INVALID_ARGUMENT);
    }
    auto& diagnostic = diagnostics[system];
    diagnostic.system_index = system;
    diagnostic.solver_device_workspace_bytes =
        solver_device_workspace_bytes;
    diagnostic.solver_host_workspace_bytes = solver_host_workspace_bytes;
    diagnostic.absolute_threshold = relative_threshold * largest;
    double smallest_retained = largest;
    for (std::size_t item = 0; item < naux; ++item) {
      const double value = eigenvalues[offset + item];
      if (!std::isfinite(value)) {
        detail = "CUDA DF metric eigensolver returned a non-finite eigenvalue";
        return fail_plan(candidate, VIBEQC_STATUS_CUDA_ERROR);
      }
      if (value <= diagnostic.absolute_threshold) continue;
      scales[offset + item] = 1.0 / std::sqrt(value);
      ++diagnostic.effective_rank;
      smallest_retained = std::min(smallest_retained, value);
    }
    if (diagnostic.effective_rank == 0) {
      detail = "CUDA DF metric threshold removed every auxiliary direction";
      return fail_plan(candidate, VIBEQC_STATUS_INVALID_ARGUMENT);
    }
    diagnostic.condition_number = largest / smallest_retained;
  }

  cuda_error = cudaMemcpyAsync(setup.scales, scales.data(),
                               scales.size() * sizeof(double),
                               cudaMemcpyHostToDevice, candidate->stream);
  if (cuda_error != cudaSuccess) {
    return fail_plan(
        candidate,
        cuda_failure(cuda_error, "upload CUDA DF metric scales", detail));
  }
  scale_eigenvectors_kernel<<<blocks_for(all_metric_elements), kThreads, 0,
                              candidate->stream>>>(all_metric_elements, naux,
                                                   setup.metrics, setup.scales,
                                                   setup.scaled_eigenvectors);
  cuda_error = cudaPeekAtLastError();
  if (cuda_error != cudaSuccess) {
    return fail_plan(
        candidate,
        cuda_failure(cuda_error, "scale CUDA DF metric eigenvectors", detail));
  }

  const double one = 1.0;
  const double zero = 0.0;
  blas_status = cublasDgemmStridedBatched(
      candidate->blas, CUBLAS_OP_N, CUBLAS_OP_T, static_cast<int>(naux),
      static_cast<int>(naux), static_cast<int>(naux), &one,
      setup.scaled_eigenvectors, static_cast<int>(naux),
      static_cast<long long>(metric_elements), setup.metrics,
      static_cast<int>(naux), static_cast<long long>(metric_elements), &zero,
      setup.inverse_square_roots, static_cast<int>(naux),
      static_cast<long long>(metric_elements), static_cast<int>(batch_size));
  if (blas_status != CUBLAS_STATUS_SUCCESS) {
    return fail_plan(
        candidate,
        blas_failure(blas_status,
                     "construct CUDA DF metric inverse square root", detail));
  }
  if (candidate->streamed) {
    try {
      candidate->streamed_inverse_square_roots.resize(
          batch_size * metric_elements);
    } catch (const std::bad_alloc&) {
      detail = "host allocation for streamed CUDA DF metric inverse failed";
      return fail_plan(candidate, VIBEQC_STATUS_OUT_OF_MEMORY);
    }
    cuda_error = cudaMemcpyAsync(
        candidate->streamed_inverse_square_roots.data(),
        setup.inverse_square_roots,
        candidate->streamed_inverse_square_roots.size() * sizeof(double),
        cudaMemcpyDeviceToHost, candidate->stream);
    if (cuda_error != cudaSuccess) {
      return fail_plan(
          candidate,
          cuda_failure(cuda_error, "read streamed CUDA DF metric inverse",
                       detail));
    }
  } else {
    blas_status = cublasDgemmStridedBatched(
        candidate->blas, CUBLAS_OP_N, CUBLAS_OP_N, static_cast<int>(naux),
        static_cast<int>(matrix_elements), static_cast<int>(naux), &one,
        setup.inverse_square_roots, static_cast<int>(naux),
        static_cast<long long>(metric_elements), setup.raw_three_center,
        static_cast<int>(naux),
        static_cast<long long>(tensor_elements_per_system), &zero,
        candidate->three_center, static_cast<int>(naux),
        static_cast<long long>(tensor_elements_per_system),
        static_cast<int>(batch_size));
    if (blas_status != CUBLAS_STATUS_SUCCESS) {
      return fail_plan(
          candidate,
          blas_failure(blas_status, "transform CUDA DF three-center tensor",
                       detail));
    }
  }
  // Publish a conservative allocation accounting record.  Setup buffers are
  // still live at this point, so the peak includes both permanent contraction
  // storage and metric-factorization workspace; host-side solver workspace is
  // intentionally excluded from the device-byte figures.
  const std::size_t persistent_device_bytes =
      6 * matrix_bytes + auxiliary_bytes +
      (candidate->streamed ? 0 : tensor_bytes) + 3 * tile_bytes +
      matrix_bytes + (candidate->streamed ? tile_bytes : 0);
  const std::size_t setup_device_bytes =
      3 * metric_bytes + 2 * auxiliary_vector_bytes +
      (candidate->streamed ? 0 : tensor_bytes) +
      solver_device_workspace_bytes + solver_info_bytes;
  const std::size_t peak_device_bytes =
      persistent_device_bytes + setup_device_bytes;
  const std::size_t host_resident_bytes =
      candidate->streamed
          ? candidate->streamed_raw_three_center.size() * sizeof(double) +
                candidate->streamed_inverse_square_roots.size() * sizeof(double)
          : 0;
  const std::size_t host_peak_bytes =
      host_resident_bytes + metrics.size() * sizeof(double) +
      three_center.size() * sizeof(double) +
      eigenvalues.size() * sizeof(double) + scales.size() * sizeof(double) +
      solver_info.size() * sizeof(int) + solver_host_workspace_bytes;
  for (auto& diagnostic : diagnostics) {
    diagnostic.device_resident_bytes = persistent_device_bytes;
    diagnostic.peak_device_bytes = peak_device_bytes;
    diagnostic.host_resident_bytes = host_resident_bytes;
    diagnostic.peak_host_bytes = host_peak_bytes;
    diagnostic.auxiliary_tile = auxiliary_tile;
    diagnostic.streamed = candidate->streamed;
  }
  cuda_error = cudaStreamSynchronize(candidate->stream);
  if (cuda_error != cudaSuccess) {
    return fail_plan(
        candidate,
        cuda_failure(cuda_error, "finish CUDA DF plan preparation", detail));
  }

  (void)cusolverDnDestroyParams(candidate->solver_parameters);
  candidate->solver_parameters = nullptr;
  (void)cusolverDnDestroy(candidate->solver);
  candidate->solver = nullptr;
  *plan = candidate;
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status create_cuda_density_fitting_jk_plan(
    int device_id, std::size_t batch_size, std::size_t nbf, std::size_t naux,
    const std::vector<double>& metrics, const std::vector<double>& three_center,
    double relative_threshold, std::size_t auxiliary_tile,
    CudaDensityFittingJkPlan** plan,
    std::vector<CudaDensityFittingMetricDiagnostic>& diagnostics,
    std::string& detail) {
  // Preserve the original API's full-AO-pair staging semantics.  New planner
  // callers use the tiled entry point below to bound the streamed path.
  return create_cuda_density_fitting_jk_plan_tiled(
      device_id, batch_size, nbf, naux, metrics, three_center,
      relative_threshold, auxiliary_tile, 0, plan, diagnostics,
      detail);
}

vibeqc_status execute_cuda_density_fitting_rhf_jk(
    CudaDensityFittingJkPlan* plan, const std::vector<double>& density,
    std::vector<double>& coulomb, std::vector<double>& exchange,
    std::string& detail) {
  detail.clear();
  if (!validate_execution_input(plan, density, detail)) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t elements = plan->batch_size * plan->matrix_elements;
  vibeqc_status status = prepare_outputs(elements, coulomb, exchange, detail);
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  cudaError_t cuda_error = cudaSetDevice(plan->device_id);
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "select CUDA DF device", detail);
  }
  const std::size_t bytes = elements * sizeof(double);
  cuda_error = cudaMemcpyAsync(plan->primary_density, density.data(), bytes,
                               cudaMemcpyHostToDevice, plan->stream);
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "upload RHF CUDA DF density", detail);
  }
  status = build_coulomb(*plan, plan->primary_density, detail);
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = build_exchange(*plan, plan->primary_density, plan->alpha_exchange,
                            detail);
  }
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  cuda_error = cudaMemcpyAsync(coulomb.data(), plan->coulomb, bytes,
                               cudaMemcpyDeviceToHost, plan->stream);
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpyAsync(exchange.data(), plan->alpha_exchange, bytes,
                                 cudaMemcpyDeviceToHost, plan->stream);
  }
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaStreamSynchronize(plan->stream);
  }
  return cuda_error == cudaSuccess
             ? VIBEQC_STATUS_SUCCESS
             : cuda_failure(cuda_error, "finish RHF CUDA DF J/K", detail);
}

vibeqc_status execute_cuda_density_fitting_uhf_jk(
    CudaDensityFittingJkPlan* plan, const std::vector<double>& alpha_density,
    const std::vector<double>& beta_density, std::vector<double>& coulomb,
    std::vector<double>& alpha_exchange, std::vector<double>& beta_exchange,
    std::string& detail) {
  detail.clear();
  if (!validate_execution_input(plan, alpha_density, detail) ||
      !validate_execution_input(plan, beta_density, detail)) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t elements = plan->batch_size * plan->matrix_elements;
  vibeqc_status status =
      prepare_outputs(elements, coulomb, alpha_exchange, detail);
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  try {
    beta_exchange.assign(elements, 0.0);
  } catch (const std::bad_alloc&) {
    detail = "host allocation for CUDA DF beta exchange output failed";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
  cudaError_t cuda_error = cudaSetDevice(plan->device_id);
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "select CUDA DF device", detail);
  }
  const std::size_t bytes = elements * sizeof(double);
  cuda_error = cudaMemcpyAsync(plan->primary_density, alpha_density.data(),
                               bytes, cudaMemcpyHostToDevice, plan->stream);
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpyAsync(plan->secondary_density, beta_density.data(),
                                 bytes, cudaMemcpyHostToDevice, plan->stream);
  }
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "upload UHF CUDA DF densities", detail);
  }
  sum_spin_density_kernel<<<blocks_for(elements), kThreads, 0, plan->stream>>>(
      elements, plan->primary_density, plan->secondary_density,
      plan->total_density);
  cuda_error = cudaPeekAtLastError();
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "sum UHF CUDA DF density", detail);
  }
  status = build_coulomb(*plan, plan->total_density, detail);
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = build_exchange(*plan, plan->primary_density, plan->alpha_exchange,
                            detail);
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = build_exchange(*plan, plan->secondary_density, plan->beta_exchange,
                            detail);
  }
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  cuda_error = cudaMemcpyAsync(coulomb.data(), plan->coulomb, bytes,
                               cudaMemcpyDeviceToHost, plan->stream);
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpyAsync(alpha_exchange.data(), plan->alpha_exchange,
                                 bytes, cudaMemcpyDeviceToHost, plan->stream);
  }
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaMemcpyAsync(beta_exchange.data(), plan->beta_exchange,
                                 bytes, cudaMemcpyDeviceToHost, plan->stream);
  }
  if (cuda_error == cudaSuccess) {
    cuda_error = cudaStreamSynchronize(plan->stream);
  }
  return cuda_error == cudaSuccess
             ? VIBEQC_STATUS_SUCCESS
             : cuda_failure(cuda_error, "finish UHF CUDA DF J/K", detail);
}

vibeqc_status execute_cuda_density_fitting_rhf_jk_device(
    CudaDensityFittingJkPlan* plan, const double* density, double* coulomb,
    double* exchange, std::string& detail) {
  detail.clear();
  if (plan == nullptr || density == nullptr || coulomb == nullptr ||
      exchange == nullptr) {
    detail = "CUDA DF device RHF J/K pointers are invalid";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  cudaError_t cuda_error = cudaSetDevice(plan->device_id);
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "select CUDA DF device", detail);
  }
  vibeqc_status status = build_coulomb(*plan, density, detail);
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = build_exchange(*plan, density, plan->alpha_exchange, detail,
                            true);
  }
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  const std::size_t bytes = plan->batch_size * plan->matrix_elements *
                            sizeof(double);
  if (coulomb != plan->coulomb) {
    cuda_error = cudaMemcpyAsync(coulomb, plan->coulomb, bytes,
                                 cudaMemcpyDeviceToDevice, plan->stream);
  }
  if (cuda_error == cudaSuccess && exchange != plan->alpha_exchange) {
    cuda_error = cudaMemcpyAsync(exchange, plan->alpha_exchange, bytes,
                                 cudaMemcpyDeviceToDevice, plan->stream);
  }
  return cuda_error == cudaSuccess
             ? VIBEQC_STATUS_SUCCESS
             : cuda_failure(cuda_error, "copy CUDA DF device J/K outputs", detail);
}

vibeqc_status execute_cuda_density_fitting_uhf_jk_device(
    CudaDensityFittingJkPlan* plan, const double* alpha_density,
    const double* beta_density, double* coulomb, double* alpha_exchange,
    double* beta_exchange, std::string& detail) {
  detail.clear();
  if (plan == nullptr || alpha_density == nullptr || beta_density == nullptr ||
      coulomb == nullptr || alpha_exchange == nullptr || beta_exchange == nullptr) {
    detail = "CUDA DF device UHF J/K pointers are invalid";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  cudaError_t cuda_error = cudaSetDevice(plan->device_id);
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "select CUDA DF device", detail);
  }
  sum_spin_density_kernel<<<blocks_for(plan->batch_size * plan->matrix_elements),
                            kThreads, 0, plan->stream>>>(
      plan->batch_size * plan->matrix_elements, alpha_density, beta_density,
      plan->total_density);
  cuda_error = cudaPeekAtLastError();
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "sum CUDA DF device UHF density", detail);
  }
  vibeqc_status status = build_coulomb(*plan, plan->total_density, detail);
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = build_exchange(*plan, alpha_density, plan->alpha_exchange, detail,
                            true);
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = build_exchange(*plan, beta_density, plan->beta_exchange, detail,
                            true);
  }
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  const std::size_t bytes = plan->batch_size * plan->matrix_elements *
                            sizeof(double);
  if (coulomb != plan->coulomb) {
    cuda_error = cudaMemcpyAsync(coulomb, plan->coulomb, bytes,
                                 cudaMemcpyDeviceToDevice, plan->stream);
  }
  if (cuda_error == cudaSuccess && alpha_exchange != plan->alpha_exchange) {
    cuda_error = cudaMemcpyAsync(alpha_exchange, plan->alpha_exchange, bytes,
                                 cudaMemcpyDeviceToDevice, plan->stream);
  }
  if (cuda_error == cudaSuccess && beta_exchange != plan->beta_exchange) {
    cuda_error = cudaMemcpyAsync(beta_exchange, plan->beta_exchange, bytes,
                                 cudaMemcpyDeviceToDevice, plan->stream);
  }
  return cuda_error == cudaSuccess
             ? VIBEQC_STATUS_SUCCESS
             : cuda_failure(cuda_error, "copy CUDA DF device UHF J/K outputs",
                            detail);
}

namespace {

struct DeviceSolver {
  cusolverDnHandle_t handle{};
  syevjInfo_t jacobi{};
  cusolverDnParams_t parameters{};
  double* workspace{};
  void* host_workspace{};
  std::size_t workspace_bytes{};
  std::size_t host_workspace_bytes{};
  bool xsyev{};
  int lwork{};
  ~DeviceSolver() {
    (void)cudaFree(workspace);
    std::free(host_workspace);
    if (parameters != nullptr) (void)cusolverDnDestroyParams(parameters);
    if (jacobi != nullptr) (void)cusolverDnDestroySyevjInfo(jacobi);
    if (handle != nullptr) (void)cusolverDnDestroy(handle);
  }
};

/** Best-effort replay object for a fixed DF SCF iteration. */
struct DeviceIterationGraph {
  int device_id{-1};
  cudaStream_t stream{};
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  void reset() noexcept {
    if (device_id >= 0) (void)cudaSetDevice(device_id);
    if (executable != nullptr) {
      (void)cudaGraphExecDestroy(executable);
      executable = nullptr;
    }
    if (graph != nullptr) {
      (void)cudaGraphDestroy(graph);
      graph = nullptr;
    }
  }
  ~DeviceIterationGraph() {
    reset();
  }
};

/**
 * Device allocations and Graph executable retained by a prepared DF plan.
 * Inputs are refreshed at the start of each replay, while the fixed topology
 * and solver workspace stay resident until the plan is destroyed or rebuilt.
 */
struct PersistentScfState {
  int device_id{-1};
  bool unrestricted{};
  std::size_t batch_size{};
  std::size_t nbf{};
  std::size_t expected{};
  std::vector<void*> allocations;
  DeviceSolver solver;
  DeviceIterationGraph graph;
  bool graph_replay{};
  double energy_tolerance{};
  double density_tolerance{};

  // Shared RHF state.
  double* d_hcore{};
  double* d_orthogonalizer{};
  double* d_density{};
  double* d_next_density{};
  double* d_fock{};
  double* d_temporary{};
  double* d_eigenvalues{};
  std::int32_t* d_occupied{};
  double* d_nuclear{};
  double* d_energy{};
  double* d_previous_energy{};
  double* d_energy_change{};
  double* d_density_rms{};
  std::uint8_t* d_active{};
  std::uint8_t* d_converged{};
  std::uint32_t* d_iterations{};
  int* d_info{};

  // Additional UHF state.
  double* d_alpha_density{};
  double* d_beta_density{};
  double* d_next_alpha{};
  double* d_next_beta{};
  double* d_alpha_fock{};
  double* d_beta_fock{};
  double* d_alpha_eigenvalues{};
  double* d_beta_eigenvalues{};
  std::int32_t* d_alpha_occupied{};
  std::int32_t* d_beta_occupied{};
  int* d_alpha_info{};
  int* d_beta_info{};

  ~PersistentScfState() {
    if (device_id >= 0) (void)cudaSetDevice(device_id);
    for (void* pointer : allocations) (void)cudaFree(pointer);
  }
};

void destroy_persistent_scf_state(void*& opaque) noexcept {
  delete static_cast<PersistentScfState*>(opaque);
  opaque = nullptr;
}

/** Temporary device storage for one raw-tensor force-response contraction. */
struct ForceResponseScratch {
  int device_id{-1};
  std::vector<void*> pointers;
  ~ForceResponseScratch() {
    if (device_id >= 0) (void)cudaSetDevice(device_id);
    for (void* pointer : pointers) (void)cudaFree(pointer);
  }
};

vibeqc_status allocate_force_buffer(ForceResponseScratch& scratch,
                                    std::size_t bytes, void** pointer,
                                    const char* description,
                                    std::string& detail) {
  const vibeqc_status status =
      allocate_device(pointer, bytes, description, detail);
  if (status == VIBEQC_STATUS_SUCCESS) scratch.pointers.push_back(*pointer);
  return status;
}

/**
 * Contract one spin density's raw DF force response.  All arrays are already
 * resident on the device and the result is accumulated into `output`; the
 * caller controls the Coulomb and exchange coefficients (1/4 exchange for RHF,
 * or exchange-only spin passes for UHF).
 */
vibeqc_status launch_force_density_response(
    CudaDensityFittingJkPlan& plan, const double* raw,
    const double* derivative_raw, const double* inverse,
    const double* derivative_inverse, const double* density,
    double coulomb_coefficient, double exchange_coefficient, double* output,
    // Reusable scratch arrays.
    double* density_column_major, double* charge, double* derivative_charge,
    double* auxiliary_matrices, double* derivative_auxiliary_matrices,
    double* transformed, double* derivative_transformed, double* response,
    double* derivative_response, double* exchange_quadratic,
    double* derivative_exchange_quadratic, std::string& detail) {
  const std::size_t matrix_elements = plan.matrix_elements;
  const std::size_t naux = plan.naux;
  const std::size_t nbf = plan.nbf;
  const std::size_t tensor_elements = matrix_elements * naux;
  const int matrix = static_cast<int>(matrix_elements);
  const int auxiliary = static_cast<int>(naux);
  const int basis = static_cast<int>(nbf);
  const double one = 1.0;
  const double zero = 0.0;

  transpose_density_kernel<<<blocks_for(matrix_elements), kThreads, 0,
                             plan.stream>>>(nbf, density,
                                             density_column_major);
  cudaError_t cuda_error = cudaPeekAtLastError();
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "transpose CUDA DF force density", detail);
  }
  cublasStatus_t blas_status = cublasDgemm(
      plan.blas, CUBLAS_OP_N, CUBLAS_OP_N, auxiliary, 1, matrix, &one, raw,
      auxiliary, density, matrix, &zero, charge, auxiliary);
  if (blas_status == CUBLAS_STATUS_SUCCESS) {
    blas_status = cublasDgemm(
        plan.blas, CUBLAS_OP_N, CUBLAS_OP_N, auxiliary, 1, matrix, &one,
        derivative_raw, auxiliary, density, matrix, &zero,
        derivative_charge, auxiliary);
  }
  if (blas_status != CUBLAS_STATUS_SUCCESS) {
    return blas_failure(blas_status, "CUDA DF force charge contraction", detail);
  }

  gather_force_auxiliary_matrices_kernel<<<blocks_for(tensor_elements),
                                           kThreads, 0, plan.stream>>>(
      nbf, naux, raw, auxiliary_matrices);
  gather_force_auxiliary_matrices_kernel<<<blocks_for(tensor_elements),
                                           kThreads, 0, plan.stream>>>(
      nbf, naux, derivative_raw, derivative_auxiliary_matrices);
  cuda_error = cudaPeekAtLastError();
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "gather CUDA DF force tensors", detail);
  }

  blas_status = cublasDgemmStridedBatched(
      plan.blas, CUBLAS_OP_N, CUBLAS_OP_N, basis, basis, basis, &one,
      auxiliary_matrices, basis, static_cast<long long>(matrix_elements),
      density_column_major, basis, 0, &zero, transformed, basis,
      static_cast<long long>(matrix_elements), auxiliary);
  if (blas_status == CUBLAS_STATUS_SUCCESS) {
    blas_status = cublasDgemmStridedBatched(
        plan.blas, CUBLAS_OP_N, CUBLAS_OP_N, basis, basis, basis, &one,
        derivative_auxiliary_matrices, basis,
        static_cast<long long>(matrix_elements), density_column_major, basis,
        0, &zero, derivative_transformed, basis,
        static_cast<long long>(matrix_elements), auxiliary);
  }
  if (blas_status == CUBLAS_STATUS_SUCCESS) {
    blas_status = cublasDgemmStridedBatched(
        plan.blas, CUBLAS_OP_T, CUBLAS_OP_N, basis, basis, basis, &one,
        density_column_major, basis, 0, transformed, basis,
        static_cast<long long>(matrix_elements), &zero, response, basis,
        static_cast<long long>(matrix_elements), auxiliary);
  }
  if (blas_status == CUBLAS_STATUS_SUCCESS) {
    blas_status = cublasDgemmStridedBatched(
        plan.blas, CUBLAS_OP_T, CUBLAS_OP_N, basis, basis, basis, &one,
        density_column_major, basis, 0, derivative_transformed, basis,
        static_cast<long long>(matrix_elements), &zero, derivative_response,
        basis,
        static_cast<long long>(matrix_elements), auxiliary);
  }
  if (blas_status != CUBLAS_STATUS_SUCCESS) {
    return blas_failure(blas_status, "CUDA DF force exchange GEMM", detail);
  }

  blas_status = cublasDgemm(
      plan.blas, CUBLAS_OP_T, CUBLAS_OP_N, auxiliary, auxiliary, matrix,
      &one, response, matrix, auxiliary_matrices, matrix, &zero,
      exchange_quadratic, auxiliary);
  if (blas_status == CUBLAS_STATUS_SUCCESS) {
    blas_status = cublasDgemm(
        plan.blas, CUBLAS_OP_T, CUBLAS_OP_N, auxiliary, auxiliary, matrix,
        &one, derivative_response, matrix, auxiliary_matrices, matrix, &zero,
        derivative_exchange_quadratic, auxiliary);
  }
  if (blas_status == CUBLAS_STATUS_SUCCESS) {
    blas_status = cublasDgemm(
        plan.blas, CUBLAS_OP_T, CUBLAS_OP_N, auxiliary, auxiliary, matrix,
        &one, response, matrix, derivative_auxiliary_matrices, matrix, &one,
        derivative_exchange_quadratic, auxiliary);
  }
  if (blas_status != CUBLAS_STATUS_SUCCESS) {
    return blas_failure(blas_status, "CUDA DF force quadratic contraction",
                        detail);
  }
  cuda_error = cudaMemsetAsync(output, 0, sizeof(double), plan.stream);
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "zero CUDA DF force response", detail);
  }
  reduce_force_response_kernel<<<4, kThreads, 0, plan.stream>>>(
      naux, charge, derivative_charge, inverse, derivative_inverse,
      exchange_quadratic, derivative_exchange_quadratic, coulomb_coefficient,
      exchange_coefficient, output);
  cuda_error = cudaPeekAtLastError();
  return cuda_error == cudaSuccess
             ? VIBEQC_STATUS_SUCCESS
             : cuda_failure(cuda_error, "reduce CUDA DF force response", detail);
}

bool valid_force_vector(const std::vector<double>& values,
                        std::size_t expected) {
  return values.size() == expected && finite_values(values);
}

vibeqc_status execute_cuda_density_fitting_force_response_impl(
    CudaDensityFittingJkPlan* plan,
    const std::vector<double>& raw_three_center,
    const std::vector<double>& metric_inverse,
    const std::vector<double>& three_center_derivative,
    const std::vector<double>& metric_inverse_derivative,
    std::size_t coordinate_count, const std::vector<double>& density,
    double coulomb_coefficient, double exchange_coefficient,
    std::vector<double>& derivative, std::string& detail) {
  detail.clear();
  derivative.clear();
  if (plan == nullptr || coordinate_count == 0) {
    detail = "CUDA DF force-response plan or coordinate count is invalid";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t matrix_elements = plan->matrix_elements;
  const std::size_t tensor_elements = plan->tensor_elements_per_system;
  const std::size_t metric_elements = plan->naux * plan->naux;
  // A bucket finalizer may submit one item at a time while reusing the shared
  // homogeneous plan.  Accept both packed-batch inputs and a single-system
  // slice; the latter keeps the response device-resident without allocating a
  // second plan solely for force assembly.
  const std::size_t single_density_elements = matrix_elements;
  const bool single_item =
      raw_three_center.size() == tensor_elements &&
      metric_inverse.size() == metric_elements &&
      density.size() == single_density_elements;
  const std::size_t contraction_batch_size = single_item ? 1 : plan->batch_size;
  std::size_t batch_coordinates = 0;
  std::size_t expected_raw = 0;
  std::size_t expected_metric = 0;
  std::size_t expected_derivative_raw = 0;
  std::size_t expected_derivative_metric = 0;
  if (!checked_multiply(contraction_batch_size, coordinate_count,
                        batch_coordinates) ||
      !checked_multiply(contraction_batch_size, tensor_elements, expected_raw) ||
      !checked_multiply(contraction_batch_size, metric_elements, expected_metric) ||
      !checked_multiply(batch_coordinates, tensor_elements,
                        expected_derivative_raw) ||
      !checked_multiply(batch_coordinates, metric_elements,
                        expected_derivative_metric)) {
    detail = "CUDA DF force-response dimensions overflow size_t";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t expected_density =
      contraction_batch_size * matrix_elements;
  if (!valid_force_vector(raw_three_center, expected_raw) ||
      !valid_force_vector(metric_inverse, expected_metric) ||
      !valid_force_vector(three_center_derivative, expected_derivative_raw) ||
      !valid_force_vector(metric_inverse_derivative,
                          expected_derivative_metric) ||
      !valid_force_vector(density, expected_density) ||
      !(coulomb_coefficient >= 0.0) || !std::isfinite(coulomb_coefficient) ||
      !(exchange_coefficient >= 0.0) || !std::isfinite(exchange_coefficient)) {
    detail = "CUDA DF force-response buffers have invalid dimensions or values";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  try {
    derivative.assign(batch_coordinates, 0.0);
  } catch (const std::bad_alloc&) {
    detail = "host allocation for CUDA DF force response failed";
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
  cudaError_t cuda_error = cudaSetDevice(plan->device_id);
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "select CUDA DF force device", detail);
  }
  ForceResponseScratch scratch{plan->device_id};
  double* d_raw = nullptr;
  double* d_derivative_raw = nullptr;
  double* d_inverse = nullptr;
  double* d_derivative_inverse = nullptr;
  double* d_density = nullptr;
  double* d_density_column_major = nullptr;
  double* d_charge = nullptr;
  double* d_derivative_charge = nullptr;
  double* d_auxiliary_matrices = nullptr;
  double* d_derivative_auxiliary_matrices = nullptr;
  double* d_transformed = nullptr;
  double* d_derivative_transformed = nullptr;
  double* d_response = nullptr;
  double* d_derivative_response = nullptr;
  double* d_exchange_quadratic = nullptr;
  double* d_derivative_exchange_quadratic = nullptr;
  double* d_output = nullptr;
  auto allocate = [&](void** pointer, std::size_t bytes,
                      const char* description) {
    return allocate_force_buffer(scratch, bytes, pointer, description, detail);
  };
  vibeqc_status status = allocate(reinterpret_cast<void**>(&d_raw),
                                   tensor_elements * sizeof(double),
                                   "allocate CUDA DF force tensor");
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate(reinterpret_cast<void**>(&d_derivative_raw),
                      tensor_elements * sizeof(double),
                      "allocate CUDA DF force derivative tensor");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate(reinterpret_cast<void**>(&d_inverse),
                      metric_elements * sizeof(double),
                      "allocate CUDA DF force metric inverse");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate(reinterpret_cast<void**>(&d_derivative_inverse),
                      metric_elements * sizeof(double),
                      "allocate CUDA DF force metric response");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate(reinterpret_cast<void**>(&d_density),
                      matrix_elements * sizeof(double),
                      "allocate CUDA DF force density");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate(reinterpret_cast<void**>(&d_density_column_major),
                      matrix_elements * sizeof(double),
                      "allocate CUDA DF force density transpose");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate(reinterpret_cast<void**>(&d_charge),
                      plan->naux * sizeof(double),
                      "allocate CUDA DF force charge");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate(reinterpret_cast<void**>(&d_derivative_charge),
                      plan->naux * sizeof(double),
                      "allocate CUDA DF force derivative charge");
  }
  const std::size_t batched_tensor_bytes =
      tensor_elements * sizeof(double);
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate(reinterpret_cast<void**>(&d_auxiliary_matrices),
                      batched_tensor_bytes,
                      "allocate CUDA DF force auxiliary matrices");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate(reinterpret_cast<void**>(&d_derivative_auxiliary_matrices),
                      batched_tensor_bytes,
                      "allocate CUDA DF force derivative matrices");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate(reinterpret_cast<void**>(&d_transformed),
                      batched_tensor_bytes,
                      "allocate CUDA DF force exchange intermediates");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate(reinterpret_cast<void**>(&d_derivative_transformed),
                      batched_tensor_bytes,
                      "allocate CUDA DF force derivative intermediates");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate(reinterpret_cast<void**>(&d_response),
                      batched_tensor_bytes,
                      "allocate CUDA DF force response matrices");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate(reinterpret_cast<void**>(&d_derivative_response),
                      batched_tensor_bytes,
                      "allocate CUDA DF force derivative responses");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate(reinterpret_cast<void**>(&d_exchange_quadratic),
                      metric_elements * sizeof(double),
                      "allocate CUDA DF force exchange quadratic");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate(reinterpret_cast<void**>(&d_derivative_exchange_quadratic),
                      metric_elements * sizeof(double),
                      "allocate CUDA DF force derivative quadratic");
  }
  if (status == VIBEQC_STATUS_SUCCESS) {
    status = allocate(reinterpret_cast<void**>(&d_output), sizeof(double),
                      "allocate CUDA DF force output");
  }
  if (status != VIBEQC_STATUS_SUCCESS) return status;

  const std::size_t matrix_bytes = matrix_elements * sizeof(double);
  const std::size_t tensor_bytes = tensor_elements * sizeof(double);
  const std::size_t metric_bytes = metric_elements * sizeof(double);
  for (std::size_t system = 0; system < contraction_batch_size; ++system) {
    const double* raw = raw_three_center.data() + system * tensor_elements;
    const double* inverse = metric_inverse.data() + system * metric_elements;
    cuda_error = cudaMemcpyAsync(d_raw, raw, tensor_bytes,
                                 cudaMemcpyHostToDevice, plan->stream);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaMemcpyAsync(d_inverse, inverse, metric_bytes,
                                   cudaMemcpyHostToDevice, plan->stream);
    }
    if (cuda_error != cudaSuccess) {
      return cuda_failure(cuda_error, "upload CUDA DF force values", detail);
    }
    const double* system_density = density.data() + system * matrix_elements;
    cuda_error = cudaMemcpyAsync(d_density, system_density, matrix_bytes,
                                 cudaMemcpyHostToDevice, plan->stream);
    if (cuda_error != cudaSuccess) {
      return cuda_failure(cuda_error, "upload CUDA DF force density", detail);
    }
    for (std::size_t coordinate = 0; coordinate < coordinate_count;
         ++coordinate) {
      const std::size_t derivative_system_offset =
          (system * coordinate_count + coordinate);
      const double* derivative_raw =
          three_center_derivative.data() +
          derivative_system_offset * tensor_elements;
      const double* derivative_inverse =
          metric_inverse_derivative.data() +
          derivative_system_offset * metric_elements;
      cuda_error = cudaMemcpyAsync(d_derivative_raw, derivative_raw,
                                   tensor_bytes, cudaMemcpyHostToDevice,
                                   plan->stream);
      if (cuda_error == cudaSuccess) {
        cuda_error = cudaMemcpyAsync(d_derivative_inverse, derivative_inverse,
                                     metric_bytes, cudaMemcpyHostToDevice,
                                     plan->stream);
      }
      if (cuda_error != cudaSuccess) {
        return cuda_failure(cuda_error, "upload CUDA DF force derivatives",
                            detail);
      }
      status = launch_force_density_response(
          *plan, d_raw, d_derivative_raw, d_inverse, d_derivative_inverse,
          d_density, coulomb_coefficient, exchange_coefficient, d_output,
          d_density_column_major, d_charge, d_derivative_charge,
          d_auxiliary_matrices, d_derivative_auxiliary_matrices, d_transformed,
          d_derivative_transformed, d_response, d_derivative_response,
          d_exchange_quadratic, d_derivative_exchange_quadratic, detail);
      if (status != VIBEQC_STATUS_SUCCESS) return status;
      cuda_error = cudaMemcpyAsync(
          derivative.data() + derivative_system_offset, d_output,
          sizeof(double), cudaMemcpyDeviceToHost, plan->stream);
      if (cuda_error != cudaSuccess) {
        return cuda_failure(cuda_error, "read CUDA DF force response",
                            detail);
      }
    }
  }
  cuda_error = cudaStreamSynchronize(plan->stream);
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "finish CUDA DF force response", detail);
  }
  if (!finite_values(derivative)) {
    detail = "CUDA DF force response produced non-finite values";
    return VIBEQC_STATUS_NUMERICAL_FAILURE;
  }
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status execute_cuda_density_fitting_rhf_force_response_internal(
    CudaDensityFittingJkPlan* plan,
    const std::vector<double>& raw_three_center,
    const std::vector<double>& metric_inverse,
    const std::vector<double>& three_center_derivative,
    const std::vector<double>& metric_inverse_derivative,
    std::size_t coordinate_count, const std::vector<double>& density,
    std::vector<double>& derivative, std::string& detail) {
  return execute_cuda_density_fitting_force_response_impl(
      plan, raw_three_center, metric_inverse, three_center_derivative,
      metric_inverse_derivative, coordinate_count, density, 1.0, 0.25,
      derivative, detail);
}

vibeqc_status execute_cuda_density_fitting_uhf_force_response_internal(
    CudaDensityFittingJkPlan* plan,
    const std::vector<double>& raw_three_center,
    const std::vector<double>& metric_inverse,
    const std::vector<double>& three_center_derivative,
    const std::vector<double>& metric_inverse_derivative,
    std::size_t coordinate_count, const std::vector<double>& alpha_density,
    const std::vector<double>& beta_density, std::vector<double>& derivative,
    std::string& detail) {
  if (plan == nullptr || alpha_density.size() != beta_density.size()) {
    detail = "CUDA DF UHF force-response densities are invalid";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  std::vector<double> total_density(alpha_density.size(), 0.0);
  for (std::size_t item = 0; item < total_density.size(); ++item) {
    total_density[item] = alpha_density[item] + beta_density[item];
  }
  std::vector<double> coulomb;
  std::vector<double> alpha;
  std::vector<double> beta;
  vibeqc_status status = execute_cuda_density_fitting_force_response_impl(
      plan, raw_three_center, metric_inverse, three_center_derivative,
      metric_inverse_derivative, coordinate_count, total_density, 1.0, 0.0,
      coulomb, detail);
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  status = execute_cuda_density_fitting_force_response_impl(
      plan, raw_three_center, metric_inverse, three_center_derivative,
      metric_inverse_derivative, coordinate_count, alpha_density, 0.0, 0.5,
      alpha, detail);
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  status = execute_cuda_density_fitting_force_response_impl(
      plan, raw_three_center, metric_inverse, three_center_derivative,
      metric_inverse_derivative, coordinate_count, beta_density, 0.0, 0.5,
      beta, detail);
  if (status != VIBEQC_STATUS_SUCCESS) return status;
  derivative.resize(coulomb.size());
  for (std::size_t item = 0; item < derivative.size(); ++item) {
    // The spin passes request exchange-only response, so three contractions
    // directly form J(total) - K_a/2 - K_b/2.
    derivative[item] = coulomb[item] + alpha[item] + beta[item];
  }
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status scf_gemm(CudaDensityFittingJkPlan& plan, bool transpose_left,
                       std::size_t batch_size, std::size_t nbf,
                       const double* left, const double* right, double* output,
                       std::string& detail) {
  const double one = 1.0;
  const double zero = 0.0;
  const std::size_t matrix_elements = nbf * nbf;
  const cublasStatus_t status = cublasDgemmStridedBatched(
      plan.blas, transpose_left ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N,
      static_cast<int>(nbf), static_cast<int>(nbf), static_cast<int>(nbf),
      &one, left, static_cast<int>(nbf), static_cast<long long>(matrix_elements),
      right, static_cast<int>(nbf), static_cast<long long>(matrix_elements),
      &zero, output, static_cast<int>(nbf), static_cast<long long>(matrix_elements),
      static_cast<int>(batch_size));
  return status == CUBLAS_STATUS_SUCCESS
             ? VIBEQC_STATUS_SUCCESS
             : blas_failure(status, "CUDA DF device matrix product", detail);
}

vibeqc_status setup_device_solver(CudaDensityFittingJkPlan& plan,
                                  std::size_t nbf, std::size_t batch_size,
                                  double* eigensystem, double* eigenvalues,
                                  DeviceSolver& solver, std::string& detail) {
  cusolverStatus_t status = cusolverDnCreate(&solver.handle);
  if (status == CUSOLVER_STATUS_SUCCESS) {
    status = cusolverDnSetStream(solver.handle, plan.stream);
  }
  solver.xsyev = nbf > 32;
  if (status == CUSOLVER_STATUS_SUCCESS && !solver.xsyev) {
    status = cusolverDnCreateSyevjInfo(&solver.jacobi);
    if (status == CUSOLVER_STATUS_SUCCESS) {
      status = cusolverDnXsyevjSetTolerance(solver.jacobi, 1.0e-13);
    }
    if (status == CUSOLVER_STATUS_SUCCESS) {
      status = cusolverDnXsyevjSetMaxSweeps(solver.jacobi, 100);
    }
    if (status == CUSOLVER_STATUS_SUCCESS) {
      status = cusolverDnXsyevjSetSortEig(solver.jacobi, 1);
    }
  } else if (status == CUSOLVER_STATUS_SUCCESS) {
    status = cusolverDnCreateParams(&solver.parameters);
  }
  if (status != CUSOLVER_STATUS_SUCCESS) {
    return solver_failure(status, "initialize CUDA DF SCF eigensolver", detail);
  }
  if (!solver.xsyev) {
    status = cusolverDnDsyevjBatched_bufferSize(
        solver.handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        static_cast<int>(nbf), eigensystem, static_cast<int>(nbf), eigenvalues,
        &solver.lwork, solver.jacobi, static_cast<int>(batch_size));
    if (status != CUSOLVER_STATUS_SUCCESS || solver.lwork <= 0) {
      return solver_failure(status == CUSOLVER_STATUS_SUCCESS
                                ? CUSOLVER_STATUS_INTERNAL_ERROR
                                : status,
                            "size CUDA DF SCF eigensolver", detail);
    }
    return allocate_device(
        reinterpret_cast<void**>(&solver.workspace),
        static_cast<std::size_t>(solver.lwork) * sizeof(double),
        "allocate CUDA DF SCF eigensolver workspace", detail);
  }
  std::size_t device_bytes = 0;
  std::size_t host_bytes = 0;
  status = cusolverDnXsyevBatched_bufferSize(
      solver.handle, solver.parameters, CUSOLVER_EIG_MODE_VECTOR,
      CUBLAS_FILL_MODE_LOWER, static_cast<int>(nbf), CUDA_R_64F, eigensystem,
      static_cast<int>(nbf), CUDA_R_64F, eigenvalues, CUDA_R_64F,
      &device_bytes, &host_bytes, static_cast<int>(batch_size));
  if (status != CUSOLVER_STATUS_SUCCESS || device_bytes == 0) {
    return solver_failure(status == CUSOLVER_STATUS_SUCCESS
                              ? CUSOLVER_STATUS_INTERNAL_ERROR
                              : status,
                          "size CUDA DF SCF generic eigensolver", detail);
  }
  solver.workspace_bytes = device_bytes;
  solver.host_workspace_bytes = host_bytes;
  vibeqc_status allocation = allocate_device(
      reinterpret_cast<void**>(&solver.workspace), device_bytes,
      "allocate CUDA DF SCF generic eigensolver workspace", detail);
  if (allocation != VIBEQC_STATUS_SUCCESS) return allocation;
  if (host_bytes != 0) {
    solver.host_workspace = std::malloc(host_bytes);
    if (solver.host_workspace == nullptr) {
      detail = "host allocation for CUDA DF SCF generic eigensolver failed";
      return VIBEQC_STATUS_OUT_OF_MEMORY;
    }
  }
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status solve_device_batch(DeviceSolver& solver, std::size_t nbf,
                                 std::size_t batch_size, double* eigensystem,
                                 double* eigenvalues, int* info,
                                 std::string& detail) {
  cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
  if (!solver.xsyev) {
    status = cusolverDnDsyevjBatched(
        solver.handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
        static_cast<int>(nbf), eigensystem, static_cast<int>(nbf), eigenvalues,
        solver.workspace, solver.lwork, info, solver.jacobi,
        static_cast<int>(batch_size));
  } else {
    status = cusolverDnXsyevBatched(
        solver.handle, solver.parameters, CUSOLVER_EIG_MODE_VECTOR,
        CUBLAS_FILL_MODE_LOWER, static_cast<int>(nbf), CUDA_R_64F, eigensystem,
        static_cast<int>(nbf), CUDA_R_64F, eigenvalues, CUDA_R_64F,
        solver.workspace, solver.workspace_bytes, solver.host_workspace,
        solver.host_workspace_bytes, info, static_cast<int>(batch_size));
  }
  return status == CUSOLVER_STATUS_SUCCESS
             ? VIBEQC_STATUS_SUCCESS
             : solver_failure(status, "CUDA DF SCF eigensolve", detail);
}

}  // namespace

vibeqc_status execute_cuda_density_fitting_rhf_force_response(
    CudaDensityFittingJkPlan* plan,
    const std::vector<double>& raw_three_center,
    const std::vector<double>& metric_inverse,
    const std::vector<double>& three_center_derivative,
    const std::vector<double>& metric_inverse_derivative,
    std::size_t coordinate_count, const std::vector<double>& density,
    std::vector<double>& derivative, std::string& detail) {
  return execute_cuda_density_fitting_rhf_force_response_internal(
      plan, raw_three_center, metric_inverse, three_center_derivative,
      metric_inverse_derivative, coordinate_count, density, derivative,
      detail);
}

vibeqc_status execute_cuda_density_fitting_uhf_force_response(
    CudaDensityFittingJkPlan* plan,
    const std::vector<double>& raw_three_center,
    const std::vector<double>& metric_inverse,
    const std::vector<double>& three_center_derivative,
    const std::vector<double>& metric_inverse_derivative,
    std::size_t coordinate_count, const std::vector<double>& alpha_density,
    const std::vector<double>& beta_density, std::vector<double>& derivative,
    std::string& detail) {
  derivative.clear();
  return execute_cuda_density_fitting_uhf_force_response_internal(
      plan, raw_three_center, metric_inverse, three_center_derivative,
      metric_inverse_derivative, coordinate_count, alpha_density,
      beta_density, derivative, detail);
}

vibeqc_status run_cuda_density_fitting_rhf_device_scf(
    CudaDensityFittingJkPlan* plan, const std::vector<double>& hcore,
    const std::vector<double>& orthogonalizer,
    const std::vector<double>& initial_density,
    const std::vector<std::int32_t>& occupied,
    const std::vector<double>& nuclear_repulsion, unsigned max_iterations,
    double energy_tolerance, double density_tolerance,
    std::vector<double>& final_density,
    std::vector<CudaDensityFittingDeviceScfItem>& results,
    std::string& detail) {
  detail.clear();
  if (plan == nullptr || max_iterations == 0 || !(energy_tolerance > 0.0) ||
      !(density_tolerance > 0.0) || !std::isfinite(energy_tolerance) ||
      !std::isfinite(density_tolerance)) {
    detail = "CUDA DF device RHF SCF arguments are invalid";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t batch_size = plan->batch_size;
  const std::size_t matrix_elements = plan->matrix_elements;
  const std::size_t expected = batch_size * matrix_elements;
  if (hcore.size() != expected || orthogonalizer.size() != expected ||
      initial_density.size() != expected || occupied.size() != batch_size ||
      nuclear_repulsion.size() != batch_size ||
      !finite_values(hcore) || !finite_values(orthogonalizer) ||
      !finite_values(initial_density) || !finite_values(nuclear_repulsion)) {
    detail = "CUDA DF device RHF SCF buffers have invalid dimensions or values";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  for (std::int32_t value : occupied) {
    if (value < 0 || static_cast<std::size_t>(value) > plan->nbf) {
      detail = "CUDA DF device RHF occupation is invalid";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
  }
  final_density.clear();
  results.assign(batch_size, {});
  cudaError_t cuda_error = cudaSetDevice(plan->device_id);
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "select CUDA DF device", detail);
  }
  PersistentScfState* state =
      static_cast<PersistentScfState*>(plan->persistent_scf_state);
  const bool compatible =
      state != nullptr && !state->unrestricted &&
      state->device_id == plan->device_id && state->batch_size == batch_size &&
      state->nbf == plan->nbf && state->expected == expected;
  if (!compatible) {
    destroy_persistent_scf_state(plan->persistent_scf_state);
    state = new (std::nothrow) PersistentScfState{};
    if (state == nullptr) {
      detail = "host allocation for persistent CUDA DF RHF state failed";
      return VIBEQC_STATUS_OUT_OF_MEMORY;
    }
    state->device_id = plan->device_id;
    state->batch_size = batch_size;
    state->nbf = plan->nbf;
    state->expected = expected;
    state->graph.device_id = plan->device_id;
    state->graph.stream = plan->stream;
    auto allocate = [&](void** pointer, std::size_t bytes,
                        const char* description) {
      const vibeqc_status allocation =
          allocate_device(pointer, bytes, description, detail);
      if (allocation == VIBEQC_STATUS_SUCCESS) {
        state->allocations.push_back(*pointer);
      }
      return allocation;
    };
    vibeqc_status status = allocate(
        reinterpret_cast<void**>(&state->d_hcore), expected * sizeof(double),
        "allocate CUDA DF SCF Hcore");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_orthogonalizer),
                        expected * sizeof(double),
                        "allocate CUDA DF SCF orthogonalizer");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_density),
                        expected * sizeof(double), "allocate CUDA DF SCF density");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_next_density),
                        expected * sizeof(double),
                        "allocate CUDA DF SCF next density");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_fock),
                        expected * sizeof(double), "allocate CUDA DF SCF Fock");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_temporary),
                        expected * sizeof(double),
                        "allocate CUDA DF SCF eigensolver temporary");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_eigenvalues),
                        batch_size * plan->nbf * sizeof(double),
                        "allocate CUDA DF SCF eigenvalues");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_occupied),
                        batch_size * sizeof(std::int32_t),
                        "allocate CUDA DF SCF occupations");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_nuclear),
                        batch_size * sizeof(double),
                        "allocate CUDA DF SCF nuclear energies");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_energy),
                        batch_size * sizeof(double),
                        "allocate CUDA DF SCF energies");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_previous_energy),
                        batch_size * sizeof(double),
                        "allocate CUDA DF SCF previous energies");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_energy_change),
                        batch_size * sizeof(double),
                        "allocate CUDA DF SCF energy changes");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_density_rms),
                        batch_size * sizeof(double),
                        "allocate CUDA DF SCF density RMS");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_active),
                        batch_size * sizeof(std::uint8_t),
                        "allocate CUDA DF SCF active mask");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_converged),
                        batch_size * sizeof(std::uint8_t),
                        "allocate CUDA DF SCF converged mask");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_iterations),
                        batch_size * sizeof(std::uint32_t),
                        "allocate CUDA DF SCF iteration counters");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_info),
                        batch_size * sizeof(int),
                        "allocate CUDA DF SCF solver status");
    if (status != VIBEQC_STATUS_SUCCESS) {
      delete state;
      return status;
    }
    status = setup_device_solver(*plan, plan->nbf, batch_size,
                                 state->d_temporary, state->d_eigenvalues,
                                 state->solver, detail);
    if (status != VIBEQC_STATUS_SUCCESS) {
      delete state;
      return status;
    }
    plan->persistent_scf_state = state;
  }
  double* d_hcore = state->d_hcore;
  double* d_orthogonalizer = state->d_orthogonalizer;
  double* d_density = state->d_density;
  double* d_next_density = state->d_next_density;
  double* d_fock = state->d_fock;
  double* d_temporary = state->d_temporary;
  double* d_eigenvalues = state->d_eigenvalues;
  std::int32_t* d_occupied = state->d_occupied;
  double* d_nuclear = state->d_nuclear;
  double* d_energy = state->d_energy;
  double* d_previous_energy = state->d_previous_energy;
  double* d_energy_change = state->d_energy_change;
  double* d_density_rms = state->d_density_rms;
  std::uint8_t* d_active = state->d_active;
  std::uint8_t* d_converged = state->d_converged;
  std::uint32_t* d_iterations = state->d_iterations;
  int* d_info = state->d_info;
  vibeqc_status status = VIBEQC_STATUS_SUCCESS;
  const std::size_t matrix_bytes = expected * sizeof(double);
  cuda_error = cudaMemcpyAsync(d_hcore, hcore.data(), matrix_bytes, cudaMemcpyHostToDevice, plan->stream);
  if (cuda_error == cudaSuccess) cuda_error = cudaMemcpyAsync(d_orthogonalizer, orthogonalizer.data(), matrix_bytes, cudaMemcpyHostToDevice, plan->stream);
  if (cuda_error == cudaSuccess) cuda_error = cudaMemcpyAsync(d_density, initial_density.data(), matrix_bytes, cudaMemcpyHostToDevice, plan->stream);
  if (cuda_error == cudaSuccess) cuda_error = cudaMemcpyAsync(d_occupied, occupied.data(), batch_size * sizeof(std::int32_t), cudaMemcpyHostToDevice, plan->stream);
  if (cuda_error == cudaSuccess) cuda_error = cudaMemcpyAsync(d_nuclear, nuclear_repulsion.data(), batch_size * sizeof(double), cudaMemcpyHostToDevice, plan->stream);
  if (cuda_error == cudaSuccess) cuda_error = cudaMemsetAsync(d_converged, 0, batch_size * sizeof(std::uint8_t), plan->stream);
  if (cuda_error == cudaSuccess) cuda_error = cudaMemsetAsync(d_iterations, 0, batch_size * sizeof(std::uint32_t), plan->stream);
  if (cuda_error == cudaSuccess) cuda_error = cudaMemsetAsync(
      d_active, 1, batch_size * sizeof(std::uint8_t), plan->stream);
  if (cuda_error != cudaSuccess) return cuda_failure(cuda_error, "upload CUDA DF device RHF SCF state", detail);
  std::vector<double> initial_previous(batch_size, std::numeric_limits<double>::infinity());
  cuda_error = cudaMemcpyAsync(d_previous_energy, initial_previous.data(), batch_size * sizeof(double), cudaMemcpyHostToDevice, plan->stream);
  if (cuda_error != cudaSuccess) return cuda_failure(cuda_error, "initialize CUDA DF device RHF energy state", detail);
  std::vector<double> host_energy(batch_size), host_energy_change(batch_size),
      host_density_rms(batch_size);
  std::vector<std::uint8_t> host_converged(batch_size);
  std::vector<std::uint32_t> host_iterations(batch_size);
  std::vector<int> host_info(batch_size);
  const bool options_changed =
      state->energy_tolerance != energy_tolerance ||
      state->density_tolerance != density_tolerance;
  if (options_changed) {
    state->graph.reset();
    state->graph.device_id = plan->device_id;
    state->graph.stream = plan->stream;
    state->graph_replay = false;
  }
  state->energy_tolerance = energy_tolerance;
  state->density_tolerance = density_tolerance;
  DeviceIterationGraph& iteration_graph = state->graph;
  const auto launch_iteration = [&]() -> vibeqc_status {
    vibeqc_status iteration_status = execute_cuda_density_fitting_rhf_jk_device(
        plan, d_density, plan->coulomb, plan->alpha_exchange, detail);
    if (iteration_status != VIBEQC_STATUS_SUCCESS) return iteration_status;
    assemble_rhf_fock_kernel<<<blocks_for(expected), kThreads, 0,
                                plan->stream>>>(
        expected, d_hcore, plan->coulomb, plan->alpha_exchange, d_fock);
    cudaError_t iteration_error = cudaPeekAtLastError();
    if (iteration_error != cudaSuccess) {
      return cuda_failure(iteration_error,
                          "assemble CUDA DF device RHF Fock", detail);
    }
    compute_device_energy_kernel<<<static_cast<unsigned>(batch_size), 32, 0,
                                   plan->stream>>>(
        batch_size, plan->nbf, d_density, d_hcore, d_fock, d_nuclear,
        d_energy);
    iteration_status = scf_gemm(*plan, false, batch_size, plan->nbf, d_fock,
                                d_orthogonalizer, d_temporary, detail);
    if (iteration_status == VIBEQC_STATUS_SUCCESS) {
      iteration_status = scf_gemm(
          *plan, true, batch_size, plan->nbf, d_orthogonalizer, d_temporary,
          d_fock, detail);
    }
    if (iteration_status != VIBEQC_STATUS_SUCCESS) return iteration_status;
    iteration_status = solve_device_batch(state->solver, plan->nbf, batch_size,
                                          d_fock, d_eigenvalues, d_info,
                                          detail);
    if (iteration_status != VIBEQC_STATUS_SUCCESS) return iteration_status;
    iteration_status = scf_gemm(*plan, false, batch_size, plan->nbf,
                                d_orthogonalizer, d_fock, d_temporary,
                                detail);
    if (iteration_status != VIBEQC_STATUS_SUCCESS) return iteration_status;
    build_device_density_kernel<<<blocks_for(expected), kThreads, 0,
                                  plan->stream>>>(
        batch_size, plan->nbf, d_occupied, d_temporary, 2.0,
        d_next_density);
    update_device_convergence_kernel<<<static_cast<unsigned>(batch_size), 32,
                                       0, plan->stream>>>(
        batch_size, plan->nbf, energy_tolerance, density_tolerance, d_energy,
        d_previous_energy, d_next_density, d_density, d_active, d_converged,
        d_iterations, d_energy_change, d_density_rms);
    iteration_error = cudaPeekAtLastError();
    return iteration_error == cudaSuccess
               ? VIBEQC_STATUS_SUCCESS
               : cuda_failure(iteration_error,
                              "advance CUDA DF device RHF SCF", detail);
  };
  bool graph_replay = state->graph_replay;
  // The streamed J/K path stages host tiles and fences between tiles, which is
  // intentionally outside Graph capture.  Fixed transformed tensors and the
  // Jacobi provider are Graph-safe and use one captured iteration replay.
  if (!graph_replay && !plan->streamed && !state->solver.xsyev) {
    cuda_error = cudaStreamSynchronize(plan->stream);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaStreamBeginCapture(
          plan->stream, cudaStreamCaptureModeThreadLocal);
    }
    if (cuda_error == cudaSuccess) {
      status = launch_iteration();
      cudaGraph_t captured = nullptr;
      const cudaError_t end_error =
          cudaStreamEndCapture(plan->stream, &captured);
      if (status == VIBEQC_STATUS_SUCCESS && end_error == cudaSuccess &&
          captured != nullptr) {
        iteration_graph.graph = captured;
        cuda_error = cudaGraphInstantiate(
            &iteration_graph.executable, iteration_graph.graph, 0U);
        if (cuda_error == cudaSuccess) {
          cuda_error = cudaGraphUpload(iteration_graph.executable,
                                       plan->stream);
        }
        if (cuda_error == cudaSuccess) {
          cuda_error = cudaStreamSynchronize(plan->stream);
          graph_replay = cuda_error == cudaSuccess;
          state->graph_replay = graph_replay;
        }
      } else {
        if (captured != nullptr) (void)cudaGraphDestroy(captured);
        cuda_error = end_error;
      }
    }
    // Graph capture is an optimization.  A provider/capture limitation falls
    // through to the same direct launch sequence without changing semantics.
    if (!graph_replay) {
      cuda_error = cudaSuccess;
    }
  }
  bool all_converged = false;
  for (unsigned iteration = 0; iteration < max_iterations && !all_converged;
       ++iteration) {
    if (graph_replay) {
      cuda_error = cudaGraphLaunch(iteration_graph.executable, plan->stream);
      if (cuda_error != cudaSuccess) {
        return cuda_failure(cuda_error, "replay CUDA DF RHF SCF Graph", detail);
      }
    } else {
      status = launch_iteration();
      if (status != VIBEQC_STATUS_SUCCESS) return status;
    }
    cuda_error = cudaMemcpyAsync(host_energy.data(), d_energy,
                                 batch_size * sizeof(double), cudaMemcpyDeviceToHost, plan->stream);
    if (cuda_error == cudaSuccess) cuda_error = cudaMemcpyAsync(host_energy_change.data(), d_energy_change, batch_size * sizeof(double), cudaMemcpyDeviceToHost, plan->stream);
    if (cuda_error == cudaSuccess) cuda_error = cudaMemcpyAsync(host_density_rms.data(), d_density_rms, batch_size * sizeof(double), cudaMemcpyDeviceToHost, plan->stream);
    if (cuda_error == cudaSuccess) cuda_error = cudaMemcpyAsync(host_converged.data(), d_converged, batch_size * sizeof(std::uint8_t), cudaMemcpyDeviceToHost, plan->stream);
    if (cuda_error == cudaSuccess) cuda_error = cudaMemcpyAsync(host_iterations.data(), d_iterations, batch_size * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, plan->stream);
    if (cuda_error == cudaSuccess) cuda_error = cudaMemcpyAsync(host_info.data(), d_info, batch_size * sizeof(int), cudaMemcpyDeviceToHost, plan->stream);
    if (cuda_error == cudaSuccess) cuda_error = cudaStreamSynchronize(plan->stream);
    if (cuda_error != cudaSuccess) return cuda_failure(cuda_error, "read CUDA DF device RHF SCF records", detail);
    if (std::any_of(host_info.begin(), host_info.end(), [](int value) {
          return value != 0;
        })) {
      detail = "CUDA DF device RHF eigensolver did not converge";
      return VIBEQC_STATUS_CUDA_ERROR;
    }
    all_converged = std::all_of(host_converged.begin(), host_converged.end(), [](std::uint8_t value) { return value != 0; });
  }
  final_density.resize(expected);
  cuda_error = cudaMemcpyAsync(final_density.data(), d_density, matrix_bytes,
                               cudaMemcpyDeviceToHost, plan->stream);
  if (cuda_error == cudaSuccess) cuda_error = cudaStreamSynchronize(plan->stream);
  if (cuda_error != cudaSuccess) return cuda_failure(cuda_error, "read CUDA DF device RHF density", detail);
  if (!finite_values(final_density)) {
    detail = "CUDA DF device RHF SCF produced non-finite density";
    return VIBEQC_STATUS_NUMERICAL_FAILURE;
  }
  for (std::size_t system = 0; system < batch_size; ++system) {
    auto& result = results[system];
    result.status = VIBEQC_STATUS_SUCCESS;
    result.converged = host_converged[system] != 0;
    result.iterations = host_iterations[system];
    result.energy = host_energy[system];
    result.energy_change = host_energy_change[system];
    result.density_rms = host_density_rms[system];
    if (!result.converged) result.status = VIBEQC_STATUS_SCF_NOT_CONVERGED;
  }
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status run_cuda_density_fitting_uhf_device_scf(
    CudaDensityFittingJkPlan* plan, const std::vector<double>& hcore,
    const std::vector<double>& orthogonalizer,
    const std::vector<double>& initial_alpha_density,
    const std::vector<double>& initial_beta_density,
    const std::vector<std::int32_t>& alpha_occupied,
    const std::vector<std::int32_t>& beta_occupied,
    const std::vector<double>& nuclear_repulsion, unsigned max_iterations,
    double energy_tolerance, double density_tolerance,
    std::vector<double>& final_alpha_density,
    std::vector<double>& final_beta_density,
    std::vector<CudaDensityFittingDeviceScfItem>& results,
    std::string& detail) {
  detail.clear();
  if (plan == nullptr || max_iterations == 0 || !(energy_tolerance > 0.0) ||
      !(density_tolerance > 0.0) || !std::isfinite(energy_tolerance) ||
      !std::isfinite(density_tolerance)) {
    detail = "CUDA DF device UHF SCF arguments are invalid";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t batch_size = plan->batch_size;
  const std::size_t matrix_elements = plan->matrix_elements;
  const std::size_t expected = batch_size * matrix_elements;
  if (hcore.size() != expected || orthogonalizer.size() != expected ||
      initial_alpha_density.size() != expected ||
      initial_beta_density.size() != expected ||
      alpha_occupied.size() != batch_size || beta_occupied.size() != batch_size ||
      nuclear_repulsion.size() != batch_size || !finite_values(hcore) ||
      !finite_values(orthogonalizer) || !finite_values(initial_alpha_density) ||
      !finite_values(initial_beta_density) || !finite_values(nuclear_repulsion)) {
    detail = "CUDA DF device UHF SCF buffers have invalid dimensions or values";
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  for (std::size_t system = 0; system < batch_size; ++system) {
    if (alpha_occupied[system] < 0 || beta_occupied[system] < 0 ||
        static_cast<std::size_t>(alpha_occupied[system]) > plan->nbf ||
        static_cast<std::size_t>(beta_occupied[system]) > plan->nbf) {
      detail = "CUDA DF device UHF occupation is invalid";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
  }
  final_alpha_density.clear();
  final_beta_density.clear();
  results.assign(batch_size, {});
  cudaError_t cuda_error = cudaSetDevice(plan->device_id);
  if (cuda_error != cudaSuccess) {
    return cuda_failure(cuda_error, "select CUDA DF device", detail);
  }
  PersistentScfState* state =
      static_cast<PersistentScfState*>(plan->persistent_scf_state);
  const bool compatible =
      state != nullptr && state->unrestricted &&
      state->device_id == plan->device_id && state->batch_size == batch_size &&
      state->nbf == plan->nbf && state->expected == expected;
  if (!compatible) {
    destroy_persistent_scf_state(plan->persistent_scf_state);
    state = new (std::nothrow) PersistentScfState{};
    if (state == nullptr) {
      detail = "host allocation for persistent CUDA DF UHF state failed";
      return VIBEQC_STATUS_OUT_OF_MEMORY;
    }
    state->device_id = plan->device_id;
    state->unrestricted = true;
    state->batch_size = batch_size;
    state->nbf = plan->nbf;
    state->expected = expected;
    state->graph.device_id = plan->device_id;
    state->graph.stream = plan->stream;
    auto allocate = [&](void** pointer, std::size_t bytes,
                        const char* description) {
      const vibeqc_status allocation =
          allocate_device(pointer, bytes, description, detail);
      if (allocation == VIBEQC_STATUS_SUCCESS) {
        state->allocations.push_back(*pointer);
      }
      return allocation;
    };
    vibeqc_status status = allocate(
        reinterpret_cast<void**>(&state->d_hcore), expected * sizeof(double),
        "allocate CUDA DF SCF UHF Hcore");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_orthogonalizer),
                        expected * sizeof(double),
                        "allocate CUDA DF SCF UHF orthogonalizer");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_alpha_density),
                        expected * sizeof(double),
                        "allocate CUDA DF SCF alpha density");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_beta_density),
                        expected * sizeof(double),
                        "allocate CUDA DF SCF beta density");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_next_alpha),
                        expected * sizeof(double),
                        "allocate CUDA DF SCF next alpha density");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_next_beta),
                        expected * sizeof(double),
                        "allocate CUDA DF SCF next beta density");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_alpha_fock),
                        expected * sizeof(double),
                        "allocate CUDA DF SCF alpha Fock");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_beta_fock),
                        expected * sizeof(double),
                        "allocate CUDA DF SCF beta Fock");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_temporary),
                        expected * sizeof(double),
                        "allocate CUDA DF SCF UHF eigensolver temporary");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_alpha_eigenvalues),
                        batch_size * plan->nbf * sizeof(double),
                        "allocate CUDA DF SCF alpha eigenvalues");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_beta_eigenvalues),
                        batch_size * plan->nbf * sizeof(double),
                        "allocate CUDA DF SCF beta eigenvalues");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_alpha_occupied),
                        batch_size * sizeof(std::int32_t),
                        "allocate CUDA DF SCF alpha occupations");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_beta_occupied),
                        batch_size * sizeof(std::int32_t),
                        "allocate CUDA DF SCF beta occupations");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_nuclear),
                        batch_size * sizeof(double),
                        "allocate CUDA DF SCF UHF nuclear energies");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_energy),
                        batch_size * sizeof(double),
                        "allocate CUDA DF SCF UHF energies");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_previous_energy),
                        batch_size * sizeof(double),
                        "allocate CUDA DF SCF UHF previous energies");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_energy_change),
                        batch_size * sizeof(double),
                        "allocate CUDA DF SCF UHF energy changes");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_density_rms),
                        batch_size * sizeof(double),
                        "allocate CUDA DF SCF UHF density RMS");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_active),
                        batch_size * sizeof(std::uint8_t),
                        "allocate CUDA DF SCF UHF active mask");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_converged),
                        batch_size * sizeof(std::uint8_t),
                        "allocate CUDA DF SCF UHF converged mask");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_iterations),
                        batch_size * sizeof(std::uint32_t),
                        "allocate CUDA DF SCF UHF iteration counters");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_alpha_info),
                        batch_size * sizeof(int),
                        "allocate CUDA DF SCF alpha solver status");
    if (status == VIBEQC_STATUS_SUCCESS)
      status = allocate(reinterpret_cast<void**>(&state->d_beta_info),
                        batch_size * sizeof(int),
                        "allocate CUDA DF SCF beta solver status");
    if (status != VIBEQC_STATUS_SUCCESS) {
      delete state;
      return status;
    }
    status = setup_device_solver(*plan, plan->nbf, batch_size,
                                 state->d_temporary,
                                 state->d_alpha_eigenvalues, state->solver,
                                 detail);
    if (status != VIBEQC_STATUS_SUCCESS) {
      delete state;
      return status;
    }
    plan->persistent_scf_state = state;
  }
  double* d_hcore = state->d_hcore;
  double* d_orthogonalizer = state->d_orthogonalizer;
  double* d_alpha_density = state->d_alpha_density;
  double* d_beta_density = state->d_beta_density;
  double* d_next_alpha = state->d_next_alpha;
  double* d_next_beta = state->d_next_beta;
  double* d_alpha_fock = state->d_alpha_fock;
  double* d_beta_fock = state->d_beta_fock;
  double* d_temporary = state->d_temporary;
  double* d_alpha_eigenvalues = state->d_alpha_eigenvalues;
  double* d_beta_eigenvalues = state->d_beta_eigenvalues;
  std::int32_t* d_alpha_occupied = state->d_alpha_occupied;
  std::int32_t* d_beta_occupied = state->d_beta_occupied;
  double* d_nuclear = state->d_nuclear;
  double* d_energy = state->d_energy;
  double* d_previous_energy = state->d_previous_energy;
  double* d_energy_change = state->d_energy_change;
  double* d_density_rms = state->d_density_rms;
  std::uint8_t* d_active = state->d_active;
  std::uint8_t* d_converged = state->d_converged;
  std::uint32_t* d_iterations = state->d_iterations;
  int* d_alpha_info = state->d_alpha_info;
  int* d_beta_info = state->d_beta_info;
  vibeqc_status status = VIBEQC_STATUS_SUCCESS;
  const std::size_t matrix_bytes = expected * sizeof(double);
  cuda_error = cudaMemcpyAsync(d_hcore, hcore.data(), matrix_bytes, cudaMemcpyHostToDevice, plan->stream);
  if (cuda_error == cudaSuccess) cuda_error = cudaMemcpyAsync(d_orthogonalizer, orthogonalizer.data(), matrix_bytes, cudaMemcpyHostToDevice, plan->stream);
  if (cuda_error == cudaSuccess) cuda_error = cudaMemcpyAsync(d_alpha_density, initial_alpha_density.data(), matrix_bytes, cudaMemcpyHostToDevice, plan->stream);
  if (cuda_error == cudaSuccess) cuda_error = cudaMemcpyAsync(d_beta_density, initial_beta_density.data(), matrix_bytes, cudaMemcpyHostToDevice, plan->stream);
  if (cuda_error == cudaSuccess) cuda_error = cudaMemcpyAsync(d_alpha_occupied, alpha_occupied.data(), batch_size * sizeof(std::int32_t), cudaMemcpyHostToDevice, plan->stream);
  if (cuda_error == cudaSuccess) cuda_error = cudaMemcpyAsync(d_beta_occupied, beta_occupied.data(), batch_size * sizeof(std::int32_t), cudaMemcpyHostToDevice, plan->stream);
  if (cuda_error == cudaSuccess) cuda_error = cudaMemcpyAsync(d_nuclear, nuclear_repulsion.data(), batch_size * sizeof(double), cudaMemcpyHostToDevice, plan->stream);
  if (cuda_error == cudaSuccess) cuda_error = cudaMemsetAsync(d_converged, 0, batch_size * sizeof(std::uint8_t), plan->stream);
  if (cuda_error == cudaSuccess) cuda_error = cudaMemsetAsync(d_iterations, 0, batch_size * sizeof(std::uint32_t), plan->stream);
  if (cuda_error == cudaSuccess) cuda_error = cudaMemsetAsync(d_active, 1, batch_size * sizeof(std::uint8_t), plan->stream);
  if (cuda_error != cudaSuccess) return cuda_failure(cuda_error, "upload CUDA DF device UHF SCF state", detail);
  std::vector<double> initial_previous(batch_size, std::numeric_limits<double>::infinity());
  cuda_error = cudaMemcpyAsync(d_previous_energy, initial_previous.data(), batch_size * sizeof(double), cudaMemcpyHostToDevice, plan->stream);
  if (cuda_error != cudaSuccess) return cuda_failure(cuda_error, "initialize CUDA DF device UHF energy state", detail);
  std::vector<double> host_energy(batch_size), host_energy_change(batch_size), host_density_rms(batch_size);
  std::vector<std::uint8_t> host_converged(batch_size);
  std::vector<std::uint32_t> host_iterations(batch_size);
  std::vector<int> host_alpha_info(batch_size), host_beta_info(batch_size);
  const bool options_changed =
      state->energy_tolerance != energy_tolerance ||
      state->density_tolerance != density_tolerance;
  if (options_changed) {
    state->graph.reset();
    state->graph.device_id = plan->device_id;
    state->graph.stream = plan->stream;
    state->graph_replay = false;
  }
  state->energy_tolerance = energy_tolerance;
  state->density_tolerance = density_tolerance;
  DeviceIterationGraph& iteration_graph = state->graph;
  const auto launch_iteration = [&]() -> vibeqc_status {
    vibeqc_status iteration_status = execute_cuda_density_fitting_uhf_jk_device(
        plan, d_alpha_density, d_beta_density, plan->coulomb,
        plan->alpha_exchange, plan->beta_exchange, detail);
    if (iteration_status != VIBEQC_STATUS_SUCCESS) return iteration_status;
    assemble_uhf_fock_kernel<<<blocks_for(expected), kThreads, 0,
                                plan->stream>>>(
        expected, d_hcore, plan->coulomb, plan->alpha_exchange,
        plan->beta_exchange, d_alpha_fock, d_beta_fock);
    cudaError_t iteration_error = cudaPeekAtLastError();
    if (iteration_error != cudaSuccess) {
      return cuda_failure(iteration_error,
                          "assemble CUDA DF device UHF Fock", detail);
    }
    compute_device_uhf_energy_kernel<<<static_cast<unsigned>(batch_size), 32,
                                       0, plan->stream>>>(
        batch_size, plan->nbf, d_alpha_density, d_beta_density, d_hcore,
        d_alpha_fock, d_beta_fock, d_nuclear, d_energy);

    iteration_status = scf_gemm(*plan, false, batch_size, plan->nbf,
                                d_alpha_fock, d_orthogonalizer, d_temporary,
                                detail);
    if (iteration_status == VIBEQC_STATUS_SUCCESS) {
      iteration_status = scf_gemm(
          *plan, true, batch_size, plan->nbf, d_orthogonalizer, d_temporary,
          d_alpha_fock, detail);
    }
    if (iteration_status == VIBEQC_STATUS_SUCCESS) {
      iteration_status = solve_device_batch(
          state->solver, plan->nbf, batch_size, d_alpha_fock,
          d_alpha_eigenvalues,
          d_alpha_info, detail);
    }
    if (iteration_status == VIBEQC_STATUS_SUCCESS) {
      iteration_status = scf_gemm(
          *plan, false, batch_size, plan->nbf, d_orthogonalizer, d_alpha_fock,
          d_temporary, detail);
    }
    if (iteration_status == VIBEQC_STATUS_SUCCESS) {
      build_device_density_kernel<<<blocks_for(expected), kThreads, 0,
                                    plan->stream>>>(
          batch_size, plan->nbf, d_alpha_occupied, d_temporary, 1.0,
          d_next_alpha);
    }
    if (iteration_status != VIBEQC_STATUS_SUCCESS) return iteration_status;

    iteration_status = scf_gemm(
        *plan, false, batch_size, plan->nbf, d_beta_fock, d_orthogonalizer,
        d_temporary, detail);
    if (iteration_status == VIBEQC_STATUS_SUCCESS) {
      iteration_status = scf_gemm(
          *plan, true, batch_size, plan->nbf, d_orthogonalizer, d_temporary,
          d_beta_fock, detail);
    }
    if (iteration_status == VIBEQC_STATUS_SUCCESS) {
      iteration_status = solve_device_batch(
          state->solver, plan->nbf, batch_size, d_beta_fock,
          d_beta_eigenvalues,
          d_beta_info, detail);
    }
    if (iteration_status == VIBEQC_STATUS_SUCCESS) {
      iteration_status = scf_gemm(
          *plan, false, batch_size, plan->nbf, d_orthogonalizer, d_beta_fock,
          d_temporary, detail);
    }
    if (iteration_status == VIBEQC_STATUS_SUCCESS) {
      build_device_density_kernel<<<blocks_for(expected), kThreads, 0,
                                    plan->stream>>>(
          batch_size, plan->nbf, d_beta_occupied, d_temporary, 1.0,
          d_next_beta);
      update_device_uhf_convergence_kernel<<<static_cast<unsigned>(batch_size),
                                             32, 0, plan->stream>>>(
          batch_size, plan->nbf, energy_tolerance, density_tolerance, d_energy,
          d_previous_energy, d_next_alpha, d_next_beta, d_alpha_density,
          d_beta_density, d_active, d_converged, d_iterations,
          d_energy_change, d_density_rms);
    }
    iteration_error = cudaPeekAtLastError();
    return iteration_error == cudaSuccess
               ? VIBEQC_STATUS_SUCCESS
               : cuda_failure(iteration_error,
                              "advance CUDA DF device UHF SCF", detail);
  };
  bool graph_replay = state->graph_replay;
  if (!graph_replay && !plan->streamed && !state->solver.xsyev) {
    cuda_error = cudaStreamSynchronize(plan->stream);
    if (cuda_error == cudaSuccess) {
      cuda_error = cudaStreamBeginCapture(
          plan->stream, cudaStreamCaptureModeThreadLocal);
    }
    if (cuda_error == cudaSuccess) {
      status = launch_iteration();
      cudaGraph_t captured = nullptr;
      const cudaError_t end_error =
          cudaStreamEndCapture(plan->stream, &captured);
      if (status == VIBEQC_STATUS_SUCCESS && end_error == cudaSuccess &&
          captured != nullptr) {
        iteration_graph.graph = captured;
        cuda_error = cudaGraphInstantiate(
            &iteration_graph.executable, iteration_graph.graph, 0U);
        if (cuda_error == cudaSuccess) {
          cuda_error = cudaGraphUpload(iteration_graph.executable,
                                       plan->stream);
        }
        if (cuda_error == cudaSuccess) {
          cuda_error = cudaStreamSynchronize(plan->stream);
          graph_replay = cuda_error == cudaSuccess;
          state->graph_replay = graph_replay;
        }
      } else {
        if (captured != nullptr) (void)cudaGraphDestroy(captured);
        cuda_error = end_error;
      }
    }
    if (!graph_replay) cuda_error = cudaSuccess;
  }
  bool all_converged = false;
  for (unsigned iteration = 0; iteration < max_iterations && !all_converged;
       ++iteration) {
    if (graph_replay) {
      cuda_error = cudaGraphLaunch(iteration_graph.executable, plan->stream);
      if (cuda_error != cudaSuccess) {
        return cuda_failure(cuda_error, "replay CUDA DF UHF SCF Graph", detail);
      }
    } else {
      status = launch_iteration();
      if (status != VIBEQC_STATUS_SUCCESS) return status;
    }
    cuda_error = cudaMemcpyAsync(host_energy.data(), d_energy, batch_size * sizeof(double), cudaMemcpyDeviceToHost, plan->stream);
    if (cuda_error == cudaSuccess) cuda_error = cudaMemcpyAsync(host_energy_change.data(), d_energy_change, batch_size * sizeof(double), cudaMemcpyDeviceToHost, plan->stream);
    if (cuda_error == cudaSuccess) cuda_error = cudaMemcpyAsync(host_density_rms.data(), d_density_rms, batch_size * sizeof(double), cudaMemcpyDeviceToHost, plan->stream);
    if (cuda_error == cudaSuccess) cuda_error = cudaMemcpyAsync(host_converged.data(), d_converged, batch_size * sizeof(std::uint8_t), cudaMemcpyDeviceToHost, plan->stream);
    if (cuda_error == cudaSuccess) cuda_error = cudaMemcpyAsync(host_iterations.data(), d_iterations, batch_size * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, plan->stream);
    if (cuda_error == cudaSuccess) cuda_error = cudaMemcpyAsync(host_alpha_info.data(), d_alpha_info, batch_size * sizeof(int), cudaMemcpyDeviceToHost, plan->stream);
    if (cuda_error == cudaSuccess) cuda_error = cudaMemcpyAsync(host_beta_info.data(), d_beta_info, batch_size * sizeof(int), cudaMemcpyDeviceToHost, plan->stream);
    if (cuda_error == cudaSuccess) cuda_error = cudaStreamSynchronize(plan->stream);
    if (cuda_error != cudaSuccess) return cuda_failure(cuda_error, "read CUDA DF device UHF SCF records", detail);
    if (std::any_of(host_alpha_info.begin(), host_alpha_info.end(), [](int value) {
          return value != 0;
        }) ||
        std::any_of(host_beta_info.begin(), host_beta_info.end(), [](int value) {
          return value != 0;
        })) {
      detail = "CUDA DF device UHF eigensolver did not converge";
      return VIBEQC_STATUS_CUDA_ERROR;
    }
    all_converged = std::all_of(host_converged.begin(), host_converged.end(), [](std::uint8_t value) { return value != 0; });
  }
  final_alpha_density.resize(expected);
  final_beta_density.resize(expected);
  cuda_error = cudaMemcpyAsync(final_alpha_density.data(), d_alpha_density, matrix_bytes, cudaMemcpyDeviceToHost, plan->stream);
  if (cuda_error == cudaSuccess) cuda_error = cudaMemcpyAsync(final_beta_density.data(), d_beta_density, matrix_bytes, cudaMemcpyDeviceToHost, plan->stream);
  if (cuda_error == cudaSuccess) cuda_error = cudaStreamSynchronize(plan->stream);
  if (cuda_error != cudaSuccess) return cuda_failure(cuda_error, "read CUDA DF device UHF density", detail);
  if (!finite_values(final_alpha_density) || !finite_values(final_beta_density)) {
    detail = "CUDA DF device UHF SCF produced non-finite density";
    return VIBEQC_STATUS_NUMERICAL_FAILURE;
  }
  for (std::size_t system = 0; system < batch_size; ++system) {
    auto& result = results[system];
    result.status = VIBEQC_STATUS_SUCCESS;
    result.converged = host_converged[system] != 0;
    result.iterations = host_iterations[system];
    result.energy = host_energy[system];
    result.energy_change = host_energy_change[system];
    result.density_rms = host_density_rms[system];
    if (!result.converged) result.status = VIBEQC_STATUS_SCF_NOT_CONVERGED;
  }
  return VIBEQC_STATUS_SUCCESS;
}

void destroy_cuda_density_fitting_jk_plan(
    CudaDensityFittingJkPlan* plan) noexcept {
  if (plan == nullptr) return;
  release(*plan);
  delete plan;
}

}  // namespace vibeqc::scf
