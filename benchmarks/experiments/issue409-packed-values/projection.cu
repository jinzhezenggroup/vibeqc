// Fixed-input #409 experiment. Packing/capture is diagnostic setup only: this
// executable does not implement or qualify a packed molecular producer.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void checked(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
void checked(cublasStatus_t e) {
  if (e != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS failure " + std::to_string(e));
}
struct Buffer {
  double* p{};
  explicit Buffer(std::size_t count) {
    checked(cudaMalloc(&p, std::max<std::size_t>(1, count) * sizeof(double)));
  }
  ~Buffer() { cudaFree(p); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};

// The lower triangle has unit weights. Symmetry is a precondition checked by
// the input planner; the dense compatibility control retains the original B.
__device__ std::size_t pair_index(std::size_t mu, std::size_t nu) {
  const auto hi = mu > nu ? mu : nu, lo = mu > nu ? nu : mu;
  return hi * (hi + 1) / 2 + lo;
}
__global__ void pack(int n, int a, const double* full, double* packed) {
  const auto count = std::size_t(n) * n * a;
  for (auto k = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; k < count;
       k += std::size_t(gridDim.x) * blockDim.x) {
    const auto q = k % a, mu = k / a / n, nu = k / a % n;
    if (nu <= mu) packed[(mu * (mu + 1) / 2 + nu) * a + q] = full[k];
  }
}

// Bounded [mu,nu,q] expansion matches the existing batched projection's
// contracted AO order. U keeps its original full-Q leading dimension.
__global__ void unpack(int n, int a, int begin, int count, const double* packed, double* expanded) {
  const auto elements = std::size_t(n) * n * count;
  for (auto k = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; k < elements;
       k += std::size_t(gridDim.x) * blockDim.x) {
    const auto q = k % count, mu = k / count / n, nu = k / count % n;
    expanded[k] = packed[pair_index(mu, nu) * a + begin + q];
  }
}

/** Tiled FP64 contraction with an on-load triangular iterator.
 * Each block fixes mu and a rectangle (Q,occupied); shared factors/coefficient
 * tiles are reused across 16 occupied columns. Every nu is reduced once in
 * ascending order. No atomics, expanded global B, or alternate Gram layout.
 * This is a SIMT candidate, not a claim to use CUTLASS or cuTENSOR.
 */
template <int Q>
__global__ void direct(int n, int a, int rank, const double* packed, const double* c, double* u) {
  constexpr int R = 16, K = 32;
  __shared__ double bs[K][Q];
  __shared__ double cs[R][K + 1];
  const int x = threadIdx.x, y = threadIdx.y;
  const int mu = blockIdx.z, q = blockIdx.x * Q + x, rb = blockIdx.y * R;
  const int thread = y * Q + x, threads = Q * 4;
  double value[4] = {};
  for (int begin = 0; begin < n; begin += K) {
    for (int element = thread; element < K * Q; element += threads) {
      const int nu = begin + element / Q, qq = blockIdx.x * Q + element % Q;
      bs[element / Q][element % Q] = nu < n && qq < a ? packed[pair_index(mu, nu) * a + qq] : 0;
    }
    for (int element = thread; element < K * R; element += threads) {
      const int nu = begin + element % K, rr = rb + element / K;
      cs[element / K][element % K] = nu < n && rr < rank ? c[nu + std::size_t(rr) * n] : 0;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < K; ++k) {
      const double b = bs[k][x];
#pragma unroll
      for (int j = 0; j < 4; ++j) value[j] = fma(b, cs[y + j * 4][k], value[j]);
    }
    __syncthreads();
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int r = rb + y + j * 4;
    if (q < a && r < rank) u[(std::size_t(mu) * rank + r) * a + q] = value[j];
  }
}
__global__ void mirror(int n, double* k) {
  const auto index = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < std::size_t(n) * n) {
    const auto row = index % n, column = index / n;
    if (row < column) k[index] = k[column + row * n];
  }
}
__global__ void differences(std::size_t n, const double* x, const double* ref, double* summary) {
  __shared__ double max_abs[256], sum_sq[256];
  double m = 0, s = 0;
  for (auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += std::size_t(gridDim.x) * blockDim.x) {
    const double d = x[i] - ref[i];
    // Non-finite values must fail explicitly; fmax alone could hide NaNs.
    m = !isfinite(d) ? INFINITY : fmax(m, fabs(d));
    s += d * d;
  }
  max_abs[threadIdx.x] = m;
  sum_sq[threadIdx.x] = s;
  __syncthreads();
  for (int stride = 128; stride; stride /= 2) {
    if (threadIdx.x < stride) {
      max_abs[threadIdx.x] = fmax(max_abs[threadIdx.x], max_abs[threadIdx.x + stride]);
      sum_sq[threadIdx.x] += sum_sq[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (!threadIdx.x) {
    summary[blockIdx.x * 2] = max_abs[0];
    summary[blockIdx.x * 2 + 1] = sum_sq[0];
  }
}

// Bounded pageable input staging is excluded from every timed interval.
void upload(const std::string& name, double* device, std::size_t count, cudaStream_t stream) {
  std::ifstream input(name, std::ios::binary | std::ios::ate);
  if (!input || input.tellg() != static_cast<std::streamoff>(count * sizeof(double)))
    throw std::runtime_error("input extent mismatch: " + name);
  input.seekg(0);
  std::vector<double> values(1 << 20);
  for (std::size_t begin = 0; begin < count; begin += values.size()) {
    const auto size = std::min(values.size(), count - begin);
    input.read(reinterpret_cast<char*>(values.data()), size * sizeof(double));
    if (!input) throw std::runtime_error("input read failure: " + name);
    checked(cudaMemcpyAsync(device + begin, values.data(), size * sizeof(double),
                            cudaMemcpyHostToDevice, stream));
    checked(cudaStreamSynchronize(stream));
  }
}
std::array<double, 2> error(const double* x, const double* ref, std::size_t count, double* scratch,
                            cudaStream_t stream) {
  differences<<<256, 256, 0, stream>>>(count, x, ref, scratch);
  checked(cudaGetLastError());
  std::array<double, 512> host{};
  checked(cudaMemcpyAsync(host.data(), scratch, sizeof(host), cudaMemcpyDeviceToHost, stream));
  checked(cudaStreamSynchronize(stream));
  double maximum = 0, sum = 0;
  for (int i = 0; i < 256; ++i) {
    maximum = std::max(maximum, host[2 * i]);
    sum += host[2 * i + 1];
  }
  return {maximum, count ? std::sqrt(sum / count) : 0};
}
}  // namespace

int main(int argc, char** argv) try {
  if (argc != 10 || !std::getenv("SLURM_JOB_ID"))
    throw std::runtime_error(
        "finite Slurm required: n a rank repeats weight B.bin C.bin U.bin|- prefix");
  const int n = std::stoi(argv[1]), a = std::stoi(argv[2]), r = std::stoi(argv[3]);
  const int repeats = std::stoi(argv[4]);
  const double weight = std::stod(argv[5]);
  if (n < 1 || n > 2048 || a < 1 || a > 2048 || r < 0 || r > n || repeats < 1 || repeats > 31 ||
      !std::isfinite(weight) || weight <= 0)
    throw std::runtime_error("trial dimensions outside bounded domain");
  const auto matrix = std::size_t(n) * n, tensor = matrix * a;
  const auto packed_count = std::size_t(n) * (n + 1) / 2 * a, occupied = std::size_t(n) * r * a;
  cudaDeviceProp properties{};
  checked(cudaGetDeviceProperties(&properties, 0));
  cudaStream_t stream;
  checked(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cublasHandle_t blas;
  checked(cublasCreate(&blas));
  checked(cublasSetStream(blas, stream));
  checked(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST));
  checked(cublasSetMathMode(blas, CUBLAS_PEDANTIC_MATH));
  Buffer full(tensor), packed(packed_count), c(std::size_t(n) * r);
  Buffer u(occupied), reference_u(occupied), k(matrix), reference_k(matrix);
  Buffer expanded(matrix * std::min(a, 256)), errors(512);
  cudaFuncAttributes direct32{}, direct64{};
  checked(cudaFuncGetAttributes(&direct32, direct<32>));
  checked(cudaFuncGetAttributes(&direct64, direct<64>));
  std::ofstream metadata(std::string(argv[9]) + "-device.json");
  if (!metadata) throw std::runtime_error("cannot create device metadata");
  const auto owned_bytes =
      (tensor + packed_count + std::max<std::size_t>(1, std::size_t(n) * r) +
       2 * std::max<std::size_t>(1, occupied) + 2 * matrix + matrix * std::min(a, 256) + 512) *
      8;
  metadata << "{\"device_name\":\"" << properties.name << "\",\"major\":" << properties.major
           << ",\"minor\":" << properties.minor << ",\"owned_device_bytes\":" << owned_bytes
           << ",\"direct32_registers\":" << direct32.numRegs
           << ",\"direct64_registers\":" << direct64.numRegs
           << ",\"direct32_shared_bytes\":" << direct32.sharedSizeBytes
           << ",\"direct64_shared_bytes\":" << direct64.sharedSizeBytes
           << ",\"math_mode\":\"CUBLAS_PEDANTIC_MATH; FP64\"}\n";
  metadata.close();
  upload(argv[6], full.p, tensor, stream);
  upload(argv[7], c.p, std::size_t(n) * r, stream);
  if (r) {
    // Match the native SCF's canonical column-major coefficients with bounded
    // host staging. No timed transpose and no additional device allocation.
    std::vector<double> row_major(std::size_t(n) * r), column_major(row_major.size());
    checked(cudaMemcpyAsync(row_major.data(), c.p, row_major.size() * sizeof(double),
                            cudaMemcpyDeviceToHost, stream));
    checked(cudaStreamSynchronize(stream));
    for (int nu = 0; nu < n; ++nu)
      for (int ri = 0; ri < r; ++ri)
        column_major[nu + std::size_t(ri) * n] = row_major[std::size_t(nu) * r + ri];
    checked(cudaMemcpyAsync(c.p, column_major.data(), column_major.size() * sizeof(double),
                            cudaMemcpyHostToDevice, stream));
    checked(cudaStreamSynchronize(stream));
  }
  pack<<<4096, 256, 0, stream>>>(n, a, full.p, packed.p);
  checked(cudaGetLastError());
  const double one = 1, zero = 0;
  const std::array<std::string, 6> modes = {"full",      "unpack32",    "unpack128",
                                            "unpack256", "direct32x16", "direct64x16"};
  const auto projection = [&](int mode) {
    if (!r) return;
    if (mode == 0) {
      checked(cublasDgemmStridedBatched(blas, CUBLAS_OP_N, CUBLAS_OP_N, a, r, n, &one, full.p, a,
                                        std::size_t(a) * n, c.p, n, 0, &zero, u.p, a,
                                        std::size_t(a) * r, n));
    } else if (mode < 4) {
      const int tile = mode == 1 ? 32 : mode == 2 ? 128 : 256;
      for (int begin = 0; begin < a; begin += tile) {
        const auto count = std::min(tile, a - begin);
        unpack<<<4096, 256, 0, stream>>>(n, a, begin, count, packed.p, expanded.p);
        checked(cudaGetLastError());
        checked(cublasDgemmStridedBatched(blas, CUBLAS_OP_N, CUBLAS_OP_N, count, r, n, &one,
                                          expanded.p, count, std::size_t(count) * n, c.p, n, 0,
                                          &zero, u.p + begin, a, std::size_t(a) * r, n));
      }
    } else if (mode == 4) {
      direct<32><<<dim3((a + 31) / 32, (r + 15) / 16, n), dim3(32, 4), 0, stream>>>(
          n, a, r, packed.p, c.p, u.p);
    } else {
      direct<64><<<dim3((a + 63) / 64, (r + 15) / 16, n), dim3(64, 4), 0, stream>>>(
          n, a, r, packed.p, c.p, u.p);
    }
    checked(cudaGetLastError());
  };
  const auto gram = [&] {
    if (r) {
      checked(cublasDsyrk(blas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, n, a * r, &weight, u.p, a * r,
                          &zero, k.p, n));
      mirror<<<(matrix + 255) / 256, 256, 0, stream>>>(n, k.p);
      checked(cudaGetLastError());
    } else
      checked(cudaMemsetAsync(k.p, 0, matrix * sizeof(double), stream));
  };
  projection(0);
  gram();
  checked(cudaMemcpyAsync(reference_u.p, u.p, occupied * sizeof(double), cudaMemcpyDeviceToDevice,
                          stream));
  checked(cudaMemcpyAsync(reference_k.p, k.p, matrix * sizeof(double), cudaMemcpyDeviceToDevice,
                          stream));
  std::array<double, 2> captured_error{};
  if (std::string(argv[8]) != "-") {
    upload(argv[8], u.p, occupied, stream);
    captured_error = error(u.p, reference_u.p, occupied, errors.p, stream);
    if (captured_error[0] > 1e-10 || !std::isfinite(captured_error[0]))
      throw std::runtime_error("captured U differs from fixed full projection");
  }
  // Save deterministic scalar samples for independent long-double validation,
  // without copying the complete U or expanding B on the host.
  std::ofstream independent(std::string(argv[9]) + "-samples.jsonl");
  if (!independent) throw std::runtime_error("cannot create independent sample output");
  independent << std::setprecision(17);
  if (r) {
    for (int index = 0; index < 32; ++index) {
      const int mu = (index * 73 + n - 1) % n, q = (index * 41 + a - 1) % a;
      const int ri = (index * 17 + r - 1) % r;
      double value;
      checked(cudaMemcpyAsync(&value, reference_u.p + (std::size_t(mu) * r + ri) * a + q,
                              sizeof(double), cudaMemcpyDeviceToHost, stream));
      checked(cudaStreamSynchronize(stream));
      independent << "{\"mu\":" << mu << ",\"q\":" << q << ",\"i\":" << ri << ",\"value\":" << value
                  << "}\n";
    }
  }
  independent.close();
  cudaEvent_t start, projected, stop;
  checked(cudaEventCreate(&start));
  checked(cudaEventCreate(&projected));
  checked(cudaEventCreate(&stop));
  std::cout << std::setprecision(17);
  bool passed = true;
  for (int repeat = 0; repeat < repeats; ++repeat) {
    for (int order = 0; order < 6; ++order) {
      const int mode = (order + repeat) % 6;
      projection(mode);
      gram();
      checked(cudaStreamSynchronize(stream));
      const auto wall_start = std::chrono::steady_clock::now();
      checked(cudaEventRecord(start, stream));
      projection(mode);
      checked(cudaEventRecord(projected, stream));
      gram();
      checked(cudaEventRecord(stop, stream));
      checked(cudaEventSynchronize(stop));
      const double wall =
          std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
      float projection_ms{}, gram_ms{};
      checked(cudaEventElapsedTime(&projection_ms, start, projected));
      checked(cudaEventElapsedTime(&gram_ms, projected, stop));
      const auto ue = error(u.p, reference_u.p, occupied, errors.p, stream);
      const auto ke = error(k.p, reference_k.p, matrix, errors.p, stream);
      const bool good = std::isfinite(ue[0]) && std::isfinite(ue[1]) && std::isfinite(ke[0]) &&
                        std::isfinite(ke[1]) && ue[0] <= 1e-10 && ue[1] <= 1e-11 &&
                        ke[0] <= 1e-10 && ke[1] <= 1e-11;
      passed &= good;
      std::cout << "{\"mode\":\"" << modes[mode] << "\",\"repeat\":" << repeat
                << ",\"projection_seconds\":" << projection_ms / 1000.0
                << ",\"gram_mirror_seconds\":" << gram_ms / 1000.0 << ",\"wall_seconds\":" << wall
                << ",\"U_max_abs\":" << ue[0] << ",\"U_rms\":" << ue[1]
                << ",\"K_max_abs\":" << ke[0] << ",\"K_rms\":" << ke[1]
                << ",\"passed\":" << (good ? "true" : "false")
                << ",\"captured_U_max_abs\":" << captured_error[0] << "}\n"
                << std::flush;
    }
  }
  checked(cudaEventDestroy(start));
  checked(cudaEventDestroy(projected));
  checked(cudaEventDestroy(stop));
  checked(cublasDestroy(blas));
  checked(cudaStreamDestroy(stream));
  return passed ? 0 : 2;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
