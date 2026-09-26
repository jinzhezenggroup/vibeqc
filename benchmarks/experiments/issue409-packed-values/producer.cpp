/** Direct triangular generation/whitening/J diagnostic for #409.
 * Reuses the production public-basis raw tile generator. Each lower AO row
 * is a contiguous dense-pair slice, written directly to its packed owner.
 * Neither a full raw nor a full transformed tensor is allocated here.
 * The independent test supplies X and D explicitly; this is not an SCF plan
 * or a production CPU metric-factorization path.
 */
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "molecule/basis.hpp"
#include "scf/cuda_density_fitting.hpp"

namespace {
using vibeqc::core::System;
using namespace vibeqc::scf;
void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}
void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void check(cublasStatus_t error) {
  if (error != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS failure");
}
struct Buffer {
  double* p{};
  explicit Buffer(std::size_t count) { check(cudaMalloc(&p, count * sizeof(double))); }
  ~Buffer() { cudaFree(p); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};
void shells(std::istream& input, System& system, std::size_t count) {
  require(count && count <= 2048, "invalid shell count");
  for (std::size_t i = 0; i < count; ++i) {
    vibeqc::core::Shell shell;
    std::size_t primitives;
    input >> shell.atom_index >> shell.angular_momentum >> primitives;
    require(input && primitives && primitives <= 64, "invalid shell record");
    for (std::size_t p = 0; p < primitives; ++p) {
      double exponent, coefficient;
      input >> exponent >> coefficient;
      shell.primitives.push_back({exponent, coefficient});
    }
    system.shells.push_back(std::move(shell));
  }
  std::string detail;
  require(bool(input), "truncated shell input");
  require(vibeqc::molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS,
          detail);
}
std::vector<double> read(const std::string& name, std::size_t count) {
  std::ifstream file(name, std::ios::binary | std::ios::ate);
  require(file && file.tellg() == static_cast<std::streamoff>(count * sizeof(double)),
          "input extent mismatch: " + name);
  file.seekg(0);
  std::vector<double> values(count);
  file.read(reinterpret_cast<char*>(values.data()), count * sizeof(double));
  require(bool(file), "cannot read " + name);
  return values;
}
void write(const std::string& name, const double* data, std::size_t count) {
  std::ofstream file(name, std::ios::binary);
  file.write(reinterpret_cast<const char*>(data), count * sizeof(double));
  require(bool(file), "cannot write " + name);
}
}  // namespace

int main(int argc, char** argv) try {
  require(argc == 6 && std::getenv("SLURM_JOB_ID"),
          "finite Slurm required: producer basis.txt X.bin D.bin output-prefix device-budget");
  std::ifstream input(argv[1]);
  std::size_t batch;
  unsigned representation;
  input >> batch >> representation;
  require(batch && batch <= 8 && representation <= 2, "invalid batch/representation");
  std::vector<System> orbital(batch), auxiliary(batch);
  for (std::size_t item = 0; item < batch; ++item) {
    unsigned orepr = representation, arepr = representation;
    if (representation == 2) input >> orepr >> arepr;
    require(orepr <= 1 && arepr <= 1, "invalid per-basis representation");
    std::size_t atoms, oshells, ashells;
    input >> atoms >> oshells >> ashells;
    require(atoms && atoms <= 256, "invalid atoms");
    for (std::size_t i = 0; i < atoms; ++i) {
      vibeqc::core::Atom atom;
      input >> atom.atomic_number >> atom.position[0] >> atom.position[1] >> atom.position[2];
      orbital[item].atoms.push_back(atom);
    }
    auxiliary[item].atoms = orbital[item].atoms;
    orbital[item].basis_representation = orepr ? VIBEQC_BASIS_SPHERICAL : VIBEQC_BASIS_CARTESIAN;
    auxiliary[item].basis_representation = arepr ? VIBEQC_BASIS_SPHERICAL : VIBEQC_BASIS_CARTESIAN;
    shells(input, orbital[item], oshells);
    shells(input, auxiliary[item], ashells);
  }
  std::size_t n{}, a{};
  std::vector<double> metrics;
  std::string detail;
  CudaDensityFittingIntegralSource* raw_source{};
  require(create_cuda_density_fitting_integral_source(0, orbital, auxiliary, &raw_source, metrics,
                                                      n, a, detail) == VIBEQC_STATUS_SUCCESS,
          detail);
  const auto source = std::unique_ptr<CudaDensityFittingIntegralSource,
                                      decltype(&destroy_cuda_density_fitting_integral_source)>(
      raw_source, &destroy_cuda_density_fitting_integral_source);
  require(n && n <= 2048 && a && a <= 2048, "producer shape outside bounded experiment");
  const auto pairs = n * (n + 1) / 2, tensor = pairs * a;
  const auto bytes = (2 * tensor + a * a + 2 * pairs + a) * sizeof(double);
  const auto source_bytes = cuda_density_fitting_integral_source_device_bytes(source.get());
  const auto budget = std::stoull(argv[5]);
  require(bytes + source_bytes + (64ULL << 20) <= budget,
          "producer numeric/library budget exceeded");
  Buffer raw(tensor), transformed(tensor), inverse(a * a), density(pairs), charges(a), j(pairs);
  const auto xs = read(argv[2], batch * a * a), ds = read(argv[3], batch * n * n);
  std::vector<double> weights(pairs), downloaded(tensor);
  cudaStream_t stream;
  check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cublasHandle_t blas;
  check(cublasCreate(&blas));
  check(cublasSetStream(blas, stream));
  check(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST));
  const double one = 1, zero = 0;
  const std::string prefix = argv[4];
  write(prefix + "-metric.bin", metrics.data(), metrics.size());
  std::cout << std::setprecision(17);
  for (std::size_t item = 0; item < batch; ++item) {
    // Values are unit-weight packed; nonsymmetric D must use both entries.
    for (std::size_t mu = 0; mu < n; ++mu)
      for (std::size_t nu = 0; nu <= mu; ++nu)
        weights[mu * (mu + 1) / 2 + nu] =
            ds[item * n * n + mu * n + nu] + (mu == nu ? 0 : ds[item * n * n + nu * n + mu]);
    check(cudaMemcpyAsync(inverse.p, xs.data() + item * a * a, a * a * sizeof(double),
                          cudaMemcpyHostToDevice, stream));
    check(cudaMemcpyAsync(density.p, weights.data(), pairs * sizeof(double), cudaMemcpyHostToDevice,
                          stream));
    check(cudaStreamSynchronize(stream));
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t mu = 0; mu < n; ++mu)
      require(generate_cuda_density_fitting_raw_tile(source.get(), item, mu * n, mu + 1, 0, a, -1,
                                                     stream, raw.p + mu * (mu + 1) / 2 * a,
                                                     detail) == VIBEQC_STATUS_SUCCESS,
              detail);
    check(cudaStreamSynchronize(stream));
    const auto generated = std::chrono::steady_clock::now();
    // One all-Q transform. Raw A survives, including discarded directions.
    check(cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, a, pairs, a, &one, inverse.p, a, raw.p, a,
                      &zero, transformed.p, a));
    check(cudaStreamSynchronize(stream));
    const auto whitened = std::chrono::steady_clock::now();
    check(cublasDgemv(blas, CUBLAS_OP_N, a, pairs, &one, transformed.p, a, density.p, 1, &zero,
                      charges.p, 1));
    check(cublasDgemv(blas, CUBLAS_OP_T, a, pairs, &one, transformed.p, a, charges.p, 1, &zero, j.p,
                      1));
    check(cudaStreamSynchronize(stream));
    const auto contracted = std::chrono::steady_clock::now();
    const auto seconds = [](auto begin, auto end) {
      return std::chrono::duration<double>(end - begin).count();
    };
    std::cout << "{\"system\":" << item << ",\"n\":" << n << ",\"a\":" << a
              << ",\"pairs\":" << pairs << ",\"owned_numeric_device_bytes\":" << bytes
              << ",\"source_device_bytes\":" << source_bytes << ",\"generation_calls\":" << n
              << ",\"generated_raw_values\":" << tensor
              << ",\"whitening_flops\":" << 2 * pairs * a * a << ",\"J_flops\":" << 4 * pairs * a
              << ",\"generation_seconds\":" << seconds(start, generated)
              << ",\"whitening_seconds\":" << seconds(generated, whitened)
              << ",\"J_seconds\":" << seconds(whitened, contracted)
              << ",\"timing_scope\":\"single intrusive correctness call; no latency claim\"}\n";
    for (const auto& output :
         {std::pair{"raw", raw.p}, std::pair{"B", transformed.p}, std::pair{"J", j.p}}) {
      const auto count = std::string(output.first) == "J" ? pairs : tensor;
      check(cudaMemcpyAsync(downloaded.data(), output.second, count * sizeof(double),
                            cudaMemcpyDeviceToHost, stream));
      check(cudaStreamSynchronize(stream));
      write(prefix + "-" + std::to_string(item) + "-" + output.first + ".bin", downloaded.data(),
            count);
    }
  }
  check(cublasDestroy(blas));
  check(cudaStreamDestroy(stream));
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
