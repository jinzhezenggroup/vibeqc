/** Complete raw-source generation, with launch-derived semantic work counts.
 * Run only under Slurm. Coefficient reads count logical lane operations, not
 * hardware memory transactions (the compiler/cache may coalesce them). The
 * counter model follows the exact requested public ranges and normalized AO
 * expansions; it adds no atomics to timed device kernels.
 */
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "molecule/basis.hpp"
#include "scf/cuda_density_fitting.hpp"

namespace {
using Clock = std::chrono::steady_clock;
void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void require(vibeqc_status status, const std::string& detail) {
  if (status != VIBEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
}
struct AoWork {
  std::uint64_t terms{}, primitive_products{};
};
std::vector<AoWork> ao_work(const vibeqc::core::System& system) {
  std::vector<AoWork> result;
  for (const auto& shell : system.shells)
    for (const auto& expansion :
         vibeqc::molecule::ao_expansions(shell.angular_momentum, system.basis_representation)) {
      std::uint64_t nonzero = 0;
      for (const auto& term : expansion) nonzero += term.coefficient != 0;
      result.push_back({nonzero, nonzero * shell.primitives.size()});
    }
  return result;
}
}  // namespace

int main(int argc, char** argv) {
  using namespace vibeqc::scf;
  try {
    if (!std::getenv("SLURM_JOB_ID") || argc != 5)
      throw std::runtime_error("usage inside Slurm: source-probe INPUT REPEATS OUTPUT RAW-ARRAY");
    std::ifstream input(argv[1]);
    std::size_t atoms, shells, rank;
    unsigned representation;
    std::string magic;
    input >> magic >> atoms >> shells >> representation >> rank;
    if (magic != "vibeqc-occupied-v1" || !atoms || !shells)
      throw std::runtime_error("invalid source fixture");
    vibeqc::core::System system;
    system.basis_representation = static_cast<vibeqc_basis_representation>(representation);
    system.atoms.resize(atoms);
    system.shells.resize(shells);
    for (auto& atom : system.atoms)
      input >> atom.atomic_number >> atom.position[0] >> atom.position[1] >> atom.position[2];
    for (auto& shell : system.shells) {
      std::size_t primitives;
      input >> shell.atom_index >> shell.angular_momentum >> primitives;
      shell.primitives.resize(primitives);
      for (auto& primitive : shell.primitives) input >> primitive.exponent >> primitive.coefficient;
    }
    if (!input) throw std::runtime_error("truncated source fixture");
    std::string detail;
    require(vibeqc::molecule::validate_and_normalize(system, detail), detail);
    CudaDensityFittingIntegralSource* raw_source = nullptr;
    std::vector<double> metric;
    std::size_t n, a;
    const auto setup_start = Clock::now();
    require(create_cuda_density_fitting_integral_source(0, {system}, {system}, &raw_source, metric,
                                                        n, a, detail),
            detail);
    std::unique_ptr<CudaDensityFittingIntegralSource,
                    decltype(&destroy_cuda_density_fitting_integral_source)>
        source(raw_source, destroy_cuda_density_fitting_integral_source);
    const auto placement = cuda_density_fitting_integral_source_diagnostic(source.get());
    const auto c = vibeqc::molecule::cartesian_ao_count(system);
    const auto work = ao_work(system);
    const auto setup_seconds = std::chrono::duration<double>(Clock::now() - setup_start).count();
    const unsigned lanes = std::string(placement.value_mapping) == "primitive" ? 32 : 1;
    // Match the production resident exporter: bounded public-pair blocks,
    // complete auxiliary rows, with one host result and unchanged layout.
    const auto pairs = n * n, pair_tile = std::min<std::size_t>(pairs, 8192);
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    double* device;
    check(cudaMalloc(reinterpret_cast<void**>(&device), pair_tile * a * sizeof(double)));
    std::vector<double> values(pairs * a);
    const auto repeats = std::stoul(argv[2]);
    if (!repeats || repeats > 100) throw std::runtime_error("invalid source repeats");
    std::ofstream report(argv[3]);
    report << std::setprecision(17);
    report << "{\"operation\":\"setup\",\"seconds\":" << setup_seconds << ",\"nbf\":" << n
           << ",\"naux\":" << a << ",\"mapping\":\"" << placement.value_mapping
           << "\",\"coefficient_lanes\":" << lanes << ",\"source_device_bytes\":"
           << cuda_density_fitting_integral_source_device_bytes(source.get())
           << ",\"source_host_peak_bytes\":"
           << cuda_density_fitting_integral_source_host_peak_bytes(source.get()) << "}\n";
    for (std::size_t repeat = 0; repeat < repeats; ++repeat) {
      std::uint64_t loads = 0, sparse_loads = 0, triples = 0, primitives = 0, launches = 0,
                    outputs = 0;
      const auto start = Clock::now();
      for (std::size_t begin = 0; begin < pairs; begin += pair_tile) {
        const auto count = std::min(pair_tile, pairs - begin);
        require(generate_cuda_density_fitting_raw_tile(source.get(), 0, begin, count, 0, a, -1,
                                                       stream, device, detail),
                detail);
        check(cudaMemcpyAsync(values.data() + begin * a, device, count * a * sizeof(double),
                              cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        ++launches;
        outputs += count * a;
      }
      const auto seconds = std::chrono::duration<double>(Clock::now() - start).count();
      // Compute semantic counters after timing, over exactly the submitted
      // ranges. Primitive partitioning divides products between warp lanes;
      // expansion coefficient traversal is repeated in each participating lane.
      std::uint64_t auxiliary_terms = 0, auxiliary_primitives = 0;
      for (const auto& entry : work) {
        auxiliary_terms += entry.terms;
        auxiliary_primitives += entry.primitive_products;
      }
      for (std::size_t pair = 0; pair < pairs; ++pair) {
        const auto& first = work[pair / n];
        const auto& second = work[pair % n];
        loads += a * (c + first.terms * c + first.terms * second.terms * c);
        sparse_loads += a * (first.terms + first.terms * second.terms) +
                        first.terms * second.terms * auxiliary_terms;
        triples += first.terms * second.terms * auxiliary_terms;
        primitives += first.primitive_products * second.primitive_products * auxiliary_primitives;
      }
      report << "{\"operation\":\"raw_generation\",\"repeat\":" << repeat
             << ",\"seconds\":" << seconds << ",\"outputs\":" << outputs
             << ",\"bytes\":" << outputs * sizeof(double) << ",\"source_launches\":" << launches
             << ",\"dense_coefficient_reads\":" << loads * lanes
             << ",\"dense_zero_skips\":" << (loads - sparse_loads) * lanes
             << ",\"sparse_coefficient_reads\":" << sparse_loads * lanes
             << ",\"nonzero_expansion_triples\":" << triples
             << ",\"contracted_df_lane_calls\":" << triples * lanes
             << ",\"primitive_products\":" << primitives << "}\n";
      report.flush();
    }
    std::ofstream arrays(argv[4], std::ios::binary);
    arrays.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(double));
    if (!report || !arrays) throw std::runtime_error("failed to retain source evidence");
    check(cudaFree(device));
    check(cudaStreamDestroy(stream));
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
