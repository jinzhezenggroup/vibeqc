/** Complete fixed-density K timings including upload, contraction and readback.
 * The input freezes physical geometry/basis and converged canonical orbitals;
 * preparation is timed separately. A Python driver records source/library/input
 * hashes. Execute only under a finite Slurm allocation, with tracing disabled
 * for endpoint measurements and a separate traced invocation for components.
 */
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "molecule/basis.hpp"
#include "scf/cuda_density_fitting.hpp"

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}
void check(vibeqc_status status, const std::string& detail) {
  if (status != VIBEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    using namespace vibeqc::scf;
    if (!std::getenv("SLURM_JOB_ID") || argc != 6)
      throw std::runtime_error("usage inside Slurm: probe INPUT REPEATS AUX_TILE AO_ROWS ARRAYS");
    const auto repeats = std::stoul(argv[2]);
    if (!repeats || repeats > 1000) throw std::runtime_error("invalid repeat count");
    std::ifstream input(argv[1]);
    std::string magic;
    std::size_t atoms = 0, shells = 0, rank = 0;
    int representation = 0;
    input >> magic >> atoms >> shells >> representation >> rank;
    if (magic != "vibeqc-occupied-v1" || !atoms || !shells)
      throw std::runtime_error("invalid occupied probe input");
    vibeqc::core::System system;
    system.basis_representation = static_cast<vibeqc_basis_representation>(representation);
    system.atoms.resize(atoms);
    system.shells.resize(shells);
    for (auto& atom : system.atoms)
      input >> atom.atomic_number >> atom.position[0] >> atom.position[1] >> atom.position[2];
    for (auto& shell : system.shells) {
      std::size_t primitives = 0;
      input >> shell.atom_index >> shell.angular_momentum >> primitives;
      shell.primitives.resize(primitives);
      for (auto& primitive : shell.primitives) input >> primitive.exponent >> primitive.coefficient;
    }
    std::string detail;
    check(vibeqc::molecule::validate_and_normalize(system, detail), detail);
    const auto nbf = vibeqc::molecule::ao_count(system);
    if (rank > nbf) throw std::runtime_error("occupied rank exceeds AO dimension");
    std::vector<double> coefficients(nbf * rank), occupations(rank, 2.0);
    for (auto& value : coefficients) input >> value;
    if (!input) throw std::runtime_error("truncated occupied probe input");
    const auto start = Clock::now();
    CudaDensityFittingIntegralSource* source = nullptr;
    std::vector<double> metrics;
    std::size_t source_n = 0, naux = 0;
    check(create_cuda_density_fitting_integral_source(0, {system}, {system}, &source, metrics,
                                                      source_n, naux, detail),
          detail);
    CudaDensityFittingJkPlan* raw = nullptr;
    std::vector<CudaDensityFittingMetricDiagnostic> diagnostics;
    const auto aux_tile = std::stoul(argv[3]);
    const auto rows = std::stoul(argv[4]);
    const auto status = create_cuda_density_fitting_jk_plan_from_source(
        0, &source, 1, nbf, naux, metrics, 1e-10, aux_tile ? aux_tile : naux,
        (rows ? rows : nbf) * nbf, &raw, diagnostics, detail);
    destroy_cuda_density_fitting_integral_source(source);
    check(status, detail);
    std::unique_ptr<CudaDensityFittingJkPlan, decltype(&destroy_cuda_density_fitting_jk_plan)> plan(
        raw, destroy_cuda_density_fitting_jk_plan);
    const double setup_seconds = elapsed(start);
    const auto identity = cuda_density_fitting_factor_identity(plan.get(), 0, 1, 1);
    const OccupiedDensityFactor factor(identity, DensityFactorSpin::Restricted, nbf, coefficients,
                                       occupations);
    const CudaOccupiedDensityInput factors[]{{&factor, identity}};
    std::vector<double> density(factor.density().begin(), factor.density().end());
    std::vector<double> unused, dense, occupied;
    std::vector<std::uint8_t> selected;
    const auto run = [&](bool compressed) {
      if (compressed) {
        check(execute_cuda_density_fitting_occupied_exchange(plan.get(), density,
                                                             DensityFactorSpin::Restricted, factors,
                                                             occupied, selected, detail),
              detail);
        if (selected != std::vector<std::uint8_t>{1})
          throw std::runtime_error("fixed-K factor unexpectedly fell back");
      } else {
        check(execute_cuda_density_fitting_rhf_jk(plan.get(), density, unused, dense, detail,
                                                  {false, true}),
              detail);
      }
    };
    std::cout << std::setprecision(17);
    std::cout << "{\"operation\":\"setup\",\"seconds\":" << setup_seconds << ",\"nbf\":" << nbf
              << ",\"naux\":" << naux << ",\"rank\":" << rank
              << ",\"value_plan_resident_bytes\":" << diagnostics[0].device_resident_bytes
              << ",\"value_plan_peak_bytes\":" << diagnostics[0].peak_device_bytes
              << ",\"streamed\":" << (diagnostics[0].streamed ? "true" : "false") << "}\n";
    run(false);
    run(true);
    for (std::size_t repeat = 0; repeat < repeats; ++repeat)
      for (unsigned position = 0; position < 2; ++position) {
        const bool compressed = (repeat + position) % 2;
        const auto begin = Clock::now();
        run(compressed);
        const double seconds = elapsed(begin);
        double error = 0;
        for (std::size_t i = 0; i < dense.size(); ++i) {
          if (!std::isfinite(dense[i]) || !std::isfinite(occupied[i]))
            throw std::runtime_error("nonfinite fixed-K output");
          error = std::max(error, std::abs(dense[i] - occupied[i]));
        }
        if (error > 1e-9) throw std::runtime_error("occupied/dense K gate failed");
        std::cout << "{\"operation\":\"" << (compressed ? "occupied" : "dense")
                  << "\",\"repeat\":" << repeat << ",\"seconds\":" << seconds
                  << ",\"maximum_k_error\":" << error << "}\n";
      }
    static_assert(std::endian::native == std::endian::little);
    std::ofstream arrays(argv[5], std::ios::binary);
    // Exact float64 little-endian row-major D, dense K, occupied K, in that order.
    for (const auto* values : {&density, &dense, &occupied})
      arrays.write(reinterpret_cast<const char*>(values->data()),
                   static_cast<std::streamsize>(values->size() * sizeof(double)));
    if (!arrays) throw std::runtime_error("failed to retain fixed-K arrays");
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
