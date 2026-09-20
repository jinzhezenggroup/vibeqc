/** Untimed physical Fock diagnostic for #434 at a preserved failed density. Dumps native
 * source matrices for independent CPU cross-composition; never runs SCF or
 * manufactures an occupied lease. All additional snapshots are diagnostic cost.
 */
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda_density_fitting.hpp"
#include "scf/cuda_density_fitting_integrals.hpp"
#include "scf/cuda_df_gradient.hpp"

int main(int argc, char** argv) {
  using namespace vibeqc;
  try {
    if (argc != 3 || !std::getenv("SLURM_JOB_ID") || !std::getenv("CUDA_VISIBLE_DEVICES"))
      throw std::runtime_error("Slurm only: probe INPUT OUTPUT_DIRECTORY");
    std::cout << std::unitbuf << std::setprecision(17);
    Dl_info identity{};
    if (!dladdr(reinterpret_cast<void*>(&vibeqc_get_source_identity), &identity))
      throw std::runtime_error("cannot identify loaded native library");
    std::cout << "{\"operation\":\"identity\",\"library\":" << std::quoted(identity.dli_fname)
              << ",\"source_identity\":" << std::quoted(vibeqc_get_source_identity()) << "}\n";
    std::ifstream input(argv[1]);
    std::string magic, detail;
    std::size_t atoms{}, orbital_shells{}, auxiliary_shells{}, expected_n{}, expected_a{};
    input >> magic >> atoms >> orbital_shells >> auxiliary_shells >> expected_n >> expected_a;
    if (magic != "vibeqc-response-v1" || !atoms || !expected_n || !expected_a)
      throw std::runtime_error("invalid response fixture header");
    core::System orbital;
    orbital.basis_representation = VIBEQC_BASIS_SPHERICAL;
    orbital.atoms.resize(atoms);
    for (auto& atom : orbital.atoms)
      input >> atom.atomic_number >> atom.position[0] >> atom.position[1] >> atom.position[2];
    auto auxiliary = orbital;
    const auto read_shells = [&](core::System& system, std::size_t count) {
      system.shells.resize(count);
      for (auto& shell : system.shells) {
        std::size_t primitives{};
        input >> shell.atom_index >> shell.angular_momentum >> primitives;
        shell.primitives.resize(primitives);
        for (auto& primitive : shell.primitives)
          input >> primitive.exponent >> primitive.coefficient;
      }
    };
    read_shells(orbital, orbital_shells);
    read_shells(auxiliary, auxiliary_shells);
    const auto gpu = [](cudaError_t status) {
      if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    };
    const std::filesystem::path output(argv[2]);
    if (!std::filesystem::is_directory(output))
      throw std::runtime_error("missing output directory");
    const auto write = [&](const std::string& name, const std::vector<double>& values) {
      const auto path = output / (name + ".bin");
      if (std::filesystem::exists(path)) throw std::runtime_error("refusing to overwrite snapshot");
      std::ofstream file(path, std::ios::binary);
      file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(double));
      if (!file) throw std::runtime_error("cannot save native array");
    };
    const auto check = [&](vibeqc_status status) {
      if (status != VIBEQC_STATUS_SUCCESS) throw std::runtime_error(detail);
    };
    check(molecule::validate_and_normalize(orbital, detail));
    check(molecule::validate_and_normalize(auxiliary, detail));
    const auto n = molecule::ao_count(orbital), a = molecule::ao_count(auxiliary);
    if (n != expected_n || a != expected_a) throw std::runtime_error("basis dimensions differ");
    std::vector<double> density(n * n), overlap(n * n);
    for (auto& value : density) input >> value;
    for (auto& value : overlap) input >> value;
    if (!input) throw std::runtime_error("truncated density/overlap fixture");
    for (const auto* values : {&density, &overlap})
      if (!std::all_of(values->begin(), values->end(), [](double x) { return std::isfinite(x); }))
        throw std::runtime_error("nonfinite fixture");
    integrals::IntegralData cartesian;
    check(scf::build_cuda_one_electron_integrals(0, orbital, cartesian, detail, false, false));
    const auto one = integrals::transform_integrals(cartesian, orbital);
    double overlap_error = 0;
    for (std::size_t k = 0; k < overlap.size(); ++k)
      overlap_error = std::max(overlap_error, std::abs(one.overlap[k] - overlap[k]));
    if (!std::isfinite(overlap_error) || overlap_error > 1e-11)
      throw std::runtime_error("independent overlap convention gate failed");
    scf::CudaDensityFittingIntegralSource* source{};
    std::vector<double> metric;
    std::size_t source_n{}, source_a{};
    check(scf::create_cuda_density_fitting_integral_source(0, {orbital}, {auxiliary}, &source,
                                                           metric, source_n, source_a, detail));
    scf::CudaDensityFittingJkPlan* raw{};
    std::vector<scf::CudaDensityFittingMetricDiagnostic> diagnostics;
    const auto status = scf::create_cuda_density_fitting_jk_plan_from_source(
        0, &source, 1, n, a, metric, 1e-10, a, n * n, &raw, diagnostics, detail, true,
        {scf::DfPairStorage::Dense, 0});
    scf::destroy_cuda_density_fitting_integral_source(source);
    check(status);
    std::unique_ptr<scf::CudaDensityFittingJkPlan,
                    decltype(&scf::destroy_cuda_density_fitting_jk_plan)>
        plan(raw, scf::destroy_cuda_density_fitting_jk_plan);
    write("overlap", one.overlap);
    write("hcore", one.hcore);
    write("metric", metric);
    write("density", density);
    std::vector<double> coulomb, exchange;
    check(scf::execute_cuda_density_fitting_rhf_jk_item(plan.get(), 0, density, coulomb, exchange,
                                                        detail));
    write("coulomb", coulomb);
    write("exchange", exchange);
    auto fock = one.hcore;
    for (std::size_t k = 0; k < fock.size(); ++k) fock[k] += coulomb[k] - .5 * exchange[k];
    write("fock", fock);
    const auto download = [&](const std::string& name, const double* pointer, std::size_t count) {
      std::vector<double> values(count);
      gpu(cudaMemcpyAsync(values.data(), pointer, count * sizeof(double), cudaMemcpyDeviceToHost,
                          plan->stream));
      gpu(cudaStreamSynchronize(plan->stream));
      write(name, values);
    };
    // Device eigensystem matrices are column-major; raw/whitened factors use
    // public row-major [mu,nu,P]. Record both to separate source from fitting.
    download("metric_eigenvectors_column_major", plan->metric_eigenvectors, a * a);
    download("metric_eigenvalues", plan->metric_eigenvalues, a);
    download("inverse_square_root_column_major", plan->inverse_square_roots, a * a);
    download("whitened", plan->three_center, n * n * a);
    // Diagnostic-only allocation, outside every production resource claim.
    double* raw_snapshot{};
    gpu(cudaMalloc(reinterpret_cast<void**>(&raw_snapshot), n * n * a * sizeof(double)));
    check(scf::generate_cuda_density_fitting_raw_tile(plan->integral_source, 0, 0, n * n, 0, a, -1,
                                                      reinterpret_cast<void*>(plan->stream),
                                                      raw_snapshot, detail));
    download("raw", raw_snapshot, n * n * a);
    gpu(cudaFree(raw_snapshot));
    std::cout << "{\"operation\":\"fixed_density_fock\",\"nbf\":" << n << ",\"naux\":" << a
              << ",\"overlap_error\":" << overlap_error << ",\"streamed\":" << plan->streamed
              << ",\"resident_exchange_enabled\":" << plan->resident_exchange_enabled
              << ",\"raw_snapshot_device_bytes\":" << n * n * a * sizeof(double) << "}\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
