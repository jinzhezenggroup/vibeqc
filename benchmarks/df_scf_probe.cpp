/** Bounded native RHF energy/full-force and actual final-state diagnosis for #308.
 * The input freezes physical geometry/basis and converged canonical orbitals;
 * the endpoint includes native setup, SCF, final validation and full response.
 * A Python driver records source/library/input hashes and independent checks.
 * Execute only under a finite Slurm allocation. Component journals are
 * diagnostic; keep them separate from clean endpoint measurements.
 */
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "runtime/df_progress_trace.hpp"
#include "scf/cuda_density_fitting.hpp"
#include "scf/cuda_density_fitting_integrals.hpp"
#include "scf/mean_field.hpp"

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
    std::cout << std::unitbuf;
    Dl_info library_info{};
    if (!dladdr(reinterpret_cast<void*>(&vibeqc_get_source_identity), &library_info))
      throw std::runtime_error("cannot identify the loaded native library");
    std::cout << "{\"operation\":\"identity\",\"source_identity\":"
              << std::quoted(vibeqc_get_source_identity())
              << ",\"library\":" << std::quoted(library_info.dli_fname) << "}\n";
    vibeqc::runtime::df_progress::Scope endpoint("rhf_state_probe");
    if (!std::getenv("SLURM_JOB_ID") || !std::getenv("CUDA_VISIBLE_DEVICES") || argc != 7)
      throw std::runtime_error(
          "usage inside Slurm: probe INPUT ARRAYS cold|seeded energy|forces VALUE_RESPONSE_BUDGET "
          "MAX_ITERATIONS");
    std::ifstream input(argv[1]);
    std::string magic;
    std::size_t atoms = 0, shells = 0, rank = 0;
    int representation = 0;
    input >> magic >> atoms >> shells >> representation >> rank;
    if (magic != "vibeqc-stage-v1" || !atoms || !shells)
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
    std::vector<double> coefficients(nbf * rank);
    for (auto& value : coefficients) input >> value;
    if (!input) throw std::runtime_error("truncated occupied probe input");
    // The input overlap is independently evaluated by libcint in exactly the
    // supplied shell order. Qualify conventions before using the imported D.
    std::vector<double> reference_overlap(nbf * nbf);
    for (auto& value : reference_overlap) input >> value;
    if (!input) throw std::runtime_error("missing independent overlap");
    vibeqc::integrals::IntegralData cartesian;
    check(build_cuda_one_electron_integrals(0, system, cartesian, detail, false, false), detail);
    const auto one_electron = vibeqc::integrals::transform_integrals(cartesian, system);
    double overlap_error = 0, orthogonality_error = 0;
    std::vector<double> sc(nbf * rank, 0);
    for (std::size_t i = 0; i < nbf; ++i)
      for (std::size_t j = 0; j < nbf; ++j) {
        overlap_error = std::max(overlap_error, std::abs(one_electron.overlap[i * nbf + j] -
                                                         reference_overlap[i * nbf + j]));
        for (std::size_t a = 0; a < rank; ++a)
          sc[i * rank + a] += one_electron.overlap[i * nbf + j] * coefficients[j * rank + a];
      }
    double electrons = 0;
    for (std::size_t a = 0; a < rank; ++a)
      for (std::size_t b = 0; b < rank; ++b) {
        double value = 0;
        for (std::size_t i = 0; i < nbf; ++i)
          value += coefficients[i * rank + a] * sc[i * rank + b];
        orthogonality_error = std::max(orthogonality_error, std::abs(value - (a == b ? 1.0 : 0.0)));
        if (a == b) electrons += 2 * value;
      }
    if (!std::isfinite(overlap_error) || overlap_error > 1e-10 ||
        !std::isfinite(orthogonality_error) || orthogonality_error > 1e-9)
      throw std::runtime_error("imported independent orbital/overlap qualification failed");
    std::cout << std::setprecision(17)
              << "{\"operation\":\"seed_qualification\",\"overlap_error\":" << overlap_error
              << ",\"orthogonality_error\":" << orthogonality_error
              << ",\"electron_trace\":" << electrons << "}\n";
    const std::string scf_mode = argv[3];
    const std::string property = argv[4];
    if (property != "energy" && property != "forces")
      throw std::runtime_error("invalid property selection");
    if (scf_mode == "cold" || scf_mode == "seeded") {
      std::vector<double> initial(nbf * nbf, 0.0);
      for (std::size_t i = 0; i < nbf; ++i)
        for (std::size_t j = 0; j < nbf; ++j)
          for (std::size_t a = 0; a < rank; ++a)
            initial[i * nbf + j] += 2 * coefficients[i * rank + a] * coefficients[j * rank + a];
      ScfOptions options;
      options.max_iterations = std::stoul(argv[6]);
      if (!options.max_iterations) throw std::runtime_error("zero iteration limit");
      options.energy_tolerance = 1e-12;
      options.density_tolerance = 1e-10;
      options.screening_tolerance = 1e-12;
      options.density_fitting_relative_threshold = 1e-10;
      options.density_fitting_memory_budget_bytes = std::stoull(argv[5]);
      options.compute_forces = property == "forces";
      options.export_physical_reference = true;
      options.precision_mode = VIBEQC_PRECISION_FP64;
      std::cout << "{\"operation\":\"request\",\"mode\":" << std::quoted(scf_mode)
                << ",\"property\":" << std::quoted(property)
                << ",\"df_budget_bytes\":" << options.density_fitting_memory_budget_bytes
                << ",\"max_iterations\":" << options.max_iterations << "}\n";
      const auto scf_begin = Clock::now();
      const auto result = run_rhf_density_fitting_cuda(system, system, options, 0,
                                                       scf_mode == "seeded" ? &initial : nullptr);
      std::cout << "{\"operation\":" << std::quoted(scf_mode)
                << ",\"seconds\":" << elapsed(scf_begin)
                << ",\"converged\":" << (result.converged ? "true" : "false")
                << ",\"iterations\":" << result.iterations
                << ",\"fock_builds\":" << result.fock_builds << ",\"energy\":" << result.energy
                << ",\"density_rms\":" << result.density_rms << "}\n";
      if (!result.converged) return 2;
      if (!result.reference) throw std::runtime_error("missing physical RHF export");
      const auto& ref = *result.reference;
      std::cout << "{\"operation\":\"physical_export\",\"eigen_residual\":" << ref.eigen_residual
                << ",\"commutator_residual\":" << ref.commutator_residual
                << ",\"canonical_density_drift\":" << ref.canonical_density_drift << "}\n";
      if (options.compute_forces &&
          (ref.weighted_density.size() != nbf * nbf || result.forces.size() != 3 * atoms))
        throw std::runtime_error("missing actual force weighted density or force components");
      std::ofstream arrays(argv[2], std::ios::binary);
      for (const auto* values :
           {&ref.density, &ref.overlap, &ref.hcore, &ref.fock, &ref.coefficients,
            &ref.orbital_energies, &ref.weighted_density, &result.forces})
        arrays.write(reinterpret_cast<const char*>(values->data()),
                     values->size() * sizeof(double));
      if (!arrays) throw std::runtime_error("failed physical export write");
      return 0;
    }
    throw std::runtime_error("invalid SCF seed mode");
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
