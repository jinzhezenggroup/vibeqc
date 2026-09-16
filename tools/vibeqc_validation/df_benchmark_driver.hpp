#pragma once
// Test-only driver. Native CPU Coulomb derivatives are independent of the
// generated polynomial/Rys CUDA kernels being selected.
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>

#include "df_benchmark_api.hpp"
#include "df_shell_fixture.hpp"
#include "integrals/s_integrals.hpp"

namespace vibeqc::df_benchmark {
struct Signature {
  unsigned a{}, b{}, c{};
  std::size_t pa{}, pb{}, pc{};
  std::array<unsigned long long, 4> expected{};
};
struct Panel {
  std::size_t begin{}, count{}, repeats{};
};
struct Profile {
  std::string name;
  core::System system;
  DfDerivativePairs pairs;
  std::vector<Panel> panels;
  std::vector<Signature> signatures;
};
inline void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
inline void normalize(core::System& system) {
  std::string detail;
  require(molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS,
          detail.c_str());
}
inline std::vector<Profile> read_profiles(const char* path) {
  std::ifstream input(path);
  std::string magic;
  std::size_t count = 0;
  input >> magic >> count;
  require(magic == "VQDF4041" && count > 0 && count <= 8, "invalid DF workload schema");
  std::vector<Profile> profiles(count);
  for (auto& profile : profiles) {
    std::size_t atoms = 0, shells = 0, panels = 0, signatures = 0;
    unsigned representation = 0, pairs = 0;
    input >> profile.name >> atoms >> shells >> representation >> pairs >> panels >> signatures;
    require(atoms && atoms < 10000 && shells && shells < 10000 && panels && panels < 10000 &&
                signatures && signatures < 1000 && representation <= 1 && pairs <= 2,
            "invalid bounded DF workload extents");
    profile.system.basis_representation =
        representation ? VIBEQC_BASIS_SPHERICAL : VIBEQC_BASIS_CARTESIAN;
    profile.pairs = static_cast<DfDerivativePairs>(pairs);
    profile.system.atoms.resize(atoms);
    for (auto& atom : profile.system.atoms)
      input >> atom.atomic_number >> atom.position[0] >> atom.position[1] >> atom.position[2];
    profile.system.shells.resize(shells);
    for (auto& shell : profile.system.shells) {
      std::size_t primitives = 0;
      input >> shell.atom_index >> shell.angular_momentum >> primitives;
      require(primitives && primitives <= 100 && shell.angular_momentum <= 3, "invalid shell");
      shell.primitives.resize(primitives);
      for (auto& primitive : shell.primitives) input >> primitive.exponent >> primitive.coefficient;
    }
    profile.panels.resize(panels);
    for (auto& panel : profile.panels) input >> panel.begin >> panel.count >> panel.repeats;
    profile.signatures.resize(signatures);
    for (auto& s : profile.signatures) {
      input >> s.a >> s.b >> s.c >> s.pa >> s.pb >> s.pc;
      for (auto& value : s.expected) input >> value;
    }
    require(bool(input), "truncated workload");
    normalize(profile.system);
  }
  return profiles;
}

/** Complete small shell gradients in both public representations and all
 * response layouts. Only cross-shell AB weights are populated; ordered full
 * input is folded before the symmetric/packed single-AB launch. Partial
 * auxiliary panels exercise clipping even when C is a spherical d shell.
 */
inline double oracle_error(const Candidate& candidate, const Signature& signature) {
  double maximum = 0;
  for (bool spherical : {false, true}) {
    core::System orbital;
    orbital.atoms = {{2, {.13, -.31, .24}}, {1, {-.43, .27, .51}}, {1, {.68, -.14, -.22}}};
    orbital.basis_representation = spherical ? VIBEQC_BASIS_SPHERICAL : VIBEQC_BASIS_CARTESIAN;
    const auto shell = [](unsigned atom, unsigned angular, std::size_t primitives) {
      core::Shell result{atom, angular, {}};
      for (std::size_t i = 0; i < primitives; ++i)
        result.primitives.push_back({.57 + .23 * atom + .71 * i, i ? -.17 / i : .83});
      return result;
    };
    orbital.shells = {shell(0, signature.a, signature.pa), shell(1, signature.b, signature.pb)};
    auto auxiliary = orbital;
    auxiliary.shells = {shell(2, signature.c, signature.pc)};
    normalize(orbital);
    normalize(auxiliary);
    const auto reference = integrals::build_density_fitting_integrals(orbital, auxiliary);
    const auto n = reference.nbf, a = reference.naux, packed = n * (n + 1) / 2;
    const auto na = molecule::ao_expansions(signature.a, orbital.basis_representation).size();
    std::vector<double> dense(a * n * n), folded(a * packed), expected(9), positions;
    for (std::size_t q = 0; q < a; ++q)
      for (std::size_t i = 0; i < na; ++i)
        for (std::size_t j = na; j < n; ++j) {
          const auto k = (i * n + j) * a + q;
          const double weight = .01 * std::sin(.7 * k + .3);
          dense[q * n * n + i * n + j] = weight;
          folded[q * packed + j * (j + 1) / 2 + i] = weight;
          for (unsigned axis = 0; axis < 9; ++axis)
            expected[axis] += weight * reference.three_center_derivative[axis * n * n * a + k];
        }
    Storage storage;
    auto first = storage.basis(orbital), second = first, third = storage.basis(auxiliary);
    first.shell_ids = storage.upload(std::vector<std::int32_t>{0});
    second.shell_ids = storage.upload(std::vector<std::int32_t>{1});
    std::fill(std::begin(first.count), std::end(first.count), 0);
    std::fill(std::begin(second.count), std::end(second.count), 0);
    first.begin[signature.a] = second.begin[signature.b] = 0;
    first.count[signature.a] = second.count[signature.b] = 1;
    first.primitives = signature.pa;
    second.primitives = signature.pb;
    third.primitives = signature.pc;
    for (const auto& atom : orbital.atoms)
      positions.insert(positions.end(), atom.position.begin(), atom.position.end());
    auto* r = storage.upload(positions);
    auto* w = storage.upload(dense);
    auto* wp = storage.upload(folded);
    auto* output = storage.upload(std::vector<double>(9));
    for (auto pairs :
         {DfDerivativePairs::full, DfDerivativePairs::symmetric, DfDerivativePairs::packed}) {
      check(cudaMemset(output, 0, 9 * sizeof(double)));
      const auto stride = pairs == DfDerivativePairs::packed ? packed : n * n;
      for (std::size_t begin = 0; begin < a; ++begin)
        check(candidate.launch(first, second, third, r, begin, 1,
                               (pairs == DfDerivativePairs::packed ? wp : w) + begin * stride,
                               output, nullptr, pairs, false, nullptr));
      std::array<double, 9> result{};
      check(cudaMemcpy(result.data(), output, sizeof(result), cudaMemcpyDeviceToHost));
      for (unsigned axis = 0; axis < 9; ++axis) {
        require(std::isfinite(result[axis]), "nonfinite candidate gradient");
        maximum = std::max(maximum, std::abs(result[axis] - expected[axis]));
      }
    }
  }
  return maximum;
}

inline int run(int argc, char** argv, const std::vector<Candidate>& candidates) {
  try {
    require(argc == 2, "expected one workload file");
    require(std::getenv("SLURM_JOB_ID"), "GPU benchmark requires Slurm");
    int device = 0, driver_version = 0, runtime_version = 0;
    cudaDeviceProp properties{};
    check(cudaGetDevice(&device));
    check(cudaGetDeviceProperties(&properties, device));
    check(cudaDriverGetVersion(&driver_version));
    check(cudaRuntimeGetVersion(&runtime_version));
    std::cout << "{\"kind\":\"device\",\"architecture\":\"sm_"
              << 10 * properties.major + properties.minor
              << "\",\"driver_version\":" << driver_version
              << ",\"runtime_version\":" << runtime_version << "}" << std::endl;
    const auto profiles = read_profiles(argv[1]);
    std::cout << std::setprecision(17);
    for (const auto& profile : profiles) {
      Storage storage;
      const auto basis = storage.basis(profile.system);
      const auto n = basis.basis.nbf,
                 stride = profile.pairs == DfDerivativePairs::packed ? n * (n + 1) / 2 : n * n;
      const auto groups = storage.groups(basis, profile.system);
      const auto group = [&](unsigned angular, std::size_t primitives) {
        for (const auto& g : groups)
          if (g.count[angular] && g.primitives == primitives) return g;
        throw std::runtime_error("signature absent from real basis");
      };
      std::vector<double> positions;
      for (const auto& atom : profile.system.atoms)
        positions.insert(positions.end(), atom.position.begin(), atom.position.end());
      auto* r = storage.upload(positions);
      auto* output = storage.upload(std::vector<double>(positions.size()));
      auto* counters = storage.upload(std::vector<unsigned long long>(6));
      std::size_t maximum_panel = 0;
      for (const auto& p : profile.panels) {
        require(p.count && p.begin + p.count <= n && p.repeats, "invalid panel");
        maximum_panel = std::max(maximum_panel, p.count);
      }
      // This input is a deterministic, dense external adjoint, not a changed
      // SCF density or an approximation to physical response algebra. Angular,
      // primitive, geometry, representation and panel frequencies are real.
      std::vector<double> host_weights(maximum_panel * stride);
      for (std::size_t k = 0; k < host_weights.size(); ++k)
        host_weights[k] = .001 * (1 + double(k % 97) / 97);
      auto* weights = storage.upload(host_weights);
      cudaEvent_t start{}, stop{};
      check(cudaEventCreate(&start));
      check(cudaEventCreate(&stop));
      for (const auto& signature : profile.signatures) {
        const auto first = group(signature.a, signature.pa),
                   second = group(signature.b, signature.pb);
        const bool triangle = profile.pairs != DfDerivativePairs::full &&
                              signature.a == signature.b && signature.pa == signature.pb;
        std::vector<DfShellBasisView> thirds;
        for (const auto& panel : profile.panels) {
          auto third = group(signature.c, signature.pc);
          std::vector<std::int32_t> ids;
          std::size_t offset = 0;
          for (std::size_t s = 0; s < profile.system.shells.size(); ++s) {
            const auto& shell = profile.system.shells[s];
            const auto width =
                molecule::ao_expansions(shell.angular_momentum, profile.system.basis_representation)
                    .size();
            if (shell.angular_momentum == signature.c && shell.primitives.size() == signature.pc &&
                offset < panel.begin + panel.count && offset + width > panel.begin)
              ids.push_back(s);
            offset += width;
          }
          if (!ids.empty()) third.shell_ids = storage.upload(ids);
          third.begin[signature.c] = 0;
          third.count[signature.c] = ids.size();
          thirds.push_back(third);
        }
        for (const auto& candidate : candidates) {
          if (candidate.a != signature.a || candidate.b != signature.b ||
              candidate.c != signature.c)
            continue;
          const auto error = oracle_error(candidate, signature);
          std::array<unsigned long long, 6> work{};
          std::array<double, 5> milliseconds{};
          for (std::size_t p = 0; p < profile.panels.size(); ++p) {
            const auto& panel = profile.panels[p];
            const auto launch = [&](unsigned long long* sink) {
              check(candidate.launch(first, second, thirds[p], r, panel.begin, panel.count, weights,
                                     output, sink, profile.pairs, triangle, nullptr));
            };
            check(cudaMemset(output, 0, positions.size() * sizeof(double)));
            check(cudaMemset(counters, 0, 6 * sizeof(unsigned long long)));
            launch(counters);
            std::array<unsigned long long, 6> local{};
            check(cudaMemcpy(local.data(), counters, sizeof(local), cudaMemcpyDeviceToHost));
            for (unsigned k = 0; k < 6; ++k) work[k] += local[k] * panel.repeats;
            launch(nullptr);
            check(cudaDeviceSynchronize());
            for (unsigned sample = 0; sample < 5; ++sample) {
              check(cudaMemset(output, 0, positions.size() * sizeof(double)));
              check(cudaEventRecord(start));
              launch(nullptr);
              check(cudaEventRecord(stop));
              check(cudaEventSynchronize(stop));
              float elapsed = 0;
              check(cudaEventElapsedTime(&elapsed, start, stop));
              milliseconds[sample] += elapsed * panel.repeats;
            }
          }
          std::cout << "{\"profile\":\"" << profile.name << "\",\"candidate\":\"" << candidate.key
                    << "\",\"primitives\":[" << signature.pa << "," << signature.pb << ","
                    << signature.pc << "],\"maximum_error\":" << error
                    << ",\"numerical_passed\":" << (error <= 1e-10 ? "true" : "false")
                    << ",\"work\":{\"shell_tasks\":" << work[0]
                    << ",\"active_shell_tasks\":" << work[1]
                    << ",\"primitive_products\":" << work[3]
                    << ",\"active_component_products\":" << work[4] << "},\"milliseconds\":[";
          for (unsigned sample = 0; sample < 5; ++sample)
            std::cout << (sample ? "," : "") << milliseconds[sample];
          std::cout << "]}" << std::endl;
        }
      }
      check(cudaEventDestroy(start));
      check(cudaEventDestroy(stop));
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
}  // namespace vibeqc::df_benchmark
