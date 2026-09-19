/** Independent CPU derivative oracle for CUDA shell-pair packing.
 * Deliberately nonsymmetric sparse adjoints, interleaved shell classes, mixed
 * public AO representations and split auxiliary shells exercise the layout
 * contract independently of the HF response producer. Run only through Slurm.
 */
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../tools/vibeqc_validation/df_shell_fixture.hpp"
#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "scf/cuda/df_shell_derivatives.cuh"

namespace {
using namespace vibeqc;
void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
void check(cudaError_t error) { require(error == cudaSuccess, cudaGetErrorString(error)); }

void select_schedule(unsigned variant) {
  const char* schedules[] = {"warp", "packed", "compact"};
  // The direct launch API receives the variant explicitly. Prevent automatic
  // production policy from overriding it so all three schedules are exercised.
  require(setenv("VIBEQC_DF_SHELL_SCHEDULE", schedules[variant], 1) == 0,
          "cannot select the explicit test schedule");
}

using df_benchmark::Storage;

void exercise(bool spherical_o, bool spherical_x, bool many_signatures = false) {
  core::System orbital;
  orbital.atoms = {{2, {.1, -.2, -.7}}, {1, {.3, .1, .8}}, {1, {-.5, .4, .2}}};
  // Equal angular classes need their own triangular indexing; different
  // classes deliberately appear in an order unrelated to their angular rank.
  orbital.shells = {{0, 1, {{.8, .8}, {1.9, -.1}}},
                    {1, 0, {{1.2, 1}}},
                    {1, 3, {{.7, 1}}},
                    {0, 2, {{.9, 1}}},
                    {1, 1, {{.6, .7}, {1.7, -.1}, {2.3, .1}}},
                    {2, 0, {{.75, 1}}}};
  orbital.basis_representation = spherical_o ? VIBEQC_BASIS_SPHERICAL : VIBEQC_BASIS_CARTESIAN;
  auto auxiliary = orbital;
  auxiliary.basis_representation = spherical_x ? VIBEQC_BASIS_SPHERICAL : VIBEQC_BASIS_CARTESIAN;
  auxiliary.shells = {
      {2, 3, {{.8, 1}}}, {0, 0, {{1.1, .8}, {2.1, -.2}}}, {2, 2, {{.9, 1}}}, {1, 1, {{1.2, 1}}}};
  if (many_signatures) {
    // More than 24 same-class signatures must flush multiple bounded packets.
    // The independent full-domain oracle detects dropped/reused packet tails.
    orbital.shells.clear();
    for (unsigned nprim = 1; nprim <= 7; ++nprim) {
      // Shell fields are atom index, angular momentum, then primitives: all
      // seven signatures are S shells, spread across three physical atoms.
      core::Shell shell{nprim % 3, 0, {}};
      for (unsigned p = 0; p < nprim; ++p) shell.primitives.push_back({.6 + .2 * p, 1.0 / (p + 1)});
      orbital.shells.push_back(shell);
    }
    auxiliary.shells = {{0, 0, {{.8, 1}}}, {2, 0, {{.7, .7}, {1.3, .3}}}};
  }
  std::string detail;
  for (auto* system : {&orbital, &auxiliary})
    require(molecule::validate_and_normalize(*system, detail) == VIBEQC_STATUS_SUCCESS,
            detail.c_str());
  const auto raw = integrals::build_density_fitting_integrals(orbital, auxiliary);
  const auto n = raw.nbf, a = raw.naux, packed_size = n * (n + 1) / 2;
  std::vector<double> dense(a * n * n), packed(a * packed_size), expected(9);
  for (std::size_t p = 0; p < a; ++p)
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) {
        const auto k = (i * n + j) * a + p;
        const double w = k % 5 ? 0 : .01 * std::sin(.7 * k + .3);
        dense[p * n * n + i * n + j] = w;
        const auto hi = std::max(i, j), lo = std::min(i, j);
        packed[p * packed_size + hi * (hi + 1) / 2 + lo] += w;
        for (std::size_t c = 0; c < expected.size(); ++c)
          expected[c] += w * raw.three_center_derivative[c * n * n * a + k];
      }
  Storage storage;
  const auto o = storage.basis(orbital), x = storage.basis(auxiliary);
  std::vector<double> positions;
  for (const auto& atom : orbital.atoms)
    positions.insert(positions.end(), atom.position.begin(), atom.position.end());
  const auto* r = storage.upload(positions);
  const auto* w = storage.upload(dense);
  const auto* wp = storage.upload(packed);
  auto* output = storage.upload(std::vector<double>(9));
  auto* work = storage.upload(std::vector<unsigned long long>(6));
  for (unsigned variant = 0; variant < 3; ++variant)
    for (const auto cap : {std::size_t{1}, std::size_t{7}, a})
      for (const auto pairs : {scf::DfDerivativePairs::full, scf::DfDerivativePairs::symmetric,
                               scf::DfDerivativePairs::packed}) {
        select_schedule(variant);
        check(cudaMemset(output, 0, 9 * sizeof(double)));
        check(cudaMemset(work, 0, 6 * sizeof(unsigned long long)));
        const bool compressed = pairs == scf::DfDerivativePairs::packed;
        const auto stride = compressed ? packed_size : n * n;
        for (std::size_t begin = 0; begin < a; begin += cap)
          check(scf::launch_df_shell_derivative_panel(o, x, r, begin, std::min(cap, a - begin),
                                                      (compressed ? wp : w) + begin * stride,
                                                      output, work, nullptr, true, variant, pairs));
        std::vector<double> actual(9);
        std::array<unsigned long long, 6> counters;
        check(cudaMemcpy(actual.data(), output, 9 * sizeof(double), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(counters.data(), work, sizeof(counters), cudaMemcpyDeviceToHost));
        for (std::size_t c = 0; c < actual.size(); ++c)
          require(std::abs(actual[c] - expected[c]) < 2e-9,
                  "shell-pair derivative oracle mismatch");
        require(counters[5] == a * stride, "incorrect public weight load count");
        if (cap == a) {
          const auto ns = orbital.shells.size();
          const auto count = pairs == scf::DfDerivativePairs::full ? ns * ns : ns * (ns + 1) / 2;
          require(counters[0] == count * auxiliary.shells.size(), "incorrect shell-triple domain");
        }
      }

  // Multi-shell signature slices exercise same-group triangles, cross-group
  // rectangles, full ordered traversal, sparse nonsymmetric weights, and split
  // auxiliary panels. No production grouping helper is used by this oracle.
  const auto os = storage.groups(o, orbital), xs = storage.groups(x, auxiliary);
  if (many_signatures) {
    require(os.size() == 7 && xs.size() == 2, "packet-flush fixture lost primitive signatures");
    for (const auto& group : os) require(group.count[0] == 1, "orbital signature must be S");
    for (const auto& group : xs) require(group.count[0] == 1, "auxiliary signature must be S");
    // Even triangular modes submit 28*2=56 SSS slices; full mode submits
    // 49*2=98. Both must reuse the 24-row diagnostic buffer within a panel.
    require(os.size() * (os.size() + 1) / 2 * xs.size() > scf::DfShellDiagnostics::packet_capacity,
            "every pair mode must exceed the diagnostic packet capacity");
  }
  using Work = scf::DfShellWork;
  using Counts = std::array<unsigned long long, scf::DfShellDiagnostics::metrics>;
  using Signature = std::array<std::size_t, 6>;
  std::map<Signature, Counts> detailed;
  std::vector<unsigned long long> diagnostic_host(scf::DfShellDiagnostics::elements);
  scf::DfShellDiagnostics diagnostics{
      storage.upload(diagnostic_host), diagnostic_host.data(),
      [](unsigned la, unsigned lb, unsigned lc, std::size_t pa, std::size_t pb, std::size_t pc,
         std::span<const unsigned long long> values, void* context) {
        auto& records = *static_cast<std::map<Signature, Counts>*>(context);
        auto& counts = records[{la, lb, lc, pa, pb, pc}];
        for (unsigned metric = 0; metric < values.size(); ++metric)
          counts[metric] += values[metric];
      },
      &detailed};
  for (bool packet : {false, true})
    for (unsigned variant = 0; variant < 3; ++variant)
      for (const auto cap : {std::size_t{1}, std::size_t{7}, a})
        for (const auto pairs : {scf::DfDerivativePairs::full, scf::DfDerivativePairs::symmetric,
                                 scf::DfDerivativePairs::packed}) {
          select_schedule(variant);
          check(cudaMemset(output, 0, 9 * sizeof(double)));
          check(cudaMemset(work, 0, 6 * sizeof(unsigned long long)));
          const bool compressed = pairs == scf::DfDerivativePairs::packed;
          const auto stride = compressed ? packed_size : n * n;
          // Exercise actual counters on both launchers, sparse weights, split
          // auxiliary shells, all pair modes and both public representations.
          const bool inspect_work = variant == 2 && cap == 7;
          detailed.clear();
          auto* sink = inspect_work ? &diagnostics : nullptr;
          if (packet) {
            for (std::size_t begin = 0; begin < a; begin += cap)
              check(scf::launch_df_shell_derivative_packets(
                  os, xs, r, begin, std::min(cap, a - begin),
                  (compressed ? wp : w) + begin * stride, output, work, nullptr, true, variant,
                  pairs, sink));
          } else {
            for (std::size_t begin = 0; begin < a; begin += cap)
              for (std::size_t ia = 0; ia < os.size(); ++ia)
                for (std::size_t ib = 0; ib < os.size(); ++ib) {
                  if (pairs != scf::DfDerivativePairs::full && ia < ib) continue;
                  for (const auto& third : xs)
                    check(scf::launch_df_shell_derivative_group(
                        os[ia], os[ib], third, r, begin, std::min(cap, a - begin),
                        (compressed ? wp : w) + begin * stride, output, work, nullptr, true,
                        variant, pairs, pairs != scf::DfDerivativePairs::full && ia == ib, sink));
                }
          }
          std::vector<double> actual(9);
          std::array<unsigned long long, 6> counters;
          check(cudaMemcpy(actual.data(), output, 9 * sizeof(double), cudaMemcpyDeviceToHost));
          check(cudaMemcpy(counters.data(), work, sizeof(counters), cudaMemcpyDeviceToHost));
          for (std::size_t c = 0; c < actual.size(); ++c)
            require(std::abs(actual[c] - expected[c]) < 2e-9,
                    "primitive-signature derivative oracle mismatch");
          require(counters[5] == a * stride, "incorrect grouped public weight load count");
          if (inspect_work) {
            std::map<Signature, unsigned long long> expected_visits;
            // Reconstruct the signature domain from host shells, independently
            // of device packet indexing and the diagnostic observer.
            for (std::size_t sa = 0; sa < orbital.shells.size(); ++sa)
              for (std::size_t sb = 0; sb < orbital.shells.size(); ++sb) {
                const auto& first = orbital.shells[sa];
                const auto& second = orbital.shells[sb];
                const auto ka = std::pair(first.angular_momentum, first.primitives.size());
                const auto kb = std::pair(second.angular_momentum, second.primitives.size());
                if (pairs != scf::DfDerivativePairs::full && (ka < kb || (ka == kb && sa < sb)))
                  continue;
                for (const auto& third : auxiliary.shells)
                  expected_visits[{first.angular_momentum, second.angular_momentum,
                                   third.angular_momentum, first.primitives.size(),
                                   second.primitives.size(), third.primitives.size()}] +=
                      (a + cap - 1) / cap;
              }
            Counts totals{};
            for (const auto& [signature, observed] : detailed) {
              const auto get = [&](Work kind) { return observed[static_cast<unsigned>(kind)]; };
              require(get(Work::shell_tasks) == expected_visits.at(signature),
                      "diagnostic signature task domain differs from independent host shells");
              const auto primitive_count = get(Work::primitive_products);
              require(primitive_count == get(Work::active_shell_tasks) * signature[3] *
                                             signature[4] * signature[5],
                      "diagnostic primitive count differs from independent host signatures");
              // Production admission no longer depends on 384/768 AO counts.
              // Every active primitive executes exactly one selected lowering;
              // the explicit legacy pass independently exercises polynomial work.
              const auto rys = get(Work::rys_evaluations);
              const auto boys = get(Work::boys_evaluations);
              require(primitive_count == boys + rys &&
                          boys == get(Work::boys_series) + get(Work::boys_large_argument),
                      "Boys/Rys branch counts do not conserve primitive work");
              if (rys) {
                const auto roots = (signature[0] + signature[1] + signature[2] + 1) / 2 + 1;
                require(rys == primitive_count && get(Work::rys_roots) == roots * rys &&
                            get(Work::recurrence_states) >= get(Work::rys_roots),
                        "Rys root/recurrence counts do not conserve primitive work");
              } else {
                require(get(Work::rys_roots) == 0 && get(Work::recurrence_states) == 0,
                        "polynomial execution reported Rys work");
              }
              require(get(Work::boys_small_argument) <= get(Work::boys_series) &&
                          get(Work::boys_series_iterations) >= get(Work::boys_series) &&
                          get(Work::boys_series_iterations) <= 179 * get(Work::boys_series),
                      "Boys dynamic series counters are outside their exact loop bounds");
              require(get(Work::gradient_atomics_a) == 3 * get(Work::active_shell_tasks) &&
                          get(Work::gradient_atomics_b) == 3 * get(Work::active_shell_tasks) &&
                          get(Work::gradient_atomics_c) == 3 * get(Work::active_shell_tasks) &&
                          get(Work::gradient_atomics_shared_atom) +
                                  get(Work::gradient_atomics_distinct_atom) ==
                              9 * get(Work::active_shell_tasks),
                      "gradient atomic channels or shared-atom classification differ");
              require(get(Work::expansion_term_products) == get(Work::folding_shared_atomics),
                      "legacy folding must count one shared atomic per expansion product");
              for (unsigned metric = 0; metric < totals.size(); ++metric)
                totals[metric] += observed[metric];
            }
            require(detailed.size() == expected_visits.size(), "missing diagnostic signature");
            const std::array mapping{Work::shell_tasks,
                                     Work::active_shell_tasks,
                                     Work::public_nonzero_weights,
                                     Work::primitive_products,
                                     Work::active_component_products,
                                     Work::public_weight_loads};
            for (unsigned metric = 0; metric < mapping.size(); ++metric)
              require(totals[static_cast<unsigned>(mapping[metric])] == counters[metric],
                      "detailed work differs from the existing executed-work counters");
          }
          if (cap == a) {
            const auto ns = orbital.shells.size();
            const auto count = pairs == scf::DfDerivativePairs::full ? ns * ns : ns * (ns + 1) / 2;
            require(counters[0] == count * auxiliary.shells.size(),
                    "incorrect grouped task domain");
          }
        }
}
}  // namespace

int main() {
  if (!std::getenv("SLURM_JOB_ID")) return 77;
  try {
    for (const char* policy : {"legacy", "auto"}) {
      require(setenv("VIBEQC_DF_SHELL_POLICY", policy, 1) == 0, "cannot select the test policy");
      for (bool spherical_o : {false, true})
        for (bool spherical_x : {false, true}) exercise(spherical_o, spherical_x);
      exercise(false, false, true);
    }
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
