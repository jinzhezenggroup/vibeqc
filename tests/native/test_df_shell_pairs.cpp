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
#include <stdexcept>
#include <vector>

#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"
#include "scf/cuda/df_shell_derivatives.cuh"

namespace {
using namespace vibeqc;
void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
void check(cudaError_t error) { require(error == cudaSuccess, cudaGetErrorString(error)); }

/** Synchronous test uploads retain every allocation until the stream is drained. */
struct Storage {
  std::vector<void*> pointers;
  ~Storage() {
    (void)cudaDeviceSynchronize();
    for (auto p : pointers) (void)cudaFree(p);
  }
  template <class T>
  T* upload(const std::vector<T>& data) {
    T* p = nullptr;
    check(cudaMalloc(reinterpret_cast<void**>(&p), data.size() * sizeof(T)));
    pointers.push_back(p);
    check(cudaMemcpy(p, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
  }
  scf::DfShellBasisView basis(const core::System& system) {
    std::vector<std::int32_t> atoms, ao_shells, ids;
    std::vector<std::int64_t> primitives{0}, aos{0};
    std::vector<std::uint8_t> counts, angular;
    std::vector<double> coefficients, exponents, terms;
    for (const auto& shell : system.shells) {
      for (const auto& expansion :
           molecule::ao_expansions(shell.angular_momentum, system.basis_representation)) {
        ao_shells.push_back(atoms.size());
        counts.push_back(expansion.size());
        for (std::size_t t = 0; t < molecule::kMaximumAoExpansionTerms; ++t) {
          if (t < expansion.size()) {
            const auto& term = expansion[t];
            for (auto l : term.component) angular.push_back(l);
            terms.push_back(term.coefficient *
                            molecule::cartesian_component_normalization(term.component));
          } else {
            angular.insert(angular.end(), 3, 0);
            terms.push_back(0);
          }
        }
      }
      atoms.push_back(shell.atom_index);
      aos.push_back(ao_shells.size());
      for (const auto& p : shell.primitives) {
        exponents.push_back(p.exponent);
        coefficients.push_back(p.coefficient);
      }
      primitives.push_back(exponents.size());
    }
    scf::DfShellBasisView result;
    for (unsigned l = 0; l < 4; ++l) {
      result.begin[l] = ids.size();
      for (std::size_t s = 0; s < system.shells.size(); ++s)
        if (system.shells[s].angular_momentum == l) ids.push_back(s);
      result.count[l] = ids.size() - result.begin[l];
    }
    result.basis = {ao_shells.size(),   upload(atoms),     upload(ao_shells),
                    upload(primitives), upload(counts),    upload(angular),
                    upload(terms),      upload(exponents), upload(coefficients)};
    result.shell_ids = upload(ids);
    result.ao_offsets = upload(aos);
    return result;
  }
};

void exercise(bool spherical_o, bool spherical_x) {
  core::System orbital;
  orbital.atoms = {{2, {.1, -.2, -.7}}, {1, {.3, .1, .8}}, {1, {-.5, .4, .2}}};
  // Equal angular classes need their own triangular indexing; different
  // classes deliberately appear in an order unrelated to their angular rank.
  orbital.shells = {{0, 1, {{.8, .8}, {1.9, -.1}}},
                    {1, 0, {{1.2, 1}}},
                    {1, 3, {{.7, 1}}},
                    {0, 2, {{.9, 1}}},
                    {1, 1, {{.6, .7}, {1.7, -.1}, {2.3, .1}}}};
  orbital.basis_representation = spherical_o ? VIBEQC_BASIS_SPHERICAL : VIBEQC_BASIS_CARTESIAN;
  auto auxiliary = orbital;
  auxiliary.basis_representation = spherical_x ? VIBEQC_BASIS_SPHERICAL : VIBEQC_BASIS_CARTESIAN;
  auxiliary.shells = {
      {2, 3, {{.8, 1}}}, {0, 0, {{1.1, .8}, {2.1, -.2}}}, {2, 2, {{.9, 1}}}, {1, 1, {{1.2, 1}}}};
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
}
}  // namespace

int main() {
  if (!std::getenv("SLURM_JOB_ID")) return 77;
  try {
    for (bool spherical_o : {false, true})
      for (bool spherical_x : {false, true}) exercise(spherical_o, spherical_x);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
