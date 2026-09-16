#pragma once
// Shared test-only basis uploads, independent of production packet traversal.
#include <map>
#include <stdexcept>
#include <vector>

#include "molecule/basis.hpp"
#include "scf/cuda/df_shell_derivatives.cuh"

namespace vibeqc::df_benchmark {
inline void check(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
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
  /** Build actual multi-shell signatures independently of production metadata. */
  std::vector<scf::DfShellBasisView> groups(const scf::DfShellBasisView& base,
                                            const core::System& system) {
    std::map<std::pair<unsigned, std::size_t>, std::vector<std::int32_t>> ids;
    for (std::size_t i = 0; i < system.shells.size(); ++i) {
      const auto& shell = system.shells[i];
      ids[{shell.angular_momentum, shell.primitives.size()}].push_back(i);
    }
    std::vector<scf::DfShellBasisView> result;
    for (const auto& [signature, shells] : ids) {
      scf::DfShellBasisView view;
      view.basis = base.basis;
      view.shell_ids = upload(shells);
      view.ao_offsets = base.ao_offsets;
      view.count[signature.first] = shells.size();
      view.primitives = signature.second;
      result.push_back(view);
    }
    return result;
  }
};

}  // namespace vibeqc::df_benchmark
