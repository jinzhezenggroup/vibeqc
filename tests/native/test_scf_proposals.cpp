#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "core/types.hpp"
#include "molecule/basis.hpp"
#include "scf/mean_field.hpp"
#include "scf/proposals.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

vibeqc::core::System molecule(bool uhf) {
  vibeqc::core::System s;
  s.atoms = {{2, {0, 0, -0.7}}, {1, {0, 0, 0.7}}};
  s.shells = {
      {0, 0, {{6.36242139, 0.15432897}, {1.15892300, 0.53532814}, {0.31364979, 0.44463454}}},
      {1, 0, {{3.42525091, 0.15432897}, {0.62391373, 0.53532814}, {0.16885540, 0.44463454}}}};
  s.charge = uhf ? 0 : 1;
  s.multiplicity = uhf ? 2 : 1;
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(s, detail) == VIBEQC_STATUS_SUCCESS,
          "system normalization failed");
  return s;
}
}  // namespace

int main() {
  using namespace vibeqc::scf;
  try {
    for (bool uhf : {false, true}) {
      for (bool df : {false, true}) {
        const auto system = molecule(uhf);
        const auto run = [&](const ScfOptions& options, const std::vector<double>* seed = nullptr) {
          if (uhf)
            return df ? run_uhf_density_fitting(system, system, options, seed)
                      : run_uhf(system, options, seed);
          return df ? run_rhf_density_fitting(system, system, options, seed)
                    : run_rhf(system, options, seed);
        };
        ScfOptions options;
        const auto baseline = run(options);
        require(baseline.converged, "baseline failed");
        ScfSnapshot retained;
        std::uint64_t previous_generation = 0;
        for (int fault = 0; fault < 4; ++fault) {
          unsigned rejected = 0;
          ScfHooks hooks;
          hooks.propose = [&](const ScfSnapshot& s) {
            ScfProposal p;
            if (s.iteration != 1) return p;
            retained = s;
            require(s.generation != previous_generation, "generation was reused");
            previous_generation = s.generation;
            p = {ProposalRepresentation::ensemble_density, s.generation, s.iteration, s.density};
            if (fault == 0)
              for (double& x : p.density) x *= 2;
            if (fault == 1) p.density[0] = std::numeric_limits<double>::quiet_NaN();
            if (fault == 2) --p.generation;
            if (fault == 3) p.density[1] += 1;
            return p;
          };
          hooks.observe = [&](const ScfSnapshot& s, const ProposalDecision& d) {
            if (s.iteration == 1) {
              require(d.action == ProposalAction::rejected && d.trials == 0,
                      "malformed native proposal reached the operator");
              ++rejected;
            }
          };
          options.hooks = &hooks;
          const auto guarded = run(options);
          require(guarded.converged && rejected == 1, "rejection lost fallback solve");
          require(std::abs(guarded.energy - baseline.energy) < 1e-10, "fallback energy changed");
          require(guarded.fock_builds == guarded.iterations + 2, "Fock work count is wrong");
        }
        options.hooks = nullptr;
        options.strict_initial_density = true;
        auto invalid_seed = baseline.density;
        for (double& x : invalid_seed) x *= 2;
        bool rejected = false;
        try {
          run(options, &invalid_seed);
        } catch (const std::invalid_argument&) {
          rejected = true;
        }
        require(rejected, "strict seed silently rescaled a wrong electron count");
        require(!retained.density.empty(), "owned snapshot did not survive callback lifetime");
        require(run(options, &baseline.density).converged, "valid warm density was rejected");
      }
    }
#if VIBEQC_HAS_CUDA
    // Capability rejection occurs before touching the CUDA runtime. Host
    // callbacks must not disappear silently in either native GPU HF path.
    ScfHooks hooks;
    ScfOptions unsupported;
    unsupported.hooks = &hooks;
    for (bool uhf : {false, true}) {
      for (bool df : {false, true}) {
        const auto system = molecule(uhf);
        bool rejected = false;
        try {
          if (uhf) {
            if (df)
              run_uhf_density_fitting_cuda(system, system, unsupported, 0);
            else
              run_uhf_cuda(system, unsupported, 0);
          } else {
            if (df)
              run_rhf_density_fitting_cuda(system, system, unsupported, 0);
            else
              run_rhf_cuda(system, unsupported, 0);
          }
        } catch (const std::invalid_argument&) {
          rejected = true;
        }
        require(rejected, "CUDA silently ignored CPU proposal callbacks");
      }
    }
#endif
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
