#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/dispersion/d4_reference.hpp"
#include "model/gfn2/d4.hpp"

using namespace vibeqc::xtb::detail::gfn2;
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
struct Fixture {
  std::vector<std::int64_t> offsets{0, 2, 4};
  std::vector<std::int32_t> numbers{6, 8, 6, 8};
  std::vector<double> xyz{0, 0, 0, 2.5, 0, 0, 0, 4, 0, 2.5, 4, 0};
  std::vector<double> charges{0.1, -0.1, 0.1, -0.1};
  std::vector<double> pairs = std::vector<double>(10);
  std::vector<double> cn = std::vector<double>(12);
  std::vector<double> gradient = std::vector<double>(12, 0.25);
  std::vector<double> energy = std::vector<double>(2, 123.0);
  D4Plan plan;
  D4Workspace workspace;
  D4GeometryCache cache;
  std::string error;
  std::unique_ptr<void, decltype(&std::free)> memory{nullptr, &std::free};
  Fixture() {
    require(make_d4_plan(2, 4, offsets.data(), numbers.data(), plan, error) == 0, "plan");
    memory.reset(std::aligned_alloc(kD4WorkspaceAlignment, plan.workspace_size_bytes()));
    require(memory != nullptr, "allocate");
    require(
        bind_d4_workspace(plan, memory.get(), plan.workspace_size_bytes(), workspace, error) == 0,
        "bind");
    require(update_d4_geometry_cache_cpu(plan, xyz.data(), 1, pairs.data(), pairs.size(), cn.data(),
                                         4, workspace, cache, error) == 0,
            "cache");
  }
  vibeqc_xtb_status_t run(double* output = nullptr) {
    return add_d4_two_body_gradient_cpu(plan, cache, xyz.data(), charges.data(),
                                        output ? output : gradient.data(), workspace, error);
  }
};
int main(int argc, char** argv) {
  try {
    require(argc == 2, "case required");
    const std::string test = argv[1];
    Fixture f;
    if (test == "alias_positions") {
      const auto original = f.xyz;
      require(f.run(f.xyz.data()) == VIBEQC_XTB_STATUS_INVALID_ARGUMENT,
              "position/output overlap admitted");
      require(f.xyz == original, "aliased positions modified");
    } else if (test == "alias_cache") {
      const auto original = f.cn;
      require(f.run(f.cn.data()) == VIBEQC_XTB_STATUS_INVALID_ARGUMENT,
              "cache/output overlap admitted");
      require(f.cn == original, "aliased cache modified");
    } else if (test == "nonfinite_output") {
      f.gradient[0] = std::numeric_limits<double>::quiet_NaN();
      const auto original = f.gradient;
      require(f.run() == VIBEQC_XTB_STATUS_INVALID_ARGUMENT, "nonfinite output admitted");
      require(
          std::memcmp(f.gradient.data(), original.data(), original.size() * sizeof(double)) == 0,
          "failed output modified");
    } else if (test == "late_gradient_failure") {
      const auto original = f.gradient;
      f.xyz[9] = f.xyz[6];
      require(f.run() != VIBEQC_XTB_STATUS_SUCCESS, "coincident later system admitted");
      require(f.gradient == original, "earlier gradient published before later failure");
    } else if (test == "late_energy_failure") {
      const auto original = f.energy;
      f.xyz[9] = f.xyz[6];
      require(evaluate_d4_atm_cpu(f.plan, f.cache, f.xyz.data(), f.charges.data(), f.energy.data(),
                                  f.workspace, f.error) != 0,
              "coincident later system admitted");
      require(f.energy == original, "earlier energy published before later failure");
    } else if (test == "hotloop_shared_parity") {
      std::vector<double> cached_energy(2, 0.0);
      std::vector<double> cached_dq(4, 0.0);
      require(evaluate_d4_two_body_cpu(f.plan, f.cache, f.charges.data(), cached_energy.data(),
                                       cached_dq.data(), f.workspace, f.error) == 0,
              "cached two-body evaluation failed");
      namespace shared = vibeqc::dft::dispersion;
      auto parameters = shared::gfn2_d4_parameters();
      parameters.s9 = 0.0;
      for (int system = 0; system < 2; ++system) {
        constexpr int count = 2;
        const int begin = 2 * system;
        std::vector<double> scratch(shared::d4_unbounded_workspace_elements(count));
        std::vector<double> gradient(3 * count, 0.0);
        std::vector<double> dq(count, 0.0);
        double energy[2] = {};
        require(
            shared::evaluate_d4_fixed_charge_unbounded_cpu(
                count, f.numbers.data() + begin, f.xyz.data() + 3 * begin, f.charges.data() + begin,
                parameters, shared::gfn2_d4_host_tables(), scratch.data(), scratch.size(), energy,
                gradient.data(), dq.data()) == shared::D4Status::success,
            "shared two-body evaluation failed");
        require(std::abs(cached_energy[system] - energy[0]) < 1.0e-13,
                "cached/shared two-body energy mismatch");
        for (int atom = 0; atom < count; ++atom)
          require(std::abs(cached_dq[begin + atom] - dq[atom]) < 1.0e-13,
                  "cached/shared dE/dq mismatch");
      }
    } else if (test == "per_system_cache_replay") {
      f.charges = {0.1, -0.2, 0.35, -0.25};
      const auto original_pairs = f.pairs, original_cn = f.cn;
      for (int replay = 0; replay < 2; ++replay) {
        std::vector<double> all_energy(2, 0.0), all_dq(4, 0.0);
        require(evaluate_d4_two_body_cpu(f.plan, f.cache, f.charges.data(), all_energy.data(),
                                         all_dq.data(), f.workspace, f.error) == 0,
                "cached batch reference failed");
        std::vector<double> dq(4, 91.0);
        double energy = 123.0;
        require(evaluate_d4_two_body_system_cpu(f.plan, f.cache, 1, f.charges.data(), energy,
                                                dq.data(), f.workspace, f.error) == 0,
                "cached single-system execution failed");
        require(std::isfinite(energy) && std::abs(energy - all_energy[1]) < 1e-14,
                "single-system cached energy mismatch");
        require(dq[0] == 91.0 && dq[1] == 91.0, "single-system call changed a peer output");
        for (unsigned atom = 2; atom < 4; ++atom)
          require(std::isfinite(dq[atom]) && std::abs(dq[atom] - all_dq[atom]) < 1e-14,
                  "single-system cached potential mismatch");
        require(evaluate_d4_two_body_system_cpu(f.plan, f.cache, 1, f.charges.data(), energy,
                                                nullptr, f.workspace, f.error) == 0,
                "energy-only cached execution failed");
        require(std::isfinite(energy) && std::abs(energy - all_energy[1]) < 1e-14,
                "optional-potential cached energy mismatch");
        require(f.pairs == original_pairs && f.cn == original_cn,
                "cached charge replay changed geometry or coordination cache");
        f.charges[2] += 0.2;
        f.charges[3] -= 0.2;
      }
    } else if (test == "per_system_failure_atomic") {
      std::vector<double> dq(4, 91.0);
      double energy = 123.0;
      f.charges[2] = std::numeric_limits<double>::quiet_NaN();
      require(evaluate_d4_two_body_system_cpu(f.plan, f.cache, 1, f.charges.data(), energy,
                                              dq.data(), f.workspace, f.error) != 0,
              "nonfinite cached charges were accepted");
      require(energy == 123.0 && dq == std::vector<double>(4, 91.0),
              "failed single-system call published caller output");
    } else if (test == "success") {
      const auto original = f.gradient;
      require(f.run() == 0, "valid gradient failed");
      require(f.gradient != original, "gradient not accumulated");
      for (double value : f.gradient) require(std::isfinite(value), "nonfinite gradient");
      require(evaluate_d4_atm_cpu(f.plan, f.cache, f.xyz.data(), f.charges.data(), f.energy.data(),
                                  f.workspace, f.error) == 0,
              "valid ATM failed");
    } else
      throw std::runtime_error("unknown case");
    std::cout << test << " passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
