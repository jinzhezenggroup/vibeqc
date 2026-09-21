#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/gfn2/d4.hpp"

using namespace xtbloom::detail::gfn2;
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
  xtbloom_status_t run(double* output = nullptr) {
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
      require(f.run(f.xyz.data()) == XTBLOOM_STATUS_INVALID_ARGUMENT,
              "position/output overlap admitted");
      require(f.xyz == original, "aliased positions modified");
    } else if (test == "alias_cache") {
      const auto original = f.cn;
      require(f.run(f.cn.data()) == XTBLOOM_STATUS_INVALID_ARGUMENT,
              "cache/output overlap admitted");
      require(f.cn == original, "aliased cache modified");
    } else if (test == "nonfinite_output") {
      f.gradient[0] = std::numeric_limits<double>::quiet_NaN();
      const auto original = f.gradient;
      require(f.run() == XTBLOOM_STATUS_INVALID_ARGUMENT, "nonfinite output admitted");
      require(
          std::memcmp(f.gradient.data(), original.data(), original.size() * sizeof(double)) == 0,
          "failed output modified");
    } else if (test == "late_gradient_failure") {
      const auto original = f.gradient;
      f.xyz[9] = f.xyz[6];
      require(f.run() != XTBLOOM_STATUS_SUCCESS, "coincident later system admitted");
      require(f.gradient == original, "earlier gradient published before later failure");
    } else if (test == "late_energy_failure") {
      const auto original = f.energy;
      f.xyz[9] = f.xyz[6];
      require(evaluate_d4_atm_cpu(f.plan, f.cache, f.xyz.data(), f.charges.data(), f.energy.data(),
                                  f.workspace, f.error) != 0,
              "coincident later system admitted");
      require(f.energy == original, "earlier energy published before later failure");
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
