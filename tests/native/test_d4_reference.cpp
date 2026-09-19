#include "d4_test_cases.hpp"
using namespace d4_tests;
struct CpuEval {
  Result operator()(const Molecule& m, const D4Parameters& p) const {
    const int n = static_cast<int>(m.z.size());
    Result r(n);
    std::vector<double> w(d4_workspace_elements(n));
    r.status = evaluate_d4_fixed_charge(n, m.z.data(), m.xyz.data(), m.q.data(), p,
                                        gfn2_d4_host_tables(), w.data(), w.size(), r.energy.data(),
                                        r.gradient.data(), r.dq.data());
    return r;
  }
};
int main() {
  CpuEval eval;
  if (run_cases(eval)) return 1;
  auto m = fixture();
  auto p = gfn2_d4_parameters();
  std::vector<double> scratch(d4_workspace_elements(4));
  Result out(4);
  const auto shortage = evaluate_d4_fixed_charge(
      4, m.z.data(), m.xyz.data(), m.q.data(), p, gfn2_d4_host_tables(), scratch.data(),
      scratch.size() - 1, out.energy.data(), out.gradient.data(), out.dq.data());
  if (shortage != D4Status::invalid_argument || out.energy[0] != 17) return 2;
  const auto alias = evaluate_d4_fixed_charge(4, m.z.data(), m.xyz.data(), m.q.data(), p,
                                              gfn2_d4_host_tables(), scratch.data(), scratch.size(),
                                              out.energy.data(), m.xyz.data(), out.dq.data());
  if (alias != D4Status::invalid_argument || m.xyz != fixture().xyz || out.energy[0] != 17)
    return 3;
  const auto limit =
      evaluate_d4_fixed_charge(257, nullptr, nullptr, nullptr, p, gfn2_d4_host_tables(), nullptr, 0,
                               out.energy.data(), nullptr, nullptr);
  if (limit != D4Status::unsupported || out.energy[0] != 17) return 4;
  return 0;
}
