#pragma once

#include <cmath>
#include <limits>
#include <vector>

// Independent literal acceptance boundaries, shared only by host/device tests.
// Do not read the generated policy's tolerances to construct expected results.
struct EcpPolicyCase {
  double coarse, fine;
  bool derivative, accepted;
};
inline std::vector<EcpPolicyCase> ecp_policy_cases() {
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double max = std::numeric_limits<double>::max();
  std::vector<EcpPolicyCase> cases;
  for (bool derivative : {false, true}) {
    const double boundary = derivative ? 2e-8 : 2e-9;
    for (double sign : {-1.0, 1.0}) {
      cases.push_back({0, sign * std::nextafter(boundary, 0), derivative, true});
      cases.push_back({0, sign * boundary, derivative, true});
      cases.push_back({0, sign * std::nextafter(boundary, inf), derivative, false});
      cases.push_back({sign * boundary, 0, derivative, true});
      cases.push_back({sign * std::nextafter(boundary, inf), 0, derivative, false});
    }
    cases.push_back({-0.0, 0.0, derivative, true});
    cases.push_back({0, std::numeric_limits<double>::denorm_min(), derivative, true});
    cases.push_back({max, max, derivative, true});
    cases.push_back({-max, max, derivative, false});
    cases.push_back({max, -max, derivative, false});
    for (double invalid : {-inf, inf, nan}) {
      for (double other : {0.0, -inf, inf, nan}) {
        cases.push_back({invalid, other, derivative, false});
        cases.push_back({other, invalid, derivative, false});
      }
    }
  }
  // The same residual must reject a value but admit a derivative.
  cases.push_back({0, 1e-8, false, false});
  cases.push_back({0, 1e-8, true, true});
  return cases;
}
