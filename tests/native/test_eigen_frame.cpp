#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "scf/solver/eigen_frame.hpp"

namespace {
using Matrix = std::vector<double>;
void require(bool value, const char* detail) {
  if (!value) throw std::runtime_error(detail);
}
void checked_frames() {
  using namespace vibeqc::scf::solver;
  // An analytic generalized frame with nonidentity metric. The ordinary
  // validator never needs a production or reference eigensolver to test it.
  const Matrix f{2, 0, 0, 12}, s{2, 0, 0, 4}, values{1, 3};
  const Matrix c{1 / std::sqrt(2.0), 0, 0, .5};
  EigenFrameDiagnostic diagnostic;
  std::string detail;
  require(validate_eigen_frame(f, &s, values, c, 2, diagnostic, detail), detail.c_str());
  for (const int info : {-1, 1}) {
    diagnostic.solver_info = info;
    require(!validate_eigen_frame(f, &s, values, c, 2, diagnostic, detail),
            "nonzero solver info accepted a successful-looking frame");
  }
  diagnostic = {};
  auto bad = c;
  bad[0] *= 1.01;
  require(!validate_eigen_frame(f, &s, values, bad, 2, diagnostic, detail),
          "nonorthogonal frame passed");
  auto shifted = values;
  shifted[0] += 1e-6;
  require(!validate_eigen_frame(f, &s, shifted, c, 2, diagnostic, detail),
          "incorrect eigenvalue passed physical residual gate");
  require(!validate_eigen_frame(f, &s, Matrix{3, 1}, c, 2, diagnostic, detail),
          "unordered eigenvalues passed");
  bad[0] = std::numeric_limits<double>::quiet_NaN();
  require(!validate_eigen_frame(f, &s, values, bad, 2, diagnostic, detail),
          "nonfinite output passed");
  bad.assign(4, std::numeric_limits<double>::max());
  require(!validate_eigen_frame(f, &s, values, bad, 2, diagnostic, detail),
          "overflowing finite-input validation products passed");
  require(!validate_eigen_frame(f, &s, values, Matrix{1}, 2, diagnostic, detail),
          "truncated output passed");
  // A degenerate ordinary frame accepts rotations within its entire eigenspace.
  const double a = std::sqrt(.5);
  require(validate_eigen_frame(Matrix{2, 0, 0, 2}, nullptr, Matrix{2, 2}, Matrix{a, -a, a, a}, 2,
                               diagnostic, detail),
          "degenerate rotation was rejected");
}
}  // namespace
int main() {
  try {
    checked_frames();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
