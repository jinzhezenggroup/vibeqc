#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "dft/xc_point.hpp"

#ifndef VIBEQC_TEST_POINT_EVALUATE
#define VIBEQC_TEST_POINT_EVALUATE vibeqc::dft::point::evaluate
#endif

int main() {
  try {
    std::ifstream input(VIBEQC_SOURCE_DIR "/tests/data/xc/scf_domain.tsv");
    if (!input) throw std::runtime_error("missing independent SCF domain fixture");
    std::string line;
    std::size_t count = 0;
    while (std::getline(input, line)) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream row(line);
      int pbe, oracle;
      double rho[2], gradient[2][3], expected[9];
      row >> pbe >> oracle >> rho[0] >> rho[1];
      for (auto& spin : gradient)
        for (double& x : spin) row >> x;
      for (double& x : expected) row >> x;
      if (!row) throw std::runtime_error("malformed SCF domain fixture");
      const auto result = VIBEQC_TEST_POINT_EVALUATE(pbe != 0, rho, gradient);
      const double actual[]{result.energy,         result.rho[0],         result.rho[1],
                            result.gradient[0][0], result.gradient[0][1], result.gradient[0][2],
                            result.gradient[1][0], result.gradient[1][1], result.gradient[1][2]};
      if (!result.valid) throw std::runtime_error("valid SCF domain input was rejected");
      for (unsigned i = 0; i < 9; ++i) {
        // Relative gates on each potential coefficient prevent a tiny absolute
        // energy tolerance from hiding incorrect low-density derivatives.
        const double tolerance = 5.0e-10 * std::abs(expected[i]) + 1.0e-322;
        if (!std::isfinite(actual[i]) || std::abs(actual[i] - expected[i]) > tolerance) {
          std::cerr << "point " << count << " component " << i << " actual " << actual[i]
                    << " expected " << expected[i] << " oracle " << oracle << '\n';
          throw std::runtime_error("SCF domain energy/potential reference mismatch");
        }
      }
      ++count;
    }
    if (count != 97) throw std::runtime_error("incomplete SCF domain fixture");
    for (bool pbe : {false, true}) {
      double rho[2]{}, gradient[2][3]{};
      const auto vacuum = VIBEQC_TEST_POINT_EVALUATE(pbe, rho, gradient);
      if (!vacuum.valid || vacuum.energy != 0.0 || vacuum.rho[0] != 0.0 || vacuum.rho[1] != 0.0)
        throw std::runtime_error("incorrect analytic vacuum limit");
      rho[0] = -1.0e-30;
      if (VIBEQC_TEST_POINT_EVALUATE(pbe, rho, gradient).valid)
        throw std::runtime_error("negative density was clipped");
      rho[0] = 0.0;
      gradient[0][0] = 1.0e-30;
      if (VIBEQC_TEST_POINT_EVALUATE(pbe, rho, gradient).valid)
        throw std::runtime_error("vacuum with nonzero gradient was accepted");
      gradient[0][0] = 0.0;
      rho[0] = std::numeric_limits<double>::infinity();
      if (VIBEQC_TEST_POINT_EVALUATE(pbe, rho, gradient).valid)
        throw std::runtime_error("infinite density was accepted");
    }
    std::cout << count << " independent LDA/PBE SCF-domain E/V points passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
