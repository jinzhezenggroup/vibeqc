#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "api/ks_snapshot.hpp"

namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

/** Independent original-energy derivatives qualify both output spins, including
 * cross-spin correlation, empty-spin tangent directions and extreme tails. */
void independent_directions() {
  std::ifstream input(VIBEQC_SOURCE_DIR "/tests/data/xc/uks_response.tsv");
  require(bool(input), "missing UKS response fixture");
  std::size_t count = 0;
  for (std::string line; std::getline(input, line);) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream row(line);
    unsigned method;
    // The generator writes all columns as FP64 for a reproducible table.
    double method_value, rho[2], gradient[6], delta[2], delta_gradient[6], expected[8],
        magnitude[8];
    row >> method_value;
    method = static_cast<unsigned>(method_value);
    for (double& v : rho) row >> v;
    for (double& v : gradient) row >> v;
    for (double& v : delta) row >> v;
    for (double& v : delta_gradient) row >> v;
    for (double& v : expected) row >> v;
    for (double& v : magnitude) row >> v;
    require(bool(row), "malformed UKS fixture");
    double actual[8];
    require(vibeqc_xc_uks_response_batch_v1(method, rho, gradient, delta, delta_gradient, 1, actual,
                                            8) == VIBEQC_STATUS_SUCCESS,
            "valid UKS direction rejected");
    for (unsigned c = 0; c < 8; ++c) {
      // Scale roundoff by independently differentiated X/C terms so neither
      // exact cancellation nor tiny nonzero tail values get a blanket floor.
      const double tolerance = 3e-10 * std::abs(expected[c]) +
                               64 * std::numeric_limits<double>::epsilon() * magnitude[c] +
                               8 * std::numeric_limits<double>::denorm_min();
      if (!std::isfinite(actual[c]) || std::abs(actual[c] - expected[c]) > tolerance) {
        std::cerr << std::setprecision(17) << "point " << count << " component " << c << " actual "
                  << actual[c] << " expected " << expected[c] << " tolerance " << tolerance << '\n';
        throw std::runtime_error("independent UKS response mismatch");
      }
    }
    // Relabeling spins must permute both potentials, never average the result.
    std::swap(rho[0], rho[1]);
    std::swap(delta[0], delta[1]);
    for (unsigned k = 0; k < 3; ++k) {
      std::swap(gradient[k], gradient[3 + k]);
      std::swap(delta_gradient[k], delta_gradient[3 + k]);
    }
    double swapped[8];
    require(vibeqc_xc_uks_response_batch_v1(method, rho, gradient, delta, delta_gradient, 1,
                                            swapped, 8) == VIBEQC_STATUS_SUCCESS,
            "swapped UKS direction rejected");
    const unsigned permutation[8]{1, 0, 5, 6, 7, 2, 3, 4};
    for (unsigned c = 0; c < 8; ++c)
      require(std::abs(swapped[c] - actual[permutation[c]]) <=
                  1e-12 * std::abs(actual[permutation[c]]) +
                      8 * std::numeric_limits<double>::denorm_min(),
              "UKS spin permutation mismatch");
    ++count;
  }
  require(count == 48, "incomplete UKS reference table");
}

void layout_and_domain() {
  const auto call = [](unsigned method, const double* rho, const double* gradient,
                       const double* delta, const double* delta_gradient, std::size_t n,
                       double* output, std::size_t count) {
    return vibeqc_xc_uks_response_batch_v1(method, rho, gradient, delta, delta_gradient, n, output,
                                           count);
  };
  const double rho[4]{.7, .4, .3, .6}, delta[4]{.04, -.02, -.01, .03};
  const double gradient[12]{.1, -.03, .02, -.01, .06, .02, .03, .01, -.02, .04, -.02, .03};
  const double dg[12]{.03, .01, -.02, .01, .02, .03, -.02, .01, .02, -.01, -.02, .01};
  double output[16];
  for (unsigned method : {0U, 1U}) {
    require(call(method, rho, gradient, delta, dg, 2, output, 16) == VIBEQC_STATUS_SUCCESS,
            "UKS batch failed");
    for (unsigned p = 0; p < 2; ++p) {
      double r[2]{rho[p], rho[2 + p]}, d[2]{delta[p], delta[2 + p]}, g[6], direction[6], scalar[8];
      for (unsigned s = 0; s < 2; ++s)
        for (unsigned k = 0; k < 3; ++k) {
          g[3 * s + k] = gradient[(2 * s + p) * 3 + k];
          direction[3 * s + k] = dg[(2 * s + p) * 3 + k];
        }
      require(call(method, r, g, d, direction, 1, scalar, 8) == VIBEQC_STATUS_SUCCESS,
              "UKS scalar failed");
      for (unsigned c = 0; c < 8; ++c)
        require(scalar[c] == output[8 * p + c], "UKS batch layout mismatch");
    }
    double zero[6]{}, r[2]{.7, 0.}, d[2]{.1, 0.};
    require(call(method, r, zero, d, zero, 1, output, 8) == VIBEQC_STATUS_SUCCESS,
            "empty-spin tangent rejected");
    d[1] = .1;
    require(call(method, r, zero, d, zero, 1, output, 8) == VIBEQC_STATUS_NUMERICAL_FAILURE,
            "singular empty-spin normal accepted");
    d[1] = 0.;
    double bad_gradient[6]{0., 0., 0., .1, 0., 0.};
    require(call(method, r, bad_gradient, d, zero, 1, output, 8) == VIBEQC_STATUS_NUMERICAL_FAILURE,
            "empty spin gradient accepted");
    require(call(method, r, zero, d, bad_gradient, 1, output, 8) == VIBEQC_STATUS_NUMERICAL_FAILURE,
            "empty spin gradient direction accepted");
    for (double density : {0., 1e-280}) {
      r[0] = density;
      require(call(method, r, zero, zero, zero, 1, output, 8) == VIBEQC_STATUS_SUCCESS,
              "zero direction failed");
      for (unsigned c = 0; c < 8; ++c) require(output[c] == 0., "zero direction not exact");
    }
    for (double invalid :
         {-1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
      r[0] = invalid;
      require(call(method, r, zero, zero, zero, 1, output, 8) == VIBEQC_STATUS_NUMERICAL_FAILURE,
              "invalid density accepted");
    }
  }
  for (auto status : {call(2, rho, gradient, delta, dg, 2, output, 16),
                      call(0, nullptr, gradient, delta, dg, 2, output, 16),
                      call(0, rho, nullptr, delta, dg, 2, output, 16),
                      call(0, rho, gradient, nullptr, dg, 2, output, 16),
                      call(0, rho, gradient, delta, nullptr, 2, output, 16),
                      call(0, rho, gradient, delta, dg, 2, nullptr, 16),
                      call(0, rho, gradient, delta, dg, 0, output, 0),
                      call(0, rho, gradient, delta, dg, 2, output, 15),
                      call(0, rho, gradient, delta, dg,
                           std::numeric_limits<std::size_t>::max() / 8 + 1, output, 16)})
    require(status == VIBEQC_STATUS_INVALID_ARGUMENT, "malformed UKS ABI accepted");
}
}  // namespace

int main() {
  try {
    independent_directions();
    layout_and_domain();
    std::cout
        << "48 independent UKS directions, spin permutation, batch and domain checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
