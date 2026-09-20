#include <array>
#include <cmath>
#include <iostream>

#include "tensor/symmetric_matrix_function.hpp"
using namespace vibeqc::tensor;
int main() {
  const std::array<double, 1> q{1.0};
  const std::array<std::uint8_t, 1> keep{1};
  int failures = 0;
  for (const auto function :
       {SymmetricMatrixFunction::inverse_sqrt, SymmetricMatrixFunction::pseudoinverse}) {
    const bool inverse = function == SymmetricMatrixFunction::pseudoinverse;
    for (const bool small : {false, true}) {
      const int power = inverse ? (small ? -600 : 800) : (small ? -700 : 1000);
      const int seed_power = inverse ? (small ? -600 : 900) : (small ? -500 : 800);
      const int derivative_power = inverse ? 2 * power : 3 * power / 2 + 1;
      const auto expected = std::ldexp(-1.0, seed_power - derivative_power);
      try {
        const auto actual = symmetric_matrix_function_vjp(
            std::array<double, 1>{std::ldexp(1.0, power)}, q, keep,
            std::array<double, 1>{std::ldexp(1.0, seed_power)}, function, 0.0);
        if (actual[0] != expected) {
          ++failures;
          std::cerr << actual[0] << " != " << expected << '\n';
        }
      } catch (const std::exception& error) {
        ++failures;
        std::cerr << error.what() << "; expected " << expected << '\n';
      }
      try {
        const auto actual = symmetric_matrix_function_vjp(
            std::array<double, 2>{0.0, std::ldexp(1.0, power)},
            std::array<double, 4>{1.0, 0.0, 0.0, 1.0}, std::array<std::uint8_t, 2>{0, 1},
            std::array<double, 4>{0.0, std::ldexp(1.0, seed_power), std::ldexp(1.0, seed_power),
                                  0.0},
            function, 0.0);
        const auto cross_expected =
            std::ldexp(1.0, seed_power - derivative_power + (inverse ? 0 : 1));
        if (actual[1] != cross_expected || actual[2] != cross_expected) {
          ++failures;
          std::cerr << "cross " << actual[1] << " != " << cross_expected << '\n';
        }
      } catch (const std::exception& error) {
        ++failures;
        std::cerr << "cross " << error.what() << '\n';
      }
    }
  }
  return failures;
}
