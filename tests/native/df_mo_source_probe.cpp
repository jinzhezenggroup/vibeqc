// Standalone layout/work probe: Python supplies independent reference inputs.
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "df_mo_source_generated.hpp"
#include "tensor/cpu_linalg.hpp"

int main(int argc, char** argv) {
  using namespace generativeqc;
  std::size_t n{}, q{};
  if (!(std::cin >> n >> q)) return 1;
  const auto work = posthf::generated::df_mo_source_work(n, q);
  if (argc == 2 && std::string_view(argv[1]) == "--work") {
    std::cout << work.raw_values << ' ' << work.source_rows << ' ' << work.gemms << ' '
              << work.contraction_summands << '\n';
    return 0;
  }
  std::vector<double> raw(work.raw_values), coefficients(n * n), root(q * q);
  for (auto* values : {&raw, &coefficients, &root})
    for (auto& value : *values)
      if (!(std::cin >> value)) return 2;
  std::vector<double> first(work.raw_values), transformed(work.raw_values);
  std::size_t reads = 0, gemms = 0, summands = 0;
  auto read = [&](std::size_t mu) {
    if (mu != reads++) throw std::runtime_error("source rows reordered or repeated");
    return raw.data() + mu * n * q;
  };
  auto gemm = [&](char ta, char tb, std::size_t m, std::size_t columns, std::size_t k,
                  const double* a, const double* b, double* c) {
    ++gemms;
    summands += m * columns * k;
    tensor::cpu_gemm(tb, ta, columns, m, k, b, a, c);
  };
  posthf::generated::transform_df_mo_source(n, q, coefficients.data(), root.data(), first.data(),
                                            transformed.data(), read, gemm);
  if (reads != work.source_rows || gemms != work.gemms || summands != work.contraction_summands)
    throw std::runtime_error("source work model differs from executed contractions");
  std::cout << reads << ' ' << gemms << ' ' << summands << '\n' << std::setprecision(17);
  for (const auto* values : {&transformed, &first}) {
    for (const auto value : *values) std::cout << value << ' ';
    std::cout << '\n';
  }
  // Reject impossible shapes before reading the source or invoking the backend.
  auto reject = [&](std::size_t orbitals, std::size_t auxiliaries) {
    try {
      posthf::generated::transform_df_mo_source(orbitals, auxiliaries, nullptr, nullptr, nullptr,
                                                nullptr, read, gemm);
    } catch (const std::invalid_argument&) {
      return;
    } catch (const std::overflow_error&) {
      return;
    }
    throw std::runtime_error("invalid shape reached source execution");
  };
  reject(0, q);
  reject(n, 0);
  reject(std::numeric_limits<std::size_t>::max(), q);
  if (reads != work.source_rows || gemms != work.gemms)
    throw std::runtime_error("invalid shape performed work");
  // Source failures stop the schedule; the owner must discard partial buffers.
  const auto old_gemms = gemms;
  try {
    posthf::generated::transform_df_mo_source(
        n, q, coefficients.data(), root.data(), first.data(), transformed.data(),
        [](std::size_t) -> const double* { throw std::runtime_error("source failure"); }, gemm);
    return 3;
  } catch (const std::runtime_error&) {
    if (gemms != old_gemms) return 4;
  }
}
