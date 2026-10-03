// Standalone layout/work probe: Python supplies independent reference inputs.
#include <algorithm>
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
  const auto response_work = posthf::generated::df_mo_source_response_work(n, q);
  if (argc == 2 && std::string_view(argv[1]) == "--response-work") {
    std::cout << response_work.source_rows << ' ' << response_work.raw_values << ' '
              << response_work.output_rows << ' ' << response_work.output_values << ' '
              << response_work.gemms << ' ' << response_work.contraction_summands << ' '
              << response_work.scratch_values << '\n';
    return 0;
  }
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
  if (argc == 2 && std::string_view(argv[1]) == "--response") {
    std::vector<double> bar_b(work.raw_values), bar_a(work.raw_values), bar_c(n * n, -999),
        bar_root(q * q, -777), raw_row(n * q), bar_row(n * q);
    for (auto& value : bar_b)
      if (!(std::cin >> value)) return 2;
    std::size_t reads = 0, consumed = 0, gemms = 0, summands = 0;
    auto read = [&](std::size_t mu) {
      if (mu != reads++ % n) throw std::runtime_error("response source row order changed");
      // Reuse one row buffer to catch accidental dependence on a resident raw A.
      std::copy_n(raw.data() + mu * n * q, n * q, raw_row.data());
      return raw_row.data();
    };
    auto consume = [&](std::size_t mu, const double* values) {
      if (mu != consumed++) throw std::runtime_error("response output row order changed");
      std::copy_n(values, n * q, bar_a.data() + mu * n * q);
    };
    auto gemm = [&](char ta, char tb, std::size_t m, std::size_t columns, std::size_t k,
                    const double* a, const double* b, double* c, bool accumulate) {
      ++gemms;
      summands += m * columns * k;
      tensor::cpu_gemm(tb, ta, columns, m, k, b, a, c, 1.0, accumulate ? 1.0 : 0.0);
    };
    posthf::generated::pullback_df_mo_source(n, q, coefficients.data(), root.data(), bar_b.data(),
                                             first.data(), transformed.data(), bar_row.data(),
                                             bar_c.data(), bar_root.data(), read, consume, gemm);
    if (reads != response_work.source_rows || consumed != response_work.output_rows ||
        gemms != response_work.gemms || summands != response_work.contraction_summands)
      throw std::runtime_error("response work model differs from execution");
    // Shape refusal must precede every callback and any pointer use.
    auto reject = [&](std::size_t orbitals, std::size_t auxiliaries) {
      try {
        posthf::generated::pullback_df_mo_source(orbitals, auxiliaries, nullptr, nullptr, nullptr,
                                                 nullptr, nullptr, nullptr, nullptr, nullptr, read,
                                                 consume, gemm);
      } catch (const std::invalid_argument&) {
        return;
      } catch (const std::overflow_error&) {
        return;
      }
      throw std::runtime_error("invalid response shape reached source execution");
    };
    reject(0, q);
    reject(n, 0);
    reject(std::numeric_limits<std::size_t>::max(), q);
    if (reads != response_work.source_rows || gemms != response_work.gemms ||
        consumed != response_work.output_rows)
      throw std::runtime_error("invalid response shape performed work");
    std::cout << reads << ' ' << gemms << ' ' << summands << ' ' << consumed << '\n'
              << std::setprecision(17);
    for (const auto* values : {&bar_a, &bar_c, &bar_root}) {
      for (const auto value : *values) std::cout << value << ' ';
      std::cout << '\n';
    }
    return 0;
  }
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
