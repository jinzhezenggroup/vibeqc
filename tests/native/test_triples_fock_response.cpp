#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

#include "cc/triples_fock_response.hpp"

namespace {
using namespace generativeqc::cc;
void require(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}
void fill(std::vector<double>& values) {
  for (std::size_t k = 0; k < values.size(); ++k)
    values[k] = .02 * (static_cast<int>((7 * k + 3) % 17) - 8);
}

/** Pinned by the independent full Kronecker-resolvent reference in the Python
 * tests, using the exact rational fixture below. Same-space gaps are zero in
 * both occupied and virtual blocks; all off-diagonal moments are retained.
 */
void qualify(bool cuda) {
  constexpr std::size_t o = 2, v = 3;
  Problem p;
  p.nocc = o;
  p.nvir = v;
  p.foo.resize(o * o);
  p.fov.resize(o * v);
  p.fvv.resize(v * v);
  p.ovov.resize(o * v * o * v);
  p.ovvo.resize(o * v * v * o);
  p.oovv.resize(o * o * v * v);
  p.ovvv.resize(o * v * v * v);
  p.ovoo.resize(o * v * o * o);
  p.oooo.resize(o * o * o * o);
  p.vvvv.resize(v * v * v * v);
  p.d1.assign(o * v, -1.0);
  p.d2.assign(o * o * v * v, -2.0);
  p.initial_t1.resize(o * v);
  p.initial_t2.resize(o * o * v * v);
  fill(p.ovvv);
  fill(p.ovoo);
  fill(p.ovov);
  fill(p.fov);
  SolverResult cc;
  cc.status = SolveStatus::Converged;
  cc.t1.resize(o * v);
  cc.t2.resize(o * o * v * v);
  fill(cc.t1);
  fill(cc.t2);
  auto original = cc.t2;
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < v; ++b) {
          auto k = ((i * o + j) * v + a) * v + b, partner = ((j * o + i) * v + b) * v + a;
          cc.t2[k] = .5 * (original[k] + original[partner]);
        }
  std::vector<double> eo{-1.3, -1.3}, ev{.4, .4, 1.2};
  const double expected_foo[]{-0.0097852960725836914, 0.0020215826926775456, 0.0020215826926775456,
                              -0.0098656849586202001};
  const double expected_fvv[]{0.0067623078886356871,  -0.001852614849940457, -0.0014656348764651328,
                              -0.001852614849940457,  0.0077451511005937231, -0.003619000590268995,
                              -0.0014656348764651328, -0.003619000590268995, 0.0051435220419744822};
  TriplesResponseOptions options;
  auto solve = [&] {
#if GENERATIVEQC_HAS_CUDA
    if (cuda) return triples_fock_response_cuda(p, cc, eo, ev, 0, options);
#else
    require(!cuda, "CUDA unavailable in this build");
#endif
    return triples_fock_response_cpu(p, cc, eo, ev, options);
  };
  auto check = [&](const TriplesFockResponseResult& result, std::size_t q) {
    double error = 0;
    for (std::size_t k = 0; k < o * o; ++k)
      error = std::max(error, std::abs(result.foo[k] - expected_foo[k]));
    for (std::size_t k = 0; k < v * v; ++k)
      error = std::max(error, std::abs(result.fvv[k] - expected_fvv[k]));
    require(error < 2e-12, "independent full Fock reference mismatch");
    const auto panels = v * (v + 1) / 2, pages = (v + q - 1) / q, pairs = pages * (pages + 1) / 2;
    require(result.page_capacity == q && result.pair_panels == panels, "page selection");
    require(result.vector_pages == panels * pairs, "bounded recomputation count");
    require(result.occupied_moments == panels * pages && result.virtual_moments == panels * pairs,
            "moment work");
    const auto layout = detail::triples_fock_response_layout(o, v, q, cuda);
    require(result.numeric_capacity_bytes == layout.numeric_bytes(),
            "complete owned numeric budget");
    require(result.device_capacity_bytes == layout.device_bytes, "device budget");
    require((result.kernel_launches > 0) == cuda, "execution backend work");
    if (cuda) {
      require(result.device_copy_bytes ==
                  result.vector_pages * 2 * layout.vector_elements * sizeof(double),
              "resident vector copies");
      require(result.host_to_device_bytes ==
                  layout.inputs * sizeof(double) + result.vector_pages * q * 4 * sizeof(double),
              "complete uploads");
      require(result.device_to_host_bytes ==
                  layout.outputs * sizeof(double) +
                      (result.vector_pages + result.occupied_moments + result.virtual_moments) *
                          sizeof(int),
              "complete downloads and errors");
    }
    std::cout << (cuda ? "cuda" : "cpu") << " q=" << q << " vectors=" << result.vector_pages
              << " error=" << error << '\n';
  };
  for (auto q : {std::size_t(1), std::size_t(2), std::size_t(3)}) {
    options.batch_capacity = q;
    options.max_bytes = detail::triples_fock_response_layout(o, v, q, cuda).numeric_bytes();
    check(solve(), q);
  }
  // A requested wide page must shrink to the exact one-lane admitted bound.
  options.batch_capacity = 16;
  options.max_bytes = detail::triples_fock_response_layout(o, v, 1, cuda).numeric_bytes();
  check(solve(), 1);
  --options.max_bytes;
  bool refused = false;
  try {
    (void)solve();
  } catch (const std::length_error&) {
    refused = true;
  }
  require(refused, "one byte below minimum must refuse");
  options.max_bytes = 256ULL << 20;
  ev[0] = eo[0];
  refused = false;
  try {
    (void)solve();
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  require(refused, "closed occupied/virtual gap must refuse");
  ev[0] = .4;
  cc.t1[0] = std::numeric_limits<double>::quiet_NaN();
  refused = false;
  try {
    (void)solve();
  } catch (const std::invalid_argument&) {
    refused = true;
  }
  require(refused, "nonfinite input must refuse");
}
}  // namespace
int main(int argc, char** argv) {
  try {
    qualify(argc == 2 && std::string_view(argv[1]) == "--cuda");
    std::cout << "Triples Fock oracle, work, fallback and budget gates passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
