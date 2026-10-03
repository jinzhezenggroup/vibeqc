#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "cc/lambda_response.hpp"

namespace {
using namespace generativeqc::cc;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

/** Independent noninteracting reference: R1=(eps_a-eps_i)*t1 and
 * R2=(eps_a+eps_b-eps_i-eps_j)*t2. With an external energy source S, the
 * exact Lambda is S/D in dense coordinates, including off-diagonal pair
 * orbits. No generated derivative or CPU response supplies the expected values.
 */
void qualify(bool cuda) {
  constexpr std::size_t o = 2, v = 3, n1 = o * v, n2 = n1 * n1;
  const double occupied[o]{-20.0, -1.0}, virtuals[v]{0.3, 2.0, 17.0};
  Problem p;
  p.nocc = o;
  p.nvir = v;
  p.foo.resize(o * o);
  p.fov.resize(n1);
  p.fvv.resize(v * v);
  p.ovov.resize(n2);
  p.ovvo.resize(n2);
  p.oovv.resize(n2);
  p.ovvv.resize(o * v * v * v);
  p.ovoo.resize(o * v * o * o);
  p.oooo.resize(o * o * o * o);
  p.vvvv.resize(v * v * v * v);
  p.d1.resize(n1);
  p.d2.resize(n2);
  p.initial_t1.resize(n1);
  p.initial_t2.resize(n2);
  for (std::size_t i = 0; i < o; ++i) p.foo[i * o + i] = occupied[i];
  for (std::size_t a = 0; a < v; ++a) p.fvv[a * v + a] = virtuals[a];
  std::vector<double> source1(n1), source2(n2), expected1(n1), expected2(n2);
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t a = 0; a < v; ++a) {
      const auto k = i * v + a;
      p.d1[k] = occupied[i] - virtuals[a];
      source1[k] = 0.002 * (k + 1);
      expected1[k] = source1[k] / p.d1[k];
    }
  for (std::size_t i = 0; i < o; ++i)
    for (std::size_t j = 0; j < o; ++j)
      for (std::size_t a = 0; a < v; ++a)
        for (std::size_t b = 0; b < v; ++b) {
          const auto k = ((i * o + j) * v + a) * v + b;
          const auto mate = ((j * o + i) * v + b) * v + a;
          p.d2[k] = (occupied[i] - virtuals[a]) + (occupied[j] - virtuals[b]);
          source2[k] = 0.001 * (std::min(k, mate) + 1);
          expected2[k] = source2[k] / p.d2[k];
        }
  SolverResult cc;
  cc.status = SolveStatus::Converged;
  cc.t1.resize(n1);
  cc.t2.resize(n2);
  LambdaOptions options;
  options.gmres.absolute_tolerance = 1e-12;
  options.gmres.max_iterations = 200;
  const auto capacity = lambda_cpu_numeric_capacity(p, cc, options, true);
  // Exact admitted host capacity includes the audit/diagonal buffer already.
  options.max_bytes = capacity;
  auto solve = [&] {
#if GENERATIVEQC_HAS_CUDA
    if (cuda) return solve_lambda_cuda_with_energy_source(p, cc, source1, source2, 0, options);
#else
    require(!cuda, "CUDA unavailable in this test build");
#endif
    return solve_lambda_cpu_with_energy_source(p, cc, source1, source2, options);
  };
  auto check = [&](const LambdaResult& result, bool preconditioned) {
    require(result.converged(), "Lambda convergence");
    double error = 0.0;
    for (std::size_t k = 0; k < n1; ++k)
      error = std::max(error, std::abs(result.lambda1[k] - expected1[k]));
    for (std::size_t k = 0; k < n2; ++k)
      error = std::max(error, std::abs(result.lambda2[k] - expected2[k]));
    require(error < 1e-11, "independent noninteracting Lambda oracle mismatch");
    require(result.diagnostic.diagonal_preconditioned == preconditioned,
            "preconditioner selection");
    require((result.diagnostic.preconditioner_actions > 0) == preconditioned,
            "preconditioner work");
    require(result.diagnostic.independent_residual_norm < 1e-9, "physical residual");
    require(result.diagnostic.numeric_capacity_bytes == capacity, "unchanged numeric bound");
    if (preconditioned)
      require(result.diagnostic.iterations == 1, "diagonal solve needs one step");
    else
      require(result.diagnostic.iterations > 1, "nontrivial unpreconditioned comparison");
    std::cout << (cuda ? "cuda" : "cpu") << " preconditioned=" << preconditioned
              << " iterations=" << result.diagnostic.iterations
              << " actions=" << result.diagnostic.operator_actions << " error=" << error << '\n';
  };
  check(solve(), true);
  options.diagonal_preconditioning = false;
  require(lambda_cpu_numeric_capacity(p, cc, options, true) == capacity, "optional capacity");
  check(solve(), false);
  options.diagonal_preconditioning = true;
  const auto first = p.d1[0];
  // Alter only optional solver denominators: the physical Fock operator and
  // independent expected Lambda stay unchanged. Unsafe diagonals must fall back.
  for (const double unsafe : {0.0, 1e-14}) {
    p.d1[0] = unsafe;
    check(solve(), false);
  }
  p.d1[0] = first;
  p.d2[1] += 1.0;  // Break the (i,j,a,b)<->(j,i,b,a) denominator orbit.
  check(solve(), false);
}
}  // namespace

int main(int argc, char** argv) {
  try {
    qualify(argc == 2 && std::string_view(argv[1]) == "--cuda");
    std::cout << "Lambda preconditioner oracle, fallback and budget gates passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
