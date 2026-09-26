#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "molecule/basis.hpp"
#include "runtime/host_component_trace.hpp"
#include "runtime/resource_ledger.hpp"
#include "scf/cuda/df_plan_internal.hpp"
#include "scf/cuda_density_fitting.hpp"
#include "scf/cuda_density_fitting_eigen.hpp"
#include "scf/initial_guess/overlap.hpp"
#include "scf/mean_field.hpp"
#include "scf/reference/linalg.hpp"
#include "scf/reference/observation.hpp"
#include "scf/solver/eigen_frame.hpp"

namespace {
using namespace vibeqc;
using namespace scf;
using Matrix = std::vector<double>;
void require(bool value, const std::string& detail) {
  if (!value) throw std::runtime_error(detail);
}
unsigned reference_solves{};
std::size_t begin(const char* name, std::size_t) noexcept {
  reference_solves += std::string_view(name) == "reference_eigensolve";
  return 0;
}
void end(std::size_t, int) noexcept {}
const reference::observation::Observer observer{begin, end};

struct Fixture {
  std::size_t n;
  Matrix f, s, x, q, values;
  explicit Fixture(std::size_t dimension)
      : n(dimension), f(n * n), s(n * n), x(n * n), q(n * n), values(n) {
    const double pi = std::acos(-1.0);
    // A dense orthogonal cosine basis gives a known, degenerate spectrum.
    // Nonidentity diagonal S provides an analytic X, independent of both
    // production and reference orthogonalization. No eigenvector gauge is fixed.
    for (std::size_t i = 0; i < n; ++i) {
      const auto sii = 1.0 + .02 * (i % 7);
      s[i * n + i] = sii;
      x[i * n + i] = 1 / std::sqrt(sii);
      values[i] = n == 2 ? static_cast<double>(i) - .5 : static_cast<double>(i / 2) - .5;
      for (std::size_t k = 0; k < n; ++k)
        q[i * n + k] = std::sqrt((k ? 2.0 : 1.0) / n) * std::cos(pi * (i + .5) * k / n);
    }
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) {
        double value = 0;
        for (std::size_t k = 0; k < n; ++k) value += q[i * n + k] * values[k] * q[j * n + k];
        f[i * n + j] = value * std::sqrt(s[i * n + i] * s[j * n + j]);
      }
  }
};

using Plan =
    std::unique_ptr<CudaDensityFittingJkPlan, decltype(&destroy_cuda_density_fitting_jk_plan)>;
Plan make_plan(std::size_t n, std::size_t batch) {
  CudaDensityFittingJkPlan* raw{};
  std::vector<CudaDensityFittingMetricDiagnostic> diagnostic;
  std::string detail;
  require(create_cuda_density_fitting_jk_plan_tiled(0, batch, n, 1, Matrix(batch, 1),
                                                    Matrix(batch * n * n, 0), 1e-10, 1, n * n, &raw,
                                                    diagnostic, detail) == VIBEQC_STATUS_SUCCESS,
          detail);
  return {raw, &destroy_cuda_density_fitting_jk_plan};
}

void verify_projection(const Fixture& fixture, const Matrix& coefficients) {
  const auto n = fixture.n;
  // Occupy complete degenerate pairs, so signs and internal rotations cannot
  // affect the tested projector. This also detects transposed coefficient output.
  const auto occupied = n == 2 ? 1 : 2 * std::max<std::size_t>(1, n / 4);
  double largest = 0;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      double actual = 0, expected = 0;
      for (std::size_t k = 0; k < occupied; ++k) {
        actual += coefficients[i * n + k] * coefficients[j * n + k];
        expected += fixture.x[i * n + i] * fixture.q[i * n + k] * fixture.q[j * n + k] *
                    fixture.x[j * n + j];
      }
      largest = std::max(largest, std::abs(actual - expected));
    }
  require(largest < 2e-10, "ordinary provider changed the occupied invariant subspace");
}

void known_frame(std::size_t n, std::size_t batch) {
  Fixture fixture(n);
  auto plan = make_plan(n, batch);
  Matrix values, coefficients;
  CudaDfEigenDiagnostic diagnostic;
  std::string detail;
  reference_solves = 0;
  require(solve_cuda_density_fitting_eigen(plan.get(), fixture.f, &fixture.s, &fixture.x, values,
                                           coefficients, diagnostic, detail,
                                           batch - 1) == VIBEQC_STATUS_SUCCESS,
          detail);
  require(reference_solves == 0 && diagnostic.solver_calls == 1 && !diagnostic.workspace_reused,
          "ordinary provider fell back or lost its actual call/lifetime record");
  require(diagnostic.device_bytes <= df_eigen_device_reservation(n),
          "ordinary eigen allocation exceeded its reservation");
  for (std::size_t i = 0; i < n; ++i)
    require(std::abs(values[i] - fixture.values[i]) < 1e-10,
            "ordinary provider eigenvalue differs from analytic fixture");
  verify_projection(fixture, coefficients);
  if (n <= 12) {
    const auto independent = reference::generalized_eigen(fixture.f, fixture.x, n);
    for (std::size_t i = 0; i < n; ++i)
      require(std::abs(independent.values[i] - values[i]) < 1e-11,
              "device/independent oracle eigenvalues differ");
  }
  const auto first_values = values;
  auto ledger = std::make_shared<runtime::DeviceResourceLedger>();
  ledger->device = 0;
  ledger->limit = 1;
  ledger->active = true;
  runtime::active_device_resource_ledger = ledger;
  // A warmed frame needs no new explicit device allocation, even with only
  // one byte of new allowance. The library's private resources are separate.
  const auto replay_status = solve_cuda_density_fitting_eigen(
      plan.get(), fixture.f, &fixture.s, &fixture.x, values, coefficients, diagnostic, detail);
  runtime::active_device_resource_ledger.reset();
  require(replay_status == VIBEQC_STATUS_SUCCESS, detail);
  require(diagnostic.workspace_reused && ledger->allocations == 0 && ledger->rejected == 0,
          "ordinary warm eigen replay queried/allocated another workspace");
  for (std::size_t i = 0; i < n; ++i)
    require(std::abs(values[i] - first_values[i]) < 1e-10, "warm eigenvalue changed");
  verify_projection(fixture, coefficients);
  // Failed validation must not leak a partial frame or destroy the useful owner.
  Matrix bad_x = fixture.x;
  bad_x[0] *= 1.1;
  require(solve_cuda_density_fitting_eigen(plan.get(), fixture.f, &fixture.s, &bad_x, values,
                                           coefficients, diagnostic,
                                           detail) == VIBEQC_STATUS_NUMERICAL_FAILURE,
          "stale X passed physical eigen/metric checks");
  require(values.empty() && coefficients.empty(), "failed eigen validation published a frame");
  require(
      solve_cuda_density_fitting_eigen(plan.get(), fixture.f, &fixture.s, &fixture.x, values,
                                       coefficients, diagnostic, detail) == VIBEQC_STATUS_SUCCESS,
      detail);
  verify_projection(fixture, coefficients);
  require(solve_cuda_density_fitting_eigen(plan.get(), fixture.f, &fixture.s, &fixture.x, values,
                                           coefficients, diagnostic, detail,
                                           batch) == VIBEQC_STATUS_INVALID_ARGUMENT,
          "invalid batch item passed ordinary eigen validation");
  std::cout << "checked ordinary FP64 n=" << n << " batch=" << batch << '\n';
}

void rejected_allocation_and_info() {
  Fixture fixture(6);
  auto plan = make_plan(6, 1);
  Matrix values{123}, coefficients{123};
  CudaDfEigenDiagnostic diagnostic;
  std::string detail;
  auto ledger = std::make_shared<runtime::DeviceResourceLedger>();
  ledger->device = 0;
  ledger->limit = 1;
  ledger->active = true;
  runtime::active_device_resource_ledger = ledger;
  const auto status = solve_cuda_density_fitting_eigen(
      plan.get(), fixture.f, &fixture.s, &fixture.x, values, coefficients, diagnostic, detail);
  runtime::active_device_resource_ledger.reset();
  require(status == VIBEQC_STATUS_OUT_OF_MEMORY && ledger->rejected == 1 && ledger->live == 0,
          "ordinary eigen allocation bypassed the global budget or leaked partial state");
  require(values.empty() && coefficients.empty(), "OOM eigen call leaked old output");
  require(
      solve_cuda_density_fitting_eigen(plan.get(), fixture.f, &fixture.s, &fixture.x, values,
                                       coefficients, diagnostic, detail) == VIBEQC_STATUS_SUCCESS,
      detail);
  for (int info : {-1, 1}) {
    diagnostic.solver_info = info;
    require(!solver::validate_eigen_frame(fixture.f, &fixture.s, values, coefficients, fixture.n,
                                          diagnostic, detail),
            "nonzero solver info accepted a stale successful frame");
  }
  auto bad_f = fixture.f;
  bad_f[0] = std::numeric_limits<double>::quiet_NaN();
  require(solve_cuda_density_fitting_eigen(plan.get(), bad_f, &fixture.s, &fixture.x, values,
                                           coefficients, diagnostic,
                                           detail) == VIBEQC_STATUS_INVALID_ARGUMENT,
          "nonfinite input reached ordinary device solver");
  bad_f = fixture.f;
  bad_f[1] += .1;
  require(solve_cuda_density_fitting_eigen(plan.get(), bad_f, &fixture.s, &fixture.x, values,
                                           coefficients, diagnostic,
                                           detail) == VIBEQC_STATUS_INVALID_ARGUMENT,
          "nonsymmetric matrix silently selected one triangle");
  // The same adapter supports ordinary symmetric setup solves without S/X.
  require(
      solve_cuda_density_fitting_eigen(plan.get(), fixture.f, nullptr, nullptr, values,
                                       coefficients, diagnostic, detail) == VIBEQC_STATUS_SUCCESS,
      detail);
  const auto original = fixture.f;
  require(solve_cuda_density_fitting_eigen(plan.get(), fixture.f, &fixture.s, &fixture.x, values,
                                           fixture.f, diagnostic,
                                           detail) == VIBEQC_STATUS_INVALID_ARGUMENT &&
              fixture.f == original,
          "alias rejection corrupted an eigen input");
  const auto prior = values;
  require(solve_cuda_density_fitting_eigen(plan.get(), fixture.f, &fixture.s, &fixture.x, values,
                                           values, diagnostic,
                                           detail) == VIBEQC_STATUS_INVALID_ARGUMENT &&
              values == prior,
          "aliased eigen outputs were modified");
  require(cudaStreamBeginCapture(plan->stream, cudaStreamCaptureModeThreadLocal) == cudaSuccess,
          "cannot create capture rejection fixture");
  const auto captured = solve_cuda_density_fitting_eigen(
      plan.get(), fixture.f, &fixture.s, &fixture.x, values, coefficients, diagnostic, detail);
  cudaGraph_t graph{};
  const auto ended = cudaStreamEndCapture(plan->stream, &graph);
  if (graph) (void)cudaGraphDestroy(graph);
  require(ended == cudaSuccess && captured == VIBEQC_STATUS_INVALID_ARGUMENT,
          "ordinary eigen operation submitted work into a captured stream");
  require(
      solve_cuda_density_fitting_eigen(plan.get(), fixture.f, &fixture.s, &fixture.x, values,
                                       coefficients, diagnostic, detail) == VIBEQC_STATUS_SUCCESS,
      detail);
}

void device_overlap_cutoff() {
  auto plan = make_plan(3, 1);
  core::System system;
  system.atoms.push_back({1, {0, 0, 0}});
  initial_guess::OverlapOrthogonalizer cache;
  initial_guess::EigenOperation eigen = [&](const Matrix& matrix, const Matrix* s, const Matrix* x,
                                            std::size_t) {
    reference::EigenResult result;
    CudaDfEigenDiagnostic diagnostic;
    std::string detail;
    require(
        solve_cuda_density_fitting_eigen(plan.get(), matrix, s, x, result.values, result.vectors,
                                         diagnostic, detail) == VIBEQC_STATUS_SUCCESS,
        detail);
    return result;
  };
  // A diagonal metric makes the exact boundary independent of eigenvalue
  // roundoff. The general rotated/degenerate frame is qualified above.
  Matrix overlap{1e-10, 0, 0, 0, 1, 0, 0, 0, 2};
  const auto x = cache.get(system, overlap, 3, eigen);
  require(std::abs(x[0] - 1e5) < 1e-10, "ordinary overlap changed the admitted boundary");
  overlap[0] = std::nextafter(1e-10, 0.0);
  bool rejected = false;
  try {
    cache.get(system, overlap, 3, eigen);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected && cache.numeric_capacity_bytes() == 0, "singular device X was retained");
  overlap[0] = 1e-9;
  cache.get(system, overlap, 3, eigen);
}

void physical_reference_export() {
  // Explicit d polarization distinguishes the two layouts without relying on
  // the device/reference solver to construct any expected invariant.
  for (auto representation : {VIBEQC_BASIS_CARTESIAN, VIBEQC_BASIS_SPHERICAL}) {
    core::System system;
    system.atoms = {{1, {0, 0, -.7}}, {1, {0, 0, .7}}};
    const std::vector<core::Primitive> primitives{
        {3.42525091, .1543289673}, {.62391373, .5353281423}, {.1688554, .4446345422}};
    system.shells = {
        {0, 0, primitives}, {1, 0, primitives}, {0, 2, {{.6, 1.0}}}, {1, 2, {{.6, 1.0}}}};
    system.basis_representation = representation;
    std::string detail;
    require(molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS, detail);
    ScfOptions options;
    options.export_physical_reference = true;
    options.screening_tolerance = 0;
    options.energy_tolerance = 1e-12;
    options.density_tolerance = 1e-10;
    options.reference_memory_budget_bytes = 256ULL << 20;
    // CPU DF owns its independent finalizer and complete derivative assembly.
    const auto expected = run_rhf_density_fitting(system, system, options);
    for (bool forces : {false, true}) {
      options.compute_forces = forces;
      const char* labels[] = {"test_cuda_export_cold", "test_cuda_export_retained",
                              "test_cuda_export_device_rebuild",
                              "test_cuda_export_reference_rebuild"};
      for (unsigned mode = 0; mode < 4; ++mode) {
        setenv("VIBEQC_DF_FORCE_FINAL_REBUILD", mode >= 2 ? "1" : "0", 1);
        setenv("VIBEQC_DF_REFERENCE_FINAL_EIGEN", mode == 3 ? "1" : "0", 1);
        ScfResult actual;
        {
          runtime::host_trace::Region trace(labels[mode]);
          actual = run_rhf_density_fitting_cuda(system, system, options, 0,
                                                mode ? &expected.density : nullptr);
        }
        unsetenv("VIBEQC_DF_FORCE_FINAL_REBUILD");
        unsetenv("VIBEQC_DF_REFERENCE_FINAL_EIGEN");
        require(actual.converged && actual.reference && expected.reference,
                "DF reference export did not retain a converged frame");
        auto frame = *actual.reference;
        validate_physical_reference(frame);
        require(std::abs(actual.energy - expected.energy) < 1e-9, "DF reference energy changed");
        for (std::size_t i = 0; i < actual.density.size(); ++i)
          require(std::abs(actual.density[i] - expected.density[i]) < 1e-9,
                  "DF reference density changed");
        for (std::size_t i = 0; i < frame.orbital_energies.size(); ++i)
          require(
              std::abs(frame.orbital_energies[i] - expected.reference->orbital_energies[i]) < 1e-9,
              "DF physical canonical energies changed");
        if (forces) {
          require(actual.forces.size() == expected.forces.size(), "DF force extent changed");
          for (std::size_t i = 0; i < actual.forces.size(); ++i)
            require(std::abs(actual.forces[i] - expected.forces[i]) < 1e-8,
                    "DF physical-reference complete force changed");
        } else
          require(actual.forces.empty(), "energy-only reference export computed forces");
      }
    }
  }
}

}  // namespace

int main() {
  reference::observation::active = &observer;
  try {
    rejected_allocation_and_info();
    device_overlap_cutoff();
    physical_reference_export();
    for (const auto n : {2U, 12U, 32U, 96U, 192U, 384U, 513U})
      for (const auto batch : {1U, 4U}) known_frame(n, batch);
    reference::observation::active = nullptr;
    return 0;
  } catch (const std::exception& error) {
    runtime::active_device_resource_ledger.reset();
    reference::observation::active = nullptr;
    std::cerr << error.what() << '\n';
    return 1;
  }
}
