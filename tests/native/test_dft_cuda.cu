#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "dft/cuda_xc.hpp"
#include "dft/xc.hpp"
#include "molecule/basis.hpp"
#include "runtime/cuda_resources.cuh"

namespace {
using namespace vibeqc::dft;
void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
void close(double actual, double expected, const char* message, double tolerance = 2e-11) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
    std::cerr << std::setprecision(17) << message << ": " << actual << " versus " << expected
              << ", difference " << actual - expected << '\n';
    throw std::runtime_error(message);
  }
}

vibeqc::core::System system(unsigned l = 0, bool spherical = false) {
  vibeqc::core::System out;
  out.atoms = {{1, {0, 0, 0}}, {1, {0.1, 0.2, 1.4}}};
  out.shells = {
      {0,
       0,
       {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}},
      {1,
       0,
       {{3.425250914, 0.1543289673}, {0.6239137298, 0.5353281423}, {0.168855404, 0.4446345422}}}};
  if (l) out.shells.push_back({1, l, {{0.7, 1.0}}});
  if (spherical) out.basis_representation = VIBEQC_BASIS_SPHERICAL;
  std::string detail;
  if (vibeqc::molecule::validate_and_normalize(out, detail) != VIBEQC_STATUS_SUCCESS)
    throw std::runtime_error(detail);
  return out;
}

/** Test owner deliberately allocates exactly the component request plus a
 * canary. Production ResourcePlan will own this arena together with J/SCF. */
struct Fixture {
  cudaStream_t stream{};
  void* arena{};
  double* density{};
  CudaXcLayout layout;
  std::unique_ptr<CudaXcPlan> plan;
  std::uint64_t generation{};
  Fixture(const AoBasis& basis, const MolecularGrid& grid, std::uint32_t functional, bool uks,
          std::size_t tile, CudaXcAoPrecision ao_precision = CudaXcAoPrecision::Fp64,
          bool response = false, double exchange_scale = 1.0, double correlation_scale = 1.0)
      : layout(cuda_xc_layout(basis, grid, functional, uks, tile, ao_precision, exchange_scale,
                              correlation_scale)) {
    try {
      if (response)
        layout = cuda_xc_layout_shape(layout.natom, layout.nprimitive, layout.nao, layout.npoint,
                                      functional, uks, tile, true, ao_precision, exchange_scale,
                                      correlation_scale);
      check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      check(cudaMalloc(&arena, layout.device_bytes + 64));
      check(cudaMemset(static_cast<char*>(arena) + layout.device_bytes, 0x5a, 64));
      check(cudaMalloc(&density, layout.spins * layout.nao * layout.nao * sizeof(double)));
      plan = std::make_unique<CudaXcPlan>(layout, basis.packed, grid.points(), grid.weights(),
                                          arena, layout.device_bytes, stream);
    } catch (...) {
      cleanup();
      throw;
    }
  }
  void cleanup() {
    plan.reset();
    if (stream) cudaStreamSynchronize(stream);
    if (density) cudaFree(density);
    if (arena) cudaFree(arena);
    if (stream) cudaStreamDestroy(stream);
  }
  ~Fixture() { cleanup(); }
  void submit(const std::vector<double>& d) {
    check(cudaMemcpyAsync(density, d.data(), d.size() * sizeof(double), cudaMemcpyHostToDevice,
                          stream));
    // Reference input transfer is an explicit test stage. Complete it before
    // a temporary host density can die; the measured native enqueue follows.
    check(cudaStreamSynchronize(stream));
    const auto before = plan->transfers();
    plan->enqueue(density, d.size(), ++generation);
    const auto after = plan->transfers();
    require(after.output_d2h_bytes == before.output_d2h_bytes &&
                after.setup_h2d_bytes == before.setup_h2d_bytes &&
                after.synchronizations == before.synchronizations,
            "XC enqueue staged data or synchronized");
  }
  CudaXcScalars scalars() { return plan->read_scalars(generation); }
  std::vector<double> potential() { return plan->download_potential(generation); }
  void canary() {
    unsigned char bytes[64]{};
    check(cudaMemcpy(bytes, static_cast<char*>(arena) + layout.device_bytes, 64,
                     cudaMemcpyDeviceToHost));
    require(std::all_of(std::begin(bytes), std::end(bytes), [](auto b) { return b == 0x5a; }),
            "XC arena exceeded its exact resource request");
  }
};

std::vector<double> density(std::size_t n, unsigned spins) {
  std::vector<double> d(spins * n * n);
  for (unsigned s = 0; s < spins; ++s)
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j)
        d[(s * n + i) * n + j] =
            (s == 0 ? 0.7 : 0.3) * ((i == j ? 0.2 : 0.0) + 0.1 / ((i + 1.0) * (j + 1.0)));
  return d;
}

void compare(Fixture& fixture, const AoBasis& basis, const MolecularGrid& grid,
             const std::vector<double>& d, const std::vector<double>& empty_spin_reference = {}) {
  fixture.submit(d);
  const auto result = fixture.scalars();
  require(result.error == 0,
          "valid density failed device XC evaluation: n=" + std::to_string(basis.nao) +
              " functional=" + std::to_string(fixture.layout.functional) +
              " spins=" + std::to_string(fixture.layout.spins) +
              " tile=" + std::to_string(fixture.layout.tile_points) +
              " error=" + std::to_string(result.error));
  const auto v = fixture.potential();
  const auto& l = fixture.layout;
  if (l.spins == 1) {
    const auto ref = l.functional == 4U   ? integrate_wb97mv_rks(basis, grid, d, 17)
                     : l.functional == 3U ? integrate_b3lyp_rks(basis, grid, d, 17)
                     : l.functional == 2U ? integrate_r2scan_rks(basis, grid, d, 17)
                     : l.functional == 1U
                         ? integrate_pbe_rks_with_tail_scaled(basis, grid, d, 17, {},
                                                              l.exchange_scale, l.correlation_scale)
                         : integrate_lda_xc_pw_rks(basis, grid, d, 17);
    close(result.energy, ref.energy, "RKS CPU/CUDA XC energy");
    close(result.electrons[0] + result.electrons[1], ref.electrons, "RKS electrons");
    for (std::size_t i = 0; i < v.size(); ++i)
      close(v[i], ref.potential[i], "RKS CPU/CUDA V", 2e-11 + 2e-12 * std::abs(ref.potential[i]));
  } else {
    const auto elements = l.nao * l.nao;
    const std::vector<double> a(d.begin(), d.begin() + elements), b(d.begin() + elements, d.end());
    const auto ref =
        l.functional == 4U   ? integrate_wb97mv_uks(basis, grid, a, b, 17)
        : l.functional == 3U ? integrate_b3lyp_uks(basis, grid, a, b, 17)
        : l.functional == 2U ? integrate_r2scan_uks(basis, grid, a, b, 17)
        : l.functional == 1U
            ? integrate_pbe_uks_scaled(basis, grid, a, b, 17, l.exchange_scale, l.correlation_scale)
            : integrate_lda_xc_pw_uks(basis, grid, a, b, 17);
    close(result.energy, ref.energy, "UKS CPU/CUDA XC energy");
    for (unsigned s = 0; s < 2; ++s) {
      close(result.electrons[s], ref.electrons[s], "UKS electrons");
      // The explicit PBE empty-spin extension has large finite minority
      // coefficients. Retain an FP64 relative gate as well as the absolute
      // floor; an absolute-only test would reject a few ulps at |V|~1e4.
      const auto begin = d.begin() + s * elements;
      const bool empty = std::all_of(begin, begin + elements, [](double x) { return x == 0.0; });
      for (std::size_t i = 0; i < elements; ++i) {
        // Stable compiler coordinates restore the full endpoint gate even
        // when independently evaluated AO features differ by a few ulps.
        const double expected = ref.potential[s][i];
        if (empty && !empty_spin_reference.empty()) {
          const double same_input = empty_spin_reference[s * elements + i];
          if (std::abs(v[s * elements + i] - same_input) > 2e-11 + 2e-12 * std::abs(same_input))
            std::cerr << "functional=" << l.functional << " spin=" << s << " matrix=" << i
                      << " tile=" << l.tile_points << " points=" << grid.point_count() << '\n';
          close(v[s * elements + i], same_input, "UKS same-input minority V",
                2e-11 + 2e-12 * std::abs(same_input));
        }
        const double tolerance = 2e-11 + 2e-12 * std::abs(expected);
        if (std::abs(v[s * elements + i] - expected) > tolerance)
          std::cerr << "functional=" << l.functional << " spin=" << s << " matrix=" << i
                    << " tile=" << l.tile_points << " points=" << grid.point_count() << '\n';
        close(v[s * elements + i], expected, "UKS CPU/CUDA V", tolerance);
      }
    }
  }
  fixture.canary();
}

/** Check AO/features against independent host contractions, then evaluate
 * r2SCAN or WB97M-V on the captured device features. This distinguishes point
 * math disagreements from feature-rounding sensitivity without relaxing the
 * complete CPU endpoint gate. The independent r2SCAN boundary fixtures live
 * in test_r2scan_boundary_codegen.
 */
std::vector<double> empty_spin_reference(const AoBasis& basis, const MolecularGrid& grid,
                                         const std::vector<double>& d, std::uint32_t functional) {
  const auto count = grid.point_count(), n = basis.nao;
  // A test-only full tile retains the inputs after enqueue. Production and the
  // endpoint checked against this reference keep their bounded 257-point tile.
  Fixture capture(basis, grid, functional, true, count);
  capture.submit(d);
  require(capture.scalars().error == 0, "meta-GGA reference capture failed");
  const auto& l = capture.layout;
  std::vector<double> ao(l.jets * count * n), features(l.spins * l.feature_terms * count);
  std::vector<double> coefficients(functional == 4U ? features.size() : 0U);
  // Mirror the borrowed arena's packed basis/grid, AO, density-work, features
  // order. No production inspection API or host transfer is introduced.
  const auto* device = static_cast<const double*>(capture.arena) + l.packed_elements + 4 * count;
  check(cudaMemcpy(ao.data(), device, ao.size() * sizeof(double), cudaMemcpyDeviceToHost));
  device += (l.jets + l.spins * l.work_jets) * count * n;
  check(cudaMemcpy(features.data(), device, features.size() * sizeof(double),
                   cudaMemcpyDeviceToHost));
  if (functional == 4U)
    check(cudaMemcpy(coefficients.data(), device + features.size(),
                     coefficients.size() * sizeof(double), cudaMemcpyDeviceToHost));
  if (functional == 4U && std::getenv("VIBEQC_WB97MV_DIAGNOSTIC_FILE")) {
    std::ofstream output(std::getenv("VIBEQC_WB97MV_DIAGNOSTIC_FILE"), std::ios::binary);
    const std::uint64_t header[]{count, n};
    output.write(reinterpret_cast<const char*>(header), sizeof(header));
    for (const auto& values : {ao, features, coefficients, grid.weights()})
      output.write(reinterpret_cast<const char*>(values.data()),
                   static_cast<std::streamsize>(values.size() * sizeof(double)));
    require(static_cast<bool>(output), "unable to capture WB97M-V point diagnostics");
  }
  std::vector<double> cpu_ao(ao.size()), expected(l.spins * n * n);
  double max_point_difference = 0.0;
  std::size_t worst_point = 0;
  double worst_cpu = 0.0, worst_gpu = 0.0, worst_density = 0.0;
  double worst_sigma = 0.0, worst_tau = 0.0;
  basis.evaluate(grid.points().data(), count, 1, 0, n, cpu_ao.data(), cpu_ao.size());
  for (std::size_t i = 0; i < ao.size(); ++i)
    close(ao[i], cpu_ao[i], "captured meta-GGA AO", 2e-15 + 2e-13 * std::abs(cpu_ao[i]));
  for (std::size_t p = 0; p < count; ++p) {
    const auto phi = [&](unsigned jet, std::size_t mu) { return ao[(jet * count + p) * n + mu]; };
    double rho[2]{}, gradient[2][3]{}, tau[2]{};
    for (unsigned spin = 0; spin < 2; ++spin) {
      double reference[5]{};
      // Independent double AO-pair contraction checks the generated D*AO
      // feature path before its rounded inputs become the XC point reference.
      for (std::size_t mu = 0; mu < n; ++mu)
        for (std::size_t nu = 0; nu < n; ++nu) {
          const double value = d[(spin * n + mu) * n + nu];
          reference[0] += phi(0, mu) * value * phi(0, nu);
          for (unsigned k = 0; k < 3; ++k) {
            reference[k + 1] += value * (phi(k + 1, mu) * phi(0, nu) + phi(0, mu) * phi(k + 1, nu));
            reference[4] += 0.5 * phi(k + 1, mu) * value * phi(k + 1, nu);
          }
        }
      for (unsigned k = 0; k < 5; ++k)
        close(features[(spin * 5 + k) * count + p], reference[k], "captured meta-GGA feature",
              2e-15 + 2e-13 * std::abs(reference[k]));
      rho[spin] = features[spin * 5 * count + p];
      for (unsigned k = 0; k < 3; ++k) gradient[spin][k] = features[(spin * 5 + k + 1) * count + p];
      tau[spin] = features[(spin * 5 + 4) * count + p];
    }
    const auto accumulate = [&](const auto& xc) {
      for (unsigned spin = 0; spin < 2; ++spin)
        for (std::size_t mu = 0; mu < n; ++mu)
          for (std::size_t nu = 0; nu < n; ++nu) {
            double value = xc.rho[spin] * phi(0, mu) * phi(0, nu);
            for (unsigned k = 0; k < 3; ++k) {
              value += xc.gradient[spin][k] *
                       (phi(k + 1, mu) * phi(0, nu) + phi(0, mu) * phi(k + 1, nu));
              value += xc.kinetic[spin] * phi(k + 1, mu) * phi(k + 1, nu);
            }
            expected[(spin * n + mu) * n + nu] += grid.weights()[p] * value;
          }
    };
    if (functional == 4U) {
      const auto xc = evaluate_wb97mv_point(rho, gradient, tau);
      const double gpu = coefficients[5 * count + p];
      const double difference = std::abs(gpu - xc.rho[1]);
      if (difference > max_point_difference) {
        max_point_difference = difference;
        worst_point = p;
        worst_cpu = xc.rho[1];
        worst_gpu = gpu;
        worst_density = rho[0];
        worst_sigma = gradient[0][0] * gradient[0][0] + gradient[0][1] * gradient[0][1] +
                      gradient[0][2] * gradient[0][2];
        worst_tau = tau[0];
      }
      accumulate(xc);
    } else {
      accumulate(evaluate_r2scan_point(rho, gradient, tau));
    }
  }
  if (functional == 4U && max_point_difference > 2e-11 + 2e-12 * std::abs(worst_cpu))
    std::cerr << std::setprecision(17) << "worst WB97M point=" << worst_point
              << " rho_a=" << worst_density << " v_b cpu=" << worst_cpu
              << " sigma_aa=" << worst_sigma << " tau_a=" << worst_tau << " gpu=" << worst_gpu
              << " diff=" << max_point_difference << '\n';
  capture.canary();
  return expected;
}

__global__ void halve_density(double* d, std::size_t n) {
  for (std::size_t i = threadIdx.x; i < n; i += blockDim.x) d[i] *= 0.5;
}

void variational_and_state(const AoBasis& basis, const MolecularGrid& grid,
                           std::uint32_t functional, std::size_t tile = 7) {
  Fixture good(basis, grid, functional, true, tile), bad(basis, grid, functional, true, tile + 4);
  auto d = density(basis.nao, 2);
  good.submit(d);
  auto invalid = d;
  invalid[0] = std::numeric_limits<double>::quiet_NaN();
  bad.submit(invalid);
  require(bad.scalars().error != 0, "nonfinite spin density was accepted");
  require(good.scalars().error == 0, "one failed plan contaminated another stream");
  const auto v = good.potential();
  const std::size_t n = basis.nao, elements = n * n;
  for (unsigned spin = 0; spin < 2; ++spin) {
    std::vector<double> direction(d.size());
    // A symmetric off-diagonal perturbation tests both AO legs and spin isolation.
    direction[spin * elements + 1] = direction[spin * elements + n] = 0.1;
    direction[spin * elements] = -0.04;
    double contraction = 0;
    for (std::size_t i = 0; i < d.size(); ++i) contraction += v[i] * direction[i];
    for (double step : {1e-4, 3e-5}) {
      auto plus = d, minus = d;
      for (std::size_t i = 0; i < d.size(); ++i) {
        plus[i] += step * direction[i];
        minus[i] -= step * direction[i];
      }
      good.submit(plus);
      const auto ep = good.scalars();
      good.submit(minus);
      const auto em = good.scalars();
      require(ep.error == 0 && em.error == 0, "variational density was rejected");
      close((ep.energy - em.energy) / (2 * step), contraction, "GPU delta E = Tr(V delta D)", 2e-8);
    }
  }
  compare(bad, basis, grid, d);  // A new generation clears the preceding numerical failure.
  const auto old = bad.generation;
  halve_density<<<1, 32, 0, bad.stream>>>(bad.density, d.size());
  check(cudaGetLastError());
  bad.plan->enqueue(bad.density, d.size(), ++bad.generation);
  require(bad.scalars().error == 0, "device-produced density was rejected");
  for (auto& x : d) x *= 0.5;
  const std::vector<double> a(d.begin(), d.begin() + elements), b(d.begin() + elements, d.end());
  const auto ref = functional == 4U   ? integrate_wb97mv_uks(basis, grid, a, b)
                   : functional == 3U ? integrate_b3lyp_uks(basis, grid, a, b)
                   : functional == 2U ? integrate_r2scan_uks(basis, grid, a, b)
                   : functional == 1U ? integrate_pbe_uks(basis, grid, a, b)
                                      : integrate_lda_xc_pw_uks(basis, grid, a, b);
  close(bad.scalars().energy, ref.energy, "XC ignored the current device density");
  bool stale = false;
  try {
    (void)bad.plan->view(old);
  } catch (const std::invalid_argument&) {
    stale = true;
  }
  require(stale, "stale GPU XC result view was accepted");
  stale = false;
  try {
    bad.plan->enqueue(bad.density, d.size(), old);
  } catch (const std::invalid_argument&) {
    stale = true;
  }
  require(stale, "stale GPU density generation was accepted");
  good.canary();
  bad.canary();
}

void graph_capture(const AoBasis& basis, const MolecularGrid& grid, unsigned functional,
                   bool unrestricted, std::size_t tile = 9) {
  Fixture captured(basis, grid, functional, unrestricted, tile);
  auto d = density(basis.nao, unrestricted ? 2 : 1);
  check(cudaMemcpyAsync(captured.density, d.data(), d.size() * sizeof(double),
                        cudaMemcpyHostToDevice, captured.stream));
  check(cudaStreamSynchronize(captured.stream));

  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  try {
    check(cudaStreamBeginCapture(captured.stream, cudaStreamCaptureModeGlobal));
    captured.plan->enqueue(captured.density, d.size(), ++captured.generation);
    check(cudaStreamEndCapture(captured.stream, &graph));
    check(cudaGraphInstantiate(&executable, graph, 0));
    for (unsigned replay = 0; replay < 2; ++replay) {
      // The immutable consumer survives graph replay, but the density does not.
      // Compare its complete outputs against a fresh independently checked plan.
      check(cudaMemcpyAsync(captured.density, d.data(), d.size() * sizeof(double),
                            cudaMemcpyHostToDevice, captured.stream));
      check(cudaGraphLaunch(executable, captured.stream));
      check(cudaStreamSynchronize(captured.stream));
      const auto result = captured.scalars();
      require(result.error == 0, "captured XC result was invalid");
      Fixture fresh(basis, grid, functional, unrestricted, 7);
      compare(fresh, basis, grid, d);
      close(result.energy, fresh.scalars().energy, "captured XC energy");
      const auto actual = captured.potential(), expected = fresh.potential();
      for (std::size_t i = 0; i < actual.size(); ++i)
        close(actual[i], expected[i], "captured XC potential");
      captured.canary();
      for (double& value : d) value *= 0.7;
    }
  } catch (...) {
    if (executable) cudaGraphExecDestroy(executable);
    if (graph) cudaGraphDestroy(graph);
    throw;
  }
  check(cudaGraphExecDestroy(executable));
  check(cudaGraphDestroy(graph));
}

/** Exercise both response consumers through the production pipeline, including
 * a captured signed direction changed between replays. CPU physical-potential
 * differences at two steps provide an oracle independent of response AD. */
void matrix_response_case(const AoBasis& basis, const MolecularGrid& grid, unsigned functional,
                          bool unrestricted, std::size_t tile) {
  Fixture response(basis, grid, functional, unrestricted, tile, CudaXcAoPrecision::Fp64, true);
  const auto d = density(basis.nao, unrestricted ? 2 : 1);
  std::vector<double> direction(d.size());
  const auto n = basis.nao, matrix = n * n, spins = response.layout.spins;
  for (std::size_t spin = 0; spin < spins; ++spin) {
    direction[spin * matrix] = -0.013 * (spin + 1);
    direction[spin * matrix + 1] = direction[spin * matrix + n] = 0.007 * (spin + 1);
  }
  vibeqc::runtime::OwnedCudaBuffer<double> device_direction(0, d.size(), response.stream);
  check(cudaMemcpy(response.density, d.data(), d.size() * sizeof(double), cudaMemcpyHostToDevice));
  const auto independent = [&](double step) {
    auto perturbed = d;
    for (std::size_t i = 0; i < d.size(); ++i) perturbed[i] += step * direction[i];
    if (!unrestricted)
      return functional ? integrate_pbe_rks_with_tail(basis, grid, perturbed, 23).potential
                        : integrate_lda_xc_pw_rks(basis, grid, perturbed, 23).potential;
    const std::vector<double> a(perturbed.begin(), perturbed.begin() + matrix),
        b(perturbed.begin() + matrix, perturbed.end());
    const auto ref = functional ? integrate_pbe_uks(basis, grid, a, b, 23)
                                : integrate_lda_xc_pw_uks(basis, grid, a, b, 23);
    auto potential = ref.potential[0];
    potential.insert(potential.end(), ref.potential[1].begin(), ref.potential[1].end());
    return potential;
  };
  cudaGraph_t graph{};
  cudaGraphExec_t executable{};
  try {
    check(cudaStreamBeginCapture(response.stream, cudaStreamCaptureModeGlobal));
    response.plan->enqueue_response(response.density, device_direction.get(), d.size(),
                                    ++response.generation);
    check(cudaStreamEndCapture(response.stream, &graph));
    check(cudaGraphInstantiate(&executable, graph, 0));
    for (unsigned replay = 0; replay < 2; ++replay) {
      check(cudaMemcpyAsync(device_direction.get(), direction.data(), d.size() * sizeof(double),
                            cudaMemcpyHostToDevice, response.stream));
      check(cudaGraphLaunch(executable, response.stream));
      require(response.scalars().error == 0, "signed matrix response was rejected");
      const auto actual = response.potential();
      for (double step : {1e-4, 3e-5}) {
        const auto plus = independent(step), minus = independent(-step);
        for (std::size_t i = 0; i < actual.size(); ++i)
          close(actual[i], (plus[i] - minus[i]) / (2 * step),
                "signed matrix response versus independent potential difference", 1e-7);
      }
      response.canary();
      for (double& value : direction) value *= -0.4;
    }
    // Exact vacuum with zero tangent has an independently known zero response.
    // Reusing the same captured entry also catches stale point coefficients.
    check(cudaMemsetAsync(response.density, 0, d.size() * sizeof(double), response.stream));
    check(cudaMemsetAsync(device_direction.get(), 0, d.size() * sizeof(double), response.stream));
    check(cudaGraphLaunch(executable, response.stream));
    require(response.scalars().error == 0, "vacuum matrix response was rejected");
    for (double value : response.potential())
      require(value == 0.0, "vacuum matrix response retained a stale coefficient");
    response.canary();
  } catch (...) {
    if (executable) cudaGraphExecDestroy(executable);
    if (graph) cudaGraphDestroy(graph);
    throw;
  }
  check(cudaGraphExecDestroy(executable));
  check(cudaGraphDestroy(graph));
}

void matrix_schedule_cases() {
  // Cross the generated matrix-tile boundary with two distinct f shells.
  // Cartesian/spherical shapes and partial point tiles exercise both matrix
  // tails and the final scalar fallback, using the independent CPU integrator.
  for (bool cooperative : {false, true})
    for (bool spherical : {false, true}) {
      auto large = system(3, spherical);
      large.shells.push_back({0, 3, {{0.51, 1.0}}});
      large.shells.push_back({0, 0, {{0.22, 1.0}}});
      if (cooperative) {
        // Cross the warp-reduction boundary with partial final AO groups.
        large.shells.push_back({0, 3, {{0.39, 1.0}}});
        large.shells.push_back({1, 3, {{0.32, 1.0}}});
        large.shells.push_back({1, 2, {{0.27, 1.0}}});
      }
      const AoBasis large_basis(large);
      require(!cooperative || large_basis.nao > 32, "cooperative fixture must span a warp");
      require(large_basis.nao >= 16 && large_basis.nao % 16 != 0,
              "matrix schedule fixture must have a partial AO block");
      const MolecularGrid large_grid(large, {1, 3, 3, 4, 3, 1e-12});
      for (std::uint32_t functional : {0U, 1U, 2U})
        for (bool uks : {false, true})
          for (std::size_t tile : {17U, 31U, 64U}) {
            Fixture test(large_basis, large_grid, functional, uks, tile);
            compare(test, large_basis, large_grid, density(large_basis.nao, uks ? 2 : 1));
          }
      graph_capture(large_basis, large_grid, 1U, false, 33);
      for (unsigned functional : {0U, 1U})
        matrix_response_case(large_basis, large_grid, functional, true, 19);
      for (std::uint32_t functional : {0U, 1U, 2U})
        variational_and_state(large_basis, large_grid, functional, 17);
    }
}
}  // namespace

int main(int argc, char** argv) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  try {
    matrix_schedule_cases();
    if (argc == 2 && std::string(argv[1]) == "--matrix-schedule") {
      std::cout << "CUDA XC matrix-tail, spin, functional, variational and capture gates passed\n";
      return 0;
    }
    const auto molecule = system();
    const AoBasis basis(molecule);
    const MolecularGrid grid(molecule, {1, 2, 2, 4, 3, 1e-12});
    for (unsigned functional : {0U, 1U, 2U, 3U, 4U})
      for (bool unrestricted : {false, true}) {
        graph_capture(basis, grid, functional, unrestricted);
        if (functional < 2U)
          for (std::size_t tile : {1U, 7U, 129U})
            matrix_response_case(basis, grid, functional, unrestricted, tile);
      }
    {
      for (bool unrestricted : {false, true})
        for (std::uint32_t functional : {0U, 1U}) {
          Fixture strict(basis, grid, functional, unrestricted, 9, CudaXcAoPrecision::Fp64);
          Fixture mixed(basis, grid, functional, unrestricted, 9,
                        CudaXcAoPrecision::Fp32ComputeFp64Storage);
          require(strict.layout.device_bytes == mixed.layout.device_bytes,
                  "FP32 AO compute candidate changed FP64 XC workspace");
          const auto d = density(basis.nao, unrestricted ? 2 : 1);
          strict.submit(d);
          mixed.submit(d);
          const auto strict_scalars = strict.scalars();
          const auto mixed_scalars = mixed.scalars();
          require(strict_scalars.error == 0 && mixed_scalars.error == 0,
                  "mixed AO candidate failed device XC evaluation");
          const auto energy_error = std::abs(mixed_scalars.energy - strict_scalars.energy);
          const auto electron_error =
              std::abs((mixed_scalars.electrons[0] + mixed_scalars.electrons[1]) -
                       (strict_scalars.electrons[0] + strict_scalars.electrons[1]));
          require(energy_error < 5e-8,
                  "FP32-compute AO semilocal energy exceeded the qualification gate");
          require(electron_error < 1e-7,
                  "FP32-compute AO electron count exceeded the qualification gate");
          const auto strict_v = strict.potential();
          const auto mixed_v = mixed.potential();
          double max_v_error = 0.0;
          for (std::size_t i = 0; i < strict_v.size(); ++i)
            max_v_error = std::max(max_v_error, std::abs(mixed_v[i] - strict_v[i]));
          require(max_v_error < 1e-7,
                  "FP32-compute AO semilocal potential exceeded the qualification gate");
          strict.canary();
          mixed.canary();
        }
      bool r2scan_rejected = false;
      try {
        (void)cuda_xc_layout(basis, grid, 2U, false, 9, CudaXcAoPrecision::Fp32ComputeFp64Storage);
      } catch (const std::invalid_argument&) {
        r2scan_rejected = true;
      }
      require(r2scan_rejected, "unqualified r2SCAN FP32-compute AO candidate was accepted");
      bool response_rejected = false;
      try {
        (void)cuda_xc_layout_shape(basis.natom, basis.nprimitive, basis.nao, grid.point_count(), 1U,
                                   false, 9, true, CudaXcAoPrecision::Fp32ComputeFp64Storage);
      } catch (const std::invalid_argument&) {
        response_rejected = true;
      }
      require(response_rejected, "unqualified response FP32-compute AO candidate was accepted");
    }
    // PBE0's semilocal branch is 0.75 PBE exchange + full PBE correlation.
    // This gate isolates CUDA XC scaling from exact-K composition in the next stack layer.
    for (bool uks : {false, true}) {
      Fixture scaled_pbe(basis, grid, 1U, uks, 17, CudaXcAoPrecision::Fp64, false, 0.75, 1.0);
      compare(scaled_pbe, basis, grid, density(basis.nao, uks ? 2 : 1));
    }
    bool scaled_r2scan_rejected = false;
    try {
      (void)cuda_xc_layout(basis, grid, 2U, false, 17, CudaXcAoPrecision::Fp64, 0.75, 1.0);
    } catch (const std::invalid_argument&) {
      scaled_r2scan_rejected = true;
    }
    require(scaled_r2scan_rejected, "unqualified scaled meta-GGA CUDA XC was accepted");

    for (std::uint32_t functional : {0U, 1U, 2U, 3U, 4U}) {
      for (bool uks : {false, true}) {
        for (std::size_t tile : {1U, 7U, 64U}) {
          Fixture test(basis, grid, functional, uks, tile);
          require(test.layout.jets == (functional == 0U ? 1U : 4U),
                  "unused AO jets were allocated");
          require(test.layout.work_jets == ((functional == 2U || functional == 4U) ? 4U : 1U),
                  "unused density-work jets were allocated");
          require(test.layout.feature_terms ==
                      (functional == 0U ? 1U : ((functional == 2U || functional == 4U) ? 5U : 4U)),
                  "CUDA XC feature layout does not match the functional");
          compare(test, basis, grid, density(basis.nao, uks ? 2 : 1));
        }
      }
      variational_and_state(basis, grid, functional);
      const MolecularGrid tail_grid(molecule);
      Fixture tail(basis, tail_grid, functional, true, 257);
      auto fully = density(basis.nao, 2);
      std::fill(fully.begin() + basis.nao * basis.nao, fully.end(), 0.0);
      const auto same_input = (functional == 2U || functional == 4U)
                                  ? empty_spin_reference(basis, tail_grid, fully, functional)
                                  : std::vector<double>{};
      compare(tail, basis, tail_grid, fully, same_input);
      if (functional == 2U) {
        // Exercise both minority channels; exchange the reference spin blocks.
        std::rotate(fully.begin(), fully.begin() + basis.nao * basis.nao, fully.end());
        auto flipped = same_input;
        std::rotate(flipped.begin(), flipped.begin() + basis.nao * basis.nao, flipped.end());
        compare(tail, basis, tail_grid, fully, flipped);
      }
      compare(tail, basis, tail_grid, std::vector<double>(fully.size()));
    }
    for (bool spherical : {false, true}) {
      const auto f = system(3, spherical);
      const AoBasis f_basis(f);
      const MolecularGrid f_grid(f, {1, 3, 3, 4, 3, 1e-12});
      Fixture f_test(f_basis, f_grid, 1U, true, 13);
      compare(f_test, f_basis, f_grid, density(f_basis.nao, 2));
    }
    // Independent PR #214 same-grid PySCF/Libxc fixture, not just CPU parity.
    Fixture independent(basis, grid, 1U, false, 9);
    independent.submit(
        {1.2007575959127958, 0.011302590336886256, 0.011302590336886256, 0.462144452714005});
    require(independent.scalars().error == 0, "independent reference density was rejected");
    close(independent.scalars().energy, -0.23010116952210713, "independent GPU PBE E", 1e-11);
    const double oracle[]{-0.1903934858683413, -0.07901244667607567, -0.07901244667607567,
                          -0.15198583764323192};
    const auto v = independent.potential();
    for (std::size_t i = 0; i < v.size(); ++i)
      close(v[i], oracle[i], "independent GPU PBE V", 1e-11);
    // Unequal-spin values from the same independent #214 h2.npz fixture.
    // Keep both potentials: equal-spin reduction alone cannot detect a spin
    // degeneracy or mixed-sigma factor error in a UKS consumer.
    const std::vector<double> spin_density{
        0.8754226489181763, 0.13425021134730328,  0.13425021134730328,  0.1929077645192876,
        0.3253349469946195, -0.12294762101041702, -0.12294762101041702, 0.2692366881947174};
    const double spin_energy[]{-0.23601166477391342, -0.23920047881983975};
    const double spin_oracle[2][8]{
        {-0.20988426234150148, -0.08482648070126539, -0.08482648070126539, -0.15648657460357385,
         -0.15888935561902634, -0.06989855886500512, -0.06989855886500512, -0.14358097629106337},
        {-0.21238938797091092, -0.08600503368570547, -0.08600503368570547, -0.15759933292601364,
         -0.15871394325917046, -0.06875820189039786, -0.06875820189039786, -0.14429309123882408}};
    for (unsigned pbe = 0; pbe < 2; ++pbe) {
      Fixture independent_spin(basis, grid, pbe, true, 9);
      independent_spin.submit(spin_density);
      const auto scalars = independent_spin.scalars();
      require(scalars.error == 0, "independent spin reference density was rejected");
      close(scalars.energy, spin_energy[pbe], "independent GPU UKS E", 1e-11);
      close(scalars.electrons[0], 0.3798061637510074, "independent alpha population", 1e-11);
      close(scalars.electrons[1], 0.15110624456362898, "independent beta population", 1e-11);
      const auto potential = independent_spin.potential();
      for (std::size_t i = 0; i < potential.size(); ++i)
        close(potential[i], spin_oracle[pbe][i], "independent GPU UKS V", 1e-11);
    }
    auto changed = molecule;
    changed.atoms[1].position[2] += 0.1;
    bool stale = false;
    try {
      (void)cuda_xc_layout(basis, MolecularGrid(changed), 1U, false);
    } catch (const std::invalid_argument&) {
      stale = true;
    }
    require(stale, "same-shape stale grid identity was accepted");
    std::cout << "Native device-buffer LDA/PBE/r2SCAN/WB97M-V semilocal RKS/UKS E/V and state "
                 "gates passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
