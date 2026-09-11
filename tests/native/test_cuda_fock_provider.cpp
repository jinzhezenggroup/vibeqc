#include <cuda_runtime_api.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "molecule/basis.hpp"
#include "scf/cuda_density_fitting.hpp"
#include "scf/cuda_direct_jk.hpp"
#include "scf/density_fitting.hpp"

namespace {
using namespace vibeqc::scf;
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t status) { require(status == cudaSuccess, cudaGetErrorString(status)); }
struct DeviceMatrix {
  double* pointer{};
  explicit DeviceMatrix(const std::vector<double>& input) {
    check(cudaMalloc(reinterpret_cast<void**>(&pointer), input.size() * sizeof(double)));
    const auto status =
        cudaMemcpy(pointer, input.data(), input.size() * sizeof(double), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
      cudaFree(pointer);
      check(status);
    }
  }
  ~DeviceMatrix() { cudaFree(pointer); }
  DeviceMatrix(const DeviceMatrix&) = delete;
  DeviceMatrix& operator=(const DeviceMatrix&) = delete;
  void verify(const std::vector<double>& expected) const {
    std::vector<double> actual(expected.size());
    check(
        cudaMemcpy(actual.data(), pointer, actual.size() * sizeof(double), cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < actual.size(); ++i)
      require(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) < 3e-12,
              "independent device DF matrix differs from CPU");
  }
};

void device_selection() {
  const std::vector<double> metric{2.0, 0.1, 0.1, 1.3};
  const std::vector<double> tensor{1.3, 0.2, 0.3, -0.1, 0.3, -0.1, 0.8, 0.6};
  const std::vector<double> a{1.2, 0.31, -0.07, 0.7}, b{0.1, -0.05, 0.13, 0.4};
  const auto transformed = orthonormalize_density_fitting_three_center(
      tensor, 2, factor_density_fitting_metric(metric, 2));
  const auto rhf = build_density_fitting_rhf_jk(transformed, a);
  const auto uhf = build_density_fitting_uhf_jk(transformed, a, b);
  for (const auto layout : {FockMatrixLayout::RowMajor, FockMatrixLayout::ColumnMajor})
    for (std::size_t tile : {0U, 1U}) {
      CudaDensityFittingJkPlan* raw{};
      std::string detail;
      std::vector<CudaDensityFittingMetricDiagnostic> diagnostics;
      require(create_cuda_density_fitting_jk_plan_tiled(0, 1, 2, 2, metric, tensor, 1e-10, tile,
                                                        tile, &raw, diagnostics,
                                                        detail) == VIBEQC_STATUS_SUCCESS,
              detail.c_str());
      std::unique_ptr<CudaDensityFittingJkPlan, decltype(&destroy_cuda_density_fitting_jk_plan)>
          plan(raw, &destroy_cuda_density_fitting_jk_plan);
      auto input_a = a, input_b = b;
      if (layout == FockMatrixLayout::ColumnMajor) {
        std::swap(input_a[1], input_a[2]);
        std::swap(input_b[1], input_b[2]);
      }
      DeviceMatrix da(input_a), db(input_b);
      for (bool j : {false, true})
        for (bool k : {false, true}) {
          const JkTermSelection terms{j, k};
          const std::vector<double> sentinel(4, 123.0);
          DeviceMatrix dj(sentinel), dka(sentinel), dkb(sentinel);
          require(execute_cuda_density_fitting_rhf_jk_device(
                      plan.get(), da.pointer, j ? dj.pointer : nullptr, k ? dka.pointer : nullptr,
                      detail, terms, layout) == VIBEQC_STATUS_SUCCESS,
                  detail.c_str());
          check(cudaDeviceSynchronize());  // Test boundary observes the plan's nonblocking stream.
          dj.verify(j ? rhf.coulomb : sentinel);
          dka.verify(k ? rhf.exchange : sentinel);
          // Give unselected outputs valid sentinels this time: the service must
          // neither write them nor assume non-null means a term was requested.
          require(execute_cuda_density_fitting_uhf_jk_device(
                      plan.get(), da.pointer, db.pointer, dj.pointer, dka.pointer, dkb.pointer,
                      detail, terms, layout) == VIBEQC_STATUS_SUCCESS,
                  detail.c_str());
          check(cudaDeviceSynchronize());
          dj.verify(j ? uhf.coulomb : sentinel);
          dka.verify(k ? uhf.alpha_exchange : sentinel);
          dkb.verify(k ? uhf.beta_exchange : sentinel);
        }
      require(execute_cuda_density_fitting_rhf_jk_device(plan.get(), da.pointer, nullptr, nullptr,
                                                         detail, {true, false}) ==
                  VIBEQC_STATUS_INVALID_ARGUMENT,
              "missing selected device J accepted");
      require(execute_cuda_density_fitting_uhf_jk_device(
                  plan.get(), da.pointer, db.pointer, nullptr, nullptr, nullptr, detail,
                  {false, true}) == VIBEQC_STATUS_INVALID_ARGUMENT,
              "missing selected device K accepted");
      da.verify(input_a);
      db.verify(input_b);
    }
}

void direct_providers(bool through_f_response) {
  for (unsigned angular : {0U, 1U, 2U, 3U})
    for (auto representation : {VIBEQC_BASIS_CARTESIAN, VIBEQC_BASIS_SPHERICAL}) {
      // Noncoincident centers and unequal primitive/basis metadata distinguish
      // items with identical dimensions. Through-f values also test public AO
      // expansion ordering. The optional numerical tier adds d/f response
      // without repeating expensive high-angular CPU derivatives for every mask.
      vibeqc::core::System first;
      first.atoms = {{1, {0.0, 0.1, -0.7}}, {1, {0.2, -0.1, 0.7}}};
      first.shells = {{0, 0, {{0.8, 0.7}, {0.2, 0.3}}}, {1, angular, {{0.6, 1.0}}}};
      first.electron_count = 2;
      first.basis_representation = representation;
      auto second = first;
      second.atoms[1].position[2] += 0.13;
      second.shells[1].primitives[0].exponent = 0.9;
      std::string detail;
      require(vibeqc::molecule::validate_and_normalize(first, detail) == VIBEQC_STATUS_SUCCESS,
              detail.c_str());
      require(vibeqc::molecule::validate_and_normalize(second, detail) == VIBEQC_STATUS_SUCCESS,
              detail.c_str());
      const bool derivatives = angular < 2 || through_f_response;
      const auto ints = vibeqc::integrals::build_integrals(first, derivatives);
      const auto other = vibeqc::integrals::build_integrals(second, derivatives);
      const std::size_t n = ints.nbf, matrix = n * n;
      std::vector<double> a(matrix), b(matrix), packed_a, packed_b;
      for (std::size_t ij = 0; ij < matrix; ++ij) {
        a[ij] = std::cos(0.3 * (ij / n) + 0.7 * (ij % n)) / n;
        b[ij] = std::sin(0.8 * (ij / n) - 0.2 * (ij % n)) / n;
      }
      packed_a = a;
      packed_a.insert(packed_a.end(), a.begin(), a.end());
      packed_b = b;
      packed_b.insert(packed_b.end(), b.begin(), b.end());
      CudaDirectJkPlan* raw{};
      CudaDirectJkDiagnostic diagnostic;
      require(create_cuda_direct_jk_plan(0, {first, second}, derivatives ? 1 : 0, 0.0,
                                         64U * 1024U * 1024U, &raw, diagnostic,
                                         detail) == VIBEQC_STATUS_SUCCESS,
              detail.c_str());
      std::unique_ptr<CudaDirectJkPlan, decltype(&destroy_cuda_direct_jk_plan)> plan(
          raw, &destroy_cuda_direct_jk_plan);
      require(diagnostic.nbf == n && diagnostic.batch_size == 2 && diagnostic.device_bytes > 0 &&
                  diagnostic.coordinates_per_item == 6 &&
                  diagnostic.host_preparation_bytes >= diagnostic.host_bytes,
              "direct provider diagnostics mismatch");
      for (bool uhf : {false, true})
        for (bool j : {false, true})
          for (bool k : {false, true}) {
            const bool response = derivatives && (angular < 2 || (uhf && j && k));
            auto spec = make_hf_fock_spec(uhf ? FockSpin::Unrestricted : FockSpin::Restricted);
            spec.derivative_order = derivatives ? 1 : 0;
            spec.coulomb.present = j;
            spec.exchange.present = k;
            spec.coulomb.coefficient = 1.7;
            spec.exchange.coefficient = -0.23;
            const auto cpu = resolve_fock_build(spec, FockBackend::Cpu, 0.0);
            std::vector<double> dj, dka, dkb;
            require(execute_cuda_direct_jk(plan.get(), spec, packed_a,
                                           uhf ? packed_b : std::vector<double>{}, dj, dka, dkb,
                                           detail) == VIBEQC_STATUS_SUCCESS,
                    detail.c_str());
            std::vector<double> ej, eka, ekb, gradient;
            for (const auto* data : {&ints, &other}) {
              const auto expected =
                  build_exact_direct_jk(cpu, n, data->eri, a, uhf ? b : std::vector<double>{});
              ej.insert(ej.end(), expected.coulomb.begin(), expected.coulomb.end());
              eka.insert(eka.end(), expected.exchange_alpha.begin(), expected.exchange_alpha.end());
              ekb.insert(ekb.end(), expected.exchange_beta.begin(), expected.exchange_beta.end());
              if (response)
                for (std::size_t coordinate = 0; coordinate < 6; ++coordinate)
                  gradient.push_back(contract_exact_direct_energy_derivative(
                      cpu, n,
                      std::span(data->eri_derivative)
                          .subspan(coordinate * matrix * matrix, matrix * matrix),
                      a, uhf ? b : std::vector<double>{}));
            }
            auto compare = [&](const auto& actual, const auto& expected) {
              require(actual.size() == expected.size(), "direct J/K selected shape mismatch");
              for (std::size_t i = 0; i < actual.size(); ++i)
                require(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) < 3e-10,
                        ("direct CUDA mismatch l=" + std::to_string(angular) +
                         " representation=" + std::to_string(representation) +
                         " spin=" + std::to_string(uhf) + " J=" + std::to_string(j) +
                         " K=" + std::to_string(k) + " element=" + std::to_string(i))
                            .c_str());
            };
            compare(dj, ej);
            compare(dka, eka);
            compare(dkb, ekb);
            std::vector<double> actual_gradient;
            if (response || !derivatives) {
              const auto status = execute_cuda_direct_energy_derivative(
                  plan.get(), spec, packed_a, uhf ? packed_b : std::vector<double>{},
                  actual_gradient, detail);
              if (response) {
                require(status == VIBEQC_STATUS_SUCCESS, detail.c_str());
                compare(actual_gradient, gradient);
              } else
                require(status == VIBEQC_STATUS_INVALID_ARGUMENT,
                        "unrequested direct derivatives executed");
            }
          }
      auto fitted = make_hf_fock_spec(FockSpin::Restricted, FockApproximation::DensityFitted);
      std::vector<double> j{123.0}, ka, kb;
      require(execute_cuda_direct_jk(plan.get(), fitted, packed_a, {}, j, ka, kb, detail) ==
                      VIBEQC_STATUS_INVALID_ARGUMENT &&
                  j == std::vector<double>{123.0},
              "direct provider accepted fitted semantics or published partial output");
      CudaDirectJkPlan* too_small{};
      require(create_cuda_direct_jk_plan(0, {first}, 0, 0.0, 1, &too_small, diagnostic, detail) ==
                      VIBEQC_STATUS_OUT_OF_MEMORY &&
                  !too_small,
              "direct source ignored its buffer budget");
      auto invalid = first;
      invalid.atoms[0].position[0] = std::numeric_limits<double>::quiet_NaN();
      require(create_cuda_direct_jk_plan(0, {invalid}, 0, 0.0, 64U * 1024U * 1024U, &too_small,
                                         diagnostic, detail) == VIBEQC_STATUS_INVALID_ARGUMENT &&
                  !too_small,
              "nonfinite direct source geometry accepted");
      if (angular == 0 && representation == VIBEQC_BASIS_CARTESIAN) {
        // Inputs are finite but deliberately outside the stable numerical
        // range. A failed bound must not silently screen the entire source.
        invalid = first;
        invalid.shells[0].primitives[0].coefficient = 1e200;
        require(create_cuda_direct_jk_plan(0, {invalid}, 0, 0.0, 64U * 1024U * 1024U, &too_small,
                                           diagnostic, detail) == VIBEQC_STATUS_NUMERICAL_FAILURE &&
                    !too_small,
                "nonfinite direct bound published a source");
        auto spec = make_hf_fock_spec(FockSpin::Restricted);
        auto huge = packed_a;
        std::fill(huge.begin(), huge.end(), std::numeric_limits<double>::max());
        require(execute_cuda_direct_jk(plan.get(), spec, huge, {}, j, ka, kb, detail) ==
                        VIBEQC_STATUS_NUMERICAL_FAILURE &&
                    j == std::vector<double>{123.0},
                "nonfinite direct matrix published partial output");
        std::fill(huge.begin(), huge.end(), 1e200);
        std::vector<double> gradient{123.0};
        require(execute_cuda_direct_energy_derivative(plan.get(), spec, huge, {}, gradient,
                                                      detail) == VIBEQC_STATUS_NUMERICAL_FAILURE &&
                    gradient == std::vector<double>{123.0},
                "nonfinite direct derivative published partial output");
        auto opposite = huge;
        for (auto& value : opposite) value = -value;
        spec.spin = FockSpin::Unrestricted;
        spec.exchange.present = false;
        require(execute_cuda_direct_energy_derivative(plan.get(), spec, huge, opposite, gradient,
                                                      detail) == VIBEQC_STATUS_SUCCESS,
                "absent direct exchange contaminated finite total-density response");
        for (double value : gradient) require(value == 0.0, "zero total-density response changed");
      }
    }
}
}  // namespace
int main(int argc, char** argv) {
  try {
    require(argc == 1 || (argc == 2 && std::string(argv[1]) == "--through-f-response"),
            "expected optional --through-f-response");
    const bool through_f_response = argc == 2;
    device_selection();
    direct_providers(through_f_response);
    std::cout << "CUDA independent J/K: DF layouts/selection and direct through-f values, "
              << (through_f_response ? "through-f" : "s/p") << " derivatives PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
