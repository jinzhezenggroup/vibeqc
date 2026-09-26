// Host-only probe of the actual CUDA descriptor admission, without a GPU or libvibeqc.
#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <string>

#include "methods/dft_admission.hpp"

int main() {
  struct Method {
    const char* exchange;
    const char* correlation;
    double exact;
  };
  const std::array<Method, 2> methods{{
      {"HYB_MGGA_X_M06_2X", "MGGA_C_M06_2X", 27.0 / 50.0},
      {"HYB_MGGA_X_MN15", "MGGA_C_MN15", 11.0 / 25.0},
  }};
  for (const auto& method : methods) {
    for (unsigned spin : {1U, 2U}) {
      std::array<vibeqc_ks_semilocal_component, 2> components{{
          {method.exchange, 1.0},
          {method.correlation, 1.0},
      }};
      vibeqc_ks_exchange_term exchange{VIBEQC_KS_EXCHANGE_FULL_RANGE, method.exact, 0.0,
                                       -method.exact / (spin == 1 ? 2.0 : 1.0)};
      vibeqc_ks_options options{};
      options.struct_size = sizeof(options);
      options.abi_version = VIBEQC_ABI_VERSION;
      options.spin_channels = spin;
      options.semilocal_components = components.data();
      options.semilocal_component_count = components.size();
      options.exchange_terms = &exchange;
      options.exchange_term_count = 1;
      options.xc_execution_schedule = VIBEQC_XC_EXECUTION_DEVICE_FUSED;
      vibeqc_method_descriptor descriptor{};
      descriptor.method = VIBEQC_METHOD_M06_2X_RKS;
      descriptor.ks_options = &options;
      descriptor.precision_mode = VIBEQC_PRECISION_FP64;
      descriptor.density_fitting_mode = VIBEQC_DENSITY_FITTING_NONE;
      const auto check = [&](vibeqc_status expected, const char* label,
                             vibeqc_backend backend = VIBEQC_BACKEND_CUDA) {
        std::string detail;
        const auto status =
            vibeqc::methods::detail::validate_split_hybrid_descriptor(descriptor, backend, detail);
        if (status != expected || (status != VIBEQC_STATUS_SUCCESS && detail.empty())) {
          std::cerr << label << ": " << status << " " << detail << '\n';
          return false;
        }
        return true;
      };
      if (!check(VIBEQC_STATUS_SUCCESS, "canonical")) return 1;
      std::swap(components[0], components[1]);
      if (!check(VIBEQC_STATUS_SUCCESS, "reordered components")) return 1;
      std::swap(components[0], components[1]);
      if (!check(VIBEQC_STATUS_NOT_IMPLEMENTED, "CPU", VIBEQC_BACKEND_CPU_REFERENCE)) return 1;
      options.exchange_terms = nullptr;
      options.exchange_term_count = 0;
      if (!check(VIBEQC_STATUS_NOT_IMPLEMENTED, "missing K")) return 1;
      options.exchange_term_count = 1;
      if (!check(VIBEQC_STATUS_INVALID_ARGUMENT, "null K pointer")) return 1;
      options.exchange_terms = &exchange;
      const auto original_exchange = exchange;
      exchange.coefficient = exchange.fock_coefficient = 0.0;
      if (!check(VIBEQC_STATUS_NOT_IMPLEMENTED, "zero K")) return 1;
      exchange = original_exchange;
      exchange.fock_coefficient *= 2.0;
      if (!check(VIBEQC_STATUS_NOT_IMPLEMENTED, "spin factor")) return 1;
      exchange = original_exchange;
      exchange.coefficient = std::numeric_limits<double>::quiet_NaN();
      if (!check(VIBEQC_STATUS_NOT_IMPLEMENTED, "nonfinite K")) return 1;
      exchange = original_exchange;
      exchange.omega = 0.3;
      if (!check(VIBEQC_STATUS_NOT_IMPLEMENTED, "range exchange")) return 1;
      exchange = original_exchange;
      options.has_nonlocal_correlation = 1;
      if (!check(VIBEQC_STATUS_NOT_IMPLEMENTED, "extra NLC")) return 1;
      options.has_nonlocal_correlation = 0;
      components[0].coefficient = 0.5;
      if (!check(VIBEQC_STATUS_NOT_IMPLEMENTED, "scaled component")) return 1;
      components[0].coefficient = 1.0;
      options.semilocal_range_omega = 0.3;
      if (!check(VIBEQC_STATUS_NOT_IMPLEMENTED, "semilocal range")) return 1;
      options.semilocal_range_omega = 0.0;
      options.xc_execution_schedule = VIBEQC_XC_EXECUTION_HOST_UNFUSED;
      if (!check(VIBEQC_STATUS_NOT_IMPLEMENTED, "host XC")) return 1;
      options.xc_execution_schedule = VIBEQC_XC_EXECUTION_DEVICE_FUSED;
      descriptor.precision_mode = VIBEQC_PRECISION_AUTO;
      if (!check(VIBEQC_STATUS_NOT_IMPLEMENTED, "mixed precision")) return 1;
      descriptor.precision_mode = VIBEQC_PRECISION_FP64;
      descriptor.density_fitting_mode = VIBEQC_DENSITY_FITTING_CUDA;
      if (!check(VIBEQC_STATUS_NOT_IMPLEMENTED, "density fitting")) return 1;
      descriptor.density_fitting_mode = VIBEQC_DENSITY_FITTING_NONE;
      options.struct_size = 0;
      if (!check(VIBEQC_STATUS_ABI_MISMATCH, "short ABI")) return 1;
    }
  }
  std::cout << "split-hybrid descriptor admission passed\n";
}
