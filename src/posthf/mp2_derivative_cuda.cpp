#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

#include "posthf/cuda_derivative.hpp"
#include "posthf/mp2_derivative.hpp"
#include "posthf/mp2_derivative_common.hpp"
#include "runtime/df_progress_trace.hpp"
#include "scf/cuda/rhf_policy.hpp"
#include "scf/cuda_one_electron_gradient.hpp"

namespace generativeqc::mp2 {
#if GENERATIVEQC_HAS_CUDA
namespace {
void check_cuda_derivative(generativeqc_status status, const std::string& detail) {
  if (status == GENERATIVEQC_STATUS_SUCCESS) return;
  const auto message = detail.empty() ? "CUDA conventional derivative failed" : detail;
  if (status == GENERATIVEQC_STATUS_OUT_OF_MEMORY) throw std::length_error(message);
  if (status == GENERATIVEQC_STATUS_INVALID_ARGUMENT) throw std::invalid_argument(message);
  throw std::runtime_error(message);
}

void add_nuclear_repulsion_gradient(const core::System& system, std::vector<double>& gradient) {
  for (std::size_t a = 0; a < system.atoms.size(); ++a)
    for (std::size_t b = 0; b < a; ++b) {
      double distance2 = 0.0;
      std::array<double, 3> displacement{};
      for (std::size_t axis = 0; axis < 3; ++axis) {
        displacement[axis] = system.atoms[a].position[axis] - system.atoms[b].position[axis];
        distance2 += displacement[axis] * displacement[axis];
      }
      if (!(distance2 > 0.0) || !std::isfinite(distance2))
        throw std::invalid_argument("nuclear repulsion derivative has coincident atoms");
      const double factor =
          static_cast<double>(system.atoms[a].ionic_charge() * system.atoms[b].ionic_charge()) /
          (distance2 * std::sqrt(distance2));
      for (std::size_t axis = 0; axis < 3; ++axis) {
        const double value = factor * displacement[axis];
        gradient[3 * a + axis] -= value;
        gradient[3 * b + axis] += value;
      }
    }
}
}  // namespace
#endif

std::vector<double> conventional_derivative_cuda(const core::System& system,
                                                 const hf::PhysicalReference& reference,
                                                 const LagrangianWeights& weights, int device_id,
                                                 std::size_t stage_budget) {
#if !GENERATIVEQC_HAS_CUDA
  (void)system;
  (void)reference;
  (void)weights;
  (void)device_id;
  (void)stage_budget;
  throw std::runtime_error("CUDA conventional derivative is unavailable in this build");
#else
  if (device_id < 0 || !stage_budget)
    throw std::invalid_argument("invalid CUDA conventional derivative request");
  runtime::df_progress::Scope trace("cuda_conventional_derivative");
  posthf::CudaShellDerivativeDiagnostic total;
  auto derivative = detail::conventional_derivative(
      system, reference, weights,
      [&](std::span<const double> overlap, std::span<const double> hcore) {
        std::vector<double> gradient;
        std::string detail;
        const auto status = scf::execute_cuda_one_electron_gradient(
            device_id, system, overlap, hcore, hcore,
            scf::cuda_policy::one_electron_derivative_mapping_requested(), stage_budget, gradient,
            detail);
        check_cuda_derivative(status, detail);
        add_nuclear_repulsion_gradient(system, gradient);
        return gradient;
      },
      [&](const std::array<std::size_t, 4>& shells, std::span<const double> local) {
        std::array<double, 12> center{};
        std::string detail;
        posthf::CudaShellDerivativeDiagnostic diagnostic;
        const auto status = posthf::contract_weighted_eri_shell_derivative_cuda(
            device_id, system, shells, local, stage_budget, center, detail,
            trace.enabled() ? &diagnostic : nullptr);
        check_cuda_derivative(status, detail);
        total.primitive_records += diagnostic.primitive_records;
        total.consumer_calls += diagnostic.consumer_calls;
        total.consumer_nanoseconds += diagnostic.consumer_nanoseconds;
        return center;
      });
  if (trace.enabled()) {
    // These completed consumer calls include record upload, primitive kernels,
    // result download, and stream/allocation lifetime. Host record construction
    // belongs to the enclosing shell callback, not consumer_ns.
    using runtime::df_progress::Scope;
    Scope::number("primitive_records", total.primitive_records);
    Scope::number("consumer_calls", total.consumer_calls);
    Scope::number("consumer_ns", total.consumer_nanoseconds);
  }
  return derivative;
#endif
}

}  // namespace generativeqc::mp2
