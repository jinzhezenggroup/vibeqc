#include <cstdlib>
#include <string_view>

#include "generated_df_production.hpp"
#include "generated_df_shell_dispatch.hpp"
#include "runtime/cuda_architecture.hpp"

namespace vibeqc::scf {
namespace {
namespace generated = generated_df_shell;
/** Query a target once per host packet call. The generated manifest owns
 * architecture/class qualification; AO dimensions do not change that kernel
 * capability. Consumer/layout admission remains with the gradient bridge.
 * Unknown targets/classes and explicit legacy controls retain their fallback.
 */
cudaError_t production_target(unsigned& architecture) {
  architecture = 0;
  const char* control = std::getenv("VIBEQC_DF_SHELL_POLICY");
  const std::string_view policy = control ? control : "auto";
  if (policy != "auto" && policy != "legacy" && policy != "candidate") return cudaErrorInvalidValue;
  if (!generated::production_policy_available || policy == "legacy") return cudaSuccess;
  int device = 0;
  auto error = cudaGetDevice(&device);
  if (error == cudaSuccess) error = runtime::cuda_architecture(device, architecture);
  return error;
}

template <unsigned A, unsigned B, unsigned C>
DfShellLaunch select_launch(unsigned architecture, DfShellLaunch context) {
  const char* mapping = std::getenv("VIBEQC_DF_SHELL_POLICY");
  const bool candidate = mapping && std::string_view(mapping) == "candidate";
  const auto choice = generated::DfProductionPolicy<A, B, C>::select(architecture, candidate);
  const bool use_choice = choice.available && (choice.qualified || candidate);
  const char* schedule = std::getenv("VIBEQC_DF_SHELL_SCHEDULE");
  if (use_choice && (!schedule || std::string_view(schedule) == "auto"))
    context.variant = choice.variant;
  context.rys = use_choice && choice.rys;
  return context;
}
}  // namespace

cudaError_t launch_df_shell_derivative_panel(DfShellBasisView o, DfShellBasisView x,
                                             const double* positions, std::size_t begin,
                                             std::size_t count, const double* weights,
                                             double* gradient, unsigned long long* counters,
                                             cudaStream_t stream, bool full_domain,
                                             unsigned variant, DfDerivativePairs pairs,
                                             DfShellDiagnostics* diagnostics) {
  if (variant > 2) return cudaErrorInvalidValue;
  unsigned architecture = 0;
  auto status = production_target(architecture);
  if (status != cudaSuccess) return status;
  const DfShellLaunch context{positions, begin,  count,   weights, gradient,
                              counters,  stream, variant, pairs,   diagnostics};
  generated_df_dispatch::for_each_class(
      [&]<unsigned A, unsigned B, unsigned C>(const DfShellDispatch& entry) {
        if (status != cudaSuccess || (!full_domain && (A > 1 || B > 1 || C > 1 || A + B + C == 0)))
          return;
        status = entry.panel(o, x, select_launch<A, B, C>(architecture, context));
      });
  return status;
}

cudaError_t launch_df_shell_derivative_group(
    DfShellBasisView first, DfShellBasisView second, DfShellBasisView x, const double* positions,
    std::size_t begin, std::size_t count, const double* weights, double* gradient,
    unsigned long long* counters, cudaStream_t stream, bool full_domain, unsigned variant,
    DfDerivativePairs pairs, bool triangle, DfShellDiagnostics* diagnostics) {
  if (variant > 2) return cudaErrorInvalidValue;
  unsigned architecture = 0;
  auto status = production_target(architecture);
  if (status != cudaSuccess) return status;
  const DfShellLaunch context{positions, begin,  count,   weights, gradient,
                              counters,  stream, variant, pairs,   diagnostics};
  generated_df_dispatch::for_each_class([&]<unsigned A, unsigned B, unsigned C>(
                                            const DfShellDispatch& entry) {
    if (status != cudaSuccess || (!full_domain && (A > 1 || B > 1 || C > 1 || A + B + C == 0)))
      return;
    status = entry.group(first, second, x, triangle, select_launch<A, B, C>(architecture, context));
  });
  return status;
}

cudaError_t launch_df_shell_derivative_packets(
    std::span<const DfShellBasisView> orbital, std::span<const DfShellBasisView> auxiliary,
    const double* positions, std::size_t begin, std::size_t count, const double* weights,
    double* gradient, unsigned long long* counters, cudaStream_t stream, bool full_domain,
    unsigned variant, DfDerivativePairs pairs, DfShellDiagnostics* diagnostics) {
  if (variant > 2) return cudaErrorInvalidValue;
  if (orbital.empty() || auxiliary.empty()) return cudaSuccess;
  unsigned architecture = 0;
  auto status = production_target(architecture);
  if (status != cudaSuccess) return status;
  const DfShellLaunch context{positions, begin,  count,   weights, gradient,
                              counters,  stream, variant, pairs,   diagnostics};
  generated_df_dispatch::for_each_class(
      [&]<unsigned A, unsigned B, unsigned C>(const DfShellDispatch& entry) {
        if (status != cudaSuccess || (!full_domain && (A > 1 || B > 1 || C > 1 || A + B + C == 0)))
          return;
        status = entry.packets(orbital, auxiliary, select_launch<A, B, C>(architecture, context));
      });
  return status;
}
}  // namespace vibeqc::scf
