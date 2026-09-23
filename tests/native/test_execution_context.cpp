#include <cmath>
#include <iostream>
#include <stdexcept>

#include "methods/method.hpp"
#include "molecule/basis.hpp"
#include "runtime/execution_context.hpp"

namespace {
// These checks remain effective in Release/NDEBUG builds.
void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void verify_tracker() {
  vibeqc::core::ContextState state;
  state.requested_backend = VIBEQC_BACKEND_CUDA;
  state.device_id = 7;

  vibeqc::runtime::ExecutionContext execution(state);
  require(execution.backend() == VIBEQC_BACKEND_CUDA, "execution.backend() == VIBEQC_BACKEND_CUDA");
  require(execution.cuda_requested(), "execution.cuda_requested()");
  require(execution.device_id() == 7, "execution.device_id() == 7");

  execution.observe_numeric_peak(vibeqc::runtime::ExecutionMemorySpace::Host, 64);
  execution.observe_numeric_peak(vibeqc::runtime::ExecutionMemorySpace::Host, 32);
  execution.observe_numeric_peak(vibeqc::runtime::ExecutionMemorySpace::Device, 96);
  execution.observe_workspace_peak(vibeqc::runtime::ExecutionMemorySpace::Host, 24);
  execution.observe_workspace_peak(vibeqc::runtime::ExecutionMemorySpace::Device, 128);
  execution.observe_workspace_peak(vibeqc::runtime::ExecutionMemorySpace::Device, 80);
  execution.observe_resource_peak(vibeqc::runtime::ExecutionResourceKind::Pinned,
                                  vibeqc::runtime::ExecutionMemorySpace::Host, 48);
  execution.observe_resource_peak(vibeqc::runtime::ExecutionResourceKind::ProviderRetained,
                                  vibeqc::runtime::ExecutionMemorySpace::Device, 144);
  execution.observe_resource_peak(vibeqc::runtime::ExecutionResourceKind::ProviderRetained,
                                  vibeqc::runtime::ExecutionMemorySpace::Device, 96);
  execution.observe_resource_peak(vibeqc::runtime::ExecutionResourceKind::Staging,
                                  vibeqc::runtime::ExecutionMemorySpace::Host, 40);
  execution.observe_resource_peak(vibeqc::runtime::ExecutionResourceKind::CaptureRetained,
                                  vibeqc::runtime::ExecutionMemorySpace::Device, 256);

  const auto resources = execution.resources();
  require(resources.host_numeric_peak_bytes == 64, "resources.host_numeric_peak_bytes == 64");
  require(resources.device_numeric_peak_bytes == 96, "resources.device_numeric_peak_bytes == 96");
  require(resources.host_workspace_peak_bytes == 24, "resources.host_workspace_peak_bytes == 24");
  require(resources.device_workspace_peak_bytes == 128,
          "resources.device_workspace_peak_bytes == 128");
  require(resources.numeric_observations == 3, "resources.numeric_observations == 3");
  require(resources.workspace_observations == 3, "resources.workspace_observations == 3");

  const auto& numeric = resources.observation(vibeqc::runtime::ExecutionResourceKind::Numeric);
  require(numeric.host_peak_bytes == 64 && numeric.device_peak_bytes == 96,
          "numeric resource detail lost host/device peaks");
  require(numeric.host_observations == 2 && numeric.device_observations == 1,
          "numeric resource detail lost per-space observation counts");
  const auto& scratch = resources.observation(vibeqc::runtime::ExecutionResourceKind::Scratch);
  require(scratch.host_peak_bytes == 24 && scratch.device_peak_bytes == 128,
          "workspace compatibility did not map to scratch");
  require(scratch.host_observations == 1 && scratch.device_observations == 2,
          "scratch resource detail lost per-space observation counts");
  const auto& pinned = resources.observation(vibeqc::runtime::ExecutionResourceKind::Pinned);
  require(pinned.host_peak_bytes == 48 && pinned.host_observations == 1 &&
              pinned.device_observations == 0,
          "pinned observation lost explicit memory-space identity");
  const auto& provider =
      resources.observation(vibeqc::runtime::ExecutionResourceKind::ProviderRetained);
  require(provider.device_peak_bytes == 144 && provider.device_observations == 2 &&
              provider.host_observations == 0,
          "provider-retained observation lost high-water/count semantics");
  const auto& staging = resources.observation(vibeqc::runtime::ExecutionResourceKind::Staging);
  require(staging.host_peak_bytes == 40 && staging.host_observations == 1,
          "staging observation lost host accounting");
  const auto& capture =
      resources.observation(vibeqc::runtime::ExecutionResourceKind::CaptureRetained);
  require(capture.device_peak_bytes == 256 && capture.device_observations == 1,
          "capture-retained observation lost device accounting");

  execution.reset_resources();
  const auto reset = execution.resources();
  require(reset.host_numeric_peak_bytes == 0, "reset.host_numeric_peak_bytes == 0");
  require(reset.device_numeric_peak_bytes == 0, "reset.device_numeric_peak_bytes == 0");
  require(reset.host_workspace_peak_bytes == 0, "reset.host_workspace_peak_bytes == 0");
  require(reset.device_workspace_peak_bytes == 0, "reset.device_workspace_peak_bytes == 0");
  require(reset.numeric_observations == 0, "reset.numeric_observations == 0");
  require(reset.workspace_observations == 0, "reset.workspace_observations == 0");
  require(reset.observation(vibeqc::runtime::ExecutionResourceKind::Pinned).host_observations == 0,
          "reset retained a pinned observation");
  require(reset.observation(vibeqc::runtime::ExecutionResourceKind::ProviderRetained)
                  .device_observations == 0,
          "reset retained a provider observation");
  require(reset.observation(vibeqc::runtime::ExecutionResourceKind::Staging).host_observations == 0,
          "reset retained a staging observation");
  require(reset.observation(vibeqc::runtime::ExecutionResourceKind::CaptureRetained)
                  .device_observations == 0,
          "reset retained a capture observation");
}

// Exercise the actual prepared owners so a disconnected telemetry adapter
// cannot pass solely through the header-level tracker tests.
void verify_prepared_resources() {
  vibeqc::core::System system;
  system.atoms = {{1, {0, 0, -0.7}}, {1, {0, 0, 0.7}}};
  const std::vector<vibeqc::core::Primitive> primitives{
      {3.42525091, 0.1543289673}, {0.62391373, 0.5353281423}, {0.1688554, 0.4446345422}};
  system.shells = {{0, 0, primitives}, {1, 0, primitives}};
  std::string detail;
  require(vibeqc::molecule::validate_and_normalize(system, detail) == VIBEQC_STATUS_SUCCESS,
          "H2 setup");
  vibeqc::core::ContextState context;
  for (const auto method : {VIBEQC_METHOD_RHF, VIBEQC_METHOD_RCCSD, VIBEQC_METHOD_RCCSD_T}) {
    vibeqc_method_descriptor descriptor{};
    descriptor.struct_size = sizeof(descriptor);
    descriptor.abi_version = VIBEQC_ABI_VERSION;
    descriptor.method = method;
    descriptor.precision_mode = VIBEQC_PRECISION_FP64;
    descriptor.max_iterations = 200;
    descriptor.energy_tolerance = 1e-12;
    descriptor.density_tolerance = 1e-11;
    descriptor.ccsd_max_iterations = 150;
    descriptor.ccsd_diis_history = 6;
    auto owner = vibeqc::methods::prepare_calculation(context, system, descriptor);
    require(owner->execution_resources().numeric_observations == 0,
            "preparation fabricated a numeric observation");
    for (unsigned replay = 1; replay <= 2; ++replay) {
      const auto result = owner->execute(false);
      require(result.convergence.converged && std::isfinite(result.energy) &&
                  result.executed_backend == VIBEQC_BACKEND_CPU_REFERENCE,
              "prepared CPU endpoint failed");
      const auto snapshot = owner->execution_resources();
      require(snapshot.device_numeric_peak_bytes == 0 && snapshot.device_workspace_peak_bytes == 0,
              "CPU endpoint fabricated device observations");
      require(snapshot.observation(vibeqc::runtime::ExecutionResourceKind::Numeric)
                      .device_observations == 0 &&
                  snapshot.observation(vibeqc::runtime::ExecutionResourceKind::Scratch)
                          .device_observations == 0,
              "CPU endpoint fabricated device resource observations");
      if (method == VIBEQC_METHOD_RHF) {
        require(snapshot.numeric_observations == 0 && snapshot.host_numeric_peak_bytes == 0,
                "HF without telemetry must remain explicitly unobserved");
        continue;
      }
      const auto diagnostic = owner->correlation_diagnostic();
      require(diagnostic.has_value(), "CC endpoint omitted its independent diagnostic");
      require(snapshot.numeric_observations == replay && snapshot.host_numeric_peak_bytes > 0 &&
                  snapshot.host_numeric_peak_bytes == diagnostic->numeric_capacity_bytes,
              "CC numeric observation lost the method diagnostic or replay count");
      require(snapshot.observation(vibeqc::runtime::ExecutionResourceKind::Numeric)
                      .host_observations == replay,
              "CC numeric observation lost per-space accounting");
      if (method == VIBEQC_METHOD_RCCSD_T) {
        require(snapshot.workspace_observations == replay &&
                    snapshot.host_workspace_peak_bytes == diagnostic->ccsd_t_workspace_bytes,
                "triples workspace observation lost its source semantics");
        require(snapshot.observation(vibeqc::runtime::ExecutionResourceKind::Scratch)
                        .host_observations == replay,
                "triples scratch observation lost per-space accounting");
      } else {
        require(snapshot.workspace_observations == 0,
                "RCCSD numeric capacity was relabelled as scratch workspace");
      }
    }
    if (method == VIBEQC_METHOD_RCCSD_T) {
      const auto force = owner->execute(true);
      require(force.convergence.converged && force.forces.size() == 3 * system.atoms.size(),
              "prepared RCCSD(T) force endpoint failed");
      const auto diagnostic = owner->correlation_diagnostic();
      const auto snapshot = owner->execution_resources();
      require(diagnostic && snapshot.numeric_observations == 4 &&
                  snapshot.host_numeric_peak_bytes == diagnostic->numeric_capacity_bytes &&
                  snapshot.host_numeric_peak_bytes >= diagnostic->planned_endpoint_peak_bytes,
              "prepared force capacity was omitted from execution observations");
      require(snapshot.workspace_observations == 4 &&
                  snapshot.host_workspace_peak_bytes >= diagnostic->response_workspace_bytes,
              "prepared orbital response workspace was omitted from execution observations");
    }
  }
}
}  // namespace

int main() {
  try {
    verify_tracker();
    verify_prepared_resources();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
