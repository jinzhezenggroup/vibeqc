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

  const auto resources = execution.resources();
  require(resources.host_numeric_peak_bytes == 64, "resources.host_numeric_peak_bytes == 64");
  require(resources.device_numeric_peak_bytes == 96, "resources.device_numeric_peak_bytes == 96");
  require(resources.host_workspace_peak_bytes == 24, "resources.host_workspace_peak_bytes == 24");
  require(resources.device_workspace_peak_bytes == 128,
          "resources.device_workspace_peak_bytes == 128");
  require(resources.numeric_observations == 3, "resources.numeric_observations == 3");
  require(resources.workspace_observations == 3, "resources.workspace_observations == 3");

  execution.reset_resources();
  const auto reset = execution.resources();
  require(reset.host_numeric_peak_bytes == 0, "reset.host_numeric_peak_bytes == 0");
  require(reset.device_numeric_peak_bytes == 0, "reset.device_numeric_peak_bytes == 0");
  require(reset.host_workspace_peak_bytes == 0, "reset.host_workspace_peak_bytes == 0");
  require(reset.device_workspace_peak_bytes == 0, "reset.device_workspace_peak_bytes == 0");
  require(reset.numeric_observations == 0, "reset.numeric_observations == 0");
  require(reset.workspace_observations == 0, "reset.workspace_observations == 0");
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
      if (method == VIBEQC_METHOD_RCCSD_T) {
        require(snapshot.workspace_observations == replay &&
                    snapshot.host_workspace_peak_bytes == diagnostic->ccsd_t_workspace_bytes,
                "triples workspace observation lost its source semantics");
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
