#include <cassert>

#include "runtime/execution_context.hpp"

int main() {
  vibeqc::core::ContextState state;
  state.requested_backend = VIBEQC_BACKEND_CUDA;
  state.device_id = 7;

  vibeqc::runtime::ExecutionContext execution(state);
  assert(execution.backend() == VIBEQC_BACKEND_CUDA);
  assert(execution.cuda_requested());
  assert(execution.device_id() == 7);

  execution.observe_numeric_peak(vibeqc::runtime::ExecutionMemorySpace::Host, 64);
  execution.observe_numeric_peak(vibeqc::runtime::ExecutionMemorySpace::Host, 32);
  execution.observe_numeric_peak(vibeqc::runtime::ExecutionMemorySpace::Device, 96);
  execution.observe_workspace_peak(vibeqc::runtime::ExecutionMemorySpace::Host, 24);
  execution.observe_workspace_peak(vibeqc::runtime::ExecutionMemorySpace::Device, 128);
  execution.observe_workspace_peak(vibeqc::runtime::ExecutionMemorySpace::Device, 80);

  const auto resources = execution.resources();
  assert(resources.host_numeric_peak_bytes == 64);
  assert(resources.device_numeric_peak_bytes == 96);
  assert(resources.host_workspace_peak_bytes == 24);
  assert(resources.device_workspace_peak_bytes == 128);
  assert(resources.numeric_observations == 3);
  assert(resources.workspace_observations == 3);

  execution.reset_resources();
  const auto reset = execution.resources();
  assert(reset.host_numeric_peak_bytes == 0);
  assert(reset.device_numeric_peak_bytes == 0);
  assert(reset.host_workspace_peak_bytes == 0);
  assert(reset.device_workspace_peak_bytes == 0);
  assert(reset.numeric_observations == 0);
  assert(reset.workspace_observations == 0);
  return 0;
}
