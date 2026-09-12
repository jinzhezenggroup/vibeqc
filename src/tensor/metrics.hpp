#pragma once
#include <cstdint>
namespace vibeqc_tensor {
/** Shared plain-data diagnostics used by the native and generated tensor ABIs. */
struct Metrics {
  uint64_t owned_device_bytes = 0;
  uint64_t provider_retained_bytes = 0;
  uint64_t prepare_device_delta = 0;
  uint64_t observed_device_delta = 0;
  double device_ms = 0;
  double input_ms = 0;
  double output_ms = 0;
  double packing_ms = 0;
  double library_ms = 0;
  double kernel_ms = 0;
};
}  // namespace vibeqc_tensor
