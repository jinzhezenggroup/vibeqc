#include <cstdio>
#include <exception>
#include <stdexcept>

#include "runtime/resource_ledger.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/cuda_batch.hpp"
#include "scf/density_fitting.hpp"

extern "C" {

/** Private prepared-request ledger. Creating it performs no CUDA operation. */
void* vibeqc_resource_ledger_create_v1(std::size_t bytes, int device) {
  if (device < 0) return nullptr;
  try {
    return new std::shared_ptr<vibeqc::runtime::DeviceResourceLedger>(
        std::make_shared<vibeqc::runtime::DeviceResourceLedger>(
            vibeqc::runtime::DeviceResourceLedger{bytes, device}));
  } catch (const std::bad_alloc&) {
    return nullptr;
  }
}

void vibeqc_resource_ledger_destroy_v1(void* handle) {
  delete static_cast<std::shared_ptr<vibeqc::runtime::DeviceResourceLedger>*>(handle);
}

/** Bind only within an active, synchronous observation. Reset the peak to
 * retained live state so warm replay measurements include pre-existing caches.
 */
int vibeqc_resource_ledger_bind_v1(void* handle) {
  using namespace vibeqc::runtime;
  if (!cpu_resource_observation.active || active_device_resource_ledger || handle == nullptr)
    return 1;
  auto ledger = *static_cast<std::shared_ptr<DeviceResourceLedger>*>(handle);
  std::lock_guard<std::mutex> lock(device_resource_mutex);
  if (ledger->active) return 1;
  ledger->active = true;
  ledger->peak = ledger->live;
  ledger->allocations = 0;
  ledger->rejected = 0;
  active_device_resource_ledger = std::move(ledger);
  return 0;
}

int vibeqc_resource_ledger_read_v1(void* handle, std::uint64_t* values) {
  if (handle == nullptr || values == nullptr) return 1;
  const auto& ledger =
      *static_cast<std::shared_ptr<vibeqc::runtime::DeviceResourceLedger>*>(handle);
  std::lock_guard<std::mutex> lock(vibeqc::runtime::device_resource_mutex);
  values[0] = ledger->live;
  values[1] = ledger->peak;
  values[2] = ledger->allocations;
  values[3] = ledger->rejected;
  return 0;
}

/** Allocation-inventory contract implemented by resources_hf.py. Increment
 * when changing CPU allocation shapes/lifetimes beyond those documented
 * bounds. The current object-capacity allowance is validated on LP64 hosts.
 */
int vibeqc_cpu_resource_inventory_version_v1() { return sizeof(void*) == 8 ? 1 : 0; }

/** Private execution-scope bridge. The selected CPU plan can explicitly cap
 * fleet concurrency as well as collect samples. No device query/allocation.
 */
int vibeqc_resource_tracking_begin_v1(unsigned cpu_worker_limit) {
  auto& observation = vibeqc::runtime::cpu_resource_observation;
  if (observation.active || cpu_worker_limit > 1) return 1;
  observation = {true, 0, 0, cpu_worker_limit};
  return 0;
}

/** Return the observed scope even after failed solves; zero samples means
 * unavailable data, never evidence of a zero-byte scientific execution.
 */
int vibeqc_resource_tracking_end_v1(std::uint64_t* peak, std::uint64_t* samples) {
  auto& observation = vibeqc::runtime::cpu_resource_observation;
  if (peak == nullptr || samples == nullptr || !observation.active) return 1;
  *peak = observation.peak_bytes;
  *samples = observation.samples;
  observation.active = false;
  observation.cpu_worker_limit = 0;
  if (vibeqc::runtime::active_device_resource_ledger) {
    std::lock_guard<std::mutex> lock(vibeqc::runtime::device_resource_mutex);
    vibeqc::runtime::active_device_resource_ledger->active = false;
  }
  vibeqc::runtime::active_device_resource_ledger.reset();
  return 0;
}

/** Read the optional CUDA arena samples before ending the synchronous scope. */
int vibeqc_resource_tracking_cuda_v1(std::uint64_t* peak, std::uint64_t* samples) {
  const auto& observation = vibeqc::runtime::cpu_resource_observation;
  if (peak == nullptr || samples == nullptr || !observation.active) return 1;
  *peak = observation.cuda_arena_peak_bytes;
  *samples = observation.cuda_arena_samples;
  return 0;
}

/** No runtime initialization, even in a CUDA-linked library. Status 2 means
 * the requested shape/provider is outside this allocation-inventory contract.
 */
int vibeqc_resource_small_hf_cuda_v1(std::size_t nbf, std::size_t direct_nbf, std::size_t atoms,
                                     std::size_t shells, std::size_t primitives,
                                     std::size_t diis_history, std::size_t spins,
                                     std::uint64_t* output) {
  if (output == nullptr) return 1;
#if VIBEQC_HAS_CUDA
  std::size_t arena_bytes = 0;
  std::size_t plan_bytes = 0;
  if (!vibeqc::scf::small_hf_cuda_resource_layout(nbf, direct_nbf, atoms, shells, primitives,
                                                  diis_history, spins, arena_bytes, plan_bytes))
    return 2;
  output[0] = arena_bytes;
  output[1] = plan_bytes;
  return 0;
#else
  return 2;
#endif
}

/** Shape-only adapter to the existing DF planner. No integrals, context or
 * workspace is allocated. Fixed source/other-provider bytes are supplied by
 * the composing planner, not independently spent again inside this budget.
 */
int vibeqc_resource_df_tiles_v1(std::size_t batch, std::size_t nbf, std::size_t naux,
                                std::size_t occupied, std::size_t budget,
                                std::size_t fixed_device_bytes, std::uint64_t* output,
                                std::size_t count, char* error, std::size_t error_size) {
  if (output == nullptr || count != 6 || error == nullptr || error_size == 0) return 1;
  try {
    const auto plan = vibeqc::scf::plan_density_fitting_tiles(batch, nbf, naux, occupied, budget,
                                                              fixed_device_bytes);
    output[0] = plan.batch_tile;
    output[1] = plan.ao_pair_tile;
    output[2] = plan.auxiliary_tile;
    output[3] = plan.occupied_tile;
    output[4] = plan.peak_workspace_bytes;
    output[5] = plan.stores_full_three_center ? 1 : 0;
    error[0] = '\0';
    return 0;
  } catch (const std::exception& exception) {
    std::snprintf(error, error_size, "%s", exception.what());
    return 1;
  }
}

int vibeqc_resource_df_source_bytes_v1(std::size_t batch, std::size_t atoms, std::size_t shells,
                                       std::size_t cartesian_aos, std::size_t primitives,
                                       std::size_t transforms, std::uint64_t* output) {
  if (output == nullptr) return 1;
  try {
    *output = vibeqc::scf::density_fitting_source_metadata_bytes(
        batch, atoms, shells, cartesian_aos, primitives, transforms);
    return 0;
  } catch (const std::overflow_error&) {
    return 1;
  }
}
}
