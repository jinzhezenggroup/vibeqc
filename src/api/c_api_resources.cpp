#include <algorithm>
#include <cstdio>
#include <exception>
#include <iterator>
#include <limits>
#include <stdexcept>

#include "dft/scf_diagnostic.hpp"
#include "runtime/resource_ledger.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/cuda_batch.hpp"
#include "scf/density_fitting.hpp"

#if VIBEQC_HAS_CUDA
#include "dft/cuda_ks.hpp"
#include "dft/cuda_xc.hpp"
#include "scf/cuda_direct_jk.hpp"
#endif

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

/** KS v1 host bounds use LP64 metadata and a 128-byte iteration-record
 * allowance. Keep this separate from HF's derivative-inclusive inventory. */
int vibeqc_ks_resource_inventory_version_v1() {
  return sizeof(void*) == 8 && sizeof(vibeqc::dft::ScfIteration) <= 128 ? 1 : 0;
}

/** Pure shape bridge to allocator-owned KS/XC and common #202 direct-J
 * layouts. It never constructs a grid, density, integral or CUDA context. */
int vibeqc_resource_ks_cuda_v1(std::size_t nao, std::size_t atoms, std::size_t shells,
                               std::size_t primitives, std::size_t points, std::size_t diis_history,
                               std::size_t spins, std::size_t pbe, std::size_t tile_points,
                               std::uint64_t* output, std::size_t count) {
  if (!output || count != 3 || diis_history > 64 || (spins != 1 && spins != 2) || pbe > 1) return 1;
#if VIBEQC_HAS_CUDA
  try {
    const auto state = vibeqc::dft::cuda_ks_state_bytes(nao, spins, diis_history);
    const auto xc = vibeqc::dft::cuda_xc_layout_shape(atoms, primitives, nao, points, pbe != 0,
                                                      spins == 2, tile_points);
    const auto direct =
        vibeqc::scf::cuda_direct_coulomb_device_bytes(1, nao, atoms, shells, primitives);
    output[0] = state;
    output[1] = xc.device_bytes;
    output[2] = direct;
    return 0;
  } catch (...) {
    return 1;
  }
#else
  return 2;
#endif
}

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
int vibeqc_resource_df_tiles_v3(std::size_t batch, std::size_t nbf, std::size_t naux,
                                std::size_t occupied, std::size_t budget,
                                std::size_t fixed_device_bytes, unsigned generated_source,
                                std::size_t automatic_rhf_rank, std::uint64_t* output,
                                std::size_t count, char* error, std::size_t error_size) {
  if (output == nullptr || (count != 6 && count != 7) || error == nullptr || error_size == 0 ||
      generated_source > 1)
    return 1;
  try {
    const auto plan = vibeqc::scf::plan_density_fitting_tiles(
        batch, nbf, naux, occupied, budget, fixed_device_bytes, generated_source != 0,
        automatic_rhf_rank);
    output[0] = plan.batch_tile;
    output[1] = plan.ao_pair_tile;
    output[2] = plan.auxiliary_tile;
    output[3] = plan.occupied_tile;
    output[4] = plan.peak_workspace_bytes;
    output[5] = plan.stores_full_three_center ? 1 : 0;
    if (count == 7) output[6] = plan.automatic_rhf_rank;
    error[0] = '\0';
    return 0;
  } catch (const std::exception& exception) {
    std::snprintf(error, error_size, "%s", exception.what());
    return 1;
  }
}

/** Legacy shape queries have no reference/occupation authorization. */
int vibeqc_resource_df_tiles_v2(std::size_t batch, std::size_t nbf, std::size_t naux,
                                std::size_t occupied, std::size_t budget,
                                std::size_t fixed_device_bytes, unsigned generated_source,
                                std::uint64_t* output, std::size_t count, char* error,
                                std::size_t error_size) {
  return vibeqc_resource_df_tiles_v3(batch, nbf, naux, occupied, budget, fixed_device_bytes,
                                     generated_source, 0, output, count, error, error_size);
}

/** Preserve the original host-tensor capacity query ABI. */
int vibeqc_resource_df_tiles_v1(std::size_t batch, std::size_t nbf, std::size_t naux,
                                std::size_t occupied, std::size_t budget,
                                std::size_t fixed_device_bytes, std::uint64_t* output,
                                std::size_t count, char* error, std::size_t error_size) {
  return vibeqc_resource_df_tiles_v2(batch, nbf, naux, occupied, budget, fixed_device_bytes, 0,
                                     output, count, error, error_size);
}

/** Explicit physical-source packed representation, without changing the dense
 * query ABI or inferring storage from environment variables. The final four
 * fields report one immutable factor's bytes, total mutable scratch bytes,
 * complete-projection capacity and each bounded panel's capacity in doubles.
 * Raw A and transformed B are distinct allocations of the same factor size.
 * rank_capacity=0 reserves no complete U, while exact bounded K still exists.
 */
int vibeqc_resource_df_packed_tiles_v2(std::size_t batch, std::size_t nbf, std::size_t naux,
                                       std::size_t rank_capacity, std::size_t budget,
                                       std::size_t fixed_device_bytes,
                                       std::size_t automatic_rhf_rank, std::uint64_t* output,
                                       std::size_t count, char* error, std::size_t error_size) {
  if (!output || (count != 10 && count != 11) || !error || !error_size) return 1;
  try {
    const auto plan = vibeqc::scf::plan_packed_density_fitting_tiles(
        batch, nbf, naux, rank_capacity, budget, fixed_device_bytes, automatic_rhf_rank);
    const auto capacity =
        vibeqc::scf::df_packed_value_capacity(batch, nbf, naux, rank_capacity, plan.auxiliary_tile);
    const std::uint64_t values[]{
        plan.batch_tile,         plan.ao_pair_tile,         plan.auxiliary_tile,
        plan.occupied_tile,      plan.peak_workspace_bytes, 1,
        capacity.factor_bytes,   capacity.scratch_bytes,    capacity.projection_elements,
        capacity.panel_elements, plan.automatic_rhf_rank};
    std::copy_n(values, count, output);
    error[0] = '\0';
    return 0;
  } catch (const std::exception& exception) {
    std::snprintf(error, error_size, "%s", exception.what());
    return 1;
  }
}

/** Preserve the packed shape ABI without guessing the method from rank capacity. */
int vibeqc_resource_df_packed_tiles_v1(std::size_t batch, std::size_t nbf, std::size_t naux,
                                       std::size_t rank_capacity, std::size_t budget,
                                       std::size_t fixed_device_bytes, std::uint64_t* output,
                                       std::size_t count, char* error, std::size_t error_size) {
  return vibeqc_resource_df_packed_tiles_v2(batch, nbf, naux, rank_capacity, budget,
                                            fixed_device_bytes, 0, output, count, error,
                                            error_size);
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

/** Shape-only capacity of the shared device DIIS owner; never probes CUDA. */
int vibeqc_resource_df_diis_bytes_v1(std::size_t batch, std::size_t nbf, unsigned history,
                                     std::uint64_t* output) {
  if (!output || !batch || !nbf) return 1;
  const auto bytes = vibeqc::scf::density_fitting_scf_diis_device_bytes(batch, nbf, history);
  if (bytes == std::numeric_limits<std::size_t>::max()) return 1;
  *output = bytes;
  return 0;
}
}
