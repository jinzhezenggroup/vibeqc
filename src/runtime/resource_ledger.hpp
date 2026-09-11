#ifndef VIBEQC_RUNTIME_RESOURCE_LEDGER_HPP
#define VIBEQC_RUNTIME_RESOURCE_LEDGER_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace vibeqc::runtime {

/** Numeric device allocations owned by one prepared resource request.
 *
 * The shared owner outlives a Python observation scope: cached native buffers
 * remain charged between calls, and can be freed on a different host thread.
 * CUDA context, graphs, pools and library-internal allocations are explicitly
 * outside this ledger. Their allowances are withheld by the global planner.
 */
struct DeviceResourceLedger {
  std::size_t limit{};
  int device{};
  std::size_t live{};
  std::size_t peak{};
  std::size_t allocations{};
  std::size_t rejected{};
  bool active{};
};

struct DeviceAllocationOwner {
  std::shared_ptr<DeviceResourceLedger> ledger;
  std::size_t bytes{};
  std::uint64_t generation{};
};

inline std::mutex device_resource_mutex;
inline std::unordered_map<void*, DeviceAllocationOwner> device_allocation_owners;
inline std::uint64_t device_allocation_generation{};
inline thread_local std::shared_ptr<DeviceResourceLedger> active_device_resource_ledger;

}  // namespace vibeqc::runtime

#endif
