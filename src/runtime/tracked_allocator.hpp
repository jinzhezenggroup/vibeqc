#pragma once

#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace vibeqc::runtime {

struct AllocationSnapshot {
  std::size_t live_bytes{};
  std::size_t peak_bytes{};
  std::size_t allocation_count{};
};

template <class T>
class TrackedAllocator;

/** Actual simultaneously live payload allocated by one explicit ownership domain.
 * Not RSS, a capacity plan, or allocator/driver overhead. The counter has no
 * global/thread-local state; unrelated calculations cannot contaminate it.
 */
class AllocationCounter {
 public:
  AllocationSnapshot snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
  }

 private:
  template <class T>
  friend class TrackedAllocator;

  mutable std::mutex mutex_;
  AllocationSnapshot state_;
};

/** Standard allocator with shared accounting lifetime, including moved results.
 * All vector propagation operations carry the domain with the storage. An
 * empty/default allocator is unobserved and allocates no counter of its own.
 */
template <class T>
class TrackedAllocator {
 public:
  using value_type = T;
  using propagate_on_container_copy_assignment = std::true_type;
  using propagate_on_container_move_assignment = std::true_type;
  using propagate_on_container_swap = std::true_type;
  using is_always_equal = std::false_type;

  TrackedAllocator() noexcept = default;
  explicit TrackedAllocator(std::shared_ptr<AllocationCounter> counter) noexcept
      : counter_(std::move(counter)) {}
  template <class U>
  TrackedAllocator(const TrackedAllocator<U>& other) noexcept : counter_(other.counter()) {}

  [[nodiscard]] T* allocate(std::size_t count) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
      throw std::bad_array_new_length();
    if (!counter_) return std::allocator<T>{}.allocate(count);
    // Both allocator events and their bookkeeping are serialized within this
    // domain, so snapshots cannot miss concurrently acquired/released storage.
    std::lock_guard<std::mutex> lock(counter_->mutex_);
    auto& state = counter_->state_;
    const auto bytes = count * sizeof(T);
    const auto maximum = std::numeric_limits<std::size_t>::max();
    if (bytes > maximum - state.live_bytes || state.allocation_count == maximum)
      throw std::overflow_error("owned allocation accounting overflow");
    T* pointer = std::allocator<T>{}.allocate(count);
    state.live_bytes += bytes;
    if (state.live_bytes > state.peak_bytes) state.peak_bytes = state.live_bytes;
    ++state.allocation_count;
    return pointer;
  }

  void deallocate(T* pointer, std::size_t count) noexcept {
    // Serialize release with acquisition. In concurrent use recording a release
    // before free would miss overlapping lifetimes; recording after free without
    // the lock could overcount a subsequent allocation in another thread.
    if (counter_) {
      std::lock_guard<std::mutex> lock(counter_->mutex_);
      std::allocator<T>{}.deallocate(pointer, count);
      const auto bytes = count * sizeof(T);
      if (bytes > counter_->state_.live_bytes) std::terminate();
      counter_->state_.live_bytes -= bytes;
    } else {
      std::allocator<T>{}.deallocate(pointer, count);
    }
  }

  std::shared_ptr<AllocationCounter> counter() const noexcept { return counter_; }

  template <class U>
  bool operator==(const TrackedAllocator<U>& other) const noexcept {
    return counter_ == other.counter();
  }

 private:
  std::shared_ptr<AllocationCounter> counter_;
};

template <class T>
using TrackedVector = std::vector<T, TrackedAllocator<T>>;

}  // namespace vibeqc::runtime
