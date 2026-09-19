#include <array>
#include <cstddef>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>

#include "runtime/tracked_allocator.hpp"

namespace {
using vibeqc::runtime::AllocationCounter;
using vibeqc::runtime::TrackedAllocator;
using vibeqc::runtime::TrackedVector;

void require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

void actual_overlap_and_reallocation() {
  const auto counter = std::make_shared<AllocationCounter>();
  const TrackedAllocator<double> allocator(counter);
  {
    TrackedVector<double> retained(8, 1.0, allocator);
    const auto retained_bytes = retained.capacity() * sizeof(double);
    {
      TrackedVector<double> first(13, 2.0, allocator);
      require(counter->snapshot().live_bytes == retained_bytes + first.capacity() * sizeof(double),
              "simultaneous storage is not counted");
    }
    {
      TrackedVector<double> second(11, 3.0, allocator);
      require(counter->snapshot().peak_bytes == retained_bytes + 13 * sizeof(double),
              "disjoint phase maxima were added instead of taking a live high-water mark");
    }
    retained.reserve(32);
    require(counter->snapshot().peak_bytes == retained_bytes + retained.capacity() * sizeof(double),
            "vector growth missed old/new overlapping allocations");
    require(counter->snapshot().allocation_count == 4, "unexpected allocation event count");
  }
  require(counter->snapshot().live_bytes == 0, "destruction leaked accounting");
  require(counter->snapshot().peak_bytes > 0, "destruction erased the historical peak");
}

void copy_move_swap_and_surviving_result() {
  const auto first = std::make_shared<AllocationCounter>();
  const auto second = std::make_shared<AllocationCounter>();
  {
    TrackedVector<int> a(7, 1, TrackedAllocator<int>(first));
    auto copy = a;
    require(first->snapshot().live_bytes == 14 * sizeof(int), "copy not in source domain");
    TrackedVector<int> b(3, 2, TrackedAllocator<int>(second));
    a.swap(b);
    require(first->snapshot().live_bytes == 14 * sizeof(int) &&
                second->snapshot().live_bytes == 3 * sizeof(int),
            "swap changed ownership counts");
    b = std::move(a);
    require(first->snapshot().live_bytes == 7 * sizeof(int) &&
                second->snapshot().live_bytes == 3 * sizeof(int),
            "move assignment lost the allocation domain");
    copy = b;
    require(first->snapshot().live_bytes == 0 && second->snapshot().live_bytes == 6 * sizeof(int),
            "copy assignment did not propagate the domain");
  }
  require(first->snapshot().live_bytes == 0 && second->snapshot().live_bytes == 0,
          "copy/move/swap left live buffers");
  std::weak_ptr<AllocationCounter> lifetime;
  {
    auto result = [&] {
      const auto owner = std::make_shared<AllocationCounter>();
      lifetime = owner;
      return TrackedVector<double>(9, 1.0, TrackedAllocator<double>(owner));
    }();
    require(!lifetime.expired(), "returned buffer outlived its counter");
    require(lifetime.lock()->snapshot().live_bytes == result.capacity() * sizeof(double),
            "returned buffer is not retained");
  }
  require(lifetime.expired(), "counter ownership leaked");
}

void exceptions_and_refusals() {
  const auto counter = std::make_shared<AllocationCounter>();
  try {
    TrackedVector<double> values(17, 0.0, TrackedAllocator<double>(counter));
    throw std::runtime_error("injected owner failure");
  } catch (const std::runtime_error&) {
  }
  require(counter->snapshot().live_bytes == 0, "exception unwinding leaked live bytes");
  const auto before = counter->snapshot();
  bool failed = false;
  try {
    auto allocator = TrackedAllocator<double>(counter);
    const auto size = std::numeric_limits<std::size_t>::max() / sizeof(double) + 1;
    auto* pointer = allocator.allocate(size);
    allocator.deallocate(pointer, size);
  } catch (const std::bad_array_new_length&) {
    failed = true;
  }
  const auto after = counter->snapshot();
  require(failed && before.peak_bytes == after.peak_bytes &&
              before.allocation_count == after.allocation_count && after.live_bytes == 0,
          "failed allocation mutated the ledger");
  TrackedVector<double> empty;
  require(!empty.get_allocator().counter(), "default empty buffer allocated an observer");
}

void concurrent_live_storage_and_foreign_destruction() {
  const auto counter = std::make_shared<AllocationCounter>();
  std::array<std::promise<void>, 4> ready;
  std::promise<void> release;
  const auto go = release.get_future().share();
  std::array<std::thread, 4> workers;
  for (std::size_t i = 0; i < workers.size(); ++i) {
    workers[i] = std::thread([&, i] {
      TrackedVector<double> values(i + 1, 0.0, TrackedAllocator<double>(counter));
      ready[i].set_value();
      go.wait();
    });
  }
  for (auto& signal : ready) signal.get_future().wait();
  const auto overlap = counter->snapshot();
  release.set_value();
  for (auto& worker : workers) worker.join();
  require(overlap.live_bytes == 10 * sizeof(double) && overlap.peak_bytes == overlap.live_bytes,
          "concurrent allocations did not share a simultaneous-live counter");
  require(counter->snapshot().live_bytes == 0, "concurrent releases leaked accounting");
  TrackedVector<double> payload(6, 0.0, TrackedAllocator<double>(counter));
  std::thread destroy([owned = std::move(payload)]() mutable { owned.clear(); });
  destroy.join();
  require(counter->snapshot().live_bytes == 0, "cross-thread destruction lost ownership");
}
}  // namespace

int main() {
  try {
    actual_overlap_and_reallocation();
    copy_move_swap_and_surviving_result();
    exceptions_and_refusals();
    concurrent_live_storage_and_foreign_destruction();
    std::cout << "Tracked allocator contracts passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
