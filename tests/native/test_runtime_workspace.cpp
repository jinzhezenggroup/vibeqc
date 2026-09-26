#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "runtime/bounded_workspace.hpp"

int main() {
  using vibeqc::runtime::AsyncGeneration;
  using vibeqc::runtime::BorrowedWorkspace;
  using vibeqc::runtime::checked_add;
  using vibeqc::runtime::checked_multiply;
  using vibeqc::runtime::ranges_overlap;
  using vibeqc::runtime::WorkspaceLayout;

  std::size_t value = 0;
  assert(checked_add(7, 9, value) && value == 16);
  assert(!checked_add(std::numeric_limits<std::size_t>::max(), 1, value));
  assert(checked_multiply(7, 9, value) && value == 63);
  assert(!checked_multiply(std::numeric_limits<std::size_t>::max(), 2, value));

  WorkspaceLayout layout;
  std::size_t byte_offset = 0;
  std::size_t double_offset = 0;
  assert(layout.append<std::uint8_t>(3, byte_offset));
  assert(layout.append<double>(2, double_offset));
  assert(byte_offset == 0);
  assert(double_offset % alignof(double) == 0);
  assert(layout.bytes() == double_offset + 2 * sizeof(double));

  alignas(double) std::array<std::byte, 64> storage{};
  BorrowedWorkspace workspace(storage.data(), storage.size());
  const auto doubles = workspace.view<double>(0, 2);
  assert(doubles.data == reinterpret_cast<double*>(storage.data()));
  assert(doubles.elements == 2);
  BorrowedWorkspace empty(nullptr, 0);
  const auto empty_view = empty.view<std::byte>(0, 0);
  assert(empty_view.data == nullptr && empty_view.elements == 0);

  vibeqc::runtime::ResourcePlan plan{64, 32, 16, alignof(double)};
  assert(plan.valid());
  plan.workspace_bytes = 65;
  assert(!plan.valid());

  bool bounds_failed = false;
  try {
    (void)workspace.view<double>(60, 1);
  } catch (const std::out_of_range&) {
    bounds_failed = true;
  }
  assert(bounds_failed);

  assert(ranges_overlap(storage.data(), 16, storage.data() + 8, 16));
  assert(!ranges_overlap(storage.data(), 8, storage.data() + 8, 8));

  AsyncGeneration generation;
  generation.begin(1);
  assert(generation.published() == 0);
  generation.commit(1);
  generation.require(1);

  generation.begin(2);
  assert(generation.published() == 0);
  bool stale_failed = false;
  try {
    generation.require(1);
  } catch (const std::invalid_argument&) {
    stale_failed = true;
  }
  assert(stale_failed);

  // Simulate an enqueue failure: without commit(), no result is publishable.
  bool failed_result_hidden = false;
  try {
    generation.require(2);
  } catch (const std::invalid_argument&) {
    failed_result_hidden = true;
  }
  assert(failed_result_hidden);
  generation.begin(3);
  generation.commit(3);
  generation.require(3);

  return 0;
}
