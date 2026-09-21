#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace vibeqc::runtime {

/** Bounded by-value packet for homogeneous task-range descriptors.
 *
 * Slice owns the scientific/runtime descriptor. This common layer owns only
 * finite packet capacity and host scheduling metadata; it never materializes
 * individual logical tasks.
 */
template <class Slice, unsigned Capacity>
struct HomogeneousTaskPacket {
  static_assert(Capacity > 0);
  static_assert(std::is_trivially_copyable_v<Slice>);
  static constexpr unsigned capacity = Capacity;
  Slice slices[capacity]{};
  unsigned count{}, blocks{};
};

/** Stable heavy-first ordering shared by runtime consumers.
 *
 * Cost must describe work within one already-legal homogeneous range. It is a
 * profitability/order hint only and cannot make an illegal range executable.
 */
template <class Packet, class Cost>
void order_homogeneous_task_packet(Packet& packet, Cost cost) {
  std::sort(packet.slices, packet.slices + packet.count, [&](const auto& left, const auto& right) {
    const auto left_cost = cost(left), right_cost = cost(right);
    return left_cost != right_cost ? left_cost > right_cost : left.first_block < right.first_block;
  });
}

/** Assign block prefixes without exceeding CUDA's unsigned grid dimension.
 *
 * Slice must expose tasks and first_block. Returning false leaves the packet
 * unsuitable for launch; callers retain their existing fallback/error.
 */
template <class Packet>
bool finalize_homogeneous_task_packet(
    Packet& packet, std::size_t tasks_per_block,
    std::size_t maximum_blocks = std::numeric_limits<unsigned>::max()) {
  if (!tasks_per_block || maximum_blocks > std::numeric_limits<unsigned>::max()) return false;
  packet.blocks = 0;
  for (unsigned index = 0; index < packet.count; ++index) {
    auto& slice = packet.slices[index];
    slice.first_block = packet.blocks;
    const auto blocks = slice.tasks / tasks_per_block + (slice.tasks % tasks_per_block != 0);
    if (blocks > maximum_blocks - packet.blocks) return false;
    packet.blocks += static_cast<unsigned>(blocks);
  }
  return true;
}

}  // namespace vibeqc::runtime
