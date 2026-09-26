#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace vibeqc::runtime {

/** Overflow-checked arithmetic shared by runtime/resource planners.
 * Failure never publishes a partial result.
 */
inline bool checked_add(std::size_t first, std::size_t second, std::size_t& result) noexcept {
  if (second > std::numeric_limits<std::size_t>::max() - first) return false;
  result = first + second;
  return true;
}

inline bool checked_multiply(std::size_t first, std::size_t second, std::size_t& result) noexcept {
  if (first != 0 && second > std::numeric_limits<std::size_t>::max() / first) return false;
  result = first * second;
  return true;
}

inline bool checked_align_up(std::size_t value, std::size_t alignment,
                             std::size_t& result) noexcept {
  if (alignment == 0) return false;
  const auto remainder = value % alignment;
  if (remainder == 0) {
    result = value;
    return true;
  }
  return checked_add(value, alignment - remainder, result);
}

inline std::size_t size_add(std::size_t first, std::size_t second,
                            const char* message = "size overflow") {
  std::size_t result = 0;
  if (!checked_add(first, second, result)) throw std::overflow_error(message);
  return result;
}

inline std::size_t size_mul(std::size_t first, std::size_t second,
                            const char* message = "size overflow") {
  std::size_t result = 0;
  if (!checked_multiply(first, second, result)) throw std::overflow_error(message);
  return result;
}

/** Transactional layout cursor for one bounded arena/workspace. */
class WorkspaceLayout {
 public:
  template <class T>
  bool append(std::size_t count, std::size_t& offset) noexcept {
    static_assert(!std::is_void_v<T>);
    std::size_t bytes = 0;
    return checked_multiply(count, sizeof(T), bytes) && append_bytes(bytes, alignof(T), offset);
  }

  bool append_bytes(std::size_t bytes, std::size_t alignment, std::size_t& offset) noexcept {
    std::size_t aligned = 0;
    std::size_t next = 0;
    if (!checked_align_up(bytes_, alignment, aligned) || !checked_add(aligned, bytes, next))
      return false;
    offset = aligned;
    bytes_ = next;
    return true;
  }

  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

 private:
  std::size_t bytes_{};
};

template <class T>
struct TensorView {
  T* data{};
  std::size_t elements{};

  [[nodiscard]] explicit operator bool() const noexcept { return data != nullptr || elements == 0; }
  [[nodiscard]] std::size_t size_bytes() const {
    return size_mul(elements, sizeof(T), "tensor view size overflow");
  }
};

/** Non-owning view over storage whose owner must outlive every returned view. */
class BorrowedWorkspace {
 public:
  BorrowedWorkspace() = default;
  BorrowedWorkspace(void* data, std::size_t bytes)
      : data_(static_cast<std::byte*>(data)), bytes_(bytes) {
    if (!data_ && bytes_) throw std::invalid_argument("null bounded workspace");
  }

  template <class T>
  [[nodiscard]] TensorView<T> view(std::size_t offset, std::size_t count) const {
    const auto bytes = size_mul(count, sizeof(T), "workspace view size overflow");
    const auto end = size_add(offset, bytes, "workspace view offset overflow");
    if (end > bytes_) throw std::out_of_range("workspace view exceeds prepared bounds");
    auto* pointer = offset == 0 ? data_ : data_ + offset;
    if (reinterpret_cast<std::uintptr_t>(pointer) % alignof(T) != 0)
      throw std::invalid_argument("workspace view is misaligned");
    return {reinterpret_cast<T*>(pointer), count};
  }

  [[nodiscard]] void* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

 private:
  std::byte* data_{};
  std::size_t bytes_{};
};

inline bool ranges_overlap(const void* first, std::size_t first_bytes, const void* second,
                           std::size_t second_bytes) {
  if (!first_bytes || !second_bytes) return false;
  if (!first || !second) throw std::invalid_argument("null non-empty buffer range");
  const auto a = reinterpret_cast<std::uintptr_t>(first);
  const auto b = reinterpret_cast<std::uintptr_t>(second);
  std::size_t a_end = 0;
  std::size_t b_end = 0;
  if (!checked_add(a, first_bytes, a_end) || !checked_add(b, second_bytes, b_end))
    throw std::overflow_error("buffer range overflow");
  return a < b_end && b < a_end;
}

/** Generation gate for asynchronous borrowed results.
 * begin() invalidates the previous published view before work is enqueued.
 * commit() publishes only a successfully submitted generation.  If an
 * operation fails between them, no stale result remains observable.
 */
class AsyncGeneration {
 public:
  void begin(std::uint64_t generation) {
    if (!generation || generation <= submitted_)
      throw std::invalid_argument("asynchronous generation is stale");
    published_ = 0;
    submitted_ = generation;
  }

  void commit(std::uint64_t generation) {
    if (!generation || generation != submitted_)
      throw std::invalid_argument("asynchronous generation was not submitted");
    published_ = generation;
  }

  /** Revoke a published generation before a second in-place asynchronous
   * phase mutates its buffers. A later successful phase may republish the same
   * submitted generation with commit(); failure leaves no partial result
   * observable.
   */
  void revoke(std::uint64_t generation) {
    if (!generation || generation != submitted_)
      throw std::invalid_argument("asynchronous generation was not submitted");
    if (published_ == generation) published_ = 0;
  }

  void require(std::uint64_t generation) const {
    if (!generation || generation != published_)
      throw std::invalid_argument("asynchronous result generation is stale");
  }

  [[nodiscard]] std::uint64_t submitted() const noexcept { return submitted_; }
  [[nodiscard]] std::uint64_t published() const noexcept { return published_; }

 private:
  std::uint64_t submitted_{};
  std::uint64_t published_{};
};

/** Generic deterministic resource metadata. Scientific identity and
 * invalidation remain method-owned.
 */
struct ResourcePlan {
  std::size_t device_bytes{};
  std::size_t workspace_bytes{};
  std::size_t host_bytes{};
  std::size_t alignment{1};

  [[nodiscard]] bool valid() const noexcept {
    return alignment != 0 && workspace_bytes <= device_bytes;
  }
};

}  // namespace vibeqc::runtime
