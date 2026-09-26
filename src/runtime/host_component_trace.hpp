#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "runtime/df_progress_trace.hpp"
#include "scf/reference/observation.hpp"

namespace vibeqc::runtime::host_trace {

/** Host counterpart of #206's CUDA component ledger. Opt in with a fresh
 * VIBEQC_DF_HOST_TRACE JSONL path. Nested wall/thread-CPU intervals are
 * inclusive; consumers subtract immediate children and never add CPU/GPU
 * clocks. No allocation, clock read, file I/O or CUDA fence occurs disabled.
 * Diagnostic runs include tracing overhead and cannot replace clean timings.
 */
namespace observation = scf::reference::observation;
using observation::active_reason;
using observation::EigenReason;
using observation::Reason;
inline thread_local long long active_item = -1;

/** An item is a source index within the enclosing native bucket, not a CUDA
 * spin-expanded slot. The endpoint/bucket scope disambiguates ragged groups. */
class Item {
 public:
  explicit Item(std::size_t value) noexcept : previous_(active_item) {
    active_item = static_cast<long long>(value);
  }
  ~Item() { active_item = previous_; }
  Item(const Item&) = delete;
  Item& operator=(const Item&) = delete;

 private:
  long long previous_;
};

namespace detail {
using Clock = std::chrono::steady_clock;
inline double thread_seconds() noexcept {
#if defined(CLOCK_THREAD_CPUTIME_ID)
  timespec value{};
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) == 0)
    return static_cast<double>(value.tv_sec) + 1e-9 * value.tv_nsec;
#endif
  // Never substitute process CPU time: another thread's work is not this item.
  return -1;
}
inline const char* reason_name(EigenReason reason) noexcept {
  switch (reason) {
    case EigenReason::overlap:
      return "overlap";
    case EigenReason::core_guess:
      return "core_guess";
    case EigenReason::final_fock:
      return "final_fock";
    case EigenReason::reference_export:
      return "reference_export";
    case EigenReason::fallback:
      return "fallback";
    case EigenReason::iteration:
      return "iteration";
    case EigenReason::seed_validation:
      return "seed_validation";
    default:
      return "unspecified";
  }
}
struct Region {
  const char* name{};
  EigenReason reason{};
  long long parent{-1}, item{-1};
  std::size_t nbf{};
  Clock::time_point start;
  double cpu_start{}, wall_ms{}, cpu_ms{-1};
  int exceptions{};
  bool finished{}, failed{};
};
struct State {
  std::string path;
  std::vector<Region> regions;
  long long current{-1};
  std::uint64_t id{};
  bool valid{true};
};
inline thread_local State* active{};
inline std::atomic<std::uint64_t> next_id{0};
inline std::mutex output_mutex;

inline std::size_t begin(const char* name, std::size_t nbf) noexcept {
  if (!active) return std::numeric_limits<std::size_t>::max();
  try {
    if (active->regions.size() >= 65536) {
      active->valid = false;
      return std::numeric_limits<std::size_t>::max();
    }
    const auto index = active->regions.size();
    active->regions.push_back({name, active_reason, active->current, active_item, nbf, Clock::now(),
                               thread_seconds(), 0, -1, std::uncaught_exceptions()});
    active->current = static_cast<long long>(index);
    return index;
  } catch (...) {
    active->valid = false;
    return std::numeric_limits<std::size_t>::max();
  }
}
inline void end(std::size_t index, int exceptions) noexcept {
  if (!active || index >= active->regions.size()) return;
  auto& row = active->regions[index];
  row.wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - row.start).count();
  const auto cpu_end = thread_seconds();
  if (row.cpu_start >= 0 && cpu_end >= row.cpu_start) row.cpu_ms = 1000 * (cpu_end - row.cpu_start);
  row.finished = true;
  row.failed = std::uncaught_exceptions() > exceptions;
  if (active->current != static_cast<long long>(index)) active->valid = false;
  active->current = row.parent;
}
inline const observation::Observer observer{begin, end};

inline void write(const State& state) {
  std::lock_guard<std::mutex> lock(output_mutex);
  auto* file = std::fopen(state.path.c_str(), "a");
  if (!file) return;  // The runner rejects missing/incomplete trace evidence.
  std::fprintf(file,
               "{\"schema\":\"vibeqc.df_host_trace\",\"version\":1,\"id\":%llu,"
               "\"valid\":%s,\"regions\":[",
               static_cast<unsigned long long>(state.id), state.valid ? "true" : "false");
  bool comma = false;
  for (const auto& row : state.regions) {
    // Names/reasons are internal static identifiers, never user input.
    std::fprintf(file,
                 "%s{\"name\":\"%s\",\"reason\":\"%s\",\"parent\":%lld,\"item\":%lld,"
                 "\"nbf\":%zu,\"wall_ms\":%.17g,\"cpu_ms\":",
                 comma ? "," : "", row.name, reason_name(row.reason), row.parent, row.item, row.nbf,
                 row.wall_ms);
    if (row.cpu_ms >= 0)
      std::fprintf(file, "%.17g", row.cpu_ms);
    else
      std::fputs("null", file);
    std::fprintf(file, ",\"finished\":%s,\"failed\":%s}", row.finished ? "true" : "false",
                 row.failed ? "true" : "false");
    comma = true;
  }
  std::fputs("]}\n", file);
  std::fclose(file);
}
}  // namespace detail

/** Scope around actual work; the reference_eigensolve leaf counts executions,
 * including exceptions. It must not be placed around a plan/capture declaration.
 * Scope names must be static ASCII identifiers without JSON metacharacters.
 */
class Region {
 public:
  explicit Region(const char* name, std::size_t nbf = 0) noexcept : progress_(name) {
    df_progress::number("nbf", nbf);
    df_progress::number("item_plus_one", static_cast<std::uint64_t>(active_item + 1));
    auto* state = detail::active;
    try {
      if (!state) {
        const char* path = std::getenv("VIBEQC_DF_HOST_TRACE");
        if (!path || !*path) return;
        owner_ = std::make_unique<detail::State>();
        state = owner_.get();
        state->path = path;
        state->id = detail::next_id.fetch_add(1, std::memory_order_relaxed);
      }
      detail::active = state;
      index_ = detail::begin(name, nbf);
      if (index_ == std::numeric_limits<std::size_t>::max()) {
        if (owner_) {
          detail::active = nullptr;
          // Preserve an explicitly invalid root so partial evidence cannot
          // be mistaken for a successful zero-call execution.
          try {
            detail::write(*owner_);
          } catch (...) {
          }
        }
        return;
      }
      if (owner_) {
        previous_observer_ = observation::active;
        observation::active = &detail::observer;
      }
      state_ = state;
    } catch (...) {
      if (state) state->valid = false;
      owner_.reset();
    }
  }
  ~Region() { finish(); }
  void finish() noexcept {
    progress_.finish();
    if (!state_) return;
    detail::end(index_, state_->regions[index_].exceptions);
    state_ = nullptr;
    if (owner_) {
      observation::active = previous_observer_;
      detail::active = nullptr;
      try {
        detail::write(*owner_);
      } catch (...) {
      }
      owner_.reset();
    }
  }
  Region(const Region&) = delete;
  Region& operator=(const Region&) = delete;

 private:
  df_progress::Scope progress_;
  std::unique_ptr<detail::State> owner_;
  detail::State* state_{};
  const observation::Observer* previous_observer_{};
  std::size_t index_{};
};

/** Preserve C++17 prvalue elision for nonmovable prepared providers. */
template <class Function>
decltype(auto) call(const char* name, Function&& function) {
  Region scope(name);
  return function();
}
template <class Function>
decltype(auto) with_reason(EigenReason reason, Function&& function) {
  Reason scope(reason);
  return function();
}
}  // namespace vibeqc::runtime::host_trace
