#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

namespace vibeqc::runtime::df_progress {

/** Append-only diagnostic journal, independent of the completed host/CUDA
 * ledgers. Each line is closed before returning, so a killed process retains
 * its last BEGIN and submitted work. This is process-crash visibility, not an
 * fsync durability guarantee. A missing END is incomplete evidence.
 *
 * Set VIBEQC_DF_PROGRESS_TRACE to a fresh JSONL path. Disabled scopes perform
 * no allocation, clock read or I/O. Host END means returned (or exception),
 * never numerical convergence. CUDA callers explicitly distinguish completed
 * stream work from graph construction. Names/statuses are internal identifiers.
 */
class Scope {
 public:
  explicit Scope(const char* name, const char* execution = "host") noexcept {
    const char* path = std::getenv("VIBEQC_DF_PROGRESS_TRACE");
    if (!path || !*path) return;
    try {
      state_ = std::make_unique<State>();
      state_->path = path;
      state_->name = name;
      state_->execution = execution;
      state_->id = next_id_.fetch_add(1, std::memory_order_relaxed);
      state_->previous = active_;
      state_->parent = active_ ? static_cast<std::int64_t>(active_->id) : -1;
      state_->exceptions = std::uncaught_exceptions();
      state_->start = Clock::now();
      active_ = state_.get();
      emit(*state_, "BEGIN", "started");
    } catch (...) {
      state_.reset();
    }
  }
  ~Scope() { finish(); }
  Scope(const Scope&) = delete;
  Scope& operator=(const Scope&) = delete;

  bool enabled() const noexcept { return state_ != nullptr; }

  void finish(const char* status = "returned") noexcept {
    if (!state_) return;
    emit(*state_, "END", std::uncaught_exceptions() > state_->exceptions ? "exception" : status);
    active_ = state_->previous;
    state_.reset();
  }

  /** A value is an observation at this scope, not an additive total. Work
   * increments use distinct counter names and remain qualified by execution.
   */
  static void number(const char* name, std::uint64_t value) noexcept {
    if (active_) emit(*active_, "VALUE", "observed", name, value);
  }
  static void label(const char* name, const char* value) noexcept {
    if (active_) emit(*active_, "VALUE", "observed", name, 0, value);
  }

 private:
  using Clock = std::chrono::steady_clock;
  struct State {
    std::string path;
    const char* name{};
    const char* execution{};
    std::uint64_t id{};
    std::int64_t parent{-1};
    int exceptions{};
    Clock::time_point start;
    State* previous{};
  };
  inline static thread_local State* active_{};
  inline static std::atomic<std::uint64_t> next_id_{0};
  inline static std::mutex mutex_;
  std::unique_ptr<State> state_;

  static void quoted(std::FILE* file, std::string_view value) noexcept {
    std::fputc('"', file);
    for (unsigned char c : value) {
      if (c == '"' || c == '\\') std::fputc('\\', file);
      if (c < 32)
        std::fprintf(file, "\\u%04x", c);
      else
        std::fputc(c, file);
    }
    std::fputc('"', file);
  }
  static void emit(const State& state, const char* event, const char* status,
                   const char* key = nullptr, std::uint64_t value = 0,
                   const char* label = nullptr) noexcept {
    try {
      const auto now = Clock::now();
      const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch());
      const double elapsed = std::chrono::duration<double, std::milli>(now - state.start).count();
      std::lock_guard<std::mutex> lock(mutex_);
      auto* file = std::fopen(state.path.c_str(), "a");
      if (!file) return;  // The evidence reader rejects missing/truncated records.
      std::fprintf(file,
                   "{\"schema\":\"vibeqc.df_progress\",\"version\":1,\"id\":%llu,"
                   "\"parent\":%lld,\"time_ns\":%lld,\"elapsed_ms\":%.17g,"
                   "\"event\":\"%s\",\"status\":\"%s\",\"execution\":\"%s\",\"name\":",
                   static_cast<unsigned long long>(state.id), static_cast<long long>(state.parent),
                   static_cast<long long>(ns.count()), elapsed, event, status, state.execution);
      quoted(file, state.name);
      if (key) {
        std::fputs(",\"key\":", file);
        quoted(file, key);
        std::fputs(",\"value\":", file);
        if (label)
          quoted(file, label);
        else
          std::fprintf(file, "%llu", static_cast<unsigned long long>(value));
      }
      std::fputs("}\n", file);
      std::fclose(file);
    } catch (...) {
      // Observation must preserve scientific exceptions and return statuses.
    }
  }
};

inline void number(const char* name, std::uint64_t value) noexcept { Scope::number(name, value); }
inline void label(const char* name, const char* value) noexcept { Scope::label(name, value); }

}  // namespace vibeqc::runtime::df_progress
