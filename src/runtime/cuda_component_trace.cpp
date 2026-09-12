#include "runtime/cuda_component_trace.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define VIBEQC_DF_TRACE_NVTX 1
#else
#define VIBEQC_DF_TRACE_NVTX 0
#endif

namespace vibeqc::runtime::cuda_trace {
namespace {
using Clock = std::chrono::steady_clock;
constexpr std::size_t kMaximumRegions = 65536;
constexpr std::size_t kMaximumTiles = 65536;
std::atomic<std::uint64_t> next_operation{0};
std::mutex output_mutex;

double milliseconds(Clock::time_point begin, Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

void quoted(std::FILE* output, std::string_view value) {
  std::fputc('"', output);
  for (const unsigned char c : value) {
    if (c == '"' || c == '\\') std::fputc('\\', output);
    if (c < 32)
      std::fprintf(output, "\\u%04x", c);
    else
      std::fputc(c, output);
  }
  std::fputc('"', output);
}

void push_range(const char* name) {
#if VIBEQC_DF_TRACE_NVTX
  nvtxRangePushA(name);
#else
  (void)name;
#endif
}
void pop_range() {
#if VIBEQC_DF_TRACE_NVTX
  nvtxRangePop();
#endif
}
}  // namespace

struct TraceOperation::State {
  struct Region {
    const char* name{};
    std::int64_t parent{-1};
    Clock::time_point begin{}, end{};
    cudaEvent_t first{}, last{};
    bool finished{};
  };
  using Tile = std::array<std::uint64_t, 7>;
  std::string path;
  const char* operation{};
  cudaStream_t stream{};
  TraceShape shape;
  State* previous{};
  std::int64_t current{-1};
  std::uint64_t id{};
  bool capture{};
  bool invalid{};
  int cuda_error{};
  std::size_t dropped_regions{}, dropped_tiles{};
  std::vector<Region> regions;
  std::map<Tile, std::uint64_t> tiles;
  std::map<std::string_view, std::uint64_t> counters;

  void check(cudaError_t error) noexcept {
    if (error != cudaSuccess) {
      invalid = true;
      if (!cuda_error) cuda_error = static_cast<int>(error);
    }
  }

  std::size_t begin(const char* name) {
    if (regions.size() == kMaximumRegions) {
      ++dropped_regions;
      invalid = true;
      return kMaximumRegions;
    }
    const auto index = regions.size();
    regions.push_back({.name = name, .parent = current, .begin = Clock::now()});
    auto& region = regions.back();
    if (!capture) {
      check(cudaEventCreate(&region.first));
      check(cudaEventCreate(&region.last));
      if (region.first) check(cudaEventRecord(region.first, stream));
    }
    current = static_cast<std::int64_t>(index);
    push_range(name);
    return index;
  }

  void finish(std::size_t index) noexcept {
    if (index >= regions.size() || regions[index].finished) return;
    auto& region = regions[index];
    if (current != static_cast<std::int64_t>(index)) invalid = true;
    if (!capture && region.last) check(cudaEventRecord(region.last, stream));
    region.end = Clock::now();
    region.finished = true;
    current = region.parent;
    pop_range();
  }

  ~State() {
    for (const auto& region : regions) {
      if (region.first) (void)cudaEventDestroy(region.first);
      if (region.last) (void)cudaEventDestroy(region.last);
    }
  }

  void write(double synchronization_ms) {
    std::vector<double> gpu_ms(regions.size(), -1.0);
    for (std::size_t i = 0; i < regions.size(); ++i) {
      const auto& region = regions[i];
      if (!region.finished) invalid = true;
      if (capture) continue;
      if (!region.finished || !region.first || !region.last) {
        invalid = true;
        continue;
      }
      float elapsed = 0;
      const auto status = cudaEventElapsedTime(&elapsed, region.first, region.last);
      check(status);
      if (status == cudaSuccess && std::isfinite(elapsed) && elapsed >= 0)
        gpu_ms[i] = elapsed;
      else
        invalid = true;
    }
    // Serialize one complete record per operation. The sink is diagnostic and
    // cannot throw through a scientific execution or corrupt another thread's
    // record. Consumers reject invalid/truncated traces before aggregation.
    std::lock_guard<std::mutex> lock(output_mutex);
    std::FILE* output = std::fopen(path.c_str(), "a");
    if (!output) return;
    std::fprintf(output, "{\"schema\":\"vibeqc.df_trace\",\"version\":1,\"id\":%llu,\"operation\":",
                 static_cast<unsigned long long>(id));
    quoted(output, operation);
    std::fprintf(
        output,
        ",\"execution\":\"%s\",\"valid\":%s,\"cuda_error\":%d,\"nvtx\":%s,"
        "\"systems\":%zu,\"system_offset\":%zu,\"nbf\":%zu,\"naux\":%zu,\"source_backed\":%s,"
        "\"streamed\":%s,"
        "\"final_synchronization_ms\":%.9g,\"host_completion_ms\":%.9g,"
        "\"profiler_event_count\":%zu,\"dropped_regions\":%zu,\"dropped_tiles\":%zu,\"regions\":[",
        capture ? "graph_capture" : "stream", invalid ? "false" : "true", cuda_error,
        VIBEQC_DF_TRACE_NVTX ? "true" : "false", shape.systems, shape.system_offset, shape.nbf,
        shape.naux, shape.source_backed ? "true" : "false", shape.streamed ? "true" : "false",
        synchronization_ms, milliseconds(regions[0].begin, regions[0].end) + synchronization_ms,
        capture ? 0U : 2 * regions.size(), dropped_regions, dropped_tiles);
    bool comma = false;
    for (std::size_t i = 0; i < regions.size(); ++i) {
      const auto& region = regions[i];
      if (comma) std::fputc(',', output);
      comma = true;
      std::fputs("{\"name\":", output);
      quoted(output, region.name);
      std::fprintf(output, ",\"parent\":%lld,\"host_ms\":%.9g,\"gpu_ms\":",
                   static_cast<long long>(region.parent), milliseconds(region.begin, region.end));
      if (gpu_ms[i] >= 0)
        std::fprintf(output, "%.9g", gpu_ms[i]);
      else
        std::fputs("null", output);
      std::fputc('}', output);
    }
    std::fputs("],\"counters\":{", output);
    comma = false;
    for (const auto& [name, value] : counters) {
      if (comma) std::fputc(',', output);
      comma = true;
      quoted(output, name);
      std::fprintf(output, ":%llu", static_cast<unsigned long long>(value));
    }
    std::fputs("},\"tiles\":[", output);
    comma = false;
    for (const auto& [tile, count] : tiles) {
      if (comma) std::fputc(',', output);
      comma = true;
      std::fprintf(
          output,
          "{\"system\":%llu,\"pair_begin\":%llu,\"pair_count\":%llu,"
          "\"auxiliary_begin\":%llu,\"auxiliary_count\":%llu,"
          "\"derivative_coordinate\":%lld,\"transformed\":%s,\"productions\":%llu}",
          static_cast<unsigned long long>(tile[0]), static_cast<unsigned long long>(tile[1]),
          static_cast<unsigned long long>(tile[2]), static_cast<unsigned long long>(tile[3]),
          static_cast<unsigned long long>(tile[4]), static_cast<long long>(tile[5]),
          tile[6] ? "true" : "false", static_cast<unsigned long long>(count));
    }
    std::fputs("]}\n", output);
    std::fclose(output);
  }
};

thread_local TraceOperation::State* TraceOperation::active_ = nullptr;

TraceOperation::TraceOperation(const char* operation, cudaStream_t stream,
                               TraceShape shape) noexcept {
  const char* path = std::getenv("VIBEQC_DF_TRACE");
  if (!path || !*path) return;
  try {
    state_ = std::make_unique<State>();
    state_->path = path;
    state_->operation = operation;
    state_->stream = stream;
    state_->shape = shape;
    state_->id = next_operation.fetch_add(1, std::memory_order_relaxed);
    cudaStreamCaptureStatus capture{};
    state_->check(cudaStreamIsCapturing(stream, &capture));
    state_->capture = capture != cudaStreamCaptureStatusNone;
    state_->previous = active_;
    state_->begin(operation);
    active_ = state_.get();
  } catch (...) {
    state_.reset();
  }
}

TraceOperation::~TraceOperation() {
  if (!state_) return;
  state_->finish(0);
  active_ = state_->previous;
  const auto start = Clock::now();
  if (!state_->capture && !state_->regions.empty() && state_->regions[0].last)
    state_->check(cudaEventSynchronize(state_->regions[0].last));
  const auto elapsed = milliseconds(start, Clock::now());
  try {
    state_->write(elapsed);
  } catch (...) {
    // Profiling must not replace a calculation's status while unwinding.
  }
}

TraceRegion::TraceRegion(const char* name, cudaStream_t stream) noexcept {
  auto* active = TraceOperation::active_;
  if (!active || active->stream != stream) return;
  try {
    index_ = active->begin(name);
    if (index_ != kMaximumRegions) state_ = active;
  } catch (...) {
    active->invalid = true;
  }
}
TraceRegion::~TraceRegion() { finish(); }
void TraceRegion::finish() noexcept {
  if (state_) state_->finish(index_);
  state_ = nullptr;
}

void trace_counter(const char* name, std::uint64_t count) noexcept {
  auto* active = TraceOperation::active_;
  if (!active) return;
  try {
    auto& value = active->counters[name];
    if (count > std::numeric_limits<std::uint64_t>::max() - value)
      active->invalid = true;
    else
      value += count;
  } catch (...) {
    active->invalid = true;
  }
}

void trace_tile(std::size_t system, std::size_t pair_begin, std::size_t pair_count,
                std::size_t auxiliary_begin, std::size_t auxiliary_count,
                std::int64_t derivative_coordinate, bool transformed) noexcept {
  auto* active = TraceOperation::active_;
  if (!active || !pair_count || !auxiliary_count) return;
  try {
    const TraceOperation::State::Tile key{system,
                                          pair_begin,
                                          pair_count,
                                          auxiliary_begin,
                                          auxiliary_count,
                                          static_cast<std::uint64_t>(derivative_coordinate),
                                          static_cast<std::uint64_t>(transformed)};
    auto found = active->tiles.find(key);
    if (found == active->tiles.end() && active->tiles.size() == kMaximumTiles) {
      ++active->dropped_tiles;
      active->invalid = true;
      return;
    }
    auto& count = active->tiles[key];
    if (count == std::numeric_limits<std::uint64_t>::max())
      active->invalid = true;
    else
      ++count;
    const bool derivative = derivative_coordinate >= 0;
    trace_counter(derivative    ? "derivative_tile_productions"
                  : transformed ? "transformed_tile_productions"
                                : "raw_tile_productions",
                  1);
    if (pair_count > std::numeric_limits<std::uint64_t>::max() / auxiliary_count / sizeof(double)) {
      active->invalid = true;
      return;
    }
    trace_counter(derivative    ? "derivative_value_bytes"
                  : transformed ? "transformed_value_bytes"
                                : "raw_value_bytes",
                  pair_count * auxiliary_count * sizeof(double));
  } catch (...) {
    active->invalid = true;
  }
}
}  // namespace vibeqc::runtime::cuda_trace
