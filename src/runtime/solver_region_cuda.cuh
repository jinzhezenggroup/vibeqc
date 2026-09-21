// Method-neutral CUDA execution owner for bounded SolverRegion bodies.
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "cuda_graph_region.cuh"

namespace vibeqc::runtime {

enum class SolverRegionCompletionMode : std::uint8_t { Scalar = 0, PerItemMask = 1 };

struct SolverRegionCudaBinding {
  GraphBinding graph;
  std::uint32_t max_steps_per_checkpoint = 1;
  SolverRegionCompletionMode completion = SolverRegionCompletionMode::Scalar;
  bool replay_enabled = false;
};

struct SolverRegionCudaMetrics {
  std::uint64_t submissions = 0;
  std::uint64_t submitted_steps = 0;
  std::uint64_t checkpoints = 0;
  std::uint32_t last_width = 0;
};

class SolverRegionCudaExecutor {
 public:
  SolverRegionCudaMetrics metrics;

  template <class F>
  unsigned submit(const SolverRegionCudaBinding& binding, unsigned requested_steps,
                  unsigned remaining_steps, bool profile, F&& submit_step) {
    if (!binding.max_steps_per_checkpoint)
      throw std::invalid_argument("solver-region checkpoint width must be positive");
    if (!requested_steps || !remaining_steps)
      throw std::invalid_argument("solver-region submission requires remaining work");
    const auto width =
        std::min({requested_steps, remaining_steps, binding.max_steps_per_checkpoint});
    auto replay_binding = binding.graph;
    replay_binding.qualification += ":solver-region:w" + std::to_string(width) + ":b" +
                                    std::to_string(binding.max_steps_per_checkpoint) + ":c" +
                                    std::to_string(static_cast<unsigned>(binding.completion));
    auto region = [&] {
      for (unsigned slot = 0; slot < width; ++slot) submit_step(slot);
    };
    replay_.submit(replay_binding, binding.replay_enabled, profile, region);
    ++metrics.submissions;
    metrics.submitted_steps += width;
    metrics.last_width = width;
    return width;
  }

  void checkpoint() noexcept { ++metrics.checkpoints; }

  void invalidate() {
    replay_.invalidate();
    metrics.last_width = 0;
  }

  const GraphMetrics& replay_metrics() const noexcept { return replay_.metrics; }
  const std::string& replay_reason() const noexcept { return replay_.reason; }

 private:
  CudaGraphRegion replay_;
};

}  // namespace vibeqc::runtime
