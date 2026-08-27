#ifndef VIBEQC_API_HANDLES_HPP
#define VIBEQC_API_HANDLES_HPP

#include "core/types.hpp"
#include "methods/method.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct vibeqc_context {
  vibeqc::core::ContextState state;
  std::string last_detail;
};

struct vibeqc_system {
  vibeqc::core::System data;
};

struct vibeqc_calculation {
  vibeqc_context* context{};
  std::unique_ptr<vibeqc::methods::PreparedCalculation> plan;
};

struct vibeqc_batch {
  vibeqc_context* context{};
  std::unique_ptr<vibeqc::methods::PreparedBatch> plan;
  std::vector<std::uint32_t> atom_counts;
};

#endif
