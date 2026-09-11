#ifndef VIBEQC_API_HANDLES_HPP
#define VIBEQC_API_HANDLES_HPP

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/types.hpp"
#include "methods/method.hpp"
#include "scf/types.hpp"

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
  /** How the precision policy resolved for the most recent successful run. */
  vibeqc::scf::PrecisionProvenance precision{};
  /** True only after a run completes and populates \p precision. */
  bool precision_available{false};
};

struct vibeqc_batch {
  vibeqc_context* context{};
  std::unique_ptr<vibeqc::methods::PreparedBatch> plan;
  std::vector<std::uint32_t> atom_counts;
  std::vector<std::uint64_t> last_fock_builds;
  /** Input-ordered completed-run records; invalid/throwing items stay unavailable. */
  std::vector<std::optional<vibeqc::scf::PrecisionProvenance>> precision;
};

#endif
