#pragma once

#include <array>

#include "cc/triples_response.hpp"

namespace generativeqc::cc::detail {

/** Shared input gates; both backends must accept the same scientific domain. */
double validate_triples_response(const Problem&, const SolverResult&, const std::vector<double>&,
                                 const std::vector<double>&, const TriplesResponseOptions&);

struct TriplesResponseLayout {
  std::array<std::size_t, 8> sizes{};
  std::size_t outputs{}, arena{}, q{}, host_bytes{}, device_bytes{};
  std::size_t numeric_bytes() const { return host_bytes + device_bytes; }
};

/** Charge all owned numeric storage, including both sides of CUDA transfers.
 * Borrowed problem, amplitudes and orbital energies belong to the caller's
 * retained phase. The device allocation has one final 256-byte alignment pad.
 */
TriplesResponseLayout triples_response_layout(std::size_t o, std::size_t v,
                                              std::size_t batch_capacity, bool cuda);

}  // namespace generativeqc::cc::detail
