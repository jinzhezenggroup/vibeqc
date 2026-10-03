#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "cc/triples_response.hpp"

namespace generativeqc::cc {

/** Complete same-space standard-(T) Fock cotangents, in dense MO coordinates.
 * Includes diagonal orbital-energy sources. Consumers must replace, rather
 * than also add, the old diagonal plus canonical-gap response with these seeds.
 */
struct TriplesFockResponseResult {
  std::vector<double> foo, fvv;
  std::size_t page_capacity{}, pair_panels{}, vector_pages{}, occupied_moments{}, virtual_moments{};
  std::size_t numeric_capacity_bytes{}, device_capacity_bytes{};
  std::size_t host_to_device_bytes{}, device_to_host_bytes{}, device_copy_bytes{},
      kernel_launches{};
  const char* program_hash{};
};

namespace detail {
struct TriplesFockLayout {
  std::array<std::size_t, 8> input_sizes{};
  std::size_t q{}, inputs{}, outputs{}, vector_elements{}, arena{}, host_bytes{}, device_bytes{};
  std::size_t numeric_bytes() const { return host_bytes + device_bytes; }
};

/** Complete owned numeric storage for two bounded vector pages and scratch.
 * Borrowed scientific inputs remain charged to the caller. CUDA additionally
 * owns resident input copies; all device transfers and the error flag are charged.
 */
TriplesFockLayout triples_fock_response_layout(std::size_t o, std::size_t v, std::size_t capacity,
                                               bool cuda);
}  // namespace detail

TriplesFockResponseResult triples_fock_response_cpu(const Problem&, const SolverResult&,
                                                    const std::vector<double>& eps_o,
                                                    const std::vector<double>& eps_v,
                                                    const TriplesResponseOptions& = {});
#if GENERATIVEQC_HAS_CUDA
TriplesFockResponseResult triples_fock_response_cuda(const Problem&, const SolverResult&,
                                                     const std::vector<double>& eps_o,
                                                     const std::vector<double>& eps_v, int device,
                                                     const TriplesResponseOptions& = {});
#endif

}  // namespace generativeqc::cc
