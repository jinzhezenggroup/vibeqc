#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "cc/solver.hpp"

namespace generativeqc::cc {

struct TriplesResponseOptions {
  double denominator_threshold{1e-10};
  std::size_t max_bytes{256ULL << 20};
  std::size_t batch_capacity{16};
};

struct TriplesResponseResult {
  // Integral-source cotangents are accumulated directly in the symmetry
  // convention consumed by the RCCSD(T) Hamiltonian response.  The producer
  // publishes only this canonical projected representation.
  std::vector<double> ovvv;
  std::vector<double> ovoo;
  std::vector<double> ovov;
  std::vector<double> fov;
  std::vector<double> t1;
  std::vector<double> t2;
  std::vector<double> eps_o;
  std::vector<double> eps_v;
  std::size_t pages{};
  std::size_t arena_bytes{};
  std::size_t numeric_capacity_bytes{};
  std::size_t device_capacity_bytes{};
  std::size_t host_to_device_bytes{};
  std::size_t device_to_host_bytes{};
  std::size_t kernel_launches{};
  double minimum_absolute_denominator{};
  const char* program_hash{};
  std::string reason;
};

TriplesResponseResult triples_response_cpu(const Problem& problem, const SolverResult& cc,
                                           const std::vector<double>& eps_o,
                                           const std::vector<double>& eps_v,
                                           const TriplesResponseOptions& options = {});

/** Evaluate all eight standard-(T) cotangents on the selected CUDA device.
 * Inputs remain resident across triangular virtual-triple pages. Only bounded
 * controls are uploaded per page; projected cotangents are downloaded once.
 * Insufficient budget or device failure is explicit; no CPU fallback occurs.
 */
#if GENERATIVEQC_HAS_CUDA
TriplesResponseResult triples_response_cuda(const Problem& problem, const SolverResult& cc,
                                            const std::vector<double>& eps_o,
                                            const std::vector<double>& eps_v, int device,
                                            const TriplesResponseOptions& options = {});
#endif

}  // namespace generativeqc::cc
