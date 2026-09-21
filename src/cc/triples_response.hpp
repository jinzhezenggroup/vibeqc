#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "cc/solver.hpp"

namespace vibeqc::cc {

struct TriplesResponseOptions {
  double denominator_threshold{1e-10};
  std::size_t max_bytes{256ULL << 20};
  std::size_t batch_capacity{16};
};

struct TriplesResponseResult {
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
  double minimum_absolute_denominator{};
  const char* program_hash{};
  std::string reason;
};

TriplesResponseResult triples_response_cpu(const Problem& problem, const SolverResult& cc,
                                           const std::vector<double>& eps_o,
                                           const std::vector<double>& eps_v,
                                           const TriplesResponseOptions& options = {});

}  // namespace vibeqc::cc
