#pragma once

#include "posthf/native_provider.hpp"

namespace vibeqc::mp2 {
struct Energy {
  double opposite_spin{}, same_spin{}, minimum_denominator{};
  std::size_t numeric_capacity_bytes{}, tiles{};
  const char* equation_hash{};
  vibeqc_tensor::Metrics metrics;
  std::size_t mo_transfer_bytes{};
};
Energy conventional_energy(const scf::PhysicalReference& reference, const posthf::RawSource& source,
                           std::size_t budget, double denominator_threshold,
                           unsigned virtual_tile = 8, bool cuda = false, int device = 0);
}  // namespace vibeqc::mp2
