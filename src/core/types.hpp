#ifndef VIBEQC_CORE_TYPES_HPP
#define VIBEQC_CORE_TYPES_HPP

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "vibeqc/vibeqc.h"

namespace vibeqc::core {

struct Atom {
  int atomic_number{};
  std::array<double, 3> position{};
};

struct Primitive {
  double exponent{};
  double coefficient{};
};

struct Shell {
  std::uint32_t atom_index{};
  std::uint32_t angular_momentum{};
  std::vector<Primitive> primitives;
};

struct System {
  std::vector<Atom> atoms;
  std::vector<Shell> shells;
  int charge{};
  unsigned multiplicity{1};
  int electron_count{};
  vibeqc_basis_representation basis_representation{VIBEQC_BASIS_CARTESIAN};
};

struct ContextState {
  vibeqc_backend requested_backend{VIBEQC_BACKEND_CPU_REFERENCE};
  vibeqc_backend executed_backend{VIBEQC_BACKEND_CPU_REFERENCE};
  int device_id{0};
  std::string device_name;
  int compute_capability_major{};
  int compute_capability_minor{};
  int warp_size{};
  int maximum_threads_per_sm{};
  int maximum_blocks_per_sm{};
  int registers_per_sm{};
  int maximum_registers_per_thread{};
  std::size_t shared_memory_per_block{};
  std::size_t shared_memory_per_sm{};
  int multiprocessor_count{};
  std::string aot_profile_name{"generic_cuda"};
  bool aot_profile_tuned{};
  bool aot_profile_portable{true};
  bool aot_profile_compatible{};
};

}  // namespace vibeqc::core

#endif
