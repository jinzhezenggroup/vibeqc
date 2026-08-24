#ifndef QCE_CORE_TYPES_HPP
#define QCE_CORE_TYPES_HPP

#include "qce/qce.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace qce::core {

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
  qce_basis_representation basis_representation{QCE_BASIS_CARTESIAN};
};

struct ScfOptions {
  unsigned max_iterations{100};
  unsigned diis_history{8};
  double energy_tolerance{1.0e-10};
  double density_tolerance{1.0e-8};
  double screening_tolerance{1.0e-12};
};

struct ScfResult {
  double energy{};
  std::vector<double> forces;
  // The converged AO density is retained internally for topology-compatible
  // warm starts. RHF stores one N x N matrix; UHF stores alpha then beta
  // matrices. It is never exposed as an implicit public buffer.
  std::vector<double> density;
  unsigned iterations{};
  double energy_change{};
  double density_rms{};
  bool converged{};
  bool initial_density_used{};
};

struct ContextState {
  qce_backend requested_backend{QCE_BACKEND_CPU_REFERENCE};
  qce_backend executed_backend{QCE_BACKEND_CPU_REFERENCE};
  int device_id{0};
  std::string device_name;
};

}  // namespace qce::core

#endif
