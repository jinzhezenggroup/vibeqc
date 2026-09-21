#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/electronic_reference.hpp"
#include "dft/grid.hpp"
#include "dft/scf_diagnostic.hpp"
#include "scf/solver/final_state.hpp"

namespace vibeqc::dft {

/** Immutable semilocal model and owner identity not represented by the common
 * mean-field determinant identity. The owner and solve epoch prevent a
 * numerically identical replacement plan from authorizing an old state. */
struct KsModelIdentity {
  std::uint32_t version{1};
  std::uint32_t scf_domain_version{1};
  GridSpec grid;
  std::size_t tile_points{};
  /** 0=LDA, 1=PBE, 2=r2SCAN; part of immutable model provenance. */
  std::uint32_t functional{};
  unsigned spins{};
  // CPU is exactly -1; CUDA is a nonnegative visible device ordinal. The
  // determinant's resolved Fock backend is authoritative, never inferred here.
  int device{};
  std::uint64_t owner{};
  double semilocal_exchange_scale{1.0};
  double semilocal_correlation_scale{1.0};
  bool operator==(const KsModelIdentity&) const = default;
};

struct KsFinalStateIdentity {
  scf::solver::FinalStateIdentity determinant;
  KsModelIdentity model;
  bool operator==(const KsFinalStateIdentity&) const = default;
};

/** Physical KS evaluation at exactly the tagged density. Components use the
 * DFT energy convention and are not replaced by the quadratic HF formula. */
struct KsPhysicalState {
  KsFinalStateIdentity identity;
  bool physical{};
  std::vector<scf::reference::Matrix> density, fock;
  EnergyComponents components;
  double reported_energy{};
  double physical_residual{};
};

/** Canonical frame produced from the physical non-DIIS Fock. */
struct KsFinalStateCandidate {
  KsFinalStateIdentity identity;
  std::uint64_t fock_density_generation{};
  bool physical_origin{};
  std::vector<scf::reference::EigenResult> spins;
};

struct KsFinalStateDiagnostic {
  scf::solver::FinalStateDiagnostic determinant;
  double component_energy{};
  double reported_energy_error{};
  double physical_residual{};
};

/** Detached output authorized for a later stationary-gradient consumer. W is
 * absent unless explicitly requested after every identity and numerical gate
 * has passed. */
struct VerifiedKsFinalState {
  KsFinalStateIdentity identity;
  std::vector<scf::reference::Matrix> density, fock, weighted_density;
  std::vector<scf::reference::EigenResult> orbitals;
  EnergyComponents components;
  KsFinalStateDiagnostic diagnostic;
};

/** Borrow a validated RKS/UKS snapshot through the common core reference
 * contract. The caller retains overlap/hcore and state storage ownership. */
core::ElectronicReferenceView electronic_reference(const VerifiedKsFinalState& state,
                                                   const scf::reference::Matrix& overlap,
                                                   const scf::reference::Matrix& hcore);

bool validate_ks_final_state(const KsFinalStateIdentity& current,
                             const scf::reference::Matrix& overlap,
                             const scf::reference::Matrix& hcore, const KsPhysicalState& physical,
                             const KsFinalStateCandidate& candidate,
                             const scf::solver::FinalStateLimits& limits,
                             bool compute_weighted_density, VerifiedKsFinalState& output,
                             std::string& detail);

}  // namespace vibeqc::dft
