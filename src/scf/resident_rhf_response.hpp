#pragma once

#include <cstddef>
#include <memory>
#include <span>

#include "response/resident_krylov.hpp"

namespace vibeqc::scf {

class PreparedFockPlan;

/** Bind the existing CUDA resident RHF response owner to the method-neutral
 * Krylov interface without exposing the C handle or duplicating response
 * mathematics. The prepared Fock plan retains the exact J/K source and must
 * outlive the returned backend.
 */
std::unique_ptr<response::ResidentKrylovBackend> make_resident_rhf_krylov_backend(
    PreparedFockPlan& plan, std::span<const double> coefficients,
    std::span<const double> orbital_energies, std::size_t nocc, std::size_t vector_slots,
    std::size_t device_budget_bytes);

}  // namespace vibeqc::scf
