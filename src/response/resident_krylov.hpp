#pragma once

#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>

#include "response/native_gmres.hpp"

namespace vibeqc::response {

/** Method-neutral resident vector storage/execution contract for GMRES.
 *
 * The controller owns scalar Arnoldi/Hessenberg bookkeeping on the host while
 * this backend owns dimension-sized Krylov vectors and the physical operator
 * action. Implementations may use CUDA, another accelerator, or a test double;
 * no device-specific type crosses this interface.
 */
class ResidentKrylovBackend {
 public:
  virtual ~ResidentKrylovBackend() = default;

  [[nodiscard]] virtual std::size_t dimension() const noexcept = 0;
  [[nodiscard]] virtual std::size_t vector_slots() const noexcept = 0;
  [[nodiscard]] virtual std::size_t owned_resident_bytes() const noexcept = 0;

  virtual void upload(std::size_t slot, std::span<const double> values) = 0;
  virtual void download(std::size_t slot, std::span<double> values) = 0;
  virtual void zero(std::size_t slot) = 0;
  virtual void copy(std::size_t destination, std::size_t source) = 0;
  virtual void scale(std::size_t slot, double alpha) = 0;
  virtual void axpy(std::size_t destination, double alpha, std::size_t source) = 0;
  [[nodiscard]] virtual double dot(std::size_t left, std::size_t right) = 0;
  [[nodiscard]] virtual double norm(std::size_t slot) = 0;
  virtual void apply(std::size_t destination, std::size_t source) = 0;
};

struct ResidentGmresWorkspace {
  std::size_t vector_slots{};
  std::size_t host_scalar_bytes{};
  std::size_t host_result_bytes{};
};

struct ResidentGmresResult {
  GmresResult result;
  ResidentGmresWorkspace workspace;
  std::size_t resident_owned_bytes{};

  [[nodiscard]] bool converged() const noexcept { return result.converged(); }
};

/** Exact slot/scalar requirement for the shared restarted-GMRES controller.
 *
 * Dimension-sized iteration vectors remain resident. Only Hessenberg/Givens
 * coefficients and the final returned solution are host-owned. The slot
 * inventory is x/RHS/residual, (restart+1) Arnoldi vectors, restart
 * preconditioned vectors, work/candidate/candidate-image/candidate-residual,
 * best-x and best-residual.
 */
inline ResidentGmresWorkspace resident_gmres_workspace(const GmresPlan& plan) {
  const auto expected = prepare_gmres(plan.dimension, plan.options);
  if (plan.restart != expected.restart || plan.workspace_bytes != expected.workspace_bytes)
    throw std::invalid_argument("resident GMRES plan does not match its dimensions and options");

  constexpr auto maximum = std::numeric_limits<std::size_t>::max();
  if (plan.restart > (maximum - 10) / 2)
    throw std::overflow_error("resident GMRES vector-slot count overflow");
  const auto slots = 2 * plan.restart + 10;

  // H[(restart+1),restart], cosine, sine, transformed and coefficients.
  if (plan.restart && plan.restart > maximum / plan.restart)
    throw std::overflow_error("resident GMRES scalar workspace overflow");
  const auto square = plan.restart * plan.restart;
  if (plan.restart > (maximum - square - 1) / 5)
    throw std::overflow_error("resident GMRES scalar workspace overflow");
  const auto scalar_elements = square + 5 * plan.restart + 1;
  if (scalar_elements > maximum / sizeof(double) || plan.dimension > maximum / sizeof(double))
    throw std::overflow_error("resident GMRES host workspace overflow");
  return {slots, scalar_elements * sizeof(double), plan.dimension * sizeof(double)};
}

/** Validate a resident backend before any upload or operator action. */
inline ResidentGmresWorkspace validate_resident_gmres_backend(
    const GmresPlan& plan, const ResidentKrylovBackend& backend) {
  const auto workspace = resident_gmres_workspace(plan);
  if (backend.dimension() != plan.dimension)
    throw std::invalid_argument("resident GMRES backend dimension does not match the plan");
  if (backend.vector_slots() < workspace.vector_slots)
    throw std::length_error("resident GMRES backend has insufficient vector slots");
  if (!backend.owned_resident_bytes())
    throw std::invalid_argument("resident GMRES backend must report owned resident storage");
  return workspace;
}

/** Execute restarted GMRES while keeping all dimension-sized iteration vectors
 * in backend-owned storage. The returned solution is downloaded exactly once.
 *
 * The current resident contract is intentionally unpreconditioned; it matches
 * the physical RHF Z-vector and current RCCSD Lambda consumers. Elementwise
 * preconditioners require a separate resident primitive rather than a hidden
 * host round trip.
 */
ResidentGmresResult solve_gmres_resident(const GmresPlan& plan, ResidentKrylovBackend& backend,
                                         std::span<const double> rhs,
                                         std::span<const double> initial_guess = {});

}  // namespace vibeqc::response
