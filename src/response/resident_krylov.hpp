#pragma once

#include <cstddef>
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
};

/** Exact slot/scalar requirement for the shared restarted-GMRES controller.
 *
 * Dimension-sized vectors remain resident. Only Hessenberg/Givens coefficients
 * and scalar reductions are host-owned. The slot inventory mirrors the existing
 * host solver: x/image/residual, (restart+1) Arnoldi vectors, restart
 * preconditioned vectors, work/candidate/candidate-image/candidate-residual,
 * best-x and best-residual.
 */
inline ResidentGmresWorkspace resident_gmres_workspace(const GmresPlan& plan) {
  const auto expected = prepare_gmres(plan.dimension, plan.options);
  if (plan.restart != expected.restart || plan.workspace_bytes != expected.workspace_bytes)
    throw std::invalid_argument("resident GMRES plan does not match its dimensions and options");

  if (plan.restart > (static_cast<std::size_t>(-1) - 10) / 2)
    throw std::overflow_error("resident GMRES vector-slot count overflow");
  const auto slots = 2 * plan.restart + 10;

  // H[(restart+1),restart], cosine, sine, transformed and coefficients.
  if (plan.restart &&
      plan.restart > (static_cast<std::size_t>(-1) - 5 * plan.restart - 1) / plan.restart)
    throw std::overflow_error("resident GMRES scalar workspace overflow");
  const auto scalar_elements = plan.restart * plan.restart + 5 * plan.restart + 1;
  if (scalar_elements > static_cast<std::size_t>(-1) / sizeof(double))
    throw std::overflow_error("resident GMRES scalar workspace overflow");
  return {slots, scalar_elements * sizeof(double)};
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

}  // namespace vibeqc::response
