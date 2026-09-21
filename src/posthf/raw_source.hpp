#pragma once

#include <array>
#include <memory>

#include "integrals/electron_interaction_source.hpp"
#include "posthf/capacity.hpp"

namespace vibeqc::posthf {

/** Values-only public-AO tiles from the independent contracted CPU evaluator.
 *
 * RawSource remains the compatibility owner used by post-HF call sites, but its
 * numerical surface now implements the method-neutral integrals-layer source
 * contract. New consumers should depend on ElectronInteractionSource rather than
 * this concrete implementation.
 *
 * The source owns normalized basis/geometry data. It never allocates molecular
 * four-index tensors or nuclear derivative jets. Offsets are global public-AO
 * indices; the Python CG02 adapter additionally checks shell-local requests.
 */
class RawSource final : public integrals::ElectronInteractionSource {
 public:
  using Operator = integrals::ElectronInteractionOperator;

  RawSource(core::System orbital, const core::System* auxiliary = nullptr);
  ~RawSource() override;
  RawSource(const RawSource&) = delete;
  RawSource& operator=(const RawSource&) = delete;
  const core::System& orbital() const override;
  const core::System& auxiliary() const;
  std::size_t nbf() const override;
  std::size_t naux() const override;
  std::size_t retained_numeric_bytes() const override;
  bool supports(Operator op) const noexcept override {
    switch (op) {
      case Operator::overlap:
      case Operator::hcore:
      case Operator::eri:
        return true;
      case Operator::metric:
      case Operator::three_center:
        return naux() != 0;
    }
    return false;
  }
  /** Evaluate exactly the requested row-major tile into caller-owned storage.
   * Empty extents at valid endpoints are legal. Invalid ranges and arithmetic
   * overflow fail before writes. Recurrence scratch is independent of tile size.
   */
  void read(Operator op, const std::array<std::size_t, 4>& begin,
            const std::array<std::size_t, 4>& count, double* out,
            std::size_t elements) const override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace vibeqc::posthf
