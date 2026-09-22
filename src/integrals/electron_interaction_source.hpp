#ifndef VIBEQC_INTEGRALS_ELECTRON_INTERACTION_SOURCE_HPP
#define VIBEQC_INTEGRALS_ELECTRON_INTERACTION_SOURCE_HPP

#include <array>
#include <cstddef>

#include "core/types.hpp"

namespace vibeqc::integrals {

/** Method-neutral AO interaction operators exposed by a bounded source.
 *
 * The source contract deliberately stops below SCF/DFT/CC semantics: consumers
 * ask only for mathematically defined AO-space tiles. Fock assembly, orbital
 * transforms, correlation equations and response remain owned by their method
 * layers.
 */
enum class ElectronInteractionOperator { overlap, hcore, eri, metric, three_center };

/** Read-only AO interaction source shared by mean-field and post-HF adapters.
 *
 * Implementations own their normalized geometry/basis state and may generate,
 * cache or stream values internally. A consumer must check supports() before
 * requesting an optional operator. read() is transactional with respect to the
 * caller-owned output buffer: validation must complete before publishing values.
 */
class ElectronInteractionSource {
 public:
  using Operator = ElectronInteractionOperator;

  virtual ~ElectronInteractionSource() = default;
  virtual const core::System& orbital() const = 0;
  virtual std::size_t nbf() const = 0;
  virtual std::size_t naux() const = 0;
  /** Numeric bytes retained by this source while a consumer borrows it.
   * Consumers use this value for endpoint memory admission; it must not omit
   * resident value tensors merely because they are immutable or shared.
   */
  virtual std::size_t retained_numeric_bytes() const = 0;
  virtual bool supports(Operator op) const noexcept = 0;

  virtual void read(Operator op, const std::array<std::size_t, 4>& begin,
                    const std::array<std::size_t, 4>& count, double* out,
                    std::size_t elements) const = 0;
};

}  // namespace vibeqc::integrals
#endif
