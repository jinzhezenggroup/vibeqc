#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>

#include "core/types.hpp"
#include "generativeqc/generativeqc.h"

namespace generativeqc::posthf {

/** Successful consumer work for one shell; timings include its existing fence.
 * Caller record construction is excluded from consumer_nanoseconds. No extra
 * CUDA synchronization is introduced; a null diagnostic avoids clock reads.
 */
struct CudaShellDerivativeDiagnostic {
  std::size_t primitive_records{};
  std::size_t consumer_calls{};
  std::uint64_t consumer_nanoseconds{};
};

/** Contract one public-AO shell-quartet cotangent with CUDA ERI derivatives.
 *
 * The four center derivatives remain independent; the caller owns physical-
 * atom scatter. Caller weights, output, and system storage are outside
 * stage_budget. Output is replaced only after complete success.
 */
generativeqc_status contract_weighted_eri_shell_derivative_cuda(
    int device_id, const core::System& system, const std::array<std::size_t, 4>& shell_indices,
    std::span<const double> weights, std::size_t stage_budget,
    std::array<double, 12>& center_gradient, std::string& detail,
    CudaShellDerivativeDiagnostic* diagnostic = nullptr);

/** Bounded stream of shell cotangents into an atom-ordered gradient.
 *
 * append consumes its weights immediately. Several quartets may share one
 * primitive upload; finish drains the final partial batch before publication.
 * Four center slots remain distinct until their results are added to atoms.
 * The borrowed system/gradient must remain stable until finish; errors throw
 * and the caller must discard its partial gradient. No CPU fallback is used.
 *
 * Numeric storage is capped by both stage_budget and source_scratch_bytes.
 * Below the batching admission threshold, append uses the one-shell consumer.
 * Buffers are allocated lazily so one-electron work need not overlap them.
 */
class CudaEriDerivativeBatch {
 public:
  CudaEriDerivativeBatch(int device_id, const core::System& system, std::size_t stage_budget,
                         bool trace = false);
  ~CudaEriDerivativeBatch();
  CudaEriDerivativeBatch(const CudaEriDerivativeBatch&) = delete;
  CudaEriDerivativeBatch& operator=(const CudaEriDerivativeBatch&) = delete;
  void append(const std::array<std::size_t, 4>& shells, std::span<const double> weights,
              std::span<double> gradient);
  void finish(std::span<double> gradient);
  const CudaShellDerivativeDiagnostic& diagnostic() const;
  bool batched() const;
  /** Planned batch payload; zero selects the separately bounded shell path. */
  std::size_t numeric_capacity_bytes() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace generativeqc::posthf
