#ifndef VIBEQC_XTB_CPU_DISPATCH_FEATURES_HPP
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#define VIBEQC_XTB_CPU_DISPATCH_FEATURES_HPP

#include <cstdint>
#include <string>

#include "runtime/types.hpp"

namespace vibeqc::xtb::detail {

/* CPU ISA selection is internal implementation state, frozen when a CPU
 * context is created. It is deliberately absent from the stable public ABI. */
enum class CpuIsa : std::uint8_t { kBaseline, kAvx2Fma };

/* Keep the individual architectural and OS-state conditions visible so the
 * selector can be exhaustively tested without executing an AVX instruction. */
struct CpuFeatureSnapshot {
  bool avx = false;
  bool avx2 = false;
  bool fma = false;
  bool os_xsave = false;
  bool xmm_ymm_state = false;

  [[nodiscard]] bool supports_avx2_fma() const noexcept {
    return avx && avx2 && fma && os_xsave && xmm_ymm_state;
  }
};

[[nodiscard]] const char* cpu_isa_name(CpuIsa isa) noexcept;
[[nodiscard]] bool cpu_avx2_fma_kernels_built() noexcept;
[[nodiscard]] CpuFeatureSnapshot detect_cpu_features() noexcept;

/* Resolve an explicit testable request. A null request has the same meaning as
 * `auto`; all non-null values are matched exactly and without whitespace or
 * case normalization so evidence cannot silently describe a different mode. */
vibeqc_xtb_status_t resolve_cpu_isa_request(const char* request, bool avx2_kernels_built,
                                         const CpuFeatureSnapshot& features, CpuIsa& selected,
                                         std::string& error);

/* Read VIBEQC_XTB_CPU_ISA once for one CPU context and freeze the resulting ISA.
 * CUDA callers do not invoke this function and therefore ignore the variable. */
vibeqc_xtb_status_t resolve_cpu_isa_from_environment(CpuIsa& selected, std::string& error);

}  // namespace vibeqc::xtb::detail

#endif  // VIBEQC_XTB_CPU_DISPATCH_FEATURES_HPP
