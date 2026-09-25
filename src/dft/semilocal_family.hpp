#pragma once

#include <cstdint>
#include <stdexcept>

namespace vibeqc::dft {

/** Native curated semilocal execution identity shared by CPU and CUDA KS.
 *
 * These stable codes are execution selectors for already-admitted native
 * implementations. They do not grant method/backend capability by themselves.
 */
enum class SemilocalFamily : std::uint32_t {
  Lda = 0,
  Pbe = 1,
  R2scan = 2,
  B3lyp = 3,
  Wb97mv = 4,
};

constexpr std::uint32_t semilocal_family_code(SemilocalFamily family) noexcept {
  return static_cast<std::uint32_t>(family);
}

constexpr const char* semilocal_family_name(SemilocalFamily family) noexcept {
  switch (family) {
    case SemilocalFamily::Pbe:
      return "PBE";
    case SemilocalFamily::R2scan:
      return "R2SCAN";
    case SemilocalFamily::B3lyp:
      return "B3LYP";
    case SemilocalFamily::Wb97mv:
      return "WB97M-V";
    default:
      return "LDA";
  }
}

constexpr std::uint32_t semilocal_family_domain_version(SemilocalFamily family) noexcept {
  return family == SemilocalFamily::Wb97mv ? 3U : family == SemilocalFamily::B3lyp ? 2U : 1U;
}

constexpr bool semilocal_family_has_cuda_ks(SemilocalFamily family) noexcept {
  return family == SemilocalFamily::Lda || family == SemilocalFamily::Pbe ||
         family == SemilocalFamily::R2scan || family == SemilocalFamily::B3lyp;
}

inline SemilocalFamily semilocal_family_from_code(std::uint32_t code) {
  switch (code) {
    case semilocal_family_code(SemilocalFamily::Lda):
      return SemilocalFamily::Lda;
    case semilocal_family_code(SemilocalFamily::Pbe):
      return SemilocalFamily::Pbe;
    case semilocal_family_code(SemilocalFamily::R2scan):
      return SemilocalFamily::R2scan;
    case semilocal_family_code(SemilocalFamily::B3lyp):
      return SemilocalFamily::B3lyp;
    case semilocal_family_code(SemilocalFamily::Wb97mv):
      return SemilocalFamily::Wb97mv;
    default:
      throw std::invalid_argument("unknown native KS semilocal family code");
  }
}

}  // namespace vibeqc::dft
