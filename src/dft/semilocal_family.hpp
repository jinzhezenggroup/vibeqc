#pragma once

#include <array>
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

struct SemilocalFamilyMetadata {
  SemilocalFamily family;
  const char* name;
  const char* scf_domain;
  std::uint32_t domain_version;
  bool cuda_ks;
  std::array<const char*, 4> component_ids;
  std::array<double, 4> component_coefficients;
  std::uint32_t component_count;
  double range_omega;
  bool component_coefficients_are_native_scales;
};

inline constexpr const char* kSemilocalScfDomain = "semilocal-scaled-v1/pbe-spin-c2-1e-18";
inline constexpr const char* kB3lypScfDomain = "b3lyp-vwn-rpa-tail-v1/density-vacuum-1e-18";
inline constexpr const char* kWb97mvScfDomain = "libxc-7.0/work-mgga-v1/smooth-lr-a1.35-order16";

inline constexpr std::array<SemilocalFamilyMetadata, 5> kSemilocalFamilyMetadata{{
    {SemilocalFamily::Lda,
     "LDA",
     kSemilocalScfDomain,
     1U,
     true,
     {"LDA_X", "LDA_C_PW", nullptr, nullptr},
     {1.0, 1.0, 0.0, 0.0},
     2U,
     0.0,
     false},
    {SemilocalFamily::Pbe,
     "PBE",
     kSemilocalScfDomain,
     1U,
     true,
     {"GGA_X_PBE", "GGA_C_PBE", nullptr, nullptr},
     {1.0, 1.0, 0.0, 0.0},
     2U,
     0.0,
     true},
    {SemilocalFamily::R2scan,
     "R2SCAN",
     kSemilocalScfDomain,
     1U,
     true,
     {"MGGA_X_R2SCAN", "MGGA_C_R2SCAN", nullptr, nullptr},
     {1.0, 1.0, 0.0, 0.0},
     2U,
     0.0,
     false},
    {SemilocalFamily::B3lyp,
     "B3LYP",
     kB3lypScfDomain,
     2U,
     true,
     {"LDA_X", "GGA_X_B88", "LDA_C_VWN_RPA", "GGA_C_LYP"},
     {0.08, 0.72, 0.19, 0.81},
     4U,
     0.0,
     false},
    {SemilocalFamily::Wb97mv,
     "WB97M-V",
     kWb97mvScfDomain,
     3U,
     true,
     {"MGGA_X_WB97M_V", "MGGA_C_WB97M_V", nullptr, nullptr},
     {1.0, 1.0, 0.0, 0.0},
     2U,
     0.3,
     false},
}};

constexpr std::uint32_t semilocal_family_code(SemilocalFamily family) noexcept {
  return static_cast<std::uint32_t>(family);
}

constexpr const SemilocalFamilyMetadata* semilocal_family_metadata_from_code(
    std::uint32_t code) noexcept {
  if (code >= kSemilocalFamilyMetadata.size()) return nullptr;
  const auto& metadata = kSemilocalFamilyMetadata[code];
  return semilocal_family_code(metadata.family) == code ? &metadata : nullptr;
}

constexpr const SemilocalFamilyMetadata& semilocal_family_metadata(
    SemilocalFamily family) noexcept {
  return kSemilocalFamilyMetadata[semilocal_family_code(family)];
}

constexpr const char* semilocal_family_name(SemilocalFamily family) noexcept {
  return semilocal_family_metadata(family).name;
}

constexpr const char* semilocal_family_scf_domain(SemilocalFamily family) noexcept {
  return semilocal_family_metadata(family).scf_domain;
}

constexpr std::uint32_t semilocal_family_domain_version(SemilocalFamily family) noexcept {
  return semilocal_family_metadata(family).domain_version;
}

constexpr bool semilocal_family_has_cuda_ks(SemilocalFamily family) noexcept {
  return semilocal_family_metadata(family).cuda_ks;
}

inline SemilocalFamily semilocal_family_from_code(std::uint32_t code) {
  if (const auto* metadata = semilocal_family_metadata_from_code(code)) return metadata->family;
  throw std::invalid_argument("unknown native KS semilocal family code");
}

static_assert(semilocal_family_code(SemilocalFamily::Lda) == 0U);
static_assert(semilocal_family_code(SemilocalFamily::Wb97mv) + 1U ==
              kSemilocalFamilyMetadata.size());

}  // namespace vibeqc::dft
