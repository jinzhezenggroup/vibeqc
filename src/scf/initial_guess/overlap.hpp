#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/types.hpp"
#include "scf/initial_guess/eigen_operation.hpp"
#include "scf/reference/linalg.hpp"
#include "scf/reference/observation.hpp"

namespace vibeqc::scf::initial_guess {

/** Preserve the full symmetric S^(-1/2) and historical strict '<1e-10' cutoff.
 * Only decomposition is substituted; no Cholesky representation or subspace
 * truncation is introduced. The provider validates its eigenframe separately. */
inline reference::Matrix symmetric_overlap(const reference::Matrix& overlap, std::size_t n,
                                           const EigenOperation& eigen) {
  if (!eigen) return reference::symmetric_orthogonalizer(overlap, n);
  reference::observation::Reason reason(reference::observation::EigenReason::overlap);
  reference::observation::Scope trace("overlap_orthogonalization", n);
  auto frame = eigen(overlap, nullptr, nullptr, n);
  if (frame.values.size() != n || frame.vectors.size() != n * n)
    throw std::runtime_error("overlap provider returned an invalid eigenframe shape");
  auto scaled = frame.vectors;
  for (std::size_t column = 0; column < n; ++column) {
    if (!std::isfinite(frame.values[column]) || frame.values[column] < 1e-10)
      throw std::runtime_error("overlap matrix is singular or severely linearly dependent");
    const double factor = 1.0 / std::sqrt(frame.values[column]);
    for (std::size_t row = 0; row < n; ++row) scaled[row * n + column] *= factor;
  }
  auto x = reference::multiply(scaled, reference::transpose(frame.vectors, n), n);
  const auto metric =
      reference::multiply(reference::transpose(x, n), reference::multiply(overlap, x, n), n);
  for (std::size_t row = 0; row < n; ++row)
    for (std::size_t column = 0; column < n; ++column) {
      const auto value = metric[row * n + column];
      if (!std::isfinite(value) || std::abs(value - (row == column ? 1.0 : 0.0)) > 1e-8)
        throw std::runtime_error("overlap orthogonalizer failed its metric identity check");
    }
  return x;
}

/** One bounded overlap cache inside an existing prepared system owner.
 * The owner fixes ordered basis/representation and backend/device; replacing
 * that owner replaces this cache. Coordinates and the actual S values are
 * checked directly, never inferred from dimensions or a fallible hash.
 * The current numerical policy is the reference FP64 symmetric S^(-1/2)
 * with its fixed 1e-10 singularity threshold. A future change to that policy
 * must invalidate the owner, as for the other prepared numerical inputs.
 * This object shares its owner's external-serialization contract.
 */
class OverlapOrthogonalizer {
 public:
  const reference::Matrix& get(const core::System& system, const reference::Matrix& overlap,
                               std::size_t n, const EigenOperation& eigen = {}) {
    if (n == 0 || n > std::numeric_limits<std::size_t>::max() / n || overlap.size() != n * n) {
      clear();
      throw std::invalid_argument("overlap cache requires a nonempty square AO matrix");
    }
    bool geometry_matches = coordinates_.size() == 3 * system.atoms.size();
    for (std::size_t atom = 0; geometry_matches && atom < system.atoms.size(); ++atom)
      for (std::size_t xyz = 0; xyz < 3; ++xyz)
        geometry_matches =
            geometry_matches && coordinates_[3 * atom + xyz] == system.atoms[atom].position[xyz];
    if (!orthogonalizer_.empty() && geometry_matches && overlap_ == overlap) {
      reference::observation::Scope hit("overlap_cache_hit", n);
      return orthogonalizer_;
    }
    reference::observation::Scope miss("overlap_cache_miss", n);
    // Drop obsolete state before allocating a replacement. A failed solve or
    // allocation leaves this item empty and cannot publish its old X as new.
    clear();
    if (!std::all_of(overlap.begin(), overlap.end(),
                     [](double value) { return std::isfinite(value); }))
      throw std::runtime_error("overlap matrix contains nonfinite values");
    auto orthogonalizer = symmetric_overlap(overlap, n, eigen);
    if (!std::all_of(orthogonalizer.begin(), orthogonalizer.end(),
                     [](double value) { return std::isfinite(value); }))
      throw std::runtime_error("overlap orthogonalization produced nonfinite values");
    reference::Matrix source = overlap;
    std::vector<double> coordinates;
    coordinates.reserve(3 * system.atoms.size());
    for (const auto& atom : system.atoms)
      coordinates.insert(coordinates.end(), atom.position.begin(), atom.position.end());
    overlap_ = std::move(source);
    coordinates_ = std::move(coordinates);
    orthogonalizer_ = std::move(orthogonalizer);
    return orthogonalizer_;
  }

  void clear() noexcept {
    reference::Matrix{}.swap(overlap_);
    reference::Matrix{}.swap(orthogonalizer_);
    std::vector<double>{}.swap(coordinates_);
  }

  /** Numeric capacity retained between calls; control objects have the
   * prepared owner's existing metadata allowance. No device copy is added. */
  std::size_t numeric_capacity_bytes() const noexcept {
    return sizeof(double) *
           (overlap_.capacity() + orthogonalizer_.capacity() + coordinates_.capacity());
  }

 private:
  reference::Matrix overlap_, orthogonalizer_;
  std::vector<double> coordinates_;
};

/** Select a prepared X or the original solve. The private diagnostic bypass
 * restores actual overlap decomposition for causal ablation; it never mutates
 * cached state. A later enabled call still checks the real S and coordinates. */
inline reference::Matrix prepare_overlap_orthogonalizer(const core::System& system,
                                                        const reference::Matrix& overlap,
                                                        std::size_t n,
                                                        OverlapOrthogonalizer* cache = nullptr,
                                                        const EigenOperation& eigen = {}) {
  const char* rebuild = std::getenv("VIBEQC_DF_REBUILD_OVERLAP");
  if (cache && !(rebuild && rebuild[0] == '1' && rebuild[1] == '\0'))
    return cache->get(system, overlap, n, eigen);
  return symmetric_overlap(overlap, n, eigen);
}

}  // namespace vibeqc::scf::initial_guess
