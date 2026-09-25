#include "scf/initial_guess/density.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>

#include "core/types.hpp"
#include "integrals/s_integrals.hpp"
#include "scf/reference/mean_field.hpp"
#include "scf/reference/observation.hpp"

namespace vibeqc::scf::initial_guess {

using reference::density_from_orbitals;
using reference::generalized_eigen;
using reference::index;

void mix_open_shell_frontier_orbitals(Matrix& beta_coefficients, std::size_t n,
                                      std::size_t alpha_occupied, std::size_t beta_occupied) {
  if (alpha_occupied == beta_occupied || beta_occupied == 0 || beta_occupied >= n) {
    return;
  }
  // Exact molecular symmetry can make a core-Hamiltonian UHF guess an
  // excited-state fixed point (for example, the sigma-hole state of linear
  // OH). A 45-degree orthogonal HOMO/LUMO rotation preserves electron count
  // and S-orthonormality while moving the seed outside that excited state's
  // basin. The converged orbitals, not this seed angle, define the result.
  constexpr double cosine = 0.7071067811865476;
  constexpr double sine = 0.7071067811865476;
  const std::size_t occupied_orbital = beta_occupied - 1;
  const std::size_t virtual_orbital = beta_occupied;
  for (std::size_t row = 0; row < n; ++row) {
    const double occupied_value = beta_coefficients[index(row, occupied_orbital, n)];
    const double virtual_value = beta_coefficients[index(row, virtual_orbital, n)];
    beta_coefficients[index(row, occupied_orbital, n)] =
        cosine * occupied_value + sine * virtual_value;
    beta_coefficients[index(row, virtual_orbital, n)] =
        -sine * occupied_value + cosine * virtual_value;
  }
}

std::pair<std::size_t, std::size_t> spin_occupations(const core::System& system) {
  const std::size_t electrons = static_cast<std::size_t>(system.electron_count);
  const std::size_t spin_excess = static_cast<std::size_t>(system.multiplicity - 1);
  if (spin_excess > electrons || ((electrons + spin_excess) & 1U) != 0U) {
    throw std::invalid_argument(
        "electron count and multiplicity do not define integral UHF occupations");
  }
  const std::size_t alpha = (electrons + spin_excess) / 2;
  return {alpha, electrons - alpha};
}

void normalize_spin_density(Matrix& density, const Matrix& overlap, std::size_t n,
                            std::size_t target_electrons) {
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      const double symmetric = 0.5 * (density[index(i, j, n)] + density[index(j, i, n)]);
      density[index(i, j, n)] = symmetric;
      density[index(j, i, n)] = symmetric;
    }
  }
  if (target_electrons == 0) {
    std::fill(density.begin(), density.end(), 0.0);
    return;
  }
  double electron_trace = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      electron_trace += density[index(i, j, n)] * overlap[index(j, i, n)];
    }
  }
  if (!(electron_trace > 0.0) || !std::isfinite(electron_trace)) {
    throw std::invalid_argument("initial spin density has an invalid electron trace");
  }
  const double scale = static_cast<double>(target_electrons) / electron_trace;
  for (double& value : density) value *= scale;
}

Matrix normalized_warm_density(const core::System& system, const integrals::IntegralData& ints,
                               const Matrix& input) {
  const auto n = ints.nbf;
  const auto* initial_density = &input;
  if (initial_density->size() != n * n ||
      !std::all_of(initial_density->begin(), initial_density->end(),
                   [](double value) { return std::isfinite(value); })) {
    throw std::invalid_argument("initial density does not match the finite AO matrix topology");
  }
  Matrix density = *initial_density;
  // A density from the same AO topology but a different geometry is a useful
  // guess, although its electron trace changes with the new overlap. Restore
  // symmetry and electron count before entering SCF so warm starts do not
  // introduce a geometry-dependent charge error.
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      const double symmetric = 0.5 * (density[index(i, j, n)] + density[index(j, i, n)]);
      density[index(i, j, n)] = symmetric;
      density[index(j, i, n)] = symmetric;
    }
  }
  double electron_trace = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      electron_trace += density[index(i, j, n)] * ints.overlap[index(j, i, n)];
    }
  }
  if (!(electron_trace > 0.0) || !std::isfinite(electron_trace)) {
    throw std::invalid_argument("initial density has an invalid electron trace");
  }
  const double trace_scale = static_cast<double>(system.electron_count) / electron_trace;
  for (double& value : density) {
    value *= trace_scale;
    if (!std::isfinite(value)) {
      throw std::invalid_argument("initial density normalization produced a non-finite AO matrix");
    }
  }
  return density;
}

std::pair<Matrix, Matrix> normalized_warm_uhf_density(const integrals::IntegralData& ints,
                                                      std::size_t alpha_occupied,
                                                      std::size_t beta_occupied,
                                                      const Matrix& input) {
  const auto n = ints.nbf;
  const auto matrix_size = n * n;
  const auto* initial_density = &input;
  if (initial_density->size() != 2 * matrix_size ||
      !std::all_of(initial_density->begin(), initial_density->end(),
                   [](double value) { return std::isfinite(value); })) {
    throw std::invalid_argument(
        "initial UHF density must contain finite alpha and beta AO matrices");
  }
  Matrix alpha(initial_density->begin(), initial_density->begin() + matrix_size);
  Matrix beta(initial_density->begin() + matrix_size, initial_density->end());
  normalize_spin_density(alpha, ints.overlap, n, alpha_occupied);
  normalize_spin_density(beta, ints.overlap, n, beta_occupied);
  return {std::move(alpha), std::move(beta)};
}

Matrix charge_guided_lowdin_density(const core::System& system, const integrals::IntegralData& ints,
                                    const Matrix& orthogonalizer, const Matrix& input,
                                    std::span<const double> atomic_charges) {
  const std::size_t n = ints.nbf;
  if (n == 0 || ints.overlap.size() != n * n || orthogonalizer.size() != n * n ||
      input.size() != n * n) {
    throw std::invalid_argument("charge-guided seed has inconsistent AO matrix dimensions");
  }
  if (atomic_charges.size() != system.atoms.size() ||
      !std::all_of(atomic_charges.begin(), atomic_charges.end(),
                   [](double value) { return std::isfinite(value); })) {
    throw std::invalid_argument("charge-guided seed requires one finite charge per atom");
  }
  if (system.electron_count < 0) {
    throw std::invalid_argument("charge-guided seed requires a nonnegative electron count");
  }

  double charge_sum = 0.0;
  for (double charge : atomic_charges) charge_sum += charge;
  const double charge_tolerance =
      1.0e-8 * std::max(1.0, static_cast<double>(atomic_charges.size()));
  if (!std::isfinite(charge_sum) ||
      std::abs(charge_sum - static_cast<double>(system.charge)) > charge_tolerance) {
    throw std::invalid_argument("atomic charges do not reproduce the molecular charge");
  }

  std::vector<std::size_t> ao_atoms;
  ao_atoms.reserve(n);
  for (const core::Shell& shell : system.shells) {
    if (shell.atom_index >= system.atoms.size()) {
      throw std::invalid_argument("charge-guided seed shell references an invalid atom");
    }
    const std::uint64_t angular = shell.angular_momentum;
    const std::uint64_t count64 = system.basis_representation == VIBEQC_BASIS_SPHERICAL
                                      ? 2u * angular + 1u
                                      : (angular + 1u) * (angular + 2u) / 2u;
    const std::size_t count =
        count64 > std::numeric_limits<std::size_t>::max() ? 0u : static_cast<std::size_t>(count64);
    if (count == 0 || ao_atoms.size() > n || count > n - ao_atoms.size()) {
      throw std::invalid_argument("charge-guided seed AO ownership exceeds the basis");
    }
    ao_atoms.insert(ao_atoms.end(), count, static_cast<std::size_t>(shell.atom_index));
  }
  if (ao_atoms.size() != n) {
    throw std::invalid_argument("charge-guided seed AO ownership does not match the basis");
  }

  Matrix normalized = normalized_warm_density(system, ints, input);
  Matrix overlap_square_root = reference::multiply(ints.overlap, orthogonalizer, n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      const double symmetric =
          0.5 * (overlap_square_root[index(i, j, n)] + overlap_square_root[index(j, i, n)]);
      overlap_square_root[index(i, j, n)] = symmetric;
      overlap_square_root[index(j, i, n)] = symmetric;
    }
  }
  Matrix orthogonal_density = reference::multiply(
      reference::multiply(overlap_square_root, normalized, n), overlap_square_root, n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      const double symmetric =
          0.5 * (orthogonal_density[index(i, j, n)] + orthogonal_density[index(j, i, n)]);
      orthogonal_density[index(i, j, n)] = symmetric;
      orthogonal_density[index(j, i, n)] = symmetric;
    }
  }

  std::vector<double> current(system.atoms.size(), 0.0);
  for (std::size_t ao = 0; ao < n; ++ao) {
    current[ao_atoms[ao]] += orthogonal_density[index(ao, ao, n)];
  }
  std::vector<double> scale(system.atoms.size(), 0.0);
  constexpr double population_tolerance = 1.0e-12;
  double target_sum = 0.0;
  for (std::size_t atom = 0; atom < system.atoms.size(); ++atom) {
    double target = static_cast<double>(system.atoms[atom].ionic_charge()) - atomic_charges[atom];
    if (target < -population_tolerance || current[atom] < -population_tolerance) {
      throw std::invalid_argument("charge-guided seed produced a negative atomic population");
    }
    if (target < 0.0) target = 0.0;
    if (current[atom] < 0.0) current[atom] = 0.0;
    target_sum += target;
    if (target <= population_tolerance) {
      scale[atom] = 0.0;
    } else {
      if (current[atom] <= population_tolerance) {
        throw std::invalid_argument("charge-guided seed cannot populate an empty atomic AO block");
      }
      const double ratio = target / current[atom];
      if (!(ratio > 0.0) || !std::isfinite(ratio)) {
        throw std::invalid_argument("charge-guided seed has an invalid atomic population ratio");
      }
      scale[atom] = std::sqrt(ratio);
    }
  }
  if (std::abs(target_sum - static_cast<double>(system.electron_count)) > charge_tolerance) {
    throw std::invalid_argument("charge-guided seed populations do not match the electron count");
  }

  for (std::size_t i = 0; i < n; ++i) {
    const double row_scale = scale[ao_atoms[i]];
    for (std::size_t j = 0; j < n; ++j) {
      orthogonal_density[index(i, j, n)] *= row_scale * scale[ao_atoms[j]];
    }
  }

  Matrix density = reference::multiply(reference::multiply(orthogonalizer, orthogonal_density, n),
                                       orthogonalizer, n);
  normalize_spin_density(density, ints.overlap, n, static_cast<std::size_t>(system.electron_count));
  if (!std::all_of(density.begin(), density.end(),
                   [](double value) { return std::isfinite(value); })) {
    throw std::invalid_argument("charge-guided seed produced a non-finite AO density");
  }
  return density;
}

OccupiedProjectionResult project_occupied_density(
    const integrals::IntegralData& target, const Matrix& target_orthogonalizer,
    std::span<const double> source_overlap, std::span<const double> cross_overlap,
    std::size_t source_nbf, std::span<const double> source_coefficients, std::size_t occupied,
    double occupation, double maximum_residual) {
  const std::size_t target_nbf = target.nbf;
  const auto finite = [](std::span<const double> values) {
    return std::all_of(values.begin(), values.end(),
                       [](double value) { return std::isfinite(value); });
  };
  if (target_nbf == 0 || source_nbf == 0 ||
      target_nbf > std::numeric_limits<std::size_t>::max() / target_nbf ||
      source_nbf > std::numeric_limits<std::size_t>::max() / source_nbf ||
      target_nbf > std::numeric_limits<std::size_t>::max() / source_nbf ||
      target.overlap.size() != target_nbf * target_nbf ||
      target_orthogonalizer.size() != target_nbf * target_nbf ||
      source_overlap.size() != source_nbf * source_nbf ||
      cross_overlap.size() != target_nbf * source_nbf ||
      source_coefficients.size() != source_nbf * source_nbf || occupied > source_nbf ||
      occupied > target_nbf || !(occupation > 0.0) || !std::isfinite(occupation) ||
      maximum_residual < 0.0 || !std::isfinite(maximum_residual) || !finite(target.overlap) ||
      !finite(target_orthogonalizer) || !finite(source_overlap) || !finite(cross_overlap) ||
      !finite(source_coefficients)) {
    throw std::invalid_argument("occupied projection has inconsistent or non-finite dimensions");
  }

  OccupiedProjectionResult result;
  result.density.assign(target_nbf * target_nbf, 0.0);
  if (occupied == 0) {
    result.minimum_projected_norm = 1.0;
    return result;
  }

  // Qualify the supplied source occupied columns before any target projection.
  std::vector<double> source_metric_times_c(source_nbf * occupied, 0.0);
  for (std::size_t mu = 0; mu < source_nbf; ++mu) {
    for (std::size_t orbital = 0; orbital < occupied; ++orbital) {
      double value = 0.0;
      for (std::size_t nu = 0; nu < source_nbf; ++nu)
        value +=
            source_overlap[mu * source_nbf + nu] * source_coefficients[nu * source_nbf + orbital];
      source_metric_times_c[mu * occupied + orbital] = value;
    }
  }
  double source_orthogonality_error = 0.0;
  for (std::size_t first = 0; first < occupied; ++first) {
    for (std::size_t second = 0; second < occupied; ++second) {
      double value = 0.0;
      for (std::size_t mu = 0; mu < source_nbf; ++mu)
        value += source_coefficients[mu * source_nbf + first] *
                 source_metric_times_c[mu * occupied + second];
      const double expected = first == second ? 1.0 : 0.0;
      if (!std::isfinite(value))
        throw std::invalid_argument("source occupied metric contraction is non-finite");
      source_orthogonality_error = std::max(source_orthogonality_error, std::abs(value - expected));
    }
  }
  result.source_metric_orthogonality_error = source_orthogonality_error;
  if (!std::isfinite(source_orthogonality_error) || source_orthogonality_error > 1.0e-7)
    throw std::invalid_argument("source occupied orbitals are not orthonormal in their metric");

  // B = S_ts C_s(occ), then Y = S_tt^(-1/2) B.  Y^T Y is the
  // projected-subspace norm before target-metric reorthonormalization.
  std::vector<double> cross_times_c(target_nbf * occupied, 0.0);
  for (std::size_t mu = 0; mu < target_nbf; ++mu) {
    for (std::size_t orbital = 0; orbital < occupied; ++orbital) {
      double value = 0.0;
      for (std::size_t nu = 0; nu < source_nbf; ++nu)
        value +=
            cross_overlap[mu * source_nbf + nu] * source_coefficients[nu * source_nbf + orbital];
      cross_times_c[mu * occupied + orbital] = value;
    }
  }
  std::vector<double> orthogonal_projection(target_nbf * occupied, 0.0);
  for (std::size_t mu = 0; mu < target_nbf; ++mu) {
    for (std::size_t orbital = 0; orbital < occupied; ++orbital) {
      double value = 0.0;
      for (std::size_t nu = 0; nu < target_nbf; ++nu)
        value += target_orthogonalizer[index(mu, nu, target_nbf)] *
                 cross_times_c[nu * occupied + orbital];
      orthogonal_projection[mu * occupied + orbital] = value;
    }
  }

  Matrix gram(occupied * occupied, 0.0);
  for (std::size_t first = 0; first < occupied; ++first) {
    for (std::size_t second = 0; second < occupied; ++second) {
      double value = 0.0;
      for (std::size_t mu = 0; mu < target_nbf; ++mu)
        value += orthogonal_projection[mu * occupied + first] *
                 orthogonal_projection[mu * occupied + second];
      gram[index(first, second, occupied)] = value;
    }
  }
  auto spectral = reference::symmetric_eigen(std::move(gram), occupied);
  const double minimum_norm = spectral.values.front();
  const double maximum_norm = spectral.values.back();
  if (!(minimum_norm > 0.0) || !std::isfinite(minimum_norm) || !std::isfinite(maximum_norm) ||
      minimum_norm <= 1.0e-10 * maximum_norm || maximum_norm > 1.0 + 1.0e-7) {
    throw std::invalid_argument("occupied projection lost rank or exceeds the source norm");
  }
  result.minimum_projected_norm = minimum_norm;
  // Roundoff in an identity Gram can otherwise become an artificial O(sqrt(epsilon))
  // residual. Preserve every larger projected-norm loss.
  const double norm_loss = std::max(0.0, 1.0 - minimum_norm);
  result.projection_residual =
      norm_loss <= 8.0 * std::numeric_limits<double>::epsilon() ? 0.0 : std::sqrt(norm_loss);
  if (result.projection_residual > maximum_residual)
    throw std::invalid_argument("occupied projection loses too much source occupied norm");

  Matrix inverse_sqrt_gram(occupied * occupied, 0.0);
  for (std::size_t row = 0; row < occupied; ++row) {
    for (std::size_t column = 0; column < occupied; ++column) {
      double value = 0.0;
      for (std::size_t eigen = 0; eigen < occupied; ++eigen)
        value += spectral.vectors[index(row, eigen, occupied)] *
                 spectral.vectors[index(column, eigen, occupied)] /
                 std::sqrt(spectral.values[eigen]);
      inverse_sqrt_gram[index(row, column, occupied)] = value;
    }
  }

  // P = S_tt^(-1) B = X (X B), then C_t = P (P^T S P)^(-1/2).
  std::vector<double> projected(target_nbf * occupied, 0.0);
  for (std::size_t mu = 0; mu < target_nbf; ++mu) {
    for (std::size_t orbital = 0; orbital < occupied; ++orbital) {
      double value = 0.0;
      for (std::size_t nu = 0; nu < target_nbf; ++nu)
        value += target_orthogonalizer[index(mu, nu, target_nbf)] *
                 orthogonal_projection[nu * occupied + orbital];
      projected[mu * occupied + orbital] = value;
    }
  }
  std::vector<double> target_coefficients(target_nbf * occupied, 0.0);
  for (std::size_t mu = 0; mu < target_nbf; ++mu) {
    for (std::size_t orbital = 0; orbital < occupied; ++orbital) {
      double value = 0.0;
      for (std::size_t source = 0; source < occupied; ++source)
        value +=
            projected[mu * occupied + source] * inverse_sqrt_gram[index(source, orbital, occupied)];
      target_coefficients[mu * occupied + orbital] = value;
    }
  }

  std::vector<double> target_metric_times_c(target_nbf * occupied, 0.0);
  for (std::size_t mu = 0; mu < target_nbf; ++mu) {
    for (std::size_t orbital = 0; orbital < occupied; ++orbital) {
      double value = 0.0;
      for (std::size_t nu = 0; nu < target_nbf; ++nu)
        value += target.overlap[index(mu, nu, target_nbf)] *
                 target_coefficients[nu * occupied + orbital];
      target_metric_times_c[mu * occupied + orbital] = value;
    }
  }
  double target_orthogonality_error = 0.0;
  for (std::size_t first = 0; first < occupied; ++first) {
    for (std::size_t second = 0; second < occupied; ++second) {
      double value = 0.0;
      for (std::size_t mu = 0; mu < target_nbf; ++mu)
        value += target_coefficients[mu * occupied + first] *
                 target_metric_times_c[mu * occupied + second];
      const double expected = first == second ? 1.0 : 0.0;
      if (!std::isfinite(value))
        throw std::invalid_argument("target occupied metric contraction is non-finite");
      target_orthogonality_error = std::max(target_orthogonality_error, std::abs(value - expected));
    }
  }
  result.target_metric_orthogonality_error = target_orthogonality_error;
  if (!std::isfinite(target_orthogonality_error) || target_orthogonality_error > 1.0e-7)
    throw std::invalid_argument("projected occupied orbitals failed target-metric orthogonality");

  result.coefficients = target_coefficients;
  for (std::size_t mu = 0; mu < target_nbf; ++mu) {
    for (std::size_t nu = 0; nu < target_nbf; ++nu) {
      double value = 0.0;
      for (std::size_t orbital = 0; orbital < occupied; ++orbital)
        value += target_coefficients[mu * occupied + orbital] *
                 target_coefficients[nu * occupied + orbital];
      result.density[index(mu, nu, target_nbf)] = occupation * value;
    }
  }
  if (!finite(result.density))
    throw std::invalid_argument("occupied projection produced a non-finite target density");

  double electron_trace = 0.0;
  for (std::size_t mu = 0; mu < target_nbf; ++mu)
    for (std::size_t nu = 0; nu < target_nbf; ++nu)
      electron_trace +=
          result.density[index(mu, nu, target_nbf)] * target.overlap[index(nu, mu, target_nbf)];
  const double expected_trace = occupation * static_cast<double>(occupied);
  if (!std::isfinite(electron_trace) ||
      std::abs(electron_trace - expected_trace) > 1.0e-7 * std::max(1.0, expected_trace))
    throw std::invalid_argument("occupied projection failed the target electron trace");

  return result;
}

OccupiedCompletionResult complete_occupied_density(const integrals::IntegralData& target,
                                                   std::span<const double> seeded_coefficients,
                                                   std::size_t seeded_occupied,
                                                   std::span<const double> reference_coefficients,
                                                   std::size_t target_occupied, double occupation,
                                                   double minimum_complement_norm) {
  const std::size_t n = target.nbf;
  const auto finite = [](std::span<const double> values) {
    return std::all_of(values.begin(), values.end(),
                       [](double value) { return std::isfinite(value); });
  };
  if (n == 0 || n > std::numeric_limits<std::size_t>::max() / n ||
      seeded_occupied > target_occupied || target_occupied > n ||
      (seeded_occupied != 0 && (n > std::numeric_limits<std::size_t>::max() / seeded_occupied ||
                                seeded_coefficients.size() != n * seeded_occupied)) ||
      reference_coefficients.size() != n * n || target.overlap.size() != n * n ||
      !(occupation > 0.0) || !std::isfinite(occupation) || !(minimum_complement_norm > 0.0) ||
      !std::isfinite(minimum_complement_norm) || !finite(target.overlap) ||
      !finite(seeded_coefficients) || !finite(reference_coefficients)) {
    throw std::invalid_argument("occupied completion has inconsistent or non-finite inputs");
  }

  OccupiedCompletionResult result;
  result.coefficients.assign(n * target_occupied, 0.0);
  Matrix metric_coefficients(n * target_occupied, 0.0);
  for (std::size_t mu = 0; mu < n; ++mu) {
    for (std::size_t orbital = 0; orbital < seeded_occupied; ++orbital)
      result.coefficients[mu * target_occupied + orbital] =
          seeded_coefficients[mu * seeded_occupied + orbital];
  }
  for (std::size_t mu = 0; mu < n; ++mu) {
    for (std::size_t orbital = 0; orbital < seeded_occupied; ++orbital) {
      double value = 0.0;
      for (std::size_t nu = 0; nu < n; ++nu)
        value +=
            target.overlap[index(mu, nu, n)] * result.coefficients[nu * target_occupied + orbital];
      metric_coefficients[mu * target_occupied + orbital] = value;
    }
  }

  auto metric_column_dot = [&](std::size_t first, std::size_t second) {
    double value = 0.0;
    for (std::size_t mu = 0; mu < n; ++mu)
      value += result.coefficients[mu * target_occupied + first] *
               metric_coefficients[mu * target_occupied + second];
    return value;
  };

  double seeded_error = 0.0;
  for (std::size_t first = 0; first < seeded_occupied; ++first) {
    for (std::size_t second = 0; second < seeded_occupied; ++second) {
      const double expected = first == second ? 1.0 : 0.0;
      seeded_error = std::max(seeded_error, std::abs(metric_column_dot(first, second) - expected));
    }
  }
  if (!std::isfinite(seeded_error) || seeded_error > 1.0e-7)
    throw std::invalid_argument("seeded occupied subspace is not target-metric orthonormal");

  std::size_t accepted = seeded_occupied;
  double minimum_added_norm = std::numeric_limits<double>::infinity();
  Matrix residual(n);
  Matrix metric_residual(n);
  for (std::size_t reference = 0; reference < n && accepted < target_occupied; ++reference) {
    for (std::size_t mu = 0; mu < n; ++mu)
      residual[mu] = reference_coefficients[index(mu, reference, n)];

    // Two-pass metric Gram-Schmidt. Caching S*C keeps every projection O(n)
    // and the one S*residual update per candidate O(n^2), so completing up to
    // O(n) target orbitals remains O(n^3), not O(n^4).
    for (unsigned pass = 0; pass < 2u; ++pass) {
      for (std::size_t previous = 0; previous < accepted; ++previous) {
        double projection = 0.0;
        for (std::size_t mu = 0; mu < n; ++mu)
          projection += metric_coefficients[mu * target_occupied + previous] * residual[mu];
        for (std::size_t mu = 0; mu < n; ++mu)
          residual[mu] -= projection * result.coefficients[mu * target_occupied + previous];
      }
    }

    for (std::size_t mu = 0; mu < n; ++mu) {
      double value = 0.0;
      for (std::size_t nu = 0; nu < n; ++nu)
        value += target.overlap[index(mu, nu, n)] * residual[nu];
      metric_residual[mu] = value;
    }
    double norm2 = 0.0;
    for (std::size_t mu = 0; mu < n; ++mu) norm2 += residual[mu] * metric_residual[mu];
    if (!std::isfinite(norm2) || norm2 < -1.0e-12)
      throw std::invalid_argument("target reference completion produced an invalid metric norm");
    const double norm = std::sqrt(std::max(0.0, norm2));
    if (norm < minimum_complement_norm) continue;

    minimum_added_norm = std::min(minimum_added_norm, norm);
    const double inverse_norm = 1.0 / norm;
    for (std::size_t mu = 0; mu < n; ++mu) {
      result.coefficients[mu * target_occupied + accepted] = residual[mu] * inverse_norm;
      metric_coefficients[mu * target_occupied + accepted] = metric_residual[mu] * inverse_norm;
    }
    ++accepted;
  }
  if (accepted != target_occupied)
    throw std::invalid_argument("target reference orbitals cannot complete the occupied subspace");

  result.added_orbitals = target_occupied - seeded_occupied;
  result.minimum_added_norm = result.added_orbitals == 0 ? 1.0 : minimum_added_norm;

  double orthogonality_error = 0.0;
  for (std::size_t first = 0; first < target_occupied; ++first) {
    for (std::size_t second = 0; second < target_occupied; ++second) {
      const double expected = first == second ? 1.0 : 0.0;
      orthogonality_error =
          std::max(orthogonality_error, std::abs(metric_column_dot(first, second) - expected));
    }
  }
  result.metric_orthogonality_error = orthogonality_error;
  if (!std::isfinite(orthogonality_error) || orthogonality_error > 1.0e-7)
    throw std::invalid_argument("completed occupied subspace is not target-metric orthonormal");

  result.density.assign(n * n, 0.0);
  for (std::size_t mu = 0; mu < n; ++mu) {
    for (std::size_t nu = 0; nu < n; ++nu) {
      double value = 0.0;
      for (std::size_t orbital = 0; orbital < target_occupied; ++orbital)
        value += result.coefficients[mu * target_occupied + orbital] *
                 result.coefficients[nu * target_occupied + orbital];
      result.density[index(mu, nu, n)] = occupation * value;
    }
  }

  if (!finite(result.density))
    throw std::invalid_argument("occupied completion produced a non-finite target density");

  double trace = 0.0;
  for (std::size_t orbital = 0; orbital < target_occupied; ++orbital)
    trace += metric_column_dot(orbital, orbital);
  trace *= occupation;
  const double expected_trace = occupation * static_cast<double>(target_occupied);
  if (!std::isfinite(trace) ||
      std::abs(trace - expected_trace) > 1.0e-7 * std::max(1.0, expected_trace))
    throw std::invalid_argument("completed occupied density failed the target electron trace");

  return result;
}

std::pair<Matrix, Matrix> prepare_initial_uhf_density(
    const integrals::IntegralData& ints, const Matrix& orthogonalizer, std::size_t alpha_occupied,
    std::size_t beta_occupied, const std::vector<double>* initial_density,
    std::optional<EigenResult>& alpha_orbitals, std::optional<EigenResult>& beta_orbitals,
    InitialOrbitalRequest request, const EigenOperation& eigen) {
  const std::size_t n = ints.nbf;
  scf::reference::observation::Reason reason(scf::reference::observation::EigenReason::core_guess);
  scf::reference::observation::Scope trace("initial_density", n);
  alpha_orbitals.reset();
  beta_orbitals.reset();
  const auto core_frame = [&] {
    return eigen ? eigen(ints.hcore, &ints.overlap, &orthogonalizer, n)
                 : generalized_eigen(ints.hcore, orthogonalizer, n);
  };
  if (initial_density == nullptr) {
    alpha_orbitals = core_frame();
    beta_orbitals = alpha_orbitals;
    mix_open_shell_frontier_orbitals(beta_orbitals->vectors, n, alpha_occupied, beta_occupied);
    return {
        density_from_orbitals(alpha_orbitals->vectors, n, alpha_occupied, 1.0),
        density_from_orbitals(beta_orbitals->vectors, n, beta_occupied, 1.0),
    };
  }
  auto [alpha, beta] =
      normalized_warm_uhf_density(ints, alpha_occupied, beta_occupied, *initial_density);
  if (request == InitialOrbitalRequest::RequireCoreFrame) {
    // A requested warm frame follows the historical unperturbed convention.
    alpha_orbitals = core_frame();
    beta_orbitals = alpha_orbitals;
  }
  return {std::move(alpha), std::move(beta)};
}

Matrix prepare_initial_density(const core::System& system, const integrals::IntegralData& ints,
                               const Matrix& orthogonalizer, std::size_t occupied,
                               const std::vector<double>* initial_density,
                               std::optional<EigenResult>& orbitals, InitialOrbitalRequest request,
                               const EigenOperation& eigen,
                               const RestrictedInitialDensityProvider& provider) {
  const std::size_t n = ints.nbf;
  scf::reference::observation::Reason reason(scf::reference::observation::EigenReason::core_guess);
  scf::reference::observation::Scope trace("initial_density", n);
  orbitals.reset();
  const auto core_frame = [&] {
    return eigen ? eigen(ints.hcore, &ints.overlap, &orthogonalizer, n)
                 : generalized_eigen(ints.hcore, orthogonalizer, n);
  };
  if (initial_density == nullptr) {
    orbitals = core_frame();
    Matrix core_density = density_from_orbitals(orbitals->vectors, n, occupied);
    if (!provider) return core_density;
    try {
      auto candidate = provider({system, ints, orthogonalizer, core_density, occupied});
      if (!candidate) return core_density;
      Matrix density = normalized_warm_density(system, ints, *candidate);
      // An accepted matrix-only proposal is not represented by the core
      // orbitals. Keep that frame only for consumers that explicitly request it.
      if (request == InitialOrbitalRequest::ColdDensityOnly) orbitals.reset();
      return density;
    } catch (const std::bad_alloc&) {
      throw;
    } catch (const std::exception&) {
      // Initial-guess accelerators are optional. A rejected/failed proposal
      // must preserve the canonical core start and its already-built frame.
      return core_density;
    }
  }
  Matrix density = normalized_warm_density(system, ints, *initial_density);
  if (request == InitialOrbitalRequest::RequireCoreFrame) orbitals = core_frame();
  return density;
}

}  // namespace vibeqc::scf::initial_guess
