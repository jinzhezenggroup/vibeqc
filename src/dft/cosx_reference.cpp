#include "dft/cosx_reference.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "dft/ao_grid.hpp"
#include "integrals/s_integrals.hpp"
#include "molecule/basis.hpp"

namespace vibeqc::dft {
namespace {

std::size_t checked_product(std::size_t a, std::size_t b) {
  if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
    throw std::overflow_error("COSX reference extent overflows size_t");
  }
  return a * b;
}

void validate_spec(const CosxReferenceSpec& spec) {
  if (spec.version != 1 || !spec.symmetrize || spec.overlap_fitting || spec.screening) {
    throw std::invalid_argument(
        "COSX reference v1 requires explicit symmetrization with no fitting or screening");
  }
}

}  // namespace

CosxReferenceResult build_cosx_reference(const core::System& system,
                                         std::span<const double> points_xyz,
                                         std::span<const double> weights,
                                         std::span<const double> density,
                                         CosxDensityConvention convention, CosxReferenceSpec spec) {
  validate_spec(spec);
  if (points_xyz.size() % 3 != 0 || points_xyz.empty()) {
    throw std::invalid_argument("COSX reference points must be nonempty xyz triples");
  }
  const std::size_t npoint = points_xyz.size() / 3;
  if (weights.size() != npoint) {
    throw std::invalid_argument("COSX reference weight count does not match the point count");
  }
  const std::size_t nbf = molecule::ao_count(system);
  const std::size_t matrix_size = checked_product(nbf, nbf);
  if (nbf == 0 || density.size() != matrix_size) {
    throw std::invalid_argument("COSX reference density does not match the AO basis");
  }
  if (convention != CosxDensityConvention::spin_resolved &&
      convention != CosxDensityConvention::rhf_spin_summed) {
    throw std::invalid_argument("unknown COSX density convention");
  }
  for (double value : density) {
    if (!std::isfinite(value)) throw std::invalid_argument("nonfinite COSX density");
  }
  for (double value : weights) {
    if (!std::isfinite(value)) throw std::invalid_argument("nonfinite COSX quadrature weight");
  }

  const AoBasis basis(system);
  if (basis.nao != nbf) throw std::logic_error("COSX AO basis dimension mismatch");
  std::vector<double> ao(checked_product(npoint, nbf));
  basis.evaluate(points_xyz.data(), npoint, 0, 0, nbf, ao.data(), ao.size());

  const auto esp = integrals::build_esp_integrals(system, points_xyz);
  if (esp.nbf != nbf || esp.npoint != npoint ||
      esp.values.size() != checked_product(npoint, matrix_size)) {
    throw std::logic_error("COSX ESP reference dimensions are inconsistent");
  }

  CosxReferenceResult result;
  result.nbf = nbf;
  result.npoint = npoint;
  result.spec = spec;
  result.convention = convention;
  result.raw_exchange.assign(matrix_size, 0.0);
  result.exchange.resize(matrix_size);

  std::vector<double> projected(nbf), potential(nbf);
  for (std::size_t point = 0; point < npoint; ++point) {
    const double* phi = ao.data() + point * nbf;
    const double* esp_matrix = esp.values.data() + point * matrix_size;

    // projected[l] = sum_k phi[k] D[k,l].
    std::fill(projected.begin(), projected.end(), 0.0);
    for (std::size_t k = 0; k < nbf; ++k) {
      for (std::size_t l = 0; l < nbf; ++l) {
        projected[l] += phi[k] * density[k * nbf + l];
      }
    }
    // potential[j] = sum_l <j|1/r_g|l> projected[l].
    for (std::size_t j = 0; j < nbf; ++j) {
      double value = 0.0;
      for (std::size_t l = 0; l < nbf; ++l) {
        value += esp_matrix[j * nbf + l] * projected[l];
      }
      potential[j] = value;
    }
    const double weight = weights[point];
    for (std::size_t i = 0; i < nbf; ++i) {
      for (std::size_t j = 0; j < nbf; ++j) {
        result.raw_exchange[i * nbf + j] += weight * phi[i] * potential[j];
      }
    }
  }

  for (std::size_t i = 0; i < nbf; ++i) {
    for (std::size_t j = 0; j < nbf; ++j) {
      result.exchange[i * nbf + j] =
          0.5 * (result.raw_exchange[i * nbf + j] + result.raw_exchange[j * nbf + i]);
    }
  }

  double contraction = 0.0;
  for (std::size_t element = 0; element < matrix_size; ++element) {
    contraction += density[element] * result.exchange[element];
  }
  const double factor = convention == CosxDensityConvention::rhf_spin_summed ? -0.25 : -0.5;
  result.exchange_energy = factor * contraction;
  if (!std::isfinite(result.exchange_energy)) {
    throw std::runtime_error("nonfinite COSX reference exchange energy");
  }
  return result;
}

CosxPointDerivativeResult build_cosx_point_derivative_reference(
    const core::System& system, std::span<const double> points_xyz, std::span<const double> weights,
    std::span<const double> density, CosxDensityConvention convention, CosxReferenceSpec spec) {
  CosxPointDerivativeResult result;
  result.value = build_cosx_reference(system, points_xyz, weights, density, convention, spec);
  const std::size_t nbf = result.value.nbf;
  const std::size_t npoint = result.value.npoint;
  const std::size_t matrix_size = checked_product(nbf, nbf);

  const AoBasis basis(system);
  const std::size_t ao_jet_elements = checked_product(checked_product(4, npoint), nbf);
  std::vector<double> ao_jets(ao_jet_elements);
  basis.evaluate(points_xyz.data(), npoint, 1, 0, nbf, ao_jets.data(), ao_jets.size());

  const auto esp = integrals::build_esp_integrals_with_probe_derivatives(system, points_xyz);
  if (esp.nbf != nbf || esp.npoint != npoint ||
      esp.values.size() != checked_product(npoint, matrix_size) ||
      esp.probe_derivative.size() != checked_product(checked_product(3, npoint), matrix_size)) {
    throw std::logic_error("COSX ESP derivative reference dimensions are inconsistent");
  }

  result.point_gradient.resize(checked_product(3, npoint));
  std::vector<double> projected(nbf), potential(nbf), projected_derivative(nbf),
      potential_derivative(nbf);
  const double energy_factor = convention == CosxDensityConvention::rhf_spin_summed ? -0.25 : -0.5;
  const std::size_t ao_jet_stride = checked_product(npoint, nbf);

  for (std::size_t point = 0; point < npoint; ++point) {
    const double* phi = ao_jets.data() + point * nbf;
    const double* esp_matrix = esp.values.data() + point * matrix_size;

    std::fill(projected.begin(), projected.end(), 0.0);
    for (std::size_t k = 0; k < nbf; ++k)
      for (std::size_t l = 0; l < nbf; ++l) projected[l] += phi[k] * density[k * nbf + l];

    for (std::size_t j = 0; j < nbf; ++j) {
      double value = 0.0;
      for (std::size_t l = 0; l < nbf; ++l) value += esp_matrix[j * nbf + l] * projected[l];
      potential[j] = value;
    }

    for (unsigned axis = 0; axis < 3; ++axis) {
      const double* phi_derivative = ao_jets.data() + (axis + 1) * ao_jet_stride + point * nbf;
      const double* esp_derivative = esp.probe_derivative.data() + (3 * point + axis) * matrix_size;

      std::fill(projected_derivative.begin(), projected_derivative.end(), 0.0);
      for (std::size_t k = 0; k < nbf; ++k)
        for (std::size_t l = 0; l < nbf; ++l)
          projected_derivative[l] += phi_derivative[k] * density[k * nbf + l];

      for (std::size_t j = 0; j < nbf; ++j) {
        double value = 0.0;
        for (std::size_t l = 0; l < nbf; ++l) {
          value += esp_derivative[j * nbf + l] * projected[l] +
                   esp_matrix[j * nbf + l] * projected_derivative[l];
        }
        potential_derivative[j] = value;
      }

      double contraction = 0.0;
      for (std::size_t i = 0; i < nbf; ++i) {
        for (std::size_t j = 0; j < nbf; ++j) {
          const double raw_ij = phi_derivative[i] * potential[j] + phi[i] * potential_derivative[j];
          const double raw_ji = phi_derivative[j] * potential[i] + phi[j] * potential_derivative[i];
          contraction += density[i * nbf + j] * 0.5 * (raw_ij + raw_ji);
        }
      }
      const double derivative = energy_factor * weights[point] * contraction;
      if (!std::isfinite(derivative))
        throw std::runtime_error("nonfinite COSX explicit-point derivative");
      result.point_gradient[3 * point + axis] = derivative;
    }
  }
  return result;
}

CosxMolecularDerivativeResult build_cosx_molecular_derivative_reference(
    const MolecularGrid& grid, std::span<const double> density, CosxDensityConvention convention,
    CosxReferenceSpec spec) {
  const auto& system = grid.system();
  CosxMolecularDerivativeResult result;
  result.value =
      build_cosx_reference(system, grid.points(), grid.weights(), density, convention, spec);
  const std::size_t nbf = result.value.nbf;
  const std::size_t npoint = result.value.npoint;
  const std::size_t matrix_size = checked_product(nbf, nbf);
  const std::size_t ncoord = checked_product(system.atoms.size(), std::size_t{3});
  if (grid.owners().size() != npoint)
    throw std::logic_error("COSX molecular derivative grid owner count is inconsistent");

  const AoBasis basis(system);
  const std::size_t jet_stride = checked_product(npoint, nbf);
  std::vector<double> ao_jets(checked_product(std::size_t{4}, jet_stride));
  basis.evaluate(grid.points().data(), npoint, 1, 0, nbf, ao_jets.data(), ao_jets.size());
  const auto esp = integrals::build_esp_integrals(system, grid.points());
  if (esp.nbf != nbf || esp.npoint != npoint ||
      esp.values.size() != checked_product(npoint, matrix_size))
    throw std::logic_error("COSX molecular ESP dimensions are inconsistent");

  std::vector<std::size_t> ao_atoms;
  ao_atoms.reserve(nbf);
  for (const auto& shell : system.shells) {
    const auto expansions =
        molecule::ao_expansions(shell.angular_momentum, system.basis_representation);
    for (std::size_t ao = 0; ao < expansions.size(); ++ao) ao_atoms.push_back(shell.atom_index);
  }
  if (ao_atoms.size() != nbf) throw std::logic_error("COSX molecular AO ownership is inconsistent");

  result.nuclear_gradient.assign(ncoord, 0.0);
  std::vector<double> projected(nbf), symmetric_projection(nbf), potential(nbf),
      left_potential(nbf), phi_cotangent(nbf), esp_cotangent(matrix_size),
      weight_sensitivity(npoint);
  const double energy_factor = convention == CosxDensityConvention::rhf_spin_summed ? -0.25 : -0.5;

  for (std::size_t point = 0; point < npoint; ++point) {
    const double* phi = ao_jets.data() + point * nbf;
    const double* esp_matrix = esp.values.data() + point * matrix_size;
    std::fill(projected.begin(), projected.end(), 0.0);
    std::fill(symmetric_projection.begin(), symmetric_projection.end(), 0.0);
    for (std::size_t i = 0; i < nbf; ++i) {
      for (std::size_t j = 0; j < nbf; ++j) {
        projected[j] += phi[i] * density[i * nbf + j];
        symmetric_projection[j] += 0.5 * phi[i] * (density[i * nbf + j] + density[j * nbf + i]);
      }
    }
    std::fill(potential.begin(), potential.end(), 0.0);
    std::fill(left_potential.begin(), left_potential.end(), 0.0);
    for (std::size_t j = 0; j < nbf; ++j) {
      for (std::size_t l = 0; l < nbf; ++l) {
        potential[j] += esp_matrix[j * nbf + l] * projected[l];
        left_potential[l] += esp_matrix[j * nbf + l] * symmetric_projection[j];
      }
    }
    double scalar = 0.0;
    for (std::size_t j = 0; j < nbf; ++j) scalar += symmetric_projection[j] * potential[j];
    weight_sensitivity[point] = energy_factor * scalar;

    const double weighted_factor = energy_factor * grid.weights()[point];
    for (std::size_t i = 0; i < nbf; ++i) {
      double from_left = 0.0, from_right = 0.0;
      for (std::size_t j = 0; j < nbf; ++j) {
        from_left += 0.5 * (density[i * nbf + j] + density[j * nbf + i]) * potential[j];
        from_right += density[i * nbf + j] * left_potential[j];
      }
      phi_cotangent[i] = weighted_factor * (from_left + from_right);
    }
    for (std::size_t j = 0; j < nbf; ++j)
      for (std::size_t l = 0; l < nbf; ++l)
        esp_cotangent[j * nbf + l] = weighted_factor * symmetric_projection[j] * projected[l];

    const std::size_t owner = grid.owners()[point];
    if (owner >= system.atoms.size())
      throw std::logic_error("COSX molecular derivative grid owner is out of range");
    for (std::size_t ao = 0; ao < nbf; ++ao) {
      const std::size_t atom = ao_atoms[ao];
      if (atom >= system.atoms.size())
        throw std::logic_error("COSX molecular derivative AO owner is out of range");
      for (unsigned axis = 0; axis < 3; ++axis) {
        const double spatial = ao_jets[(axis + 1) * jet_stride + point * nbf + ao];
        const double response = phi_cotangent[ao] * spatial;
        result.nuclear_gradient[3 * owner + axis] += response;
        result.nuclear_gradient[3 * atom + axis] -= response;
      }
    }

    const auto esp_response = integrals::contract_weighted_esp_geometry_derivative(
        system, std::span<const double>(grid.points().data() + 3 * point, 3), esp_cotangent);
    if (esp_response.nuclear_derivative.size() != ncoord)
      throw std::logic_error("COSX contracted ESP derivative size is inconsistent");
    for (std::size_t coordinate = 0; coordinate < ncoord; ++coordinate)
      result.nuclear_gradient[coordinate] += esp_response.nuclear_derivative[coordinate];
    for (unsigned axis = 0; axis < 3; ++axis)
      result.nuclear_gradient[3 * owner + axis] += esp_response.probe_derivative[axis];
  }

  const auto weight_response = grid.contract_weight_derivative(weight_sensitivity);
  if (weight_response.size() != ncoord)
    throw std::logic_error("COSX grid weight derivative size is inconsistent");
  for (std::size_t coordinate = 0; coordinate < ncoord; ++coordinate)
    result.nuclear_gradient[coordinate] += weight_response[coordinate];
  if (!std::all_of(result.nuclear_gradient.begin(), result.nuclear_gradient.end(),
                   [](double value) { return std::isfinite(value); }))
    throw std::runtime_error("nonfinite COSX molecular derivative");
  return result;
}

}  // namespace vibeqc::dft
