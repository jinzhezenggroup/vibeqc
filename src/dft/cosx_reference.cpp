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

}  // namespace vibeqc::dft
