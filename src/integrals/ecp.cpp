#include "integrals/ecp.hpp"

#include <algorithm>
#include <stdexcept>

#include "generated_ecp_ao.cuh"
#include "molecule/basis.hpp"

namespace vibeqc::integrals {
namespace {
constexpr int projector_count = generated::ecp_projector_count;
struct Component {
  unsigned x, y, z;
  double coefficient;
};
struct AO {
  const core::Shell* shell;
  int primitive_offset{}, primitive_count{}, term_count{};
  double x, y, z;
  Component components[3];
};
struct Jet {
  double v[4];
};
}  // namespace

void ecp_quadrature(unsigned radial, unsigned angular, std::vector<EcpRadialPoint>& radii,
                    std::vector<EcpSpherePoint>& sphere) {
  generated::ecp_make_grid(radial, angular, radii, sphere);
}

EcpData ecp_integrals(const core::System& system, unsigned radial, unsigned angular,
                      bool derivatives) {
  if (radial < 16 || radial > 512 || angular < 8 || angular > 96)
    throw std::invalid_argument("ECP quadrature requires 16..512 radial and 8..96 polar points");
  const auto n = molecule::ao_count(system);
  if (n > 256 || system.atoms.size() > 128)
    throw std::invalid_argument("ECP baseline supports at most 256 AOs and 128 atoms");
  const auto size = n * n;
  std::vector<AO> aos;
  for (const auto& shell : system.shells) {
    const auto& position = system.atoms[shell.atom_index].position;
    const auto expansions =
        molecule::ao_expansions(shell.angular_momentum, system.basis_representation);
    if (std::any_of(expansions.begin(), expansions.end(),
                    [](const auto& expansion) { return expansion.size() > 3; }))
      throw std::invalid_argument("ECP AO expansion exceeds validated domain");
    for (const auto& expansion : expansions) {
      AO ao{&shell,
            0,
            static_cast<int>(shell.primitives.size()),
            static_cast<int>(expansion.size()),
            position[0],
            position[1],
            position[2],
            {}};
      for (unsigned t = 0; t < expansion.size(); ++t) {
        const auto& v = expansion[t];
        ao.components[t] = {v.component[0], v.component[1], v.component[2],
                            generated::ecp_component_coefficient(v.component[0], v.component[1],
                                                                 v.component[2], v.coefficient)};
      }
      aos.push_back(ao);
    }
  }
  EcpData out{n,
              derivatives ? 3 * system.atoms.size() : 0,
              std::vector<double>(size),
              std::vector<double>(size),
              {},
              {}};
  out.local_derivative.resize(out.ncoord * size);
  out.nonlocal_derivative.resize(out.ncoord * size);
  if (system.ecp_terms.empty()) return out;
  std::vector<EcpRadialPoint> radii;
  std::vector<EcpSpherePoint> sphere;
  ecp_quadrature(radial, angular, radii, sphere);
  // A radial layer is still the lifetime boundary. Share its potentials and
  // projected AO jets across every pair, independently of radial grid length.
  std::vector<Jet> values(n * sphere.size()), projections(n * projector_count);
  for (std::size_t center = 0; center < system.atoms.size(); ++center) {
    if (!system.atoms[center].ecp_core) continue;
    const auto& c = system.atoms[center].position;
    for (const auto& radial_point : radii) {
      double potentials[generated::ecp_max_projector_angular + 2];
      generated::ecp_potentials(system.ecp_terms.data(), system.ecp_terms.size(), radial_point,
                                center, potentials);
      if (std::all_of(std::begin(potentials), std::end(potentials),
                      [](double v) { return v == 0; }))
        continue;
      for (std::size_t a = 0; a < n; ++a) {
        for (std::size_t q = 0; q < sphere.size(); ++q)
          values[a * sphere.size() + q] =
              generated::ecp_evaluate_ao<Jet>(aos[a], aos[a].shell->primitives.data(), sphere[q],
                                              radial_point.r, c[0], c[1], c[2], derivatives);
        for (int m = 0; m < projector_count; ++m)
          projections[a * projector_count + m] = generated::ecp_project(
              values.data() + a * sphere.size(), sphere.data(), sphere.size(), m, derivatives);
      }
      for (std::size_t a = 0; a < n; ++a)
        for (std::size_t b = 0; b <= a; ++b) {
          double parts[2][10];
          generated::ecp_contract_potentials(
              potentials, sphere.data(), sphere.size(), values.data() + a * sphere.size(),
              values.data() + b * sphere.size(), projections.data() + a * projector_count,
              projections.data() + b * projector_count, derivatives, parts);
          for (unsigned part = 0; part < 2; ++part) {
            const auto& v = parts[part];
            auto& matrix = part == 0 ? out.local : out.nonlocal;
            auto& gradient = part == 0 ? out.local_derivative : out.nonlocal_derivative;
            for (unsigned transpose = 0; transpose < (a == b ? 1U : 2U); ++transpose) {
              const auto item = transpose ? b * n + a : a * n + b;
              matrix[item] += v[0];
              if (derivatives)
                for (unsigned d = 0; d < 3; ++d) {
                  gradient[(aos[a].shell->atom_index * 3 + d) * size + item] += v[d + 1];
                  gradient[(aos[b].shell->atom_index * 3 + d) * size + item] += v[d + 4];
                  gradient[(center * 3 + d) * size + item] += v[d + 7];
                }
            }
          }
        }
    }
  }
  return out;
}

EcpData checked_ecp_integrals(const core::System& system, bool derivatives) {
  const auto coarse = ecp_integrals(system, generated::ecp_coarse_radial_points,
                                    generated::ecp_coarse_polar_points, derivatives);
  auto fine = ecp_integrals(system, generated::ecp_refined_radial_points,
                            generated::ecp_refined_polar_points, derivatives);
  const auto check = [](const auto& a, const auto& b, bool derivative) {
    for (std::size_t i = 0; i < a.size(); ++i)
      if (!generated::ecp_grid_pair_accepted(a[i], b[i], derivative))
        throw std::runtime_error(
            "ECP quadrature convergence gate failed; this input is outside the validated execution "
            "domain");
  };
  check(coarse.local, fine.local, false);
  check(coarse.nonlocal, fine.nonlocal, false);
  check(coarse.local_derivative, fine.local_derivative, true);
  check(coarse.nonlocal_derivative, fine.nonlocal_derivative, true);
  return fine;
}

void add_ecp(const EcpData& ecp, std::vector<double>& hcore, std::vector<double>& derivative) {
  if (hcore.size() != ecp.local.size() || derivative.size() != ecp.local_derivative.size())
    throw std::invalid_argument("ECP and one-electron matrix layouts differ");
  for (std::size_t i = 0; i < hcore.size(); ++i)
    hcore[i] = generated::ecp_add_operator(hcore[i], ecp.local[i], ecp.nonlocal[i]);
  for (std::size_t i = 0; i < derivative.size(); ++i)
    derivative[i] = generated::ecp_add_operator(derivative[i], ecp.local_derivative[i],
                                                ecp.nonlocal_derivative[i]);
}
}  // namespace vibeqc::integrals
