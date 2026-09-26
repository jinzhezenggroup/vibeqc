#ifndef VIBEQC_SCF_GRADIENT_HF_GRADIENT_HPP
#define VIBEQC_SCF_GRADIENT_HF_GRADIENT_HPP
#include <span>

#include "integrals/s_integrals.hpp"
#include "scf/reference/linalg.hpp"

namespace vibeqc::scf::gradient {
using reference::Matrix;
/** Assemble stationary restricted HF forces from separately owned derivatives.
 * two_electron is the resolved provider's positive energy derivative. D and W
 * include occupation factors. Inputs use full dense AO and nuclear-coordinate
 * ordering with dimensions validated by the provider/solver. Preserve the
 * nuclear, one-electron, overlap/Pulay and two-electron accumulation order;
 * negate exactly once when returning forces. No provider or solver is owned.
 */
std::vector<double> analytic_forces(const integrals::IntegralData& ints, const Matrix& density,
                                    const Matrix& weighted_density,
                                    std::span<const double> two_electron);
/** Unrestricted assembly; the two spin densities and weighted densities each
 * carry unit occupations, while the provider derivative already includes both.
 */
std::vector<double> analytic_uhf_forces(const integrals::IntegralData& ints,
                                        const Matrix& alpha_density, const Matrix& beta_density,
                                        const Matrix& alpha_weighted_density,
                                        const Matrix& beta_weighted_density,
                                        std::span<const double> two_electron);
}  // namespace vibeqc::scf::gradient
#endif
