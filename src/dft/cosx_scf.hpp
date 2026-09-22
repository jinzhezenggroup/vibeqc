#ifndef VIBEQC_DFT_COSX_SCF_HPP
#define VIBEQC_DFT_COSX_SCF_HPP

#include <vector>

#include "dft/cosx_fock_provider.hpp"
#include "scf/types.hpp"

namespace vibeqc::dft {

/** Host-controlled RHF using the prepared RI-J/COSX-K value/force provider. */
scf::ScfResult run_cosx_rhf(PreparedCosxFockPlan& plan, const scf::ScfOptions& options,
                            const std::vector<double>* initial_density = nullptr);

/** Host-controlled UHF using independent alpha/beta COSX value/force exchange. */
scf::ScfResult run_cosx_uhf(PreparedCosxFockPlan& plan, const scf::ScfOptions& options,
                            const std::vector<double>* initial_density = nullptr);

}  // namespace vibeqc::dft

#endif
