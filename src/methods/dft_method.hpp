#ifndef VIBEQC_METHODS_DFT_METHOD_HPP
#define VIBEQC_METHODS_DFT_METHOD_HPP

#include <array>
#include <memory>
#include <string>

#include "dft/cuda_ks_final_state.hpp"
#include "methods/method.hpp"

namespace vibeqc::methods::detail {

/** Detached derivative inputs copied from the same owner as the #162 state.
 * The token remains authoritative: this copy alone never proves freshness. */
struct KsDerivativeSnapshot {
  dft::VerifiedKsFinalState state;
  core::System system;
  std::vector<double> overlap, packed_basis, points, weights;
  std::vector<std::uint32_t> grid_owners;
  std::vector<double> atomic_weights;
  std::uint64_t export_d2h_bytes{}, export_reads{}, export_synchronizations{};
};

vibeqc_status read_dft_derivative_state(PreparedBatch& batch, std::size_t index,
                                        const dft::CudaKsFinalStateToken& expected,
                                        KsDerivativeSnapshot& output, std::string& detail);

/** Five explicit CUDA stationary sources: hcore, overlap/Pulay, J, SR-K,
 * LR-K. The live token binds D/W, geometry, radial parameters and spin.
 * XC, nonlocal correlation and nuclear repulsion are separate consumers. */
vibeqc_status dft_cuda_integral_gradient(PreparedBatch& batch, std::size_t index,
                                         const dft::CudaKsFinalStateToken& expected,
                                         std::vector<double>& output, std::size_t maximum_bytes,
                                         std::array<std::uint64_t, 9>& work, std::string& detail);

vibeqc_status validate_dft_system(vibeqc_method method, const core::System& system,
                                  std::string& detail);

/** Internal #163 handoff. These helpers accept prepared CPU RKS/CUDA KS owners;
 * they do not extend public result layouts or authorize force execution. */
vibeqc_status dft_final_state_token(const PreparedCalculation& calculation,
                                    dft::CudaKsFinalStateToken& token, std::string& detail);
vibeqc_status read_dft_final_state(PreparedCalculation& calculation,
                                   const dft::CudaKsFinalStateToken& expected,
                                   bool compute_weighted_density, dft::VerifiedKsFinalState& state,
                                   std::string& detail);
vibeqc_status dft_final_state_token(const PreparedBatch& batch, std::size_t index,
                                    dft::CudaKsFinalStateToken& token, std::string& detail);
vibeqc_status read_dft_final_state(PreparedBatch& batch, std::size_t index,
                                   const dft::CudaKsFinalStateToken& expected,
                                   bool compute_weighted_density, dft::VerifiedKsFinalState& state,
                                   std::string& detail);

std::unique_ptr<PreparedCalculation> prepare_dft_calculation(
    const Capabilities& capabilities, core::ContextState& context, const core::System& system,
    const vibeqc_method_descriptor& descriptor);

std::unique_ptr<PreparedBatch> prepare_dft_batch(const Capabilities& capabilities,
                                                 core::ContextState& context,
                                                 std::vector<core::System> systems,
                                                 const vibeqc_method_descriptor& descriptor,
                                                 vibeqc_batch_flags flags);

}  // namespace vibeqc::methods::detail

#endif
