#ifndef VIBEQC_METHODS_DFT_METHOD_HPP
#define VIBEQC_METHODS_DFT_METHOD_HPP

#include <memory>
#include <string>

#include "dft/cuda_ks_final_state.hpp"
#include "methods/method.hpp"

namespace vibeqc::methods::detail {

vibeqc_status validate_dft_system(vibeqc_method method, const core::System& system,
                                  std::string& detail);

/** Internal #163 handoff. These helpers accept only prepared CUDA KS owners;
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
