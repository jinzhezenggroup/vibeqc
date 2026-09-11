#ifndef VIBEQC_SCF_CUDA_ONE_ELECTRON_DERIVATIVES_CUH
#define VIBEQC_SCF_CUDA_ONE_ELECTRON_DERIVATIVES_CUH

#include "scf/cuda/one_electron_values.cuh"

namespace vibeqc::scf {

/** Non-owning full [system, AO, AO] weights held fixed during differentiation.
 * A null channel means zero. Triangular ownership uses W_ij + W_ji off the
 * diagonal, so nonsymmetric external Lagrangian weights have exact semantics.
 * Channel scales permit HF to reuse D for T/V and -energy-weighted D for S
 * without allocating transformed weight matrices. This layer assumes no HF
 * occupation factors and never differentiates the supplied weights.
 */
struct OneElectronWeightView {
  const double* overlap{};
  const double* kinetic{};
  const double* attraction{};
  double overlap_scale{1.0}, kinetic_scale{1.0}, attraction_scale{1.0};
};

/** Accumulate sum(W_S*dS + W_T*dT + W_V*dV) into an existing O(Natom) buffer.
 * Output is the energy gradient, multiplied by output_sign (HF forces use -1).
 * Nuclear repulsion belongs to the independent caller and is never included.
 * The buffer is not cleared; accumulation may follow an existing nuclear term.
 * An optional active mask isolates failed batch items. Schedule 0 owns AO pairs
 * by thread, 1 uses shell-pair warp lanes, and 2 uses one serial owner per system
 * for deterministic diagnostics. No derivative tensors, allocation, transfer,
 * or stream synchronization occurs in this launch.
 */
cudaError_t launch_generated_one_electron_gradient(
    const OneElectronDeviceView& batch, const std::int32_t* pair_first,
    const std::int32_t* pair_second, std::size_t pair_count, const OneElectronWeightView& weights,
    const std::uint8_t* active, unsigned schedule, double output_sign, double* gradient,
    cudaStream_t stream);

}  // namespace vibeqc::scf

#endif
