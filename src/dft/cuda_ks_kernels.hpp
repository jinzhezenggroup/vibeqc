#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace vibeqc::dft::cuda_ks_detail {
struct Scalars {
  double one_electron{}, hartree{}, exact_exchange{}, xc{}, residual{}, density_change{};
  double residual_rms{}, density_rms{};
  // Snapshot admission checks the largest AO commutator entry, not its RMS.
  double maximum_residual{};
  double electrons[2]{}, grid_electrons[2]{};
  double energy_change{};
  int failure{};
};

struct Control {
  double previous_energy{};
  std::uint32_t iterations{};
  int active{}, converged{}, failed{};
};

void reset_control(cudaStream_t stream, unsigned spins, int occupied_alpha, int occupied_beta,
                   Control* control, std::uint8_t* enabled, std::uint8_t* spin_enabled);
void assemble_fock(cudaStream_t stream, std::size_t n, unsigned spins, const double* hcore,
                   const double* coulomb, const double* exchange, double exchange_coefficient,
                   const double* range_exchange, double range_exchange_coefficient,
                   const double* potential, const std::uint8_t* enabled, double* fock);
void stabilize_uks_proposal(cudaStream_t stream, std::size_t n, const double* overlap,
                            const double* occupied_projector, const std::uint8_t* enabled,
                            double* proposal_fock);
void diagnostics(cudaStream_t stream, std::size_t n, unsigned spins, const double* density,
                 const double* proposal, const double* residual, const double* hcore,
                 const double* overlap, const double* coulomb, const double* exchange,
                 double exchange_coefficient, const double* range_exchange,
                 double range_exchange_coefficient, const double* xc_totals, const int* xc_error,
                 const int* jk_error, const int* range_jk_error, const int* solver_info,
                 const std::uint8_t* enabled, Scalars* output);
void advance(cudaStream_t stream, std::size_t n, unsigned spins, double nuclear_repulsion,
             int occupied_alpha, int occupied_beta, double energy_tolerance,
             double density_tolerance, unsigned max_iterations, bool warm_updates, Scalars* current,
             Control* control, const double* proposal, double* density, double* warm,
             std::uint8_t* enabled, std::uint8_t* spin_enabled);
}  // namespace vibeqc::dft::cuda_ks_detail
