#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace vibeqc::scf::cuda_df {
// Fixed-size two-stage reductions keep the scratch independent of AO count.
// Norms are merged with hypot, and signed traces/energies use compensation.
inline constexpr unsigned validation_threads = 128;
inline unsigned validation_block_count(std::size_t n) {
  const auto blocks = (n * n + validation_threads - 1) / validation_threads;
  return static_cast<unsigned>(blocks < 128 ? blocks : 128);
}
// Bit zero rejects numerical products; this bit denotes a broken provider
// input/identity and must abort selection without attempting correction.
inline constexpr int validation_input_failure = 2;
struct ValidationPartial {
  double norm_f, norm_c, norm_rhs, norm_residual, norm_density;
  double eigen, metric, canonical, density, idempotency, commutator;
  double electrons, energy;
  // A full-width status word leaves no unwritten trailing padding in the
  // packet copied to the host (including under CUDA initcheck).
  std::uint64_t invalid;
};
// CuMetal rejects implicit initialization of shared variables. Keep the packet
// trivial and value-initialize local accumulators at each reduction entry point.
static_assert(std::is_trivial_v<ValidationPartial>);
struct ValidationInputs {
  std::size_t n{}, occupied{};
  double weight{};
  const double *f{}, *s{}, *h{}, *d{}, *c{}, *values{};
  const double* expected_density{};
  const std::uint64_t* generation{};
  const int* info{};
  std::uint64_t expected_generation{};
  bool physical_fock{};
};
// J/K use the existing row-major provider layout; validation uses column major.
void launch_validation_fock(cudaStream_t stream, std::size_t n, const double* hcore,
                            const double* coulomb, const double* exchange, double exchange_weight,
                            double* fock);
void launch_validation_eigen(cudaStream_t stream, ValidationInputs inputs, const double* fc,
                             const double* sc, const double* gram, const double* canonical,
                             ValidationPartial* partial);
void launch_validation_density(cudaStream_t stream, ValidationInputs inputs,
                               const double* reconstructed, const double* ds, const double* dsd,
                               ValidationPartial* partial);
void launch_validation_commutator(cudaStream_t stream, std::size_t n, const double* fds,
                                  const double* sdf, ValidationPartial* partial);
void launch_validation_finish(cudaStream_t stream, ValidationPartial* partial,
                              ValidationPartial* result, unsigned stages, unsigned blocks);
// Scale occupied columns before GEMM; negative orbital energies are supported.
void launch_validation_columns(cudaStream_t stream, std::size_t n, std::size_t occupied,
                               const double* c, const double* values, double weight,
                               double* columns);
}  // namespace vibeqc::scf::cuda_df
