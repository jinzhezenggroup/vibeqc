#pragma once

#include <cstddef>
#include <vector>

namespace generativeqc::cc::triples {

/** Complete standalone DF (T) evaluation, including staged inputs and BLAS allowance. */
struct DFCudaResult {
  double energy{};
  double minimum_absolute_denominator{};
  double seconds{};
  std::size_t virtual_triples{}, occupied_tiles{};
  std::size_t workspace_bytes{}, arena_bytes{}, provider_retained_bytes{};
  std::size_t panel_capacity{}, panel_gemms{}, moment_gemms{};
  std::size_t epilogue_kernels{}, reduction_kernels{}, epilogue_points{};
  std::size_t contraction_summands{}, h2d_bytes{}, d2h_bytes{};
};

/** Complete fixed-canonical-input triples pullback; all arrays are detached.
 * Bvv is an unprojected dense Frobenius cotangent on the symmetric physical
 * factor domain. eps_o/eps_v describe canonical denominator derivatives;
 * they do not by themselves certify same-space Fock response at degeneracy.
 */
struct DFCudaResponseResult {
  DFCudaResult diagnostic;
  std::vector<double> bov, bvv, ovoo, ovov, fov, t1, t2, eps_o, eps_v;
  std::size_t numeric_capacity_bytes{}, borrowed_host_bytes{};
  std::size_t reverse_gemms{}, reverse_kernels{}, audit_kernels{}, scalar_response_evaluations{};
};

/** Standard closed-shell (T) on a supplied physical DF Hamiltonian.
 * Q-major B_ov/B_vv replace resident ovvv. Other inputs retain the canonical
 * spatial-MO layout. The owner uploads each input once, builds at most three
 * occupied integral panels and six virtual W cubes, and drains before result
 * publication. Tight budgets fall back to one panel without changing equations.
 * max_bytes covers this owner's numeric storage plus the bounded BLAS allowance;
 * callers composing endpoints separately charge their retained host/CC state.
 */
#if GENERATIVEQC_HAS_CUDA
DFCudaResult evaluate_df_cuda(std::size_t o, std::size_t v, std::size_t q, const double* bov,
                              const double* bvv, const double* ovoo, const double* ovov,
                              const double* fov, const double* t1, const double* t2,
                              const double* eps_o, const double* eps_v,
                              double denominator_threshold, std::size_t max_bytes, int device,
                              std::size_t max_panel_buffers = 3);
/** Differentiate the complete occupied-tile energy on CUDA.
 * Includes both the original energy and all nine input cotangents, staged once.
 * Complete admission charges borrowed host input values, detached outputs,
 * owned CUDA/provider storage and caller_bytes for all other live numeric state.
 * No CPU mathematical fallback or complete ovvv/rank-six tensor is constructed.
 * A one-panel fallback preserves the same equations when three panels do not fit.
 * This internal fixed-frame derivative is not a complete molecular force.
 */
DFCudaResponseResult pullback_df_cuda(std::size_t o, std::size_t v, std::size_t q,
                                      const double* bov, const double* bvv, const double* ovoo,
                                      const double* ovov, const double* fov, const double* t1,
                                      const double* t2, const double* eps_o, const double* eps_v,
                                      double denominator_threshold, std::size_t max_bytes,
                                      int device, std::size_t caller_bytes = 0,
                                      std::size_t max_panel_buffers = 3);
#endif

}  // namespace generativeqc::cc::triples
