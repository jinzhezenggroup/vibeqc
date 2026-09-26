#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace vibeqc::scf::cuda_df {

/** Unit-weight lower-pair density: diagonal once, off-diagonal D_mn+D_nm.
 * The sum is invariant to row/column-major interpretation even for nonsymmetric D.
 */
void launch_pack_df_density(cudaStream_t stream, std::size_t n, const double* density,
                            double* packed);

/** Expand one symmetric packed AO matrix into public row-major storage. */
void launch_scatter_df_pairs(cudaStream_t stream, std::size_t n, const double* packed,
                             double* matrix);

/** Expand a bounded raw/transformed lower-pair panel. Output is either
 * [q,row,nu] or [row,nu,q], with compact active extents and no hidden padding.
 * The caller checks offsets/capacity and owns all buffers on the same stream.
 */
void launch_unpack_df_values(cudaStream_t stream, std::size_t n, std::size_t a,
                             std::size_t row_begin, std::size_t rows, std::size_t qbegin,
                             std::size_t qcount, bool auxiliary_major, const double* packed,
                             double* output);

/** FP64 occupied contraction directly from lower pairs into U[mu,i,Q].
 * rank>0 and complete U capacity are checked by the caller. The fixed 64x16
 * tiled iterator preserves nu order and never materializes a dense B panel.
 */
void launch_project_packed_df(cudaStream_t stream, std::size_t n, std::size_t a, std::size_t rank,
                              bool column_major, const double* packed, const double* coefficients,
                              double* projection);

}  // namespace vibeqc::scf::cuda_df
