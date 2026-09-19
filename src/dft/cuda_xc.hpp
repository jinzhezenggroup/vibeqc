#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "dft/ao_grid.hpp"
#include "dft/grid.hpp"
#include "runtime/bounded_workspace.hpp"

namespace vibeqc::dft {

/** Exact explicit storage request for ordinary-stream semilocal XC. The
 * method's ResourcePlan supplies one arena of device_bytes; this component
 * does not allocate CUDA memory or introduce another user memory budget.
 * Host quadrature is prepared separately and uploaded once. Tiles retain
 * only value jets for LDA and value/first jets for PBE, with no tau panel. */
struct CudaXcLayout {
  std::size_t natom{}, nprimitive{}, nao{}, npoint{}, tile_points{}, spins{}, jets{};
  std::size_t packed_elements{}, device_bytes{};
  bool pbe{};
};

CudaXcLayout cuda_xc_layout(const AoBasis& basis, const MolecularGrid& grid, bool pbe,
                            bool unrestricted, std::size_t tile_points = 256);

/** Metadata-only counterpart of the same layout: does not construct a grid,
 * normalize basis data, initialize CUDA or allocate any numerical buffer. */
CudaXcLayout cuda_xc_layout_shape(std::size_t atoms, std::size_t primitives, std::size_t nao,
                                  std::size_t points, bool pbe, bool unrestricted,
                                  std::size_t tile_points = 256);

struct CudaXcTransfers {
  std::uint64_t setup_h2d_bytes{}, output_d2h_bytes{}, synchronizations{}, evaluations{};
};

/** Borrowed current result on the plan's stream. potential is row-major
 * [spin,AO,AO]; restricted input/output uses total D and one potential.
 * totals is [E_xc,N_alpha,N_beta]. error==0 means numerically valid only after
 * the stream reaches this result. A later enqueue invalidates every view. */
struct CudaXcView {
  std::uint64_t generation{};
  std::size_t nao{}, spins{};
  const double *potential{}, *totals{};
  const int* error{};
  cudaStream_t stream{};
};

struct CudaXcScalars {
  double energy{};
  std::array<double, 2> electrons{};
  int error{};
};

/** Immutable geometry/basis/grid/functional owner with borrowed device arena
 * and stream. Both must outlive this object; destruction drains the stream.
 * Input density remains caller-owned and must survive the enqueued work.
 * Concurrent access is not supported; independent plans isolate failures.
 * Every enqueue fully rebuilds features/E/V from a strictly newer density
 * generation. Only explicit scalar/output access performs D2H or a fence. */
class CudaXcPlan {
 public:
  CudaXcPlan(const AoBasis& basis, const MolecularGrid& grid, bool pbe, bool unrestricted,
             std::size_t tile_points, void* arena, std::size_t arena_bytes, cudaStream_t stream);
  ~CudaXcPlan();
  CudaXcPlan(const CudaXcPlan&) = delete;
  CudaXcPlan& operator=(const CudaXcPlan&) = delete;

  const CudaXcLayout& layout() const noexcept { return layout_; }
  const CudaXcTransfers& transfers() const noexcept { return transfers_; }
  void enqueue(const double* density, std::size_t elements, std::uint64_t generation);
  CudaXcView view(std::uint64_t generation) const;
  CudaXcScalars read_scalars(std::uint64_t generation);
  /** Explicit user/reference matrix export, never called by enqueue. */
  std::vector<double> download_potential(std::uint64_t generation);

 private:
  void check_device() const;
  CudaXcLayout layout_;
  CudaXcTransfers transfers_;
  int device_{};
  void* arena_{};
  cudaStream_t stream_{};
  vibeqc::runtime::AsyncGeneration generations_;
  double *basis_{}, *points_{}, *weights_{}, *ao_{}, *work_{}, *features_{}, *coefficients_{},
      *point_totals_{}, *potential_{}, *totals_{};
  int* error_{};
};

namespace cuda_xc_detail {
/** Allocation-free launch adapter compiled with the existing generated AO
 * policy. Scientific AO/ingredient arithmetic has one shared generator. */
void enqueue(const CudaXcLayout& layout, cudaStream_t stream, const double* basis,
             const double* points, const double* weights, const double* density, double* ao,
             double* work, double* features, double* coefficients, double* point_totals,
             double* potential, double* totals, int* error);
}  // namespace cuda_xc_detail
}  // namespace vibeqc::dft
