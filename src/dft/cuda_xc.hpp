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
 * only the AO jets and density-product panels required by the functional. */
enum class CudaXcAoPrecision : std::uint8_t {
  Fp64 = 0,
  Fp32ComputeFp64Storage = 1,
};

/** Compiler-selected point entry. The immutable functional/response key is
 * resolved at preparation; runtime execution only binds validated device data.
 * Spin remains a layout argument and every entry uses the same FP64 contract. */
using CudaXcPointLauncher = void (*)(cudaStream_t, const double*, const double*, std::size_t,
                                     std::size_t, double*, double*, int*, const double*);

struct CudaXcLayout {
  std::size_t natom{}, nprimitive{}, nao{}, npoint{}, tile_points{}, spins{}, jets{};
  std::size_t work_jets{}, feature_terms{}, packed_elements{}, device_bytes{};
  /** 0=LDA, 1=PBE, 2=r2SCAN, 4=omegaB97M-V semilocal. */
  std::uint32_t functional{};
  bool response{};
  CudaXcAoPrecision ao_precision{CudaXcAoPrecision::Fp64};
};

CudaXcLayout cuda_xc_layout(const AoBasis& basis, const MolecularGrid& grid,
                            std::uint32_t functional, bool unrestricted,
                            std::size_t tile_points = 256,
                            CudaXcAoPrecision ao_precision = CudaXcAoPrecision::Fp64);

/** Metadata-only counterpart of the same layout: does not construct a grid,
 * normalize basis data, initialize CUDA or allocate any numerical buffer. */
CudaXcLayout cuda_xc_layout_shape(std::size_t atoms, std::size_t primitives, std::size_t nao,
                                  std::size_t points, std::uint32_t functional, bool unrestricted,
                                  std::size_t tile_points = 256, bool response = false,
                                  CudaXcAoPrecision ao_precision = CudaXcAoPrecision::Fp64);

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
  CudaXcPlan(const AoBasis& basis, const MolecularGrid& grid, std::uint32_t functional,
             bool unrestricted, std::size_t tile_points, void* arena, std::size_t arena_bytes,
             cudaStream_t stream, CudaXcAoPrecision ao_precision = CudaXcAoPrecision::Fp64);
  /** Private explicit-source constructor for a validated native snapshot.
   * The caller proves packed basis/quadrature identity; setup copies them into
   * the same bounded arena used by SCF. No grid is regenerated for response. */
  CudaXcPlan(CudaXcLayout layout, const std::vector<double>& packed_basis,
             const std::vector<double>& points, const std::vector<double>& weights, void* arena,
             std::size_t arena_bytes, cudaStream_t stream);
  ~CudaXcPlan();
  CudaXcPlan(const CudaXcPlan&) = delete;
  CudaXcPlan& operator=(const CudaXcPlan&) = delete;

  const CudaXcLayout& layout() const noexcept { return layout_; }
  const CudaXcTransfers& transfers() const noexcept { return transfers_; }
  void enqueue(const double* density, std::size_t elements, std::uint64_t generation);
  /** Differentiate the fixed native density on GPU, including AO/feature and
   * matrix assembly. Signed directions use the same input layout as density. */
  void enqueue_response(const double* density, const double* direction, std::size_t elements,
                        std::uint64_t generation);
  CudaXcView view(std::uint64_t generation) const;
  CudaXcScalars read_scalars(std::uint64_t generation);
  /** Explicit user/reference matrix export, never called by enqueue. */
  std::vector<double> download_potential(std::uint64_t generation);

 private:
  void check_device() const;
  void enqueue_impl(const double* density, const double* direction, std::size_t elements,
                    std::uint64_t generation);
  CudaXcLayout layout_;
  CudaXcPointLauncher point_launcher_{};
  CudaXcTransfers transfers_;
  int device_{};
  void* arena_{};
  cudaStream_t stream_{};
  vibeqc::runtime::AsyncGeneration generations_;
  double *basis_{}, *points_{}, *weights_{}, *ao_{}, *work_{}, *features_{}, *coefficients_{},
      *point_totals_{}, *potential_{}, *totals_{}, *delta_features_{};
  int* error_{};
};

namespace cuda_xc_detail {
/** Emitted finite admission selector; performs no CUDA calls or allocation. */
CudaXcPointLauncher resolve_point_launcher(std::uint32_t functional, bool response);
/** Allocation-free launch adapter compiled with the existing generated AO
 * policy. Scientific AO/ingredient arithmetic has one shared generator. */
void enqueue(const CudaXcLayout& layout, CudaXcPointLauncher point_launcher, cudaStream_t stream,
             const double* basis, const double* points, const double* weights,
             const double* density, double* ao, double* work, double* features,
             double* coefficients, double* point_totals, double* potential, double* totals,
             int* error, const double* direction = nullptr, double* delta_features = nullptr);
}  // namespace cuda_xc_detail
}  // namespace vibeqc::dft
