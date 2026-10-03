#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "dft/ao_grid.hpp"
#include "dft/ao_selection_work.hpp"
#include "dft/grid.hpp"
#include "runtime/bounded_workspace.hpp"

namespace generativeqc::dft {

/** Exact explicit storage request for ordinary-stream semilocal XC. The
 * method's ResourcePlan supplies one arena of device_bytes; this component
 * does not allocate CUDA memory or introduce another user memory budget.
 * Host quadrature is prepared separately and uploaded once. Tiles retain
 * only the AO jets and density-product panels required by the functional. */
enum class CudaXcAoPrecision : std::uint8_t {
  Fp64 = 0,
  Fp32ComputeFp64Storage = 1,
};

/** Density-times-AO arithmetic. Mixed evaluates products in explicit RN FP32
 * while keeping storage, the long reduction, point XC, Vxc and scalar reductions FP64.
 * Mixed is qualified for LDA, PBE and r2SCAN; omegaB97M-V requires Fp64. */
enum class CudaXcDensityPrecision : std::uint8_t {
  Fp64 = 0,
  Fp32ComputeFp64Accumulate = 1,
};

/** Compiler-selected point entry. The immutable functional/response key is
 * resolved at preparation; runtime execution only binds validated device data.
 * Spin remains a layout argument and every entry uses the same FP64 contract. */
using CudaXcPointLauncher = void (*)(cudaStream_t, const double*, const double*, std::size_t,
                                     std::size_t, double*, double*, int*, std::uint32_t, double,
                                     double, const double*);

struct CudaXcLayout {
  std::size_t natom{}, nprimitive{}, nao{}, npoint{}, tile_points{}, spins{}, jets{};
  std::size_t work_jets{}, feature_terms{}, packed_elements{}, device_bytes{};
  /** 0=LDA, 1=PBE, 2=r2SCAN, 4=omegaB97M-V semilocal. */
  std::uint32_t functional{};
  /** Independent semilocal X/C weights resolved by MethodIR. Exact exchange is
   * owned by the prepared Fock provider and is never folded into these scales. */
  double exchange_scale{1.0}, correlation_scale{1.0};
  bool response{};
  CudaXcAoPrecision ao_precision{CudaXcAoPrecision::Fp64};
  /** Full-grid points/weights are borrowed from an immutable MolecularGrid
   * device owner instead of occupying this arena. */
  bool borrowed_grid{};
  /** Explicit local maps are optional; their device indices and retained host
   * offsets are charged separately. Full-capacity AO scratch remains bounded
   * by nao, never by the mean selected column count. */
  bool local_ao{};
  std::size_t ao_map_entries{}, host_ao_map_bytes{};
};

/** Explicit CSR maps for the immutable point-tile sequence. Every local map
 * is sorted, unique, and in range; empty tiles are legal. These indices define
 * the caller's selected scientific domain, not an error-certified cutoff. */
struct CudaXcAoTiles {
  std::vector<std::size_t> offsets, indices;
};

/** Worst-case admission for sampled-jet discovery. The device bound includes
 * the dense arena and one global-capacity map per tile; host peak includes
 * those maps, offsets, and the current tile's flags. No mean AO count enters
 * admission. Discovery borrows the already charged AO/work scratch. */
struct CudaXcAoSelectionResources {
  std::size_t device_bytes{}, host_peak_bytes{}, max_entries{}, tiles{};
};
CudaXcAoSelectionResources cuda_xc_ao_selection_resources(const CudaXcLayout& dense);

/** Validate maps and charge their storage on top of the ordinary dense layout.
 * Only physical FP64 execution is admitted; response retains its dense route.
 * No discovery, screening threshold, CUDA allocation or GPU work occurs here. */
CudaXcLayout cuda_xc_local_ao_layout(CudaXcLayout dense, const CudaXcAoTiles& maps);

CudaXcLayout cuda_xc_layout(const AoBasis& basis, const MolecularGrid& grid,
                            std::uint32_t functional, bool unrestricted,
                            std::size_t tile_points = 256,
                            CudaXcAoPrecision ao_precision = CudaXcAoPrecision::Fp64,
                            double exchange_scale = 1.0, double correlation_scale = 1.0,
                            bool borrow_resident_grid = false);

/** Metadata-only counterpart of the same layout: does not construct a grid,
 * normalize basis data, initialize CUDA or allocate any numerical buffer. */
CudaXcLayout cuda_xc_layout_shape(std::size_t atoms, std::size_t primitives, std::size_t nao,
                                  std::size_t points, std::uint32_t functional, bool unrestricted,
                                  std::size_t tile_points = 256, bool response = false,
                                  CudaXcAoPrecision ao_precision = CudaXcAoPrecision::Fp64,
                                  double exchange_scale = 1.0, double correlation_scale = 1.0,
                                  bool borrow_resident_grid = false);

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

struct CudaXcGridView {
  const double *points{}, *weights{};
  std::size_t point_count{};
  cudaStream_t stream{};
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
             cudaStream_t stream, CudaXcAoPrecision ao_precision = CudaXcAoPrecision::Fp64,
             double exchange_scale = 1.0, double correlation_scale = 1.0,
             bool borrow_resident_grid = false);
  /** Private explicit-source constructor for a validated native snapshot.
   * The caller proves packed basis/quadrature identity; setup copies them into
   * the same bounded arena used by SCF. No grid is regenerated for response. */
  CudaXcPlan(CudaXcLayout layout, const std::vector<double>& packed_basis,
             const std::vector<double>& points, const std::vector<double>& weights, void* arena,
             std::size_t arena_bytes, cudaStream_t stream, CudaMolecularGridView borrowed_grid = {},
             const CudaXcAoTiles* ao_maps = nullptr);
  ~CudaXcPlan();
  CudaXcPlan(const CudaXcPlan&) = delete;
  CudaXcPlan& operator=(const CudaXcPlan&) = delete;

  const CudaXcLayout& layout() const noexcept { return layout_; }
  const CudaXcTransfers& transfers() const noexcept { return transfers_; }
  /** Explicit setup-only policy; no density work or external oracle is used.
   * Returns false without discovery if either numeric budget is insufficient.
   * A successful selection is immutable for the lifetime of this geometry
   * owner. A positive sampled-jet cutoff requires endpoint qualification. */
  bool select_local_ao(double cutoff, std::size_t max_host_bytes);
  const CudaXcAoSelectionWork& ao_selection_work() const noexcept { return ao_selection_work_; }
  /** Borrow immutable device quadrature owned by this plan. */
  CudaXcGridView grid_view() const;
  void enqueue(const double* density, std::size_t elements, std::uint64_t generation,
               CudaXcDensityPrecision precision = CudaXcDensityPrecision::Fp64);
  /** Enqueue only the stable device body for a shared replay region. The caller
   * must publish exactly one logical generation per physically submitted body
   * after the runtime chooses warmup/capture/replay/fallback. */
  CudaXcView enqueue_replay_body(const double* density, std::size_t elements,
                                 CudaXcDensityPrecision precision = CudaXcDensityPrecision::Fp64);
  /** Replay-runtime counterpart that also exports total rho/grad-rho for a
   * resident nonlocal consumer. Logical generation publication remains owned
   * by publish_submitted_generation() after the runtime selects physical work. */
  CudaXcView enqueue_replay_density_features(const double* density, std::size_t elements,
                                             double* total_density, double* total_gradient);
  /** Execute the ordinary physical XC evaluation while also publishing total
   * rho and grad-rho to caller-owned full-grid device buffers. This adds no
   * plan-owned storage and is admitted only for GGA/meta-GGA ingredient sets. */
  void enqueue_density_features(const double* density, std::size_t elements,
                                std::uint64_t generation, double* total_density,
                                double* total_gradient);
  /** Accumulate a total-density nonlocal contribution into the already
   * submitted physical XC potential/totals for the same generation. All
   * inputs are caller-owned full-grid device arrays/scalars. */
  void enqueue_nonlocal_potential(std::uint64_t generation, const double* effective_weights,
                                  const double* total_gradient, const double* vrho,
                                  const double* vsigma, const double* nonlocal_energy);
  /** Replay-runtime counterpart for the same nonlocal AO assembly. It mutates
   * the current semilocal potential/totals but performs no generation
   * publication; the surrounding SolverRegion publishes the physical body once. */
  void enqueue_replay_nonlocal_potential(const double* effective_weights,
                                         const double* total_gradient, const double* vrho,
                                         const double* vsigma, const double* nonlocal_energy);
  /** Differentiate the fixed native density on GPU, including AO/feature and
   * matrix assembly. Signed directions use the same input layout as density. */
  void enqueue_response(const double* density, const double* direction, std::size_t elements,
                        std::uint64_t generation);
  /** Publish host generation/accounting after a replay-runtime body has been
   * physically submitted. This performs no numerical launch or transfer. */
  void publish_submitted_generation(std::uint64_t generation);
  CudaXcView view(std::uint64_t generation) const;
  CudaXcScalars read_scalars(std::uint64_t generation);
  /** Explicit user/reference matrix export, never called by enqueue. */
  std::vector<double> download_potential(std::uint64_t generation);

 private:
  void check_device() const;
  void enqueue_impl(const double* density, const double* direction, std::size_t elements,
                    std::uint64_t generation, CudaXcDensityPrecision precision,
                    double* total_density = nullptr, double* total_gradient = nullptr,
                    bool publish_generation = true);
  void enqueue_nonlocal_potential_impl(std::uint64_t generation, bool publish_generation,
                                       const double* effective_weights,
                                       const double* total_gradient, const double* vrho,
                                       const double* vsigma, const double* nonlocal_energy);
  CudaXcLayout layout_;
  CudaXcPointLauncher point_launcher_{};
  CudaXcTransfers transfers_;
  CudaXcAoSelectionWork ao_selection_work_;
  bool evaluation_started_{};
  int device_{};
  void* arena_{};
  std::size_t arena_bytes_{};
  cudaStream_t stream_{};
  std::shared_ptr<const void> grid_lifetime_;
  generativeqc::runtime::AsyncGeneration generations_;
  double *basis_{}, *points_{}, *weights_{}, *ao_{}, *work_{}, *features_{}, *coefficients_{},
      *point_totals_{}, *potential_{}, *totals_{}, *delta_features_{};
  int* error_{};
  std::size_t* ao_ids_{};
  // Immutable host offsets determine launch shapes and survive graph capture.
  // Only offsets are retained here; device indices live in the caller's arena.
  std::vector<std::size_t> ao_offsets_;
};

namespace cuda_xc_detail {
/** Populate one host flag per global AO from all actual jets in a point tile.
 * The caller lends full-capacity panels and owns stream/error lifetimes. */
void select_ao(const CudaXcLayout& layout, cudaStream_t stream, const double* basis,
               const double* points, std::size_t count, double cutoff, double* ao, double* work,
               int* error, unsigned* host_flags);
/** Emitted finite admission selector; performs no CUDA calls or allocation. */
CudaXcPointLauncher resolve_point_launcher(std::uint32_t functional, bool response);
/** Allocation-free launch adapter compiled with the existing generated AO
 * policy. Scientific AO/ingredient arithmetic has one shared generator. */
void enqueue(const CudaXcLayout& layout, CudaXcPointLauncher point_launcher, cudaStream_t stream,
             const double* basis, const double* points, const double* weights,
             const double* density, double* ao, double* work, double* features,
             double* coefficients, double* point_totals, double* potential, double* totals,
             int* error, CudaXcDensityPrecision precision, const double* direction = nullptr,
             double* delta_features = nullptr, double* total_density = nullptr,
             double* total_gradient = nullptr, const std::vector<std::size_t>& ao_offsets = {},
             const std::size_t* ao_ids = nullptr);
void enqueue_nonlocal_potential(const CudaXcLayout& layout, cudaStream_t stream,
                                const double* basis, const double* points,
                                const double* effective_weights, const double* total_gradient,
                                const double* vrho, const double* vsigma,
                                const double* nonlocal_energy, double* ao, double* coefficients,
                                double* potential, double* totals, int* error, double* work,
                                const std::vector<std::size_t>& ao_offsets = {},
                                const std::size_t* ao_ids = nullptr);
}  // namespace cuda_xc_detail
}  // namespace generativeqc::dft
