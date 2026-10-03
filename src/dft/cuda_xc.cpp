#include "dft/cuda_xc.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "generated_split_hybrid_registry.cuh"
#include "generativeqc/generativeqc.hpp"
#include "tensor/cuda_error.hpp"

#if defined(GENERATIVEQC_TEST_HOOKS)
namespace {
thread_local cudaError_t fail_next_xc_status = cudaSuccess;
thread_local cudaError_t fail_next_nonlocal_xc_status = cudaSuccess;
}  // namespace
extern "C" void xc_cuda_fail_next_runtime_for_test_v1() { fail_next_xc_status = cudaErrorUnknown; }
extern "C" void xc_cuda_fail_next_allocation_for_test_v1() {
  fail_next_xc_status = cudaErrorMemoryAllocation;
}
extern "C" void xc_cuda_fail_next_nonlocal_runtime_for_test_v1() {
  fail_next_nonlocal_xc_status = cudaErrorUnknown;
}
extern "C" void xc_cuda_fail_next_nonlocal_allocation_for_test_v1() {
  fail_next_nonlocal_xc_status = cudaErrorMemoryAllocation;
}
#endif

namespace generativeqc::dft {
namespace {
using generativeqc::runtime::size_add;
using generativeqc::runtime::size_mul;
void check(cudaError_t status) {
  if (status == cudaErrorMemoryAllocation) throw std::bad_alloc();
  if (status != cudaSuccess)
    throw generativeqc::Error(GENERATIVEQC_STATUS_CUDA_ERROR, cudaGetErrorString(status));
}
void device_pointer(const void* pointer, int device) {
  if (!pointer) throw std::invalid_argument("null CUDA XC device buffer");
  cudaPointerAttributes attributes{};
  check(cudaPointerGetAttributes(&attributes, pointer));
  if (attributes.type != cudaMemoryTypeDevice || attributes.device != device)
    throw std::invalid_argument("CUDA XC requires a device buffer on the current device");
}
}  // namespace

CudaXcLayout cuda_xc_layout(const AoBasis& basis, const MolecularGrid& grid,
                            std::uint32_t functional, bool unrestricted, std::size_t tile_points,
                            CudaXcAoPrecision ao_precision, double exchange_scale,
                            double correlation_scale, bool borrow_resident_grid) {
  // Equal dimensions alone cannot bind a grid to its current geometry/basis.
  const AoBasis grid_basis(grid.system());
  if (basis.nao != grid_basis.nao || basis.natom != grid_basis.natom ||
      basis.nprimitive != grid_basis.nprimitive || basis.packed != grid_basis.packed)
    throw std::invalid_argument("CUDA XC grid/basis identity mismatch");
  if (borrow_resident_grid && !grid.cuda_view())
    throw std::invalid_argument("CUDA XC requested a resident grid from a host-only owner");
  return cuda_xc_layout_shape(basis.natom, basis.nprimitive, basis.nao, grid.point_count(),
                              functional, unrestricted, tile_points, false, ao_precision,
                              exchange_scale, correlation_scale, borrow_resident_grid);
}

CudaXcLayout cuda_xc_layout_shape(std::size_t atoms, std::size_t primitives, std::size_t nao,
                                  std::size_t points, std::uint32_t functional, bool unrestricted,
                                  std::size_t tile_points, bool response,
                                  CudaXcAoPrecision ao_precision, double exchange_scale,
                                  double correlation_scale, bool borrow_resident_grid) {
  const bool generated_split_hybrid = generated::split_hybrid_registered(functional);
  const bool supported_functional = functional <= 4U || generated_split_hybrid;
  if (!atoms || !primitives || !nao || !points || !tile_points || tile_points > INT_MAX ||
      atoms > INT_MAX || primitives > INT_MAX || nao > INT_MAX || !supported_functional)
    throw std::invalid_argument("invalid CUDA XC resource shape");
  if (!std::isfinite(exchange_scale) || !std::isfinite(correlation_scale) || exchange_scale < 0.0 ||
      correlation_scale < 0.0)
    throw std::invalid_argument("invalid CUDA XC component scale");
  if (functional != 1U && (exchange_scale != 1.0 || correlation_scale != 1.0))
    throw std::invalid_argument("scaled CUDA XC is currently qualified for PBE only");
  if (response && (exchange_scale != 1.0 || correlation_scale != 1.0))
    throw std::invalid_argument("scaled CUDA XC response is not qualified");
  if (response && functional > 1U)
    throw std::invalid_argument("CUDA XC response supports LDA/PBE only");
  if (ao_precision != CudaXcAoPrecision::Fp64 &&
      ao_precision != CudaXcAoPrecision::Fp32ComputeFp64Storage)
    throw std::invalid_argument("unknown CUDA XC AO precision");
  if (ao_precision == CudaXcAoPrecision::Fp32ComputeFp64Storage && functional > 1U)
    throw std::invalid_argument("non-LDA/PBE CUDA XC currently requires strict FP64 AO evaluation");
  if (ao_precision == CudaXcAoPrecision::Fp32ComputeFp64Storage && response)
    throw std::invalid_argument("CUDA XC response currently requires strict FP64 AO evaluation");
  constexpr auto overflow = "CUDA XC storage overflow";
  const auto packed =
      size_add(size_add(size_mul(3, atoms, overflow), size_mul(2, primitives, overflow), overflow),
               size_mul(16, nao, overflow), overflow);
  const bool meta_gga = functional == 2U || functional == 4U ||
                        (generated_split_hybrid && generated::split_hybrid_is_mgga(functional));
  const auto ao_jets = functional == 0U ? 1U : 4U;
  const auto work_jets = meta_gga ? 4U : 1U;
  const auto feature_terms = functional == 0U ? 1U : (meta_gga ? 5U : 4U);
  CudaXcLayout out{atoms,
                   primitives,
                   nao,
                   points,
                   std::min(tile_points, points),
                   unrestricted ? 2U : 1U,
                   ao_jets,
                   work_jets,
                   feature_terms,
                   packed,
                   0,
                   functional,
                   exchange_scale,
                   correlation_scale,
                   response,
                   ao_precision};
  out.borrowed_grid = borrow_resident_grid;
  std::size_t elements = out.packed_elements;
  if (!out.borrowed_grid)
    elements = size_add(elements, size_mul(4, out.npoint, overflow), overflow);
  const auto panel = size_mul(out.tile_points, out.nao, overflow);
  const auto panel_terms =
      size_add(out.jets, size_mul(out.spins, out.work_jets, overflow), overflow);
  elements = size_add(elements, size_mul(panel_terms, panel, overflow), overflow);
  auto feature_storage = size_mul(response ? 3 : 2, out.spins, overflow);
  feature_storage = size_mul(feature_storage, out.feature_terms, overflow);
  feature_storage = size_add(feature_storage, 3, overflow);
  elements = size_add(elements, size_mul(feature_storage, out.tile_points, overflow), overflow);
  const auto matrix = size_mul(out.nao, out.nao, overflow);
  elements = size_add(elements, size_mul(out.spins, matrix, overflow), overflow);
  elements = size_add(elements, 3, overflow);
  // Preserve the historical double-sized error slot so resource bounds and
  // diagnostics remain byte-for-byte unchanged.
  out.device_bytes = size_mul(size_add(elements, 1, overflow), sizeof(double), overflow);
  return out;
}

CudaXcLayout cuda_xc_local_ao_layout(CudaXcLayout dense, const CudaXcAoTiles& maps) {
  if (dense.local_ao || dense.ao_map_entries || dense.host_ao_map_bytes || dense.response ||
      dense.ao_precision != CudaXcAoPrecision::Fp64 || !dense.tile_points || !dense.npoint)
    throw std::invalid_argument("local CUDA XC maps require a dense physical FP64 layout");
  const auto tiles = 1 + (dense.npoint - 1) / dense.tile_points;
  if (maps.offsets.size() != tiles + 1 || maps.offsets.front() != 0 ||
      maps.offsets.back() != maps.indices.size())
    throw std::invalid_argument("local CUDA XC map offsets differ from the point tile domain");
  for (std::size_t tile = 0; tile < tiles; ++tile) {
    const auto first = maps.offsets[tile], last = maps.offsets[tile + 1];
    if (first > last || last > maps.indices.size() || last - first > dense.nao)
      throw std::invalid_argument("invalid local CUDA XC map extent");
    for (auto i = first; i < last; ++i)
      if (maps.indices[i] >= dense.nao || (i > first && maps.indices[i - 1] >= maps.indices[i]))
        throw std::invalid_argument("local CUDA XC AO maps must be sorted, unique and in range");
  }
  constexpr auto overflow = "local CUDA XC map storage overflow";
  dense.local_ao = true;
  dense.ao_map_entries = maps.indices.size();
  dense.host_ao_map_bytes = size_mul(maps.offsets.size(), sizeof(std::size_t), overflow);
  dense.device_bytes = size_add(
      dense.device_bytes, size_mul(maps.indices.size(), sizeof(std::size_t), overflow), overflow);
  return dense;
}

CudaXcPlan::CudaXcPlan(const AoBasis& basis, const MolecularGrid& grid, std::uint32_t functional,
                       bool unrestricted, std::size_t tile_points, void* arena,
                       std::size_t arena_bytes, cudaStream_t stream, CudaXcAoPrecision ao_precision,
                       double exchange_scale, double correlation_scale, bool borrow_resident_grid)
    : CudaXcPlan(cuda_xc_layout(basis, grid, functional, unrestricted, tile_points, ao_precision,
                                exchange_scale, correlation_scale, borrow_resident_grid),
                 basis.packed, grid.points(), grid.weights(), arena, arena_bytes, stream,
                 borrow_resident_grid ? grid.cuda_view() : CudaMolecularGridView{}) {}

CudaXcPlan::CudaXcPlan(CudaXcLayout layout, const std::vector<double>& packed_basis,
                       const std::vector<double>& points, const std::vector<double>& weights,
                       void* arena, std::size_t arena_bytes, cudaStream_t stream,
                       CudaMolecularGridView borrowed_grid, const CudaXcAoTiles* ao_maps)
    : layout_(cuda_xc_layout_shape(layout.natom, layout.nprimitive, layout.nao, layout.npoint,
                                   layout.functional, layout.spins == 2, layout.tile_points,
                                   layout.response, layout.ao_precision, layout.exchange_scale,
                                   layout.correlation_scale, layout.borrowed_grid)),
      point_launcher_(cuda_xc_detail::resolve_point_launcher(layout_.functional, layout_.response)),
      arena_(arena),
      stream_(stream) {
  if (ao_maps) {
    layout_ = cuda_xc_local_ao_layout(layout_, *ao_maps);
    ao_offsets_ = ao_maps->offsets;
  }
  if (layout.local_ao != layout_.local_ao || layout.ao_map_entries != layout_.ao_map_entries ||
      layout.host_ao_map_bytes != layout_.host_ao_map_bytes ||
      layout.device_bytes != layout_.device_bytes)
    throw std::invalid_argument("CUDA XC local map resource layout mismatch");
  if (layout.spins != 1 && layout.spins != 2)
    throw std::invalid_argument("CUDA XC spin layout is invalid");
  if (packed_basis.size() != layout_.packed_elements || points.size() != 3 * layout_.npoint ||
      weights.size() != layout_.npoint)
    throw std::invalid_argument("CUDA XC explicit source shape mismatch");
  if (layout_.borrowed_grid != static_cast<bool>(borrowed_grid))
    throw std::invalid_argument("CUDA XC resident-grid layout/source mismatch");
  if (arena_bytes < layout_.device_bytes ||
      reinterpret_cast<std::uintptr_t>(arena) % alignof(double))
    throw std::invalid_argument("CUDA XC arena is too small or misaligned");
  check(cudaGetDevice(&device_));
  device_pointer(arena, device_);
  if (borrowed_grid) {
    if (borrowed_grid.device != device_ || borrowed_grid.point_count != layout_.npoint ||
        borrowed_grid.device_bytes != cuda_resident_grid_bytes(layout_.npoint))
      throw std::invalid_argument("CUDA XC resident grid has incompatible device or shape");
    device_pointer(borrowed_grid.points, device_);
    device_pointer(borrowed_grid.weights, device_);
    grid_lifetime_ = borrowed_grid.lifetime;
  }
  const auto& l = layout_;
  generativeqc::runtime::BorrowedWorkspace arena_view(arena, arena_bytes);
  generativeqc::runtime::WorkspaceLayout workspace;
  auto take_double = [&](std::size_t count) {
    std::size_t offset = 0;
    if (!workspace.append<double>(count, offset))
      throw std::overflow_error("CUDA XC workspace layout overflow");
    return arena_view.view<double>(offset, count).data;
  };
  basis_ = take_double(l.packed_elements);
  double *owned_points = nullptr, *owned_weights = nullptr;
  if (l.borrowed_grid) {
    points_ = const_cast<double*>(borrowed_grid.points);
    weights_ = const_cast<double*>(borrowed_grid.weights);
  } else {
    owned_points = take_double(size_mul(3, l.npoint, "CUDA XC workspace layout overflow"));
    owned_weights = take_double(l.npoint);
    points_ = owned_points;
    weights_ = owned_weights;
  }
  const auto panel = size_mul(l.tile_points, l.nao, "CUDA XC workspace layout overflow");
  ao_ = take_double(size_mul(l.jets, panel, "CUDA XC workspace layout overflow"));
  work_ = take_double(size_mul(size_mul(l.spins, l.work_jets, "CUDA XC workspace layout overflow"),
                               panel, "CUDA XC workspace layout overflow"));
  const auto feature_panel =
      size_mul(l.spins, l.feature_terms, "CUDA XC workspace layout overflow");
  features_ =
      take_double(size_mul(feature_panel, l.tile_points, "CUDA XC workspace layout overflow"));
  coefficients_ =
      take_double(size_mul(feature_panel, l.tile_points, "CUDA XC workspace layout overflow"));
  if (l.response)
    delta_features_ =
        take_double(size_mul(feature_panel, l.tile_points, "CUDA XC workspace layout overflow"));
  point_totals_ = take_double(size_mul(3, l.tile_points, "CUDA XC workspace layout overflow"));
  const auto matrix = size_mul(l.nao, l.nao, "CUDA XC workspace layout overflow");
  potential_ = take_double(size_mul(l.spins, matrix, "CUDA XC workspace layout overflow"));
  totals_ = take_double(3);
  auto* error_storage = take_double(1);
  error_ = reinterpret_cast<int*>(error_storage);
  if (l.ao_map_entries) {
    std::size_t offset = 0;
    if (!workspace.append<std::size_t>(l.ao_map_entries, offset))
      throw std::overflow_error("local CUDA XC map storage overflow");
    ao_ids_ = arena_view.view<std::size_t>(offset, l.ao_map_entries).data;
  }
  if (workspace.bytes() != layout_.device_bytes)
    throw std::logic_error("CUDA XC workspace layout mismatch");
  try {
    check(cudaMemcpyAsync(basis_, packed_basis.data(), l.packed_elements * sizeof(double),
                          cudaMemcpyHostToDevice, stream_));
    if (!l.borrowed_grid) {
      check(cudaMemcpyAsync(owned_points, points.data(), 3 * l.npoint * sizeof(double),
                            cudaMemcpyHostToDevice, stream_));
      check(cudaMemcpyAsync(owned_weights, weights.data(), l.npoint * sizeof(double),
                            cudaMemcpyHostToDevice, stream_));
    }
    if (l.ao_map_entries)
      check(cudaMemcpyAsync(ao_ids_, ao_maps->indices.data(),
                            l.ao_map_entries * sizeof(std::size_t), cudaMemcpyHostToDevice,
                            stream_));
    // Complete setup before releasing borrowed host quadrature/basis inputs.
    check(cudaStreamSynchronize(stream_));
  } catch (...) {
    cudaStreamSynchronize(stream_);
    throw;
  }
  const auto setup_elements =
      l.borrowed_grid
          ? l.packed_elements
          : size_add(l.packed_elements, size_mul(4, l.npoint, "CUDA XC transfer size overflow"),
                     "CUDA XC transfer size overflow");
  transfers_.setup_h2d_bytes =
      size_add(size_mul(setup_elements, sizeof(double), "CUDA XC transfer size overflow"),
               size_mul(l.ao_map_entries, sizeof(std::size_t), "CUDA XC transfer size overflow"),
               "CUDA XC transfer size overflow");
  transfers_.synchronizations = 1;
}

CudaXcPlan::~CudaXcPlan() {
  int previous = 0;
  cudaGetDevice(&previous);
  cudaSetDevice(device_);
  cudaStreamSynchronize(stream_);
  cudaSetDevice(previous);
}

void CudaXcPlan::check_device() const {
  int current = -1;
  check(cudaGetDevice(&current));
  if (current != device_) throw std::invalid_argument("CUDA XC current device changed");
}

CudaXcGridView CudaXcPlan::grid_view() const {
  check_device();
  return {points_, weights_, layout_.npoint, stream_};
}

void CudaXcPlan::enqueue(const double* density, std::size_t elements, std::uint64_t generation,
                         CudaXcDensityPrecision precision) {
  if (layout_.response) throw std::invalid_argument("XC response plan requires a direction");
  enqueue_impl(density, nullptr, elements, generation, precision);
}

void CudaXcPlan::enqueue_density_features(const double* density, std::size_t elements,
                                          std::uint64_t generation, double* total_density,
                                          double* total_gradient) {
  if (layout_.response)
    throw std::invalid_argument("XC response plan cannot publish physical features");
  if (total_density == nullptr || total_gradient == nullptr)
    throw std::invalid_argument("CUDA XC density-feature export requires both output buffers");
  enqueue_impl(density, nullptr, elements, generation, CudaXcDensityPrecision::Fp64, total_density,
               total_gradient);
}

CudaXcView CudaXcPlan::enqueue_replay_density_features(const double* density, std::size_t elements,
                                                       double* total_density,
                                                       double* total_gradient) {
  if (layout_.response)
    throw std::invalid_argument("XC response plan cannot publish physical replay features");
  if (total_density == nullptr || total_gradient == nullptr)
    throw std::invalid_argument("CUDA XC replay density-feature export requires both outputs");
  enqueue_impl(density, nullptr, elements, 0, CudaXcDensityPrecision::Fp64, total_density,
               total_gradient, false);
  return {0, layout_.nao, layout_.spins, potential_, totals_, error_, stream_};
}

void CudaXcPlan::enqueue_response(const double* density, const double* direction,
                                  std::size_t elements, std::uint64_t generation) {
  if (!layout_.response) throw std::invalid_argument("XC plan was not prepared for response");
  enqueue_impl(density, direction, elements, generation, CudaXcDensityPrecision::Fp64);
}

CudaXcView CudaXcPlan::enqueue_replay_body(const double* density, std::size_t elements,
                                           CudaXcDensityPrecision precision) {
  if (layout_.response) throw std::invalid_argument("XC response plan requires a direction");
  enqueue_impl(density, nullptr, elements, 0, precision, nullptr, nullptr, false);
  return {0, layout_.nao, layout_.spins, potential_, totals_, error_, stream_};
}

void CudaXcPlan::publish_submitted_generation(std::uint64_t generation) {
  generations_.begin(generation);
  generations_.commit(generation);
  ++transfers_.evaluations;
}

void CudaXcPlan::enqueue_nonlocal_potential(std::uint64_t generation,
                                            const double* effective_weights,
                                            const double* total_gradient, const double* vrho,
                                            const double* vsigma, const double* nonlocal_energy) {
  enqueue_nonlocal_potential_impl(generation, true, effective_weights, total_gradient, vrho, vsigma,
                                  nonlocal_energy);
}

void CudaXcPlan::enqueue_replay_nonlocal_potential(const double* effective_weights,
                                                   const double* total_gradient, const double* vrho,
                                                   const double* vsigma,
                                                   const double* nonlocal_energy) {
  enqueue_nonlocal_potential_impl(0, false, effective_weights, total_gradient, vrho, vsigma,
                                  nonlocal_energy);
}

void CudaXcPlan::enqueue_nonlocal_potential_impl(std::uint64_t generation, bool publish_generation,
                                                 const double* effective_weights,
                                                 const double* total_gradient, const double* vrho,
                                                 const double* vsigma,
                                                 const double* nonlocal_energy) {
  check_device();
  if (layout_.response)
    throw std::invalid_argument("XC response plan cannot accumulate a physical nonlocal potential");
  if (layout_.feature_terms < 4 || layout_.ao_precision != CudaXcAoPrecision::Fp64)
    throw std::invalid_argument("CUDA nonlocal AO assembly requires strict-FP64 GGA ingredients");
  if (publish_generation)
    generations_.require(generation);
  else if (generation != 0)
    throw std::invalid_argument("CUDA replay nonlocal assembly received a logical generation");
  if (!effective_weights || !total_gradient || !vrho || !vsigma || !nonlocal_energy)
    throw std::invalid_argument("CUDA nonlocal AO assembly received a null device input");
  const auto scalar_bytes =
      size_mul(layout_.npoint, sizeof(double), "CUDA nonlocal AO input size overflow");
  const auto gradient_bytes =
      size_mul(size_mul(3, layout_.npoint, "CUDA nonlocal AO input size overflow"), sizeof(double),
               "CUDA nonlocal AO input size overflow");
  for (const auto* pointer : {effective_weights, vrho, vsigma, nonlocal_energy})
    device_pointer(pointer, device_);
  device_pointer(total_gradient, device_);
  const auto overlaps_arena = [&](const void* pointer, std::size_t bytes) {
    return generativeqc::runtime::ranges_overlap(pointer, bytes, arena_, layout_.device_bytes);
  };
  if (overlaps_arena(effective_weights, scalar_bytes) ||
      overlaps_arena(total_gradient, gradient_bytes) || overlaps_arena(vrho, scalar_bytes) ||
      overlaps_arena(vsigma, scalar_bytes) || overlaps_arena(nonlocal_energy, sizeof(double)))
    throw std::invalid_argument("CUDA nonlocal AO inputs alias the semilocal XC workspace");

  // Ordinary execution revokes the published semilocal generation while the
  // nonlocal contribution mutates it. Replay execution has no logical
  // generation yet; the SolverRegion publishes only after physical submission.
  if (publish_generation) generations_.revoke(generation);
  try {
#if defined(GENERATIVEQC_TEST_HOOKS)
    const auto injected = fail_next_nonlocal_xc_status;
    fail_next_nonlocal_xc_status = cudaSuccess;
    generativeqc_tensor::cuda_check(injected);
#endif
    cuda_xc_detail::enqueue_nonlocal_potential(layout_, stream_, basis_, points_, effective_weights,
                                               total_gradient, vrho, vsigma, nonlocal_energy, ao_,
                                               coefficients_, potential_, totals_, error_, work_,
                                               ao_offsets_, ao_ids_);
    if (publish_generation) generations_.commit(generation);
  } catch (const generativeqc_tensor::DeviceAllocationError&) {
    (void)cudaStreamSynchronize(stream_);
    throw std::bad_alloc();
  } catch (const generativeqc_tensor::DeviceRuntimeError& error) {
    (void)cudaStreamSynchronize(stream_);
    throw generativeqc::Error(GENERATIVEQC_STATUS_CUDA_ERROR, error.what());
  } catch (...) {
    (void)cudaStreamSynchronize(stream_);
    throw;
  }
}

void CudaXcPlan::enqueue_impl(const double* density, const double* direction, std::size_t elements,
                              std::uint64_t generation, CudaXcDensityPrecision precision,
                              double* total_density, double* total_gradient,
                              bool publish_generation) {
  check_device();
  const auto matrix = size_mul(layout_.nao, layout_.nao, "CUDA XC density size overflow");
  const auto count = size_mul(layout_.spins, matrix, "CUDA XC density size overflow");
  if (elements != count) throw std::invalid_argument("CUDA XC density size is invalid");
  if (precision != CudaXcDensityPrecision::Fp64 &&
      precision != CudaXcDensityPrecision::Fp32ComputeFp64Accumulate)
    throw std::invalid_argument("unknown CUDA XC density precision");
  if (layout_.local_ao && precision != CudaXcDensityPrecision::Fp64)
    throw std::invalid_argument("local CUDA XC maps require FP64 density contraction");
  if (precision == CudaXcDensityPrecision::Fp32ComputeFp64Accumulate && layout_.functional > 2U)
    throw std::invalid_argument(
        "mixed CUDA XC density precision is not qualified for this functional");
  if (publish_generation && (!generation || generation <= generations_.submitted()))
    throw std::invalid_argument("CUDA XC density generation is stale");
  device_pointer(density, device_);
  const auto input_bytes = size_mul(count, sizeof(double), "CUDA XC density size overflow");
  if (generativeqc::runtime::ranges_overlap(density, input_bytes, arena_, layout_.device_bytes))
    throw std::invalid_argument("CUDA XC density aliases its workspace");
  if ((total_density == nullptr) != (total_gradient == nullptr))
    throw std::invalid_argument("CUDA XC density-feature outputs must be provided together");
  if (total_density) {
    if (layout_.feature_terms < 4)
      throw std::invalid_argument("CUDA XC density-gradient export requires GGA ingredients");
    const auto rho_bytes =
        size_mul(layout_.npoint, sizeof(double), "CUDA XC feature export size overflow");
    const auto gradient_bytes =
        size_mul(size_mul(3, layout_.npoint, "CUDA XC feature export size overflow"),
                 sizeof(double), "CUDA XC feature export size overflow");
    device_pointer(total_density, device_);
    device_pointer(total_gradient, device_);
    if (generativeqc::runtime::ranges_overlap(total_density, rho_bytes, arena_,
                                              layout_.device_bytes) ||
        generativeqc::runtime::ranges_overlap(total_gradient, gradient_bytes, arena_,
                                              layout_.device_bytes) ||
        generativeqc::runtime::ranges_overlap(total_density, rho_bytes, density, input_bytes) ||
        generativeqc::runtime::ranges_overlap(total_gradient, gradient_bytes, density,
                                              input_bytes) ||
        generativeqc::runtime::ranges_overlap(total_density, rho_bytes, total_gradient,
                                              gradient_bytes))
      throw std::invalid_argument("CUDA XC density-feature outputs alias live input/workspace");
  }
  if (layout_.response) {
    device_pointer(direction, device_);
    if (generativeqc::runtime::ranges_overlap(direction, input_bytes, arena_, layout_.device_bytes))
      throw std::invalid_argument("CUDA XC direction aliases its workspace");
  }
  if (publish_generation) generations_.begin(generation);
  try {
#if defined(GENERATIVEQC_TEST_HOOKS)
    // Exercise the generated executor's real exception types without leaving a
    // failed CUDA context behind; explicit replay must retain the last-good seed.
    const auto injected = fail_next_xc_status;
    fail_next_xc_status = cudaSuccess;
    generativeqc_tensor::cuda_check(injected);
#endif
    cuda_xc_detail::enqueue(layout_, point_launcher_, stream_, basis_, points_, weights_, density,
                            ao_, work_, features_, coefficients_, point_totals_, potential_,
                            totals_, error_, precision, direction, delta_features_, total_density,
                            total_gradient, ao_offsets_, ao_ids_);
  } catch (const generativeqc_tensor::DeviceAllocationError&) {
    // The generated executor has a separate exception vocabulary. Translate at
    // this native owner boundary so both single-point and batch APIs preserve it.
    throw std::bad_alloc();
  } catch (const generativeqc_tensor::DeviceRuntimeError& error) {
    throw generativeqc::Error(GENERATIVEQC_STATUS_CUDA_ERROR, error.what());
  }
  if (publish_generation) {
    generations_.commit(generation);
    ++transfers_.evaluations;
  }
}

CudaXcView CudaXcPlan::view(std::uint64_t generation) const {
  check_device();
  generations_.require(generation);
  return {generation, layout_.nao, layout_.spins, potential_, totals_, error_, stream_};
}

CudaXcScalars CudaXcPlan::read_scalars(std::uint64_t generation) {
  const auto result = view(generation);
  double values[3]{};
  CudaXcScalars out;
  try {
    check(cudaMemcpyAsync(values, result.totals, sizeof(values), cudaMemcpyDeviceToHost, stream_));
    check(cudaMemcpyAsync(&out.error, result.error, sizeof(out.error), cudaMemcpyDeviceToHost,
                          stream_));
    check(cudaStreamSynchronize(stream_));
  } catch (...) {
    cudaStreamSynchronize(stream_);
    throw;
  }
  out.energy = values[0];
  out.electrons = {values[1], values[2]};
  transfers_.output_d2h_bytes += sizeof(values) + sizeof(out.error);
  ++transfers_.synchronizations;
  return out;
}

std::vector<double> CudaXcPlan::download_potential(std::uint64_t generation) {
  const auto result = view(generation);
  std::vector<double> output(layout_.spins * layout_.nao * layout_.nao);
  try {
    check(cudaMemcpyAsync(output.data(), result.potential, output.size() * sizeof(double),
                          cudaMemcpyDeviceToHost, stream_));
    check(cudaStreamSynchronize(stream_));
  } catch (...) {
    cudaStreamSynchronize(stream_);
    throw;
  }
  transfers_.output_d2h_bytes += output.size() * sizeof(double);
  ++transfers_.synchronizations;
  return output;
}
}  // namespace generativeqc::dft
