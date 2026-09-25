#include "dft/cuda_xc.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "tensor/cuda_error.hpp"
#include "vibeqc/vibeqc.hpp"

#if defined(VIBEQC_TEST_HOOKS)
namespace {
thread_local cudaError_t fail_next_xc_status = cudaSuccess;
}  // namespace
extern "C" void xc_cuda_fail_next_runtime_for_test_v1() { fail_next_xc_status = cudaErrorUnknown; }
extern "C" void xc_cuda_fail_next_allocation_for_test_v1() {
  fail_next_xc_status = cudaErrorMemoryAllocation;
}
#endif

namespace vibeqc::dft {
namespace {
using vibeqc::runtime::size_add;
using vibeqc::runtime::size_mul;
void check(cudaError_t status) {
  if (status == cudaErrorMemoryAllocation) throw std::bad_alloc();
  if (status != cudaSuccess)
    throw vibeqc::Error(VIBEQC_STATUS_CUDA_ERROR, cudaGetErrorString(status));
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
                            double correlation_scale) {
  // Equal dimensions alone cannot bind a grid to its current geometry/basis.
  const AoBasis grid_basis(grid.system());
  if (basis.nao != grid_basis.nao || basis.natom != grid_basis.natom ||
      basis.nprimitive != grid_basis.nprimitive || basis.packed != grid_basis.packed)
    throw std::invalid_argument("CUDA XC grid/basis identity mismatch");
  return cuda_xc_layout_shape(basis.natom, basis.nprimitive, basis.nao, grid.point_count(),
                              functional, unrestricted, tile_points, false, ao_precision,
                              exchange_scale, correlation_scale);
}

CudaXcLayout cuda_xc_layout_shape(std::size_t atoms, std::size_t primitives, std::size_t nao,
                                  std::size_t points, std::uint32_t functional, bool unrestricted,
                                  std::size_t tile_points, bool response,
                                  CudaXcAoPrecision ao_precision, double exchange_scale,
                                  double correlation_scale) {
  const bool supported_functional = functional <= 2U || functional == 4U;
  if (!atoms || !primitives || !nao || !points || !tile_points || tile_points > INT_MAX ||
      atoms > INT_MAX || primitives > INT_MAX || nao > INT_MAX || !supported_functional)
    throw std::invalid_argument("invalid CUDA XC resource shape");
  if (!std::isfinite(exchange_scale) || !std::isfinite(correlation_scale) || exchange_scale < 0.0 ||
      correlation_scale < 0.0)
    throw std::invalid_argument("invalid CUDA XC component scale");
  if (functional > 1U && (exchange_scale != 1.0 || correlation_scale != 1.0))
    throw std::invalid_argument("scaled meta-GGA CUDA XC is not qualified");
  if (response && (exchange_scale != 1.0 || correlation_scale != 1.0))
    throw std::invalid_argument("scaled CUDA XC response is not qualified");
  if (response && functional > 1U)
    throw std::invalid_argument("CUDA XC response supports LDA/PBE only");
  if (ao_precision != CudaXcAoPrecision::Fp64 &&
      ao_precision != CudaXcAoPrecision::Fp32ComputeFp64Storage)
    throw std::invalid_argument("unknown CUDA XC AO precision");
  if (ao_precision == CudaXcAoPrecision::Fp32ComputeFp64Storage && functional > 1U)
    throw std::invalid_argument("meta-GGA CUDA XC currently requires strict FP64 AO evaluation");
  if (ao_precision == CudaXcAoPrecision::Fp32ComputeFp64Storage && response)
    throw std::invalid_argument("CUDA XC response currently requires strict FP64 AO evaluation");
  constexpr auto overflow = "CUDA XC storage overflow";
  const auto packed =
      size_add(size_add(size_mul(3, atoms, overflow), size_mul(2, primitives, overflow), overflow),
               size_mul(16, nao, overflow), overflow);
  const bool meta_gga = functional == 2U || functional == 4U;
  const auto ao_jets = functional == 0U ? 1U : 4U;
  const auto work_jets = meta_gga ? 4U : 1U;
  const auto feature_terms = functional == 0U ? 1U : (functional == 1U ? 4U : 5U);
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
  std::size_t elements = size_add(out.packed_elements, size_mul(4, out.npoint, overflow), overflow);
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

CudaXcPlan::CudaXcPlan(const AoBasis& basis, const MolecularGrid& grid, std::uint32_t functional,
                       bool unrestricted, std::size_t tile_points, void* arena,
                       std::size_t arena_bytes, cudaStream_t stream, CudaXcAoPrecision ao_precision,
                       double exchange_scale, double correlation_scale)
    : CudaXcPlan(cuda_xc_layout(basis, grid, functional, unrestricted, tile_points, ao_precision,
                                exchange_scale, correlation_scale),
                 basis.packed, grid.points(), grid.weights(), arena, arena_bytes, stream) {}

CudaXcPlan::CudaXcPlan(CudaXcLayout layout, const std::vector<double>& packed_basis,
                       const std::vector<double>& points, const std::vector<double>& weights,
                       void* arena, std::size_t arena_bytes, cudaStream_t stream)
    : layout_(cuda_xc_layout_shape(layout.natom, layout.nprimitive, layout.nao, layout.npoint,
                                   layout.functional, layout.spins == 2, layout.tile_points,
                                   layout.response, layout.ao_precision, layout.exchange_scale,
                                   layout.correlation_scale)),
      point_launcher_(cuda_xc_detail::resolve_point_launcher(layout_.functional, layout_.response)),
      arena_(arena),
      stream_(stream) {
  if (layout.spins != 1 && layout.spins != 2)
    throw std::invalid_argument("CUDA XC spin layout is invalid");
  if (packed_basis.size() != layout_.packed_elements || points.size() != 3 * layout_.npoint ||
      weights.size() != layout_.npoint)
    throw std::invalid_argument("CUDA XC explicit source shape mismatch");
  if (arena_bytes < layout_.device_bytes ||
      reinterpret_cast<std::uintptr_t>(arena) % alignof(double))
    throw std::invalid_argument("CUDA XC arena is too small or misaligned");
  check(cudaGetDevice(&device_));
  device_pointer(arena, device_);
  const auto& l = layout_;
  vibeqc::runtime::BorrowedWorkspace arena_view(arena, arena_bytes);
  vibeqc::runtime::WorkspaceLayout workspace;
  auto take_double = [&](std::size_t count) {
    std::size_t offset = 0;
    if (!workspace.append<double>(count, offset))
      throw std::overflow_error("CUDA XC workspace layout overflow");
    return arena_view.view<double>(offset, count).data;
  };
  basis_ = take_double(l.packed_elements);
  points_ = take_double(size_mul(3, l.npoint, "CUDA XC workspace layout overflow"));
  weights_ = take_double(l.npoint);
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
  if (workspace.bytes() != layout_.device_bytes)
    throw std::logic_error("CUDA XC workspace layout mismatch");
  try {
    check(cudaMemcpyAsync(basis_, packed_basis.data(), l.packed_elements * sizeof(double),
                          cudaMemcpyHostToDevice, stream_));
    check(cudaMemcpyAsync(points_, points.data(), 3 * l.npoint * sizeof(double),
                          cudaMemcpyHostToDevice, stream_));
    check(cudaMemcpyAsync(weights_, weights.data(), l.npoint * sizeof(double),
                          cudaMemcpyHostToDevice, stream_));
    // Complete setup before releasing borrowed host quadrature/basis inputs.
    check(cudaStreamSynchronize(stream_));
  } catch (...) {
    cudaStreamSynchronize(stream_);
    throw;
  }
  transfers_.setup_h2d_bytes =
      size_mul(size_add(l.packed_elements, size_mul(4, l.npoint, "CUDA XC transfer size overflow"),
                        "CUDA XC transfer size overflow"),
               sizeof(double), "CUDA XC transfer size overflow");
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

void CudaXcPlan::enqueue(const double* density, std::size_t elements, std::uint64_t generation) {
  if (layout_.response) throw std::invalid_argument("XC response plan requires a direction");
  enqueue_impl(density, nullptr, elements, generation);
}

void CudaXcPlan::enqueue_response(const double* density, const double* direction,
                                  std::size_t elements, std::uint64_t generation) {
  if (!layout_.response) throw std::invalid_argument("XC plan was not prepared for response");
  enqueue_impl(density, direction, elements, generation);
}

void CudaXcPlan::enqueue_impl(const double* density, const double* direction, std::size_t elements,
                              std::uint64_t generation) {
  check_device();
  const auto matrix = size_mul(layout_.nao, layout_.nao, "CUDA XC density size overflow");
  const auto count = size_mul(layout_.spins, matrix, "CUDA XC density size overflow");
  if (elements != count) throw std::invalid_argument("CUDA XC density size is invalid");
  if (!generation || generation <= generations_.submitted())
    throw std::invalid_argument("CUDA XC density generation is stale");
  device_pointer(density, device_);
  const auto input_bytes = size_mul(count, sizeof(double), "CUDA XC density size overflow");
  if (vibeqc::runtime::ranges_overlap(density, input_bytes, arena_, layout_.device_bytes))
    throw std::invalid_argument("CUDA XC density aliases its workspace");
  if (layout_.response) {
    device_pointer(direction, device_);
    if (vibeqc::runtime::ranges_overlap(direction, input_bytes, arena_, layout_.device_bytes))
      throw std::invalid_argument("CUDA XC direction aliases its workspace");
  }
  generations_.begin(generation);
  try {
#if defined(VIBEQC_TEST_HOOKS)
    // Exercise the generated executor's real exception types without leaving a
    // failed CUDA context behind; explicit replay must retain the last-good seed.
    const auto injected = fail_next_xc_status;
    fail_next_xc_status = cudaSuccess;
    vibeqc_tensor::cuda_check(injected);
#endif
    cuda_xc_detail::enqueue(layout_, point_launcher_, stream_, basis_, points_, weights_, density,
                            ao_, work_, features_, coefficients_, point_totals_, potential_,
                            totals_, error_, direction, delta_features_);
  } catch (const vibeqc_tensor::DeviceAllocationError&) {
    // The generated executor has a separate exception vocabulary. Translate at
    // this native owner boundary so both single-point and batch APIs preserve it.
    throw std::bad_alloc();
  } catch (const vibeqc_tensor::DeviceRuntimeError& error) {
    throw vibeqc::Error(VIBEQC_STATUS_CUDA_ERROR, error.what());
  }
  generations_.commit(generation);
  ++transfers_.evaluations;
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
}  // namespace vibeqc::dft
