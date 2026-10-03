#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <mutex>
#include <span>
#include <vector>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "dft/nonlocal_correlation/vv10_runtime.hpp"
#include "generativeqc/generativeqc.h"
#if GENERATIVEQC_HAS_CUDA
#include "dft/grid_task_view.cuh"
#include "runtime/cuda_resources.cuh"
#endif

struct generativeqc_nonlocal_plan {
  generativeqc_context* context{};
  std::unique_ptr<generativeqc::dft::nlc::Vv10Plan> plan;
};

#if GENERATIVEQC_HAS_CUDA
struct generativeqc_nonlocal_cuda_force {
  generativeqc_context* context{};
  generativeqc::dft::nlc::Vv10Parameters parameters{};
  generativeqc::dft::nlc::Vv10CudaDeviceLayout layout{};
  int device{-1};
  std::size_t point_count{}, next_offset{}, device_bytes{};
  double density_threshold{};
  std::uint64_t generation{};
  cudaStream_t stream{};
  bool executed{};

  generativeqc::runtime::OwnedCudaBuffer<double> arena;
  generativeqc::runtime::OwnedCudaBuffer<int> errors;
  generativeqc::runtime::OwnedCudaEvent source_ready;
  double *coordinates{}, *weights{}, *raw_density{}, *raw_gradient{}, *effective_weights{},
      *effective_density{}, *effective_gradient{}, *seeds{}, *point_derivative{}, *workspace{};

  ~generativeqc_nonlocal_cuda_force() {
    if (!stream || device < 0) return;
    int previous = 0;
    (void)cudaGetDevice(&previous);
    (void)cudaSetDevice(device);
    (void)cudaStreamSynchronize(stream);
    (void)cudaSetDevice(previous);
  }
};
#endif

namespace {

generativeqc::dft::nlc::Vv10Variant variant(generativeqc_nonlocal_variant value) {
  using generativeqc::dft::nlc::Vv10Variant;
  if (value == GENERATIVEQC_NONLOCAL_VV10) return Vv10Variant::vv10;
  if (value == GENERATIVEQC_NONLOCAL_RVV10) return Vv10Variant::rvv10;
  throw std::invalid_argument("unsupported VV10/rVV10 kernel variant");
}

template <class T>
std::span<T> optional_span(T* pointer, std::uint32_t count, const char* label) {
  if ((pointer == nullptr) != (count == 0))
    throw std::invalid_argument(std::string(label) + " pointer/count disagree");
  return pointer ? std::span<T>(pointer, count) : std::span<T>{};
}

#if GENERATIVEQC_HAS_CUDA
// The dry query and allocating owner share this inventory. Fifteen full-grid
// arrays cover xyz(3), immutable weights(1), rho/gradient(4), effective weights(1)
// and seeds(6). Domain preparation overwrites private rho/gradient in place;
// the later pair kernels reuse the dead gradient panel for point derivatives.
// Pair workspace and three sticky error flags remain separately charged.
std::size_t nonlocal_force_arena_doubles(
    const generativeqc::dft::nlc::Vv10CudaDeviceLayout& layout) {
  using generativeqc::runtime::size_add;
  using generativeqc::runtime::size_mul;
  return size_add(
      size_mul(std::size_t{15}, layout.point_count, "resident nonlocal force extent overflow"),
      layout.workspace_bytes / sizeof(double), "resident nonlocal force extent overflow");
}

std::size_t nonlocal_force_device_bytes(
    const generativeqc::dft::nlc::Vv10CudaDeviceLayout& layout) {
  using generativeqc::runtime::size_add;
  using generativeqc::runtime::size_mul;
  return size_add(size_mul(nonlocal_force_arena_doubles(layout), sizeof(double),
                           "resident nonlocal force byte extent overflow"),
                  std::size_t{3} * sizeof(int), "resident nonlocal force error extent overflow");
}

void drain_failed_seed_source(cudaStream_t stream) noexcept {
  if (stream) (void)cudaStreamSynchronize(stream);
}
#endif

}  // namespace

extern "C" {

generativeqc_status generativeqc_nonlocal_plan_prepare(
    generativeqc_context* context, const generativeqc_nonlocal_descriptor* model,
    generativeqc_nonlocal_plan** plan) {
  if (!context || !model || !plan) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  *plan = nullptr;
  if (!generativeqc::api::valid_descriptor(model)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  try {
    generativeqc_status status = GENERATIVEQC_STATUS_INTERNAL_ERROR;
    const generativeqc::dft::nlc::Vv10Parameters parameters{variant(model->variant), model->b,
                                                            model->c, model->coefficient};
    auto native = generativeqc::dft::nlc::Vv10Plan::prepare(
        context->state.executed_backend, context->state.device_id, model->point_count,
        model->tile_points, parameters, model->maximum_bytes, context->last_detail, status);
    if (!native) return status;
    auto owner = std::make_unique<generativeqc_nonlocal_plan>();
    owner->context = context;
    owner->plan = std::move(native);
    *plan = owner.release();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
}

void generativeqc_nonlocal_plan_destroy(generativeqc_nonlocal_plan* plan) { delete plan; }

generativeqc_status generativeqc_nonlocal_plan_get_diagnostic(
    const generativeqc_nonlocal_plan* plan, generativeqc_nonlocal_runtime_diagnostic* diagnostic) {
  if (!plan || !diagnostic) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!generativeqc::api::valid_descriptor(diagnostic)) return GENERATIVEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(plan->context->mutex);
  const auto& resources = plan->plan->resources();
  diagnostic->backend = plan->plan->backend();
  diagnostic->workspace_bytes = resources.workspace_bytes;
  diagnostic->host_workspace_bytes = resources.host_workspace_bytes;
  diagnostic->device_workspace_bytes = resources.device_workspace_bytes;
  diagnostic->maximum_bytes = resources.maximum_bytes;
  diagnostic->pair_evaluations = resources.pair_evaluations;
  diagnostic->tiles = resources.tiles;
  diagnostic->point_count = resources.point_count;
  diagnostic->tile_points = resources.tile_points;
  return GENERATIVEQC_STATUS_SUCCESS;
}

generativeqc_status generativeqc_nonlocal_plan_execute(
    generativeqc_nonlocal_plan* plan, const generativeqc_nonlocal_input_descriptor* input,
    generativeqc_nonlocal_result_descriptor* result) {
  if (!plan || !input || !result) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (!generativeqc::api::valid_descriptor(input) || !generativeqc::api::valid_descriptor(result))
    return GENERATIVEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(plan->context->mutex);
  try {
    const auto points = plan->plan->resources().point_count;
    if (!input->coordinates || !input->weights || !input->density || !input->density_gradient ||
        input->coordinate_count != 3u * points || input->weight_count != points ||
        input->density_count != points || input->density_gradient_count != 3u * points) {
      plan->context->last_detail = "VV10 input arrays do not match the prepared point count";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    auto vrho = optional_span(result->vrho, result->vrho_count, "vrho");
    auto vsigma = optional_span(result->vsigma, result->vsigma_count, "vsigma");
    auto point =
        optional_span(result->point_derivative, result->point_derivative_count, "point_derivative");
    auto weight = optional_span(result->weight_derivative, result->weight_derivative_count,
                                "weight_derivative");
    const bool publish_features = !vrho.empty() || !vsigma.empty();
    if (publish_features && (vrho.size() != points || vsigma.size() != points)) {
      plan->context->last_detail = "VV10 feature outputs do not match the prepared point count";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    const bool publish_geometry = !point.empty() || !weight.empty();
    if (publish_geometry && (point.size() != 3u * points || weight.size() != points)) {
      plan->context->last_detail = "VV10 geometry outputs do not match the prepared point count";
      return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
    }
    std::vector<double> staged_vrho(publish_features ? points : 0u);
    std::vector<double> staged_vsigma(publish_features ? points : 0u);
    std::vector<double> staged_point(publish_geometry ? 3u * points : 0u);
    std::vector<double> staged_weight(publish_geometry ? points : 0u);
    double energy{};
    const auto status = plan->plan->execute(
        std::span<const double>(input->coordinates, input->coordinate_count),
        std::span<const double>(input->weights, input->weight_count),
        std::span<const double>(input->density, input->density_count),
        std::span<const double>(input->density_gradient, input->density_gradient_count), energy,
        staged_vrho, staged_vsigma, staged_point, staged_weight, plan->context->last_detail);
    if (status != GENERATIVEQC_STATUS_SUCCESS) return status;
    if (publish_features) {
      std::copy(staged_vrho.begin(), staged_vrho.end(), vrho.begin());
      std::copy(staged_vsigma.begin(), staged_vsigma.end(), vsigma.begin());
    }
    if (publish_geometry) {
      std::copy(staged_point.begin(), staged_point.end(), point.begin());
      std::copy(staged_weight.begin(), staged_weight.end(), weight.begin());
    }
    result->energy = energy;
    result->executed_backend = plan->plan->backend();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&plan->context->last_detail);
  }
}

#if GENERATIVEQC_HAS_CUDA
/** Query the exact numeric device allocation without a CUDA context or work. */
GENERATIVEQC_API generativeqc_status generativeqc_internal_nonlocal_cuda_force_bytes_v1(
    std::uint32_t point_count, std::uint32_t tile_points, std::uint64_t* bytes) {
  if (!bytes) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  *bytes = 0;
  try {
    const auto layout =
        generativeqc::dft::nlc::vv10_cuda_device_layout(point_count, tile_points, true, true, true);
    *bytes = nonlocal_force_device_bytes(layout);
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(nullptr);
  }
}

GENERATIVEQC_API generativeqc_status generativeqc_internal_nonlocal_cuda_force_create_v1(
    generativeqc_context* context, const generativeqc_nonlocal_descriptor* model,
    const double* coordinates, std::size_t coordinate_count, const double* weights,
    std::size_t weight_count, double density_threshold, generativeqc_nonlocal_cuda_force** output) {
  if (output) *output = nullptr;
  if (!context || !model || !output || !coordinates || !weights ||
      !generativeqc::api::valid_descriptor(model))
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  try {
    if (context->state.executed_backend != GENERATIVEQC_BACKEND_CUDA ||
        context->state.device_id < 0)
      return GENERATIVEQC_STATUS_NOT_IMPLEMENTED;
    const auto n = static_cast<std::size_t>(model->point_count);
    if (!n || !model->tile_points || coordinate_count != 3 * n || weight_count != n ||
        !std::isfinite(density_threshold) || density_threshold <= 0.0)
      throw std::invalid_argument("invalid resident nonlocal CUDA force shape/policy");
    for (std::size_t i = 0; i < coordinate_count; ++i)
      if (!std::isfinite(coordinates[i]))
        throw std::invalid_argument("resident nonlocal CUDA coordinates must be finite");
    for (std::size_t i = 0; i < weight_count; ++i)
      if (!std::isfinite(weights[i]))
        throw std::invalid_argument("resident nonlocal CUDA weights must be finite");

    auto result = std::make_unique<generativeqc_nonlocal_cuda_force>();
    result->context = context;
    result->device = context->state.device_id;
    result->point_count = n;
    result->density_threshold = density_threshold;
    result->parameters = {variant(model->variant), model->b, model->c, model->coefficient};
    if (!std::isfinite(result->parameters.b) || result->parameters.b <= 0.0 ||
        !std::isfinite(result->parameters.c) || result->parameters.c <= 0.0 ||
        !std::isfinite(result->parameters.coefficient) || result->parameters.coefficient <= 0.0)
      throw std::invalid_argument("resident nonlocal CUDA parameters must be finite and positive");
    result->layout =
        generativeqc::dft::nlc::vv10_cuda_device_layout(n, model->tile_points, true, true, true);

    const auto workspace_doubles = result->layout.workspace_bytes / sizeof(double);
    const auto doubles = nonlocal_force_arena_doubles(result->layout);
    result->device_bytes = nonlocal_force_device_bytes(result->layout);
    if (!model->maximum_bytes || result->device_bytes > model->maximum_bytes)
      throw std::bad_alloc();

    generativeqc::runtime::CudaDeviceScope device(result->device);
    generativeqc::runtime::OwnedCudaStream setup(result->device);
    result->source_ready.create(result->device, cudaEventDisableTiming);
    result->arena.allocate(result->device, doubles);
    result->errors.allocate(result->device, 3);
    auto* cursor = result->arena.get();
    auto take = [&](std::size_t count) {
      auto* pointer = cursor;
      cursor += count;
      return pointer;
    };
    result->coordinates = take(3 * n);
    result->weights = take(n);
    result->raw_density = take(n);
    result->raw_gradient = take(3 * n);
    result->effective_weights = take(n);
    // MolecularV1 loads a point's rho/gradient before replacing those values.
    // Keep original weights separate: a reset may unscreen an earlier row.
    result->effective_density = result->raw_density;
    result->effective_gradient = result->raw_gradient;
    result->seeds = take(6 * n);
    // Local-scale construction is the last gradient reader. Ordered later
    // pair kernels may overwrite it; packing then reads this AoS panel into
    // the separate SoA seeds. Aliasing the seeds themselves would race.
    result->point_derivative = result->raw_gradient;
    result->workspace = take(workspace_doubles);
    if (cursor != result->arena.get() + doubles)
      throw std::logic_error("resident nonlocal CUDA force arena partition mismatch");

    generativeqc::runtime::cuda_resource_check(
        cudaMemcpyAsync(result->coordinates, coordinates, coordinate_count * sizeof(double),
                        cudaMemcpyHostToDevice, setup.get()));
    generativeqc::runtime::cuda_resource_check(
        cudaMemcpyAsync(result->weights, weights, weight_count * sizeof(double),
                        cudaMemcpyHostToDevice, setup.get()));
    generativeqc::runtime::cuda_resource_check(
        cudaMemsetAsync(result->errors.get(), 0, 3 * sizeof(int), setup.get()));
    setup.synchronize();
    *output = result.release();
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&context->last_detail);
  }
}

GENERATIVEQC_API void generativeqc_internal_nonlocal_cuda_force_destroy_v1(
    generativeqc_nonlocal_cuda_force* owner) {
  delete owner;
}

GENERATIVEQC_API generativeqc_status generativeqc_internal_nonlocal_cuda_force_collect_v1(
    generativeqc_nonlocal_cuda_force* owner, const generativeqc::dft::GridTaskView* view,
    std::size_t offset) {
  if (!owner || !view) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(owner->context->mutex);
  try {
    if (owner->executed || offset != owner->next_offset || !view->stream ||
        offset > owner->point_count || view->npoint > owner->point_count - offset)
      throw std::invalid_argument("invalid resident nonlocal CUDA feature tile order");
    if (!owner->stream)
      owner->stream = view->stream;
    else if (owner->stream != view->stream)
      throw std::invalid_argument("resident nonlocal CUDA feature stream changed");
    generativeqc::runtime::CudaDeviceScope device(owner->device);
    if (offset == 0)
      generativeqc::runtime::cuda_resource_check(
          cudaMemsetAsync(owner->errors.get(), 0, sizeof(int), owner->stream));
    generativeqc::dft::nlc::enqueue_vv10_collect_total_features_cuda(
        owner->stream, *view, offset, owner->point_count, owner->raw_density, owner->raw_gradient,
        owner->errors.get());
    owner->next_offset += view->npoint;
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&owner->context->last_detail);
  }
}

GENERATIVEQC_API generativeqc_status generativeqc_internal_nonlocal_cuda_force_seed_device_v1(
    generativeqc_nonlocal_cuda_force* owner, generativeqc_context* expected_context, int device,
    const double* density, const double* gradient, std::size_t point_count, void* source_stream_raw,
    const generativeqc::dft::GridTaskView* view) {
  if (!owner || !expected_context || !density || !gradient || !source_stream_raw || !view)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  if (owner->context != expected_context) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(owner->context->mutex);
  try {
    if (owner->executed || owner->next_offset != 0 || device != owner->device ||
        point_count != owner->point_count || !view->stream)
      throw std::invalid_argument("invalid resident nonlocal CUDA feature seed");
    if (!owner->stream)
      owner->stream = view->stream;
    else if (owner->stream != view->stream)
      throw std::invalid_argument("resident nonlocal CUDA feature stream changed");
    generativeqc::runtime::CudaDeviceScope scope(owner->device);
    const auto source_stream = reinterpret_cast<cudaStream_t>(source_stream_raw);
    generativeqc::runtime::cuda_resource_check(
        cudaMemsetAsync(owner->errors.get(), 0, sizeof(int), owner->stream));
    // Read the final KS feature generation on its producer stream. A later KS
    // begin/teardown therefore cannot overtake these D2D reads. When the grid
    // consumer uses another stream, bridge the dependency with one device event
    // rather than a host fence; owner teardown synchronizes the consumer stream,
    // which in turn waits for this source-side copy to complete.
    try {
      generativeqc::runtime::cuda_resource_check(
          cudaMemcpyAsync(owner->raw_density, density, point_count * sizeof(double),
                          cudaMemcpyDeviceToDevice, source_stream));
      generativeqc::runtime::cuda_resource_check(
          cudaMemcpyAsync(owner->raw_gradient, gradient, 3 * point_count * sizeof(double),
                          cudaMemcpyDeviceToDevice, source_stream));
      if (source_stream != owner->stream) {
        owner->source_ready.record(source_stream);
        generativeqc::runtime::cuda_resource_check(
            cudaStreamWaitEvent(owner->stream, owner->source_ready.get(), 0));
      }
    } catch (...) {
      // A failed cross-stream handoff may already have queued a D2D read into
      // owner-owned destination storage. Drain only on this exceptional path
      // so owner teardown cannot free that storage while the source stream is
      // still writing it. Successful execution remains fence-free on the host.
      drain_failed_seed_source(source_stream);
      throw;
    }
    owner->next_offset = owner->point_count;
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&owner->context->last_detail);
  }
}

GENERATIVEQC_API generativeqc_status
generativeqc_internal_nonlocal_cuda_force_execute_v1(generativeqc_nonlocal_cuda_force* owner) {
  if (!owner) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(owner->context->mutex);
  try {
    if (owner->executed || !owner->stream || owner->next_offset != owner->point_count)
      throw std::invalid_argument("resident nonlocal CUDA force owner is not fully populated");
    generativeqc::runtime::CudaDeviceScope device(owner->device);
    auto* errors = owner->errors.get();
    generativeqc::dft::nlc::enqueue_vv10_molecular_domain_cuda(
        owner->stream, owner->point_count, owner->density_threshold, owner->weights,
        owner->raw_density, owner->raw_gradient, owner->effective_weights, owner->effective_density,
        owner->effective_gradient, errors + 1);
    generativeqc::dft::nlc::enqueue_vv10_cuda_device(
        owner->layout, owner->parameters, owner->device, owner->stream, owner->coordinates,
        owner->effective_weights, owner->effective_density, owner->effective_gradient,
        owner->workspace, owner->layout.workspace_bytes, owner->workspace, owner->seeds,
        owner->seeds + owner->point_count, owner->point_derivative,
        owner->seeds + 5 * owner->point_count, errors + 2);
    generativeqc::dft::nlc::enqueue_vv10_pack_force_seeds_cuda(
        owner->stream, owner->point_count, owner->effective_weights, owner->point_derivative,
        owner->seeds, errors, errors + 1, errors + 2);
    owner->executed = true;
    ++owner->generation;
    return GENERATIVEQC_STATUS_SUCCESS;
  } catch (...) {
    return generativeqc::api::map_exception(&owner->context->last_detail);
  }
}

GENERATIVEQC_API generativeqc_status generativeqc_internal_nonlocal_cuda_force_seed_view_v1(
    const generativeqc_nonlocal_cuda_force* owner, void** seeds, std::size_t* stride, void** stream,
    std::uint64_t* generation) {
  if (!owner || !seeds || !stride || !stream || !generation)
    return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(owner->context->mutex);
  if (!owner->executed || !owner->stream) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  *seeds = owner->seeds;
  *stride = owner->point_count;
  *stream = reinterpret_cast<void*>(owner->stream);
  *generation = owner->generation;
  return GENERATIVEQC_STATUS_SUCCESS;
}

GENERATIVEQC_API generativeqc_status
generativeqc_internal_nonlocal_cuda_force_reset_v1(generativeqc_nonlocal_cuda_force* owner) {
  if (!owner) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(owner->context->mutex);
  if (!owner->executed) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  owner->next_offset = 0;
  owner->executed = false;
  return GENERATIVEQC_STATUS_SUCCESS;
}

GENERATIVEQC_API generativeqc_status generativeqc_internal_nonlocal_cuda_force_metrics_v1(
    const generativeqc_nonlocal_cuda_force* owner, std::uint64_t* values, std::size_t count) {
  if (!owner || !values || count != 6) return GENERATIVEQC_STATUS_INVALID_ARGUMENT;
  std::lock_guard<std::recursive_mutex> lock(owner->context->mutex);
  const std::array<std::uint64_t, 6> metrics{
      owner->device_bytes, owner->point_count,        owner->next_offset,
      owner->generation,   owner->executed ? 1u : 0u, owner->stream ? 1u : 0u,
  };
  std::copy(metrics.begin(), metrics.end(), values);
  return GENERATIVEQC_STATUS_SUCCESS;
}
#endif

}  // extern "C"
