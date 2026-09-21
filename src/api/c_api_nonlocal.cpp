#include <memory>
#include <mutex>
#include <span>

#include "api/error.hpp"
#include "api/handles.hpp"
#include "dft/nonlocal_correlation/vv10_runtime.hpp"
#include "vibeqc/vibeqc.h"

struct vibeqc_nonlocal_plan {
  vibeqc_context* context{};
  std::unique_ptr<vibeqc::dft::nlc::Vv10CpuPlan> plan;
};

namespace {

vibeqc::dft::nlc::Vv10Variant variant(vibeqc_nonlocal_variant value) {
  using vibeqc::dft::nlc::Vv10Variant;
  if (value == VIBEQC_NONLOCAL_VV10) return Vv10Variant::vv10;
  if (value == VIBEQC_NONLOCAL_RVV10) return Vv10Variant::rvv10;
  throw std::invalid_argument("unsupported VV10/rVV10 kernel variant");
}

template <class T>
std::span<T> optional_span(T* pointer, std::uint32_t count, const char* label) {
  if ((pointer == nullptr) != (count == 0))
    throw std::invalid_argument(std::string(label) + " pointer/count disagree");
  return pointer ? std::span<T>(pointer, count) : std::span<T>{};
}

}  // namespace

extern "C" {

vibeqc_status vibeqc_nonlocal_plan_prepare(vibeqc_context* context,
                                           const vibeqc_nonlocal_descriptor* model,
                                           vibeqc_nonlocal_plan** plan) {
  if (!context || !model || !plan) return VIBEQC_STATUS_INVALID_ARGUMENT;
  *plan = nullptr;
  if (!vibeqc::api::valid_descriptor(model)) return VIBEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(context->mutex);
  try {
    vibeqc_status status = VIBEQC_STATUS_INTERNAL_ERROR;
    const vibeqc::dft::nlc::Vv10Parameters parameters{variant(model->variant), model->b, model->c,
                                                      model->coefficient};
    auto native = vibeqc::dft::nlc::Vv10CpuPlan::prepare(
        context->state.executed_backend, model->point_count, model->tile_points, parameters,
        model->maximum_bytes, context->last_detail, status);
    if (!native) return status;
    auto owner = std::make_unique<vibeqc_nonlocal_plan>();
    owner->context = context;
    owner->plan = std::move(native);
    *plan = owner.release();
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&context->last_detail);
  }
}

void vibeqc_nonlocal_plan_destroy(vibeqc_nonlocal_plan* plan) { delete plan; }

vibeqc_status vibeqc_nonlocal_plan_get_diagnostic(const vibeqc_nonlocal_plan* plan,
                                                  vibeqc_nonlocal_runtime_diagnostic* diagnostic) {
  if (!plan || !diagnostic) return VIBEQC_STATUS_INVALID_ARGUMENT;
  if (!vibeqc::api::valid_descriptor(diagnostic)) return VIBEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(plan->context->mutex);
  const auto& resources = plan->plan->resources();
  diagnostic->backend = plan->plan->backend();
  diagnostic->workspace_bytes = resources.workspace_bytes;
  diagnostic->maximum_bytes = resources.maximum_bytes;
  diagnostic->pair_evaluations = resources.pair_evaluations;
  diagnostic->point_count = resources.point_count;
  diagnostic->tile_points = resources.tile_points;
  return VIBEQC_STATUS_SUCCESS;
}

vibeqc_status vibeqc_nonlocal_plan_execute(vibeqc_nonlocal_plan* plan,
                                           const vibeqc_nonlocal_input_descriptor* input,
                                           vibeqc_nonlocal_result_descriptor* result) {
  if (!plan || !input || !result) return VIBEQC_STATUS_INVALID_ARGUMENT;
  if (!vibeqc::api::valid_descriptor(input) || !vibeqc::api::valid_descriptor(result))
    return VIBEQC_STATUS_ABI_MISMATCH;
  std::lock_guard<std::recursive_mutex> lock(plan->context->mutex);
  try {
    const auto points = plan->plan->resources().point_count;
    if (!input->coordinates || !input->weights || !input->density || !input->density_gradient ||
        input->coordinate_count != 3u * points || input->weight_count != points ||
        input->density_count != points || input->density_gradient_count != 3u * points) {
      plan->context->last_detail = "VV10 input arrays do not match the prepared point count";
      return VIBEQC_STATUS_INVALID_ARGUMENT;
    }
    auto vrho = optional_span(result->vrho, result->vrho_count, "vrho");
    auto vsigma = optional_span(result->vsigma, result->vsigma_count, "vsigma");
    auto point =
        optional_span(result->point_derivative, result->point_derivative_count, "point_derivative");
    auto weight = optional_span(result->weight_derivative, result->weight_derivative_count,
                                "weight_derivative");
    double energy{};
    const auto status = plan->plan->execute(
        std::span<const double>(input->coordinates, input->coordinate_count),
        std::span<const double>(input->weights, input->weight_count),
        std::span<const double>(input->density, input->density_count),
        std::span<const double>(input->density_gradient, input->density_gradient_count), energy,
        vrho, vsigma, point, weight, plan->context->last_detail);
    if (status != VIBEQC_STATUS_SUCCESS) return status;
    result->energy = energy;
    result->executed_backend = plan->plan->backend();
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return vibeqc::api::map_exception(&plan->context->last_detail);
  }
}

}  // extern "C"
