// Version 1 resident TensorIR span operations over the existing Context owner.
//
// Built with -I/32 so every span index, element count and division below uses
// 32-bit signed arithmetic. Guarded callers are also checked. Each span is
// (byte_offset, byte_count): both are emitted to match exactly one named slot,
// which is the only protocol a caller must not modify. No function here
// allocates, creates handles or reorders independent plans; plain transfer and
// copy synchronize with the host before returning.
#pragma once
#include "cuda_runtime.cuh"

namespace vibeqc_resident {
using namespace vibeqc_tensor;
struct Span {
  size_t offset, bytes;
};

inline const Span& checked(const Span* spans, size_t count, size_t index, size_t bytes) {
  if (index >= count || spans[index].bytes != bytes)
    throw std::invalid_argument("resident span index/length mismatch");
  return spans[index];
}

inline int transfer(void* pointer, const Span* spans, size_t count, size_t slot, void* host,
                    size_t bytes, bool upload, char* error, size_t size) {
  if (!pointer || (!host && bytes)) {
    error_text(error, size, "null resident transfer");
    return 1;
  }
  auto& ctx = *static_cast<Context*>(pointer);
  std::unique_lock<std::mutex> lock(ctx.mutex, std::try_to_lock);
  if (!lock.owns_lock()) {
    error_text(error, size, "resident plan is busy");
    return 1;
  }
  try {
    ctx.check_device();
    const auto span = checked(spans, count, slot, bytes);
    if (span.offset > ctx.metrics.owned_device_bytes ||
        bytes > ctx.metrics.owned_device_bytes - span.offset)
      throw std::invalid_argument("resident span exceeds allocation");
    if (bytes) {
      cuda_check(cudaMemcpyAsync(
          upload ? ctx.arena + span.offset : host, upload ? host : ctx.arena + span.offset, bytes,
          upload ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost, ctx.stream));
    }
    cuda_check(cudaStreamSynchronize(ctx.stream));
    return 0;
  } catch (const std::exception& e) {
    cudaStreamSynchronize(ctx.stream);
    error_text(error, size, e.what());
    return 1;
  }
}

inline int copy(void* source, const Span* source_spans, size_t source_count, size_t source_slot,
                void* target, const Span* target_spans, size_t target_count, size_t target_slot,
                size_t bytes, char* error, size_t size) {
  if (!source || !target) {
    error_text(error, size, "null resident copy owner");
    return 1;
  }
  auto& src = *static_cast<Context*>(source);
  auto& dst = *static_cast<Context*>(target);
  std::unique_lock<std::mutex> first(src.mutex, std::defer_lock);
  std::unique_lock<std::mutex> second(dst.mutex, std::defer_lock);
  if (source == target)
    first.lock();
  else
    std::lock(first, second);
  try {
    // While both locks are held no other resident operation can run on either
    // owner; ordinary tensor_run calls only lock their own owner, so a
    // cross-plan copy is conservative but always safe.
    src.check_device();
    dst.check_device();
    if (src.device != dst.device) throw std::invalid_argument("resident device mismatch");
    const auto from = checked(source_spans, source_count, source_slot, bytes);
    const auto to = checked(target_spans, target_count, target_slot, bytes);
    if (from.offset > src.metrics.owned_device_bytes ||
        bytes > src.metrics.owned_device_bytes - from.offset ||
        to.offset > dst.metrics.owned_device_bytes ||
        bytes > dst.metrics.owned_device_bytes - to.offset)
      throw std::invalid_argument("resident span exceeds allocation");
    if (source == target && from.offset != to.offset && from.offset < to.offset + bytes &&
        to.offset < from.offset + bytes)
      throw std::invalid_argument("overlapping resident spans");
    if (source != target) cuda_check(cudaStreamSynchronize(src.stream));
    if (bytes && (source != target || from.offset != to.offset))
      cuda_check(cudaMemcpyAsync(dst.arena + to.offset, src.arena + from.offset, bytes,
                                 cudaMemcpyDeviceToDevice, dst.stream));
    cuda_check(cudaStreamSynchronize(dst.stream));
    return 0;
  } catch (const std::exception& e) {
    cudaStreamSynchronize(dst.stream);
    error_text(error, size, e.what());
    return 1;
  }
}
}  // namespace vibeqc_resident
