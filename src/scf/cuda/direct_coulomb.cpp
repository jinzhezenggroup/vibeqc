#include "scf/cuda/direct_coulomb.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "runtime/bounded_workspace.hpp"
#include "runtime/cuda_target_info.hpp"
#include "runtime/resource_cuda.cuh"
#include "runtime/resource_usage.hpp"
#include "scf/aot_shell_registry.hpp"
#include "scf/cuda/basis_transform_kernels.hpp"
#include "scf/cuda/df_jk_kernels.hpp"
#include "scf/cuda/direct_bounded_dddd.hpp"
#include "scf/cuda/direct_constants.hpp"
#include "scf/cuda/direct_density_bounds.hpp"
#include "scf/cuda/direct_jk_kernels.hpp"
#include "scf/cuda/direct_pair_cache.hpp"
#include "scf/cuda/direct_schwarz_kernels.hpp"
#include "scf/cuda/metadata_upload.hpp"
#include "scf/cuda/queue_plan.hpp"

namespace generativeqc::scf::cuda_execution {
namespace {
void check(cudaError_t error) {
  if (error != cudaSuccess) throw error;
}
std::size_t product(std::size_t a, std::size_t b) { return runtime::size_mul(a, b); }
unsigned blocks(std::size_t elements) { return static_cast<unsigned>((elements + 127) / 128); }
}  // namespace

GeneratedCoulombPlan::~GeneratedCoulombPlan() {
  // The outer provider still owns this stream and all borrowed geometry.
  if (stream) (void)cudaStreamSynchronize(stream);
  for (void* pointer : allocations) (void)runtime::resource_cuda_free(pointer);
}

std::unique_ptr<GeneratedCoulombPlan> prepare_generated_coulomb(
    const HostBatch& host, DeviceBatch borrowed, cudaStream_t stream, int device, double screening,
    std::size_t budget, bool allow_bounded_shell_fallback,
    detail::BoundedDirectHostSchedule* bounded_schedule) try {
  cudaDeviceProp properties{};
  check(cudaGetDeviceProperties(&properties, device));
  generated::select_profile_for_device(device, properties.major, properties.minor);
  const auto schedule = cuda_policy::resolve_direct_jk_schedule_policy(
      runtime::cuda_target_info_from_properties(properties));
  const auto present = present_direct_shell_class_mask(host);
  const auto generated_mask =
      generated::enabled_fock_shell_class_mask() & kGeneratedStreamingFockShellClassMask;
  const auto value_mask = generated_mask | kNativeStreamingFockShellClassMask;
  const auto value_class_mask = present & value_mask;
  const bool value_capability = value_class_mask == present;
  if (!value_capability && !allow_bounded_shell_fallback) return {};
  std::vector<std::uint32_t> order, offsets;
  if (!make_bounded_stream_shell_pair_order(host, order, offsets)) return {};
  const std::size_t batch = borrowed.batch_size;
  const auto matrix = product(product(batch, host.nbf), host.nbf);
  const auto cartesian = product(product(batch, host.direct_nbf), host.direct_nbf);
  const auto rectangular = product(product(batch, host.nbf), host.direct_nbf);
  // The HF packer omits C for Cartesian and tiny persistent-ERI systems.
  // Reconstruct only that metadata from its existing normalized AO expansion;
  // no basis/recurrence equations are introduced by this runtime adapter.
  std::vector<double> expanded_transform;
  const auto* transform = &host.ao_to_direct_transform;
  if (transform->empty()) {
    expanded_transform.assign(rectangular, 0.0);
    for (std::size_t ao = 0; ao < batch * host.nbf; ++ao) {
      const auto shell = host.ao_shells[ao];
      const auto system = ao / host.nbf;
      for (unsigned term = 0; term < host.ao_term_counts[ao]; ++term) {
        const auto term_index = ao * 3 + term;
        for (auto source = host.shell_direct_ao_offsets[shell];
             source < host.shell_direct_ao_offsets[shell + 1]; ++source) {
          if (!std::equal(host.ao_term_angular.begin() + term_index * 3,
                          host.ao_term_angular.begin() + term_index * 3 + 3,
                          host.direct_ao_angular.begin() + source * 3))
            continue;
          const auto column = static_cast<std::size_t>(source) - system * host.direct_nbf;
          expanded_transform[system * host.nbf * host.direct_nbf + ao % host.nbf +
                             column * host.nbf] =
              host.ao_term_coefficients[term_index] / host.direct_ao_coefficients[source];
        }
      }
    }
    transform = &expanded_transform;
  }
  const auto pairs = host.shell_pair_first.size();
  const auto primitive_pairs = static_cast<std::size_t>(host.shell_pair_primitive_offsets.back());
  const auto classes = detail::kDirectQuartetShellClassCount;
  std::size_t required = 0;
  const auto charge = [&](std::size_t count, std::size_t width) {
    required = runtime::size_add(required, product(count, width));
  };
#define COULOMB_METADATA(F)        \
  F(system_shell_offsets);         \
  F(system_shell_pair_offsets);    \
  F(shell_ao_offsets);             \
  F(shell_direct_ao_offsets);      \
  F(shell_pair_systems);           \
  F(shell_pair_first);             \
  F(shell_pair_second);            \
  F(shell_pair_primitive_offsets); \
  F(direct_ao_shells);             \
  F(direct_ao_angular);            \
  F(direct_ao_coefficients)
#define COUNT(field) charge(host.field.size(), sizeof(host.field[0]))
  COULOMB_METADATA(COUNT);
#undef COUNT
  charge(rectangular, sizeof(double));  // C is also present for identity transforms.
  charge(order.size() + offsets.size(), sizeof(std::uint32_t));
  charge(primitive_pairs, sizeof(PrimitivePairData));
  charge(cartesian, 3 * sizeof(double));  // Density, J and Cartesian Schwarz.
  charge(rectangular, sizeof(double));
  charge(matrix, 2 * sizeof(double));  // Total spin density and zero one-electron term.
  charge(pairs, sizeof(double));
  charge(batch, sizeof(std::uint8_t));
  charge(classes, sizeof(std::uint32_t));
  charge(1, sizeof(GeneratedShellPairStream));
  if (required > budget || cartesian > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    return {};
  auto plan = std::make_unique<GeneratedCoulombPlan>();
  plan->batch = borrowed;
  plan->batch.total_shell_pairs = pairs;
  plan->stream = stream;
  plan->screening = screening;
  plan->class_mask = present;
  plan->value_class_mask = value_class_mask;
  plan->value_capability = value_capability;
  // Reuse the target-legal HF worker and recurrence-stack policy; a prepared
  // KS owner must not rely on an earlier HF call having raised the CUDA limit.
  plan->worker_blocks = static_cast<unsigned>(properties.multiProcessorCount) *
                        schedule.persistent_quartet_warps_per_sm;
  try {
    // Growing the optional recurrence stack can reserve device memory too.
    // Keep that failure on the same bounded fallback path as explicit buffers.
    std::size_t stack_limit = 0;
    check(cudaDeviceGetLimit(&stack_limit, cudaLimitStackSize));
    if (stack_limit < schedule.cuda_stack_limit_bytes)
      check(cudaDeviceSetLimit(cudaLimitStackSize, schedule.cuda_stack_limit_bytes));
    auto allocate = [&](std::size_t count, std::size_t width, const void* values = nullptr) {
      void* pointer{};
      const auto bytes = product(count, width);
      check(runtime::resource_cuda_malloc(&pointer, bytes));
      try {
        plan->allocations.push_back(pointer);
      } catch (...) {
        (void)runtime::resource_cuda_free(pointer);
        throw;
      }
      plan->device_bytes += bytes;
      if (values) check(cudaMemcpyAsync(pointer, values, bytes, cudaMemcpyHostToDevice, stream));
      return pointer;
    };
#define UPLOAD(field)                                           \
  plan->batch.field = static_cast<decltype(plan->batch.field)>( \
      allocate(host.field.size(), sizeof(host.field[0]), host.field.data()))
    COULOMB_METADATA(UPLOAD);
#undef UPLOAD
#undef COULOMB_METADATA
    plan->batch.ao_to_direct_transform =
        static_cast<const double*>(allocate(transform->size(), sizeof(double), transform->data()));
    auto* primitive_cache =
        static_cast<PrimitivePairData*>(allocate(primitive_pairs, sizeof(PrimitivePairData)));
    plan->batch.shell_primitive_pairs = primitive_cache;
    auto doubles = [&](std::size_t count) {
      return static_cast<double*>(allocate(count, sizeof(double)));
    };
    plan->density = doubles(cartesian);
    plan->coulomb = doubles(cartesian);
    plan->schwarz = doubles(cartesian);
    plan->temporary = doubles(rectangular);
    plan->total_density = doubles(matrix);
    plan->zero = doubles(matrix);
    plan->shell_bounds = doubles(pairs);
    plan->active = static_cast<std::uint8_t*>(allocate(batch, sizeof(std::uint8_t)));
    plan->heads = static_cast<std::uint32_t*>(allocate(classes, sizeof(std::uint32_t)));
    check(cudaMemsetAsync(plan->active, 1, batch, stream));
    check(cudaMemsetAsync(plan->zero, 0, matrix * sizeof(double), stream));
    check(cudaMemsetAsync(plan->shell_bounds, 0, pairs * sizeof(double), stream));
    launch_build_shell_primitive_pair_cache_kernel(static_cast<unsigned>(pairs),
                                                   detail::kDirectQuartetThreads, 0, stream,
                                                   plan->batch, primitive_cache);
    check(cudaGetLastError());
    const auto cartesian_pairs = host.direct_nbf * (host.direct_nbf + 1) / 2;
    launch_build_schwarz_and_shell_pair_bounds_packed_kernel(
        static_cast<unsigned>(batch * cartesian_pairs), kSchwarzThreads, 0, stream, plan->batch,
        cartesian_pairs, plan->schwarz, plan->shell_bounds);
    check(cudaGetLastError());
    std::vector<double> bounds(pairs);
    check(cudaMemcpyAsync(bounds.data(), plan->shell_bounds, pairs * sizeof(double),
                          cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    for (double bound : bounds)
      if (!std::isfinite(bound)) throw std::runtime_error("nonfinite generated J Schwarz bound");
    // Reuse the required geometry readback, not a new force-time download.
    if (bounded_schedule)
      *bounded_schedule =
          detail::make_bounded_direct_schedule(host.system_shell_pair_offsets, bounds, screening);
    // Descending bounds are a correctness precondition for generated ket-tail
    // termination, not just a performance ordering. Sort every class/system.
    for (std::size_t cls = 0; cls < detail::kDirectShellPairClassCount; ++cls)
      for (std::size_t item = 0; item < batch; ++item) {
        const auto index = cls * (batch + 1) + item;
        std::stable_sort(order.begin() + offsets[index], order.begin() + offsets[index + 1],
                         [&](auto a, auto b) { return bounds[a] > bounds[b]; });
      }
    const auto* device_order =
        static_cast<const std::uint32_t*>(allocate(order.size(), sizeof(order[0]), order.data()));
    const auto* device_offsets = static_cast<const std::uint32_t*>(
        allocate(offsets.size(), sizeof(offsets[0]), offsets.data()));
    plan->pair_order = device_order;
    plan->pair_class_offsets = device_offsets;
    const auto& b = plan->batch;
    const GeneratedShellPairStream topology{b.batch_size,
                                            static_cast<std::uint32_t>(b.direct_nbf),
                                            b.system_shell_offsets,
                                            b.system_shell_pair_offsets,
                                            b.shell_atoms,
                                            b.shell_angular,
                                            b.shell_direct_ao_offsets,
                                            b.shell_primitive_offsets,
                                            b.shell_pair_systems,
                                            b.shell_pair_first,
                                            b.shell_pair_second,
                                            plan->pair_order,
                                            plan->pair_class_offsets,
                                            plan->shell_bounds,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            plan->active,
                                            detail::GeneratedFockConsumer::Coulomb};
    plan->topology =
        static_cast<GeneratedShellPairStream*>(allocate(1, sizeof(topology), &topology));
    check(cudaStreamSynchronize(stream));
    if (plan->device_bytes != required) throw std::logic_error("generated J inventory drift");
    plan->host_preparation_bytes =
        sizeof(*plan) +
        runtime::vector_capacities(plan->allocations, order, offsets, bounds, expanded_transform);
    if (bounded_schedule)
      plan->host_preparation_bytes = runtime::size_add(
          plan->host_preparation_bytes,
          runtime::vector_capacities(bounded_schedule->pair_order, bounded_schedule->block_prefix));
    return plan;
  } catch (cudaError_t error) {
    if (error != cudaErrorMemoryAllocation) throw;
    // The optional owner drains and releases itself before the generic path
    // resumes. A failed allocation must not poison later launch checks.
    (void)cudaGetLastError();
    return {};
  } catch (const std::bad_alloc&) {
    return {};
  }
} catch (const std::bad_alloc&) {
  // Host topology, transform and owner allocations are optional too.
  return {};
}

GeneratedExchangePlan::~GeneratedExchangePlan() {
  if (shared && shared->stream) (void)cudaStreamSynchronize(shared->stream);
  for (void* pointer : allocations) (void)runtime::resource_cuda_free(pointer);
}

std::unique_ptr<GeneratedExchangePlan> prepare_generated_exchange(
    const HostBatch& host, DeviceBatch borrowed, cudaStream_t stream, int device, double screening,
    std::size_t budget, bool force_capability, bool allow_bounded_shell_fallback) try {
  detail::BoundedDirectHostSchedule bounded_schedule;
  const bool indexed = force_capability && cuda_policy::bounded_schwarz_schedule_requested();
  auto shared = prepare_generated_coulomb(host, borrowed, stream, device, screening, budget,
                                          allow_bounded_shell_fallback,
                                          indexed ? &bounded_schedule : nullptr);
  if (!shared) return {};
  const bool bounded_value_capability = allow_bounded_shell_fallback && !shared->value_capability;
  const bool bounded_resources = force_capability || bounded_value_capability;

  const std::size_t batch = static_cast<std::size_t>(borrowed.batch_size);
  const auto matrix = product(product(batch, host.nbf), host.nbf);
  const auto cartesian = product(product(batch, host.direct_nbf), host.direct_nbf);
  const auto rectangular = product(product(batch, host.nbf), host.direct_nbf);
  const auto pairs = host.shell_pair_first.size();
  constexpr std::size_t pair_classes = detail::kDirectShellPairClassCount;
  constexpr std::size_t quartet_classes = detail::kDirectQuartetShellClassCount;
  const auto atoms = host.atomic_numbers.size();
  const auto pair_blocks = static_cast<std::size_t>(host.system_shell_pair_block_offsets.back());
  if (bounded_resources && pair_blocks > std::numeric_limits<unsigned>::max()) return {};

  std::size_t additional = 0;
  const auto charge = [&](std::size_t count, std::size_t width) {
    additional = runtime::size_add(additional, product(count, width));
  };
  charge(product(2, matrix), sizeof(double));       // Interleaved public alpha/beta input.
  charge(product(2, cartesian), sizeof(double));    // Direct alpha/beta density.
  charge(product(2, cartesian), sizeof(double));    // Direct raw K output.
  charge(product(2, rectangular), sizeof(double));  // Density transform scratch.
  charge(product(2, rectangular), sizeof(double));  // Fock transform scratch.
  charge(product(2, matrix), sizeof(double));       // Interleaved public raw K.
  charge(pairs, sizeof(ShellPairDensityBounds));
  charge(batch, sizeof(double));
  charge(product(batch, pair_classes), sizeof(double));
  charge(quartet_classes, sizeof(std::uint32_t));
  charge(1, sizeof(GeneratedShellPairStream));
  if (bounded_resources) {
    charge(pairs, sizeof(std::uint32_t));
    charge(pair_blocks, sizeof(double));
    charge(1, sizeof(unsigned long long));
    charge(host.system_shell_pair_block_offsets.size(), sizeof(std::int64_t));
    charge(host.system_shell_pair_block_quartet_offsets.size(), sizeof(std::int64_t));
  }
  if (force_capability) charge(product(atoms, 9), sizeof(double));
  if (bounded_value_capability) charge(quartet_classes, sizeof(std::uint32_t));
  if (shared->device_bytes > budget || additional > budget - shared->device_bytes) return {};
  if (!bounded_schedule.block_prefix.empty()) {
    const auto prefix_bytes = product(bounded_schedule.block_prefix.size(), sizeof(std::uint64_t));
    // A tight lease can still use sorted blocks with the old triangular index.
    if (prefix_bytes <= budget - shared->device_bytes - additional)
      charge(bounded_schedule.block_prefix.size(), sizeof(std::uint64_t));
    else
      bounded_schedule.block_prefix.clear();
  }

  // The owner drains H2D on failed preparation before this staging is freed.
  std::vector<std::uint32_t> bounded_pair_order;
  auto plan = std::make_unique<GeneratedExchangePlan>();
  plan->shared = std::move(shared);
  plan->force_capability = force_capability;
  plan->bounded_value_capability = bounded_value_capability;
  plan->device_bytes = plan->shared->device_bytes;
  auto allocate = [&](std::size_t count, std::size_t width, const void* values = nullptr) {
    void* pointer{};
    const auto bytes = product(count, width);
    check(runtime::resource_cuda_malloc(&pointer, bytes));
    try {
      plan->allocations.push_back(pointer);
    } catch (...) {
      (void)runtime::resource_cuda_free(pointer);
      throw;
    }
    plan->device_bytes = runtime::size_add(plan->device_bytes, bytes);
    if (values) check(cudaMemcpyAsync(pointer, values, bytes, cudaMemcpyHostToDevice, stream));
    return pointer;
  };
  auto doubles = [&](std::size_t count) {
    return static_cast<double*>(allocate(count, sizeof(double)));
  };
  plan->public_spin = doubles(product(2, matrix));
  plan->direct_spin = doubles(product(2, cartesian));
  plan->direct_exchange = doubles(product(2, cartesian));
  plan->density_temporary = doubles(product(2, rectangular));
  plan->fock_temporary = doubles(product(2, rectangular));
  plan->public_exchange = doubles(product(2, matrix));
  plan->shell_pair_density_bounds =
      static_cast<ShellPairDensityBounds*>(allocate(pairs, sizeof(ShellPairDensityBounds)));
  plan->system_density_bounds = doubles(batch);
  plan->system_pair_density_bounds = doubles(product(batch, pair_classes));
  plan->heads = static_cast<std::uint32_t*>(allocate(quartet_classes, sizeof(std::uint32_t)));
  if (bounded_resources) {
    if (indexed)
      bounded_pair_order = std::move(bounded_schedule.pair_order);
    else {
      bounded_pair_order.resize(pairs);
      std::iota(bounded_pair_order.begin(), bounded_pair_order.end(), 0U);
    }
    plan->bounded_pair_order = static_cast<const std::uint32_t*>(
        allocate(pairs, sizeof(std::uint32_t), bounded_pair_order.data()));
    if (!bounded_schedule.block_prefix.empty()) {
      if (bounded_schedule.block_prefix.size() != pair_blocks + 1U)
        throw std::logic_error("bounded Schwarz row inventory drift");
      plan->bounded_block_domain = {static_cast<const std::uint64_t*>(allocate(
                                        bounded_schedule.block_prefix.size(), sizeof(std::uint64_t),
                                        bounded_schedule.block_prefix.data())),
                                    pair_blocks, bounded_schedule.block_prefix.back()};
    }
    plan->shell_pair_block_bounds = doubles(pair_blocks);
    plan->force_cursor = static_cast<unsigned long long*>(allocate(1, sizeof(unsigned long long)));
    plan->shared->batch.total_shell_pair_blocks = host.system_shell_pair_block_offsets.back();
    plan->shared->batch.total_shell_pair_block_quartets =
        host.system_shell_pair_block_quartet_offsets.back();
    plan->shared->batch.system_shell_pair_block_offsets = static_cast<const std::int64_t*>(
        allocate(host.system_shell_pair_block_offsets.size(), sizeof(std::int64_t),
                 host.system_shell_pair_block_offsets.data()));
    plan->shared->batch.system_shell_pair_block_quartet_offsets = static_cast<const std::int64_t*>(
        allocate(host.system_shell_pair_block_quartet_offsets.size(), sizeof(std::int64_t),
                 host.system_shell_pair_block_quartet_offsets.data()));
  }
  if (force_capability) plan->force = doubles(product(atoms, 9));
  if (bounded_value_capability) {
    plan->bounded_value_overflow =
        static_cast<std::uint32_t*>(allocate(quartet_classes, sizeof(std::uint32_t)));
    check(cudaMemsetAsync(plan->bounded_value_overflow, 0, quartet_classes * sizeof(std::uint32_t),
                          stream));
  }

  const auto& b = plan->shared->batch;
  const GeneratedShellPairStream topology{
      b.batch_size,
      static_cast<std::uint32_t>(b.direct_nbf),
      b.system_shell_offsets,
      b.system_shell_pair_offsets,
      b.shell_atoms,
      b.shell_angular,
      b.shell_direct_ao_offsets,
      b.shell_primitive_offsets,
      b.shell_pair_systems,
      b.shell_pair_first,
      b.shell_pair_second,
      plan->shared->pair_order,
      plan->shared->pair_class_offsets,
      plan->shared->shell_bounds,
      reinterpret_cast<const detail::GeneratedShellPairDensityBounds*>(
          plan->shell_pair_density_bounds),
      plan->system_density_bounds,
      plan->system_pair_density_bounds,
      nullptr,
      plan->shared->active,
      detail::GeneratedFockConsumer::Exchange};
  plan->topology = static_cast<GeneratedShellPairStream*>(allocate(1, sizeof(topology), &topology));
  if (bounded_resources) {
    launch_reduce_bounded_shell_pair_block_bounds_kernel(
        static_cast<unsigned>(pair_blocks), 128U, 128U * sizeof(double), stream, b,
        plan->bounded_pair_order, plan->shared->shell_bounds, plan->shell_pair_block_bounds);
    check(cudaGetLastError());
  }
  check(cudaStreamSynchronize(stream));
  if (plan->device_bytes != runtime::size_add(plan->shared->device_bytes, additional))
    throw std::logic_error("generated K inventory drift");
  plan->host_preparation_bytes = runtime::size_add(
      plan->shared->host_preparation_bytes,
      sizeof(*plan) + runtime::vector_capacities(plan->allocations, bounded_pair_order,
                                                 bounded_schedule.block_prefix));
  return plan;
} catch (cudaError_t error) {
  if (error != cudaErrorMemoryAllocation) throw;
  (void)cudaGetLastError();
  return {};
} catch (const std::bad_alloc&) {
  return {};
}

namespace {

cudaError_t prepare_generated_exchange_density(GeneratedExchangePlan& p, bool unrestricted,
                                               const double* alpha, const double* beta) {
  if (alpha == nullptr || (unrestricted ? beta == nullptr : beta != nullptr))
    return cudaErrorInvalidValue;
  auto& shared = *p.shared;
  const auto b = shared.batch;
  const std::size_t batch = static_cast<std::size_t>(b.batch_size);
  const std::size_t public_matrix = product(static_cast<std::size_t>(b.nbf), b.nbf);
  const std::size_t direct_matrix = product(static_cast<std::size_t>(b.direct_nbf), b.direct_nbf);
  const std::size_t public_rectangular = product(static_cast<std::size_t>(b.nbf), b.direct_nbf);
  const std::size_t spin_count = unrestricted ? 2U : 1U;
  const std::size_t cartesian = product(batch, direct_matrix);
  const std::size_t rectangular = product(batch, public_rectangular);

  for (std::size_t system = 0; system < batch; ++system) {
    auto error = cudaMemcpyAsync(p.public_spin + (system * spin_count) * public_matrix,
                                 alpha + system * public_matrix, public_matrix * sizeof(double),
                                 cudaMemcpyDeviceToDevice, shared.stream);
    if (error != cudaSuccess) return error;
    if (unrestricted) {
      error = cudaMemcpyAsync(p.public_spin + (system * spin_count + 1U) * public_matrix,
                              beta + system * public_matrix, public_matrix * sizeof(double),
                              cudaMemcpyDeviceToDevice, shared.stream);
      if (error != cudaSuccess) return error;
    }
  }

  launch_transform_density_to_direct_right_kernel(
      blocks(product(spin_count, rectangular)), 128, 0, shared.stream, b.batch_size,
      static_cast<std::int32_t>(spin_count), b.nbf, b.direct_nbf, b.ao_to_direct_transform,
      p.public_spin, shared.active, p.density_temporary);
  launch_transform_density_to_direct_left_kernel(
      blocks(product(spin_count, cartesian)), 128, 0, shared.stream, b.batch_size,
      static_cast<std::int32_t>(spin_count), b.nbf, b.direct_nbf, b.ao_to_direct_transform,
      p.density_temporary, shared.active, p.direct_spin);
  auto error = cudaGetLastError();
  if (error != cudaSuccess) return error;

  constexpr unsigned threads = 128U;
  launch_reduce_shell_pair_density_bounds_kernel(
      unrestricted, static_cast<unsigned>(b.total_shell_pairs), threads,
      3U * threads * sizeof(double), shared.stream, b, p.direct_spin, shared.active,
      p.shell_pair_density_bounds);
  launch_reduce_bounded_system_density_bounds_kernel(
      static_cast<unsigned>(batch), threads,
      detail::kDirectShellPairClassCount * threads * sizeof(double), shared.stream, b,
      p.shell_pair_density_bounds, p.system_density_bounds, p.system_pair_density_bounds);
  return cudaGetLastError();
}

cudaError_t enqueue_generated_coulomb_direct(GeneratedCoulombPlan& p, const double* density,
                                             const double* beta);
cudaError_t project_generated_coulomb(GeneratedCoulombPlan& p, double* coulomb);

cudaError_t enqueue_generated_exchange_prepared(GeneratedExchangePlan& p, bool unrestricted,
                                                double* alpha_exchange, double* beta_exchange,
                                                DirectCoulombRange range, double omega) {
  const bool full_range = range == DirectCoulombRange::Full;
  cudaError_t error = cudaSuccess;
  auto& shared = *p.shared;
  const auto b = shared.batch;
  const std::size_t batch = static_cast<std::size_t>(b.batch_size);
  const std::size_t public_matrix = product(static_cast<std::size_t>(b.nbf), b.nbf);
  const std::size_t direct_matrix = product(static_cast<std::size_t>(b.direct_nbf), b.direct_nbf);
  const std::size_t public_rectangular = product(static_cast<std::size_t>(b.nbf), b.direct_nbf);
  const std::size_t spin_count = unrestricted ? 2U : 1U;
  const std::size_t matrix = product(batch, public_matrix);
  const std::size_t cartesian = product(batch, direct_matrix);
  const std::size_t rectangular = product(batch, public_rectangular);

  error = cudaMemsetAsync(p.direct_exchange, 0,
                          product(product(spin_count, cartesian), sizeof(double)), shared.stream);
  if (error != cudaSuccess) return error;

  if (full_range) {
    error = cudaMemsetAsync(
        p.heads, 0, detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t), shared.stream);
    if (error != cudaSuccess) return error;

    std::size_t count = 0;
    const auto* kernels = generated::selected_fock_shell_kernels(count);
    for (std::size_t i = 0; i < count; ++i) {
      const auto cls = kernels[i].shell_class;
      if (!(shared.class_mask & kGeneratedStreamingFockShellClassMask & (std::uint64_t{1} << cls)))
        continue;
      error = generated::launch_shell_class_streaming_fock(
          cls, shared.stream, unrestricted, shared.worker_blocks, p.topology,
          b.shell_pair_primitive_offsets, b.shell_primitive_pairs, b.direct_ao_coefficients,
          b.positions, shared.screening, false, 0, shared.schwarz, p.direct_spin, p.direct_exchange,
          p.heads + cls, nullptr, nullptr);
      if (error != cudaSuccess) return error;
    }
    if (shared.class_mask & kNativeStreamingFockShellClassMask) {
      launch_bounded_direct_dddd_streaming_kernel(
          unrestricted, DirectScreeningPurpose::Fock, false, shared.worker_blocks, 32, 0,
          shared.stream, b, p.topology, shared.screening, shared.schwarz, p.direct_spin,
          shared.active, p.direct_exchange, p.heads + kDdddShellClass, nullptr, nullptr);
      error = cudaGetLastError();
      if (error != cudaSuccess) return error;
    }
    if (p.bounded_value_capability) {
      error = cudaMemsetAsync(p.force_cursor, 0, sizeof(unsigned long long), shared.stream);
      if (error != cudaSuccess) return error;
      launch_bounded_shell_fock_source(
          unrestricted, shared.worker_blocks, shared.stream, b, shared.screening,
          shared.shell_bounds, p.shell_pair_density_bounds, p.bounded_pair_order,
          p.shell_pair_block_bounds, p.system_density_bounds, shared.value_class_mask,
          p.bounded_value_overflow, shared.schwarz, p.direct_spin, shared.active, p.direct_exchange,
          p.force_cursor, false, true);
      error = cudaGetLastError();
      if (error != cudaSuccess) return error;
    }
  } else {
    error = cudaMemsetAsync(p.force_cursor, 0, sizeof(unsigned long long), shared.stream);
    if (error != cudaSuccess) return error;
    launch_bounded_shell_range_exchange_source(
        unrestricted, shared.worker_blocks, shared.stream, b, shared.screening, shared.shell_bounds,
        p.shell_pair_density_bounds, p.bounded_pair_order, p.shell_pair_block_bounds,
        p.system_density_bounds, p.bounded_value_overflow, shared.schwarz, p.direct_spin,
        shared.active, p.direct_exchange, p.force_cursor, range, omega);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }

  launch_transform_direct_fock_left_kernel(
      blocks(product(spin_count, rectangular)), 128, 0, shared.stream, b.batch_size,
      static_cast<std::int32_t>(spin_count), b.nbf, b.direct_nbf, b.ao_to_direct_transform,
      p.direct_exchange, shared.active, p.fock_temporary);
  launch_transform_direct_fock_right_kernel(
      blocks(product(spin_count, matrix)), 128, 0, shared.stream, b.batch_size,
      static_cast<std::int32_t>(spin_count), b.nbf, b.direct_nbf, b.ao_to_direct_transform,
      p.fock_temporary, shared.zero, shared.active, p.public_exchange);
  error = cudaGetLastError();
  if (error != cudaSuccess) return error;

  for (std::size_t system = 0; system < batch; ++system) {
    error =
        cudaMemcpyAsync(alpha_exchange + system * public_matrix,
                        p.public_exchange + (system * spin_count) * public_matrix,
                        public_matrix * sizeof(double), cudaMemcpyDeviceToDevice, shared.stream);
    if (error != cudaSuccess) return error;
    if (unrestricted) {
      error =
          cudaMemcpyAsync(beta_exchange + system * public_matrix,
                          p.public_exchange + (system * spin_count + 1U) * public_matrix,
                          public_matrix * sizeof(double), cudaMemcpyDeviceToDevice, shared.stream);
      if (error != cudaSuccess) return error;
    }
  }
  return cudaGetLastError();
}

cudaError_t enqueue_generated_coulomb_prepared(GeneratedExchangePlan& p, bool unrestricted,
                                               const double* alpha, const double* beta,
                                               double* coulomb) {
  auto& shared = *p.shared;
  auto error = enqueue_generated_coulomb_direct(shared, alpha, beta);
  if (error != cudaSuccess) return error;
  if (p.bounded_value_capability) {
    error = cudaMemsetAsync(p.force_cursor, 0, sizeof(unsigned long long), shared.stream);
    if (error != cudaSuccess) return error;
    launch_bounded_shell_fock_source(
        false, shared.worker_blocks, shared.stream, shared.batch, shared.screening,
        shared.shell_bounds, p.shell_pair_density_bounds, p.bounded_pair_order,
        p.shell_pair_block_bounds, p.system_density_bounds, shared.value_class_mask,
        p.bounded_value_overflow, shared.schwarz, shared.density, shared.active, shared.coulomb,
        p.force_cursor, true, false);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }
  return project_generated_coulomb(shared, coulomb);
}

}  // namespace

cudaError_t enqueue_generated_exchange(GeneratedExchangePlan& p, bool unrestricted,
                                       const double* alpha, const double* beta,
                                       double* alpha_exchange, double* beta_exchange,
                                       DirectCoulombRange range, double omega) {
  const bool full_range = range == DirectCoulombRange::Full;
  if (p.shared == nullptr || (!p.shared->value_capability && !p.bounded_value_capability) ||
      (!full_range && !p.bounded_value_capability))
    return cudaErrorNotSupported;
  if (alpha_exchange == nullptr ||
      (unrestricted ? beta_exchange == nullptr : beta_exchange != nullptr) ||
      (!full_range && (!std::isfinite(omega) || omega <= 0.0)))
    return cudaErrorInvalidValue;
  auto error = prepare_generated_exchange_density(p, unrestricted, alpha, beta);
  return error == cudaSuccess ? enqueue_generated_exchange_prepared(p, unrestricted, alpha_exchange,
                                                                    beta_exchange, range, omega)
                              : error;
}

cudaError_t execute_generated_full_range_energy_derivatives(
    GeneratedExchangePlan& p, bool unrestricted, const double* alpha, const double* beta,
    double coulomb_coefficient, double exchange_coefficient, std::vector<double>& derivatives) {
  if (!p.force_capability || p.bounded_pair_order == nullptr ||
      p.shell_pair_block_bounds == nullptr || p.force == nullptr || p.force_cursor == nullptr ||
      !std::isfinite(coulomb_coefficient) || !std::isfinite(exchange_coefficient))
    return cudaErrorInvalidValue;
  auto error = prepare_generated_exchange_density(p, unrestricted, alpha, beta);
  if (error != cudaSuccess) return error;

  auto& shared = *p.shared;
  const auto b = shared.batch;
  const std::size_t coordinates = static_cast<std::size_t>(b.total_atoms) * 3U;
  std::vector<double> result(2U * coordinates);
  // Drain any pending D2H before result is destroyed on failure or exception.
  struct HostResultDrain {
    cudaStream_t stream;
    bool active{true};
    ~HostResultDrain() {
      if (active) (void)cudaStreamSynchronize(stream);
    }
  } drain{shared.stream};
  error = cudaMemsetAsync(p.heads, 0, detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t),
                          shared.stream);
  if (error != cudaSuccess) return error;
  error = cudaMemsetAsync(p.force, 0, result.size() * sizeof(double), shared.stream);
  if (error != cudaSuccess) return error;
  error = cudaMemsetAsync(p.force_cursor, 0, sizeof(unsigned long long), shared.stream);
  if (error != cudaSuccess) return error;
  if (coulomb_coefficient != 0.0 || exchange_coefficient != 0.0) {
    launch_bounded_shell_energy_derivative(
        unrestricted, shared.worker_blocks, shared.stream, b, shared.screening, shared.shell_bounds,
        p.shell_pair_density_bounds, p.bounded_pair_order, p.shell_pair_block_bounds,
        p.system_density_bounds, p.heads, shared.schwarz, p.direct_spin, shared.active, p.force,
        p.force_cursor, coulomb_coefficient, exchange_coefficient, p.bounded_block_domain);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
  }
  error = cudaMemcpyAsync(result.data(), p.force, result.size() * sizeof(double),
                          cudaMemcpyDeviceToHost, shared.stream);
  if (error != cudaSuccess) return error;
  error = cudaStreamSynchronize(shared.stream);
  if (error != cudaSuccess) return error;
  drain.active = false;
  // Native shell force kernels accumulate -dE/dR. This API publishes the
  // derivative convention used by the stationary integral-source reducer.
  for (double& value : result) value = -value;
  derivatives = std::move(result);
  return cudaSuccess;
}

cudaError_t execute_generated_rsh_energy_derivatives(GeneratedExchangePlan& p, bool unrestricted,
                                                     const double* alpha, const double* beta,
                                                     double coulomb_coefficient,
                                                     double short_exchange_coefficient,
                                                     double long_exchange_coefficient, double omega,
                                                     std::vector<double>& derivatives) {
  if (!p.force_capability || p.bounded_pair_order == nullptr ||
      p.shell_pair_block_bounds == nullptr || p.force == nullptr || p.force_cursor == nullptr ||
      !std::isfinite(coulomb_coefficient) || !std::isfinite(short_exchange_coefficient) ||
      !std::isfinite(long_exchange_coefficient) || !std::isfinite(omega) || omega <= 0.0)
    return cudaErrorInvalidValue;
  // Bounded decomposition: the fused full-range J/K shell workers plus the
  // packaged omega=0.3 LR workers use two traversals of specialized sources.
  // Keep the three stationary source meanings: SR = Full - LR, with each
  // exchange source carrying its own functional coefficient. Other omegas
  // retain the fused source traversal below.
  if (omega == 0.3) {
    std::vector<double> full;
    auto error = execute_generated_full_range_energy_derivatives(
        p, unrestricted, alpha, beta, coulomb_coefficient, short_exchange_coefficient, full);
    if (error != cudaSuccess) return error;
    auto& shared = *p.shared;
    const auto b = shared.batch;
    const std::size_t coordinates = static_cast<std::size_t>(b.total_atoms) * 3U;
    if (short_exchange_coefficient == 0.0 && long_exchange_coefficient == 0.0) {
      // Disabled exchange must not form potentially overflowing spin products.
      // The full helper already drained its download and supplied a zero K row.
      full.resize(3U * coordinates, 0.0);
      derivatives = std::move(full);
      return cudaSuccess;
    }
    std::vector<double> long_force(coordinates);
    struct HostResultDrain {
      cudaStream_t stream;
      bool active{true};
      ~HostResultDrain() {
        if (active) (void)cudaStreamSynchronize(stream);
      }
    } drain{shared.stream};
    error = cudaMemsetAsync(
        p.heads, 0, detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t), shared.stream);
    if (error != cudaSuccess) return error;
    error = cudaMemsetAsync(p.force, 0, coordinates * sizeof(double), shared.stream);
    if (error != cudaSuccess) return error;
    error = cudaMemsetAsync(p.force_cursor, 0, sizeof(unsigned long long), shared.stream);
    if (error != cudaSuccess) return error;
    launch_bounded_shell_range_exchange_derivative(
        unrestricted, shared.worker_blocks, shared.stream, b, shared.screening, shared.shell_bounds,
        p.shell_pair_density_bounds, p.bounded_pair_order, p.shell_pair_block_bounds,
        p.system_density_bounds, p.heads, shared.schwarz, p.direct_spin, shared.active, p.force,
        p.force_cursor, DirectCoulombRange::Long, omega, 1.0);
    error = cudaGetLastError();
    if (error != cudaSuccess) return error;
    error = cudaMemcpyAsync(long_force.data(), p.force, coordinates * sizeof(double),
                            cudaMemcpyDeviceToHost, shared.stream);
    if (error != cudaSuccess) return error;
    error = cudaStreamSynchronize(shared.stream);
    if (error != cudaSuccess) return error;
    drain.active = false;
    std::vector<double> result(3U * coordinates);
    for (std::size_t coordinate = 0; coordinate < coordinates; ++coordinate) {
      // The full helper publishes derivatives; the shell worker returns force.
      const double long_derivative = -long_force[coordinate];
      result[coordinate] = full[coordinate];
      result[coordinates + coordinate] =
          full[coordinates + coordinate] - short_exchange_coefficient * long_derivative;
      result[2U * coordinates + coordinate] = long_exchange_coefficient * long_derivative;
    }
    derivatives = std::move(result);
    return cudaSuccess;
  }
  auto error = prepare_generated_exchange_density(p, unrestricted, alpha, beta);
  if (error != cudaSuccess) return error;

  auto& shared = *p.shared;
  const auto b = shared.batch;
  const std::size_t coordinates = static_cast<std::size_t>(b.total_atoms) * 3U;
  std::vector<double> result(3U * coordinates);
  // The fused D2H may still be pending when submission or synchronization
  // fails. The retained device owner cannot protect this local host result.
  struct HostResultDrain {
    cudaStream_t stream;
    bool active{true};
    ~HostResultDrain() {
      if (active) (void)cudaStreamSynchronize(stream);
    }
  } drain{shared.stream};
  error = cudaMemsetAsync(p.heads, 0, detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t),
                          shared.stream);
  if (error != cudaSuccess) return error;

  error = cudaMemsetAsync(p.force, 0, 3U * coordinates * sizeof(double), shared.stream);
  if (error != cudaSuccess) return error;
  error = cudaMemsetAsync(p.force_cursor, 0, sizeof(unsigned long long), shared.stream);
  if (error != cudaSuccess) return error;
  launch_bounded_shell_rsh_derivatives(
      unrestricted, shared.worker_blocks, shared.stream, b, shared.screening, shared.shell_bounds,
      p.shell_pair_density_bounds, p.bounded_pair_order, p.shell_pair_block_bounds,
      p.system_density_bounds, p.heads, shared.schwarz, p.direct_spin, shared.active, p.force,
      p.force_cursor, omega, coulomb_coefficient, short_exchange_coefficient,
      long_exchange_coefficient);
  error = cudaGetLastError();
  if (error != cudaSuccess) return error;
  error = cudaMemcpyAsync(result.data(), p.force, 3U * coordinates * sizeof(double),
                          cudaMemcpyDeviceToHost, shared.stream);
  if (error != cudaSuccess) return error;
  error = cudaStreamSynchronize(shared.stream);
  if (error != cudaSuccess) return error;
  drain.active = false;
  for (double& value : result) value = -value;
  derivatives = std::move(result);
  return cudaSuccess;
}

namespace {

cudaError_t enqueue_generated_coulomb_direct(GeneratedCoulombPlan& p, const double* density,
                                             const double* beta) {
  const auto b = p.batch;
  const auto matrix = std::size_t(b.batch_size) * b.nbf * b.nbf;
  const auto cartesian = std::size_t(b.batch_size) * b.direct_nbf * b.direct_nbf;
  const auto rectangle = std::size_t(b.batch_size) * b.nbf * b.direct_nbf;
  if (beta) {
    cuda_df::launch_sum_spin_density_kernel(blocks(matrix), 128, 0, p.stream, matrix, density, beta,
                                            p.total_density);
    density = p.total_density;
  }
  launch_transform_density_to_direct_right_kernel(blocks(rectangle), 128, 0, p.stream, b.batch_size,
                                                  1, b.nbf, b.direct_nbf, b.ao_to_direct_transform,
                                                  density, p.active, p.temporary);
  launch_transform_density_to_direct_left_kernel(blocks(cartesian), 128, 0, p.stream, b.batch_size,
                                                 1, b.nbf, b.direct_nbf, b.ao_to_direct_transform,
                                                 p.temporary, p.active, p.density);
  auto error = cudaMemsetAsync(p.coulomb, 0, cartesian * sizeof(double), p.stream);
  if (error != cudaSuccess) return error;
  error = cudaMemsetAsync(p.heads, 0, detail::kDirectQuartetShellClassCount * sizeof(std::uint32_t),
                          p.stream);
  if (error != cudaSuccess) return error;
  std::size_t count = 0;
  const auto* kernels = generated::selected_fock_shell_kernels(count);
  for (std::size_t i = 0; i < count; ++i) {
    const auto cls = kernels[i].shell_class;
    if (!(p.value_class_mask & kGeneratedStreamingFockShellClassMask & (std::uint64_t{1} << cls)))
      continue;
    error = generated::launch_shell_class_streaming_fock(
        cls, p.stream, false, p.worker_blocks, p.topology, b.shell_pair_primitive_offsets,
        b.shell_primitive_pairs, b.direct_ao_coefficients, b.positions, p.screening, false, 0,
        p.schwarz, p.density, p.coulomb, p.heads + cls, nullptr, nullptr);
    if (error != cudaSuccess) return error;
  }
  if (p.value_class_mask & kNativeStreamingFockShellClassMask) {
    launch_bounded_direct_dddd_streaming_kernel(
        false, DirectScreeningPurpose::Fock, false, p.worker_blocks, 32, 0, p.stream, b, p.topology,
        p.screening, p.schwarz, p.density, p.active, p.coulomb, p.heads + kDdddShellClass, nullptr,
        nullptr);
  }
  return cudaGetLastError();
}

cudaError_t project_generated_coulomb(GeneratedCoulombPlan& p, double* coulomb) {
  if (coulomb == nullptr) return cudaErrorInvalidValue;
  const auto b = p.batch;
  const auto matrix = std::size_t(b.batch_size) * b.nbf * b.nbf;
  const auto rectangle = std::size_t(b.batch_size) * b.nbf * b.direct_nbf;
  launch_transform_direct_fock_left_kernel(blocks(rectangle), 128, 0, p.stream, b.batch_size, 1,
                                           b.nbf, b.direct_nbf, b.ao_to_direct_transform, p.coulomb,
                                           p.active, p.temporary);
  launch_transform_direct_fock_right_kernel(blocks(matrix), 128, 0, p.stream, b.batch_size, 1,
                                            b.nbf, b.direct_nbf, b.ao_to_direct_transform,
                                            p.temporary, p.zero, p.active, coulomb);
  return cudaGetLastError();
}

}  // namespace

cudaError_t enqueue_generated_coulomb(GeneratedCoulombPlan& p, const double* density,
                                      const double* beta, double* coulomb) {
  if (!p.value_capability) return cudaErrorNotSupported;
  auto error = enqueue_generated_coulomb_direct(p, density, beta);
  return error == cudaSuccess ? project_generated_coulomb(p, coulomb) : error;
}

cudaError_t enqueue_generated_coulomb(GeneratedExchangePlan& p, bool unrestricted,
                                      const double* alpha, const double* beta, double* coulomb) {
  if (p.shared == nullptr || (!p.shared->value_capability && !p.bounded_value_capability) ||
      coulomb == nullptr)
    return cudaErrorNotSupported;
  auto error = prepare_generated_exchange_density(p, unrestricted, alpha, beta);
  return error == cudaSuccess
             ? enqueue_generated_coulomb_prepared(p, unrestricted, alpha, beta, coulomb)
             : error;
}

cudaError_t enqueue_generated_rsh_values(GeneratedExchangePlan& p, bool unrestricted,
                                         const double* alpha, const double* beta, double* coulomb,
                                         double* full_alpha_exchange, double* full_beta_exchange,
                                         double* range_alpha_exchange, double* range_beta_exchange,
                                         DirectCoulombRange range, double omega) {
  if (p.shared == nullptr || !p.bounded_value_capability || coulomb == nullptr ||
      full_alpha_exchange == nullptr || range_alpha_exchange == nullptr ||
      (unrestricted ? (full_beta_exchange == nullptr || range_beta_exchange == nullptr)
                    : (full_beta_exchange != nullptr || range_beta_exchange != nullptr)) ||
      range == DirectCoulombRange::Full || !std::isfinite(omega) || omega <= 0.0)
    return cudaErrorNotSupported;
  auto error = prepare_generated_exchange_density(p, unrestricted, alpha, beta);
  if (error != cudaSuccess) return error;
  error = enqueue_generated_coulomb_prepared(p, unrestricted, alpha, beta, coulomb);
  if (error != cudaSuccess) return error;
  error = enqueue_generated_exchange_prepared(p, unrestricted, full_alpha_exchange,
                                              full_beta_exchange, DirectCoulombRange::Full, 0.0);
  if (error != cudaSuccess) return error;
  return enqueue_generated_exchange_prepared(p, unrestricted, range_alpha_exchange,
                                             range_beta_exchange, range, omega);
}

}  // namespace generativeqc::scf::cuda_execution
