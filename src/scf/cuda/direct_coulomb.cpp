#include "scf/cuda/direct_coulomb.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
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
#include "scf/cuda/direct_pair_cache.hpp"
#include "scf/cuda/direct_schwarz_kernels.hpp"
#include "scf/cuda/metadata_upload.hpp"
#include "scf/cuda/queue_plan.hpp"

namespace vibeqc::scf::cuda_execution {
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

std::unique_ptr<GeneratedCoulombPlan> prepare_generated_coulomb(const HostBatch& host,
                                                                DeviceBatch borrowed,
                                                                cudaStream_t stream, int device,
                                                                double screening,
                                                                std::size_t budget) {
  cudaDeviceProp properties{};
  check(cudaGetDeviceProperties(&properties, device));
  generated::select_profile_for_device(device, properties.major, properties.minor);
  const auto schedule = cuda_policy::resolve_direct_jk_schedule_policy(
      runtime::cuda_target_info_from_properties(properties));
  const auto present = present_direct_shell_class_mask(host);
  const auto generated_mask =
      generated::enabled_fock_shell_class_mask() & kGeneratedStreamingFockShellClassMask;
  if ((present & ~(generated_mask | kNativeStreamingFockShellClassMask)) != 0) return {};
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
  // Reuse the target-legal HF worker and recurrence-stack policy; a prepared
  // KS owner must not rely on an earlier HF call having raised the CUDA limit.
  plan->worker_blocks = static_cast<unsigned>(properties.multiProcessorCount) *
                        schedule.persistent_quartet_warps_per_sm;
  std::size_t stack_limit = 0;
  check(cudaDeviceGetLimit(&stack_limit, cudaLimitStackSize));
  if (stack_limit < schedule.cuda_stack_limit_bytes)
    check(cudaDeviceSetLimit(cudaLimitStackSize, schedule.cuda_stack_limit_bytes));
  try {
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
                                            device_order,
                                            device_offsets,
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
}

cudaError_t enqueue_generated_coulomb(GeneratedCoulombPlan& p, const double* density,
                                      const double* beta, double* coulomb) {
  const auto b = p.batch;
  const auto matrix = std::size_t(b.batch_size) * b.nbf * b.nbf;
  const auto cartesian = std::size_t(b.batch_size) * b.direct_nbf * b.direct_nbf;
  const auto rectangle = std::size_t(b.batch_size) * b.nbf * b.direct_nbf;
  if (beta) {
    cuda_df::launch_sum_spin_density_kernel(blocks(matrix), 128, 0, p.stream, matrix, density, beta,
                                            p.total_density);
    density = p.total_density;
  }
  // J depends only on the symmetric density, so the public row-major input
  // may be viewed transposed by the column-major transform. The antisymmetric
  // component cancels exactly under ERI pair symmetry, including UHF sums.
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
  // Keep the geometry-only Schwarz contract of the public provider. The
  // Coulomb consumer disables HF's extra density-dependent screening gates.
  std::size_t count = 0;
  const auto* kernels = generated::selected_fock_shell_kernels(count);
  for (std::size_t i = 0; i < count; ++i) {
    const auto cls = kernels[i].shell_class;
    if (!(p.class_mask & kGeneratedStreamingFockShellClassMask & (std::uint64_t{1} << cls)))
      continue;
    error = generated::launch_shell_class_streaming_fock(
        cls, p.stream, false, p.worker_blocks, p.topology, b.shell_pair_primitive_offsets,
        b.shell_primitive_pairs, b.direct_ao_coefficients, b.positions, p.screening, false, 0,
        p.schwarz, p.density, p.coulomb, p.heads + cls, nullptr, nullptr);
    if (error != cudaSuccess) return error;
  }
  if (p.class_mask & kNativeStreamingFockShellClassMask) {
    launch_bounded_direct_dddd_streaming_kernel(
        false, DirectScreeningPurpose::Fock, false, p.worker_blocks, 32, 0, p.stream, b, p.topology,
        p.screening, p.schwarz, p.density, p.active, p.coulomb, p.heads + kDdddShellClass, nullptr,
        nullptr);
  }
  launch_transform_direct_fock_left_kernel(blocks(rectangle), 128, 0, p.stream, b.batch_size, 1,
                                           b.nbf, b.direct_nbf, b.ao_to_direct_transform, p.coulomb,
                                           p.active, p.temporary);
  launch_transform_direct_fock_right_kernel(blocks(matrix), 128, 0, p.stream, b.batch_size, 1,
                                            b.nbf, b.direct_nbf, b.ao_to_direct_transform,
                                            p.temporary, p.zero, p.active, coulomb);
  return cudaGetLastError();
}
}  // namespace vibeqc::scf::cuda_execution
