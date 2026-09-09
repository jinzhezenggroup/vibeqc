#include "scf/fleet.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <map>
#include <new>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>

#include "molecule/basis.hpp"
#include "runtime/resource_usage.hpp"
#include "scf/mean_field.hpp"

namespace vibeqc::scf {
namespace {

using WorkloadKey = std::tuple<std::size_t, int, int, std::size_t>;

WorkloadKey workload_key(const core::System& system, vibeqc_method method) {
  std::size_t primitive_count = 0;
  for (const auto& shell : system.shells) {
    primitive_count += shell.primitives.size();
  }
  const int spin_excess = static_cast<int>(system.multiplicity) - 1;
  const int alpha = method == VIBEQC_METHOD_UHF ? (system.electron_count + spin_excess) / 2
                                                : system.electron_count / 2;
  const int beta = method == VIBEQC_METHOD_UHF ? system.electron_count - alpha : alpha;
  return {molecule::ao_count(system), alpha, beta, primitive_count};
}

vibeqc_status exception_status() {
  try {
    throw;
  } catch (const std::bad_alloc&) {
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  } catch (const std::invalid_argument&) {
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  } catch (const std::exception&) {
    return VIBEQC_STATUS_NUMERICAL_FAILURE;
  } catch (...) {
    return VIBEQC_STATUS_INTERNAL_ERROR;
  }
}

bool valid_coordinates(const std::vector<double>& coordinates, std::size_t atom_count) {
  return coordinates.size() == atom_count * 3 &&
         std::all_of(coordinates.begin(), coordinates.end(),
                     [](double value) { return std::isfinite(value); });
}

void apply_coordinates(core::System& system, const std::vector<double>& coordinates) {
  for (std::size_t atom = 0; atom < system.atoms.size(); ++atom) {
    for (std::size_t axis = 0; axis < 3; ++axis) {
      system.atoms[atom].position[axis] = coordinates[atom * 3 + axis];
    }
  }
}

// Capture the actual geometry used to build a DF bucket.  Keeping this
// snapshot alongside the opaque CUDA plan lets FleetPlan distinguish a warm
// replay (same coordinates and topology) from a geometry update that requires
// rebuilding geometry-derived metric/three-center tensors.
void append_geometry_positions(const core::System& system, std::vector<double>& positions) {
  positions.reserve(positions.size() + system.atoms.size() * 3);
  for (const auto& atom : system.atoms) {
    positions.push_back(atom.position[0]);
    positions.push_back(atom.position[1]);
    positions.push_back(atom.position[2]);
  }
}

/**
 * Reuse an auxiliary shell topology at the coordinates of one fleet item.
 * The auxiliary basis is fixed for a prepared batch, while its Gaussian
 * centers follow the item geometry on every replay.
 */
core::System auxiliary_for_geometry(const std::optional<core::System>& auxiliary_template,
                                    const core::System& system) {
  if (!auxiliary_template.has_value()) return system;
  core::System auxiliary = *auxiliary_template;
  auxiliary.atoms = system.atoms;
  auxiliary.charge = system.charge;
  auxiliary.multiplicity = system.multiplicity;
  auxiliary.electron_count = system.electron_count;
  return auxiliary;
}

/** Merge additive PPPS counters without rounding derived efficiencies. */
void merge_ppps_queue_profile(CudaPppsQueueProfile& aggregate, const CudaPppsQueueProfile& source) {
  aggregate.descriptor_slots += source.descriptor_slots;
  aggregate.non_empty_descriptors += source.non_empty_descriptors;
  aggregate.tasks += source.tasks;
  aggregate.primitive_work += source.primitive_work;
  if (aggregate.ket_count_histogram.size() < source.ket_count_histogram.size()) {
    aggregate.ket_count_histogram.resize(source.ket_count_histogram.size(), 0U);
  }
  for (std::size_t index = 0; index < source.ket_count_histogram.size(); ++index) {
    aggregate.ket_count_histogram[index] += source.ket_count_histogram[index];
  }
  aggregate.primitive_warp_slots += source.primitive_warp_slots;
  for (std::size_t index = 0; index < kPppsProfileBlockThreads.size(); ++index) {
    aggregate.lane_slots[index] += source.lane_slots[index];
    aggregate.task_schedule_ideal[index] += source.task_schedule_ideal[index];
    aggregate.task_schedule_makespan[index] += source.task_schedule_makespan[index];
    aggregate.primitive_schedule_ideal[index] += source.primitive_schedule_ideal[index];
    aggregate.primitive_schedule_makespan[index] += source.primitive_schedule_makespan[index];
  }
  for (std::size_t orientation = 0; orientation < CudaPppsQueueProfile::kOrientationCount;
       ++orientation) {
    aggregate.orientation_tasks[orientation] += source.orientation_tasks[orientation];
    aggregate.orientation_primitive_work[orientation] +=
        source.orientation_primitive_work[orientation];
  }
  for (std::size_t bucket = 0; bucket < CudaPppsQueueProfile::kPrimitivePairBucketCount; ++bucket) {
    aggregate.bra_primitive_tasks[bucket] += source.bra_primitive_tasks[bucket];
    aggregate.bra_primitive_work[bucket] += source.bra_primitive_work[bucket];
    aggregate.ket_primitive_tasks[bucket] += source.ket_primitive_tasks[bucket];
    aggregate.ket_primitive_work[bucket] += source.ket_primitive_work[bucket];
  }
}

}  // namespace

FleetPlan::FleetPlan(std::vector<core::System> systems, vibeqc_method method, ScfOptions options,
                     bool warm_starts_enabled, bool cuda_fock_enabled,
                     bool shell_class_profiling_enabled,
                     bool inactive_eigensolver_profiling_enabled, int device_id,
                     std::optional<core::System> auxiliary_template,
                     bool cuda_density_fitting_enabled)
    : systems_(std::move(systems)),
      method_(method),
      options_(options),
      warm_starts_enabled_(warm_starts_enabled),
      cuda_fock_enabled_(cuda_fock_enabled),
      cuda_density_fitting_enabled_(cuda_density_fitting_enabled),
      shell_class_profiling_enabled_(shell_class_profiling_enabled),
      inactive_eigensolver_profiling_enabled_(inactive_eigensolver_profiling_enabled),
      device_id_(device_id),
      auxiliary_template_(std::move(auxiliary_template)),
      execution_order_(systems_.size()),
      bucket_ids_(systems_.size()),
      warm_densities_(systems_.size()) {
  const bool fitted = options_.density_fitting_mode != VIBEQC_DENSITY_FITTING_NONE;
  const FockBackend backend =
      cuda_fock_enabled_ || cuda_density_fitting_enabled_ ? FockBackend::Cuda : FockBackend::Cpu;
  const ResolvedFockBuild expected = resolve_fock_build(
      make_hf_fock_spec(
          method_ == VIBEQC_METHOD_UHF ? FockSpin::Unrestricted : FockSpin::Restricted,
          fitted ? FockApproximation::DensityFitted : FockApproximation::Exact),
      backend, options_.screening_tolerance, options_.density_fitting_relative_threshold);
  if (options_.resolved_fock_build.has_value() && *options_.resolved_fock_build != expected) {
    throw std::invalid_argument("fleet options disagree with the resolved HF Fock strategy");
  }
  options_.resolved_fock_build = expected;
  if (!fitted) require_exact_direct_strategy(expected, expected.spec.spin, backend);
  std::iota(execution_order_.begin(), execution_order_.end(), 0);
  std::stable_sort(execution_order_.begin(), execution_order_.end(),
                   [&](std::size_t a, std::size_t b) {
                     return workload_key(systems_[a], method_) < workload_key(systems_[b], method_);
                   });

  std::map<WorkloadKey, std::size_t> buckets;
  for (const std::size_t system_index : execution_order_) {
    const WorkloadKey key = workload_key(systems_[system_index], method_);
    auto [iterator, inserted] = buckets.emplace(key, buckets.size());
    (void)inserted;
    bucket_ids_[system_index] = iterator->second;
  }
  cuda_bucket_plans_.resize(buckets.size(), nullptr);
  cuda_density_fitting_plans_.resize(buckets.size(), nullptr);
  cuda_density_fitting_positions_.resize(buckets.size());
  cuda_density_fitting_batch_sizes_.resize(buckets.size(), 0);
  cuda_density_fitting_data_.resize(buckets.size());
  cuda_density_fitting_diagnostics_.resize(buckets.size());
}

FleetPlan::~FleetPlan() {
  for (CudaRhfBucketPlan* plan : cuda_bucket_plans_) {
    destroy_rhf_cuda_bucket_plan(plan);
  }
  for (CudaDensityFittingJkPlan* plan : cuda_density_fitting_plans_) {
    destroy_cuda_density_fitting_jk_plan(plan);
  }
}

std::vector<FleetItemResult> FleetPlan::execute(
    const std::vector<std::optional<std::vector<double>>>& coordinates) {
  if (!coordinates.empty() && coordinates.size() != systems_.size()) {
    throw std::invalid_argument("fleet coordinate list does not match system count");
  }
  last_shell_class_profile_.reset();
  last_ppps_queue_profile_.reset();
  last_eigensolver_diagnostics_.clear();
  last_density_fitting_metric_diagnostics_.clear();
  last_inactive_eigensolver_profile_.clear();
  std::vector<FleetItemResult> results(systems_.size());
  const auto execute_one = [&](std::size_t system_index) {
    FleetItemResult& item = results[system_index];
    item.bucket_id = bucket_ids_[system_index];
    item.executed_backend = VIBEQC_BACKEND_CPU_REFERENCE;
    core::System execution_system = systems_[system_index];
    if (!coordinates.empty() && coordinates[system_index].has_value()) {
      if (!valid_coordinates(*coordinates[system_index], execution_system.atoms.size())) {
        item.status = VIBEQC_STATUS_INVALID_ARGUMENT;
        return;
      }
      apply_coordinates(execution_system, *coordinates[system_index]);
    }

    const bool has_warm_density = warm_starts_enabled_ && warm_densities_[system_index].has_value();
    item.warm_start_used = has_warm_density;
    const ResolvedFockBuild& strategy = *options_.resolved_fock_build;
    const bool use_cuda_density_fitting =
        strategy.legacy_density_fitting && strategy.backend == FockBackend::Cuda;
    const auto evaluate = [&](const std::vector<double>* initial_density) {
      if (strategy.legacy_density_fitting) {
        const core::System auxiliary =
            auxiliary_for_geometry(auxiliary_template_, execution_system);
        if (use_cuda_density_fitting) {
          return method_ == VIBEQC_METHOD_UHF
                     ? run_uhf_density_fitting_cuda(execution_system, auxiliary, options_,
                                                    device_id_, initial_density)
                     : run_rhf_density_fitting_cuda(execution_system, auxiliary, options_,
                                                    device_id_, initial_density);
        }
        return method_ == VIBEQC_METHOD_UHF
                   ? run_uhf_density_fitting(execution_system, auxiliary, options_, initial_density)
                   : run_rhf_density_fitting(execution_system, auxiliary, options_,
                                             initial_density);
      }
      return method_ == VIBEQC_METHOD_UHF ? run_uhf(execution_system, options_, initial_density)
                                          : run_rhf(execution_system, options_, initial_density);
    };
    try {
      const std::vector<double>* initial_density =
          has_warm_density ? &warm_densities_[system_index]->density : nullptr;
      item.scf = evaluate(initial_density);
      if (use_cuda_density_fitting) {
        item.executed_backend = VIBEQC_BACKEND_CUDA;
      }
      if (has_warm_density && !item.scf.converged) {
        // A geometry change can make an otherwise topology-compatible density
        // a poor numerical guess. Retry cold so warm starts never reduce the
        // robustness of independent fleet items.
        item.warm_start_fallback = true;
        item.scf = evaluate(nullptr);
      }
      item.status = item.scf.converged ? VIBEQC_STATUS_SUCCESS : VIBEQC_STATUS_SCF_NOT_CONVERGED;
    } catch (...) {
      if (has_warm_density) {
        try {
          item.warm_start_fallback = true;
          item.scf = evaluate(nullptr);
          item.status =
              item.scf.converged ? VIBEQC_STATUS_SUCCESS : VIBEQC_STATUS_SCF_NOT_CONVERGED;
        } catch (...) {
          item.status = exception_status();
        }
      } else {
        item.status = exception_status();
      }
    }

    if (item.status == VIBEQC_STATUS_SUCCESS && warm_starts_enabled_ &&
        warm_start_updates_enabled_) {
      retain_warm_state(system_index, execution_system, item.scf);
    }
  };

  // Systems within a bucket have compatible matrix and primitive dimensions.
  // The reference backend dispatches them to a bounded native worker group;
  // production CUDA backends can lower the same bucket to batched kernels and
  // batched small-matrix library calls without changing result semantics.
  std::size_t bucket_begin = 0;
  while (bucket_begin < execution_order_.size()) {
    std::size_t bucket_end = bucket_begin + 1;
    const std::size_t bucket = bucket_ids_[execution_order_[bucket_begin]];
    while (bucket_end < execution_order_.size() &&
           bucket_ids_[execution_order_[bucket_end]] == bucket) {
      ++bucket_end;
    }
    const std::size_t bucket_size = bucket_end - bucket_begin;
    const std::size_t hardware_threads = std::max<unsigned>(1, std::thread::hardware_concurrency());
    // An accepted global CPU resource plan may require serialized items.
    // This explicit cap is independent of the hardware thread count; without
    // it a largest-item workspace estimate would undercount concurrent solves.
    const auto requested_workers = runtime::cpu_resource_observation.cpu_worker_limit;
    const std::size_t worker_count = std::min(
        bucket_size, requested_workers ? std::min<std::size_t>(hardware_threads, requested_workers)
                                       : hardware_threads);
    if (cuda_fock_enabled_) {
      std::vector<core::System> cuda_systems;
      std::vector<std::size_t> original_indices;
      std::vector<const std::vector<double>*> initial_densities;
      cuda_systems.reserve(bucket_size);
      original_indices.reserve(bucket_size);
      initial_densities.reserve(bucket_size);
      for (std::size_t position = bucket_begin; position < bucket_end; ++position) {
        const std::size_t system_index = execution_order_[position];
        FleetItemResult& item = results[system_index];
        item.bucket_id = bucket_ids_[system_index];
        core::System execution_system = systems_[system_index];
        if (!coordinates.empty() && coordinates[system_index].has_value()) {
          if (!valid_coordinates(*coordinates[system_index], execution_system.atoms.size())) {
            item.status = VIBEQC_STATUS_INVALID_ARGUMENT;
            continue;
          }
          apply_coordinates(execution_system, *coordinates[system_index]);
        }
        const bool has_warm_density =
            warm_starts_enabled_ && warm_densities_[system_index].has_value();
        item.warm_start_used = has_warm_density;
        cuda_systems.push_back(std::move(execution_system));
        original_indices.push_back(system_index);
        initial_densities.push_back(has_warm_density ? &warm_densities_[system_index]->density
                                                     : nullptr);
      }

      if (!cuda_systems.empty()) {
        std::vector<RhfBucketItem> cuda_results =
            method_ == VIBEQC_METHOD_UHF
                ? run_uhf_cuda_bucket_cached(&cuda_bucket_plans_[bucket], cuda_systems, options_,
                                             initial_densities, device_id_,
                                             shell_class_profiling_enabled_,
                                             inactive_eigensolver_profiling_enabled_)
                : run_rhf_cuda_bucket_cached(&cuda_bucket_plans_[bucket], cuda_systems, options_,
                                             initial_densities, device_id_,
                                             shell_class_profiling_enabled_,
                                             inactive_eigensolver_profiling_enabled_);
        CudaEigensolverDiagnostic eigensolver_diagnostic;
        if (get_rhf_cuda_eigensolver_diagnostic(cuda_bucket_plans_[bucket],
                                                eigensolver_diagnostic)) {
          eigensolver_diagnostic.bucket_id = bucket;
          last_eigensolver_diagnostics_.push_back(eigensolver_diagnostic);
        }
        if (inactive_eigensolver_profiling_enabled_) {
          CudaInactiveEigensolverProfile bucket_profile;
          if (get_rhf_cuda_inactive_eigensolver_profile(cuda_bucket_plans_[bucket],
                                                        bucket_profile)) {
            for (auto& entry : bucket_profile) entry.bucket_id = bucket;
            last_inactive_eigensolver_profile_.insert(last_inactive_eigensolver_profile_.end(),
                                                      bucket_profile.begin(), bucket_profile.end());
          }
        }
        if (shell_class_profiling_enabled_) {
          CudaRhfShellClassProfile bucket_profile{};
          if (get_rhf_cuda_shell_class_profile(cuda_bucket_plans_[bucket], bucket_profile)) {
            if (!last_shell_class_profile_.has_value()) {
              last_shell_class_profile_.emplace();
            }
            for (std::size_t shell_class = 0; shell_class < bucket_profile.size(); ++shell_class) {
              auto& aggregate = (*last_shell_class_profile_)[shell_class];
              const auto& entry = bucket_profile[shell_class];
              aggregate.shell_quartets += entry.shell_quartets;
              aggregate.tiles += entry.tiles;
              aggregate.ao_quartets += entry.ao_quartets;
              aggregate.primitive_quartets += entry.primitive_quartets;
            }
          }
          CudaPppsQueueProfile bucket_ppps_profile;
          if (get_rhf_cuda_ppps_queue_profile(cuda_bucket_plans_[bucket], bucket_ppps_profile)) {
            if (!last_ppps_queue_profile_.has_value()) {
              last_ppps_queue_profile_.emplace();
            }
            merge_ppps_queue_profile(*last_ppps_queue_profile_, bucket_ppps_profile);
          }
        }
        for (std::size_t slot = 0; slot < cuda_results.size(); ++slot) {
          const std::size_t system_index = original_indices[slot];
          FleetItemResult& item = results[system_index];
          item.status = cuda_results[slot].status;
          item.scf = std::move(cuda_results[slot].scf);
          item.executed_backend = VIBEQC_BACKEND_CUDA;

          if (item.warm_start_used && item.status != VIBEQC_STATUS_SUCCESS &&
              item.status != VIBEQC_STATUS_CUDA_ERROR &&
              item.status != VIBEQC_STATUS_OUT_OF_MEMORY) {
            item.warm_start_fallback = true;
            const std::vector<core::System> cold_system{cuda_systems[slot]};
            const std::vector<const std::vector<double>*> cold_density{nullptr};
            std::vector<RhfBucketItem> cold =
                method_ == VIBEQC_METHOD_UHF
                    ? run_uhf_cuda_bucket(cold_system, options_, cold_density, device_id_, false)
                    : run_rhf_cuda_bucket(cold_system, options_, cold_density, device_id_, false);
            item.status = cold.front().status;
            item.scf = std::move(cold.front().scf);
          }
          if (item.status == VIBEQC_STATUS_SUCCESS && warm_starts_enabled_ &&
              warm_start_updates_enabled_) {
            retain_warm_state(system_index, cuda_systems[slot], item.scf);
          }
        }
      }
    } else if (cuda_density_fitting_enabled_) {
      std::vector<core::System> df_systems;
      std::vector<std::size_t> original_indices;
      std::vector<const std::vector<double>*> initial_densities;
      std::vector<double> bucket_positions;
      bool malformed_coordinate = false;
      df_systems.reserve(bucket_size);
      original_indices.reserve(bucket_size);
      initial_densities.reserve(bucket_size);
      for (std::size_t position = bucket_begin; position < bucket_end; ++position) {
        const std::size_t system_index = execution_order_[position];
        FleetItemResult& item = results[system_index];
        item.bucket_id = bucket_ids_[system_index];
        core::System execution_system = systems_[system_index];
        if (!coordinates.empty() && coordinates[system_index].has_value()) {
          if (!valid_coordinates(*coordinates[system_index], execution_system.atoms.size())) {
            item.status = VIBEQC_STATUS_INVALID_ARGUMENT;
            malformed_coordinate = true;
            continue;
          }
          apply_coordinates(execution_system, *coordinates[system_index]);
        }
        const bool has_warm_density =
            warm_starts_enabled_ && warm_densities_[system_index].has_value();
        item.warm_start_used = has_warm_density;
        df_systems.push_back(std::move(execution_system));
        original_indices.push_back(system_index);
        initial_densities.push_back(has_warm_density ? &warm_densities_[system_index]->density
                                                     : nullptr);
        append_geometry_positions(df_systems.back(), bucket_positions);
      }

      if (!df_systems.empty()) {
        // A malformed item changes the batch shape.  Invalidate any previous
        // plan before executing the surviving items so a later corrected
        // replay cannot accidentally reuse a plan for a different subset.
        if (malformed_coordinate || bucket_positions != cuda_density_fitting_positions_[bucket] ||
            cuda_density_fitting_batch_sizes_[bucket] != df_systems.size()) {
          destroy_cuda_density_fitting_jk_plan(cuda_density_fitting_plans_[bucket]);
          cuda_density_fitting_plans_[bucket] = nullptr;
          cuda_density_fitting_positions_[bucket].clear();
          cuda_density_fitting_batch_sizes_[bucket] = 0;
          cuda_density_fitting_data_[bucket].clear();
          cuda_density_fitting_diagnostics_[bucket].clear();
        }
        std::vector<CudaDensityFittingMetricDiagnostic> bucket_metric_diagnostics;
        // A positive memory budget opts into bounded transient preparation;
        // retaining derivative tensors between calls would turn that budget
        // into an unbounded Fleet-level host reservation. The default path
        // keeps prepared records for warm replay, while budgeted calls rebuild
        // only their bounded chunks.
        auto* prepared_cache = options_.density_fitting_memory_budget_bytes == 0
                                   ? &cuda_density_fitting_data_[bucket]
                                   : nullptr;
        if (prepared_cache == nullptr) {
          cuda_density_fitting_data_[bucket].clear();
        }
        std::vector<RhfBucketItem> df_results =
            method_ == VIBEQC_METHOD_UHF
                ? run_uhf_density_fitting_cuda_bucket_cached(
                      &cuda_density_fitting_plans_[bucket], df_systems, auxiliary_template_,
                      options_, initial_densities, device_id_, &bucket_metric_diagnostics,
                      prepared_cache)
                : run_rhf_density_fitting_cuda_bucket_cached(
                      &cuda_density_fitting_plans_[bucket], df_systems, auxiliary_template_,
                      options_, initial_densities, device_id_, &bucket_metric_diagnostics,
                      prepared_cache);
        if (!malformed_coordinate &&
            std::all_of(df_results.begin(), df_results.end(), [](const RhfBucketItem& result) {
              return result.status == VIBEQC_STATUS_SUCCESS;
            })) {
          cuda_density_fitting_positions_[bucket] = std::move(bucket_positions);
          cuda_density_fitting_batch_sizes_[bucket] = df_systems.size();
        } else {
          // Keep a failed batch from being mistaken for a complete cached
          // topology on the next replay.  The item results themselves remain
          // isolated and are still returned in caller order.
          destroy_cuda_density_fitting_jk_plan(cuda_density_fitting_plans_[bucket]);
          cuda_density_fitting_plans_[bucket] = nullptr;
          cuda_density_fitting_positions_[bucket].clear();
          cuda_density_fitting_batch_sizes_[bucket] = 0;
          cuda_density_fitting_data_[bucket].clear();
          cuda_density_fitting_diagnostics_[bucket].clear();
        }
        if (bucket_metric_diagnostics.empty()) {
          // Cached plans skip metric setup, so surface the original records
          // consistently on every replay.
          last_density_fitting_metric_diagnostics_.insert(
              last_density_fitting_metric_diagnostics_.end(),
              cuda_density_fitting_diagnostics_[bucket].begin(),
              cuda_density_fitting_diagnostics_[bucket].end());
        }
        for (auto& diagnostic : bucket_metric_diagnostics) {
          diagnostic.bucket_id = bucket;
          // The CUDA bucket may omit malformed coordinate items before plan
          // construction. Translate the plan-local slot back to the caller's
          // original input index so diagnostics remain actionable alongside
          // `FleetItemResult` records.
          if (diagnostic.system_index < original_indices.size()) {
            diagnostic.system_index = original_indices[diagnostic.system_index];
          }
          last_density_fitting_metric_diagnostics_.push_back(diagnostic);
        }
        if (!bucket_metric_diagnostics.empty()) {
          cuda_density_fitting_diagnostics_[bucket] = bucket_metric_diagnostics;
        }
        for (std::size_t slot = 0; slot < df_results.size(); ++slot) {
          const std::size_t system_index = original_indices[slot];
          FleetItemResult& item = results[system_index];
          item.status = df_results[slot].status;
          item.scf = std::move(df_results[slot].scf);
          item.executed_backend = VIBEQC_BACKEND_CUDA;

          // A warm density can be a poor guess after a geometry replay. Keep
          // the batch's failure isolation, but retry that one item cold when
          // the failure is numerical rather than a shared CUDA/OOM failure.
          if (item.warm_start_used && item.status != VIBEQC_STATUS_SUCCESS &&
              item.status != VIBEQC_STATUS_CUDA_ERROR &&
              item.status != VIBEQC_STATUS_OUT_OF_MEMORY) {
            item.warm_start_fallback = true;
            try {
              const core::System auxiliary =
                  auxiliary_for_geometry(auxiliary_template_, df_systems[slot]);
              item.scf = method_ == VIBEQC_METHOD_UHF
                             ? run_uhf_density_fitting_cuda(df_systems[slot], auxiliary, options_,
                                                            device_id_, nullptr)
                             : run_rhf_density_fitting_cuda(df_systems[slot], auxiliary, options_,
                                                            device_id_, nullptr);
              item.status =
                  item.scf.converged ? VIBEQC_STATUS_SUCCESS : VIBEQC_STATUS_SCF_NOT_CONVERGED;
            } catch (...) {
              item.status = exception_status();
            }
          }
          if (item.status == VIBEQC_STATUS_SUCCESS && warm_starts_enabled_ &&
              warm_start_updates_enabled_) {
            retain_warm_state(system_index, df_systems[slot], item.scf);
          }
        }
      }
    } else if (worker_count == 1) {
      // One worker can own a multi-item bucket under the resource policy.
      for (std::size_t position = bucket_begin; position < bucket_end; ++position) {
        execute_one(execution_order_[position]);
      }
    } else {
      std::atomic<std::size_t> next{bucket_begin};
      std::vector<std::thread> workers;
      workers.reserve(worker_count);
      for (std::size_t worker = 0; worker < worker_count; ++worker) {
        workers.emplace_back([&] {
          while (true) {
            const std::size_t position = next.fetch_add(1);
            if (position >= bucket_end) break;
            execute_one(execution_order_[position]);
          }
        });
      }
      for (auto& worker : workers) worker.join();
    }
    if (runtime::cpu_resource_observation.active && cuda_fock_enabled_) {
      // Every bucket cache remains live. Sampling only the most recent
      // bucket would miss retained arenas from earlier ragged shapes.
      std::size_t resident_bytes = 0;
      for (const auto* plan : cuda_bucket_plans_)
        resident_bytes = runtime::add_capacity(resident_bytes, hf_cuda_owned_device_bytes(plan));
      runtime::sample_cuda_arena_capacity(resident_bytes);
    }
    bucket_begin = bucket_end;
  }
  return results;
}

std::size_t FleetPlan::warm_density_size(std::size_t index) const {
  const auto n = molecule::ao_count(systems_.at(index));
  const std::size_t spins = method_ == VIBEQC_METHOD_UHF ? 2 : 1;
  if (n == 0 || n > std::numeric_limits<std::size_t>::max() / n / spins / sizeof(double))
    throw std::invalid_argument("warm density dimensions overflow");
  return spins * n * n;
}

const std::optional<HfWarmState>& FleetPlan::warm_state(std::size_t index) const {
  return warm_densities_.at(index);
}

void FleetPlan::retain_warm_state(std::size_t index, const core::System& system,
                                  const ScfResult& result) {
  auto& retained = warm_densities_[index];
  if (retained && retained->density.size() == result.density.size() &&
      retained->coordinates.size() == 3 * system.atoms.size()) {
    // Fixed topology keeps the existing buffer capacity and pointer stable on
    // warm replays. Nothing in this update can allocate or leave a density
    // paired with the previous geometry after an allocation failure.
    std::copy(result.density.begin(), result.density.end(), retained->density.begin());
    for (std::size_t atom = 0; atom < system.atoms.size(); ++atom)
      std::copy(system.atoms[atom].position.begin(), system.atoms[atom].position.end(),
                retained->coordinates.begin() + 3 * atom);
    retained->energy = result.energy;
    retained->energy_change = result.energy_change;
    retained->density_rms = result.density_rms;
    retained->iterations = result.iterations;
    return;
  }
  HfWarmState state;
  state.density = result.density;
  state.coordinates.reserve(3 * system.atoms.size());
  for (const auto& atom : system.atoms)
    state.coordinates.insert(state.coordinates.end(), atom.position.begin(), atom.position.end());
  state.energy = result.energy;
  state.energy_change = result.energy_change;
  state.density_rms = result.density_rms;
  state.iterations = result.iterations;
  warm_densities_[index] = std::move(state);
}

void FleetPlan::restore_warm_states(std::vector<std::optional<HfWarmState>> states) {
  if (!warm_starts_enabled_ || states.size() != size())
    throw std::invalid_argument("checkpoint restore requires a matching warm-enabled fleet");
  for (std::size_t i = 0; i < size(); ++i) {
    if (!states[i]) continue;
    const auto& state = *states[i];
    if (state.density.size() != warm_density_size(i) ||
        !valid_coordinates(state.coordinates, systems_[i].atoms.size()) || state.iterations < 0 ||
        !std::isfinite(state.energy) || !std::isfinite(state.energy_change) ||
        !std::isfinite(state.density_rms) || state.density_rms < 0)
      throw std::invalid_argument("invalid checkpoint state dimensions or diagnostics");
    auto source = systems_[i];
    apply_coordinates(source, state.coordinates);
    validate_hf_warm_density(source, method_, state.density);
  }
  // Every allocation/validation above completes before this no-throw commit.
  // A pre-existing device seed/energy must never supersede an imported seed.
  for (auto* plan : cuda_bucket_plans_) clear_rhf_cuda_bucket_warm_starts(plan);
  for (std::size_t i = 0; i < size(); ++i)
    if (states[i]) warm_densities_[i].swap(states[i]);
}

void FleetPlan::clear_warm_starts() {
  for (auto& density : warm_densities_) density.reset();
  for (CudaRhfBucketPlan* plan : cuda_bucket_plans_) {
    clear_rhf_cuda_bucket_warm_starts(plan);
  }
}

void FleetPlan::set_warm_start_updates(bool enabled) noexcept {
  if (warm_start_updates_enabled_ == enabled) return;
  // The CUDA plan owns the energy associated with its current returned
  // density. Freeze that pair at the same transition where Fleet stops
  // replacing the corresponding host dm0 snapshots.
  for (CudaRhfBucketPlan* plan : cuda_bucket_plans_) {
    set_rhf_cuda_bucket_warm_start_updates(plan, enabled);
  }
  warm_start_updates_enabled_ = enabled;
}

}  // namespace vibeqc::scf
