#include "scf/fleet.hpp"

#include "molecule/basis.hpp"
#include "scf/mean_field.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <map>
#include <new>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <thread>
#include <utility>

namespace vibeqc::scf {
namespace {

using WorkloadKey = std::tuple<std::size_t, int, int, std::size_t>;

WorkloadKey workload_key(const core::System& system, vibeqc_method method) {
  std::size_t primitive_count = 0;
  for (const auto& shell : system.shells) {
    primitive_count += shell.primitives.size();
  }
  const int spin_excess = static_cast<int>(system.multiplicity) - 1;
  const int alpha = method == VIBEQC_METHOD_UHF
      ? (system.electron_count + spin_excess) / 2
      : system.electron_count / 2;
  const int beta = method == VIBEQC_METHOD_UHF
      ? system.electron_count - alpha
      : alpha;
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

bool valid_coordinates(const std::vector<double>& coordinates,
                       std::size_t atom_count) {
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

}  // namespace

FleetPlan::FleetPlan(std::vector<core::System> systems,
                     vibeqc_method method,
                     ScfOptions options,
                     bool warm_starts_enabled,
                     bool cuda_fock_enabled,
                     bool shell_class_profiling_enabled,
                     int device_id)
    : systems_(std::move(systems)),
      method_(method),
      options_(options),
      warm_starts_enabled_(warm_starts_enabled),
      cuda_fock_enabled_(cuda_fock_enabled),
      shell_class_profiling_enabled_(shell_class_profiling_enabled),
      device_id_(device_id),
      execution_order_(systems_.size()),
      bucket_ids_(systems_.size()),
      warm_densities_(systems_.size()) {
  std::iota(execution_order_.begin(), execution_order_.end(), 0);
  std::stable_sort(execution_order_.begin(), execution_order_.end(),
                   [&](std::size_t a, std::size_t b) {
                     return workload_key(systems_[a], method_) <
                            workload_key(systems_[b], method_);
                   });

  std::map<WorkloadKey, std::size_t> buckets;
  for (const std::size_t system_index : execution_order_) {
    const WorkloadKey key = workload_key(systems_[system_index], method_);
    auto [iterator, inserted] = buckets.emplace(key, buckets.size());
    (void)inserted;
    bucket_ids_[system_index] = iterator->second;
  }
  cuda_bucket_plans_.resize(buckets.size(), nullptr);
}

FleetPlan::~FleetPlan() {
  for (CudaRhfBucketPlan* plan : cuda_bucket_plans_) {
    destroy_rhf_cuda_bucket_plan(plan);
  }
}

std::vector<FleetItemResult> FleetPlan::execute(
    const std::vector<std::optional<std::vector<double>>>& coordinates) {
  if (!coordinates.empty() && coordinates.size() != systems_.size()) {
    throw std::invalid_argument("fleet coordinate list does not match system count");
  }
  last_shell_class_profile_.reset();
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

    const bool has_warm_density =
        warm_starts_enabled_ && warm_densities_[system_index].has_value();
    item.warm_start_used = has_warm_density;
    try {
      item.scf = method_ == VIBEQC_METHOD_UHF
          ? run_uhf(execution_system, options_,
                    has_warm_density ? &*warm_densities_[system_index] : nullptr)
          : run_rhf(execution_system, options_,
                    has_warm_density ? &*warm_densities_[system_index] : nullptr);
      if (has_warm_density && !item.scf.converged) {
        // A geometry change can make an otherwise topology-compatible density
        // a poor numerical guess. Retry cold so warm starts never reduce the
        // robustness of independent fleet items.
        item.warm_start_fallback = true;
        item.scf = method_ == VIBEQC_METHOD_UHF
            ? run_uhf(execution_system, options_, nullptr)
            : run_rhf(execution_system, options_, nullptr);
      }
      item.status = item.scf.converged ? VIBEQC_STATUS_SUCCESS
                                       : VIBEQC_STATUS_SCF_NOT_CONVERGED;
    } catch (...) {
      if (has_warm_density) {
        try {
          item.warm_start_fallback = true;
          item.scf = method_ == VIBEQC_METHOD_UHF
              ? run_uhf(execution_system, options_, nullptr)
              : run_rhf(execution_system, options_, nullptr);
          item.status = item.scf.converged ? VIBEQC_STATUS_SUCCESS
                                           : VIBEQC_STATUS_SCF_NOT_CONVERGED;
        } catch (...) {
          item.status = exception_status();
        }
      } else {
        item.status = exception_status();
      }
    }

    if (item.status == VIBEQC_STATUS_SUCCESS && warm_starts_enabled_) {
      warm_densities_[system_index] = item.scf.density;
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
    const std::size_t hardware_threads =
        std::max<unsigned>(1, std::thread::hardware_concurrency());
    const std::size_t worker_count = std::min(bucket_size, hardware_threads);
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
          if (!valid_coordinates(*coordinates[system_index],
                                 execution_system.atoms.size())) {
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
        initial_densities.push_back(
            has_warm_density ? &*warm_densities_[system_index] : nullptr);
      }

      if (!cuda_systems.empty()) {
        std::vector<RhfBucketItem> cuda_results = method_ == VIBEQC_METHOD_UHF
            ? run_uhf_cuda_bucket_cached(
                  &cuda_bucket_plans_[bucket], cuda_systems, options_,
                  initial_densities, device_id_,
                  shell_class_profiling_enabled_)
            : run_rhf_cuda_bucket_cached(
                  &cuda_bucket_plans_[bucket], cuda_systems, options_,
                  initial_densities, device_id_,
                  shell_class_profiling_enabled_);
        if (shell_class_profiling_enabled_) {
          CudaRhfShellClassProfile bucket_profile{};
          if (get_rhf_cuda_shell_class_profile(
                  cuda_bucket_plans_[bucket], bucket_profile)) {
            if (!last_shell_class_profile_.has_value()) {
              last_shell_class_profile_.emplace();
            }
            for (std::size_t shell_class = 0;
                 shell_class < bucket_profile.size(); ++shell_class) {
              auto& aggregate = (*last_shell_class_profile_)[shell_class];
              const auto& entry = bucket_profile[shell_class];
              aggregate.shell_quartets += entry.shell_quartets;
              aggregate.tiles += entry.tiles;
              aggregate.ao_quartets += entry.ao_quartets;
              aggregate.primitive_quartets += entry.primitive_quartets;
            }
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
            std::vector<RhfBucketItem> cold = method_ == VIBEQC_METHOD_UHF
                ? run_uhf_cuda_bucket(
                      cold_system, options_, cold_density, device_id_, false)
                : run_rhf_cuda_bucket(
                      cold_system, options_, cold_density, device_id_, false);
            item.status = cold.front().status;
            item.scf = std::move(cold.front().scf);
          }
          if (item.status == VIBEQC_STATUS_SUCCESS && warm_starts_enabled_) {
            warm_densities_[system_index] = item.scf.density;
          }
        }
      }
    } else if (worker_count == 1) {
      execute_one(execution_order_[bucket_begin]);
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
    bucket_begin = bucket_end;
  }
  return results;
}

void FleetPlan::clear_warm_starts() {
  for (auto& density : warm_densities_) density.reset();
}

}  // namespace vibeqc::scf
