#include "vibeqc/vibeqc.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

vibeqc_system* make_hydrogen_system(vibeqc_context* context,
                                 const std::vector<std::array<double, 3>>& positions,
                                 int charge) {
  std::vector<vibeqc_atom> atoms;
  std::vector<vibeqc_shell> shells;
  std::vector<vibeqc_primitive> primitives;
  atoms.reserve(positions.size());
  shells.reserve(positions.size());
  primitives.reserve(positions.size() * 3);
  for (std::size_t atom = 0; atom < positions.size(); ++atom) {
    atoms.push_back({1, positions[atom][0], positions[atom][1], positions[atom][2]});
    const std::uint32_t offset = static_cast<std::uint32_t>(primitives.size());
    primitives.push_back({3.42525091, 0.15432897});
    primitives.push_back({0.62391373, 0.53532814});
    primitives.push_back({0.16885540, 0.44463454});
    shells.push_back({static_cast<std::uint32_t>(atom), 0, offset, 3});
  }
  vibeqc_system_descriptor descriptor{
      sizeof(vibeqc_system_descriptor), VIBEQC_ABI_VERSION,
      atoms.data(), static_cast<std::uint32_t>(atoms.size()),
      shells.data(), static_cast<std::uint32_t>(shells.size()),
      primitives.data(), static_cast<std::uint32_t>(primitives.size()), charge, 1};
  vibeqc_system* system = nullptr;
  require(vibeqc_system_create(context, &descriptor, &system) == VIBEQC_STATUS_SUCCESS,
          "failed to create hydrogen test system");
  return system;
}

vibeqc_batch_item_result_descriptor output(double* forces, std::uint32_t count) {
  return {sizeof(vibeqc_batch_item_result_descriptor), VIBEQC_ABI_VERSION,
          VIBEQC_STATUS_INTERNAL_ERROR, 0.0, forces, count, 0, 0.0, 0.0, 0,
          VIBEQC_BACKEND_CPU_REFERENCE, 0, 0, 0};
}

}  // namespace

int main() {
  try {
    vibeqc_context_descriptor context_descriptor{
        sizeof(vibeqc_context_descriptor), VIBEQC_ABI_VERSION, 0,
        VIBEQC_BACKEND_CPU_REFERENCE};
    vibeqc_context* context = nullptr;
    require(vibeqc_context_create(&context_descriptor, &context) == VIBEQC_STATUS_SUCCESS,
            "batch context creation failed");

    vibeqc_system* h2 = make_hydrogen_system(
        context, {{{0.0, 0.0, -0.7}}, {{0.0, 0.0, 0.7}}}, 0);
    vibeqc_system* h3_plus = make_hydrogen_system(
        context, {{{-1.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}, {{1.0, 0.0, 0.0}}}, 1);
    vibeqc_system* h4 = make_hydrogen_system(
        context,
        {{{-1.0, -1.0, 0.0}}, {{-1.0, 1.0, 0.0}},
         {{1.0, -1.0, 0.0}}, {{1.0, 1.0, 0.0}}},
        0);
    const std::array<const vibeqc_system*, 3> systems{{h2, h4, h3_plus}};

    vibeqc_method_descriptor method{
        sizeof(vibeqc_method_descriptor), VIBEQC_ABI_VERSION, VIBEQC_METHOD_RHF,
        100, 8, 1.0e-12, 1.0e-10, 1.0e-14};
    vibeqc_batch* batch = nullptr;
    require(vibeqc_batch_prepare(context, systems.data(), systems.size(), &method,
                              VIBEQC_BATCH_ENABLE_WARM_STARTS, &batch) ==
                VIBEQC_STATUS_SUCCESS,
            "batch preparation failed");
    require(vibeqc_batch_get_system_count(batch) == systems.size(),
            "batch system count is incorrect");
    require(vibeqc_batch_set_warm_start_updates(nullptr, 0) ==
                VIBEQC_STATUS_INVALID_ARGUMENT &&
                vibeqc_batch_set_warm_start_updates(batch, 2) ==
                    VIBEQC_STATUS_INVALID_ARGUMENT,
            "warm-start update policy accepted an invalid argument");

    std::array<double, 6> h2_forces{};
    std::array<double, 12> h4_forces{};
    std::array<double, 9> h3_forces{};
    std::array<vibeqc_batch_item_result_descriptor, 3> first{{
        output(h2_forces.data(), h2_forces.size()),
        output(h4_forces.data(), h4_forces.size()),
        output(h3_forces.data(), h3_forces.size()),
    }};
    require(vibeqc_batch_execute(batch, nullptr, 0, first.data(), first.size()) ==
                VIBEQC_STATUS_SUCCESS,
            "first batch execution failed structurally");
    std::array<vibeqc_shell_class_profile_entry,
               VIBEQC_DIRECT_SHELL_CLASS_COUNT> profile{};
    require(vibeqc_batch_get_last_shell_class_profile(
                batch, profile.data(), profile.size()) ==
                VIBEQC_STATUS_NOT_IMPLEMENTED,
            "a non-profiled batch unexpectedly published shell-class data");
    vibeqc_ppps_queue_profile ppps_profile{};
    require(vibeqc_batch_get_last_ppps_queue_profile(batch, &ppps_profile) ==
                VIBEQC_STATUS_NOT_IMPLEMENTED &&
                vibeqc_batch_get_last_ppps_queue_profile(nullptr,
                                                         &ppps_profile) ==
                    VIBEQC_STATUS_INVALID_ARGUMENT,
            "a non-profiled batch unexpectedly published PPPS queue data");
    std::uint32_t eigensolver_diagnostic_count = 0;
    require(vibeqc_batch_get_last_eigensolver_diagnostics(
                batch, nullptr, 0, &eigensolver_diagnostic_count) ==
                VIBEQC_STATUS_NOT_IMPLEMENTED &&
                vibeqc_batch_get_last_eigensolver_diagnostics(
                    nullptr, nullptr, 0, &eigensolver_diagnostic_count) ==
                    VIBEQC_STATUS_INVALID_ARGUMENT &&
                vibeqc_batch_get_last_eigensolver_diagnostics(
                    batch, nullptr, 1, &eigensolver_diagnostic_count) ==
                    VIBEQC_STATUS_INVALID_ARGUMENT,
            "CPU batch unexpectedly published CUDA eigensolver evidence");
    std::uint32_t inactive_profile_count = 0;
    require(vibeqc_batch_get_last_inactive_eigensolver_profile(
                batch, nullptr, 0, &inactive_profile_count) ==
                VIBEQC_STATUS_NOT_IMPLEMENTED &&
                vibeqc_batch_get_last_inactive_eigensolver_profile(
                    nullptr, nullptr, 0, &inactive_profile_count) ==
                    VIBEQC_STATUS_INVALID_ARGUMENT &&
                vibeqc_batch_get_last_inactive_eigensolver_profile(
                    batch, nullptr, 1, &inactive_profile_count) ==
                    VIBEQC_STATUS_INVALID_ARGUMENT,
            "CPU batch unexpectedly published inactive eigensolver data");
    for (const auto& item : first) {
      require(item.status == VIBEQC_STATUS_SUCCESS && item.converged == 1,
              "a valid first-run batch item failed");
      require(item.warm_start_used == 0, "first execution unexpectedly used a warm start");
    }
    require(std::abs(first[0].energy - (-1.11671432506255)) < 2.0e-9,
            "ragged batch H2 energy is incorrect");
    require(first[0].bucket_id != first[1].bucket_id &&
                first[0].bucket_id != first[2].bucket_id &&
                first[1].bucket_id != first[2].bucket_id,
            "different workload shapes were not assigned distinct buckets");

    // Freeze the post-cold snapshots so every warm replay uses one fixed dm0,
    // matching controlled backend-comparison benchmark semantics.
    require(vibeqc_batch_set_warm_start_updates(batch, 0) ==
                VIBEQC_STATUS_SUCCESS,
            "failed to freeze batch warm-start snapshots");

    std::array<vibeqc_batch_item_result_descriptor, 3> second{{
        output(h2_forces.data(), h2_forces.size()),
        output(h4_forces.data(), h4_forces.size()),
        output(h3_forces.data(), h3_forces.size()),
    }};
    require(vibeqc_batch_execute(batch, nullptr, 0, second.data(), second.size()) ==
                VIBEQC_STATUS_SUCCESS,
            "warm batch execution failed structurally");
    for (std::size_t i = 0; i < second.size(); ++i) {
      require(second[i].status == VIBEQC_STATUS_SUCCESS &&
                  second[i].warm_start_used == 1,
              "prepared batch did not reuse a converged per-system density");
      require(second[i].iterations <= first[i].iterations,
              "warm start increased the SCF iteration count");
    }
    require(second[2].iterations < first[2].iterations,
            "the nontrivial H3+ workload did not benefit from its warm start");

    std::array<double, 6> invalid_h2_coordinates{
        0.0, 0.0, std::numeric_limits<double>::quiet_NaN(),
        0.0, 0.0, 0.7};
    std::array<vibeqc_batch_input_descriptor, 3> inputs{{
        {sizeof(vibeqc_batch_input_descriptor), VIBEQC_ABI_VERSION,
         invalid_h2_coordinates.data(), invalid_h2_coordinates.size()},
        {sizeof(vibeqc_batch_input_descriptor), VIBEQC_ABI_VERSION, nullptr, 0},
        {sizeof(vibeqc_batch_input_descriptor), VIBEQC_ABI_VERSION, nullptr, 0},
    }};
    std::array<vibeqc_batch_item_result_descriptor, 3> isolated{{
        output(h2_forces.data(), h2_forces.size()),
        output(h4_forces.data(), h4_forces.size()),
        output(h3_forces.data(), h3_forces.size()),
    }};
    require(vibeqc_batch_execute(batch, inputs.data(), inputs.size(), isolated.data(),
                              isolated.size()) == VIBEQC_STATUS_SUCCESS,
            "an item-level failure incorrectly aborted the batch call");
    require(isolated[0].status == VIBEQC_STATUS_INVALID_ARGUMENT,
            "invalid coordinates were not isolated to their item");
    require(isolated[1].status == VIBEQC_STATUS_SUCCESS &&
                isolated[2].status == VIBEQC_STATUS_SUCCESS,
            "one failed system prevented valid neighbors from completing");

    require(vibeqc_batch_clear_warm_starts(batch) == VIBEQC_STATUS_SUCCESS,
            "failed to clear batch warm starts");
    std::array<vibeqc_batch_item_result_descriptor, 3> cold_again{{
        output(h2_forces.data(), h2_forces.size()),
        output(h4_forces.data(), h4_forces.size()),
        output(h3_forces.data(), h3_forces.size()),
    }};
    require(vibeqc_batch_execute(batch, nullptr, 0, cold_again.data(),
                              cold_again.size()) == VIBEQC_STATUS_SUCCESS,
            "cold batch execution after clearing warm state failed");
    for (const auto& item : cold_again) {
      require(item.warm_start_used == 0,
              "cleared warm-start state was unexpectedly reused");
    }

    vibeqc_batch_destroy(batch);
    vibeqc_system_destroy(h2);
    vibeqc_system_destroy(h3_plus);
    vibeqc_system_destroy(h4);
    vibeqc_context_destroy(context);
    std::cout << "ragged batch, bucketing, isolation, and warm starts: PASS\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
