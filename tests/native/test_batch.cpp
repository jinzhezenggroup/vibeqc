#include "qce/qce.h"

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

qce_system* make_hydrogen_system(qce_context* context,
                                 const std::vector<std::array<double, 3>>& positions,
                                 int charge) {
  std::vector<qce_atom> atoms;
  std::vector<qce_shell> shells;
  std::vector<qce_primitive> primitives;
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
  qce_system_descriptor descriptor{
      sizeof(qce_system_descriptor), QCE_ABI_VERSION,
      atoms.data(), static_cast<std::uint32_t>(atoms.size()),
      shells.data(), static_cast<std::uint32_t>(shells.size()),
      primitives.data(), static_cast<std::uint32_t>(primitives.size()), charge, 1};
  qce_system* system = nullptr;
  require(qce_system_create(context, &descriptor, &system) == QCE_STATUS_SUCCESS,
          "failed to create hydrogen test system");
  return system;
}

qce_batch_item_result_descriptor output(double* forces, std::uint32_t count) {
  return {sizeof(qce_batch_item_result_descriptor), QCE_ABI_VERSION,
          QCE_STATUS_INTERNAL_ERROR, 0.0, forces, count, 0, 0.0, 0.0, 0,
          QCE_BACKEND_CPU_REFERENCE, 0, 0, 0};
}

}  // namespace

int main() {
  try {
    qce_context_descriptor context_descriptor{
        sizeof(qce_context_descriptor), QCE_ABI_VERSION, 0,
        QCE_BACKEND_CPU_REFERENCE};
    qce_context* context = nullptr;
    require(qce_context_create(&context_descriptor, &context) == QCE_STATUS_SUCCESS,
            "batch context creation failed");

    qce_system* h2 = make_hydrogen_system(
        context, {{{0.0, 0.0, -0.7}}, {{0.0, 0.0, 0.7}}}, 0);
    qce_system* h3_plus = make_hydrogen_system(
        context, {{{-1.0, 0.0, 0.0}}, {{0.0, 0.0, 0.0}}, {{1.0, 0.0, 0.0}}}, 1);
    qce_system* h4 = make_hydrogen_system(
        context,
        {{{-1.0, -1.0, 0.0}}, {{-1.0, 1.0, 0.0}},
         {{1.0, -1.0, 0.0}}, {{1.0, 1.0, 0.0}}},
        0);
    const std::array<const qce_system*, 3> systems{{h2, h4, h3_plus}};

    qce_method_descriptor method{
        sizeof(qce_method_descriptor), QCE_ABI_VERSION, QCE_METHOD_RHF,
        100, 8, 1.0e-12, 1.0e-10, 1.0e-14};
    qce_batch* batch = nullptr;
    require(qce_batch_prepare(context, systems.data(), systems.size(), &method,
                              QCE_BATCH_ENABLE_WARM_STARTS, &batch) ==
                QCE_STATUS_SUCCESS,
            "batch preparation failed");
    require(qce_batch_get_system_count(batch) == systems.size(),
            "batch system count is incorrect");

    std::array<double, 6> h2_forces{};
    std::array<double, 12> h4_forces{};
    std::array<double, 9> h3_forces{};
    std::array<qce_batch_item_result_descriptor, 3> first{{
        output(h2_forces.data(), h2_forces.size()),
        output(h4_forces.data(), h4_forces.size()),
        output(h3_forces.data(), h3_forces.size()),
    }};
    require(qce_batch_execute(batch, nullptr, 0, first.data(), first.size()) ==
                QCE_STATUS_SUCCESS,
            "first batch execution failed structurally");
    std::array<qce_shell_class_profile_entry,
               QCE_DIRECT_SHELL_CLASS_COUNT> profile{};
    require(qce_batch_get_last_shell_class_profile(
                batch, profile.data(), profile.size()) ==
                QCE_STATUS_NOT_IMPLEMENTED,
            "a non-profiled batch unexpectedly published shell-class data");
    for (const auto& item : first) {
      require(item.status == QCE_STATUS_SUCCESS && item.converged == 1,
              "a valid first-run batch item failed");
      require(item.warm_start_used == 0, "first execution unexpectedly used a warm start");
    }
    require(std::abs(first[0].energy - (-1.11671432506255)) < 2.0e-9,
            "ragged batch H2 energy is incorrect");
    require(first[0].bucket_id != first[1].bucket_id &&
                first[0].bucket_id != first[2].bucket_id &&
                first[1].bucket_id != first[2].bucket_id,
            "different workload shapes were not assigned distinct buckets");

    std::array<qce_batch_item_result_descriptor, 3> second{{
        output(h2_forces.data(), h2_forces.size()),
        output(h4_forces.data(), h4_forces.size()),
        output(h3_forces.data(), h3_forces.size()),
    }};
    require(qce_batch_execute(batch, nullptr, 0, second.data(), second.size()) ==
                QCE_STATUS_SUCCESS,
            "warm batch execution failed structurally");
    for (std::size_t i = 0; i < second.size(); ++i) {
      require(second[i].status == QCE_STATUS_SUCCESS &&
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
    std::array<qce_batch_input_descriptor, 3> inputs{{
        {sizeof(qce_batch_input_descriptor), QCE_ABI_VERSION,
         invalid_h2_coordinates.data(), invalid_h2_coordinates.size()},
        {sizeof(qce_batch_input_descriptor), QCE_ABI_VERSION, nullptr, 0},
        {sizeof(qce_batch_input_descriptor), QCE_ABI_VERSION, nullptr, 0},
    }};
    std::array<qce_batch_item_result_descriptor, 3> isolated{{
        output(h2_forces.data(), h2_forces.size()),
        output(h4_forces.data(), h4_forces.size()),
        output(h3_forces.data(), h3_forces.size()),
    }};
    require(qce_batch_execute(batch, inputs.data(), inputs.size(), isolated.data(),
                              isolated.size()) == QCE_STATUS_SUCCESS,
            "an item-level failure incorrectly aborted the batch call");
    require(isolated[0].status == QCE_STATUS_INVALID_ARGUMENT,
            "invalid coordinates were not isolated to their item");
    require(isolated[1].status == QCE_STATUS_SUCCESS &&
                isolated[2].status == QCE_STATUS_SUCCESS,
            "one failed system prevented valid neighbors from completing");

    require(qce_batch_clear_warm_starts(batch) == QCE_STATUS_SUCCESS,
            "failed to clear batch warm starts");
    std::array<qce_batch_item_result_descriptor, 3> cold_again{{
        output(h2_forces.data(), h2_forces.size()),
        output(h4_forces.data(), h4_forces.size()),
        output(h3_forces.data(), h3_forces.size()),
    }};
    require(qce_batch_execute(batch, nullptr, 0, cold_again.data(),
                              cold_again.size()) == QCE_STATUS_SUCCESS,
            "cold batch execution after clearing warm state failed");
    for (const auto& item : cold_again) {
      require(item.warm_start_used == 0,
              "cleared warm-start state was unexpectedly reused");
    }

    qce_batch_destroy(batch);
    qce_system_destroy(h2);
    qce_system_destroy(h3_plus);
    qce_system_destroy(h4);
    qce_context_destroy(context);
    std::cout << "ragged batch, bucketing, isolation, and warm starts: PASS\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
