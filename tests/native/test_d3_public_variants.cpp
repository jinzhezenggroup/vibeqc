#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/dispersion/d3_zero.hpp"
#include "vibeqc/vibeqc.h"

namespace {

constexpr std::array<std::int32_t, 4> kZeroNumbers{6, 8, 1, 17};
constexpr std::array<double, 12> kZeroCoordinates{0.1,  0.3, -0.2, 2.4, -0.6, 0.8,
                                                  -1.7, 0.8, 0.0,  3.7, 2.1,  1.3};
constexpr double kZeroOracleEnergy = -4.742941049083972e-04;
constexpr std::array<double, 12> kZeroOracleGradient{
    7.264892589154373e-05,   3.3364010522326386e-05,  3.071926911995039e-05,
    -1.1415410870098772e-04, 4.7324964067888234e-05,  -2.2581043945442164e-05,
    3.1066772795464696e-05,  -5.9017428411558095e-05, 2.0461347070702326e-06,
    1.0438410013979293e-05,  -2.167154617865652e-05,  -1.0184359881578459e-05};

constexpr std::array<std::int32_t, 4> kAtmNumbers{6, 8, 7, 1};
constexpr std::array<double, 12> kAtmCoordinates{0.0, 0.0, 0.0, 2.5,  0.1, 0.0,
                                                 0.6, 2.7, 0.2, -1.2, 0.8, 2.4};
constexpr double kAtmOracleEnergy = 3.3280853885481534e-08;
constexpr std::array<double, 12> kAtmOracleGradient{
    3.72472344498586336e-09,  -1.85479558309227088e-08, -3.61045307549533526e-08,
    2.60708265967875448e-08,  -2.63615669102038128e-08, 2.56660793687209677e-09,
    -3.96599503356355378e-09, 5.05725536818215585e-08,  -1.60861908260602298e-08,
    -2.58295550081963018e-08, -5.66303094069503694e-09, 4.96241136441330152e-08};

struct Context {
  vibeqc_context* value{};
  Context() {
    vibeqc_context_descriptor descriptor{sizeof(vibeqc_context_descriptor), VIBEQC_ABI_VERSION, 0,
                                         VIBEQC_BACKEND_CPU_REFERENCE};
    if (vibeqc_context_create(&descriptor, &value) != VIBEQC_STATUS_SUCCESS)
      throw std::runtime_error("D3 public-test context creation failed");
  }
  ~Context() { vibeqc_context_destroy(value); }
  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;
};

struct PublicResult {
  double energy{};
  std::array<double, 12> gradient{};
  std::string variant;
};

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void require_close(double actual, double expected, double tolerance, const char* message) {
  if (std::abs(actual - expected) > tolerance) {
    std::cerr << message << ": expected " << expected << ", got " << actual << '\n';
    throw std::runtime_error(message);
  }
}

template <std::size_t N>
PublicResult execute_public(Context& context, const std::array<std::int32_t, N>& numbers,
                            const std::array<double, 3 * N>& prepared,
                            const vibeqc_d3_bj_descriptor& model,
                            const std::array<double, 3 * N>* changed = nullptr,
                            bool want_gradient = true) {
  vibeqc_d3_system_descriptor system{sizeof(vibeqc_d3_system_descriptor), VIBEQC_ABI_VERSION,
                                     numbers.data(), prepared.data(),
                                     static_cast<std::uint32_t>(numbers.size())};
  vibeqc_d3_batch* batch{};
  const auto prepare_status = vibeqc_d3_batch_prepare(context.value, &system, 1, &model, &batch);
  if (prepare_status != VIBEQC_STATUS_SUCCESS) {
    const auto* detail = vibeqc_context_get_last_detail(context.value);
    throw std::runtime_error(detail ? detail : "D3 public prepare failed");
  }
  const char* raw_variant = vibeqc_d3_batch_variant_identity(batch);
  require(raw_variant != nullptr, "D3 public variant identity is missing");

  std::array<double, 3 * N> gradient{};
  vibeqc_d3_batch_item_result_descriptor result{
      sizeof(vibeqc_d3_batch_item_result_descriptor),
      VIBEQC_ABI_VERSION,
      VIBEQC_STATUS_INTERNAL_ERROR,
      0.0,
      want_gradient ? gradient.data() : nullptr,
      want_gradient ? static_cast<std::uint32_t>(gradient.size()) : 0u,
      VIBEQC_BACKEND_CPU_REFERENCE};
  vibeqc_d3_batch_input_descriptor input{
      sizeof(vibeqc_d3_batch_input_descriptor), VIBEQC_ABI_VERSION,
      changed ? changed->data() : nullptr,
      changed ? static_cast<std::uint32_t>(changed->size()) : 0u};
  const auto status =
      vibeqc_d3_batch_execute(batch, changed ? &input : nullptr, changed ? 1u : 0u, &result, 1u);
  std::string variant(raw_variant);
  vibeqc_d3_batch_destroy(batch);
  require(status == VIBEQC_STATUS_SUCCESS && result.status == VIBEQC_STATUS_SUCCESS,
          "D3 public execution failed");
  require(result.executed_backend == VIBEQC_BACKEND_CPU_REFERENCE,
          "D3 public execution reported the wrong backend");

  PublicResult output{};
  output.energy = result.energy;
  output.gradient = gradient;
  output.variant = std::move(variant);
  return output;
}

vibeqc_d3_bj_descriptor zero_model(std::uint64_t maximum_bytes = 64u << 20) {
  return {sizeof(vibeqc_d3_bj_descriptor),
          VIBEQC_ABI_VERSION,
          VIBEQC_D3_DAMPING_ZERO,
          1.0,
          0.722,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          maximum_bytes,
          1.217,
          1.0,
          14.0,
          0.0,
          0.0};
}

vibeqc_d3_bj_descriptor bj_model(double s9) {
  return {sizeof(vibeqc_d3_bj_descriptor),
          VIBEQC_ABI_VERSION,
          VIBEQC_D3_DAMPING_BJ,
          1.0,
          0.7875,
          0.4289,
          4.4407,
          s9,
          0.0,
          0.0,
          0.0,
          64u << 20,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0};
}

void test_identities() {
  require(std::strcmp(vibeqc_d3_provider_identity(), "vibeqc-native-d3-v2") == 0,
          "unexpected D3 provider identity");
  require(std::strcmp(vibeqc_d3_scheduler_identity(), "ragged-system-cooperative-pair-v1") == 0,
          "unexpected D3 scheduler identity");
  require(std::strlen(vibeqc_d3_table_sha256()) == 64, "missing D3 table identity");
  require(std::strlen(vibeqc_d3_radii_sha256()) == 64, "missing D3 radii identity");
}

void test_zero_public_oracle_and_replay(Context& context) {
  const auto model = zero_model();
  const auto result = execute_public(context, kZeroNumbers, kZeroCoordinates, model);
  require(result.variant == "d3.zero-two-body", "zero-damping variant identity mismatch");
  require_close(result.energy, kZeroOracleEnergy, 3.0e-15, "public zero oracle energy mismatch");
  for (std::size_t q = 0; q < result.gradient.size(); ++q)
    require_close(result.gradient[q], kZeroOracleGradient[q], 3.0e-13,
                  "public zero oracle gradient mismatch");

  auto changed = kZeroCoordinates;
  changed[0] += 0.031;
  changed[7] -= 0.019;
  const auto replay = execute_public(context, kZeroNumbers, kZeroCoordinates, model, &changed);

  using namespace vibeqc::dft::dispersion;
  std::vector<double> workspace(d3_workspace_elements(kZeroNumbers.size()));
  double reference_energy{};
  std::array<double, 12> reference_gradient{};
  const auto zero = D3ZeroParameters{1.0, 0.722, 1.217, 1.0, 14.0, 0.0, 0.0, 0.0};
  require(evaluate_d3_zero(kZeroNumbers.size(), kZeroNumbers.data(), changed.data(), zero,
                           d3_host_tables(), workspace.data(), workspace.size(), &reference_energy,
                           reference_gradient.data()) == D3Status::success,
          "zero changed-geometry reference failed");
  require_close(replay.energy, reference_energy, 3.0e-15, "zero changed-geometry energy mismatch");
  for (std::size_t q = 0; q < replay.gradient.size(); ++q)
    require_close(replay.gradient[q], reference_gradient[q], 3.0e-13,
                  "zero changed-geometry gradient mismatch");

  const auto energy_only =
      execute_public(context, kZeroNumbers, kZeroCoordinates, model, nullptr, false);
  require_close(energy_only.energy, kZeroOracleEnergy, 3.0e-15,
                "zero energy-only publication mismatch");
}

void test_atm_exactly_once(Context& context) {
  const auto two_body = execute_public(context, kAtmNumbers, kAtmCoordinates, bj_model(0.0));
  const auto with_atm = execute_public(context, kAtmNumbers, kAtmCoordinates, bj_model(1.0));
  require(two_body.variant == "d3.bj-two-body", "BJ variant identity mismatch");
  require(with_atm.variant == "d3.bj-atm", "BJ+ATM variant identity mismatch");
  require_close(with_atm.energy - two_body.energy, kAtmOracleEnergy, 2.0e-18,
                "ATM public composition is missing or duplicated");
  for (std::size_t q = 0; q < kAtmOracleGradient.size(); ++q)
    require_close(with_atm.gradient[q] - two_body.gradient[q], kAtmOracleGradient[q], 3.0e-17,
                  "ATM public gradient composition is missing or duplicated");
}

void test_fail_closed_and_resources(Context& context) {
  vibeqc_d3_system_descriptor system{sizeof(vibeqc_d3_system_descriptor), VIBEQC_ABI_VERSION,
                                     kZeroNumbers.data(), kZeroCoordinates.data(),
                                     static_cast<std::uint32_t>(kZeroNumbers.size())};
  vibeqc_d3_batch* batch{};

  auto unsupported = zero_model();
  unsupported.s9 = 1.0;
  require(vibeqc_d3_batch_prepare(context.value, &system, 1, &unsupported, &batch) ==
                  VIBEQC_STATUS_NOT_IMPLEMENTED &&
              batch == nullptr,
          "zero+ATM must fail closed");

  unsupported = zero_model();
  unsupported.damping = 99;
  require(vibeqc_d3_batch_prepare(context.value, &system, 1, &unsupported, &batch) ==
                  VIBEQC_STATUS_NOT_IMPLEMENTED &&
              batch == nullptr,
          "unknown D3 damping must fail closed");

  auto tiny = zero_model(1);
  require(vibeqc_d3_batch_prepare(context.value, &system, 1, &tiny, &batch) ==
                  VIBEQC_STATUS_OUT_OF_MEMORY &&
              batch == nullptr,
          "D3 maximum_bytes must fail closed");

  auto legacy = bj_model(0.0);
  legacy.struct_size = static_cast<std::uint32_t>(offsetof(vibeqc_d3_bj_descriptor, rs6));
  require(vibeqc_d3_batch_prepare(context.value, &system, 1, &legacy, &batch) ==
                  VIBEQC_STATUS_SUCCESS &&
              batch != nullptr,
          "legacy BJ descriptor prefix stopped working");
  require(std::strcmp(vibeqc_d3_batch_variant_identity(batch), "d3.bj-two-body") == 0,
          "legacy BJ descriptor acquired the wrong capability");
  vibeqc_d3_batch_destroy(batch);
}

}  // namespace

int main() {
  try {
    test_identities();
    Context context;
    test_zero_public_oracle_and_replay(context);
    test_atm_exactly_once(context);
    test_fail_closed_and_resources(context);
    std::cout << "D3 public zero/ATM capability, replay, provenance, exactly-once and "
                 "fail-closed tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
