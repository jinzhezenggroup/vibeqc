#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "dft/dispersion/d4_eeq.hpp"
#include "vibeqc/vibeqc.h"

namespace vibeqc::dft::dispersion {

inline constexpr char kD4EEQTableSha256[] =
    "f0caaa42a0604d2d0c429e34eb84f5078769058ab02cccc55ae492746c6bc34c";
inline constexpr char kD4EEQChargeParameterSha256[] =
    "02b8bee49c10b4c31914caf149d9f58164e58d6dc7ae2ab21e9d12f5bc22797a";
inline constexpr char kD4ProductionProviderIdentity[] = "vibeqc-native-d4-bj-eeq-v1";
inline constexpr char kD4ProductionSchedulerIdentity[] =
    "bounded-eeq-workers-cooperative-fixed-charge-v1";

struct D4ResourceUsage {
  std::uint64_t plan_host_bytes{};
  std::uint64_t execution_host_bytes{};
  std::uint64_t device_bytes{};
  std::uint64_t table_bytes{};
  std::uint64_t workspace_bytes{};
  std::uint64_t maximum_bytes{};
  std::uint64_t total_atoms{};
  std::uint32_t system_count{};
  std::uint32_t maximum_atoms{};
  std::uint32_t worker_blocks{};
  std::uint32_t workspace_slots{};
};

struct D4RuntimeCounters {
  std::uint64_t execution_count{};
  std::uint64_t unchanged_geometry_replays{};
  std::uint64_t changed_geometry_replays{};
  std::uint64_t coordinate_h2d_bytes{};
  std::uint64_t kernel_launches{};
};

struct D4CudaOwner;

class D4Plan {
 public:
  static std::unique_ptr<D4Plan> prepare(
      vibeqc_backend backend, int device_id, std::vector<std::uint32_t> offsets,
      std::vector<std::int32_t> atomic_numbers, std::vector<double> total_charges,
      std::vector<double> default_coordinates, D4Parameters parameters, D4EEQProfile profile,
      std::uint64_t maximum_bytes, std::string& detail, vibeqc_status& status);
  ~D4Plan();

  D4Plan(const D4Plan&) = delete;
  D4Plan& operator=(const D4Plan&) = delete;

  [[nodiscard]] std::uint32_t system_count() const noexcept {
    return static_cast<std::uint32_t>(offsets_.size() - 1);
  }
  [[nodiscard]] std::uint32_t atom_count(std::uint32_t system) const noexcept {
    return offsets_[system + 1] - offsets_[system];
  }
  [[nodiscard]] vibeqc_backend backend() const noexcept { return backend_; }
  [[nodiscard]] D4EEQProfile profile() const noexcept { return profile_; }
  [[nodiscard]] const D4Parameters& parameters() const noexcept { return parameters_; }
  [[nodiscard]] const D4ResourceUsage& resources() const noexcept { return resources_; }
  [[nodiscard]] const D4RuntimeCounters& counters() const noexcept { return counters_; }
  [[nodiscard]] double total_charge(std::uint32_t system) const noexcept {
    return total_charges_[system];
  }
  [[nodiscard]] std::span<const double> default_coordinates(std::uint32_t system) const noexcept {
    const auto begin = static_cast<std::size_t>(offsets_[system]) * 3;
    return std::span<const double>(default_coordinates_.data() + begin,
                                   static_cast<std::size_t>(atom_count(system)) * 3);
  }

  vibeqc_status execute(std::span<const double> packed_coordinates,
                        std::span<const std::uint8_t> active,
                        std::span<const std::uint8_t> want_gradient,
                        std::vector<D4Status>& statuses, std::vector<double>& energy_components,
                        std::vector<double>& packed_gradients, std::vector<double>& packed_charges,
                        std::string& detail);

 private:
  D4Plan(vibeqc_backend backend, int device_id, std::vector<std::uint32_t> offsets,
         std::vector<std::int32_t> atomic_numbers, std::vector<double> total_charges,
         std::vector<double> default_coordinates, D4Parameters parameters, D4EEQProfile profile,
         D4ResourceUsage resources)
      : backend_(backend),
        device_id_(device_id),
        offsets_(std::move(offsets)),
        atomic_numbers_(std::move(atomic_numbers)),
        total_charges_(std::move(total_charges)),
        default_coordinates_(std::move(default_coordinates)),
        last_coordinates_(default_coordinates_),
        parameters_(parameters),
        profile_(profile),
        resources_(resources) {}

  vibeqc_backend backend_{};
  int device_id_{};
  std::vector<std::uint32_t> offsets_;
  std::vector<std::int32_t> atomic_numbers_;
  std::vector<double> total_charges_;
  std::vector<double> default_coordinates_;
  std::vector<double> last_coordinates_;
  bool cuda_coordinates_valid_{true};
  D4Parameters parameters_{};
  D4EEQProfile profile_{};
  D4ResourceUsage resources_{};
  D4RuntimeCounters counters_{};
  D4CudaOwner* cuda_{};
};

D4CudaOwner* create_d4_cuda_owner(int device_id, std::span<const std::uint32_t> offsets,
                                  std::span<const std::int32_t> atomic_numbers,
                                  std::span<const double> total_charges,
                                  std::span<const double> default_coordinates, D4EEQProfile profile,
                                  const D4ResourceUsage& resources, std::string& detail,
                                  vibeqc_status& status);
void destroy_d4_cuda_owner(D4CudaOwner* owner) noexcept;
vibeqc_status execute_d4_cuda(D4CudaOwner* owner, const D4Parameters& parameters,
                              D4EEQProfile profile, std::span<const double> coordinates,
                              bool coordinates_changed, std::span<const std::uint8_t> active,
                              std::span<const std::uint8_t> want_gradient,
                              std::vector<D4Status>& statuses,
                              std::vector<double>& energy_components,
                              std::vector<double>& gradients, std::vector<double>& charges,
                              D4RuntimeCounters& counters, std::string& detail);

}  // namespace vibeqc::dft::dispersion
