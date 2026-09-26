#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/dispersion/d3_model.hpp"
#include "vibeqc/vibeqc.h"

namespace vibeqc::dft::dispersion {

inline constexpr const char* kD3ProductionProviderIdentity = "vibeqc-native-d3-v2";
inline constexpr const char* kD3ProductionSchedulerIdentity = "ragged-system-cooperative-pair-v1";

// Aggregate ragged storage is not subject to the per-system physics cap.
inline std::size_t d3_ragged_workspace_elements(std::size_t atoms) {
  const auto per_atom = d3_workspace_elements(1);
  if (atoms > std::numeric_limits<std::size_t>::max() / sizeof(double) / per_atom)
    throw std::overflow_error("D3 aggregate workspace byte extent overflow");
  return per_atom * atoms;
}

struct D3ResourceUsage {
  std::uint64_t plan_host_bytes{};
  std::uint64_t execution_host_bytes{};
  std::uint64_t device_bytes{};
  std::uint64_t table_bytes{};
  std::uint64_t workspace_bytes{};
  std::uint64_t maximum_bytes{};
  std::uint64_t total_atoms{};
  std::uint32_t system_count{};
  std::uint32_t maximum_atoms{};
};

struct D3CudaOwner;

class D3Plan {
 public:
  static std::unique_ptr<D3Plan> prepare(vibeqc_backend backend, int device_id,
                                         std::vector<std::uint32_t> offsets,
                                         std::vector<std::int32_t> atomic_numbers,
                                         std::vector<double> default_coordinates,
                                         D3ModelParameters parameters, std::uint64_t maximum_bytes,
                                         std::string& detail, vibeqc_status& status);
  static std::unique_ptr<D3Plan> prepare(vibeqc_backend backend, int device_id,
                                         std::vector<std::uint32_t> offsets,
                                         std::vector<std::int32_t> atomic_numbers,
                                         std::vector<double> default_coordinates,
                                         D3Parameters parameters, std::uint64_t maximum_bytes,
                                         std::string& detail, vibeqc_status& status) {
    D3ModelParameters model{};
    model.damping = D3Damping::bj;
    model.bj = parameters;
    model.atm.s9 = 0.0;
    return prepare(backend, device_id, std::move(offsets), std::move(atomic_numbers),
                   std::move(default_coordinates), model, maximum_bytes, detail, status);
  }
  ~D3Plan();

  D3Plan(const D3Plan&) = delete;
  D3Plan& operator=(const D3Plan&) = delete;

  [[nodiscard]] std::uint32_t system_count() const noexcept {
    return static_cast<std::uint32_t>(offsets_.size() - 1);
  }
  [[nodiscard]] std::uint32_t atom_count(std::uint32_t system) const noexcept {
    return offsets_[system + 1] - offsets_[system];
  }
  [[nodiscard]] vibeqc_backend backend() const noexcept { return backend_; }
  [[nodiscard]] const D3ModelParameters& parameters() const noexcept { return parameters_; }
  [[nodiscard]] const D3ResourceUsage& resources() const noexcept { return resources_; }
  [[nodiscard]] std::span<const double> default_coordinates(std::uint32_t system) const noexcept {
    const auto begin = static_cast<std::size_t>(offsets_[system]) * 3;
    return std::span<const double>(default_coordinates_.data() + begin,
                                   static_cast<std::size_t>(atom_count(system)) * 3);
  }

  // active[i]==0 skips the item and leaves status ownership to the caller.
  vibeqc_status execute(std::span<const double> packed_coordinates,
                        std::span<const std::uint8_t> active,
                        std::span<const std::uint8_t> want_gradient,
                        std::vector<D3Status>& statuses, std::vector<double>& energies,
                        std::vector<double>& packed_gradients, std::string& detail);

 private:
  D3Plan(vibeqc_backend backend, int device_id, std::vector<std::uint32_t> offsets,
         std::vector<std::int32_t> atomic_numbers, std::vector<double> default_coordinates,
         D3ModelParameters parameters, D3ResourceUsage resources)
      : backend_(backend),
        device_id_(device_id),
        offsets_(std::move(offsets)),
        atomic_numbers_(std::move(atomic_numbers)),
        default_coordinates_(std::move(default_coordinates)),
        parameters_(parameters),
        resources_(resources) {}

  vibeqc_backend backend_{};
  int device_id_{};
  std::vector<std::uint32_t> offsets_;
  std::vector<std::int32_t> atomic_numbers_;
  std::vector<double> default_coordinates_;
  D3ModelParameters parameters_;
  D3ResourceUsage resources_;
  D3CudaOwner* cuda_{};
};

D3CudaOwner* create_d3_cuda_owner(int device_id, std::span<const std::uint32_t> offsets,
                                  std::span<const std::int32_t> atomic_numbers,
                                  const D3ResourceUsage& resources, std::string& detail,
                                  vibeqc_status& status);
void destroy_d3_cuda_owner(D3CudaOwner* owner) noexcept;
vibeqc_status execute_d3_cuda(D3CudaOwner* owner, const D3ModelParameters& parameters,
                              std::span<const double> coordinates,
                              std::span<const std::uint8_t> active,
                              std::span<const std::uint8_t> want_gradient,
                              std::vector<D3Status>& statuses, std::vector<double>& energies,
                              std::vector<double>& gradients, std::string& detail);

}  // namespace vibeqc::dft::dispersion
