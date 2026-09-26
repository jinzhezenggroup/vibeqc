#include "model/gfn2/d4.hpp"
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include "data/parameters/gfn2.hpp"
#include "dft/dispersion/d4_data.hpp"
#include "dft/dispersion/d4_reference.hpp"

namespace vibeqc::xtb::detail::gfn2 {

struct D4PlanData {
  std::int64_t batch_size = 0;
  std::int64_t total_atoms = 0;
  std::int64_t total_pairs = 0;
  std::vector<std::int64_t> atom_offsets;
  std::vector<std::int64_t> pair_offsets;
  std::vector<std::uint8_t> element_indices;
  std::vector<std::int32_t> atomic_numbers;
  std::vector<double> pair_coordination_radii;
  std::vector<double> pair_en_factors;
  std::vector<double> pair_rrij;
  std::vector<double> pair_damping_radii;
  std::size_t workspace_size_bytes = 0u;
  std::size_t pair_scratch_offset = 0u;
  std::size_t coordination_scratch_offset = 0u;
  std::size_t weight_offset = 0u;
  std::size_t weight_cn_offset = 0u;
  std::size_t weight_charge_offset = 0u;
  std::size_t atom_scratch_offset = 0u;
  std::size_t coordination_adjoint_offset = 0u;
  std::size_t batch_scratch_offset = 0u;
  std::size_t gradient_scratch_offset = 0u;
  std::size_t shared_scratch_offset = 0u;
  std::size_t shared_scratch_elements = 0u;
};

namespace {

namespace d4_data = ::vibeqc::dft::dispersion::data;
namespace d4_math = ::vibeqc::dft::dispersion::math;
using D4ElementData = d4_data::D4ElementData;

constexpr double kCoordinationCutoff = 30.0;
constexpr double kTwoBodyCutoff = 50.0;
constexpr double kAtmCutoff = 25.0;
constexpr double kD4CutoffSwitchWidth = 0.05;
constexpr double kMinimumDistanceSquared = 1.0e-12;
constexpr double kAtmExponent = 16.0;

static_assert(d4_data::kElementCount == parameters::gfn2::kElementCount,
              "D4 and GFN2 element domains must match");
static_assert(parameters::gfn2::kGlobal.dispersion_self_consistent,
              "GFN2 parameter data must select self-consistent D4");
static_assert(!parameters::gfn2::kGlobal.dispersion_smooth,
              "the molecular GFN2 D4 cache assumes sharp pair-list cutoffs; periodic D4 uses an "
              "explicit smooth outer switch");

bool checked_add_size(std::size_t first, std::size_t second, std::size_t& result) {
  if (first > std::numeric_limits<std::size_t>::max() - second) {
    return false;
  }
  result = first + second;
  return true;
}

bool checked_multiply_size(std::size_t first, std::size_t second, std::size_t& result) {
  if (first != 0u && second > std::numeric_limits<std::size_t>::max() / first) {
    return false;
  }
  result = first * second;
  return true;
}

struct AddressRange {
  std::uintptr_t begin = 0u;
  std::uintptr_t end = 0u;
};

bool make_range(const void* pointer, std::size_t bytes, AddressRange& range) {
  if (bytes != 0u && pointer == nullptr) {
    return false;
  }
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return false;
  }
  range = {begin, begin + bytes};
  return true;
}

bool ranges_overlap(const AddressRange& first, const AddressRange& second) {
  return first.begin < second.end && second.begin < first.end;
}

bool aligned(const void* pointer, std::size_t alignment) {
  return pointer != nullptr && reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

template <typename T>
T* offset_pointer(void* base, std::size_t offset) {
  return reinterpret_cast<T*>(static_cast<std::byte*>(base) + offset);
}

template <std::size_t N>
bool pairwise_disjoint(const std::array<AddressRange, N>& ranges) {
  for (std::size_t first = 0u; first < N; ++first) {
    for (std::size_t second = first + 1u; second < N; ++second) {
      if (ranges_overlap(ranges[first], ranges[second])) {
        return false;
      }
    }
  }
  return true;
}

template <typename T>
bool overlaps_vector(const AddressRange& active, const std::vector<T>& values) {
  std::size_t bytes = 0u;
  AddressRange storage;
  return checked_multiply_size(values.capacity(), sizeof(T), bytes) &&
         make_range(values.data(), bytes, storage) && ranges_overlap(active, storage);
}

bool align_up(std::size_t value, std::size_t alignment, std::size_t& result) {
  if (alignment == 0u || (alignment & (alignment - 1u)) != 0u) {
    return false;
  }
  const std::size_t mask = alignment - 1u;
  if (value > std::numeric_limits<std::size_t>::max() - mask) {
    return false;
  }
  result = (value + mask) & ~mask;
  return true;
}

bool append_segment(std::size_t bytes, std::size_t& cursor, std::size_t& offset) {
  if (!align_up(cursor, alignof(double), offset)) {
    return false;
  }
  return checked_add_size(offset, bytes, cursor);
}

bool count_bytes(std::size_t elements, std::size_t& bytes) {
  return checked_multiply_size(elements, sizeof(double), bytes);
}

bool valid_count(std::int64_t value) {
  return value >= 0 && static_cast<std::uint64_t>(value) <=
                           static_cast<std::uint64_t>(std::numeric_limits<std::ptrdiff_t>::max());
}

vibeqc_xtb_status_t validate_plan(const D4Plan& plan, std::string& error) {
  if (!plan.sealed() || plan.batch_size() <= 0 || plan.total_atoms() <= 0 ||
      plan.total_pairs() < 0) {
    error = "D4 plan is not sealed or has invalid extents";
    return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
  }
  return VIBEQC_XTB_STATUS_SUCCESS;
}

vibeqc_xtb_status_t validate_workspace(const D4Plan& plan, const D4Workspace& workspace,
                                       std::string& error) {
  const D4PlanData& data = *plan.identity();
  if (workspace.plan_identity != plan.identity() ||
      !aligned(workspace.workspace_base, kD4WorkspaceAlignment) ||
      workspace.workspace_size_bytes < plan.workspace_size_bytes() ||
      workspace.pair_scratch !=
          offset_pointer<double>(workspace.workspace_base, data.pair_scratch_offset) ||
      workspace.coordination_scratch !=
          offset_pointer<double>(workspace.workspace_base, data.coordination_scratch_offset) ||
      workspace.weights != offset_pointer<double>(workspace.workspace_base, data.weight_offset) ||
      workspace.weight_cn_derivatives !=
          offset_pointer<double>(workspace.workspace_base, data.weight_cn_offset) ||
      workspace.weight_charge_derivatives !=
          offset_pointer<double>(workspace.workspace_base, data.weight_charge_offset) ||
      workspace.atom_scratch !=
          offset_pointer<double>(workspace.workspace_base, data.atom_scratch_offset) ||
      workspace.coordination_adjoints !=
          offset_pointer<double>(workspace.workspace_base, data.coordination_adjoint_offset) ||
      workspace.batch_scratch !=
          offset_pointer<double>(workspace.workspace_base, data.batch_scratch_offset) ||
      workspace.gradient_scratch !=
          offset_pointer<double>(workspace.workspace_base, data.gradient_scratch_offset) ||
      workspace.pair_elements !=
          plan.total_pairs() * static_cast<std::int64_t>(kD4PairDataElements) ||
      workspace.coordination_elements != plan.total_atoms() ||
      workspace.weight_elements !=
          plan.total_atoms() * static_cast<std::int64_t>(kD4MaximumReferences) ||
      workspace.atom_elements != plan.total_atoms() ||
      workspace.batch_elements != plan.batch_size() ||
      workspace.gradient_elements != plan.total_atoms() * 3) {
    error = "D4 workspace is incomplete or belongs to another plan";
    return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
  }
  return VIBEQC_XTB_STATUS_SUCCESS;
}

vibeqc_xtb_status_t validate_cache(const D4Plan& plan, const D4GeometryCache& cache,
                                   std::string& error) {
  if (cache.plan_identity != plan.identity() || cache.geometry_generation == 0u ||
      !aligned(cache.pair_data, alignof(double)) ||
      !aligned(cache.coordination_numbers, alignof(double)) ||
      cache.pair_data_elements !=
          plan.total_pairs() * static_cast<std::int64_t>(kD4PairDataElements) ||
      cache.coordination_elements != plan.total_atoms()) {
    error = "D4 geometry cache is incomplete or belongs to another plan";
    return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
  }
  return VIBEQC_XTB_STATUS_SUCCESS;
}

bool finite_values(const double* values, std::size_t count) {
  if (values == nullptr) {
    return false;
  }
  for (std::size_t index = 0; index < count; ++index) {
    if (!std::isfinite(values[index])) {
      return false;
    }
  }
  return true;
}

template <std::size_t N, std::size_t M>
bool valid_call_storage(const D4Plan& plan, const D4Workspace& workspace,
                        const std::array<AddressRange, N>& numerical,
                        const std::array<AddressRange, M>& controls) {
  AddressRange workspace_range;
  if (!make_range(workspace.workspace_base, plan.workspace_size_bytes(), workspace_range) ||
      plan.overlaps_storage(workspace.workspace_base, plan.workspace_size_bytes()) ||
      !pairwise_disjoint(numerical) || !pairwise_disjoint(controls)) {
    return false;
  }
  for (const AddressRange& range : numerical) {
    const std::size_t bytes = static_cast<std::size_t>(range.end - range.begin);
    if (ranges_overlap(range, workspace_range) ||
        plan.overlaps_storage(reinterpret_cast<const void*>(range.begin), bytes)) {
      return false;
    }
    for (const AddressRange& control : controls) {
      if (ranges_overlap(range, control)) {
        return false;
      }
    }
  }
  for (const AddressRange& control : controls) {
    if (ranges_overlap(workspace_range, control)) {
      return false;
    }
  }
  return true;
}

const D4ElementData& element(const D4PlanData& data, std::int64_t atom) {
  return d4_data::kElements[data.element_indices[static_cast<std::size_t>(atom)]];
}

void prepare_weight_slice(const D4PlanData& data, const double* coordination, const double* charges,
                          std::int64_t atom_begin, std::int64_t atom_end, bool /* derivatives */,
                          const D4Workspace& workspace) {
  namespace shared = ::vibeqc::dft::dispersion;
  const int count = static_cast<int>(atom_end - atom_begin);
  const std::size_t weight_begin = static_cast<std::size_t>(atom_begin) * kD4MaximumReferences;
  shared::prepare_d4_cached_weights(
      count, data.atomic_numbers.data() + atom_begin, coordination + atom_begin,
      charges + atom_begin, shared::gfn2_d4_parameters(), shared::gfn2_d4_host_tables(),
      workspace.weights + weight_begin, workspace.weight_cn_derivatives + weight_begin,
      workspace.weight_charge_derivatives + weight_begin);
}

vibeqc_xtb_status_t prepare_weights(const D4PlanData& data, const double* coordination,
                                    const double* charges, bool derivatives,
                                    const D4Workspace& workspace, std::string& error) {
  const std::size_t atom_count = static_cast<std::size_t>(data.total_atoms);
  if (!finite_values(coordination, atom_count) || !finite_values(charges, atom_count)) {
    error = "D4 coordination numbers and charges must be finite";
    return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
  }
  prepare_weight_slice(data, coordination, charges, 0, data.total_atoms, derivatives, workspace);
  return VIBEQC_XTB_STATUS_SUCCESS;
}

using PairCoefficient = ::vibeqc::dft::dispersion::D4CachedPairCoefficient;

PairCoefficient pair_coefficient(const D4PlanData& data, std::int64_t first, std::int64_t second,
                                 const D4Workspace& workspace, bool /* derivatives */) {
  namespace shared = ::vibeqc::dft::dispersion;
  return shared::d4_cached_pair_coefficient(
      static_cast<int>(first), static_cast<int>(second), data.atomic_numbers.data(),
      shared::gfn2_d4_host_tables(), workspace.weights, workspace.weight_cn_derivatives,
      workspace.weight_charge_derivatives);
}

vibeqc_xtb_status_t evaluate_shared_molecular_d4_component(
    const D4Plan& plan, const D4GeometryCache& cache, const double* positions,
    const double* atomic_charges, bool include_two_body, bool include_atm, double* energies,
    double* gradients, const D4Workspace& workspace, std::string& error) {
  vibeqc_xtb_status_t status = validate_plan(plan, error);
  if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
  status = validate_workspace(plan, workspace, error);
  if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
  status = validate_cache(plan, cache, error);
  if (status != VIBEQC_XTB_STATUS_SUCCESS) return status;
  if (positions == nullptr || atomic_charges == nullptr || (!include_two_body && !include_atm) ||
      (energies == nullptr && gradients == nullptr)) {
    error = "shared molecular D4 adapter received incomplete inputs";
    return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
  }

  namespace shared = ::vibeqc::dft::dispersion;
  const D4PlanData& data = *plan.identity();
  const std::size_t atoms = static_cast<std::size_t>(data.total_atoms);
  const std::size_t systems = static_cast<std::size_t>(data.batch_size);
  const std::size_t coordinates = 3u * atoms;
  if (!aligned(positions, alignof(double)) || !aligned(atomic_charges, alignof(double)) ||
      (energies && !aligned(energies, alignof(double))) ||
      (gradients && !aligned(gradients, alignof(double)))) {
    error = "shared molecular D4 requires aligned inputs and finite gradient outputs";
    return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
  }
  std::array<AddressRange, 6> numerical{};
  std::array<AddressRange, 4> controls{};
  if (!make_range(positions, coordinates * sizeof(double), numerical[0]) ||
      !make_range(atomic_charges, atoms * sizeof(double), numerical[1]) ||
      !make_range(cache.pair_data,
                  static_cast<std::size_t>(cache.pair_data_elements) * sizeof(double),
                  numerical[2]) ||
      !make_range(cache.coordination_numbers, atoms * sizeof(double), numerical[3]) ||
      !make_range(energies, energies ? systems * sizeof(double) : 0u, numerical[4]) ||
      !make_range(gradients, gradients ? coordinates * sizeof(double) : 0u, numerical[5]) ||
      !make_range(&plan, sizeof(plan), controls[0]) ||
      !make_range(&cache, sizeof(cache), controls[1]) ||
      !make_range(&workspace, sizeof(workspace), controls[2]) ||
      !make_range(&error, sizeof(error), controls[3]) ||
      !valid_call_storage(plan, workspace, numerical, controls)) {
    error = "shared molecular D4 buffers overlap numerical, plan, workspace, or descriptors";
    return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
  }
  if (gradients && !finite_values(gradients, coordinates)) {
    error = "shared molecular D4 requires finite gradient outputs";
    return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
  }
  auto parameters = shared::gfn2_d4_parameters();
  if (!include_two_body) {
    parameters.s6 = 0.0;
    parameters.s8 = 0.0;
  }
  if (!include_atm) parameters.s9 = 0.0;
  const auto tables = shared::gfn2_d4_host_tables();

  std::array<double, 2> candidate_energy{};

  for (std::int64_t system = 0; system < data.batch_size; ++system) {
    const std::int64_t begin = data.atom_offsets[static_cast<std::size_t>(system)];
    const std::int64_t end = data.atom_offsets[static_cast<std::size_t>(system + 1)];
    const std::int64_t count64 = end - begin;
    if (count64 <= 0 || count64 > std::numeric_limits<int>::max()) {
      error = "shared molecular D4 adapter does not support this system size";
      return VIBEQC_XTB_STATUS_NOT_SUPPORTED;
    }
    const int count = static_cast<int>(count64);
    const std::size_t required_workspace = shared::d4_unbounded_workspace_elements(count);
    if (required_workspace == 0u) {
      error = "shared molecular D4 adapter workspace size overflowed";
      return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
    }

    double* shared_workspace =
        offset_pointer<double>(workspace.workspace_base, data.shared_scratch_offset);
    const std::size_t shared_workspace_elements = data.shared_scratch_elements;
    double* candidate_gradient = workspace.gradient_scratch + 3 * begin;
    double* candidate_dq = workspace.coordination_adjoints + begin;

    const auto shared_status = shared::evaluate_d4_fixed_charge_unbounded_cpu(
        count, data.atomic_numbers.data() + begin, positions + 3 * begin, atomic_charges + begin,
        parameters, tables, shared_workspace, shared_workspace_elements, candidate_energy.data(),
        candidate_gradient, candidate_dq);
    if (shared_status != shared::D4Status::success) {
      switch (shared_status) {
        case shared::D4Status::invalid_argument:
          error = "shared molecular D4 adapter rejected its numerical inputs";
          return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
        case shared::D4Status::unsupported:
          error = "shared molecular D4 adapter does not support the requested GFN2 case";
          return VIBEQC_XTB_STATUS_NOT_SUPPORTED;
        case shared::D4Status::numerical_failure:
          error = "shared molecular D4 adapter encountered a numerical failure";
          return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
        case shared::D4Status::success:
          break;
      }
    }
    workspace.batch_scratch[system] =
        (include_two_body ? candidate_energy[0] : 0.0) + (include_atm ? candidate_energy[1] : 0.0);
  }
  // Validate every final accumulation before publishing any batch member.
  if (gradients != nullptr) {
    for (std::int64_t coordinate = 0; coordinate < 3 * data.total_atoms; ++coordinate) {
      if (!std::isfinite(gradients[coordinate] + workspace.gradient_scratch[coordinate])) {
        error = "shared molecular D4 gradient accumulation overflowed";
        return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
      }
    }
  }
  if (energies != nullptr) std::copy_n(workspace.batch_scratch, data.batch_size, energies);
  if (gradients != nullptr)
    for (std::int64_t coordinate = 0; coordinate < 3 * data.total_atoms; ++coordinate)
      gradients[coordinate] += workspace.gradient_scratch[coordinate];

  error.clear();
  return VIBEQC_XTB_STATUS_SUCCESS;
}

}  // namespace

D4Plan::D4Plan(std::shared_ptr<const D4PlanData> data) noexcept : data_(std::move(data)) {}

bool D4Plan::sealed() const noexcept { return static_cast<bool>(data_); }

std::int64_t D4Plan::batch_size() const noexcept { return data_ ? data_->batch_size : 0; }

std::int64_t D4Plan::total_atoms() const noexcept { return data_ ? data_->total_atoms : 0; }

std::int64_t D4Plan::total_pairs() const noexcept { return data_ ? data_->total_pairs : 0; }

const std::vector<std::int64_t>& D4Plan::atom_offsets() const noexcept {
  static const std::vector<std::int64_t> empty;
  return data_ ? data_->atom_offsets : empty;
}

const std::vector<std::int64_t>& D4Plan::pair_offsets() const noexcept {
  static const std::vector<std::int64_t> empty;
  return data_ ? data_->pair_offsets : empty;
}

bool D4Plan::matches_atomic_numbers(const std::int32_t* atomic_numbers) const noexcept {
  if (!data_ || atomic_numbers == nullptr) {
    return false;
  }
  for (std::int64_t atom = 0; atom < data_->total_atoms; ++atom) {
    if (atomic_numbers[atom] != data_->atomic_numbers[static_cast<std::size_t>(atom)]) {
      return false;
    }
  }
  return true;
}

bool D4Plan::overlaps_storage(const void* data, std::size_t size_bytes) const noexcept {
  if (size_bytes == 0u) {
    return false;
  }
  AddressRange active;
  AddressRange descriptor;
  AddressRange plan_data;
  if (data_ == nullptr || !make_range(data, size_bytes, active) ||
      !make_range(this, sizeof(*this), descriptor) ||
      !make_range(data_.get(), sizeof(*data_), plan_data)) {
    return true;
  }
  if (ranges_overlap(active, descriptor) || ranges_overlap(active, plan_data)) {
    return true;
  }
  return overlaps_vector(active, data_->atom_offsets) ||
         overlaps_vector(active, data_->pair_offsets) ||
         overlaps_vector(active, data_->element_indices) ||
         overlaps_vector(active, data_->atomic_numbers) ||
         overlaps_vector(active, data_->pair_coordination_radii) ||
         overlaps_vector(active, data_->pair_en_factors) ||
         overlaps_vector(active, data_->pair_rrij) ||
         overlaps_vector(active, data_->pair_damping_radii);
}

std::size_t D4Plan::workspace_size_bytes() const noexcept {
  return data_ ? data_->workspace_size_bytes : 0u;
}

std::size_t D4Plan::resident_bytes() const noexcept {
  if (data_ == nullptr) {
    return 0u;
  }
  return sizeof(*data_) + data_->atom_offsets.capacity() * sizeof(std::int64_t) +
         data_->pair_offsets.capacity() * sizeof(std::int64_t) +
         data_->element_indices.capacity() * sizeof(std::uint8_t) +
         data_->atomic_numbers.capacity() * sizeof(std::int32_t) +
         data_->pair_coordination_radii.capacity() * sizeof(double) +
         data_->pair_en_factors.capacity() * sizeof(double) +
         data_->pair_rrij.capacity() * sizeof(double) +
         data_->pair_damping_radii.capacity() * sizeof(double);
}

const D4PlanData* D4Plan::identity() const noexcept { return data_.get(); }

vibeqc_xtb_status_t make_d4_plan(std::int64_t batch_size, std::int64_t total_atoms,
                                 const std::int64_t* atom_offsets,
                                 const std::int32_t* atomic_numbers, D4Plan& plan,
                                 std::string& error) {
  if (batch_size <= 0 || total_atoms <= 0 || !valid_count(batch_size) ||
      !valid_count(total_atoms) || atom_offsets == nullptr || atomic_numbers == nullptr ||
      atom_offsets[0] != 0 || atom_offsets[batch_size] != total_atoms) {
    error = "D4 plan requires a valid positive ragged batch";
    return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
  }
  try {
    auto created = std::make_shared<D4PlanData>();
    created->batch_size = batch_size;
    created->total_atoms = total_atoms;
    created->atom_offsets.assign(atom_offsets, atom_offsets + batch_size + 1);
    created->pair_offsets.resize(static_cast<std::size_t>(batch_size + 1), 0);
    created->element_indices.resize(static_cast<std::size_t>(total_atoms));
    created->atomic_numbers.assign(atomic_numbers, atomic_numbers + total_atoms);
    for (std::int64_t batch = 0; batch < batch_size; ++batch) {
      const std::int64_t begin = atom_offsets[batch];
      const std::int64_t end = atom_offsets[batch + 1];
      if (begin < 0 || begin > end || end > total_atoms) {
        error = "D4 atom offsets are not a valid ragged partition";
        return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
      }
      const std::uint64_t count = static_cast<std::uint64_t>(end - begin);
      if (count > 0u &&
          count - 1u >
              static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) / count) {
        error = "D4 pair count overflows";
        return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
      }
      const std::uint64_t pairs = count * (count - 1u) / 2u;
      if (pairs >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max() -
                                     created->pair_offsets[static_cast<std::size_t>(batch)])) {
        error = "D4 total pair count overflows";
        return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
      }
      created->pair_offsets[static_cast<std::size_t>(batch + 1)] =
          created->pair_offsets[static_cast<std::size_t>(batch)] + static_cast<std::int64_t>(pairs);
    }
    created->total_pairs = created->pair_offsets.back();
    for (std::int64_t atom = 0; atom < total_atoms; ++atom) {
      const std::int32_t atomic_number = atomic_numbers[atom];
      if (atomic_number <= 0 || atomic_number > static_cast<std::int32_t>(d4_data::kElementCount)) {
        error = "D4 plan contains an unsupported atomic number";
        return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
      }
      created->element_indices[static_cast<std::size_t>(atom)] =
          static_cast<std::uint8_t>(atomic_number - 1);
    }
    const std::size_t packed_pair_count = static_cast<std::size_t>(created->total_pairs);
    created->pair_coordination_radii.resize(packed_pair_count);
    created->pair_en_factors.resize(packed_pair_count);
    created->pair_rrij.resize(packed_pair_count);
    created->pair_damping_radii.resize(packed_pair_count);
    std::size_t packed_pair = 0u;
    for (std::int64_t batch = 0; batch < batch_size; ++batch) {
      const std::int64_t begin = atom_offsets[batch];
      const std::int64_t end = atom_offsets[batch + 1];
      for (std::int64_t second = begin + 1; second < end; ++second) {
        for (std::int64_t first = begin; first < second; ++first, ++packed_pair) {
          const D4ElementData& first_element = element(*created, first);
          const D4ElementData& second_element = element(*created, second);
          const auto coordination = d4_math::coordination_parameters(first_element, second_element);
          created->pair_coordination_radii[packed_pair] = coordination.radius;
          created->pair_en_factors[packed_pair] = coordination.electronegativity_factor;
          created->pair_rrij[packed_pair] = d4_math::pair_rr(first_element, second_element);
          created->pair_damping_radii[packed_pair] = d4_math::damping_radius(
              first_element, second_element, parameters::gfn2::kGlobal.dispersion_a1,
              parameters::gfn2::kGlobal.dispersion_a2);
        }
      }
    }

    const std::size_t atom_count = static_cast<std::size_t>(total_atoms);
    const std::size_t pair_count = static_cast<std::size_t>(created->total_pairs);
    const std::size_t batch_count = static_cast<std::size_t>(batch_size);
    std::size_t pair_elements = 0u;
    std::size_t weight_elements = 0u;
    std::size_t gradient_elements = 0u;
    if (!checked_multiply_size(pair_count, kD4PairDataElements, pair_elements) ||
        !checked_multiply_size(atom_count, kD4MaximumReferences, weight_elements) ||
        !checked_multiply_size(atom_count, 3u, gradient_elements)) {
      error = "D4 workspace element count overflows";
      return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
    }
    std::size_t maximum_atoms = 0u;
    for (std::size_t system = 0; system < batch_count; ++system) {
      maximum_atoms =
          std::max(maximum_atoms, static_cast<std::size_t>(created->atom_offsets[system + 1] -
                                                           created->atom_offsets[system]));
    }
    if (!checked_multiply_size(maximum_atoms, 27u, created->shared_scratch_elements)) {
      error = "shared D4 scratch extent overflows";
      return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
    }
    std::size_t cursor = 0u;
    std::size_t bytes = 0u;
    auto append_doubles = [&](std::size_t elements, std::size_t& offset) {
      return count_bytes(elements, bytes) && append_segment(bytes, cursor, offset);
    };
    if (!append_doubles(pair_elements, created->pair_scratch_offset) ||
        !append_doubles(atom_count, created->coordination_scratch_offset) ||
        !append_doubles(weight_elements, created->weight_offset) ||
        !append_doubles(weight_elements, created->weight_cn_offset) ||
        !append_doubles(weight_elements, created->weight_charge_offset) ||
        !append_doubles(atom_count, created->atom_scratch_offset) ||
        !append_doubles(atom_count, created->coordination_adjoint_offset) ||
        !append_doubles(batch_count, created->batch_scratch_offset) ||
        !append_doubles(gradient_elements, created->gradient_scratch_offset) ||
        !append_doubles(created->shared_scratch_elements, created->shared_scratch_offset) ||
        !align_up(cursor, kD4WorkspaceAlignment, created->workspace_size_bytes)) {
      error = "D4 workspace byte count overflows";
      return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
    }

    plan = D4Plan(std::move(created));
    error.clear();
    return VIBEQC_XTB_STATUS_SUCCESS;
  } catch (const std::bad_alloc&) {
    error = "failed to allocate D4 plan";
    return VIBEQC_XTB_STATUS_ALLOCATION_FAILED;
  }
}

vibeqc_xtb_status_t bind_d4_workspace(const D4Plan& plan, void* workspace,
                                      std::size_t workspace_size, D4Workspace& view,
                                      std::string& error) {
  vibeqc_xtb_status_t status = validate_plan(plan, error);
  if (status != VIBEQC_XTB_STATUS_SUCCESS) {
    return status;
  }
  AddressRange workspace_range;
  AddressRange view_range;
  AddressRange error_range;
  if (!aligned(workspace, kD4WorkspaceAlignment) || workspace_size < plan.workspace_size_bytes() ||
      !make_range(workspace, plan.workspace_size_bytes(), workspace_range) ||
      !make_range(&view, sizeof(view), view_range) ||
      !make_range(&error, sizeof(error), error_range) ||
      ranges_overlap(workspace_range, view_range) || ranges_overlap(workspace_range, error_range) ||
      ranges_overlap(view_range, error_range) ||
      plan.overlaps_storage(workspace, plan.workspace_size_bytes())) {
    error = "D4 workspace must be sufficiently large and 64-byte aligned";
    return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
  }
  const D4PlanData& data = *plan.identity();
  auto* base = static_cast<std::byte*>(workspace);
  D4Workspace bound;
  bound.workspace_base = workspace;
  bound.workspace_size_bytes = workspace_size;
  bound.pair_scratch = reinterpret_cast<double*>(base + data.pair_scratch_offset);
  bound.pair_elements = data.total_pairs * static_cast<std::int64_t>(kD4PairDataElements);
  bound.coordination_scratch = reinterpret_cast<double*>(base + data.coordination_scratch_offset);
  bound.coordination_elements = data.total_atoms;
  bound.weights = reinterpret_cast<double*>(base + data.weight_offset);
  bound.weight_cn_derivatives = reinterpret_cast<double*>(base + data.weight_cn_offset);
  bound.weight_charge_derivatives = reinterpret_cast<double*>(base + data.weight_charge_offset);
  bound.weight_elements = data.total_atoms * static_cast<std::int64_t>(kD4MaximumReferences);
  bound.atom_scratch = reinterpret_cast<double*>(base + data.atom_scratch_offset);
  bound.coordination_adjoints = reinterpret_cast<double*>(base + data.coordination_adjoint_offset);
  bound.atom_elements = data.total_atoms;
  bound.batch_scratch = reinterpret_cast<double*>(base + data.batch_scratch_offset);
  bound.batch_elements = data.batch_size;
  bound.gradient_scratch = reinterpret_cast<double*>(base + data.gradient_scratch_offset);
  bound.gradient_elements = data.total_atoms * 3;
  bound.plan_identity = plan.identity();
  view = bound;
  error.clear();
  return VIBEQC_XTB_STATUS_SUCCESS;
}

vibeqc_xtb_status_t update_d4_geometry_cache_cpu(
    const D4Plan& plan, const double* positions, std::uint64_t geometry_generation,
    double* pair_storage, std::size_t pair_storage_elements, double* coordination_storage,
    std::size_t coordination_storage_elements, const D4Workspace& workspace, D4GeometryCache& cache,
    std::string& error) {
  vibeqc_xtb_status_t status = validate_plan(plan, error);
  if (status != VIBEQC_XTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_workspace(plan, workspace, error);
  if (status != VIBEQC_XTB_STATUS_SUCCESS) {
    return status;
  }
  const D4PlanData& data = *plan.identity();
  const std::size_t atom_count = static_cast<std::size_t>(data.total_atoms);
  const std::size_t expected_pairs =
      static_cast<std::size_t>(data.total_pairs) * kD4PairDataElements;
  if (geometry_generation == 0u || !aligned(positions, alignof(double)) ||
      !aligned(pair_storage, alignof(double)) || !aligned(coordination_storage, alignof(double)) ||
      pair_storage_elements < expected_pairs || coordination_storage_elements < atom_count ||
      !finite_values(positions, atom_count * 3u)) {
    error = "D4 geometry update requires finite positions and complete output storage";
    return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
  }
  std::size_t position_bytes = 0u;
  std::size_t pair_bytes = 0u;
  std::size_t coordination_bytes = 0u;
  std::array<AddressRange, 3> numerical{};
  std::array<AddressRange, 4> controls{};
  if (!checked_multiply_size(atom_count * 3u, sizeof(double), position_bytes) ||
      !checked_multiply_size(expected_pairs, sizeof(double), pair_bytes) ||
      !checked_multiply_size(atom_count, sizeof(double), coordination_bytes) ||
      !make_range(positions, position_bytes, numerical[0]) ||
      !make_range(pair_storage, pair_bytes, numerical[1]) ||
      !make_range(coordination_storage, coordination_bytes, numerical[2]) ||
      !make_range(&plan, sizeof(plan), controls[0]) ||
      !make_range(&workspace, sizeof(workspace), controls[1]) ||
      !make_range(&cache, sizeof(cache), controls[2]) ||
      !make_range(&error, sizeof(error), controls[3]) ||
      !valid_call_storage(plan, workspace, numerical, controls)) {
    error = "D4 geometry buffers overlap numerical, plan, workspace, or descriptor storage";
    return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
  }

  std::fill_n(workspace.coordination_scratch, atom_count, 0.0);
  std::size_t packed_pair = 0u;
  for (std::int64_t batch = 0; batch < data.batch_size; ++batch) {
    const std::int64_t begin = data.atom_offsets[static_cast<std::size_t>(batch)];
    const std::int64_t end = data.atom_offsets[static_cast<std::size_t>(batch + 1)];
    for (std::int64_t second = begin + 1; second < end; ++second) {
      for (std::int64_t first = begin; first < second; ++first, ++packed_pair) {
        double* pair = workspace.pair_scratch + packed_pair * kD4PairDataElements;
        pair[0] = positions[static_cast<std::size_t>(first) * 3u] -
                  positions[static_cast<std::size_t>(second) * 3u];
        pair[1] = positions[static_cast<std::size_t>(first) * 3u + 1u] -
                  positions[static_cast<std::size_t>(second) * 3u + 1u];
        pair[2] = positions[static_cast<std::size_t>(first) * 3u + 2u] -
                  positions[static_cast<std::size_t>(second) * 3u + 2u];
        const double distance_squared = pair[0] * pair[0] + pair[1] * pair[1] + pair[2] * pair[2];
        if (distance_squared < kMinimumDistanceSquared) {
          error = "D4 geometry contains coincident atoms";
          return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
        }
        if (distance_squared <= kCoordinationCutoff * kCoordinationCutoff) {
          const double distance = std::sqrt(distance_squared);
          const auto coordination = d4_math::coordination_pair(
              d4_math::CoordinationParameters{data.pair_coordination_radii[packed_pair],
                                              data.pair_en_factors[packed_pair]},
              distance);
          workspace.coordination_scratch[first] += coordination.value;
          workspace.coordination_scratch[second] += coordination.value;
        }
        pair[3] = 0.0;
        pair[4] = 0.0;
        if (distance_squared <= kTwoBodyCutoff * kTwoBodyCutoff) {
          const auto damping = d4_math::pair_damping(
              distance_squared, data.pair_rrij[packed_pair], data.pair_damping_radii[packed_pair],
              parameters::gfn2::kGlobal.dispersion_s6, parameters::gfn2::kGlobal.dispersion_s8);
          pair[3] = damping.value;
          pair[4] = damping.derivative;
        }
        if (!std::isfinite(pair[3]) || !std::isfinite(pair[4])) {
          error = "D4 geometry cache overflowed";
          return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
        }
      }
    }
  }

  std::memcpy(pair_storage, workspace.pair_scratch, expected_pairs * sizeof(double));
  std::memcpy(coordination_storage, workspace.coordination_scratch, atom_count * sizeof(double));
  cache.pair_data = pair_storage;
  cache.pair_data_elements = static_cast<std::int64_t>(expected_pairs);
  cache.coordination_numbers = coordination_storage;
  cache.coordination_elements = data.total_atoms;
  cache.geometry_generation = geometry_generation;
  cache.plan_identity = plan.identity();
  error.clear();
  return VIBEQC_XTB_STATUS_SUCCESS;
}

vibeqc_xtb_status_t evaluate_d4_two_body_cpu(const D4Plan& plan, const D4GeometryCache& cache,
                                             const double* atomic_charges, double* energies,
                                             double* atomic_potentials,
                                             const D4Workspace& workspace, std::string& error) {
  vibeqc_xtb_status_t status = validate_plan(plan, error);
  if (status != VIBEQC_XTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_workspace(plan, workspace, error);
  if (status != VIBEQC_XTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_cache(plan, cache, error);
  if (status != VIBEQC_XTB_STATUS_SUCCESS) {
    return status;
  }
  const D4PlanData& data = *plan.identity();
  if (!aligned(atomic_charges, alignof(double)) || !aligned(energies, alignof(double)) ||
      !aligned(atomic_potentials, alignof(double))) {
    error = "D4 two-body outputs must not be NULL";
    return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
  }
  const std::size_t pair_count = static_cast<std::size_t>(cache.pair_data_elements);
  const std::size_t atom_count = static_cast<std::size_t>(data.total_atoms);
  const std::size_t batch_count = static_cast<std::size_t>(data.batch_size);
  std::size_t pair_bytes = 0u;
  std::size_t atom_bytes = 0u;
  std::size_t batch_bytes = 0u;
  std::array<AddressRange, 5> numerical{};
  std::array<AddressRange, 4> controls{};
  if (!checked_multiply_size(pair_count, sizeof(double), pair_bytes) ||
      !checked_multiply_size(atom_count, sizeof(double), atom_bytes) ||
      !checked_multiply_size(batch_count, sizeof(double), batch_bytes) ||
      !make_range(cache.pair_data, pair_bytes, numerical[0]) ||
      !make_range(cache.coordination_numbers, atom_bytes, numerical[1]) ||
      !make_range(atomic_charges, atom_bytes, numerical[2]) ||
      !make_range(energies, batch_bytes, numerical[3]) ||
      !make_range(atomic_potentials, atom_bytes, numerical[4]) ||
      !make_range(&plan, sizeof(plan), controls[0]) ||
      !make_range(&cache, sizeof(cache), controls[1]) ||
      !make_range(&workspace, sizeof(workspace), controls[2]) ||
      !make_range(&error, sizeof(error), controls[3]) ||
      !valid_call_storage(plan, workspace, numerical, controls)) {
    error = "D4 two-body buffers overlap numerical, plan, workspace, or descriptor storage";
    return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
  }
  status =
      prepare_weights(data, cache.coordination_numbers, atomic_charges, true, workspace, error);
  if (status != VIBEQC_XTB_STATUS_SUCCESS) {
    return status;
  }
  std::fill_n(workspace.batch_scratch, static_cast<std::size_t>(data.batch_size), 0.0);
  std::fill_n(workspace.atom_scratch, static_cast<std::size_t>(data.total_atoms), 0.0);
  for (std::int64_t batch = 0; batch < data.batch_size; ++batch) {
    const std::int64_t begin = data.atom_offsets[static_cast<std::size_t>(batch)];
    const std::int64_t end = data.atom_offsets[static_cast<std::size_t>(batch + 1)];
    std::size_t packed_pair =
        static_cast<std::size_t>(data.pair_offsets[static_cast<std::size_t>(batch)]);
    for (std::int64_t second = begin + 1; second < end; ++second) {
      for (std::int64_t first = begin; first < second; ++first, ++packed_pair) {
        const double damping = cache.pair_data[packed_pair * kD4PairDataElements + 3u];
        if (damping == 0.0) {
          continue;
        }
        const PairCoefficient coefficient = pair_coefficient(data, first, second, workspace, true);
        workspace.batch_scratch[batch] -= coefficient.c6 * damping;
        workspace.atom_scratch[first] -= coefficient.first_charge * damping;
        workspace.atom_scratch[second] -= coefficient.second_charge * damping;
      }
    }
  }
  if (!finite_values(workspace.batch_scratch, static_cast<std::size_t>(data.batch_size)) ||
      !finite_values(workspace.atom_scratch, static_cast<std::size_t>(data.total_atoms))) {
    error = "D4 two-body evaluation overflowed";
    return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
  }
  std::memcpy(energies, workspace.batch_scratch,
              static_cast<std::size_t>(data.batch_size) * sizeof(double));
  std::memcpy(atomic_potentials, workspace.atom_scratch,
              static_cast<std::size_t>(data.total_atoms) * sizeof(double));
  error.clear();
  return VIBEQC_XTB_STATUS_SUCCESS;
}

vibeqc_xtb_status_t evaluate_d4_two_body_system_cpu(
    const D4Plan& plan, const D4GeometryCache& cache, std::int64_t system,
    const double* atomic_charges, double& energy, double* atomic_potentials,
    const D4Workspace& workspace, std::string& error) {
  vibeqc_xtb_status_t status = validate_plan(plan, error);
  if (status != VIBEQC_XTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_workspace(plan, workspace, error);
  if (status != VIBEQC_XTB_STATUS_SUCCESS) {
    return status;
  }
  status = validate_cache(plan, cache, error);
  if (status != VIBEQC_XTB_STATUS_SUCCESS) {
    return status;
  }
  if (system < 0 || system >= plan.batch_size()) {
    error = "D4 two-body system index is out of range";
    return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
  }
  if (!aligned(atomic_charges, alignof(double)) || !aligned(&energy, alignof(double)) ||
      (atomic_potentials != nullptr && !aligned(atomic_potentials, alignof(double)))) {
    error = "D4 system two-body inputs and outputs must not be NULL or misaligned";
    return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
  }

  const D4PlanData& data = *plan.identity();
  const std::size_t atom_count = static_cast<std::size_t>(data.total_atoms);
  const std::size_t pair_element_count = static_cast<std::size_t>(cache.pair_data_elements);
  std::size_t atom_bytes = 0u;
  std::size_t pair_bytes = 0u;
  std::array<AddressRange, 5> numerical{};
  std::array<AddressRange, 4> controls{};
  if (!checked_multiply_size(atom_count, sizeof(double), atom_bytes) ||
      !checked_multiply_size(pair_element_count, sizeof(double), pair_bytes) ||
      !make_range(cache.pair_data, pair_bytes, numerical[0]) ||
      !make_range(cache.coordination_numbers, atom_bytes, numerical[1]) ||
      !make_range(atomic_charges, atom_bytes, numerical[2]) ||
      !make_range(&energy, sizeof(double), numerical[3]) ||
      !make_range(atomic_potentials, atomic_potentials == nullptr ? 0u : atom_bytes,
                  numerical[4]) ||
      !make_range(&plan, sizeof(plan), controls[0]) ||
      !make_range(&cache, sizeof(cache), controls[1]) ||
      !make_range(&workspace, sizeof(workspace), controls[2]) ||
      !make_range(&error, sizeof(error), controls[3]) ||
      !valid_call_storage(plan, workspace, numerical, controls)) {
    error = "D4 system two-body buffers overlap numerical, plan, workspace, or descriptor storage";
    return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
  }

  const std::size_t system_index = static_cast<std::size_t>(system);
  const std::int64_t atom_begin = data.atom_offsets[system_index];
  const std::int64_t atom_end = data.atom_offsets[system_index + 1u];
  const std::int64_t pair_begin = data.pair_offsets[system_index];
  const std::int64_t pair_end = data.pair_offsets[system_index + 1u];
  if (atom_begin < 0 || atom_begin > atom_end || atom_end > data.total_atoms || pair_begin < 0 ||
      pair_begin > pair_end || pair_end > data.total_pairs) {
    error = "D4 target system partition is structurally invalid";
    return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
  }
  const std::uint64_t target_atom_count = static_cast<std::uint64_t>(atom_end - atom_begin);
  if (target_atom_count > 0u &&
      target_atom_count - 1u >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) /
              target_atom_count) {
    error = "D4 target system pair count overflows";
    return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
  }
  const std::uint64_t expected_pairs = target_atom_count * (target_atom_count - 1u) / 2u;
  if (expected_pairs != static_cast<std::uint64_t>(pair_end - pair_begin)) {
    error = "D4 target atom and pair partitions disagree";
    return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
  }

  const std::size_t target_atoms = static_cast<std::size_t>(atom_end - atom_begin);
  if (!finite_values(cache.coordination_numbers + atom_begin, target_atoms) ||
      !finite_values(atomic_charges + atom_begin, target_atoms)) {
    error = "D4 target coordination numbers and charges must be finite";
    return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
  }
  for (std::int64_t atom = atom_begin; atom < atom_end; ++atom) {
    if (cache.coordination_numbers[atom] < 0.0) {
      error = "D4 target coordination numbers must be nonnegative";
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
  }
  for (std::int64_t pair_index = pair_begin; pair_index < pair_end; ++pair_index) {
    const double* pair =
        cache.pair_data + static_cast<std::size_t>(pair_index) * kD4PairDataElements;
    if (!finite_values(pair, kD4PairDataElements) || pair[3] < 0.0) {
      error = "D4 target geometry cache contains invalid numerical data";
      return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
    }
  }

  const bool derivatives = atomic_potentials != nullptr;
  prepare_weight_slice(data, cache.coordination_numbers, atomic_charges, atom_begin, atom_end,
                       derivatives, workspace);
  if (derivatives) {
    std::fill_n(workspace.atom_scratch + atom_begin, target_atoms, 0.0);
  }
  double contribution = 0.0;
  std::size_t packed_pair = static_cast<std::size_t>(pair_begin);
  for (std::int64_t second = atom_begin + 1; second < atom_end; ++second) {
    for (std::int64_t first = atom_begin; first < second; ++first, ++packed_pair) {
      const double damping = cache.pair_data[packed_pair * kD4PairDataElements + 3u];
      if (damping == 0.0) {
        continue;
      }
      const PairCoefficient coefficient =
          pair_coefficient(data, first, second, workspace, derivatives);
      contribution -= coefficient.c6 * damping;
      if (derivatives) {
        workspace.atom_scratch[first] -= coefficient.first_charge * damping;
        workspace.atom_scratch[second] -= coefficient.second_charge * damping;
      }
    }
  }
  if (packed_pair != static_cast<std::size_t>(pair_end)) {
    error = "D4 target pair enumeration disagrees with the plan";
    return VIBEQC_XTB_STATUS_INVALID_ARGUMENT;
  }
  if (!std::isfinite(contribution) ||
      (derivatives && !finite_values(workspace.atom_scratch + atom_begin, target_atoms))) {
    error = "D4 target two-body evaluation overflowed";
    return VIBEQC_XTB_STATUS_INTERNAL_ERROR;
  }

  workspace.batch_scratch[system_index] = contribution;
  if (derivatives) {
    std::memcpy(atomic_potentials + atom_begin, workspace.atom_scratch + atom_begin,
                target_atoms * sizeof(double));
  }
  energy = contribution;
  error.clear();
  return VIBEQC_XTB_STATUS_SUCCESS;
}

vibeqc_xtb_status_t add_d4_two_body_gradient_cpu(const D4Plan& plan, const D4GeometryCache& cache,
                                                 const double* positions,
                                                 const double* atomic_charges, double* gradients,
                                                 const D4Workspace& workspace, std::string& error) {
  return evaluate_shared_molecular_d4_component(plan, cache, positions, atomic_charges, true, false,
                                                nullptr, gradients, workspace, error);
}

vibeqc_xtb_status_t evaluate_d4_atm_cpu(const D4Plan& plan, const D4GeometryCache& cache,
                                        const double* positions, const double* atomic_charges,
                                        double* energies, const D4Workspace& workspace,
                                        std::string& error) {
  return evaluate_shared_molecular_d4_component(plan, cache, positions, atomic_charges, false, true,
                                                energies, nullptr, workspace, error);
}

vibeqc_xtb_status_t add_d4_atm_gradient_cpu(const D4Plan& plan, const D4GeometryCache& cache,
                                            const double* positions, const double* atomic_charges,
                                            double* gradients, const D4Workspace& workspace,
                                            std::string& error) {
  return evaluate_shared_molecular_d4_component(plan, cache, positions, atomic_charges, false, true,
                                                nullptr, gradients, workspace, error);
}

}  // namespace vibeqc::xtb::detail::gfn2
