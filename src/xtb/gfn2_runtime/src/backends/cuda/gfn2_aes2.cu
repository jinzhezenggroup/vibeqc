#include <cmath>
// xtbloom's CUDA/MKL additional permission is in CUDA_MKL_LINKING_EXCEPTION.

#include <cstddef>
#include <cstdint>
#include <limits>

#include "backends/cuda/gfn2_aes2.cuh"
#include "generated_gfn2_aes2_native.cuh"

namespace vibeqc::xtb::detail::cuda {
namespace {

constexpr int kThreadsPerBlock = 256;
constexpr std::int64_t kInt64Maximum = 9223372036854775807LL;
constexpr double kMinimumDistanceSquared = 1.0e-12;

struct SystemRanges {
  std::int64_t atom_begin;
  std::int64_t atom_end;
  std::int64_t pair_begin;
  std::int64_t pair_end;
};

struct PairAtoms {
  std::int64_t first;
  std::int64_t second;
};

__device__ bool system_is_valid(const std::uint32_t* system_errors, std::int64_t system) {
  return atomicAdd(const_cast<std::uint32_t*>(system_errors) + system, 0u) ==
         static_cast<std::uint32_t>(Gfn2AES2DeviceError::kSuccess);
}

__device__ void record_system_error(std::uint32_t* system_errors, std::int64_t system,
                                    std::uint32_t* device_error, Gfn2AES2DeviceError error) {
  const std::uint32_t code = static_cast<std::uint32_t>(error);
  if (atomicCAS(system_errors + system, static_cast<std::uint32_t>(Gfn2AES2DeviceError::kSuccess),
                code) == static_cast<std::uint32_t>(Gfn2AES2DeviceError::kSuccess)) {
    atomicCAS(device_error, static_cast<std::uint32_t>(Gfn2AES2DeviceError::kSuccess), code);
  }
}

__host__ __device__ std::int64_t triangle_count(std::int64_t value) {
  return (value & 1LL) == 0LL ? (value / 2LL) * (value - 1LL) : value * ((value - 1LL) / 2LL);
}

__device__ PairAtoms pair_atoms(const SystemRanges& ranges, std::int64_t pair) {
  const std::int64_t atom_count = ranges.atom_end - ranges.atom_begin;
  const std::int64_t local_pair = pair - ranges.pair_begin;
  std::int64_t second =
      static_cast<std::int64_t>(0.5 * (1.0 + sqrt(1.0 + 8.0 * static_cast<double>(local_pair))));
  if (second < 1) {
    second = 1;
  }
  if (second >= atom_count) {
    second = atom_count - 1;
  }
  while (second > 1 && triangle_count(second) > local_pair) {
    --second;
  }
  while (second + 1 < atom_count && triangle_count(second + 1) <= local_pair) {
    ++second;
  }
  const std::int64_t first = local_pair - triangle_count(second);
  return {ranges.atom_begin + first, ranges.atom_begin + second};
}

__device__ std::int64_t pair_index(const SystemRanges& ranges, std::int64_t first,
                                   std::int64_t second) {
  const std::int64_t local_first = first - ranges.atom_begin;
  const std::int64_t local_second = second - ranges.atom_begin;
  return ranges.pair_begin + triangle_count(local_second) + local_first;
}

__device__ bool load_and_validate_system(const Gfn2AES2DeviceBatch& batch, std::int64_t system,
                                         SystemRanges* ranges, int* valid,
                                         std::uint32_t* system_errors,
                                         std::uint32_t* device_error) {
  if (threadIdx.x == 0) {
    if (!system_is_valid(system_errors, system)) {
      *valid = 0;
    } else {
      ranges->atom_begin = batch.atom_offsets[system];
      ranges->atom_end = batch.atom_offsets[system + 1];
      ranges->pair_begin = batch.pair_offsets[system];
      ranges->pair_end = batch.pair_offsets[system + 1];
      /* Validate endpoints before subtracting so hostile INT64 extrema fail closed. */
      const bool endpoints_valid =
          ranges->atom_begin >= 0 && ranges->atom_begin <= ranges->atom_end &&
          ranges->atom_end <= batch.total_atoms && ranges->pair_begin >= 0 &&
          ranges->pair_begin <= ranges->pair_end && ranges->pair_end <= batch.total_pairs;
      *valid = endpoints_valid ? 1 : 0;
      if (*valid != 0) {
        const std::int64_t atom_count = ranges->atom_end - ranges->atom_begin;
        const std::int64_t pair_count = ranges->pair_end - ranges->pair_begin;
        const bool atom_count_representable =
            atom_count <= 1 || atom_count <= kInt64Maximum / (atom_count - 1);
        const std::int64_t expected_pairs =
            atom_count_representable ? triangle_count(atom_count) : -1;
        *valid = expected_pairs >= 0 && pair_count == expected_pairs &&
                 (system != 0 || (ranges->atom_begin == 0 && ranges->pair_begin == 0)) &&
                 (system + 1 != batch.batch_size ||
                  (ranges->atom_end == batch.total_atoms && ranges->pair_end == batch.total_pairs));
      }
      if (*valid == 0) {
        record_system_error(system_errors, system, device_error,
                            Gfn2AES2DeviceError::kInvalidOffsets);
      }
    }
  }
  __syncthreads();
  if (*valid == 0) {
    return false;
  }

  for (std::int64_t atom = ranges->atom_begin + threadIdx.x; atom < ranges->atom_end;
       atom += blockDim.x) {
    const double dipole_kernel = batch.dipole_kernel[atom];
    const double quadrupole_kernel = batch.quadrupole_kernel[atom];
    const double radius = batch.multipole_radius[atom];
    const double valence_cn = batch.multipole_valence_cn[atom];
    if (!isfinite(dipole_kernel) || !isfinite(quadrupole_kernel) || !(radius > 0.0) ||
        radius > vibeqc::xtb::generated::gfn2_aes2_multipole_max_radius || !isfinite(radius) ||
        !(valence_cn > 0.0) || !isfinite(valence_cn)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2AES2DeviceError::kInvalidElementParameter);
      atomicExch(valid, 0);
    }
  }
  __syncthreads();
  return *valid != 0;
}

__device__ double multipole_radius(const Gfn2AES2DeviceBatch& batch, std::int64_t atom,
                                   double coordination_number) {
  vibeqc::xtb::generated::Gfn2AES2RadiusResult result{};
  if (!vibeqc::xtb::generated::evaluate_gfn2_aes2_radius(
          coordination_number, batch.multipole_radius[atom], batch.multipole_valence_cn[atom],
          result)) {
    return nan("");
  }
  return result.radius;
}

__device__ double multipole_radius_cn_derivative(const Gfn2AES2DeviceBatch& batch,
                                                 std::int64_t atom, double coordination_number) {
  vibeqc::xtb::generated::Gfn2AES2RadiusResult result{};
  if (!vibeqc::xtb::generated::evaluate_gfn2_aes2_radius(
          coordination_number, batch.multipole_radius[atom], batch.multipole_valence_cn[atom],
          result)) {
    return nan("");
  }
  return result.cn_derivative;
}

__device__ bool pair_kernels(double distance, double radius, double* kernel3, double* kernel5) {
  vibeqc::xtb::generated::Gfn2AES2KernelResult result{};
  if (!vibeqc::xtb::generated::evaluate_gfn2_aes2_pair_kernels(distance, radius, result)) {
    return false;
  }
  *kernel3 = result.kernel3;
  *kernel5 = result.kernel5;
  return true;
}

__device__ bool finite_add(double contribution, double* target) {
  if (!isfinite(contribution)) {
    return false;
  }
  const double updated = *target + contribution;
  if (!isfinite(updated)) {
    return false;
  }
  *target = updated;
  return true;
}

__device__ bool finite_cache_pair(const double* pair_data) {
  return isfinite(pair_data[0]) && isfinite(pair_data[1]) && isfinite(pair_data[2]) &&
         pair_data[3] >= 0.0 && isfinite(pair_data[3]) && pair_data[4] >= 0.0 &&
         isfinite(pair_data[4]);
}

__device__ bool validate_multipoles_and_cache(
    const Gfn2AES2DeviceBatch& batch, const SystemRanges& ranges, const Gfn2AES2DeviceCache& cache,
    const double* atomic_charges, const double* atomic_dipoles, const double* atomic_quadrupoles,
    std::int64_t system, int* valid, std::uint32_t* system_errors, std::uint32_t* device_error) {
  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    bool finite = isfinite(atomic_charges[atom]);
    for (int component = 0; component < 3; ++component) {
      finite = finite && isfinite(atomic_dipoles[atom * 3 + component]);
    }
    for (int component = 0; component < 6; ++component) {
      finite = finite && isfinite(atomic_quadrupoles[atom * 6 + component]);
    }
    if (!finite) {
      record_system_error(system_errors, system, device_error,
                          Gfn2AES2DeviceError::kNonfiniteMultipole);
      atomicExch(valid, 0);
    }
  }
  for (std::int64_t pair = ranges.pair_begin + threadIdx.x; pair < ranges.pair_end;
       pair += blockDim.x) {
    if (!finite_cache_pair(cache.pair_data + pair * kGfn2AES2PairDataElements)) {
      record_system_error(system_errors, system, device_error, Gfn2AES2DeviceError::kInvalidCache);
      atomicExch(valid, 0);
    }
  }
  __syncthreads();
  return *valid != 0;
}

__global__ void geometry_preflight_kernel(Gfn2AES2DeviceBatch batch, const double* positions,
                                          const double* coordination_numbers, double* pair_scratch,
                                          std::uint32_t* system_errors,
                                          std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int valid;
  if (!load_and_validate_system(batch, system, &ranges, &valid, system_errors, device_error)) {
    return;
  }

  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    const std::int64_t coordinate = atom * 3;
    if (!isfinite(positions[coordinate]) || !isfinite(positions[coordinate + 1]) ||
        !isfinite(positions[coordinate + 2])) {
      record_system_error(system_errors, system, device_error,
                          Gfn2AES2DeviceError::kNonfinitePosition);
      atomicExch(&valid, 0);
    }
    if (!(coordination_numbers[atom] >= 0.0) || !isfinite(coordination_numbers[atom])) {
      record_system_error(system_errors, system, device_error,
                          Gfn2AES2DeviceError::kInvalidCoordination);
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }

  for (std::int64_t pair = ranges.pair_begin + threadIdx.x; pair < ranges.pair_end;
       pair += blockDim.x) {
    const PairAtoms atoms = pair_atoms(ranges, pair);
    const double dx = positions[atoms.first * 3] - positions[atoms.second * 3];
    const double dy = positions[atoms.first * 3 + 1] - positions[atoms.second * 3 + 1];
    const double dz = positions[atoms.first * 3 + 2] - positions[atoms.second * 3 + 2];
    if (!isfinite(dx) || !isfinite(dy) || !isfinite(dz)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2AES2DeviceError::kCoordinateDifferenceOverflow);
      continue;
    }
    const double distance = hypot(hypot(dx, dy), dz);
    const double distance_squared = distance * distance;
    if (!(distance > 0.0) || !isfinite(distance) || distance_squared < kMinimumDistanceSquared) {
      record_system_error(system_errors, system, device_error,
                          Gfn2AES2DeviceError::kInvalidGeometry);
      continue;
    }
    const double first_radius =
        multipole_radius(batch, atoms.first, coordination_numbers[atoms.first]);
    const double second_radius =
        multipole_radius(batch, atoms.second, coordination_numbers[atoms.second]);
    const double average_radius = 0.5 * (first_radius + second_radius);
    double kernel3 = 0.0;
    double kernel5 = 0.0;
    if (!(first_radius > 0.0) || !isfinite(first_radius) || !(second_radius > 0.0) ||
        !isfinite(second_radius) || !(average_radius > 0.0) || !isfinite(average_radius) ||
        !pair_kernels(distance, average_radius, &kernel3, &kernel5)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2AES2DeviceError::kNonfiniteKernelArithmetic);
      continue;
    }
    double* const output = pair_scratch + pair * kGfn2AES2PairDataElements;
    output[0] = dx;
    output[1] = dy;
    output[2] = dz;
    output[3] = kernel3;
    output[4] = kernel5;
  }
}

__global__ void publish_pair_cache_kernel(Gfn2AES2DeviceBatch batch, const double* pair_scratch,
                                          double* pair_data, const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!system_is_valid(system_errors, system)) {
    return;
  }
  const std::int64_t pair_begin = batch.pair_offsets[system];
  const std::int64_t pair_end = batch.pair_offsets[system + 1];
  for (std::int64_t pair = pair_begin + threadIdx.x; pair < pair_end; pair += blockDim.x) {
    const std::int64_t base = pair * kGfn2AES2PairDataElements;
    for (int component = 0; component < kGfn2AES2PairDataElements; ++component) {
      pair_data[base + component] = pair_scratch[base + component];
    }
  }
}

__global__ void potential_preflight_kernel(Gfn2AES2DeviceBatch batch, Gfn2AES2DeviceCache cache,
                                           const double* atomic_charges,
                                           const double* atomic_dipoles,
                                           const double* atomic_quadrupoles,
                                           double* potential_scratch, std::uint32_t* system_errors,
                                           std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int valid;
  if (!load_and_validate_system(batch, system, &ranges, &valid, system_errors, device_error) ||
      !validate_multipoles_and_cache(batch, ranges, cache, atomic_charges, atomic_dipoles,
                                     atomic_quadrupoles, system, &valid, system_errors,
                                     device_error)) {
    return;
  }

  const std::int64_t total_atoms = batch.total_atoms;
  double* const charge_scratch = potential_scratch;
  double* const dipole_scratch = potential_scratch + total_atoms;
  double* const quadrupole_scratch = potential_scratch + total_atoms * 4;
  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    vibeqc::xtb::generated::Gfn2AES2OnsitePotentialResult onsite{};
    bool finite_result = vibeqc::xtb::generated::evaluate_gfn2_aes2_onsite_potential(
        batch.dipole_kernel[atom], batch.quadrupole_kernel[atom], atomic_dipoles + atom * 3,
        atomic_quadrupoles + atom * 6, onsite);
    double charge_potential = 0.0;
    double dipole_potential[3];
    double quadrupole_potential[6];
    for (int component = 0; component < 3; ++component) {
      dipole_potential[component] = onsite.dipole[component];
    }
    for (int component = 0; component < 6; ++component) {
      quadrupole_potential[component] = onsite.quadrupole[component];
    }

    for (std::int64_t peer = ranges.atom_begin; finite_result && peer < ranges.atom_end; ++peer) {
      if (peer == atom) {
        continue;
      }
      const bool target_is_first = atom < peer;
      const std::int64_t first = target_is_first ? atom : peer;
      const std::int64_t second = target_is_first ? peer : atom;
      const std::int64_t pair = pair_index(ranges, first, second);
      const double* const pair_data = cache.pair_data + pair * kGfn2AES2PairDataElements;
      vibeqc::xtb::generated::Gfn2AES2PairPotentialResult pair_result{};
      finite_result = vibeqc::xtb::generated::evaluate_gfn2_aes2_pair_potential(
          pair_data[0], pair_data[1], pair_data[2], pair_data[3], pair_data[4],
          atomic_charges[first], atomic_charges[second], atomic_dipoles + first * 3,
          atomic_dipoles + second * 3, atomic_quadrupoles + first * 6,
          atomic_quadrupoles + second * 6, pair_result);
      if (!finite_result) {
        break;
      }
      const double charge = target_is_first ? pair_result.first_charge : pair_result.second_charge;
      finite_result = finite_add(charge, &charge_potential);
      for (int component = 0; finite_result && component < 3; ++component) {
        const double value = target_is_first ? pair_result.first_dipole[component]
                                             : pair_result.second_dipole[component];
        finite_result = finite_add(value, &dipole_potential[component]);
      }
      for (int component = 0; finite_result && component < 6; ++component) {
        const double value = target_is_first ? pair_result.first_quadrupole[component]
                                             : pair_result.second_quadrupole[component];
        finite_result = finite_add(value, &quadrupole_potential[component]);
      }
    }

    if (!finite_result) {
      record_system_error(system_errors, system, device_error,
                          Gfn2AES2DeviceError::kNonfinitePotentialArithmetic);
      continue;
    }
    charge_scratch[atom] = charge_potential;
    for (int component = 0; component < 3; ++component) {
      dipole_scratch[atom * 3 + component] = dipole_potential[component];
    }
    for (int component = 0; component < 6; ++component) {
      quadrupole_scratch[atom * 6 + component] = quadrupole_potential[component];
    }
  }
}

__global__ void publish_potential_kernel(Gfn2AES2DeviceBatch batch, const double* potential_scratch,
                                         double* charge_potentials, double* dipole_potentials,
                                         double* quadrupole_potentials,
                                         const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!system_is_valid(system_errors, system)) {
    return;
  }
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  const double* const charge_scratch = potential_scratch;
  const double* const dipole_scratch = potential_scratch + batch.total_atoms;
  const double* const quadrupole_scratch = potential_scratch + batch.total_atoms * 4;
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    charge_potentials[atom] = charge_scratch[atom];
    for (int component = 0; component < 3; ++component) {
      dipole_potentials[atom * 3 + component] = dipole_scratch[atom * 3 + component];
    }
    for (int component = 0; component < 6; ++component) {
      quadrupole_potentials[atom * 6 + component] = quadrupole_scratch[atom * 6 + component];
    }
  }
}

__device__ bool onsite_energy(const Gfn2AES2DeviceBatch& batch, std::int64_t atom,
                              const double* dipoles, const double* quadrupoles, double* energy) {
  return vibeqc::xtb::generated::evaluate_gfn2_aes2_onsite_energy(
      batch.dipole_kernel[atom], batch.quadrupole_kernel[atom], dipoles + atom * 3,
      quadrupoles + atom * 6, *energy);
}

__device__ bool pair_energy(const double* pair_data, std::int64_t first, std::int64_t second,
                            const double* charges, const double* dipoles, const double* quadrupoles,
                            double* energy) {
  return vibeqc::xtb::generated::evaluate_gfn2_aes2_pair_energy(
      pair_data[0], pair_data[1], pair_data[2], pair_data[3], pair_data[4], charges[first],
      charges[second], dipoles + first * 3, dipoles + second * 3, quadrupoles + first * 6,
      quadrupoles + second * 6, *energy);
}

__global__ void energy_preflight_kernel(Gfn2AES2DeviceBatch batch, Gfn2AES2DeviceCache cache,
                                        const double* atomic_charges, const double* atomic_dipoles,
                                        const double* atomic_quadrupoles, const double* energies,
                                        double* batch_scratch, std::uint32_t* system_errors,
                                        std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int valid;
  __shared__ double partial[kThreadsPerBlock];
  if (!load_and_validate_system(batch, system, &ranges, &valid, system_errors, device_error) ||
      !validate_multipoles_and_cache(batch, ranges, cache, atomic_charges, atomic_dipoles,
                                     atomic_quadrupoles, system, &valid, system_errors,
                                     device_error)) {
    return;
  }

  if (threadIdx.x == 0 && !isfinite(energies[system])) {
    record_system_error(system_errors, system, device_error,
                        Gfn2AES2DeviceError::kNonfiniteEnergySeed);
  }
  __syncthreads();

  double local_energy = 0.0;
  bool finite_result = true;
  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    double contribution = 0.0;
    finite_result = onsite_energy(batch, atom, atomic_dipoles, atomic_quadrupoles, &contribution) &&
                    finite_add(contribution, &local_energy) && finite_result;
  }
  for (std::int64_t pair = ranges.pair_begin + threadIdx.x; pair < ranges.pair_end;
       pair += blockDim.x) {
    const PairAtoms atoms = pair_atoms(ranges, pair);
    double contribution = 0.0;
    finite_result =
        pair_energy(cache.pair_data + pair * kGfn2AES2PairDataElements, atoms.first, atoms.second,
                    atomic_charges, atomic_dipoles, atomic_quadrupoles, &contribution) &&
        finite_add(contribution, &local_energy) && finite_result;
  }
  if (!finite_result) {
    record_system_error(system_errors, system, device_error,
                        Gfn2AES2DeviceError::kNonfiniteEnergyArithmetic);
  }
  partial[threadIdx.x] = local_energy;
  __syncthreads();
  for (int stride = kThreadsPerBlock / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0 && system_is_valid(system_errors, system)) {
    const double updated = energies[system] + partial[0];
    if (!isfinite(partial[0]) || !isfinite(updated)) {
      record_system_error(system_errors, system, device_error,
                          Gfn2AES2DeviceError::kNonfiniteEnergyArithmetic);
    } else {
      batch_scratch[system] = updated;
    }
  }
}

__global__ void publish_energy_kernel(std::int64_t batch_size, const double* batch_scratch,
                                      double* energies, const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system < batch_size && system_is_valid(system_errors, system)) {
    energies[system] = batch_scratch[system];
  }
}

inline constexpr std::uint32_t kSccInactiveMarker = 0xffffffffu;

__device__ void record_aes2_scc_plan_error(std::uint32_t* plan_error, Gfn2AES2DeviceError error) {
  atomicCAS(plan_error, static_cast<std::uint32_t>(Gfn2AES2DeviceError::kSuccess),
            static_cast<std::uint32_t>(error));
}

__global__ void aes2_scc_plan_preflight_kernel(Gfn2AES2DeviceBatch batch, Gfn2AES2DeviceCache cache,
                                               std::uint64_t geometry_generation,
                                               Gfn2SccIterationDeviceActivity activity,
                                               std::uint32_t* plan_error) {
  __shared__ int sequence_active;
  __shared__ int any_active;
  if (threadIdx.x == 0) {
    sequence_active = atomicAdd(const_cast<std::uint32_t*>(activity.sequence_active), 0u) == 1u;
    any_active = 0;
  }
  __syncthreads();
  if (sequence_active == 0) {
    return;
  }
  for (std::int64_t system = threadIdx.x; system < batch.batch_size; system += blockDim.x) {
    if (activity.active_mask[system] == 1u) {
      atomicExch(&any_active, 1);
    }
  }
  __syncthreads();
  if (any_active == 0 || atomicAdd(plan_error, 0u) != 0u) {
    return;
  }
  if (threadIdx.x == 0 &&
      (geometry_generation == 0u || cache.geometry_generation != geometry_generation ||
       cache.plan_token != batch.plan_token)) {
    record_aes2_scc_plan_error(plan_error, Gfn2AES2DeviceError::kCacheMismatch);
  }
  __syncthreads();
  if (atomicAdd(plan_error, 0u) != 0u) {
    return;
  }
  for (std::int64_t system = threadIdx.x; system < batch.batch_size; system += blockDim.x) {
    if (activity.active_mask[system] != 1u) {
      continue;
    }
    const std::int64_t atom_begin = batch.atom_offsets[system];
    const std::int64_t atom_end = batch.atom_offsets[system + 1];
    const std::int64_t pair_begin = batch.pair_offsets[system];
    const std::int64_t pair_end = batch.pair_offsets[system + 1];
    const bool endpoints_valid = atom_begin >= 0 && atom_begin <= atom_end &&
                                 atom_end <= batch.total_atoms && pair_begin >= 0 &&
                                 pair_begin <= pair_end && pair_end <= batch.total_pairs;
    bool valid = endpoints_valid;
    if (valid) {
      const std::int64_t atoms = atom_end - atom_begin;
      const bool triangle_representable = atoms <= 1 || atoms <= kInt64Maximum / (atoms - 1);
      valid = triangle_representable && pair_end - pair_begin == triangle_count(atoms);
    }
    if (valid && system == 0) {
      valid = atom_begin == 0 && pair_begin == 0;
    }
    if (valid && system + 1 == batch.batch_size) {
      valid = atom_end == batch.total_atoms && pair_end == batch.total_pairs;
    }
    if (!valid) {
      record_aes2_scc_plan_error(plan_error, Gfn2AES2DeviceError::kInvalidOffsets);
    }
  }
}

__global__ void aes2_scc_gate_kernel(std::int64_t batch_size,
                                     Gfn2SccIterationDeviceActivity activity,
                                     const std::uint32_t* plan_error,
                                     std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system >= batch_size) {
    return;
  }
  const bool sequence_active =
      atomicAdd(const_cast<std::uint32_t*>(activity.sequence_active), 0u) == 1u;
  if (!sequence_active) {
    atomicCAS(system_errors + system, 0u, kSccInactiveMarker);
    return;
  }
  if (activity.active_mask[system] != 1u ||
      atomicAdd(const_cast<std::uint32_t*>(plan_error), 0u) != 0u) {
    atomicCAS(system_errors + system, 0u, kSccInactiveMarker);
  }
}

__global__ void aes2_scc_cleanup_kernel(std::int64_t batch_size, std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (system < batch_size) {
    atomicCAS(system_errors + system, kSccInactiveMarker, 0u);
  }
}

__device__ bool pair_vjp(const double* pair_data, double average_radius,
                         double first_radius_cn_derivative, double second_radius_cn_derivative,
                         std::int64_t first, std::int64_t second, const double* charges,
                         const double* dipoles, const double* quadrupoles, double* pair_gradient,
                         double* first_cn_adjoint, double* second_cn_adjoint) {
  vibeqc::xtb::generated::Gfn2AES2PairVjpResult result{};
  if (!vibeqc::xtb::generated::evaluate_gfn2_aes2_pair_vjp(
          pair_data[0], pair_data[1], pair_data[2], pair_data[3], pair_data[4], average_radius,
          first_radius_cn_derivative, second_radius_cn_derivative, charges[first], charges[second],
          dipoles + first * 3, dipoles + second * 3, quadrupoles + first * 6,
          quadrupoles + second * 6, result)) {
    return false;
  }
  for (int axis = 0; axis < 3; ++axis) {
    pair_gradient[axis] = result.gradient[axis];
  }
  *first_cn_adjoint = result.first_cn_adjoint;
  *second_cn_adjoint = result.second_cn_adjoint;
  return true;
}

__global__ void vjp_preflight_kernel(Gfn2AES2DeviceBatch batch, Gfn2AES2DeviceCache cache,
                                     const double* positions, const double* coordination_numbers,
                                     const double* atomic_charges, const double* atomic_dipoles,
                                     const double* atomic_quadrupoles, const double* gradients,
                                     const double* coordination_adjoints, double* gradient_scratch,
                                     double* coordination_scratch, std::uint32_t* system_errors,
                                     std::uint32_t* device_error) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  __shared__ SystemRanges ranges;
  __shared__ int valid;
  if (!load_and_validate_system(batch, system, &ranges, &valid, system_errors, device_error) ||
      !validate_multipoles_and_cache(batch, ranges, cache, atomic_charges, atomic_dipoles,
                                     atomic_quadrupoles, system, &valid, system_errors,
                                     device_error)) {
    return;
  }

  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    const std::int64_t coordinate = atom * 3;
    if (!isfinite(positions[coordinate]) || !isfinite(positions[coordinate + 1]) ||
        !isfinite(positions[coordinate + 2])) {
      record_system_error(system_errors, system, device_error,
                          Gfn2AES2DeviceError::kNonfinitePosition);
      atomicExch(&valid, 0);
    }
    if (!(coordination_numbers[atom] >= 0.0) || !isfinite(coordination_numbers[atom])) {
      record_system_error(system_errors, system, device_error,
                          Gfn2AES2DeviceError::kInvalidCoordination);
      atomicExch(&valid, 0);
    }
    if (!isfinite(gradients[coordinate]) || !isfinite(gradients[coordinate + 1]) ||
        !isfinite(gradients[coordinate + 2]) || !isfinite(coordination_adjoints[atom])) {
      record_system_error(system_errors, system, device_error,
                          Gfn2AES2DeviceError::kNonfiniteGradientSeed);
      atomicExch(&valid, 0);
    }
  }
  __syncthreads();
  if (valid == 0) {
    return;
  }

  for (std::int64_t atom = ranges.atom_begin + threadIdx.x; atom < ranges.atom_end;
       atom += blockDim.x) {
    double gradient_contribution[3] = {0.0, 0.0, 0.0};
    double coordination_contribution = 0.0;
    bool finite_result = true;
    for (std::int64_t peer = ranges.atom_begin; finite_result && peer < ranges.atom_end; ++peer) {
      if (peer == atom) {
        continue;
      }
      const bool target_is_first = atom < peer;
      const std::int64_t first = target_is_first ? atom : peer;
      const std::int64_t second = target_is_first ? peer : atom;
      const std::int64_t pair = pair_index(ranges, first, second);
      const double* const pair_data = cache.pair_data + pair * kGfn2AES2PairDataElements;
      const double dx = positions[first * 3] - positions[second * 3];
      const double dy = positions[first * 3 + 1] - positions[second * 3 + 1];
      const double dz = positions[first * 3 + 2] - positions[second * 3 + 2];
      const double distance = hypot(hypot(dx, dy), dz);
      const double distance_squared = distance * distance;
      const double first_radius = multipole_radius(batch, first, coordination_numbers[first]);
      const double second_radius = multipole_radius(batch, second, coordination_numbers[second]);
      const double average_radius = 0.5 * (first_radius + second_radius);
      double expected_kernel3 = 0.0;
      double expected_kernel5 = 0.0;
      if (!isfinite(dx) || !isfinite(dy) || !isfinite(dz) || !(distance > 0.0) ||
          !isfinite(distance) || distance_squared < kMinimumDistanceSquared ||
          !(first_radius > 0.0) || !isfinite(first_radius) || !(second_radius > 0.0) ||
          !isfinite(second_radius) || !(average_radius > 0.0) || !isfinite(average_radius) ||
          !pair_kernels(distance, average_radius, &expected_kernel3, &expected_kernel5)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2AES2DeviceError::kNonfiniteVjpArithmetic);
        finite_result = false;
        break;
      }
      if (pair_data[0] != dx || pair_data[1] != dy || pair_data[2] != dz ||
          pair_data[3] != expected_kernel3 || pair_data[4] != expected_kernel5) {
        record_system_error(system_errors, system, device_error,
                            Gfn2AES2DeviceError::kCacheMismatch);
        finite_result = false;
        break;
      }
      const double first_cn_derivative =
          multipole_radius_cn_derivative(batch, first, coordination_numbers[first]);
      const double second_cn_derivative =
          multipole_radius_cn_derivative(batch, second, coordination_numbers[second]);
      double pair_gradient[3];
      double first_cn_adjoint = 0.0;
      double second_cn_adjoint = 0.0;
      if (!pair_vjp(pair_data, average_radius, first_cn_derivative, second_cn_derivative, first,
                    second, atomic_charges, atomic_dipoles, atomic_quadrupoles, pair_gradient,
                    &first_cn_adjoint, &second_cn_adjoint)) {
        record_system_error(system_errors, system, device_error,
                            Gfn2AES2DeviceError::kNonfiniteVjpArithmetic);
        finite_result = false;
        break;
      }
      for (int axis = 0; axis < 3; ++axis) {
        finite_result = finite_add((target_is_first ? 1.0 : -1.0) * pair_gradient[axis],
                                   &gradient_contribution[axis]) &&
                        finite_result;
      }
      finite_result = finite_add(target_is_first ? first_cn_adjoint : second_cn_adjoint,
                                 &coordination_contribution) &&
                      finite_result;
    }
    for (int axis = 0; finite_result && axis < 3; ++axis) {
      const double updated = gradients[atom * 3 + axis] + gradient_contribution[axis];
      if (!isfinite(updated)) {
        finite_result = false;
      } else {
        gradient_scratch[atom * 3 + axis] = updated;
      }
    }
    const double updated_coordination = coordination_adjoints[atom] + coordination_contribution;
    if (!isfinite(updated_coordination)) {
      finite_result = false;
    } else {
      coordination_scratch[atom] = updated_coordination;
    }
    if (!finite_result) {
      record_system_error(system_errors, system, device_error,
                          Gfn2AES2DeviceError::kNonfiniteVjpArithmetic);
    }
  }
}

__global__ void publish_vjp_kernel(Gfn2AES2DeviceBatch batch, const double* gradient_scratch,
                                   const double* coordination_scratch, double* gradients,
                                   double* coordination_adjoints,
                                   const std::uint32_t* system_errors) {
  const std::int64_t system = static_cast<std::int64_t>(blockIdx.x);
  if (!system_is_valid(system_errors, system)) {
    return;
  }
  const std::int64_t atom_begin = batch.atom_offsets[system];
  const std::int64_t atom_end = batch.atom_offsets[system + 1];
  for (std::int64_t atom = atom_begin + threadIdx.x; atom < atom_end; atom += blockDim.x) {
    for (int axis = 0; axis < 3; ++axis) {
      gradients[atom * 3 + axis] = gradient_scratch[atom * 3 + axis];
    }
    coordination_adjoints[atom] = coordination_scratch[atom];
  }
}

bool count_bytes(std::int64_t count, std::size_t element_size, std::size_t* bytes) noexcept {
  if (count < 0 ||
      static_cast<std::uint64_t>(count) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / element_size)) {
    return false;
  }
  *bytes = static_cast<std::size_t>(count) * element_size;
  return true;
}

bool ranges_overlap(const void* first, std::size_t first_bytes, const void* second,
                    std::size_t second_bytes) noexcept {
  if (first_bytes == 0u || second_bytes == 0u) {
    return false;
  }
  const auto first_begin = reinterpret_cast<std::uintptr_t>(first);
  const auto second_begin = reinterpret_cast<std::uintptr_t>(second);
  if (first_begin > std::numeric_limits<std::uintptr_t>::max() - first_bytes ||
      second_begin > std::numeric_limits<std::uintptr_t>::max() - second_bytes) {
    return true;
  }
  return first_begin < second_begin + second_bytes && second_begin < first_begin + first_bytes;
}

bool is_aligned(const void* pointer, std::size_t alignment) noexcept {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0u;
}

struct MemoryRange {
  const void* data;
  std::size_t bytes;
};

template <std::size_t N>
bool pairwise_disjoint(const MemoryRange (&ranges)[N]) noexcept {
  for (std::size_t first = 0; first < N; ++first) {
    for (std::size_t second = first + 1; second < N; ++second) {
      if (ranges_overlap(ranges[first].data, ranges[first].bytes, ranges[second].data,
                         ranges[second].bytes)) {
        return false;
      }
    }
  }
  return true;
}

struct CommonBytes {
  std::size_t atom_offsets = 0u;
  std::size_t pair_offsets = 0u;
  std::size_t atoms = 0u;
  std::size_t positions = 0u;
  std::size_t dipoles = 0u;
  std::size_t quadrupoles = 0u;
  std::size_t pair_data = 0u;
  std::size_t potentials = 0u;
  std::size_t batch = 0u;
  std::size_t system_errors = 0u;
};

cudaError_t validate_common(const Gfn2AES2DeviceBatch& batch, std::uint32_t* system_errors,
                            std::uint32_t* device_error, CommonBytes* bytes) noexcept {
  if (batch.batch_size <= 0 || batch.total_atoms <= 0 || batch.total_pairs < 0 ||
      batch.plan_token == 0u ||
      batch.batch_size > static_cast<std::int64_t>(std::numeric_limits<unsigned int>::max()) ||
      batch.batch_size == std::numeric_limits<std::int64_t>::max() ||
      batch.atom_offset_count != batch.batch_size + 1 ||
      batch.pair_offset_count != batch.batch_size + 1 ||
      batch.dipole_kernel_count != batch.total_atoms ||
      batch.quadrupole_kernel_count != batch.total_atoms ||
      batch.multipole_radius_count != batch.total_atoms ||
      batch.multipole_valence_cn_count != batch.total_atoms || batch.atom_offsets == nullptr ||
      batch.pair_offsets == nullptr || batch.dipole_kernel == nullptr ||
      batch.quadrupole_kernel == nullptr || batch.multipole_radius == nullptr ||
      batch.multipole_valence_cn == nullptr || system_errors == nullptr ||
      device_error == nullptr || !is_aligned(batch.atom_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.pair_offsets, alignof(std::int64_t)) ||
      !is_aligned(batch.dipole_kernel, alignof(double)) ||
      !is_aligned(batch.quadrupole_kernel, alignof(double)) ||
      !is_aligned(batch.multipole_radius, alignof(double)) ||
      !is_aligned(batch.multipole_valence_cn, alignof(double)) ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t))) {
    return cudaErrorInvalidValue;
  }
  std::int64_t pair_elements = 0;
  std::int64_t potential_elements = 0;
  if (batch.total_pairs > std::numeric_limits<std::int64_t>::max() / kGfn2AES2PairDataElements ||
      batch.total_atoms >
          std::numeric_limits<std::int64_t>::max() / kGfn2AES2PotentialElementsPerAtom) {
    return cudaErrorInvalidValue;
  }
  pair_elements = batch.total_pairs * kGfn2AES2PairDataElements;
  potential_elements = batch.total_atoms * kGfn2AES2PotentialElementsPerAtom;
  return count_bytes(batch.batch_size + 1, sizeof(std::int64_t), &bytes->atom_offsets) &&
                 count_bytes(batch.batch_size + 1, sizeof(std::int64_t), &bytes->pair_offsets) &&
                 count_bytes(batch.total_atoms, sizeof(double), &bytes->atoms) &&
                 count_bytes(batch.total_atoms * 3, sizeof(double), &bytes->positions) &&
                 count_bytes(batch.total_atoms * 3, sizeof(double), &bytes->dipoles) &&
                 count_bytes(batch.total_atoms * 6, sizeof(double), &bytes->quadrupoles) &&
                 count_bytes(pair_elements, sizeof(double), &bytes->pair_data) &&
                 count_bytes(potential_elements, sizeof(double), &bytes->potentials) &&
                 count_bytes(batch.batch_size, sizeof(double), &bytes->batch) &&
                 count_bytes(batch.batch_size, sizeof(std::uint32_t), &bytes->system_errors)
             ? cudaSuccess
             : cudaErrorInvalidValue;
}

cudaError_t validate_cache(const Gfn2AES2DeviceBatch& batch,
                           const Gfn2AES2DeviceCache& cache) noexcept {
  const std::int64_t required = batch.total_pairs * kGfn2AES2PairDataElements;
  return cache.plan_token == batch.plan_token && cache.pair_data_elements == required &&
                 (required == 0 ||
                  (cache.pair_data != nullptr && is_aligned(cache.pair_data, alignof(double))))
             ? cudaSuccess
             : cudaErrorInvalidValue;
}

cudaError_t validate_scc_common(const Gfn2AES2DeviceBatch& batch, const Gfn2AES2DeviceCache& cache,
                                const Gfn2SccIterationDeviceActivity& activity,
                                const Gfn2AES2DeviceWorkspace& workspace,
                                std::uint32_t* system_errors, std::uint32_t* plan_error,
                                CommonBytes* bytes) noexcept {
  cudaError_t status = validate_common(batch, system_errors, plan_error, bytes);
  if (status != cudaSuccess) {
    return status;
  }
  const std::int64_t required_pair_elements = batch.total_pairs * kGfn2AES2PairDataElements;
  if (cache.pair_data_elements != required_pair_elements ||
      (required_pair_elements != 0 &&
       (cache.pair_data == nullptr || !is_aligned(cache.pair_data, alignof(double)))) ||
      activity.active_mask == nullptr || activity.sequence_active == nullptr ||
      activity.batch_elements != batch.batch_size || activity.sequence_elements != 1 ||
      activity.plan_token != batch.plan_token || workspace.scc_peer_error_scratch == nullptr ||
      workspace.scc_peer_error_elements != 1 ||
      !is_aligned(activity.sequence_active, alignof(std::uint32_t)) ||
      !is_aligned(workspace.scc_peer_error_scratch, alignof(std::uint32_t))) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t check_launch() noexcept { return cudaGetLastError(); }

}  // namespace

cudaError_t reset_gfn2_aes2_device_errors_cuda(std::int64_t batch_size,
                                               std::uint32_t* system_errors,
                                               std::uint32_t* device_error,
                                               cudaStream_t stream) noexcept {
  std::size_t error_bytes = 0u;
  if (batch_size <= 0 || system_errors == nullptr || device_error == nullptr ||
      !is_aligned(system_errors, alignof(std::uint32_t)) ||
      !is_aligned(device_error, alignof(std::uint32_t)) ||
      !count_bytes(batch_size, sizeof(std::uint32_t), &error_bytes) ||
      ranges_overlap(system_errors, error_bytes, device_error, sizeof(*device_error))) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaMemsetAsync(system_errors, 0, error_bytes, stream);
  if (status != cudaSuccess) {
    return status;
  }
  return cudaMemsetAsync(device_error, 0, sizeof(*device_error), stream);
}

cudaError_t update_gfn2_aes2_geometry_cache_cuda(
    const Gfn2AES2DeviceBatch& batch, const double* positions, const double* coordination_numbers,
    const Gfn2AES2DeviceCache& cache, const Gfn2AES2DeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error, cudaStream_t stream) noexcept {
  CommonBytes bytes{};
  cudaError_t status = validate_common(batch, system_errors, device_error, &bytes);
  if (status != cudaSuccess) {
    return status;
  }
  status = validate_cache(batch, cache);
  const std::int64_t required_pairs = batch.total_pairs * kGfn2AES2PairDataElements;
  if (status != cudaSuccess || positions == nullptr || coordination_numbers == nullptr ||
      workspace.pair_elements < required_pairs ||
      (required_pairs != 0 && workspace.pair_scratch == nullptr) ||
      !is_aligned(positions, alignof(double)) ||
      !is_aligned(coordination_numbers, alignof(double)) ||
      (required_pairs != 0 && !is_aligned(workspace.pair_scratch, alignof(double)))) {
    return status == cudaSuccess ? cudaErrorInvalidValue : status;
  }
  const MemoryRange ranges[]{{batch.atom_offsets, bytes.atom_offsets},
                             {batch.pair_offsets, bytes.pair_offsets},
                             {batch.dipole_kernel, bytes.atoms},
                             {batch.quadrupole_kernel, bytes.atoms},
                             {batch.multipole_radius, bytes.atoms},
                             {batch.multipole_valence_cn, bytes.atoms},
                             {positions, bytes.positions},
                             {coordination_numbers, bytes.atoms},
                             {cache.pair_data, bytes.pair_data},
                             {workspace.pair_scratch, bytes.pair_data},
                             {system_errors, bytes.system_errors},
                             {device_error, sizeof(*device_error)}};
  if (!pairwise_disjoint(ranges)) {
    return cudaErrorInvalidValue;
  }
  geometry_preflight_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                              stream>>>(batch, positions, coordination_numbers,
                                        workspace.pair_scratch, system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  publish_pair_cache_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                              stream>>>(batch, workspace.pair_scratch, cache.pair_data,
                                        system_errors);
  return check_launch();
}

cudaError_t evaluate_gfn2_aes2_potential_cuda(
    const Gfn2AES2DeviceBatch& batch, const Gfn2AES2DeviceCache& cache,
    const double* atomic_charges, const double* atomic_dipoles, const double* atomic_quadrupoles,
    double* charge_potentials, double* dipole_potentials, double* quadrupole_potentials,
    const Gfn2AES2DeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* device_error, cudaStream_t stream) noexcept {
  CommonBytes bytes{};
  cudaError_t status = validate_common(batch, system_errors, device_error, &bytes);
  if (status != cudaSuccess) {
    return status;
  }
  status = validate_cache(batch, cache);
  const std::int64_t required = batch.total_atoms * kGfn2AES2PotentialElementsPerAtom;
  if (status != cudaSuccess || atomic_charges == nullptr || atomic_dipoles == nullptr ||
      atomic_quadrupoles == nullptr || charge_potentials == nullptr ||
      dipole_potentials == nullptr || quadrupole_potentials == nullptr ||
      workspace.potential_scratch == nullptr || workspace.potential_elements < required ||
      !is_aligned(atomic_charges, alignof(double)) ||
      !is_aligned(atomic_dipoles, alignof(double)) ||
      !is_aligned(atomic_quadrupoles, alignof(double)) ||
      !is_aligned(charge_potentials, alignof(double)) ||
      !is_aligned(dipole_potentials, alignof(double)) ||
      !is_aligned(quadrupole_potentials, alignof(double)) ||
      !is_aligned(workspace.potential_scratch, alignof(double))) {
    return status == cudaSuccess ? cudaErrorInvalidValue : status;
  }
  const MemoryRange ranges[]{{batch.atom_offsets, bytes.atom_offsets},
                             {batch.pair_offsets, bytes.pair_offsets},
                             {batch.dipole_kernel, bytes.atoms},
                             {batch.quadrupole_kernel, bytes.atoms},
                             {batch.multipole_radius, bytes.atoms},
                             {batch.multipole_valence_cn, bytes.atoms},
                             {cache.pair_data, bytes.pair_data},
                             {atomic_charges, bytes.atoms},
                             {atomic_dipoles, bytes.dipoles},
                             {atomic_quadrupoles, bytes.quadrupoles},
                             {charge_potentials, bytes.atoms},
                             {dipole_potentials, bytes.dipoles},
                             {quadrupole_potentials, bytes.quadrupoles},
                             {workspace.potential_scratch, bytes.potentials},
                             {system_errors, bytes.system_errors},
                             {device_error, sizeof(*device_error)}};
  if (!pairwise_disjoint(ranges)) {
    return cudaErrorInvalidValue;
  }
  potential_preflight_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                               stream>>>(batch, cache, atomic_charges, atomic_dipoles,
                                         atomic_quadrupoles, workspace.potential_scratch,
                                         system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  publish_potential_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                             stream>>>(batch, workspace.potential_scratch, charge_potentials,
                                       dipole_potentials, quadrupole_potentials, system_errors);
  return check_launch();
}

cudaError_t add_gfn2_aes2_energy_cuda(const Gfn2AES2DeviceBatch& batch,
                                      const Gfn2AES2DeviceCache& cache,
                                      const double* atomic_charges, const double* atomic_dipoles,
                                      const double* atomic_quadrupoles, double* energies,
                                      const Gfn2AES2DeviceWorkspace& workspace,
                                      std::uint32_t* system_errors, std::uint32_t* device_error,
                                      cudaStream_t stream) noexcept {
  CommonBytes bytes{};
  cudaError_t status = validate_common(batch, system_errors, device_error, &bytes);
  if (status != cudaSuccess) {
    return status;
  }
  status = validate_cache(batch, cache);
  if (status != cudaSuccess || atomic_charges == nullptr || atomic_dipoles == nullptr ||
      atomic_quadrupoles == nullptr || energies == nullptr || workspace.batch_scratch == nullptr ||
      workspace.batch_elements < batch.batch_size || !is_aligned(atomic_charges, alignof(double)) ||
      !is_aligned(atomic_dipoles, alignof(double)) ||
      !is_aligned(atomic_quadrupoles, alignof(double)) || !is_aligned(energies, alignof(double)) ||
      !is_aligned(workspace.batch_scratch, alignof(double))) {
    return status == cudaSuccess ? cudaErrorInvalidValue : status;
  }
  const MemoryRange ranges[]{{batch.atom_offsets, bytes.atom_offsets},
                             {batch.pair_offsets, bytes.pair_offsets},
                             {batch.dipole_kernel, bytes.atoms},
                             {batch.quadrupole_kernel, bytes.atoms},
                             {batch.multipole_radius, bytes.atoms},
                             {batch.multipole_valence_cn, bytes.atoms},
                             {cache.pair_data, bytes.pair_data},
                             {atomic_charges, bytes.atoms},
                             {atomic_dipoles, bytes.dipoles},
                             {atomic_quadrupoles, bytes.quadrupoles},
                             {energies, bytes.batch},
                             {workspace.batch_scratch, bytes.batch},
                             {system_errors, bytes.system_errors},
                             {device_error, sizeof(*device_error)}};
  if (!pairwise_disjoint(ranges)) {
    return cudaErrorInvalidValue;
  }
  energy_preflight_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                            stream>>>(batch, cache, atomic_charges, atomic_dipoles,
                                      atomic_quadrupoles, energies, workspace.batch_scratch,
                                      system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  const unsigned int blocks =
      static_cast<unsigned int>((batch.batch_size + kThreadsPerBlock - 1) / kThreadsPerBlock);
  publish_energy_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch.batch_size, workspace.batch_scratch, energies, system_errors);
  return check_launch();
}

cudaError_t add_gfn2_aes2_vjp_cuda(
    const Gfn2AES2DeviceBatch& batch, const Gfn2AES2DeviceCache& cache, const double* positions,
    const double* coordination_numbers, std::uint64_t geometry_generation,
    const double* atomic_charges, const double* atomic_dipoles, const double* atomic_quadrupoles,
    double* gradients, double* coordination_adjoints, const Gfn2AES2DeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* device_error, cudaStream_t stream) noexcept {
  CommonBytes bytes{};
  cudaError_t status = validate_common(batch, system_errors, device_error, &bytes);
  if (status != cudaSuccess) {
    return status;
  }
  status = validate_cache(batch, cache);
  if (status != cudaSuccess || geometry_generation != cache.geometry_generation ||
      positions == nullptr || coordination_numbers == nullptr || atomic_charges == nullptr ||
      atomic_dipoles == nullptr || atomic_quadrupoles == nullptr || gradients == nullptr ||
      coordination_adjoints == nullptr || workspace.gradient_scratch == nullptr ||
      workspace.coordination_scratch == nullptr ||
      workspace.gradient_elements < batch.total_atoms * 3 ||
      workspace.coordination_elements < batch.total_atoms ||
      !is_aligned(positions, alignof(double)) ||
      !is_aligned(coordination_numbers, alignof(double)) ||
      !is_aligned(atomic_charges, alignof(double)) ||
      !is_aligned(atomic_dipoles, alignof(double)) ||
      !is_aligned(atomic_quadrupoles, alignof(double)) || !is_aligned(gradients, alignof(double)) ||
      !is_aligned(coordination_adjoints, alignof(double)) ||
      !is_aligned(workspace.gradient_scratch, alignof(double)) ||
      !is_aligned(workspace.coordination_scratch, alignof(double))) {
    return status == cudaSuccess ? cudaErrorInvalidValue : status;
  }
  const MemoryRange ranges[]{{batch.atom_offsets, bytes.atom_offsets},
                             {batch.pair_offsets, bytes.pair_offsets},
                             {batch.dipole_kernel, bytes.atoms},
                             {batch.quadrupole_kernel, bytes.atoms},
                             {batch.multipole_radius, bytes.atoms},
                             {batch.multipole_valence_cn, bytes.atoms},
                             {cache.pair_data, bytes.pair_data},
                             {positions, bytes.positions},
                             {coordination_numbers, bytes.atoms},
                             {atomic_charges, bytes.atoms},
                             {atomic_dipoles, bytes.dipoles},
                             {atomic_quadrupoles, bytes.quadrupoles},
                             {gradients, bytes.positions},
                             {coordination_adjoints, bytes.atoms},
                             {workspace.gradient_scratch, bytes.positions},
                             {workspace.coordination_scratch, bytes.atoms},
                             {system_errors, bytes.system_errors},
                             {device_error, sizeof(*device_error)}};
  if (!pairwise_disjoint(ranges)) {
    return cudaErrorInvalidValue;
  }
  vjp_preflight_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                         stream>>>(batch, cache, positions, coordination_numbers, atomic_charges,
                                   atomic_dipoles, atomic_quadrupoles, gradients,
                                   coordination_adjoints, workspace.gradient_scratch,
                                   workspace.coordination_scratch, system_errors, device_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  publish_vjp_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0, stream>>>(
      batch, workspace.gradient_scratch, workspace.coordination_scratch, gradients,
      coordination_adjoints, system_errors);
  return check_launch();
}

cudaError_t evaluate_gfn2_aes2_scc_potential_cuda(
    const Gfn2AES2DeviceBatch& batch, const Gfn2AES2DeviceCache& cache,
    std::uint64_t geometry_generation, const Gfn2SccIterationDeviceActivity& activity,
    const double* atomic_charges, const double* atomic_dipoles, const double* atomic_quadrupoles,
    double* charge_potentials, double* dipole_potentials, double* quadrupole_potentials,
    const Gfn2AES2DeviceWorkspace& workspace, std::uint32_t* system_errors,
    std::uint32_t* plan_error, cudaStream_t stream) noexcept {
  CommonBytes bytes{};
  cudaError_t status =
      validate_scc_common(batch, cache, activity, workspace, system_errors, plan_error, &bytes);
  if (status != cudaSuccess) {
    /* Do not derive element counts from a batch that failed overflow validation. */
    return status;
  }
  const std::int64_t required = batch.total_atoms * kGfn2AES2PotentialElementsPerAtom;
  if (atomic_charges == nullptr || atomic_dipoles == nullptr || atomic_quadrupoles == nullptr ||
      charge_potentials == nullptr || dipole_potentials == nullptr ||
      quadrupole_potentials == nullptr || workspace.potential_scratch == nullptr ||
      workspace.potential_elements < required || !is_aligned(atomic_charges, alignof(double)) ||
      !is_aligned(atomic_dipoles, alignof(double)) ||
      !is_aligned(atomic_quadrupoles, alignof(double)) ||
      !is_aligned(charge_potentials, alignof(double)) ||
      !is_aligned(dipole_potentials, alignof(double)) ||
      !is_aligned(quadrupole_potentials, alignof(double)) ||
      !is_aligned(workspace.potential_scratch, alignof(double))) {
    return cudaErrorInvalidValue;
  }
  const MemoryRange ranges[]{
      {batch.atom_offsets, bytes.atom_offsets},
      {batch.pair_offsets, bytes.pair_offsets},
      {batch.dipole_kernel, bytes.atoms},
      {batch.quadrupole_kernel, bytes.atoms},
      {batch.multipole_radius, bytes.atoms},
      {batch.multipole_valence_cn, bytes.atoms},
      {cache.pair_data, bytes.pair_data},
      {activity.active_mask, static_cast<std::size_t>(batch.batch_size)},
      {activity.sequence_active, sizeof(*activity.sequence_active)},
      {atomic_charges, bytes.atoms},
      {atomic_dipoles, bytes.dipoles},
      {atomic_quadrupoles, bytes.quadrupoles},
      {charge_potentials, bytes.atoms},
      {dipole_potentials, bytes.dipoles},
      {quadrupole_potentials, bytes.quadrupoles},
      {workspace.potential_scratch, bytes.potentials},
      {workspace.scc_peer_error_scratch, sizeof(*workspace.scc_peer_error_scratch)},
      {system_errors, bytes.system_errors},
      {plan_error, sizeof(*plan_error)}};
  if (!pairwise_disjoint(ranges)) {
    return cudaErrorInvalidValue;
  }
  status = cudaMemsetAsync(workspace.scc_peer_error_scratch, 0,
                           sizeof(*workspace.scc_peer_error_scratch), stream);
  if (status != cudaSuccess) {
    return status;
  }
  aes2_scc_plan_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
      batch, cache, geometry_generation, activity, plan_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  const unsigned int gate_blocks =
      static_cast<unsigned int>((batch.batch_size + kThreadsPerBlock - 1) / kThreadsPerBlock);
  aes2_scc_gate_kernel<<<gate_blocks, kThreadsPerBlock, 0, stream>>>(batch.batch_size, activity,
                                                                     plan_error, system_errors);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  potential_preflight_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                               stream>>>(batch, cache, atomic_charges, atomic_dipoles,
                                         atomic_quadrupoles, workspace.potential_scratch,
                                         system_errors, workspace.scc_peer_error_scratch);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  publish_potential_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                             stream>>>(batch, workspace.potential_scratch, charge_potentials,
                                       dipole_potentials, quadrupole_potentials, system_errors);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  aes2_scc_cleanup_kernel<<<gate_blocks, kThreadsPerBlock, 0, stream>>>(batch.batch_size,
                                                                        system_errors);
  return check_launch();
}

cudaError_t evaluate_gfn2_aes2_scc_energy_cuda(
    const Gfn2AES2DeviceBatch& batch, const Gfn2AES2DeviceCache& cache,
    std::uint64_t geometry_generation, const Gfn2SccIterationDeviceActivity& activity,
    const double* atomic_charges, const double* atomic_dipoles, const double* atomic_quadrupoles,
    double* component_energies, const Gfn2AES2DeviceWorkspace& workspace,
    std::uint32_t* system_errors, std::uint32_t* plan_error, cudaStream_t stream) noexcept {
  CommonBytes bytes{};
  cudaError_t status =
      validate_scc_common(batch, cache, activity, workspace, system_errors, plan_error, &bytes);
  if (status != cudaSuccess || atomic_charges == nullptr || atomic_dipoles == nullptr ||
      atomic_quadrupoles == nullptr || component_energies == nullptr ||
      workspace.batch_scratch == nullptr || workspace.batch_elements < batch.batch_size ||
      !is_aligned(atomic_charges, alignof(double)) ||
      !is_aligned(atomic_dipoles, alignof(double)) ||
      !is_aligned(atomic_quadrupoles, alignof(double)) ||
      !is_aligned(component_energies, alignof(double)) ||
      !is_aligned(workspace.batch_scratch, alignof(double))) {
    return status == cudaSuccess ? cudaErrorInvalidValue : status;
  }
  const MemoryRange ranges[]{
      {batch.atom_offsets, bytes.atom_offsets},
      {batch.pair_offsets, bytes.pair_offsets},
      {batch.dipole_kernel, bytes.atoms},
      {batch.quadrupole_kernel, bytes.atoms},
      {batch.multipole_radius, bytes.atoms},
      {batch.multipole_valence_cn, bytes.atoms},
      {cache.pair_data, bytes.pair_data},
      {activity.active_mask, static_cast<std::size_t>(batch.batch_size)},
      {activity.sequence_active, sizeof(*activity.sequence_active)},
      {atomic_charges, bytes.atoms},
      {atomic_dipoles, bytes.dipoles},
      {atomic_quadrupoles, bytes.quadrupoles},
      {component_energies, bytes.batch},
      {workspace.batch_scratch, bytes.batch},
      {workspace.scc_peer_error_scratch, sizeof(*workspace.scc_peer_error_scratch)},
      {system_errors, bytes.system_errors},
      {plan_error, sizeof(*plan_error)}};
  if (!pairwise_disjoint(ranges)) {
    return cudaErrorInvalidValue;
  }
  status = cudaMemsetAsync(workspace.scc_peer_error_scratch, 0,
                           sizeof(*workspace.scc_peer_error_scratch), stream);
  if (status == cudaSuccess) {
    status = cudaMemsetAsync(workspace.batch_scratch, 0, bytes.batch, stream);
  }
  if (status != cudaSuccess) {
    return status;
  }
  aes2_scc_plan_preflight_kernel<<<1, kThreadsPerBlock, 0, stream>>>(
      batch, cache, geometry_generation, activity, plan_error);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  const unsigned int blocks =
      static_cast<unsigned int>((batch.batch_size + kThreadsPerBlock - 1) / kThreadsPerBlock);
  aes2_scc_gate_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(batch.batch_size, activity,
                                                                plan_error, system_errors);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  energy_preflight_kernel<<<static_cast<unsigned int>(batch.batch_size), kThreadsPerBlock, 0,
                            stream>>>(
      batch, cache, atomic_charges, atomic_dipoles, atomic_quadrupoles, workspace.batch_scratch,
      workspace.batch_scratch, system_errors, workspace.scc_peer_error_scratch);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  publish_energy_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      batch.batch_size, workspace.batch_scratch, component_energies, system_errors);
  status = check_launch();
  if (status != cudaSuccess) {
    return status;
  }
  aes2_scc_cleanup_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(batch.batch_size, system_errors);
  return check_launch();
}

}  // namespace vibeqc::xtb::detail::cuda
