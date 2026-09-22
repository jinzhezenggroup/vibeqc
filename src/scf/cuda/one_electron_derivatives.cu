#include <cmath>
#include <limits>

#include "generated_one_electron_derivative_policy.cuh"
#include "generated_one_electron_derivatives.cuh"
#include "molecule/basis.hpp"
#include "scf/cuda/one_electron_derivatives.cuh"

namespace vibeqc::scf {
namespace {
namespace generated = generated_one_electron_derivatives;
namespace derivative_policy = generated_one_electron_derivative_policy;
using NucleusCooperativeSchedule = derivative_policy::NucleusCooperativeSchedule;
constexpr std::size_t kTerms = molecule::kMaximumAoExpansionTerms;
constexpr unsigned kWarpThreads = 32U;
constexpr unsigned kDerivativeThreads = 128U;
constexpr unsigned kCooperativeLanes = NucleusCooperativeSchedule::group_lanes;
constexpr unsigned kCooperativeGroups = NucleusCooperativeSchedule::groups_per_block;
static_assert(NucleusCooperativeSchedule::subgroup_lanes == kWarpThreads);
static_assert(kCooperativeLanes == kWarpThreads);
static_assert(NucleusCooperativeSchedule::block_threads == kCooperativeLanes * kCooperativeGroups);

__device__ double pair_weight(const double* matrix, double scale, std::size_t offset, std::size_t i,
                              std::size_t j, std::size_t n) {
  if (!matrix) return 0.0;
  const double weight = matrix[offset + i * n + j];
  return scale * (i == j ? weight : weight + matrix[offset + j * n + i]);
}

/** Evaluate each primitive/component once and fuse its three center responses.
 * Atom ownership is applied after mathematical differentiation: coincident or
 * shared A/B/C atoms retain every contribution, including cancellation. Only
 * pair-center accumulators live in registers; each external nuclear response
 * goes directly to the bounded global gradient buffer.
 */
__device__ void contract_pair(const OneElectronDeviceView& batch, std::int64_t i, std::int64_t j,
                              const OneElectronWeightView& weights, double sign, double* gradient) {
  const std::size_t n = batch.nbf, system = i / n, offset = system * n * n;
  const double ws = pair_weight(weights.overlap, weights.overlap_scale, offset, i % n, j % n, n);
  const double wt = pair_weight(weights.kinetic, weights.kinetic_scale, offset, i % n, j % n, n);
  const double wv =
      pair_weight(weights.attraction, weights.attraction_scale, offset, i % n, j % n, n);
  if (ws == 0.0 && wt == 0.0 && wv == 0.0) return;
  const auto si = batch.ao_shells[i], sj = batch.ao_shells[j];
  const auto atom_a = batch.shell_atoms[si], atom_b = batch.shell_atoms[sj];
  const double* A = batch.positions + 3 * atom_a;
  const double* B = batch.positions + 3 * atom_b;
  double first[3]{}, second[3]{};
  for (auto a = batch.shell_primitive_offsets[si]; a < batch.shell_primitive_offsets[si + 1]; ++a) {
    for (auto b = batch.shell_primitive_offsets[sj]; b < batch.shell_primitive_offsets[sj + 1];
         ++b) {
      const auto pair =
          generated::make_pair(batch.primitive_exponents[a], batch.primitive_exponents[b], A[0],
                               A[1], A[2], B[0], B[1], B[2]);
      const double radial = batch.primitive_coefficients[a] * batch.primitive_coefficients[b];
      for (unsigned ti = 0; ti < batch.ao_term_counts[i]; ++ti) {
        const auto term_i = i * kTerms + ti;
        const auto* ai = batch.ao_term_angular + 3 * term_i;
        const auto ca = generated::component_index(ai[0], ai[1], ai[2]);
        for (unsigned tj = 0; tj < batch.ao_term_counts[j]; ++tj) {
          const auto term_j = j * kTerms + tj;
          const auto* aj = batch.ao_term_angular + 3 * term_j;
          const auto cb = generated::component_index(aj[0], aj[1], aj[2]);
          const double norm = sign * radial * batch.ao_term_coefficients[term_i] *
                              batch.ao_term_coefficients[term_j];
          if (ws != 0.0 || wt != 0.0) {
            const auto st = generated::overlap_kinetic_gradient(pair, ca, cb);
            for (unsigned axis = 0; axis < 3; ++axis) {
              const double value = norm * (ws * st.first[axis] + wt * st.second[axis]);
              first[axis] += value;
              second[axis] -= value;
            }
          }
          if (wv != 0.0) {
            for (auto atom = batch.atom_offsets[system]; atom < batch.atom_offsets[system + 1];
                 ++atom) {
              const double* C = batch.positions + 3 * atom;
              const auto v = generated::attraction_gradient(pair, ca, cb, C[0], C[1], C[2]);
              const double factor = norm * wv * batch.atomic_numbers[atom];
              for (unsigned axis = 0; axis < 3; ++axis) {
                const double da = factor * v.first[axis], db = factor * v.second[axis];
                first[axis] += da;
                second[axis] += db;
                atomicAdd(gradient + 3 * atom + axis, -da - db);
              }
            }
          }
        }
      }
    }
  }
  for (unsigned axis = 0; axis < 3; ++axis) {
    atomicAdd(gradient + 3 * atom_a + axis, first[axis]);
    atomicAdd(gradient + 3 * atom_b + axis, second[axis]);
  }
}

/** One warp owns one AO pair while lanes own nuclear centers.
 *
 * The generated derivative DAG remains the sole mathematical implementation.
 * This runtime schedule only changes ownership: lane zero evaluates the
 * nucleus-independent S/T response once, while attraction centers are striped
 * over warp lanes. Pair-center attraction responses are reduced across the
 * warp; each lane accumulates its owned nuclear response before one global
 * update per coordinate.
 */
__device__ void contract_pair_nucleus_cooperative(const OneElectronDeviceView& batch,
                                                  std::int64_t i, std::int64_t j,
                                                  const OneElectronWeightView& weights, double sign,
                                                  double* gradient,
                                                  generated::PairGeometry* shared_pair) {
  constexpr unsigned kWarpMask = 0xffffffffU;
  const unsigned lane = threadIdx.x % kCooperativeLanes;
  const std::size_t n = batch.nbf, system = i / n, offset = system * n * n;
  double ws = 0.0, wt = 0.0, wv = 0.0;
  if (lane == 0U) {
    ws = pair_weight(weights.overlap, weights.overlap_scale, offset, i % n, j % n, n);
    wt = pair_weight(weights.kinetic, weights.kinetic_scale, offset, i % n, j % n, n);
    wv = pair_weight(weights.attraction, weights.attraction_scale, offset, i % n, j % n, n);
  }
  ws = __shfl_sync(kWarpMask, ws, 0);
  wt = __shfl_sync(kWarpMask, wt, 0);
  wv = __shfl_sync(kWarpMask, wv, 0);
  if (ws == 0.0 && wt == 0.0 && wv == 0.0) return;

  const auto si = batch.ao_shells[i], sj = batch.ao_shells[j];
  const auto atom_a = batch.shell_atoms[si], atom_b = batch.shell_atoms[sj];
  const double* A = batch.positions + 3 * atom_a;
  const double* B = batch.positions + 3 * atom_b;
  double first[3]{}, second[3]{};

  // Translation-invariant S/T terms have no nuclear-center dimension.
  if (lane == 0U && (ws != 0.0 || wt != 0.0)) {
    for (auto a = batch.shell_primitive_offsets[si]; a < batch.shell_primitive_offsets[si + 1];
         ++a) {
      for (auto b = batch.shell_primitive_offsets[sj]; b < batch.shell_primitive_offsets[sj + 1];
           ++b) {
        const auto pair =
            generated::make_pair(batch.primitive_exponents[a], batch.primitive_exponents[b], A[0],
                                 A[1], A[2], B[0], B[1], B[2]);
        const double radial = batch.primitive_coefficients[a] * batch.primitive_coefficients[b];
        for (unsigned ti = 0; ti < batch.ao_term_counts[i]; ++ti) {
          const auto term_i = i * kTerms + ti;
          const auto* ai = batch.ao_term_angular + 3 * term_i;
          const auto ca = generated::component_index(ai[0], ai[1], ai[2]);
          for (unsigned tj = 0; tj < batch.ao_term_counts[j]; ++tj) {
            const auto term_j = j * kTerms + tj;
            const auto* aj = batch.ao_term_angular + 3 * term_j;
            const auto cb = generated::component_index(aj[0], aj[1], aj[2]);
            const double norm = sign * radial * batch.ao_term_coefficients[term_i] *
                                batch.ao_term_coefficients[term_j];
            const auto st = generated::overlap_kinetic_gradient(pair, ca, cb);
            for (unsigned axis = 0; axis < 3; ++axis) {
              const double value = norm * (ws * st.first[axis] + wt * st.second[axis]);
              first[axis] += value;
              second[axis] -= value;
            }
          }
        }
      }
    }
  }

  // Attraction centers are the large-system parallel dimension. All lanes
  // advance through the same nucleus tiles so the full-warp collectives remain
  // legal even on the final partial tile.
  if (wv != 0.0) {
    const auto atom_begin = batch.atom_offsets[system];
    const auto atom_end = batch.atom_offsets[system + 1];
    for (auto atom_base = atom_begin; atom_base < atom_end; atom_base += kCooperativeLanes) {
      const auto atom = atom_base + lane;
      const bool valid_atom = atom < atom_end;
      const double* C = valid_atom ? batch.positions + 3 * atom : nullptr;
      double nuclear[3]{};
      for (auto a = batch.shell_primitive_offsets[si]; a < batch.shell_primitive_offsets[si + 1];
           ++a) {
        for (auto b = batch.shell_primitive_offsets[sj]; b < batch.shell_primitive_offsets[sj + 1];
             ++b) {
          // Pair geometry depends only on the primitive pair, not the nuclear
          // center. Materialize it once per warp/tile instead of repeating the
          // exp/pow-heavy setup in every active lane.
          if (lane == 0U)
            *shared_pair =
                generated::make_pair(batch.primitive_exponents[a], batch.primitive_exponents[b],
                                     A[0], A[1], A[2], B[0], B[1], B[2]);
          __syncwarp(kWarpMask);
          const double radial = batch.primitive_coefficients[a] * batch.primitive_coefficients[b];
          for (unsigned ti = 0; ti < batch.ao_term_counts[i]; ++ti) {
            const auto term_i = i * kTerms + ti;
            const auto* ai = batch.ao_term_angular + 3 * term_i;
            const auto ca = generated::component_index(ai[0], ai[1], ai[2]);
            for (unsigned tj = 0; tj < batch.ao_term_counts[j]; ++tj) {
              const auto term_j = j * kTerms + tj;
              const auto* aj = batch.ao_term_angular + 3 * term_j;
              const auto cb = generated::component_index(aj[0], aj[1], aj[2]);
              const double norm = sign * radial * batch.ao_term_coefficients[term_i] *
                                  batch.ao_term_coefficients[term_j];
              if (valid_atom) {
                const auto v =
                    generated::attraction_gradient(*shared_pair, ca, cb, C[0], C[1], C[2]);
                const double factor = norm * wv * batch.atomic_numbers[atom];
                for (unsigned axis = 0; axis < 3; ++axis) {
                  const double da = factor * v.first[axis], db = factor * v.second[axis];
                  first[axis] += da;
                  second[axis] += db;
                  nuclear[axis] -= da + db;
                }
              }
            }
          }
          // Lane zero must not overwrite the shared pair while another lane is
          // still consuming it.
          __syncwarp(kWarpMask);
        }
      }
      if (valid_atom)
        for (unsigned axis = 0; axis < 3; ++axis)
          if (nuclear[axis] != 0.0) atomicAdd(gradient + 3 * atom + axis, nuclear[axis]);
    }
  }

  for (unsigned delta = kCooperativeLanes / 2; delta != 0; delta /= 2)
    for (unsigned axis = 0; axis < 3; ++axis) {
      first[axis] += __shfl_down_sync(kWarpMask, first[axis], delta);
      second[axis] += __shfl_down_sync(kWarpMask, second[axis], delta);
    }
  if (lane == 0U)
    for (unsigned axis = 0; axis < 3; ++axis) {
      if (first[axis] != 0.0) atomicAdd(gradient + 3 * atom_a + axis, first[axis]);
      if (second[axis] != 0.0) atomicAdd(gradient + 3 * atom_b + axis, second[axis]);
    }
}

__global__ void nucleus_cooperative_gradient(OneElectronDeviceView batch, const std::int32_t* first,
                                             const std::int32_t* second, std::size_t count,
                                             OneElectronWeightView weights,
                                             const std::uint8_t* active, double sign,
                                             double* gradient) {
  __shared__ generated::PairGeometry shared_pairs[kCooperativeGroups];
  const std::size_t warp = (std::size_t{blockIdx.x} * blockDim.x + threadIdx.x) / kCooperativeLanes;
  const std::size_t tasks = static_cast<std::size_t>(batch.batch_size) * count;
  if (warp >= tasks) return;
  const auto system = warp / count;
  if (active && !active[system]) return;
  const auto base = system * batch.nbf, pair = warp % count;
  contract_pair_nucleus_cooperative(batch, base + first[pair], base + second[pair], weights, sign,
                                    gradient, shared_pairs + threadIdx.x / kCooperativeLanes);
}

__global__ void thread_gradient(OneElectronDeviceView batch, const std::int32_t* first,
                                const std::int32_t* second, std::size_t count,
                                OneElectronWeightView weights, const std::uint8_t* active,
                                double sign, double* gradient) {
  const std::size_t task = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (task >= static_cast<std::size_t>(batch.batch_size) * count) return;
  const auto system = task / count;
  if (active && !active[system]) return;
  const auto base = system * batch.nbf, pair = task % count;
  contract_pair(batch, base + first[pair], base + second[pair], weights, sign, gradient);
}

__global__ void shell_warp_gradient(OneElectronDeviceView batch, OneElectronWeightView weights,
                                    const std::uint8_t* active, double sign, double* gradient) {
  const std::size_t task = (std::size_t{blockIdx.x} * blockDim.x + threadIdx.x) / 32;
  if (task >= batch.shell_pair_count) return;
  const auto si = batch.shell_pair_first[task], sj = batch.shell_pair_second[task];
  const auto begin_i = batch.shell_ao_offsets[si], begin_j = batch.shell_ao_offsets[sj];
  if (active && !active[begin_i / batch.nbf]) return;
  const auto ni = batch.shell_ao_offsets[si + 1] - begin_i,
             nj = batch.shell_ao_offsets[sj + 1] - begin_j;
  for (std::int64_t c = threadIdx.x % 32; c < ni * nj; c += 32) {
    const auto i = begin_i + c / nj, j = begin_j + c % nj;
    if (si == sj && i < j) continue;
    contract_pair(batch, i, j, weights, sign, gradient);
  }
}

__global__ void serial_gradient(OneElectronDeviceView batch, OneElectronWeightView weights,
                                const std::uint8_t* active, double sign, double* gradient) {
  const std::size_t system = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (system >= static_cast<std::size_t>(batch.batch_size) || (active && !active[system])) return;
  const auto base = system * batch.nbf;
  for (std::int32_t i = 0; i < batch.nbf; ++i)
    for (std::int32_t j = 0; j <= i; ++j)
      contract_pair(batch, base + i, base + j, weights, sign, gradient);
}
}  // namespace

cudaError_t launch_generated_one_electron_gradient(
    const OneElectronDeviceView& batch, const std::int32_t* pair_first,
    const std::int32_t* pair_second, std::size_t pair_count, const OneElectronWeightView& weights,
    const std::uint8_t* active, unsigned schedule, double output_sign, double* gradient,
    cudaStream_t stream) {
  if (batch.batch_size <= 0 || batch.nbf <= 0 || !gradient ||
      schedule > NucleusCooperativeSchedule::schedule_code || !std::isfinite(output_sign) ||
      !std::isfinite(weights.overlap_scale) || !std::isfinite(weights.kinetic_scale) ||
      !std::isfinite(weights.attraction_scale))
    return cudaErrorInvalidValue;
  const unsigned threads = schedule == NucleusCooperativeSchedule::schedule_code
                               ? NucleusCooperativeSchedule::block_threads
                               : kDerivativeThreads;
  const bool ao_pair_schedule =
      schedule == 0 || schedule == NucleusCooperativeSchedule::schedule_code;
  if (ao_pair_schedule && (!pair_first || !pair_second || pair_count == 0))
    return cudaErrorInvalidValue;
  // Validate products before task/grid calculations or matrix offset indexing.
  const auto systems = static_cast<std::size_t>(batch.batch_size);
  const auto n = static_cast<std::size_t>(batch.nbf);
  const auto maximum = std::numeric_limits<std::size_t>::max();
  if (n > maximum / n || systems > maximum / (n * n) ||
      (ao_pair_schedule && pair_count > maximum / systems))
    return cudaErrorInvalidValue;
  if (schedule == 1 && (!batch.shell_pair_first || !batch.shell_pair_second))
    return cudaErrorInvalidValue;
  const std::size_t tasks = schedule == 2 ? static_cast<std::size_t>(batch.batch_size)
                            : schedule == 1
                                ? batch.shell_pair_count
                                : static_cast<std::size_t>(batch.batch_size) * pair_count;
  const unsigned per_block = schedule == NucleusCooperativeSchedule::schedule_code
                                 ? NucleusCooperativeSchedule::groups_per_block
                             : schedule == 1 ? threads / kWarpThreads
                                             : threads;
  if (tasks == 0 || (tasks - 1) / per_block >= std::numeric_limits<int>::max())
    return cudaErrorInvalidValue;
  const unsigned blocks = static_cast<unsigned>((tasks - 1) / per_block + 1);
  if (schedule == 2)
    serial_gradient<<<blocks, threads, 0, stream>>>(batch, weights, active, output_sign, gradient);
  else if (schedule == 1)
    shell_warp_gradient<<<blocks, threads, 0, stream>>>(batch, weights, active, output_sign,
                                                        gradient);
  else if (schedule == NucleusCooperativeSchedule::schedule_code)
    nucleus_cooperative_gradient<<<blocks, threads, 0, stream>>>(
        batch, pair_first, pair_second, pair_count, weights, active, output_sign, gradient);
  else
    thread_gradient<<<blocks, threads, 0, stream>>>(batch, pair_first, pair_second, pair_count,
                                                    weights, active, output_sign, gradient);
  return cudaPeekAtLastError();
}

}  // namespace vibeqc::scf
