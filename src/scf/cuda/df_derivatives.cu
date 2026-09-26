#include <limits>

#include "generated_df_derivative_policy.cuh"
#include "generated_df_derivative_schedule.cuh"
#include "molecule/basis.hpp"
#include "runtime/cuda_subgroup.cuh"
#include "scf/cuda/df_derivatives.cuh"
namespace vibeqc::scf {
namespace {
namespace products = runtime::cuda_gaussian_products;
using Policy = generated_df_policy::Derivative;
constexpr std::size_t terms = molecule::kMaximumAoExpansionTerms;
using Schedule = generated_df_policy::WeightedSchedule;
constexpr unsigned threads = Schedule::block_threads;
constexpr unsigned lanes = Schedule::lanes_per_element;
constexpr unsigned elements_per_block = threads / lanes;
static_assert(threads % 32 == 0 && threads % lanes == 0);

/** Contract independent primitive partitions before the shared gradient sink. */
template <unsigned Lanes, unsigned Rank>
__device__ Policy::Accumulator cooperative_contract(const products::Factor (&factors)[Rank],
                                                    const double* positions, unsigned lane) {
  // Capture before traversal: different contraction lengths diverge within a
  // warp, but every live subgroup must rendezvous with the same original mask.
  const auto mask = __activemask();
  auto result = products::contract<Policy, terms>(factors, positions, lane, Lanes);
  runtime::subgroup_sum<Rank, Lanes>(result.gradient, mask);
  return result;
}

/** Full dense weights count once; shared atoms are summed by the runtime sink. */
template <unsigned Lanes = 1, bool SkipSpShells = false>
__device__ void contract(DfDerivativeBasisView o, DfDerivativeBasisView x, const double* positions,
                         unsigned kind, std::size_t element, double weight, double* gradient,
                         unsigned lane = 0) {
  if (weight == 0) return;
  const auto p = static_cast<std::int64_t>(element % x.nbf);
  if (kind) {
    const products::Factor factors[2]{{x, static_cast<std::int64_t>(element / x.nbf)}, {x, p}};
    const auto result = cooperative_contract<Lanes>(factors, positions, lane);
    if (lane == 0) products::scatter(factors, result, weight, gradient);
  } else {
    if constexpr (SkipSpShells) {
      // Every expansion term has the shell's total angular degree, including
      // spherical d/f terms. The shell worker consumed these s/p components.
      const auto mu = element / x.nbf / o.nbf, nu = element / x.nbf % o.nbf;
      const auto* ai = o.term_angular + 3 * terms * mu;
      const auto* aj = o.term_angular + 3 * terms * nu;
      const auto* ap = x.term_angular + 3 * terms * p;
      const unsigned li = ai[0] + ai[1] + ai[2], lj = aj[0] + aj[1] + aj[2];
      const unsigned lp = ap[0] + ap[1] + ap[2];
      if (li <= 1 && lj <= 1 && lp <= 1 && li + lj + lp != 0) return;
    }
    const products::Factor factors[3]{{o, static_cast<std::int64_t>(element / x.nbf / o.nbf)},
                                      {o, static_cast<std::int64_t>(element / x.nbf % o.nbf)},
                                      {x, p}};
    const auto result = cooperative_contract<Lanes>(factors, positions, lane);
    if (lane == 0) products::scatter(factors, result, weight, gradient);
  }
}
template <bool DistributedSink, bool SkipSpShells = false>
__global__ void derivative_tile(DfDerivativeBasisView o, DfDerivativeBasisView x,
                                const double* positions, unsigned kind, runtime::StridedRange range,
                                std::size_t count, const double* weights, unsigned schedule,
                                double* gradient, std::size_t begin, std::size_t gradient_stride,
                                unsigned gradient_copies) {
  if constexpr (DistributedSink) gradient += (blockIdx.x % gradient_copies) * gradient_stride;
  const auto thread = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (schedule) {
    if (thread == 0)
      for (std::size_t item = 0; item < count; ++item)
        contract(o, x, positions, kind, range.index(begin + item), weights[item], gradient);
  } else if (thread / lanes < count)
    contract<lanes, SkipSpShells>(o, x, positions, kind, range.index(begin + thread / lanes),
                                  weights[thread / lanes], gradient, thread % lanes);
}
}  // namespace
cudaError_t launch_df_derivative_tile(DfDerivativeBasisView o, DfDerivativeBasisView x,
                                      const double* positions, unsigned kind,
                                      runtime::StridedRange range, std::size_t count,
                                      const double* weights, unsigned schedule, double* gradient,
                                      cudaStream_t stream, std::size_t begin,
                                      std::size_t gradient_stride, unsigned gradient_copies,
                                      bool skip_sp_shells) {
  const auto maximum = std::numeric_limits<std::size_t>::max();
  if (!o.nbf || !x.nbf || kind > 1 || schedule > 1 || !positions || !weights || !gradient ||
      !count || o.nbf > maximum / o.nbf || o.nbf * o.nbf > maximum / x.nbf ||
      x.nbf > maximum / x.nbf || !gradient_copies ||
      (gradient_copies > 1 && (!gradient_stride || gradient_stride > maximum / gradient_copies)))
    return cudaErrorInvalidValue;
  const auto elements = kind ? x.nbf * x.nbf : o.nbf * o.nbf * x.nbf;
  if (!range.row_length || !range.row_stride || !range.column_stride || range.offset >= elements ||
      count - 1 > maximum - begin ||
      (count - 1) / elements_per_block >= std::numeric_limits<int>::max())
    return cudaErrorInvalidValue;
  // A transposed range is not globally monotone: its last partial row may
  // end below the preceding full row. Check both maxima without overflow.
  const auto fits = [&](std::size_t index) {
    const auto row = index / range.row_length, column = index % range.row_length;
    const auto remaining = elements - 1 - range.offset;
    return row <= remaining / range.row_stride &&
           column <= (remaining - row * range.row_stride) / range.column_stride;
  };
  const auto end = begin + count - 1;
  if (!fits(end) || (end / range.row_length > begin / range.row_length &&
                     !fits((end / range.row_length) * range.row_length - 1)))
    return cudaErrorInvalidValue;
  const auto blocks = schedule ? 1U : static_cast<unsigned>((count - 1) / elements_per_block + 1);
  if (skip_sp_shells && (kind || schedule || gradient_copies != 1)) return cudaErrorInvalidValue;
  if (skip_sp_shells)
    derivative_tile<false, true><<<blocks, threads, 0, stream>>>(
        o, x, positions, kind, range, count, weights, schedule, gradient, begin, 0, 1);
  else if (gradient_copies > 1)
    derivative_tile<true><<<blocks, threads, 0, stream>>>(o, x, positions, kind, range, count,
                                                          weights, schedule, gradient, begin,
                                                          gradient_stride, gradient_copies);
  else
    derivative_tile<false><<<blocks, threads, 0, stream>>>(
        o, x, positions, kind, range, count, weights, schedule, gradient, begin, 0, 1);
  return cudaPeekAtLastError();
}
}  // namespace vibeqc::scf
