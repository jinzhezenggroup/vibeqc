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
template <unsigned Lanes = 1>
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
    const products::Factor factors[3]{{o, static_cast<std::int64_t>(element / x.nbf / o.nbf)},
                                      {o, static_cast<std::int64_t>(element / x.nbf % o.nbf)},
                                      {x, p}};
    const auto result = cooperative_contract<Lanes>(factors, positions, lane);
    if (lane == 0) products::scatter(factors, result, weight, gradient);
  }
}
__global__ void derivative_tile(DfDerivativeBasisView o, DfDerivativeBasisView x,
                                const double* positions, unsigned kind, runtime::StridedRange range,
                                std::size_t count, const double* weights, unsigned schedule,
                                double* gradient, std::size_t begin) {
  const auto thread = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (schedule) {
    if (thread == 0)
      for (std::size_t item = 0; item < count; ++item)
        contract(o, x, positions, kind, range.index(begin + item), weights[item], gradient);
  } else if (thread / lanes < count)
    contract<lanes>(o, x, positions, kind, range.index(begin + thread / lanes),
                    weights[thread / lanes], gradient, thread % lanes);
}
}  // namespace
cudaError_t launch_df_derivative_tile(DfDerivativeBasisView o, DfDerivativeBasisView x,
                                      const double* positions, unsigned kind,
                                      runtime::StridedRange range, std::size_t count,
                                      const double* weights, unsigned schedule, double* gradient,
                                      cudaStream_t stream, std::size_t begin) {
  const auto maximum = std::numeric_limits<std::size_t>::max();
  if (!o.nbf || !x.nbf || kind > 1 || schedule > 1 || !positions || !weights || !gradient ||
      !count || o.nbf > maximum / o.nbf || o.nbf * o.nbf > maximum / x.nbf ||
      x.nbf > maximum / x.nbf)
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
  derivative_tile<<<blocks, threads, 0, stream>>>(o, x, positions, kind, range, count, weights,
                                                  schedule, gradient, begin);
  return cudaPeekAtLastError();
}
}  // namespace vibeqc::scf
