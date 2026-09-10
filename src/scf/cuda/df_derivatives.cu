#include <limits>

#include "generated_df_policy.cuh"
#include "molecule/basis.hpp"
#include "scf/cuda/df_derivatives.cuh"
namespace vibeqc::scf {
namespace {
namespace products = runtime::cuda_gaussian_products;
using Policy = generated_df_policy::Derivative;
constexpr std::size_t terms = molecule::kMaximumAoExpansionTerms;

/** Full dense weights count once; shared atoms are summed by the runtime sink. */
__device__ void contract(DfDerivativeBasisView o, DfDerivativeBasisView x, const double* positions,
                         unsigned kind, std::size_t element, double weight, double* gradient) {
  if (weight == 0) return;
  const auto p = static_cast<std::int64_t>(element % x.nbf);
  if (kind) {
    const products::Factor factors[2]{{x, static_cast<std::int64_t>(element / x.nbf)}, {x, p}};
    const auto result = products::contract<Policy, terms>(factors, positions);
    products::scatter(factors, result, weight, gradient);
  } else {
    const products::Factor factors[3]{{o, static_cast<std::int64_t>(element / x.nbf / o.nbf)},
                                      {o, static_cast<std::int64_t>(element / x.nbf % o.nbf)},
                                      {x, p}};
    const auto result = products::contract<Policy, terms>(factors, positions);
    products::scatter(factors, result, weight, gradient);
  }
}
__global__ void derivative_tile(DfDerivativeBasisView o, DfDerivativeBasisView x,
                                const double* positions, unsigned kind, std::size_t offset,
                                std::size_t count, const double* weights, unsigned schedule,
                                double* gradient, std::size_t stride) {
  const auto thread = std::size_t{blockIdx.x} * blockDim.x + threadIdx.x;
  if (schedule) {
    if (thread == 0)
      for (std::size_t item = 0; item < count; ++item)
        contract(o, x, positions, kind, offset + item * stride, weights[item], gradient);
  } else if (thread < count)
    contract(o, x, positions, kind, offset + thread * stride, weights[thread], gradient);
}
}  // namespace
cudaError_t launch_df_derivative_tile(DfDerivativeBasisView o, DfDerivativeBasisView x,
                                      const double* positions, unsigned kind, std::size_t offset,
                                      std::size_t count, const double* weights, unsigned schedule,
                                      double* gradient, cudaStream_t stream, std::size_t stride) {
  const auto maximum = std::numeric_limits<std::size_t>::max();
  if (!o.nbf || !x.nbf || kind > 1 || schedule > 1 || !positions || !weights || !gradient ||
      !count || o.nbf > maximum / o.nbf || o.nbf * o.nbf > maximum / x.nbf ||
      x.nbf > maximum / x.nbf)
    return cudaErrorInvalidValue;
  const auto elements = kind ? x.nbf * x.nbf : o.nbf * o.nbf * x.nbf;
  if (!stride || offset >= elements || count - 1 > (elements - 1 - offset) / stride ||
      (count - 1) / 128 >= std::numeric_limits<int>::max())
    return cudaErrorInvalidValue;
  const auto blocks = schedule ? 1U : static_cast<unsigned>((count - 1) / 128 + 1);
  derivative_tile<<<blocks, 128, 0, stream>>>(o, x, positions, kind, offset, count, weights,
                                              schedule, gradient, stride);
  return cudaPeekAtLastError();
}
}  // namespace vibeqc::scf
