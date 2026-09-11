"""Raw one-electron primitive fixture using the common CUDA execution protocol."""

from .raw_values_cuda import emit_raw_value_driver


def emit_one_electron_value_driver(architecture):
    """Evaluate S/T/V separately; component indices follow the public inventory."""
    kernel = r"""
namespace one = vibeqc::scf::generated_one_electron;
struct Input {
  double alpha, beta, centers[9], charge, first, second;
};
static_assert(sizeof(Input) == 14 * sizeof(double));
extern "C" __global__ void fixture_values(const Input* inputs, double* values, std::size_t count) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const auto& in = inputs[i];
  const auto* r = in.centers;
  const auto pair = one::make_pair(in.alpha, in.beta, r[0], r[1], r[2], r[3], r[4], r[5]);
  const auto st = one::overlap_kinetic(pair, in.first, in.second);
  values[3 * i] = st.overlap;
  values[3 * i + 1] = st.kinetic;
  values[3 * i + 2] = in.charge * one::attraction(pair, in.first, in.second, r[6], r[7], r[8]);
}
"""
    return emit_raw_value_driver(
        architecture,
        header="generated_one_electron_values.cuh",
        kernel=kernel,
        magic="VQOE1401",
        width=3,
    )
