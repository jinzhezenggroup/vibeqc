"""Standalone CUDA fixture driver for the exact generated DF primitive header."""

from .raw_values_cuda import emit_raw_value_driver


def emit_df_value_driver(architecture: str) -> str:
    """Use the shared resource/fixture protocol with DF's exact primitive ABI."""
    kernel = r"""namespace df = vibeqc::scf::generated_df;
struct Input {
  std::uint32_t centers_count;
  df::Angular angular[3];
  df::Vec3 centers[3];
  double exponents[3];
  double weight;
};
static_assert(sizeof(Input) == 144 && offsetof(Input, centers) == 40 &&
              offsetof(Input, exponents) == 112 && offsetof(Input, weight) == 136);

extern "C" __global__ void fixture_values(const Input* inputs, double* values, std::size_t count) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const Input& in = inputs[i];
  double value;
  if (in.centers_count == 2) {
    value = df::metric(in.exponents[0], in.centers[0], in.angular[0],
                       in.exponents[2], in.centers[2], in.angular[2]);
  } else if (in.centers_count == 3) {
    value = df::three_center(in.exponents[0], in.centers[0], in.angular[0],
        in.exponents[1], in.centers[1], in.angular[1],
        in.exponents[2], in.centers[2], in.angular[2]);
  } else { value = nan(""); }
  values[i] = in.weight * value;
}

"""
    return emit_raw_value_driver(
        architecture, header="df_values.cuh", kernel=kernel, magic="VQDF1421", width=1
    )
