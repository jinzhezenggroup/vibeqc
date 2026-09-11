"""Raw derivative CUDA fixture and host-emittable arithmetic evaluation body."""

from .raw_values_cuda import emit_raw_value_driver


def derivative_evaluation_body():
    """Write 27 channels with explicit zero external-center S/T derivatives."""
    return r"""
  const auto pair = one::make_pair(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
  const auto st = one::overlap_kinetic_gradient(pair, p[12], p[13]);
  const auto v = one::attraction_gradient(pair, p[12], p[13], p[8], p[9], p[10]);
  for (unsigned axis = 0; axis < 3; ++axis) {
    out[axis] = st.first[axis];
    out[3+axis] = -st.first[axis];
    out[6+axis] = 0.0;
    out[9+axis] = st.second[axis];
    out[12+axis] = -st.second[axis];
    out[15+axis] = 0.0;
    out[18+axis] = p[11]*v.first[axis];
    out[21+axis] = p[11]*v.second[axis];
    out[24+axis] = -p[11]*(v.first[axis]+v.second[axis]);
  }
"""


def emit_one_electron_derivative_driver(architecture):
    """Run the same raw binary protocol/resource adapter as the value gate."""
    kernel = (
        r"""
namespace one = vibeqc::scf::generated_one_electron_derivatives;
struct Input { double values[14]; };
extern "C" __global__ void fixture_values(const Input* inputs, double* values, std::size_t count) {
  const std::size_t i = static_cast<std::size_t>(blockIdx.x)*blockDim.x + threadIdx.x;
  if (i >= count) return;
  const double* p = inputs[i].values;
  double* out = values + 27*i;
"""
        + derivative_evaluation_body()
        + "\n}\n"
    )
    return emit_raw_value_driver(
        architecture,
        header="generated_one_electron_derivatives.cuh",
        kernel=kernel,
        magic="VQOE1411",
        width=27,
    )
