"""Raw DF derivative probe using the common finite-Slurm resource protocol."""

from .raw_values_cuda import emit_raw_value_driver


def emit_df_derivative_driver(architecture):
    """Keep the existing 144-byte positive-exponent primitive input contract."""
    kernel = r"""
namespace df=vibeqc::scf::generated_df_derivatives;
struct Input {
  std::uint32_t centers_count;
  df::Angular angular[3];
  df::Vec3 centers[3];
  double exponents[3];
  double weight;
};
static_assert(sizeof(Input)==144 && offsetof(Input,centers)==40 && offsetof(Input,weight)==136);
extern "C" __global__ void fixture_values(const Input* inputs,double* values,std::size_t count) {
  const auto i=std::size_t{blockIdx.x}*blockDim.x+threadIdx.x;
  if(i>=count)return;
  const auto& in=inputs[i];
  const auto r=in.centers_count==2 ? df::metric(in.exponents[0],in.centers[0],in.angular[0],in.exponents[2],in.centers[2],in.angular[2]) :
    df::three_center(in.exponents[0],in.centers[0],in.angular[0],in.exponents[1],in.centers[1],in.angular[1],in.exponents[2],in.centers[2],in.angular[2]);
  const double f[]={r.first.x,r.first.y,r.first.z,r.second.x,r.second.y,r.second.z,r.third.x,r.third.y,r.third.z};
  for(unsigned k=0;k<9;++k) values[9*i+k]=in.weight*f[k];
}
"""
    return emit_raw_value_driver(
        architecture,
        header="df_derivatives.cuh",
        kernel=kernel,
        magic="VQDF1431",
        width=9,
    )
