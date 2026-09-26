#include "integrals/generated_df_cpu.hpp"

#include "generated_df_derivatives_cpu.hpp"
#include "generated_df_values_cpu.hpp"

namespace vibeqc::integrals::generated_df_cpu {
namespace {
scf::generated_df::Vec3 value_vec(Vec3 v) { return {v.x, v.y, v.z}; }
scf::generated_df::Angular value_angular(Angular a) { return {a.x, a.y, a.z}; }
scf::generated_df_derivatives::Vec3 derivative_vec(Vec3 v) { return {v.x, v.y, v.z}; }
scf::generated_df_derivatives::Angular derivative_angular(Angular a) { return {a.x, a.y, a.z}; }
Vec3 response_vec(scf::generated_df_derivatives::Vec3 v) { return {v.x, v.y, v.z}; }
Response response(scf::generated_df_derivatives::Response value) {
  return {value.value, response_vec(value.first), response_vec(value.second),
          response_vec(value.third)};
}
}  // namespace

double metric_value(double alpha, Vec3 a_center, Angular a, double gamma, Vec3 c_center,
                    Angular c) {
  return scf::generated_df::metric(alpha, value_vec(a_center), value_angular(a), gamma,
                                   value_vec(c_center), value_angular(c));
}

double three_center_value(double alpha, Vec3 a_center, Angular a, double beta, Vec3 b_center,
                          Angular b, double gamma, Vec3 c_center, Angular c) {
  return scf::generated_df::three_center(alpha, value_vec(a_center), value_angular(a), beta,
                                         value_vec(b_center), value_angular(b), gamma,
                                         value_vec(c_center), value_angular(c));
}

Response metric_derivative(double alpha, Vec3 a_center, Angular a, double gamma, Vec3 c_center,
                           Angular c) {
  return response(scf::generated_df_derivatives::metric(
      alpha, derivative_vec(a_center), derivative_angular(a), gamma, derivative_vec(c_center),
      derivative_angular(c)));
}

Response three_center_derivative(double alpha, Vec3 a_center, Angular a, double beta, Vec3 b_center,
                                 Angular b, double gamma, Vec3 c_center, Angular c) {
  return response(scf::generated_df_derivatives::three_center(
      alpha, derivative_vec(a_center), derivative_angular(a), beta, derivative_vec(b_center),
      derivative_angular(b), gamma, derivative_vec(c_center), derivative_angular(c)));
}

}  // namespace vibeqc::integrals::generated_df_cpu
