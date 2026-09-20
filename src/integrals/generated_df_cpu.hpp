#ifndef VIBEQC_INTEGRALS_GENERATED_DF_CPU_HPP
#define VIBEQC_INTEGRALS_GENERATED_DF_CPU_HPP

namespace vibeqc::integrals::generated_df_cpu {

struct Vec3 {
  double x{}, y{}, z{};
};
struct Angular {
  unsigned x{}, y{}, z{};
};
struct Response {
  double value{};
  Vec3 first{}, second{}, third{};
};

[[nodiscard]] double metric_value(double alpha, Vec3 a_center, Angular a, double gamma,
                                  Vec3 c_center, Angular c);
[[nodiscard]] double three_center_value(double alpha, Vec3 a_center, Angular a, double beta,
                                        Vec3 b_center, Angular b, double gamma, Vec3 c_center,
                                        Angular c);
[[nodiscard]] Response metric_derivative(double alpha, Vec3 a_center, Angular a, double gamma,
                                         Vec3 c_center, Angular c);
[[nodiscard]] Response three_center_derivative(double alpha, Vec3 a_center, Angular a, double beta,
                                               Vec3 b_center, Angular b, double gamma,
                                               Vec3 c_center, Angular c);

}  // namespace vibeqc::integrals::generated_df_cpu

#endif
