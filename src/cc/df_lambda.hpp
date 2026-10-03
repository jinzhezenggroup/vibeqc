#pragma once

#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include "cc/lambda_response.hpp"

namespace generativeqc::cc::detail {

/** Resident native DF scientific actions for the existing Lambda solve.
 * Retained-core and every auxiliary derivative execute on one stream. Inputs
 * are staged once; outputs detach only after the complete sticky-error audit.
 * The full numeric bound includes borrowed problem/CC state, host Krylov and
 * publication storage, and the device arena. No ovvv/vvvv is reconstructed.
 */
class DFLambdaActions {
 public:
  DFLambdaActions(const Problem&, const SolverResult&, const LambdaOptions&, int device,
                  bool with_source, bool with_parameters);
  ~DFLambdaActions();
  DFLambdaActions(const DFLambdaActions&) = delete;
  DFLambdaActions& operator=(const DFLambdaActions&) = delete;
  void replay(double& energy, std::vector<double>& r1, std::vector<double>& r2);
  void rhs(bool independent, std::vector<double>& one, std::vector<double>& two);
  void transpose(bool independent, std::span<const double> one, std::span<const double> two,
                 std::vector<double>& out_one, std::vector<double>& out_two);
  void seeds(std::span<const double> one, std::span<const double> two);
  std::vector<double> parameter(std::string_view name, std::size_t count);
  std::pair<std::vector<double>, std::vector<double>> virtual_factors();
  const LambdaDiagnostic& diagnostic() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace generativeqc::cc::detail
