#include <cuda_runtime.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

#include "dft/xc_point_response.hpp"
#include "runtime/cuda_resources.cuh"

namespace {
struct Input {
  bool unrestricted{}, pbe{};
  double rho[2]{}, gradient[2][3]{}, delta[2]{}, delta_gradient[2][3]{};
};

__global__ void evaluate(const Input* in, vibeqc::dft::point::Value* out) {
  *out = in->unrestricted
             ? vibeqc::dft::point::unrestricted_response(in->pbe, in->rho, in->gradient, in->delta,
                                                         in->delta_gradient)
             : vibeqc::dft::point::restricted_response(in->pbe, in->rho[0], in->gradient[0],
                                                       in->delta[0], in->delta_gradient[0]);
}

/** Same 450-digit original-energy references and roundoff gates as the CPU
 * tier. This tests the device point algebra, separately from complete AO/SCF
 * acceptance; no CPU production point evaluator is used as an oracle. */
void independent_points(bool unrestricted) {
  using vibeqc::runtime::cuda_resource_check;
  vibeqc::runtime::OwnedCudaBuffer<Input> input(0, 1);
  vibeqc::runtime::OwnedCudaBuffer<vibeqc::dft::point::Value> output(0, 1);
  std::ifstream table(std::string(VIBEQC_SOURCE_DIR) + "/tests/data/xc/" +
                      (unrestricted ? "uks_response.tsv" : "rks_response.tsv"));
  if (!table) throw std::runtime_error("missing independent response fixture");
  unsigned count = 0;
  const unsigned spins = unrestricted ? 2 : 1;
  for (std::string line; std::getline(table, line);) {
    if (line.empty() || line[0] == '#') continue;
    Input in{};
    in.unrestricted = unrestricted;
    std::istringstream row(line);
    double method{}, expected[8]{}, magnitude[8]{};
    row >> method;
    in.pbe = method == 1;
    for (unsigned s = 0; s < spins; ++s) row >> in.rho[s];
    for (unsigned s = 0; s < spins; ++s)
      for (double& x : in.gradient[s]) row >> x;
    for (unsigned s = 0; s < spins; ++s) row >> in.delta[s];
    for (unsigned s = 0; s < spins; ++s)
      for (double& x : in.delta_gradient[s]) row >> x;
    for (unsigned c = 0; c < 4 * spins; ++c) row >> expected[c];
    for (unsigned c = 0; c < 4 * spins; ++c) row >> magnitude[c];
    if (!row) throw std::runtime_error("malformed independent response fixture");
    cuda_resource_check(cudaMemcpy(input.get(), &in, sizeof(in), cudaMemcpyHostToDevice));
    evaluate<<<1, 1>>>(input.get(), output.get());
    cuda_resource_check(cudaGetLastError());
    vibeqc::dft::point::Value value;
    cuda_resource_check(cudaMemcpy(&value, output.get(), sizeof(value), cudaMemcpyDeviceToHost));
    if (!value.valid) throw std::runtime_error("valid device point direction rejected");
    double actual[8]{};
    for (unsigned s = 0; s < spins; ++s) {
      actual[s] = value.rho[s];
      for (unsigned k = 0; k < 3; ++k) actual[spins + 3 * s + k] = value.gradient[s][k];
    }
    for (unsigned c = 0; c < 4 * spins; ++c) {
      const double tolerance =
          (unrestricted ? 3e-10 : 3e-11) * std::abs(expected[c]) +
          (unrestricted ? 64 : 32) * std::numeric_limits<double>::epsilon() * magnitude[c] +
          8 * std::numeric_limits<double>::denorm_min();
      if (!std::isfinite(actual[c]) || std::abs(actual[c] - expected[c]) > tolerance) {
        std::cerr << "spin blocks " << spins << " point " << count << " component " << c
                  << " value " << actual[c] << " expected " << expected[c] << '\n';
        throw std::runtime_error("independent CUDA response mismatch");
      }
    }
    ++count;
  }
  if (count != (unrestricted ? 48U : 30U)) throw std::runtime_error("incomplete response table");
  std::cout << count << " independent CUDA " << (unrestricted ? "UKS" : "RKS")
            << " point directions passed\n";
}
}  // namespace

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
  try {
    independent_points(false);
    independent_points(true);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
