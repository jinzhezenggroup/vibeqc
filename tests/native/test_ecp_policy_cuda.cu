#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <vector>

#include "ecp_policy_cases.hpp"
#include "generated_ecp_ao.cuh"

__global__ void evaluate_policy(const EcpPolicyCase* cases, int count, int* results) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count)
    results[i] = vibeqc::generated::ecp_grid_pair_accepted(cases[i].coarse, cases[i].fine,
                                                           cases[i].derivative);
}

int main() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
  EcpPolicyCase* input = nullptr;
  int* output = nullptr;
  auto check = [](cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
  };
  int status = 0;
  try {
    const auto cases = ecp_policy_cases();
    std::vector<int> results(cases.size());
    check(cudaMalloc(&input, cases.size() * sizeof(EcpPolicyCase)));
    check(cudaMalloc(&output, results.size() * sizeof(int)));
    check(cudaMemcpy(input, cases.data(), cases.size() * sizeof(EcpPolicyCase),
                     cudaMemcpyHostToDevice));
    evaluate_policy<<<(cases.size() + 63) / 64, 64>>>(input, cases.size(), output);
    check(cudaGetLastError());
    check(cudaMemcpy(results.data(), output, results.size() * sizeof(int), cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < cases.size(); ++i)
      if (results[i] != cases[i].accepted)
        throw std::runtime_error("generated CUDA ECP convergence policy changed acceptance");
    std::cout << cases.size() << " independent ECP policy boundary cases passed on CUDA\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    status = 1;
  }
  if (output) cudaFree(output);
  if (input) cudaFree(input);
  return status;
}
