#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <stdexcept>
int main() {
  cudaDeviceProp p{};
  int major = 0, minor = 0;
  const int n = 200;
  auto checked = [](cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
  };
  checked(cudaFree(nullptr));
  checked(cudaGetDeviceProperties(&p, 0));
  using clock = std::chrono::steady_clock;
  auto t = clock::now();
  for (int i = 0; i < n; ++i) checked(cudaGetDeviceProperties(&p, 0));
  double props = std::chrono::duration<double>(clock::now() - t).count() / n;
  t = clock::now();
  for (int i = 0; i < n; ++i) {
    checked(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
    checked(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, 0));
  }
  double attrs = std::chrono::duration<double>(clock::now() - t).count() / n;
  std::printf(
      "{\"full_properties_seconds\":%.12g,\"two_attributes_seconds\":%.12g,\"architecture\":%d,"
      "\"repeats\":%d}\n",
      props, attrs, 10 * major + minor, n);
}
