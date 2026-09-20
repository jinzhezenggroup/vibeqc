#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "dft/dispersion/d4_cuda.hpp"

using namespace vibeqc::dft::dispersion;

namespace {

void checked(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template <class T>
struct Buffer {
  T* ptr{};
  std::size_t count{};
  explicit Buffer(std::size_t n) : count(n) {
    if (n) checked(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)));
  }
  ~Buffer() {
    if (ptr) cudaFree(ptr);
  }
  void upload(const T* source, std::size_t n) {
    checked(cudaMemcpy(ptr, source, n * sizeof(T), cudaMemcpyHostToDevice));
  }
  void download(T* destination, std::size_t n) const {
    checked(cudaMemcpy(destination, ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
  }
};

__global__ void scalar_batch_kernel(std::uint32_t systems, const std::uint32_t* offsets,
                                    const std::int32_t* atomic_numbers, const double* coordinates,
                                    const double* charges, D4Parameters parameters, D4Tables tables,
                                    double* workspace, D4Status* statuses, double* energies,
                                    double* gradients, double* dedq) {
  const std::uint32_t system = blockIdx.x;
  if (system >= systems || threadIdx.x != 0) return;
  const std::uint32_t begin = offsets[system];
  const int atoms = static_cast<int>(offsets[system + 1] - begin);
  statuses[system] = evaluate_d4_fixed_charge(
      atoms, atomic_numbers + begin, coordinates + 3 * begin, charges + begin, parameters, tables,
      workspace + 27u * begin, 27u * atoms, energies + 2 * system, gradients + 3 * begin,
      dedq + begin);
}

float elapsed(cudaEvent_t start, cudaEvent_t stop) {
  float milliseconds = 0.0f;
  checked(cudaEventElapsedTime(&milliseconds, start, stop));
  return milliseconds;
}

bool near(double first, double second, double tolerance) {
  return std::abs(first - second) <= tolerance * std::max({1.0, std::abs(first), std::abs(second)});
}

}  // namespace

int main(int argc, char** argv) {
  int devices = 0;
  const auto probe = cudaGetDeviceCount(&devices);
  if (probe == cudaErrorNoDevice || probe == cudaErrorInsufficientDriver ||
      (probe == cudaSuccess && devices == 0))
    return 77;
  try {
    checked(probe);
    const int systems = argc > 1 ? std::atoi(argv[1]) : 4;
    const int atoms = argc > 2 ? std::atoi(argv[2]) : 64;
    const int repeats = argc > 3 ? std::atoi(argv[3]) : 10;
    if (systems < 1 || atoms < 3 || atoms > kD4MaximumAtoms || repeats < 1)
      throw std::runtime_error(
          "usage: d4_cuda_schedule_probe [systems>=1] [atoms=3..256] [repeats>=1]");

    const std::size_t total_atoms = static_cast<std::size_t>(systems) * atoms;
    std::vector<std::uint32_t> offsets(systems + 1);
    std::vector<std::int32_t> z(total_atoms, 1);
    std::vector<double> xyz(3 * total_atoms);
    std::vector<double> q(total_atoms);
    std::vector<std::uint8_t> active(systems, 1);
    for (int system = 0; system < systems; ++system) {
      offsets[system] = static_cast<std::uint32_t>(static_cast<std::size_t>(system) * atoms);
      for (int atom = 0; atom < atoms; ++atom) {
        const std::size_t packed = static_cast<std::size_t>(system) * atoms + atom;
        xyz[3 * packed] = 2.4 * (atom % 8);
        xyz[3 * packed + 1] = 2.4 * ((atom / 8) % 8);
        xyz[3 * packed + 2] = 2.4 * (atom / 64);
        q[packed] = 0.02 * ((atom % 5) - 2);
      }
    }
    offsets[systems] = static_cast<std::uint32_t>(total_atoms);

    Buffer<std::uint32_t> d_offsets(offsets.size());
    Buffer<std::int32_t> d_z(z.size());
    Buffer<double> d_xyz(xyz.size());
    Buffer<double> d_q(q.size());
    Buffer<std::uint8_t> d_active(active.size());
    Buffer<D4Status> d_status(systems);
    Buffer<double> d_energy(2u * systems);
    Buffer<double> d_gradient(3u * total_atoms);
    Buffer<double> d_dedq(total_atoms);
    Buffer<double> d_workspace(d4_cuda_workspace_elements(total_atoms));
    Buffer<data::D4ElementData> d_elements(data::kElementCount);
    Buffer<data::D4ReferenceData> d_references(data::kReferenceCount);
    Buffer<double> d_c6(data::kReferenceC6.size());
    d_offsets.upload(offsets.data(), offsets.size());
    d_z.upload(z.data(), z.size());
    d_xyz.upload(xyz.data(), xyz.size());
    d_q.upload(q.data(), q.size());
    d_active.upload(active.data(), active.size());
    d_elements.upload(data::kElements.data(), d_elements.count);
    d_references.upload(data::kReferences.data(), d_references.count);
    d_c6.upload(data::kReferenceC6.data(), d_c6.count);

    D4Tables tables{D4ReferenceModel::gfn2,
                    d_elements.ptr,
                    d_references.ptr,
                    d_c6.ptr,
                    data::kElementCount,
                    data::kReferenceCount,
                    data::kReferenceC6.size(),
                    3.0,
                    2.0};
    const auto parameters = gfn2_d4_parameters();
    const D4CudaBatch batch{static_cast<std::uint32_t>(systems),
                            static_cast<std::uint32_t>(total_atoms),
                            d_offsets.ptr,
                            d_z.ptr,
                            d_xyz.ptr,
                            d_q.ptr,
                            d_active.ptr};
    const D4CudaResult result{d_status.ptr, d_energy.ptr, d_gradient.ptr, d_dedq.ptr};

    scalar_batch_kernel<<<systems, 1>>>(systems, d_offsets.ptr, d_z.ptr, d_xyz.ptr, d_q.ptr,
                                        parameters, tables, d_workspace.ptr, d_status.ptr,
                                        d_energy.ptr, d_gradient.ptr, d_dedq.ptr);
    checked(cudaGetLastError());
    checked(cudaDeviceSynchronize());
    std::vector<double> expected_energy(2u * systems);
    std::vector<double> expected_gradient(3u * total_atoms);
    std::vector<double> expected_dedq(total_atoms);
    d_energy.download(expected_energy.data(), expected_energy.size());
    d_gradient.download(expected_gradient.data(), expected_gradient.size());
    d_dedq.download(expected_dedq.data(), expected_dedq.size());

    checked(launch_d4_fixed_charge_batched_cuda(batch, parameters, tables, d_workspace.ptr,
                                                d_workspace.count, result));
    checked(cudaDeviceSynchronize());
    std::vector<double> actual_energy(expected_energy.size());
    std::vector<double> actual_gradient(expected_gradient.size());
    std::vector<double> actual_dedq(expected_dedq.size());
    d_energy.download(actual_energy.data(), actual_energy.size());
    d_gradient.download(actual_gradient.data(), actual_gradient.size());
    d_dedq.download(actual_dedq.data(), actual_dedq.size());
    for (std::size_t i = 0; i < actual_energy.size(); ++i)
      if (!near(actual_energy[i], expected_energy[i], 2e-10))
        throw std::runtime_error("cooperative energy differs from scalar baseline");
    for (std::size_t i = 0; i < actual_gradient.size(); ++i)
      if (!near(actual_gradient[i], expected_gradient[i], 2e-9))
        throw std::runtime_error("cooperative gradient differs from scalar baseline");
    for (std::size_t i = 0; i < actual_dedq.size(); ++i)
      if (!near(actual_dedq[i], expected_dedq[i], 2e-9))
        throw std::runtime_error("cooperative charge derivative differs from scalar baseline");

    cudaEvent_t start{}, stop{};
    checked(cudaEventCreate(&start));
    checked(cudaEventCreate(&stop));
    for (int warmup = 0; warmup < 2; ++warmup) {
      scalar_batch_kernel<<<systems, 1>>>(systems, d_offsets.ptr, d_z.ptr, d_xyz.ptr, d_q.ptr,
                                          parameters, tables, d_workspace.ptr, d_status.ptr,
                                          d_energy.ptr, d_gradient.ptr, d_dedq.ptr);
      checked(launch_d4_fixed_charge_batched_cuda(batch, parameters, tables, d_workspace.ptr,
                                                  d_workspace.count, result));
    }
    checked(cudaDeviceSynchronize());

    checked(cudaEventRecord(start));
    for (int repeat = 0; repeat < repeats; ++repeat)
      scalar_batch_kernel<<<systems, 1>>>(systems, d_offsets.ptr, d_z.ptr, d_xyz.ptr, d_q.ptr,
                                          parameters, tables, d_workspace.ptr, d_status.ptr,
                                          d_energy.ptr, d_gradient.ptr, d_dedq.ptr);
    checked(cudaEventRecord(stop));
    checked(cudaEventSynchronize(stop));
    const float scalar_ms = elapsed(start, stop) / repeats;

    checked(cudaEventRecord(start));
    for (int repeat = 0; repeat < repeats; ++repeat)
      checked(launch_d4_fixed_charge_batched_cuda(batch, parameters, tables, d_workspace.ptr,
                                                  d_workspace.count, result));
    checked(cudaEventRecord(stop));
    checked(cudaEventSynchronize(stop));
    const float cooperative_ms = elapsed(start, stop) / repeats;
    checked(cudaEventDestroy(stop));
    checked(cudaEventDestroy(start));

    std::printf(
        "systems=%d atoms/system=%d repeats=%d scalar_lane_ms=%.6f cooperative_ms=%.6f "
        "speedup=%.3fx\n",
        systems, atoms, repeats, scalar_ms, cooperative_ms, scalar_ms / cooperative_ms);
    return 0;
  } catch (const std::exception& exception) {
    std::fprintf(stderr, "D4 CUDA schedule probe: %s\n", exception.what());
    return 1;
  }
}
