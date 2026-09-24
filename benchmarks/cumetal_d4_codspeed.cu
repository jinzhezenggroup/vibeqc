#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "dft/dispersion/d4_cuda.hpp"

// Compile the exact production cooperative D4 CUDA schedule into this isolated
// benchmark executable. The benchmark has no duplicate scientific kernel body.
#include "dft/dispersion/d4_cuda.cu"

using namespace vibeqc::dft::dispersion;

namespace {

constexpr std::uint32_t kSystems = 4;
constexpr std::uint32_t kAtomsPerSystem = 16;
constexpr std::uint32_t kTotalAtoms = kSystems * kAtomsPerSystem;
constexpr unsigned int kWarmupLaunches = 2;
constexpr unsigned int kLaunchesPerSample = 8;

void check(cudaError_t status, const char* expression) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(status));
}

template <class T>
class Buffer {
 public:
  explicit Buffer(std::size_t count) : count_(count) {
    if (count_)
      check(cudaMalloc(reinterpret_cast<void**>(&pointer_), count_ * sizeof(T)), "cudaMalloc");
  }
  ~Buffer() {
    if (pointer_) cudaFree(pointer_);
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;

  T* get() const { return pointer_; }
  std::size_t size() const { return count_; }

  void upload(const T* source, std::size_t count) {
    check(cudaMemcpy(pointer_, source, count * sizeof(T), cudaMemcpyHostToDevice),
          "cudaMemcpy H2D");
  }

  void download(T* destination, std::size_t count) const {
    check(cudaMemcpy(destination, pointer_, count * sizeof(T), cudaMemcpyDeviceToHost),
          "cudaMemcpy D2H");
  }

 private:
  T* pointer_{};
  std::size_t count_{};
};

bool near(double actual, double expected, double tolerance) {
  return std::isfinite(actual) && std::isfinite(expected) &&
         std::abs(actual - expected) <=
             tolerance * std::max({1.0, std::abs(actual), std::abs(expected)});
}

class BenchmarkServer {
 public:
  BenchmarkServer()
      : offsets_(kSystems + 1),
        atomic_numbers_(kTotalAtoms),
        coordinates_(3u * kTotalAtoms),
        charges_(kTotalAtoms),
        active_(kSystems, 1u),
        expected_energy_(2u * kSystems),
        expected_gradient_(3u * kTotalAtoms),
        expected_dedq_(kTotalAtoms),
        device_offsets_(offsets_.size()),
        device_atomic_numbers_(atomic_numbers_.size()),
        device_coordinates_(coordinates_.size()),
        device_charges_(charges_.size()),
        device_active_(active_.size()),
        device_statuses_(kSystems),
        device_energy_(expected_energy_.size()),
        device_gradient_(expected_gradient_.size()),
        device_dedq_(expected_dedq_.size()),
        device_workspace_(d4_cuda_workspace_elements(kTotalAtoms)),
        device_elements_(data::kElementCount),
        device_references_(data::kReferenceCount),
        device_reference_c6_(data::kReferenceC6.size()) {}

  bool initialize() {
    try {
      check(cudaSetDevice(0), "cudaSetDevice");
      cudaDeviceProp properties{};
      check(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
      prepare_inputs();
      compute_host_reference();
      upload_inputs();
      validate_production_schedule();

      check(cudaEventCreate(&begin_), "cudaEventCreate(begin)");
      check(cudaEventCreate(&end_), "cudaEventCreate(end)");
      for (unsigned int i = 0; i < kWarmupLaunches; ++i) launch_once();
      check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(warmup)");

      std::cout << "READY device=" << properties.name << " case=d4-production systems=" << kSystems
                << " atoms_per_system=" << kAtomsPerSystem << std::endl;
      return true;
    } catch (const std::exception& error) {
      std::fprintf(stderr, "FAIL: %s\n", error.what());
      return false;
    }
  }

  bool run_sample(float* milliseconds) {
    try {
      check(cudaEventRecord(begin_), "cudaEventRecord(begin)");
      for (unsigned int i = 0; i < kLaunchesPerSample; ++i) launch_once();
      check(cudaEventRecord(end_), "cudaEventRecord(end)");
      check(cudaEventSynchronize(end_), "cudaEventSynchronize(end)");
      float total = 0.0F;
      check(cudaEventElapsedTime(&total, begin_, end_), "cudaEventElapsedTime");
      *milliseconds = total / static_cast<float>(kLaunchesPerSample);
      return std::isfinite(*milliseconds) && *milliseconds > 0.0F;
    } catch (const std::exception& error) {
      std::fprintf(stderr, "FAIL: %s\n", error.what());
      return false;
    }
  }

  ~BenchmarkServer() {
    if (begin_) cudaEventDestroy(begin_);
    if (end_) cudaEventDestroy(end_);
  }

 private:
  void prepare_inputs() {
    constexpr std::int32_t elements[] = {1, 6, 8, 1};
    for (std::uint32_t system = 0; system < kSystems; ++system) {
      offsets_[system] = system * kAtomsPerSystem;
      for (std::uint32_t atom = 0; atom < kAtomsPerSystem; ++atom) {
        const std::size_t packed = static_cast<std::size_t>(system) * kAtomsPerSystem + atom;
        atomic_numbers_[packed] = elements[atom % 4u];
        const double shift = 0.025 * static_cast<double>(system);
        coordinates_[3u * packed] = 2.35 * static_cast<double>(atom % 4u) + shift;
        coordinates_[3u * packed + 1u] = 2.30 * static_cast<double>((atom / 4u) % 4u) - shift;
        coordinates_[3u * packed + 2u] = 0.17 * static_cast<double>((atom + system) % 3u);
        charges_[packed] = 0.015 * static_cast<double>(static_cast<int>(atom % 5u) - 2);
      }
    }
    offsets_[kSystems] = kTotalAtoms;
  }

  void compute_host_reference() {
    const auto parameters = gfn2_d4_parameters();
    const auto tables = gfn2_d4_host_tables();
    for (std::uint32_t system = 0; system < kSystems; ++system) {
      const std::size_t begin = offsets_[system];
      std::vector<double> workspace(d4_workspace_elements(kAtomsPerSystem));
      double energy[2]{};
      std::vector<double> gradient(3u * kAtomsPerSystem);
      std::vector<double> dedq(kAtomsPerSystem);
      const auto status = evaluate_d4_fixed_charge(
          kAtomsPerSystem, atomic_numbers_.data() + begin, coordinates_.data() + 3u * begin,
          charges_.data() + begin, parameters, tables, workspace.data(), workspace.size(), energy,
          gradient.data(), dedq.data());
      if (status != D4Status::success) throw std::runtime_error("host D4 reference failed");
      expected_energy_[2u * system] = energy[0];
      expected_energy_[2u * system + 1u] = energy[1];
      std::copy(gradient.begin(), gradient.end(), expected_gradient_.begin() + 3u * begin);
      std::copy(dedq.begin(), dedq.end(), expected_dedq_.begin() + begin);
    }
  }

  void upload_inputs() {
    device_offsets_.upload(offsets_.data(), offsets_.size());
    device_atomic_numbers_.upload(atomic_numbers_.data(), atomic_numbers_.size());
    device_coordinates_.upload(coordinates_.data(), coordinates_.size());
    device_charges_.upload(charges_.data(), charges_.size());
    device_active_.upload(active_.data(), active_.size());
    device_elements_.upload(data::kElements.data(), data::kElementCount);
    device_references_.upload(data::kReferences.data(), data::kReferenceCount);
    device_reference_c6_.upload(data::kReferenceC6.data(), data::kReferenceC6.size());
  }

  D4Tables device_tables() const {
    return {D4ReferenceModel::gfn2,
            device_elements_.get(),
            device_references_.get(),
            device_reference_c6_.get(),
            data::kElementCount,
            data::kReferenceCount,
            data::kReferenceC6.size(),
            3.0,
            2.0};
  }

  D4CudaBatch batch() const {
    return {kSystems,
            kTotalAtoms,
            device_offsets_.get(),
            device_atomic_numbers_.get(),
            device_coordinates_.get(),
            device_charges_.get(),
            device_active_.get()};
  }

  D4CudaResult result() const {
    return {device_statuses_.get(), device_energy_.get(), device_gradient_.get(),
            device_dedq_.get()};
  }

  void launch_once() {
    check(launch_d4_fixed_charge_batched_cuda(batch(), gfn2_d4_parameters(), device_tables(),
                                              device_workspace_.get(), device_workspace_.size(),
                                              result()),
          "launch_d4_fixed_charge_batched_cuda");
  }

  void validate_production_schedule() {
    launch_once();
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize(validation)");

    std::vector<D4Status> statuses(kSystems);
    std::vector<double> energy(expected_energy_.size());
    std::vector<double> gradient(expected_gradient_.size());
    std::vector<double> dedq(expected_dedq_.size());
    device_statuses_.download(statuses.data(), statuses.size());
    device_energy_.download(energy.data(), energy.size());
    device_gradient_.download(gradient.data(), gradient.size());
    device_dedq_.download(dedq.data(), dedq.size());

    for (const auto status : statuses)
      if (status != D4Status::success) throw std::runtime_error("production D4 status failed");
    for (std::size_t i = 0; i < energy.size(); ++i)
      if (!near(energy[i], expected_energy_[i], 2e-10))
        throw std::runtime_error("production D4 energy differs from host reference");
    for (std::size_t i = 0; i < gradient.size(); ++i)
      if (!near(gradient[i], expected_gradient_[i], 2e-9))
        throw std::runtime_error("production D4 gradient differs from host reference");
    for (std::size_t i = 0; i < dedq.size(); ++i)
      if (!near(dedq[i], expected_dedq_[i], 2e-9))
        throw std::runtime_error("production D4 dE/dq differs from host reference");
  }

  std::vector<std::uint32_t> offsets_;
  std::vector<std::int32_t> atomic_numbers_;
  std::vector<double> coordinates_;
  std::vector<double> charges_;
  std::vector<std::uint8_t> active_;
  std::vector<double> expected_energy_;
  std::vector<double> expected_gradient_;
  std::vector<double> expected_dedq_;

  Buffer<std::uint32_t> device_offsets_;
  Buffer<std::int32_t> device_atomic_numbers_;
  Buffer<double> device_coordinates_;
  Buffer<double> device_charges_;
  Buffer<std::uint8_t> device_active_;
  Buffer<D4Status> device_statuses_;
  Buffer<double> device_energy_;
  Buffer<double> device_gradient_;
  Buffer<double> device_dedq_;
  Buffer<double> device_workspace_;
  Buffer<data::D4ElementData> device_elements_;
  Buffer<data::D4ReferenceData> device_references_;
  Buffer<double> device_reference_c6_;

  cudaEvent_t begin_{};
  cudaEvent_t end_{};
};

}  // namespace

int main() {
  BenchmarkServer server;
  if (!server.initialize()) return 1;

  std::string command;
  while (std::getline(std::cin, command)) {
    if (command == "run d4") {
      float milliseconds = 0.0F;
      if (!server.run_sample(&milliseconds)) return 2;
      std::cout << "OK d4 " << milliseconds << std::endl;
      continue;
    }
    if (command == "quit") return 0;
    std::fprintf(stderr, "FAIL: unknown command: %s\n", command.c_str());
    return 3;
  }
  return 0;
}
