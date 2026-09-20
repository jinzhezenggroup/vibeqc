#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include "d4_test_cases.hpp"
#include "dft/dispersion/d4_cuda.hpp"
#include "dft/dispersion/d4_eeq.hpp"

using namespace d4_tests;
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
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  void upload(const T* data, std::size_t n) {
    if (n > count) throw std::runtime_error("oversize upload");
    if (n) checked(cudaMemcpy(ptr, data, n * sizeof(T), cudaMemcpyHostToDevice));
  }
  void download(T* data, std::size_t n) const {
    if (n > count) throw std::runtime_error("oversize download");
    if (n) checked(cudaMemcpy(data, ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
  }
};

Result cpu_evaluate(const Molecule& molecule, const D4Parameters& parameters) {
  Result result(static_cast<int>(molecule.z.size()));
  std::vector<double> workspace(d4_workspace_elements(static_cast<int>(molecule.z.size())));
  result.status = evaluate_d4_fixed_charge(
      static_cast<int>(molecule.z.size()), molecule.z.data(), molecule.xyz.data(),
      molecule.q.data(), parameters, gfn2_d4_host_tables(), workspace.data(), workspace.size(),
      result.energy.data(), result.gradient.data(), result.dq.data());
  return result;
}

struct Fixture {
  std::vector<Molecule> molecules{
      {{8, 1, 1}, {0, 0, 0, 1.43, 0, 1.1, -1.43, 0, 1.1}, {-0.5, 0.25, 0.25}},
      {{6}, {0, 0, 0}, {0}},
      {},
      fixture(),
  };
  std::vector<std::uint32_t> offsets{0};
  std::vector<std::int32_t> z;
  std::vector<double> xyz;
  std::vector<double> q;
  std::vector<Result> expected;

  Fixture() {
    const auto parameters = gfn2_d4_parameters();
    for (const auto& molecule : molecules) {
      z.insert(z.end(), molecule.z.begin(), molecule.z.end());
      xyz.insert(xyz.end(), molecule.xyz.begin(), molecule.xyz.end());
      q.insert(q.end(), molecule.q.begin(), molecule.q.end());
      offsets.push_back(static_cast<std::uint32_t>(z.size()));
      expected.push_back(cpu_evaluate(molecule, parameters));
    }
  }
};

struct DeviceFixture {
  Fixture host;
  Buffer<std::uint32_t> offsets{host.offsets.size()};
  Buffer<std::int32_t> z{host.z.size()};
  Buffer<double> xyz{host.xyz.size()};
  Buffer<double> q{host.q.size()};
  Buffer<std::uint8_t> active{host.molecules.size()};
  Buffer<D4Status> statuses{host.molecules.size()};
  Buffer<double> energies{2 * host.molecules.size()};
  Buffer<double> gradients{3 * host.z.size()};
  Buffer<double> dedq{host.z.size()};
  Buffer<double> workspace{d4_cuda_workspace_elements(host.z.size())};
  Buffer<data::D4ElementData> elements{data::kElementCount};
  Buffer<data::D4ReferenceData> references{data::kReferenceCount};
  Buffer<double> c6{data::kReferenceC6.size()};

  DeviceFixture() {
    offsets.upload(host.offsets.data(), host.offsets.size());
    z.upload(host.z.data(), host.z.size());
    xyz.upload(host.xyz.data(), host.xyz.size());
    q.upload(host.q.data(), host.q.size());
    elements.upload(data::kElements.data(), elements.count);
    references.upload(data::kReferences.data(), references.count);
    c6.upload(data::kReferenceC6.data(), c6.count);
  }

  D4Tables tables() const {
    return {D4ReferenceModel::gfn2,
            elements.ptr,
            references.ptr,
            c6.ptr,
            data::kElementCount,
            data::kReferenceCount,
            data::kReferenceC6.size(),
            3.0,
            2.0};
  }

  int run(bool poison_last, bool deactivate_single) {
    auto charges = host.q;
    if (poison_last) charges.back() = std::numeric_limits<double>::quiet_NaN();
    q.upload(charges.data(), charges.size());
    std::vector<std::uint8_t> mask(host.molecules.size(), 1);
    if (deactivate_single) mask[1] = 0;
    active.upload(mask.data(), mask.size());

    const D4CudaBatch batch{static_cast<std::uint32_t>(host.molecules.size()),
                            static_cast<std::uint32_t>(host.z.size()),
                            offsets.ptr,
                            z.ptr,
                            xyz.ptr,
                            q.ptr,
                            active.ptr};
    const D4CudaResult result{statuses.ptr, energies.ptr, gradients.ptr, dedq.ptr};
    checked(launch_d4_fixed_charge_batched_cuda(batch, gfn2_d4_parameters(), tables(),
                                                workspace.ptr, workspace.count, result));
    checked(cudaDeviceSynchronize());

    std::vector<D4Status> status(host.molecules.size());
    std::vector<double> energy(2 * host.molecules.size());
    std::vector<double> gradient(3 * host.z.size());
    std::vector<double> charge_derivative(host.z.size());
    statuses.download(status.data(), status.size());
    energies.download(energy.data(), energy.size());
    gradients.download(gradient.data(), gradient.size());
    dedq.download(charge_derivative.data(), charge_derivative.size());

    std::size_t atom_offset = 0;
    for (std::size_t system = 0; system < host.molecules.size(); ++system) {
      const auto atoms = host.molecules[system].z.size();
      const bool failed = poison_last && system + 1 == host.molecules.size();
      const bool inactive = deactivate_single && system == 1;
      if (failed) {
        if (status[system] != D4Status::invalid_argument) return 10;
      } else if (status[system] != D4Status::success) {
        return 11;
      }
      if (failed || inactive) {
        if (energy[2 * system] != 0.0 || energy[2 * system + 1] != 0.0) return 12;
        for (std::size_t atom = 0; atom < atoms; ++atom) {
          if (charge_derivative[atom_offset + atom] != 0.0) return 13;
          for (int axis = 0; axis < 3; ++axis)
            if (gradient[3 * (atom_offset + atom) + axis] != 0.0) return 14;
        }
      } else {
        const auto& expected = host.expected[system];
        for (int term = 0; term < 2; ++term)
          if (!near(energy[2 * system + term], expected.energy[term], 3e-13)) return 20;
        for (std::size_t atom = 0; atom < atoms; ++atom) {
          if (!near(charge_derivative[atom_offset + atom], expected.dq[atom], 5e-11)) return 21;
          for (int axis = 0; axis < 3; ++axis)
            if (!near(gradient[3 * (atom_offset + atom) + axis], expected.gradient[3 * atom + axis],
                      5e-11))
              return 22;
        }
      }
      atom_offset += atoms;
    }
    return 0;
  }
};

int run_invalid_partitions_and_recovery() {
  DeviceFixture fixture;
  const auto original = fixture.host.offsets;
  std::vector<std::vector<std::uint32_t>> cases(4, original);
  cases[0][2] = 1;  // Backward interval followed by an overlapping valid interval.
  cases[1][0] = 1;
  --cases[2].back();
  cases[3][2] = static_cast<std::uint32_t>(fixture.host.z.size() + 1);
  for (const auto& offsets : cases) {
    fixture.offsets.upload(offsets.data(), offsets.size());
    const D4CudaBatch batch{static_cast<std::uint32_t>(fixture.host.molecules.size()),
                            static_cast<std::uint32_t>(fixture.host.z.size()),
                            fixture.offsets.ptr,
                            fixture.z.ptr,
                            fixture.xyz.ptr,
                            fixture.q.ptr,
                            nullptr};
    const D4CudaResult result{fixture.statuses.ptr, fixture.energies.ptr, fixture.gradients.ptr,
                              fixture.dedq.ptr};
    checked(launch_d4_fixed_charge_batched_cuda(batch, gfn2_d4_parameters(), fixture.tables(),
                                                fixture.workspace.ptr, fixture.workspace.count,
                                                result));
    checked(cudaDeviceSynchronize());
    std::vector<D4Status> status(batch.systems);
    fixture.statuses.download(status.data(), status.size());
    for (auto value : status)
      if (value != D4Status::invalid_argument) return 40;
    for (const auto* buffer : {&fixture.energies, &fixture.gradients, &fixture.dedq}) {
      std::vector<double> values(buffer->count);
      buffer->download(values.data(), values.size());
      for (double value : values)
        if (value != 0.0) return 41;
    }
  }
  fixture.offsets.upload(original.data(), original.size());
  return fixture.run(false, false);
}

int run_all_empty_batch() {
  const std::array<std::uint32_t, 3> offsets{0u, 0u, 0u};
  Buffer<std::uint32_t> d_offsets(offsets.size());
  Buffer<D4Status> d_status(2);
  Buffer<double> d_energy(4);
  Buffer<data::D4ElementData> d_elements(data::kElementCount);
  Buffer<data::D4ReferenceData> d_references(data::kReferenceCount);
  Buffer<double> d_c6(data::kReferenceC6.size());
  d_offsets.upload(offsets.data(), offsets.size());
  d_elements.upload(data::kElements.data(), d_elements.count);
  d_references.upload(data::kReferences.data(), d_references.count);
  d_c6.upload(data::kReferenceC6.data(), d_c6.count);

  const D4Tables tables{D4ReferenceModel::gfn2,
                        d_elements.ptr,
                        d_references.ptr,
                        d_c6.ptr,
                        data::kElementCount,
                        data::kReferenceCount,
                        data::kReferenceC6.size(),
                        3.0,
                        2.0};
  const D4CudaBatch batch{2u, 0u, d_offsets.ptr, nullptr, nullptr, nullptr, nullptr};
  const D4CudaResult output{d_status.ptr, d_energy.ptr, nullptr, nullptr};
  checked(
      launch_d4_fixed_charge_batched_cuda(batch, gfn2_d4_parameters(), tables, nullptr, 0, output));
  checked(cudaDeviceSynchronize());

  std::array<D4Status, 2> status{};
  std::array<double, 4> energy{};
  d_status.download(status.data(), status.size());
  d_energy.download(energy.data(), energy.size());
  for (const auto value : status)
    if (value != D4Status::success) return 35;
  for (const double value : energy)
    if (value != 0.0) return 36;
  return 0;
}

int run_r2scan3c_fixed_charge_profile() {
  const auto molecule = fixture();
  const int atoms = static_cast<int>(molecule.z.size());
  const auto parameters = r2scan3c_d4_parameters();
  Result expected(atoms);
  std::vector<double> host_workspace(d4_workspace_elements(atoms));
  expected.status = evaluate_d4_fixed_charge(
      atoms, molecule.z.data(), molecule.xyz.data(), molecule.q.data(), parameters,
      eeq_d4_host_tables(D4EEQProfile::r2scan3c), host_workspace.data(), host_workspace.size(),
      expected.energy.data(), expected.gradient.data(), expected.dq.data());
  if (expected.status != D4Status::success) return 30;

  const std::array<std::uint32_t, 2> offsets{0u, static_cast<std::uint32_t>(atoms)};
  Buffer<std::uint32_t> d_offsets(offsets.size());
  Buffer<std::int32_t> d_z(atoms);
  Buffer<double> d_xyz(3 * atoms);
  Buffer<double> d_q(atoms);
  Buffer<D4Status> d_status(1);
  Buffer<double> d_energy(2);
  Buffer<double> d_gradient(3 * atoms);
  Buffer<double> d_dedq(atoms);
  Buffer<double> d_workspace(d4_cuda_workspace_elements(atoms));
  Buffer<data::D4ElementData> d_elements(eeq_data::kElementCount);
  Buffer<data::D4ReferenceData> d_references(eeq_data::kReferenceCount);
  Buffer<double> d_c6(eeq_data::kReferenceC6R2SCAN3C.size());
  d_offsets.upload(offsets.data(), offsets.size());
  d_z.upload(molecule.z.data(), atoms);
  d_xyz.upload(molecule.xyz.data(), 3 * atoms);
  d_q.upload(molecule.q.data(), atoms);
  d_elements.upload(eeq_data::kElements.data(), d_elements.count);
  d_references.upload(eeq_data::kReferences.data(), d_references.count);
  d_c6.upload(eeq_data::kReferenceC6R2SCAN3C.data(), d_c6.count);

  const D4Tables tables{D4ReferenceModel::eeq,
                        d_elements.ptr,
                        d_references.ptr,
                        d_c6.ptr,
                        eeq_data::kElementCount,
                        eeq_data::kReferenceCount,
                        eeq_data::kReferenceC6R2SCAN3C.size(),
                        parameters.ga,
                        parameters.gc};
  const D4CudaBatch batch{
      1u, static_cast<std::uint32_t>(atoms), d_offsets.ptr, d_z.ptr, d_xyz.ptr, d_q.ptr, nullptr};
  const D4CudaResult output{d_status.ptr, d_energy.ptr, d_gradient.ptr, d_dedq.ptr};
  checked(launch_d4_fixed_charge_batched_cuda(batch, parameters, tables, d_workspace.ptr,
                                              d_workspace.count, output));
  checked(cudaDeviceSynchronize());

  D4Status status{};
  std::array<double, 2> energy{};
  std::vector<double> gradient(3 * atoms);
  std::vector<double> dedq(atoms);
  d_status.download(&status, 1);
  d_energy.download(energy.data(), energy.size());
  d_gradient.download(gradient.data(), gradient.size());
  d_dedq.download(dedq.data(), dedq.size());
  if (status != D4Status::success) return 31;
  for (int term = 0; term < 2; ++term)
    if (!near(energy[term], expected.energy[term], 3e-13)) return 32;
  for (int coordinate = 0; coordinate < 3 * atoms; ++coordinate)
    if (!near(gradient[coordinate], expected.gradient[coordinate], 5e-11)) return 33;
  for (int atom = 0; atom < atoms; ++atom)
    if (!near(dedq[atom], expected.dq[atom], 5e-11)) return 34;
  return 0;
}

}  // namespace

int main() {
  int devices = 0;
  const auto error = cudaGetDeviceCount(&devices);
  if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
      (error == cudaSuccess && devices == 0))
    return 77;
  try {
    checked(error);
    DeviceFixture fixture;
    if (const int rc = fixture.run(false, false)) return rc;
    if (const int rc = fixture.run(true, true)) return rc;
    if (const int rc = run_invalid_partitions_and_recovery()) return rc;
    if (const int rc = run_all_empty_batch()) return rc;
    if (const int rc = run_r2scan3c_fixed_charge_profile()) return rc;
    std::puts(
        "block-cooperative D4 batch matches GFN2/r2SCAN-3c scalar oracles and isolates ragged "
        "failures");
    return 0;
  } catch (const std::exception& exception) {
    std::fprintf(stderr, "CUDA D4 schedule test: %s\n", exception.what());
    return 1;
  }
}
