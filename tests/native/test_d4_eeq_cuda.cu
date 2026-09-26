#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <vector>

#include "d4_eeq_oracle_fixtures.hpp"

using namespace d4_eeq_tests;

namespace {
bool near(double a, double b, double tolerance) { return std::fabs(a - b) <= tolerance; }

void checked(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template <class T>
struct Buffer {
  T* ptr = nullptr;
  std::size_t count = 0;
  explicit Buffer(std::size_t n) : count(n) {
    checked(cudaMalloc(reinterpret_cast<void**>(&ptr), n * sizeof(T)));
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
  void download(T* data, std::size_t n) {
    if (n > count) throw std::runtime_error("oversize download");
    if (n) checked(cudaMemcpy(data, ptr, n * sizeof(T), cudaMemcpyDeviceToHost));
  }
};

__global__ void complete_kernel(int n, const std::int32_t* z, const double* xyz, double charge,
                                D4Parameters p, D4EEQProfile profile, D4Tables d4_tables,
                                EEQTables eeq_tables, double* workspace, double* energy,
                                double* gradient, double* charges, D4Status* status) {
  *status = evaluate_complete_d4_eeq_with_tables(
      n, z, xyz, charge, p, profile, d4_tables, eeq_tables, workspace,
      complete_d4_eeq_workspace_elements(n), energy, gradient, charges);
}

__global__ void ragged_kernel(const std::int32_t* z, const double* xyz, const double* total_charge,
                              const D4Parameters* parameters, const D4EEQProfile* profiles,
                              D4Tables standard_tables, D4Tables r2scan_tables,
                              EEQTables eeq_tables, double* workspace, std::size_t workspace_stride,
                              double* energy, double* gradient, double* charges, D4Status* status) {
  constexpr int offsets[] = {0, 3, 7, 12};
  const int row = blockIdx.x;
  const int begin = offsets[row];
  const int n = offsets[row + 1] - begin;
  const auto profile = profiles[row];
  const auto tables = profile == D4EEQProfile::r2scan3c ? r2scan_tables : standard_tables;
  status[row] = evaluate_complete_d4_eeq_with_tables(
      n, z + begin, xyz + 3 * begin, total_charge[row], parameters[row], profile, tables,
      eeq_tables, workspace + row * workspace_stride, workspace_stride, energy + 2 * row,
      gradient + 3 * begin, charges + begin);
}

struct GpuEval {
  static constexpr std::size_t c6_count = eeq_data::kReferenceC6Standard.size();
  Buffer<data::D4ElementData> elements{eeq_data::kElementCount};
  Buffer<data::D4ReferenceData> references{eeq_data::kReferenceCount};
  Buffer<eeq_data::EEQChargeElementData> charge_elements{eeq_data::kElementCount};
  Buffer<double> c6_standard{c6_count};
  Buffer<double> c6_r2scan{c6_count};
  Buffer<std::int32_t> z{kMaxAtoms};
  Buffer<double> xyz{3 * kMaxAtoms};
  Buffer<double> workspace{complete_d4_eeq_workspace_elements(kMaxAtoms)};
  Buffer<double> energy{6};
  Buffer<double> gradient{3 * 12};
  Buffer<double> charges{12};
  Buffer<D4Status> status{3};

  GpuEval() {
    elements.upload(eeq_data::kElements.data(), elements.count);
    references.upload(eeq_data::kReferences.data(), references.count);
    charge_elements.upload(eeq_data::kChargeElements.data(), charge_elements.count);
    c6_standard.upload(eeq_data::kReferenceC6Standard.data(), c6_standard.count);
    c6_r2scan.upload(eeq_data::kReferenceC6R2SCAN3C.data(), c6_r2scan.count);
  }

  EEQTables eeq_tables() const {
    return {elements.ptr, charge_elements.ptr, eeq_data::kElementCount};
  }
  D4Tables d4_tables(D4EEQProfile profile) const {
    return {D4ReferenceModel::eeq,
            elements.ptr,
            references.ptr,
            profile == D4EEQProfile::r2scan3c ? c6_r2scan.ptr : c6_standard.ptr,
            eeq_data::kElementCount,
            eeq_data::kReferenceCount,
            c6_count,
            profile == D4EEQProfile::r2scan3c ? 2.0 : 3.0,
            profile == D4EEQProfile::r2scan3c ? 1.0 : 2.0};
  }

  int individual_cases() {
    for (const auto& f : kEEQOracleFixtures) {
      z.upload(f.z.data(), f.atoms);
      xyz.upload(f.xyz.data(), 3 * f.atoms);
      std::array<double, 2> e{17.0, 19.0};
      std::vector<double> g(3 * f.atoms, 23.0), q(f.atoms, 29.0);
      energy.upload(e.data(), 2);
      gradient.upload(g.data(), g.size());
      charges.upload(q.data(), q.size());
      complete_kernel<<<1, 1>>>(f.atoms, z.ptr, xyz.ptr, f.total_charge, f.parameters, f.profile,
                                d4_tables(f.profile), eeq_tables(), workspace.ptr, energy.ptr,
                                gradient.ptr, charges.ptr, status.ptr);
      checked(cudaGetLastError());
      checked(cudaDeviceSynchronize());
      D4Status st{};
      status.download(&st, 1);
      energy.download(e.data(), 2);
      gradient.download(g.data(), g.size());
      charges.download(q.data(), q.size());
      if (st != D4Status::success || !near(e[0] + e[1], f.energy, 2e-13)) return 1;
      for (int i = 0; i < f.atoms; ++i)
        if (!near(q[i], f.charges[i], 1e-10)) return 2;
      for (int i = 0; i < 3 * f.atoms; ++i)
        if (!near(g[i], f.gradient[i], 2e-12)) return 3;
    }
    return 0;
  }

  int ragged_cases() {
    constexpr int rows = 3;
    constexpr int total_atoms = 12;
    const std::size_t stride = complete_d4_eeq_workspace_elements(kMaxAtoms);
    Buffer<std::int32_t> packed_z{total_atoms};
    Buffer<double> packed_xyz{3 * total_atoms};
    Buffer<double> packed_charge{rows};
    Buffer<D4Parameters> packed_parameters{rows};
    Buffer<D4EEQProfile> packed_profiles{rows};
    Buffer<double> packed_workspace{rows * stride};

    std::array<std::int32_t, total_atoms> hz{};
    std::array<double, 3 * total_atoms> hx{};
    std::array<double, rows> hc{};
    std::array<D4Parameters, rows> hp{};
    std::array<D4EEQProfile, rows> hprof{};
    int offset = 0;
    for (int row = 0; row < rows; ++row) {
      const auto& f = kEEQOracleFixtures[row];
      std::copy_n(f.z.begin(), f.atoms, hz.begin() + offset);
      std::copy_n(f.xyz.begin(), 3 * f.atoms, hx.begin() + 3 * offset);
      hc[row] = f.total_charge;
      hp[row] = f.parameters;
      hprof[row] = f.profile;
      offset += f.atoms;
    }
    packed_z.upload(hz.data(), hz.size());
    packed_charge.upload(hc.data(), hc.size());
    packed_parameters.upload(hp.data(), hp.size());
    packed_profiles.upload(hprof.data(), hprof.size());

    for (bool poison : {false, true}) {
      auto coords = hx;
      if (poison) coords.back() = std::numeric_limits<double>::quiet_NaN();
      packed_xyz.upload(coords.data(), coords.size());
      std::array<double, 2 * rows> e{17, 19, 17, 19, 17, 19};
      std::array<double, 3 * total_atoms> g;
      std::array<double, total_atoms> q;
      g.fill(23.0);
      q.fill(29.0);
      energy.upload(e.data(), e.size());
      gradient.upload(g.data(), g.size());
      charges.upload(q.data(), q.size());
      ragged_kernel<<<rows, 1>>>(
          packed_z.ptr, packed_xyz.ptr, packed_charge.ptr, packed_parameters.ptr,
          packed_profiles.ptr, d4_tables(D4EEQProfile::standard), d4_tables(D4EEQProfile::r2scan3c),
          eeq_tables(), packed_workspace.ptr, stride, energy.ptr, gradient.ptr, charges.ptr,
          status.ptr);
      checked(cudaGetLastError());
      checked(cudaDeviceSynchronize());
      std::array<D4Status, rows> st{};
      energy.download(e.data(), e.size());
      gradient.download(g.data(), g.size());
      charges.download(q.data(), q.size());
      status.download(st.data(), st.size());
      offset = 0;
      for (int row = 0; row < rows; ++row) {
        const auto& f = kEEQOracleFixtures[row];
        if (poison && row == rows - 1) {
          if (st[row] != D4Status::invalid_argument || e[2 * row] != 17 || e[2 * row + 1] != 19)
            return 4;
          for (int i = 0; i < f.atoms; ++i)
            if (q[offset + i] != 29.0) return 5;
          for (int i = 0; i < 3 * f.atoms; ++i)
            if (g[3 * offset + i] != 23.0) return 6;
        } else {
          if (st[row] != D4Status::success || !near(e[2 * row] + e[2 * row + 1], f.energy, 2e-13))
            return 7;
          for (int i = 0; i < f.atoms; ++i)
            if (!near(q[offset + i], f.charges[i], 1e-10)) return 8;
          for (int i = 0; i < 3 * f.atoms; ++i)
            if (!near(g[3 * offset + i], f.gradient[i], 2e-12)) return 9;
        }
        offset += f.atoms;
      }
    }
    std::puts("CUDA complete D4-EEQ oracle/ragged/failure-isolation tests passed");
    return 0;
  }
};
}  // namespace

int main() {
  int devices = 0;
  const auto error = cudaGetDeviceCount(&devices);
  if (error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver ||
      (error == cudaSuccess && devices == 0))
    return 77;
  try {
    checked(error);
    GpuEval evaluate;
    if (const int rc = evaluate.individual_cases()) return rc;
    return evaluate.ragged_cases();
  } catch (const std::exception& error) {
    std::fprintf(stderr, "CUDA D4 EEQ test: %s\n", error.what());
    return 1;
  }
}
