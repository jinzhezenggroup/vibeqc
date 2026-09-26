#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "d4_test_cases.hpp"
using namespace d4_tests;
namespace {
void checked(cudaError_t err) {
  if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}
template <class T>
struct Buffer {
  T* ptr = nullptr;
  std::size_t count;
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
__global__ void evaluate_kernel(int n, const std::int32_t* z, const double* xyz, const double* q,
                                D4Parameters p, D4Tables t, double* w, double* energy, double* grad,
                                double* dq, D4Status* status) {
  *status =
      evaluate_d4_fixed_charge(n, z, xyz, q, p, t, w, d4_workspace_elements(n), energy, grad, dq);
}
// Four packed, unequal members including an empty member. Independent slices
// run concurrently; an invalid charge in the last member must not poison peers.
__global__ void ragged_kernel(const std::int32_t* z, const double* xyz, const double* q,
                              D4Parameters p, D4Tables t, double* w, double* energy, double* grad,
                              double* dq, D4Status* status) {
  constexpr int offsets[] = {0, 3, 4, 4, 8};
  const int row = blockIdx.x;
  const int b = offsets[row], n = offsets[row + 1] - b;
  status[row] = evaluate_d4_fixed_charge(n, z + b, xyz + 3 * b, q + b, p, t, w + row * 108, 108,
                                         energy + 2 * row, grad + 3 * b, dq + b);
}
struct GpuEval {
  Buffer<data::D4ElementData> elements{data::kElementCount};
  Buffer<data::D4ReferenceData> references{data::kReferenceCount};
  Buffer<double> c6{data::kReferenceC6.size()};
  Buffer<std::int32_t> z{kD4MaximumAtoms};
  Buffer<double> xyz{3 * kD4MaximumAtoms}, q{kD4MaximumAtoms};
  Buffer<double> w{d4_workspace_elements(kD4MaximumAtoms)};
  Buffer<double> energy{8}, grad{3 * kD4MaximumAtoms}, dq{kD4MaximumAtoms};
  Buffer<D4Status> status{4};
  GpuEval() {
    elements.upload(data::kElements.data(), elements.count);
    references.upload(data::kReferences.data(), references.count);
    c6.upload(data::kReferenceC6.data(), c6.count);
  }
  D4Tables tables() {
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
  Result operator()(const Molecule& m, const D4Parameters& p) {
    const int n = static_cast<int>(m.z.size());
    Result out(n);
    z.upload(m.z.data(), m.z.size());
    xyz.upload(m.xyz.data(), m.xyz.size());
    q.upload(m.q.data(), m.q.size());
    energy.upload(out.energy.data(), 2);
    grad.upload(out.gradient.data(), out.gradient.size());
    dq.upload(out.dq.data(), out.dq.size());
    evaluate_kernel<<<1, 1>>>(n, z.ptr, xyz.ptr, q.ptr, p, tables(), w.ptr, energy.ptr, grad.ptr,
                              dq.ptr, status.ptr);
    checked(cudaGetLastError());
    checked(cudaDeviceSynchronize());
    status.download(&out.status, 1);
    energy.download(out.energy.data(), 2);
    grad.download(out.gradient.data(), out.gradient.size());
    dq.download(out.dq.data(), out.dq.size());
    return out;
  }
  int ragged_test() {
    const Molecule mol[] = {{{8, 1, 1}, {0, 0, 0, 1.43, 0, 1.1, -1.43, 0, 1.1}, {-0.5, 0.25, 0.25}},
                            {{6}, {0, 0, 0}, {0}},
                            Molecule{},
                            fixture()};
    Molecule packed;
    std::vector<Result> expected;
    auto p = gfn2_d4_parameters();
    for (const auto& m : mol) {
      packed.z.insert(packed.z.end(), m.z.begin(), m.z.end());
      packed.xyz.insert(packed.xyz.end(), m.xyz.begin(), m.xyz.end());
      packed.q.insert(packed.q.end(), m.q.begin(), m.q.end());
      expected.push_back((*this)(m, p));
    }
    z.upload(packed.z.data(), packed.z.size());
    xyz.upload(packed.xyz.data(), packed.xyz.size());
    for (bool poison : {false, true}) {
      auto charges = packed.q;
      if (poison) charges.back() = std::numeric_limits<double>::quiet_NaN();
      q.upload(charges.data(), charges.size());
      std::array<double, 8> en{17, 19, 17, 19, 17, 19, 17, 19};
      std::vector<double> g(24, 23), d(8, 29);
      std::array<D4Status, 4> st;
      energy.upload(en.data(), 8);
      grad.upload(g.data(), 24);
      dq.upload(d.data(), 8);
      ragged_kernel<<<4, 1>>>(z.ptr, xyz.ptr, q.ptr, p, tables(), w.ptr, energy.ptr, grad.ptr,
                              dq.ptr, status.ptr);
      checked(cudaGetLastError());
      checked(cudaDeviceSynchronize());
      energy.download(en.data(), 8);
      grad.download(g.data(), 24);
      dq.download(d.data(), 8);
      status.download(st.data(), 4);
      int offset = 0;
      for (int row = 0; row < 4; ++row) {
        const int n = static_cast<int>(mol[row].z.size());
        if (poison && row == 3) {
          if (st[row] != D4Status::invalid_argument || en[6] != 17 || en[7] != 19) return 1;
          for (int k = 0; k < 3 * n; ++k)
            if (g[3 * offset + k] != 23) return 2;
          for (int k = 0; k < n; ++k)
            if (d[offset + k] != 29) return 3;
        } else {
          if (st[row] != D4Status::success) return 4;
          for (int k = 0; k < 2; ++k)
            if (!near(en[2 * row + k], expected[row].energy[k], 1e-15)) return 5;
          for (int k = 0; k < 3 * n; ++k)
            if (!near(g[3 * offset + k], expected[row].gradient[k], 1e-14)) return 6;
          for (int k = 0; k < n; ++k)
            if (!near(d[offset + k], expected[row].dq[k], 1e-14)) return 7;
        }
        offset += n;
      }
    }
    std::puts("CUDA ragged-batch/empty-member/peer-isolation tests passed");
    return 0;
  }
};
}  // namespace
int main() {
  int devices = 0;
  const auto err = cudaGetDeviceCount(&devices);
  if (err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver ||
      (err == cudaSuccess && devices == 0))
    return 77;
  try {
    checked(err);
    GpuEval evaluate;
    if (run_cases(evaluate)) return 1;
    return evaluate.ragged_test();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "CUDA test: %s\n", e.what());
    return 1;
  }
}
