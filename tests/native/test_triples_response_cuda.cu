#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "cc/triples_response_cuda.cuh"
#include "cc/triples_response_internal.hpp"

namespace {
using namespace generativeqc::cc;
struct Fixture {
  std::size_t o, v;
  std::array<std::vector<double>, 8> inputs, expected;
};
#include "../reference_data/cc/triples_response_cuda.hpp"

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
void check(cudaError_t code) { require(code == cudaSuccess, cudaGetErrorString(code)); }

void compare(const TriplesResponseResult& result, const Fixture& f) {
  const std::array<const std::vector<double>*, 8> actual{&result.ovvv,  &result.ovoo, &result.ovov,
                                                         &result.fov,   &result.t1,   &result.t2,
                                                         &result.eps_o, &result.eps_v};
  for (std::size_t k = 0; k < actual.size(); ++k) {
    require(actual[k]->size() == f.expected[k].size(), "response shape");
    for (std::size_t i = 0; i < actual[k]->size(); ++i)
      require(std::isfinite((*actual[k])[i]) && std::abs((*actual[k])[i] - f.expected[k][i]) <=
                                                    8e-11 * (1.0 + std::abs(f.expected[k][i])),
              "full VJP oracle mismatch");
  }
}

template <class Error, class F>
void throws(F&& action) {
  bool rejected = false;
  try {
    action();
  } catch (const Error&) {
    rejected = true;
  }
  require(rejected, "expected rejection");
}

void owner_case(const Fixture& f) {
  const auto o = f.o, v = f.v, ov = o * v;
  Problem p;
  p.nocc = o;
  p.nvir = v;
  p.reference_energy = 0.0;
  p.foo.resize(o * o);
  p.fvv.resize(v * v);
  p.ovvv = f.inputs[0];
  p.ovoo = f.inputs[1];
  p.ovov = f.inputs[2];
  p.fov = f.inputs[3];
  p.ovvo.resize(ov * ov);
  p.oovv.resize(ov * ov);
  p.oooo.resize(o * o * o * o);
  p.vvvv.resize(v * v * v * v);
  p.d1.assign(ov, -1.0);
  p.d2.assign(ov * ov, -2.0);
  p.initial_t1.resize(ov);
  p.initial_t2.resize(ov * ov);
  SolverResult cc;
  cc.status = SolveStatus::Converged;
  cc.t1 = f.inputs[4];
  cc.t2 = f.inputs[5];
  auto eo = f.inputs[6], ev = f.inputs[7];
  TriplesResponseOptions options;
  for (auto q : {1u, 2u, 3u, 16u}) {
    options.batch_capacity = q;
    const auto layout = detail::triples_response_layout(o, v, q, true);
    options.max_bytes = layout.numeric_bytes();
    const auto result = triples_response_cuda(p, cc, eo, ev, 0, options);
    compare(result, f);
    require(result.numeric_capacity_bytes == options.max_bytes, "exact budget accounting");
    require(result.device_capacity_bytes == layout.device_bytes, "device accounting");
    require(result.pages == (v * (v + 1) * (v + 2) / 6 + layout.q - 1) / layout.q,
            "triangular page count");
    require(result.host_to_device_bytes == layout.outputs * 8 + 8 + result.pages * layout.q * 40,
            "resident input upload count");
    require(result.device_to_host_bytes == layout.outputs * 8 + result.pages * sizeof(int),
            "one final cotangent download");
    require(std::string(result.program_hash) == generated::triples_response_program_hash,
            "shared program identity");
  }
  // Reduce capacity to fit the complete budget; reject before allocation if even
  // one lane cannot fit. This also covers a padded, non-divisible final page.
  options.batch_capacity = 16;
  options.max_bytes = detail::triples_response_layout(o, v, 2, true).numeric_bytes();
  compare(triples_response_cuda(p, cc, eo, ev, 0, options), f);
  options.max_bytes = detail::triples_response_layout(o, v, 1, true).numeric_bytes() - 1;
  throws<std::length_error>([&] { triples_response_cuda(p, cc, eo, ev, 0, options); });
  options.max_bytes = 64ULL << 20;
  throws<std::invalid_argument>([&] { triples_response_cuda(p, cc, eo, ev, -1, options); });
  cc.t1[0] = std::numeric_limits<double>::quiet_NaN();
  throws<std::invalid_argument>([&] { triples_response_cuda(p, cc, eo, ev, 0, options); });
  cc.t1 = f.inputs[4];
  ev[0] = eo.back();
  throws<std::invalid_argument>([&] { triples_response_cuda(p, cc, eo, ev, 0, options); });
}

/** Direct generated-page test exercises arbitrary maps and non-unit seeds,
 * independently of the triangular owner. Invalid maps must set the error flag
 * without performing an out-of-range read, including on inactive lanes.
 */
void page_case() {
  generated::TriplesResponseCudaState state{};
  state.o = 2;
  state.v = 3;
  state.q = 5;
  std::vector<void*> allocations;
  auto copy = [&](const void* input, std::size_t bytes) {
    void* pointer{};
    check(cudaMalloc(&pointer, bytes));
    allocations.push_back(pointer);
    if (input) check(cudaMemcpy(pointer, input, bytes, cudaMemcpyHostToDevice));
    return pointer;
  };
  try {
    std::array<const double**, 8> fields{
        &state.inputs.ovvv, &state.inputs.ovoo, &state.inputs.ovov,  &state.inputs.fov,
        &state.inputs.t1,   &state.inputs.t2,   &state.inputs.eps_o, &state.inputs.eps_v};
    for (std::size_t i = 0; i < fields.size(); ++i)
      *fields[i] = static_cast<double*>(
          copy(page_fixture.inputs[i].data(), page_fixture.inputs[i].size() * 8));
    state.inputs.a_map = static_cast<std::int64_t*>(copy(page_a_map.data(), 40));
    state.inputs.b_map = static_cast<std::int64_t*>(copy(page_b_map.data(), 40));
    state.inputs.c_map = static_cast<std::int64_t*>(copy(page_c_map.data(), 40));
    state.inputs.active = static_cast<double*>(copy(page_active.data(), 40));
    state.inputs.degeneracy = static_cast<double*>(copy(page_degeneracy.data(), 40));
    state.inputs.bar_triples_energy = static_cast<double*>(copy(&page_seed, 8));
    state.error = static_cast<int*>(copy(nullptr, sizeof(int)));
    state.arena = static_cast<double*>(
        copy(nullptr, generated::triples_response_arena_elements(2, 3, 5) * 8));
    auto out = generated::run_triples_response_cuda(state);
    int error{};
    check(cudaMemcpy(&error, state.error, sizeof(error), cudaMemcpyDeviceToHost));
    require(error == 0, "generated page error");
    const std::array<const double*, 8> device{out.ovvv, out.ovoo, out.ovov,  out.fov,
                                              out.t1,   out.t2,   out.eps_o, out.eps_v};
    TriplesResponseResult result;
    const std::array<std::vector<double>*, 8> host{&result.ovvv,  &result.ovoo, &result.ovov,
                                                   &result.fov,   &result.t1,   &result.t2,
                                                   &result.eps_o, &result.eps_v};
    for (std::size_t i = 0; i < host.size(); ++i) {
      host[i]->resize(page_fixture.expected[i].size());
      check(cudaMemcpy(host[i]->data(), device[i], host[i]->size() * 8, cudaMemcpyDeviceToHost));
    }
    compare(result, page_fixture);
    for (auto invalid : {std::int64_t(-1), std::int64_t(3)}) {
      auto maps = page_a_map;
      maps.back() = invalid;
      check(cudaMemcpy(const_cast<std::int64_t*>(state.inputs.a_map), maps.data(), 40,
                       cudaMemcpyHostToDevice));
      (void)generated::run_triples_response_cuda(state);
      check(cudaMemcpy(&error, state.error, sizeof(error), cudaMemcpyDeviceToHost));
      require(error != 0, "invalid map escaped error gate");
    }
  } catch (...) {
    (void)cudaDeviceSynchronize();
    for (auto pointer : allocations) (void)cudaFree(pointer);
    throw;
  }
  for (auto pointer : allocations) check(cudaFree(pointer));
}
}  // namespace

int main() {
  int devices{};
  if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
  try {
    for (const auto& fixture : fixtures) owner_case(fixture);
    page_case();
    std::cout << "native CUDA triples response: full VJP, indexed page and budget gates passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
