// Shared streaming owner for generated directional first-integral consumers.
// Program owns derivative/contraction mathematics. This file owns no HF terms.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "tensor/cuda_runtime.cuh"

namespace vibeqc::integrals::first_directional {
inline std::size_t mul(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::invalid_argument("directional storage overflow");
  return a * b;
}
inline std::size_t add(std::size_t a, std::size_t b) {
  if (a > std::numeric_limits<std::size_t>::max() - b)
    throw std::invalid_argument("directional storage overflow");
  return a + b;
}
struct NumericalFailure : std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct Mapping {
  std::size_t offsets[4], atoms[4];
};
// Stable record prefix: four exponent slots, twelve center slots, radial scale.
constexpr std::size_t stride = 17;
struct Plan {
  static constexpr uint64_t tag_value = 0x5649424544495231ULL;
  uint64_t tag = tag_value;
  char abi[65]{};
  std::size_t nbf, natoms, outputs, capacity, matrix_size, output_size, device_bytes;
  std::size_t direction_offset, output_offset, record_offset;
  bool valid = false;
  vibeqc_tensor::Context context;
  std::vector<double> candidate;

  Plan(int device, int major, int minor, std::size_t n, std::size_t atoms, std::size_t slots,
       std::size_t cap, std::size_t budget, const char* identity)
      : nbf(n),
        natoms(atoms),
        outputs(slots),
        capacity(cap),
        matrix_size(mul(n, n)),
        output_size(mul(slots, matrix_size)) {
    if (!n || !atoms || !slots || slots > 32 || !cap || cap > (1U << 20) || !identity ||
        std::strlen(identity) != 64)
      throw std::invalid_argument("invalid directional plan dimensions/ABI");
    std::memcpy(abi, identity, 65);
    direction_offset = matrix_size;
    output_offset = add(direction_offset, mul(atoms, 3));
    record_offset = add(output_offset, output_size);
    const auto error_offset = mul(add(record_offset, mul(cap, stride)), sizeof(double));
    device_bytes = add(error_offset, sizeof(double));
    if (add(device_bytes, mul(output_size, sizeof(double))) > budget) throw std::bad_alloc();
    candidate.resize(output_size);
    context.prepare(device, major, minor, device_bytes, error_offset, 0, 0, 0, false);
  }
  double* data() { return reinterpret_cast<double*>(context.arena); }
  void check(const char* identity) {
    if (tag != tag_value || std::strcmp(abi, identity) != 0)
      throw std::invalid_argument("directional runtime ABI mismatch");
    context.check_device();
  }
  void reset(const double* weights, std::size_t weight_size, const double* direction,
             std::size_t direction_size, const char* identity) {
    std::lock_guard<std::mutex> lock(context.mutex);
    check(identity);
    valid = false;
    if (!weights || !direction || weight_size != matrix_size || direction_size != natoms * 3)
      throw std::invalid_argument("directional input shape mismatch");
    for (std::size_t i = 0; i < weight_size; ++i)
      if (!std::isfinite(weights[i])) throw std::invalid_argument("nonfinite external weight");
    for (std::size_t i = 0; i < direction_size; ++i)
      if (!std::isfinite(direction[i])) throw std::invalid_argument("nonfinite direction");
    vibeqc_tensor::cuda_check(cudaMemcpyAsync(data(), weights, weight_size * sizeof(double),
                                              cudaMemcpyHostToDevice, context.stream));
    vibeqc_tensor::cuda_check(cudaMemcpyAsync(data() + direction_offset, direction,
                                              direction_size * sizeof(double),
                                              cudaMemcpyHostToDevice, context.stream));
    vibeqc_tensor::cuda_check(
        cudaMemsetAsync(data() + output_offset, 0, output_size * sizeof(double), context.stream));
    vibeqc_tensor::cuda_check(cudaMemsetAsync(context.error, 0, sizeof(int), context.stream));
    vibeqc_tensor::cuda_check(cudaStreamSynchronize(context.stream));
    valid = true;
  }
  void finish(double* output, std::size_t size, const char* identity) {
    std::lock_guard<std::mutex> lock(context.mutex);
    check(identity);
    if (!valid) throw std::runtime_error("directional plan requires successful reset/run");
    if (!output || size != output_size) throw std::invalid_argument("directional output size");
    valid = false;
    int error = 0;
    vibeqc_tensor::cuda_check(cudaMemcpyAsync(&error, context.error, sizeof(int),
                                              cudaMemcpyDeviceToHost, context.stream));
    vibeqc_tensor::cuda_check(cudaMemcpyAsync(candidate.data(), data() + output_offset,
                                              size * sizeof(double), cudaMemcpyDeviceToHost,
                                              context.stream));
    vibeqc_tensor::cuda_check(cudaStreamSynchronize(context.stream));
    if (error) throw NumericalFailure("nonfinite generated directional contribution");
    for (double value : candidate)
      if (!std::isfinite(value))
        throw NumericalFailure("nonfinite directional matrix accumulation");
    // Publication only after every program/chunk and the complete output pass.
    std::memcpy(output, candidate.data(), size * sizeof(double));
    valid = true;
  }
};

template <class Program>
__global__ void execute(const double* records, std::size_t count, Mapping mapping, std::size_t nbf,
                        const double* direction, const double* weights, double* output,
                        int* error) {
  const std::size_t work = count * Program::components;
  for (std::size_t item = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; item < work;
       item += std::size_t(blockDim.x) * gridDim.x) {
    Program::accumulate(records + (item / Program::components) * stride, item % Program::components,
                        mapping, nbf, direction, weights, output, error);
  }
}

template <class Program>
void append(Plan& p, const double* records, std::size_t count, const Mapping& mapping,
            const char* identity) {
  std::lock_guard<std::mutex> lock(p.context.mutex);
  p.check(identity);
  if (!p.valid) throw std::runtime_error("directional plan requires successful reset/run");
  p.valid = false;
  if (count > p.capacity || (count && !records) || Program::output_slots > p.outputs)
    throw std::invalid_argument("directional chunk exceeds prepared capacity");
  for (unsigned s = 0; s < Program::exponents; ++s)
    if (mapping.offsets[s] > p.nbf || Program::extent(s) > p.nbf - mapping.offsets[s])
      throw std::invalid_argument("directional AO mapping out of bounds");
  for (unsigned c = 0; c < Program::centers; ++c)
    if (mapping.atoms[c] >= p.natoms)
      throw std::invalid_argument("directional atom mapping out of bounds");
  for (std::size_t r = 0; r < count; ++r) {
    const auto* row = records + r * stride;
    for (unsigned j = 0; j < stride; ++j)
      if (!std::isfinite(row[j]) || (j < Program::exponents && row[j] <= 0))
        throw std::invalid_argument("invalid directional primitive");
  }
  if (count) {
    vibeqc_tensor::cuda_check(cudaMemcpyAsync(p.data() + p.record_offset, records,
                                              count * stride * sizeof(double),
                                              cudaMemcpyHostToDevice, p.context.stream));
    const auto work = count * Program::components;
    execute<Program><<<vibeqc_tensor::blocks(work, 64), 64, 0, p.context.stream>>>(
        p.data() + p.record_offset, count, mapping, p.nbf, p.data() + p.direction_offset, p.data(),
        p.data() + p.output_offset, p.context.error);
    vibeqc_tensor::cuda_check(cudaGetLastError());
    // The caller may refill host staging after append. One scalar status check
    // transfers no raw derivative or intermediate matrix to the host.
    int error = 0;
    vibeqc_tensor::cuda_check(cudaMemcpyAsync(&error, p.context.error, sizeof(int),
                                              cudaMemcpyDeviceToHost, p.context.stream));
    vibeqc_tensor::cuda_check(cudaStreamSynchronize(p.context.stream));
    if (error) throw NumericalFailure("generated directional primitive failed");
  }
  p.valid = true;
}

template <class Operation>
int boundary(Operation operation, char* detail, std::size_t size) {
  if (detail && size) detail[0] = 0;
  try {
    operation();
    return 0;
  } catch (const NumericalFailure& e) {
    vibeqc_tensor::error_text(detail, size, e.what());
    return 5;
  } catch (const vibeqc_tensor::DeviceAllocationError& e) {
    vibeqc_tensor::error_text(detail, size, e.what());
    return 7;
  } catch (const std::bad_alloc& e) {
    vibeqc_tensor::error_text(detail, size, e.what());
    return 7;
  } catch (const std::invalid_argument& e) {
    vibeqc_tensor::error_text(detail, size, e.what());
    return 1;
  } catch (const std::exception& e) {
    vibeqc_tensor::error_text(detail, size, e.what());
    return 8;
  }
}
}  // namespace vibeqc::integrals::first_directional
