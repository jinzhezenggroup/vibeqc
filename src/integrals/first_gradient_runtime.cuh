// Shared streaming owner for generated weighted first-integral gradient consumers.
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

namespace vibeqc::integrals::first_gradient {
inline std::size_t mul(std::size_t a, std::size_t b) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::invalid_argument("first-gradient storage overflow");
  return a * b;
}
inline std::size_t add(std::size_t a, std::size_t b) {
  if (a > std::numeric_limits<std::size_t>::max() - b)
    throw std::invalid_argument("first-gradient storage overflow");
  return a + b;
}
struct NumericalFailure : std::runtime_error {
  using std::runtime_error::runtime_error;
};
struct Mapping {
  std::size_t offsets[4], atoms[4];
};
constexpr std::size_t stride = 17;

__global__ void validate_finite_weights(const double* values, std::size_t count, int* error) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += std::size_t(blockDim.x) * gridDim.x)
    if (!isfinite(values[i])) atomicCAS(error, 0, 1);
}

__global__ inline void validate_output(const double* values, std::size_t count, int* error) {
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += std::size_t(blockDim.x) * gridDim.x)
    if (!isfinite(values[i])) atomicCAS(error, 0, 1);
}

struct Plan {
  static constexpr uint64_t tag_value = 0x5649424546475231ULL;
  uint64_t tag = tag_value;
  char abi[65]{};
  std::size_t nbf, natoms, weight_slots, capacity, matrix_size, weight_size, output_size,
      device_bytes;
  std::size_t output_offset, record_offset;
  bool valid = false;
  vibeqc_tensor::Context context;
  std::vector<double> candidate;

  Plan(int device, int major, int minor, std::size_t n, std::size_t atoms, std::size_t slots,
       std::size_t cap, std::size_t budget, const char* identity)
      : nbf(n),
        natoms(atoms),
        weight_slots(slots),
        capacity(cap),
        matrix_size(mul(n, n)),
        weight_size(mul(slots, matrix_size)),
        output_size(mul(atoms, 3)) {
    if (!n || !atoms || !slots || slots > 8 || !cap || cap > (1U << 20) || !identity ||
        std::strlen(identity) != 64)
      throw std::invalid_argument("invalid first-gradient plan dimensions/ABI");
    std::memcpy(abi, identity, 65);
    output_offset = weight_size;
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
      throw std::invalid_argument("first-gradient runtime ABI mismatch");
    context.check_device();
  }
  void reset(const double* weights, std::size_t size, const char* identity) {
    std::lock_guard<std::mutex> lock(context.mutex);
    check(identity);
    valid = false;
    if (!weights || size != weight_size)
      throw std::invalid_argument("first-gradient input shape mismatch");
    for (std::size_t i = 0; i < size; ++i)
      if (!std::isfinite(weights[i]))
        throw std::invalid_argument("nonfinite first-gradient weight");
    vibeqc_tensor::cuda_check(cudaMemcpyAsync(data(), weights, size * sizeof(double),
                                              cudaMemcpyHostToDevice, context.stream));
    vibeqc_tensor::cuda_check(
        cudaMemsetAsync(data() + output_offset, 0, output_size * sizeof(double), context.stream));
    vibeqc_tensor::cuda_check(cudaMemsetAsync(context.error, 0, sizeof(int), context.stream));
    vibeqc_tensor::cuda_check(cudaStreamSynchronize(context.stream));
    valid = true;
  }
  void reset_mixed(const double* device_weights, std::size_t device_count,
                   const double* host_weights, std::size_t host_count, const char* identity) {
    std::lock_guard<std::mutex> lock(context.mutex);
    check(identity);
    valid = false;
    if (!device_weights || !device_count || !host_weights || !host_count ||
        device_count + host_count != weight_size)
      throw std::invalid_argument("first-gradient mixed weight shape mismatch");
    for (std::size_t i = 0; i < host_count; ++i)
      if (!std::isfinite(host_weights[i]))
        throw std::invalid_argument("nonfinite first-gradient host weight");
    vibeqc_tensor::cuda_check(cudaMemcpyAsync(data(), device_weights, device_count * sizeof(double),
                                              cudaMemcpyDeviceToDevice, context.stream));
    vibeqc_tensor::cuda_check(cudaMemcpyAsync(data() + device_count, host_weights,
                                              host_count * sizeof(double), cudaMemcpyHostToDevice,
                                              context.stream));
    vibeqc_tensor::cuda_check(
        cudaMemsetAsync(data() + output_offset, 0, output_size * sizeof(double), context.stream));
    vibeqc_tensor::cuda_check(cudaMemsetAsync(context.error, 0, sizeof(int), context.stream));
    validate_finite_weights<<<vibeqc_tensor::blocks(weight_size, 128), 128, 0, context.stream>>>(
        data(), weight_size, context.error);
    vibeqc_tensor::cuda_check(cudaGetLastError());
    int error = 0;
    vibeqc_tensor::cuda_check(cudaMemcpyAsync(&error, context.error, sizeof(int),
                                              cudaMemcpyDeviceToHost, context.stream));
    vibeqc_tensor::cuda_check(cudaStreamSynchronize(context.stream));
    if (error) throw NumericalFailure("nonfinite first-gradient resident weight");
    valid = true;
  }
  const double* output_device(const char* identity) {
    std::lock_guard<std::mutex> lock(context.mutex);
    check(identity);
    if (!valid) throw std::runtime_error("first-gradient plan requires successful reset/run");
    vibeqc_tensor::cuda_check(cudaMemsetAsync(context.error, 0, sizeof(int), context.stream));
    validate_output<<<vibeqc_tensor::blocks(output_size, 128), 128, 0, context.stream>>>(
        data() + output_offset, output_size, context.error);
    vibeqc_tensor::cuda_check(cudaGetLastError());
    int error = 0;
    vibeqc_tensor::cuda_check(cudaMemcpyAsync(&error, context.error, sizeof(int),
                                              cudaMemcpyDeviceToHost, context.stream));
    vibeqc_tensor::cuda_check(cudaStreamSynchronize(context.stream));
    if (error) throw NumericalFailure("nonfinite first-gradient resident output");
    return data() + output_offset;
  }

  void finish(double* output, std::size_t size, const char* identity) {
    std::lock_guard<std::mutex> lock(context.mutex);
    check(identity);
    if (!valid) throw std::runtime_error("first-gradient plan requires successful reset/run");
    if (!output || size != output_size)
      throw std::invalid_argument("first-gradient output size mismatch");
    valid = false;
    int error = 0;
    vibeqc_tensor::cuda_check(cudaMemcpyAsync(&error, context.error, sizeof(int),
                                              cudaMemcpyDeviceToHost, context.stream));
    vibeqc_tensor::cuda_check(cudaMemcpyAsync(candidate.data(), data() + output_offset,
                                              size * sizeof(double), cudaMemcpyDeviceToHost,
                                              context.stream));
    vibeqc_tensor::cuda_check(cudaStreamSynchronize(context.stream));
    if (error) throw NumericalFailure("nonfinite generated first-gradient contribution");
    for (double value : candidate)
      if (!std::isfinite(value)) throw NumericalFailure("nonfinite first-gradient accumulation");
    std::memcpy(output, candidate.data(), size * sizeof(double));
    valid = true;
  }
};

template <class Program>
__global__ void execute(const double* records, std::size_t count, Mapping mapping, std::size_t nbf,
                        const double* weights, double* output, int* error) {
  const std::size_t work = count * Program::components;
  for (std::size_t item = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; item < work;
       item += std::size_t(blockDim.x) * gridDim.x) {
    Program::accumulate(records + (item / Program::components) * stride, item % Program::components,
                        mapping, nbf, weights, output, error);
  }
}

template <class Program>
void append(Plan& p, const double* records, std::size_t count, const Mapping& mapping,
            const char* identity) {
  std::lock_guard<std::mutex> lock(p.context.mutex);
  p.check(identity);
  if (!p.valid) throw std::runtime_error("first-gradient plan requires successful reset/run");
  p.valid = false;
  if (count > p.capacity || (count && !records) || Program::weight_slots > p.weight_slots)
    throw std::invalid_argument("first-gradient chunk exceeds prepared capacity");
  for (unsigned s = 0; s < Program::exponents; ++s)
    if (mapping.offsets[s] > p.nbf || Program::extent(s) > p.nbf - mapping.offsets[s])
      throw std::invalid_argument("first-gradient AO mapping out of bounds");
  for (unsigned c = 0; c < Program::centers; ++c)
    if (mapping.atoms[c] >= p.natoms)
      throw std::invalid_argument("first-gradient atom mapping out of bounds");
  for (std::size_t r = 0; r < count; ++r) {
    const auto* row = records + r * stride;
    for (unsigned j = 0; j < stride; ++j)
      if (!std::isfinite(row[j]) || (j < Program::exponents && row[j] <= 0))
        throw std::invalid_argument("invalid first-gradient primitive");
  }
  if (count) {
    vibeqc_tensor::cuda_check(cudaMemcpyAsync(p.data() + p.record_offset, records,
                                              count * stride * sizeof(double),
                                              cudaMemcpyHostToDevice, p.context.stream));
    const auto work = count * Program::components;
    execute<Program><<<vibeqc_tensor::blocks(work, 64), 64, 0, p.context.stream>>>(
        p.data() + p.record_offset, count, mapping, p.nbf, p.data(), p.data() + p.output_offset,
        p.context.error);
    vibeqc_tensor::cuda_check(cudaGetLastError());
    int error = 0;
    vibeqc_tensor::cuda_check(cudaMemcpyAsync(&error, p.context.error, sizeof(int),
                                              cudaMemcpyDeviceToHost, p.context.stream));
    vibeqc_tensor::cuda_check(cudaStreamSynchronize(p.context.stream));
    if (error) throw NumericalFailure("generated first-gradient primitive failed");
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
}  // namespace vibeqc::integrals::first_gradient
