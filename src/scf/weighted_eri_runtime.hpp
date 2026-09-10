#ifndef VIBEQC_SCF_WEIGHTED_ERI_RUNTIME_HPP
#define VIBEQC_SCF_WEIGHTED_ERI_RUNTIME_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>

#include "scf/cuda_weighted_eri.hpp"
#ifdef __CUDACC__
// Prepared integral providers share the arena/stream/device/error owner used
// by TensorIR and DFT. Scientific primitive evaluation is supplied by Program.
#include "tensor/cuda_runtime.cuh"
#endif

namespace vibeqc::scf::weighted_runtime {

using Result = CudaWeightedEriResult;

#ifdef __CUDACC__
#define VIBEQC_RESULT_HD __host__ __device__
#else
#define VIBEQC_RESULT_HD
#endif

/** Legacy value/gradient ABI, retained as the default result policy. */
struct GradientOutput {
  using Result = CudaWeightedEriResult;
  static constexpr unsigned count = 13;
  VIBEQC_RESULT_HD static double& element(Result& result, unsigned i) {
    return i == 0 ? result.value : result.center[(i - 1) / 3][(i - 1) % 3];
  }
};

/** A compile-time bounded coordinate tile for second-order consumers.
 * The storage owner, chunking, atomic reduction, failure handling and
 * transactional publication below remain common to both result families.
 */
template <unsigned Count>
struct CoordinateOutput {
  static_assert(Count > 0 && Count <= 12, "select one to twelve coordinate outputs per tile");
  struct Result {
    double values[Count];
  };
  static constexpr unsigned count = Count;
  VIBEQC_RESULT_HD static double& element(Result& result, unsigned i) { return result.values[i]; }
};

template <class Output>
VIBEQC_RESULT_HD void accumulate(typename Output::Result& target, typename Output::Result& value) {
  for (unsigned i = 0; i < Output::count; ++i) {
#ifdef __CUDA_ARCH__
    atomicAdd(&Output::element(target, i), Output::element(value, i));
#else
    Output::element(target, i) += Output::element(value, i);
#endif
  }
}
#undef VIBEQC_RESULT_HD

struct NumericalFailure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

inline std::size_t bytes(std::size_t count, std::size_t stride) {
  if (count > std::numeric_limits<std::size_t>::max() / stride)
    throw std::invalid_argument("weighted ERI byte count overflows size_t");
  return count * stride;
}

/** Validate the common normalized primitive prefix before any dispatch.
 * Program additionally checks the record ABI, radial identity and its bounded
 * component subset. Inputs and output publication remain separate phases.
 */
inline void validate_primitive(const CudaWeightedEriPrimitive& record, std::size_t tiles) {
  if (record.output_tile >= tiles) throw std::invalid_argument("weighted ERI output tile");
  for (unsigned c = 0; c < 4; ++c) {
    if (!std::isfinite(record.exponents[c]) || record.exponents[c] <= 0)
      throw std::invalid_argument("weighted ERI exponent must be finite and positive");
    unsigned angular = 0;
    for (unsigned a = 0; a < 3; ++a) {
      if (record.angular[c][a] > 3 || !std::isfinite(record.centers[c][a]))
        throw std::invalid_argument("weighted ERI angular component or center");
      angular += record.angular[c][a];
    }
    if (angular > 3) throw std::invalid_argument("weighted ERI supports shells through f");
  }
  for (double weight : record.weights)
    if (!std::isfinite(weight)) throw std::invalid_argument("weighted ERI weight must be finite");
}

#ifdef __CUDACC__
template <class Program, class Output>
__global__ void contract(const typename Program::Record* records, std::size_t count,
                         typename Output::Result* output, int* error) {
  using Result = typename Output::Result;
  for (std::size_t i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += std::size_t(blockDim.x) * gridDim.x) {
    Result result{};
    if (!Program::evaluate(records[i], result)) {
      atomicCAS(error, 0, 1);
      continue;
    }
    auto& target = output[Program::base(records[i]).output_tile];
    accumulate<Output>(target, result);
  }
}
#endif

/** Retained bounded storage for an immutable generated primitive program.
 *
 * The numeric budget covers the host publication candidate and, on CUDA,
 * uploaded records, compact device tile results and one error flag. Caller
 * inputs/outputs, object metadata and native call stacks/CUDA context are
 * excluded and must be accounted for by the enclosing resource plan. No
 * allocation occurs during run, including for partial/empty input chunks.
 * Execution calls serialize on the shared workspace; diagnostics and
 * destruction require caller serialization. A failed run never modifies the
 * caller's output buffer.
 */
template <class Program, class Output = GradientOutput>
class Plan {
 public:
  using Record = typename Program::Record;
  using Result = typename Output::Result;

  Plan(int device, int major, int minor, std::size_t capacity, std::size_t tiles,
       std::size_t budget)
      : capacity_(capacity), tiles_(tiles) {
    if (!capacity || !tiles || tiles > std::numeric_limits<std::uint32_t>::max())
      throw std::invalid_argument("weighted ERI capacities must be positive and representable");
    input_bytes_ = bytes(capacity, sizeof(Record));
    output_bytes_ = bytes(tiles, sizeof(Result));
    host_bytes_ = output_bytes_;
#ifdef __CUDACC__
    // Both POD strides are multiples of eight; the trailing flag is aligned.
    if (output_bytes_ > std::numeric_limits<std::size_t>::max() - sizeof(int) ||
        input_bytes_ > std::numeric_limits<std::size_t>::max() - output_bytes_ - sizeof(int))
      throw std::invalid_argument("weighted ERI arena size overflow");
    device_bytes_ = input_bytes_ + output_bytes_ + sizeof(int);
#else
    if (device != 0 || major != 0 || minor != 0)
      throw std::invalid_argument("CPU weighted ERI does not accept CUDA device controls");
#endif
    if (host_bytes_ > budget || device_bytes_ > budget - host_bytes_)
      throw std::invalid_argument("weighted ERI prepared numeric budget exceeded");
    candidate_.resize(tiles_);
#ifdef __CUDACC__
    context_.prepare(device, major, minor, device_bytes_, input_bytes_ + output_bytes_, 0, 0, 0,
                     false);
#endif
  }

  void run(const Record* records, std::size_t count, std::size_t tiles, Result* output,
           bool profile) {
#ifdef __CUDACC__
    std::lock_guard<std::mutex> lock(context_.mutex);
#else
    std::lock_guard<std::mutex> lock(mutex_);
    (void)profile;
#endif
    if (count > capacity_ || tiles > tiles_ || (count && !records) || (tiles && !output) ||
        (records && reinterpret_cast<std::uintptr_t>(records) % alignof(Record)) ||
        (output && reinterpret_cast<std::uintptr_t>(output) % alignof(Result)))
      throw std::invalid_argument("weighted ERI chunk dimensions or pointer alignment");
    for (std::size_t i = 0; i < count; ++i) {
      validate_primitive(Program::base(records[i]), tiles);
      if (!Program::validate(records[i]))
        throw std::invalid_argument("weighted ERI ABI, operator, omega or component mismatch");
    }
#ifdef __CUDACC__
    using vibeqc_tensor::cuda_check;
    context_.check_device();
    auto* device_records = reinterpret_cast<Record*>(context_.arena);
    auto* device_results = reinterpret_cast<Result*>(context_.arena + input_bytes_);
    context_.metrics.input_ms = context_.metrics.output_ms = context_.metrics.kernel_ms = 0;
    context_.section(profile, context_.metrics.input_ms, [&] {
      if (count)
        cuda_check(cudaMemcpyAsync(device_records, records, bytes(count, sizeof(Record)),
                                   cudaMemcpyHostToDevice, context_.stream));
      if (tiles)
        cuda_check(
            cudaMemsetAsync(device_results, 0, bytes(tiles, sizeof(Result)), context_.stream));
      cuda_check(cudaMemsetAsync(context_.error, 0, sizeof(int), context_.stream));
    });
    context_.section(profile, context_.metrics.kernel_ms, [&] {
      if (count) {
        // Grid-stride traversal bounds launch dimensions for large capacities.
        const auto blocks =
            static_cast<unsigned>(std::min<std::size_t>((count - 1) / 64 + 1, 65535));
        contract<Program, Output><<<blocks, 64, 0, context_.stream>>>(
            device_records, count, device_results, context_.error);
        cuda_check(cudaGetLastError());
      }
    });
    int error = 0;
    cuda_check(cudaMemcpyAsync(&error, context_.error, sizeof(error), cudaMemcpyDeviceToHost,
                               context_.stream));
    cuda_check(cudaStreamSynchronize(context_.stream));
    if (error) throw NumericalFailure("weighted ERI primitive produced a nonfinite result");
    context_.section(profile, context_.metrics.output_ms, [&] {
      if (tiles)
        cuda_check(cudaMemcpyAsync(candidate_.data(), device_results, bytes(tiles, sizeof(Result)),
                                   cudaMemcpyDeviceToHost, context_.stream));
    });
    cuda_check(cudaStreamSynchronize(context_.stream));
    context_.metrics.device_ms =
        context_.metrics.input_ms + context_.metrics.kernel_ms + context_.metrics.output_ms;
#else
    std::fill_n(candidate_.begin(), tiles, Result{});
    for (std::size_t i = 0; i < count; ++i) {
      Result value{};
      if (!Program::evaluate(records[i], value))
        throw NumericalFailure("weighted ERI primitive produced a nonfinite result");
      auto& target = candidate_[Program::base(records[i]).output_tile];
      accumulate<Output>(target, value);
    }
#endif
    for (std::size_t i = 0; i < tiles; ++i) {
      for (unsigned j = 0; j < Output::count; ++j)
        if (!std::isfinite(Output::element(candidate_[i], j)))
          throw NumericalFailure("weighted integral output overflow");
    }
    if (tiles) std::copy_n(candidate_.data(), tiles, output);
  }

  std::size_t host_bytes() const { return host_bytes_; }
  std::size_t device_bytes() const { return device_bytes_; }
#ifdef __CUDACC__
  const vibeqc_tensor::Metrics& metrics() const { return context_.metrics; }
#endif

 private:
  std::size_t capacity_, tiles_, input_bytes_{}, output_bytes_{}, host_bytes_{}, device_bytes_{};
  std::vector<Result> candidate_;
#ifdef __CUDACC__
  vibeqc_tensor::Context context_;
#else
  std::mutex mutex_;
#endif
};
}  // namespace vibeqc::scf::weighted_runtime
#endif
