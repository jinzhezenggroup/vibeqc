#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <new>
#include <stdexcept>
#include <utility>

#include "bounded_workspace.hpp"
#include "resource_cuda.cuh"

namespace vibeqc::runtime {

inline void cuda_resource_check(cudaError_t status) {
  if (status == cudaErrorMemoryAllocation) throw std::bad_alloc();
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

class CudaDeviceScope {
 public:
  using StatusChecker = void (*)(cudaError_t);

  explicit CudaDeviceScope(int device, StatusChecker check = cuda_resource_check) {
    check(cudaGetDevice(&previous_));
    check(cudaSetDevice(device));
  }
  CudaDeviceScope(const CudaDeviceScope&) = delete;
  CudaDeviceScope& operator=(const CudaDeviceScope&) = delete;
  ~CudaDeviceScope() { (void)cudaSetDevice(previous_); }

 private:
  int previous_{};
};

class OwnedCudaStream {
 public:
  OwnedCudaStream() = default;
  explicit OwnedCudaStream(int device, unsigned flags = cudaStreamNonBlocking) {
    create(device, flags);
  }
  OwnedCudaStream(const OwnedCudaStream&) = delete;
  OwnedCudaStream& operator=(const OwnedCudaStream&) = delete;
  OwnedCudaStream(OwnedCudaStream&& other) noexcept { move_from(other); }
  OwnedCudaStream& operator=(OwnedCudaStream&& other) noexcept {
    if (this != &other) {
      reset();
      move_from(other);
    }
    return *this;
  }
  ~OwnedCudaStream() { reset(); }

  void create(int device, unsigned flags = cudaStreamNonBlocking) {
    if (stream_) throw std::logic_error("CUDA stream already owned");
    CudaDeviceScope guard(device);
    cuda_resource_check(cudaStreamCreateWithFlags(&stream_, flags));
    device_ = device;
  }

  void synchronize() const {
    if (!stream_) return;
    CudaDeviceScope guard(device_);
    cuda_resource_check(cudaStreamSynchronize(stream_));
  }

  void reset() noexcept {
    if (!stream_) return;
    int previous = 0;
    (void)cudaGetDevice(&previous);
    (void)cudaSetDevice(device_);
    (void)cudaStreamSynchronize(stream_);
    (void)cudaStreamDestroy(stream_);
    (void)cudaSetDevice(previous);
    stream_ = nullptr;
    device_ = -1;
  }

  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }
  [[nodiscard]] int device() const noexcept { return device_; }
  [[nodiscard]] explicit operator bool() const noexcept { return stream_ != nullptr; }

 private:
  void move_from(OwnedCudaStream& other) noexcept {
    device_ = other.device_;
    stream_ = std::exchange(other.stream_, nullptr);
    other.device_ = -1;
  }
  int device_{-1};
  cudaStream_t stream_{};
};

class OwnedCudaEvent {
 public:
  OwnedCudaEvent() = default;
  explicit OwnedCudaEvent(int device, unsigned flags = cudaEventDefault) { create(device, flags); }
  OwnedCudaEvent(const OwnedCudaEvent&) = delete;
  OwnedCudaEvent& operator=(const OwnedCudaEvent&) = delete;
  OwnedCudaEvent(OwnedCudaEvent&& other) noexcept { move_from(other); }
  OwnedCudaEvent& operator=(OwnedCudaEvent&& other) noexcept {
    if (this != &other) {
      reset();
      move_from(other);
    }
    return *this;
  }
  ~OwnedCudaEvent() { reset(); }

  void create(int device, unsigned flags = cudaEventDefault) {
    if (event_) throw std::logic_error("CUDA event already owned");
    CudaDeviceScope guard(device);
    cuda_resource_check(cudaEventCreateWithFlags(&event_, flags));
    device_ = device;
  }

  void record(cudaStream_t stream) const { cuda_resource_check(cudaEventRecord(event_, stream)); }
  void synchronize() const { cuda_resource_check(cudaEventSynchronize(event_)); }
  [[nodiscard]] float elapsed_since(const OwnedCudaEvent& begin) const {
    float milliseconds = 0;
    cuda_resource_check(cudaEventElapsedTime(&milliseconds, begin.event_, event_));
    return milliseconds;
  }

  void reset() noexcept {
    if (!event_) return;
    int previous = 0;
    (void)cudaGetDevice(&previous);
    (void)cudaSetDevice(device_);
    (void)cudaEventDestroy(event_);
    (void)cudaSetDevice(previous);
    event_ = nullptr;
    device_ = -1;
  }

  [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

 private:
  void move_from(OwnedCudaEvent& other) noexcept {
    device_ = other.device_;
    event_ = std::exchange(other.event_, nullptr);
    other.device_ = -1;
  }
  int device_{-1};
  cudaEvent_t event_{};
};

template <class T>
struct BorrowedCudaBuffer {
  T* data{};
  std::size_t elements{};
  int device{-1};
  cudaStream_t stream{};

  [[nodiscard]] TensorView<T> view() const noexcept { return {data, elements}; }
};

template <class T>
class OwnedCudaBuffer {
 public:
  OwnedCudaBuffer() = default;
  OwnedCudaBuffer(int device, std::size_t count, cudaStream_t lifetime_stream = nullptr) {
    allocate(device, count, lifetime_stream);
  }
  OwnedCudaBuffer(const OwnedCudaBuffer&) = delete;
  OwnedCudaBuffer& operator=(const OwnedCudaBuffer&) = delete;
  OwnedCudaBuffer(OwnedCudaBuffer&& other) noexcept { move_from(other); }
  OwnedCudaBuffer& operator=(OwnedCudaBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      move_from(other);
    }
    return *this;
  }
  ~OwnedCudaBuffer() { reset(); }

  void allocate(int device, std::size_t count, cudaStream_t lifetime_stream = nullptr) {
    if (data_) throw std::logic_error("CUDA buffer already owned");
    const auto bytes = size_mul(count, sizeof(T), "CUDA buffer byte count overflow");
    CudaDeviceScope guard(device);
    cuda_resource_check(resource_cuda_malloc(&data_, bytes));
    device_ = device;
    elements_ = count;
    lifetime_stream_ = lifetime_stream;
  }

  void reset() noexcept {
    if (!data_) return;
    int previous = 0;
    (void)cudaGetDevice(&previous);
    (void)cudaSetDevice(device_);
    if (lifetime_stream_) (void)cudaStreamSynchronize(lifetime_stream_);
    (void)resource_cuda_free(data_);
    (void)cudaSetDevice(previous);
    data_ = nullptr;
    elements_ = 0;
    lifetime_stream_ = nullptr;
    device_ = -1;
  }

  [[nodiscard]] T* get() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return elements_; }
  [[nodiscard]] BorrowedCudaBuffer<T> borrow() const noexcept {
    return {data_, elements_, device_, lifetime_stream_};
  }
  [[nodiscard]] explicit operator bool() const noexcept { return data_ != nullptr; }

 private:
  void move_from(OwnedCudaBuffer& other) noexcept {
    data_ = std::exchange(other.data_, nullptr);
    elements_ = std::exchange(other.elements_, 0);
    device_ = std::exchange(other.device_, -1);
    lifetime_stream_ = std::exchange(other.lifetime_stream_, nullptr);
  }

  T* data_{};
  std::size_t elements_{};
  int device_{-1};
  cudaStream_t lifetime_stream_{};
};

}  // namespace vibeqc::runtime
