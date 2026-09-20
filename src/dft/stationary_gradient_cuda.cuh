#pragma once
// Small-domain diagnostic runtime. Graph-emitted primitive, AO pullback and
// Becke entries precede this include; compiler-emitted contraction bodies follow it.
// This header owns only resource state, validation, transfers, launches and ABI.
#include <vector>

#include "../tensor/cuda_runtime.cuh"
#include "grid_task_view.cuh"
#include "xc_point.hpp"

namespace vibeqc_stationary_cuda {
using namespace vibeqc_tensor;
constexpr size_t workers = 32, record_stride = 26, map_stride = 8;
struct Owner {
  Context context;
  size_t atoms{}, aos{}, points{}, records{}, spin_blocks{}, bytes{};
  bool failed = true;  // An owner must be reset before its first source or read.
  double *record{}, *primitive{}, *centers{}, *weights{}, *raw{}, *partial{}, *scratch{},
      *sources{}, *density{}, *weighted_density{};
  int64_t *maps{}, *ao_atoms{}, *point_atoms{};
  uint64_t uploads{}, downloads{}, launches{}, primitive_count{}, point_count{}, pair_visits{};
};
// Caps make all products below representable before any allocation or pointer
// dereference. The fixed worker count bounds O(worker*natom) adjoint scratch.
size_t allocation(size_t na, size_t n, size_t np, size_t nr, size_t ns) {
  if (!na || na > 32 || !n || n > 128 || !np || np > 4096 || !nr || nr > 4096 ||
      (ns != 1 && ns != 2) || ns != stationary_spin_blocks)
    throw std::invalid_argument("stationary CUDA shape exceeds small-domain caps");
  return 8 * (record_stride * nr + 12 * nr + 3 * na + 2 * np + workers * 9 * na + workers * 9 * na +
              21 * na + map_stride * nr + n + np + 2 * ns * n * n) +
         256;
}
template <class F>
int guarded(Owner* owner, char* error, size_t size, F f) noexcept {
  try {
    f();
    return 0;
  } catch (const std::exception& e) {
    if (owner) owner->failed = true;
    error_text(error, size, e.what());
    return 1;
  } catch (...) {
    if (owner) owner->failed = true;
    error_text(error, size, "unknown stationary CUDA failure");
    return 1;
  }
}
void check(Owner& p) {
  if (p.failed) throw std::runtime_error("failed stationary owner; reset before reuse");
  p.context.check_device();
}
void finished(Owner& p, cudaStream_t stream) {
  int failure = 0;
  cuda_check(cudaGetLastError());
  cuda_check(
      cudaMemcpyAsync(&failure, p.context.error, sizeof(int), cudaMemcpyDeviceToHost, stream));
  cuda_check(cudaStreamSynchronize(stream));
  p.downloads += sizeof(int);
  if (failure) throw std::runtime_error("nonfinite or invalid stationary CUDA source");
}
template <class T>
void upload(Owner& p, T* out, const T* in, size_t n, cudaStream_t stream) {
  if (n && !in) throw std::invalid_argument("null stationary source");
  cuda_check(cudaMemcpyAsync(out, in, n * sizeof(T), cudaMemcpyHostToDevice, stream));
  p.uploads += n * sizeof(T);
}
__global__ void primitive_kernel(unsigned kind, unsigned source, const double* records,
                                 const int64_t* maps, size_t count, const double* density,
                                 const double* weighted_density, size_t n, double* output,
                                 int* error);
__global__ void primitive_reduce(const double* input, const int64_t* maps, size_t count, size_t na,
                                 double* output, int* error);
__global__ void validate_centers(const double* centers, size_t na, double tolerance, int* error);
__global__ void geometry_kernel(vibeqc::dft::GridTaskView view, const double* work,
                                const int64_t* ao_atoms, const int64_t* owners,
                                const double* centers, size_t na, const double* weights,
                                const double* raw, double* partial, double* scratch, int* error);
__global__ void geometry_reduce(const double* partial, size_t na, double* output, int* error);
}  // namespace vibeqc_stationary_cuda

extern "C" {
int stationary_create(int device, int major, int minor, size_t na, size_t n, size_t np, size_t nr,
                      size_t ns, size_t budget, void** output, char* error, size_t size) {
  using namespace vibeqc_stationary_cuda;
  if (output) *output = nullptr;
  return guarded(nullptr, error, size, [&] {
    if (!output) throw std::invalid_argument("null stationary owner output");
    const size_t bytes = allocation(na, n, np, nr, ns);
    if (bytes > budget) throw std::invalid_argument("stationary CUDA byte budget exceeded");
    // Reject invalid visible ordinals before Context stores/switches the device.
    // A failed cudaSetDevice otherwise leaves a latched runtime error that can
    // poison an unrelated later owner, including during partial destruction.
    int device_count = 0;
    cuda_check(cudaGetDeviceCount(&device_count));
    if (device < 0 || device >= device_count)
      throw std::invalid_argument("invalid stationary CUDA device ordinal");
    auto p = std::make_unique<Owner>();
    p->atoms = na;
    p->aos = n;
    p->points = np;
    p->records = nr;
    p->spin_blocks = ns;
    p->bytes = bytes;
    p->context.prepare(device, major, minor, bytes, bytes - 256, 0, 0, 0, false);
    auto* next = reinterpret_cast<double*>(p->context.arena);
    auto take = [&](size_t count) {
      auto* ptr = next;
      next += count;
      return ptr;
    };
    p->record = take(record_stride * nr);
    p->primitive = take(12 * nr);
    p->centers = take(3 * na);
    p->weights = take(np);
    p->raw = take(np);
    p->partial = take(workers * 9 * na);
    p->scratch = take(workers * 9 * na);
    p->sources = take(21 * na);
    p->maps = reinterpret_cast<int64_t*>(take(map_stride * nr));
    p->ao_atoms = reinterpret_cast<int64_t*>(take(n));
    p->point_atoms = reinterpret_cast<int64_t*>(take(np));
    p->density = take(ns * n * n);
    p->weighted_density = take(ns * n * n);
    *output = p.release();
  });
}
int stationary_reset(void* pointer, const double* centers, const int64_t* ao_atoms,
                     const double* density, const double* weighted_density, double tolerance,
                     char* error, size_t size) {
  using namespace vibeqc_stationary_cuda;
  auto* p = static_cast<Owner*>(pointer);
  return guarded(p, error, size, [&] {
    if (!p || !std::isfinite(tolerance) || tolerance < 0)
      throw std::invalid_argument("invalid reset");
    p->context.check_device();
    p->failed = false;
    auto stream = p->context.stream;
    cuda_check(cudaMemsetAsync(p->context.error, 0, sizeof(int), stream));
    cuda_check(cudaMemsetAsync(p->sources, 0, 21 * p->atoms * 8, stream));
    upload(*p, p->centers, centers, 3 * p->atoms, stream);
    upload(*p, p->ao_atoms, ao_atoms, p->aos, stream);
    upload(*p, p->density, density, p->spin_blocks * p->aos * p->aos, stream);
    upload(*p, p->weighted_density, weighted_density, p->spin_blocks * p->aos * p->aos, stream);
    validate_centers<<<1, 1, 0, stream>>>(p->centers, p->atoms, tolerance, p->context.error);
    ++p->launches;
    p->pair_visits += p->atoms * (p->atoms - 1) / 2;
    finished(*p, stream);
  });
}
int stationary_records(void* pointer, unsigned kind, unsigned source, const double* records,
                       const int64_t* maps, size_t count, char* error, size_t size) {
  using namespace vibeqc_stationary_cuda;
  auto* p = static_cast<Owner*>(pointer);
  return guarded(p, error, size, [&] {
    if (!p || !count || count > p->records ||
        (source != 0 && source != 1 && source != 5 && source != 6))
      throw std::invalid_argument("invalid primitive tile");
    check(*p);
    auto stream = p->context.stream;
    upload(*p, p->record, records, count * record_stride, stream);
    upload(*p, p->maps, maps, count * map_stride, stream);
    primitive_kernel<<<blocks(count, 64), 64, 0, stream>>>(kind, source, p->record, p->maps, count,
                                                           p->density, p->weighted_density, p->aos,
                                                           p->primitive, p->context.error);
    primitive_reduce<<<blocks(3 * p->atoms, 64), 64, 0, stream>>>(
        p->primitive, p->maps, count, p->atoms, p->sources + source * 3 * p->atoms,
        p->context.error);
    p->launches += 2;
    p->primitive_count += count;
    finished(*p, stream);
  });
}
int stationary_geometry(void* pointer, const vibeqc::dft::GridTaskView* view, const double* work,
                        const int64_t* owners, const double* weights, const double* raw,
                        char* error, size_t size) {
  using namespace vibeqc_stationary_cuda;
  auto* p = static_cast<Owner*>(pointer);
  return guarded(p, error, size, [&] {
    if (!p || !view || view->version != 1 || view->nao != p->aos || view->nactive != p->aos ||
        view->npoint > p->points || view->jets < stationary_ao_jets || !view->features || !work ||
        !view->ao_ids || !view->ao || !view->points)
      throw std::invalid_argument("invalid geometry task lease");
    check(*p);
    auto stream = view->stream;
    // CudaGrid synchronized its producer before lending this view. Finish on
    // the SAME borrowed stream before the lease ends; retain no task pointers.
    try {
      upload(*p, p->point_atoms, owners, view->npoint, stream);
      upload(*p, p->weights, weights, view->npoint, stream);
      upload(*p, p->raw, raw, view->npoint, stream);
      geometry_kernel<<<1, workers, 0, stream>>>(*view, work, p->ao_atoms, p->point_atoms,
                                                 p->centers, p->atoms, p->weights, p->raw,
                                                 p->partial, p->scratch, p->context.error);
      geometry_reduce<<<blocks(9 * p->atoms, 64), 64, 0, stream>>>(
          p->partial, p->atoms, p->sources + 6 * p->atoms, p->context.error);
      p->launches += 2;
      p->point_count += view->npoint;
      p->pair_visits += view->npoint * p->atoms * (p->atoms - 1);
      finished(*p, stream);
    } catch (...) {
      // Even an upload/launch failure must drain the borrowed stream before
      // our arena can be freed or the grid owner can reuse its leased buffers.
      cudaStreamSynchronize(stream);
      throw;
    }
  });
}
int stationary_finish(void* pointer, double* output, size_t count, char* error, size_t size) {
  using namespace vibeqc_stationary_cuda;
  auto* p = static_cast<Owner*>(pointer);
  return guarded(p, error, size, [&] {
    if (!p || !output || count != 21 * p->atoms)
      throw std::invalid_argument("invalid source output");
    check(*p);
    finished(*p, p->context.stream);
    // Host output is touched only after every device source passed its gate.
    std::vector<double> candidate(count);
    cuda_check(cudaMemcpy(candidate.data(), p->sources, count * 8, cudaMemcpyDeviceToHost));
    p->downloads += count * 8;
    for (double v : candidate)
      if (!std::isfinite(v)) throw std::runtime_error("nonfinite gradient");
    std::copy(candidate.begin(), candidate.end(), output);
  });
}
int stationary_metrics(void* pointer, uint64_t* output, size_t count) {
  auto* p = static_cast<vibeqc_stationary_cuda::Owner*>(pointer);
  if (!p || !output || count != 8) return 1;
  const uint64_t values[]{p->bytes,           p->uploads,
                          p->downloads,       p->launches,
                          p->primitive_count, p->point_count,
                          p->pair_visits,     reinterpret_cast<uintptr_t>(p->context.stream)};
  std::copy(values, values + 8, output);
  return 0;
}
void stationary_destroy(void* pointer) {
  delete static_cast<vibeqc_stationary_cuda::Owner*>(pointer);
}
}
