/** Bounded FP64 AO spatial jets and spin density contractions on CUDA.
 * Host-built grid tiles are explicit inputs. Native normalized shell data,
 * D matrices, cuBLAS handle, stream and reusable arena are owned by the plan.
 */
#include <climits>
#include <cmath>

#include "../tensor/cuda_runtime.cuh"

namespace {
using namespace vibeqc_tensor;
struct GridPlan {
  Context context;
  bool density_ready = false;
  size_t natom{}, nprimitive{}, nao{}, capacity{}, jets{}, packed_size{};
  double *basis{}, *density{}, *points{}, *ao{}, *work{}, *features{};
};
size_t mul(size_t a, size_t b) {
  if (b && a > SIZE_MAX / b) throw std::overflow_error("grid allocation overflow");
  return a * b;
}
size_t add(size_t a, size_t b) {
  if (a > SIZE_MAX - b) throw std::overflow_error("grid allocation overflow");
  return a + b;
}
template <class F>
int guarded(char* error, size_t size, F operation) noexcept {
  try {
    operation();
    return 0;
  } catch (const std::exception& e) {
    error_text(error, size, e.what());
    return 1;
  } catch (...) {
    error_text(error, size, "unknown CUDA grid failure");
    return 1;
  }
}

__constant__ int derivatives[20][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {2, 0, 0},
                                       {1, 1, 0}, {1, 0, 1}, {0, 2, 0}, {0, 1, 1}, {0, 0, 2},
                                       {3, 0, 0}, {2, 1, 0}, {2, 0, 1}, {1, 2, 0}, {1, 1, 1},
                                       {1, 0, 2}, {0, 3, 0}, {0, 2, 1}, {0, 1, 2}, {0, 0, 3}};

__device__ double power(double x, int degree) {
  double result = 1;
  for (int k = 0; k < degree; ++k) result *= x;
  return result;
}
// Independent closed Gaussian/Leibniz derivatives, rather than the native
// CPU polynomial recurrence. Derivative jets are not Taylor coefficients.
__device__ double axis_jet(int l, int d, double a, double x) {
  const double t = -2 * a * x;
  double result = 0, falling = 1;
  for (int m = 0; m <= d && m <= l; ++m) {
    if (m) falling *= l - m + 1;
    const int g = d - m;
    double gaussian = g == 0 ? 1 : g == 1 ? t : g == 2 ? t * t - 2 * a : t * t * t - 6 * a * t;
    // Binomial coefficients for derivative orders zero through three.
    const int binomial = m == 0 || m == d ? 1 : d;
    result += binomial * falling * power(x, l - m) * gaussian;
  }
  return result;
}

__global__ void ao_kernel(const double* basis, I natom, I nprimitive, I nao, const double* points,
                          I npoint, I jets, double* output, int* error) {
  const double* primitives = basis + 3 * natom;
  const double* records = primitives + 2 * nprimitive;
  for (I index = I(blockIdx.x) * blockDim.x + threadIdx.x; index < jets * npoint * nao;
       index += I(blockDim.x) * gridDim.x) {
    const I ao = index % nao, point = index / nao % npoint, jet = index / (nao * npoint);
    const double* record = records + 16 * ao;
    const I atom = static_cast<I>(record[0]);
    const double x = points[3 * point] - basis[3 * atom];
    const double y = points[3 * point + 1] - basis[3 * atom + 1];
    const double z = points[3 * point + 2] - basis[3 * atom + 2];
    const double r2 = x * x + y * y + z * z;
    const I first = static_cast<I>(record[1]), end = first + static_cast<I>(record[2]);
    double value = 0;
    for (I p = first; p < end; ++p) {
      const double alpha = primitives[2 * p];
      const double radial = primitives[2 * p + 1] * exp(-alpha * r2);
      if (radial == 0) continue;
      for (int term = 0; term < static_cast<int>(record[3]); ++term) {
        value += radial * record[7 + 4 * term] *
                 axis_jet(static_cast<int>(record[4 + 4 * term]), derivatives[jet][0], alpha, x) *
                 axis_jet(static_cast<int>(record[5 + 4 * term]), derivatives[jet][1], alpha, y) *
                 axis_jet(static_cast<int>(record[6 + 4 * term]), derivatives[jet][2], alpha, z);
      }
    }
    output[index] = finite(value, error, 0);
  }
}

__global__ void feature_kernel(const double* ao, const double* work, I npoint, I nao,
                               double* output, int* error) {
  for (I point = I(blockIdx.x) * blockDim.x + threadIdx.x; point < npoint;
       point += I(blockDim.x) * gridDim.x) {
    double gradients[2][3]{};
    const I stride = npoint * nao;
    for (int spin = 0; spin < 2; ++spin) {
      const double* w = work + 4 * spin * stride;
      double rho = 0, tau = 0;
      for (I mu = 0; mu < nao; ++mu) {
        const I i = point * nao + mu;
        rho += ao[i] * w[i];
        for (int axis = 0; axis < 3; ++axis) {
          const double derivative = ao[(axis + 1) * stride + i];
          gradients[spin][axis] += 2 * derivative * w[i];
          tau += 0.5 * derivative * w[(axis + 1) * stride + i];
        }
      }
      output[(5 * spin) * npoint + point] = finite(rho, error, 1);
      for (int axis = 0; axis < 3; ++axis)
        output[(5 * spin + 1 + axis) * npoint + point] = finite(gradients[spin][axis], error, 1);
      output[(5 * spin + 4) * npoint + point] = finite(tau, error, 1);
    }
    double aa = 0, ab = 0, bb = 0;
    for (int k = 0; k < 3; ++k) {
      aa += gradients[0][k] * gradients[0][k];
      ab += gradients[0][k] * gradients[1][k];
      bb += gradients[1][k] * gradients[1][k];
    }
    output[10 * npoint + point] = finite(aa, error, 1);
    output[11 * npoint + point] = finite(ab, error, 1);
    output[12 * npoint + point] = finite(bb, error, 1);
  }
}
}  // namespace

extern "C" {
int grid_cuda_create_v1(int device, int major, int minor, const size_t* dimensions,
                        const double* basis, size_t capacity, unsigned order, size_t expected_bytes,
                        void** output, char* error, size_t size) {
  return guarded(error, size, [&] {
    if (!output) throw std::invalid_argument("null CUDA grid output");
    *output = nullptr;
    if (!dimensions || !basis || !capacity || capacity > INT_MAX || order > 3)
      throw std::invalid_argument("invalid CUDA grid plan");
    auto p = std::make_unique<GridPlan>();
    p->natom = dimensions[0];
    p->nprimitive = dimensions[1];
    p->nao = dimensions[2];
    if (!p->natom || !p->nprimitive || !p->nao || p->natom > INT_MAX || p->nprimitive > INT_MAX ||
        p->nao > INT_MAX)
      throw std::invalid_argument("invalid CUDA grid basis dimensions");
    p->capacity = capacity;
    p->jets = (order + 1) * (order + 2) * (order + 3) / 6;
    p->packed_size = add(add(mul(3, p->natom), mul(2, p->nprimitive)), mul(16, p->nao));
    for (size_t i = 0; i < p->packed_size; ++i)
      if (!std::isfinite(basis[i])) throw std::invalid_argument("nonfinite CUDA grid basis");
    const auto integral = [](double x, size_t limit) {
      return x >= 0 && x <= limit && x == std::floor(x);
    };
    const double* records = basis + 3 * p->natom + 2 * p->nprimitive;
    for (size_t a = 0; a < p->nao; ++a) {
      const double* r = records + 16 * a;
      if (!integral(r[0], p->natom - 1) || !integral(r[1], p->nprimitive) ||
          !integral(r[2], p->nprimitive) || r[2] < 1 || r[1] + r[2] > p->nprimitive ||
          !integral(r[3], 3) || r[3] < 1)
        throw std::invalid_argument("invalid packed AO bounds");
      for (int t = 0; t < static_cast<int>(r[3]); ++t)
        if (!integral(r[4 + 4 * t], 3) || !integral(r[5 + 4 * t], 3) ||
            !integral(r[6 + 4 * t], 3) || r[4 + 4 * t] + r[5 + 4 * t] + r[6 + 4 * t] > 3)
          throw std::invalid_argument("unsupported packed AO powers");
    }
    for (size_t i = 0; i < p->nprimitive; ++i)
      if (!(basis[3 * p->natom + 2 * i] > 0))
        throw std::invalid_argument("invalid Gaussian exponent");
    const size_t matrices = mul(2, mul(p->nao, p->nao));
    const size_t tile = mul(capacity, p->nao);
    if (mul(p->jets, tile) > static_cast<size_t>(INT64_MAX))
      throw std::invalid_argument("CUDA grid index overflow");
    const size_t elements =
        add(add(p->packed_size, matrices), add(mul(16, capacity), mul(p->jets + 8, tile)));
    const size_t numeric = mul(8, elements), error_offset = mul(add(numeric, 255) / 256, 256);
    const size_t workspace = add(error_offset, 256), bytes = add(workspace, 4U << 20);
    if (bytes != expected_bytes) throw std::invalid_argument("native/Python grid plan mismatch");
    p->context.prepare(device, major, minor, bytes, error_offset, workspace, 4U << 20, 96U << 20,
                       true);
    p->basis = reinterpret_cast<double*>(p->context.arena);
    p->density = p->basis + p->packed_size;
    p->points = p->density + matrices;
    p->ao = p->points + 3 * capacity;
    p->work = p->ao + p->jets * tile;
    p->features = p->work + 8 * tile;
    p->context.section(true, p->context.metrics.input_ms, [&] {
      cuda_check(cudaMemcpyAsync(p->basis, basis, p->packed_size * 8, cudaMemcpyHostToDevice,
                                 p->context.stream));
    });
    *output = p.release();
  });
}
void grid_cuda_destroy_v1(void* pointer) { delete static_cast<GridPlan*>(pointer); }

int grid_cuda_density_v1(void* pointer, const double* density, size_t elements, char* error,
                         size_t size) {
  return guarded(error, size, [&] {
    if (!pointer || !density) throw std::invalid_argument("null CUDA grid density");
    auto& p = *static_cast<GridPlan*>(pointer);
    auto& ctx = p.context;
    std::lock_guard<std::mutex> lock(ctx.mutex);
    ctx.check_device();
    if (elements != 2 * p.nao * p.nao) throw std::invalid_argument("density size mismatch");
    for (size_t i = 0; i < elements; ++i)
      if (!std::isfinite(density[i])) throw std::invalid_argument("nonfinite density");
    ctx.section(true, ctx.metrics.input_ms, [&] {
      cuda_check(
          cudaMemcpyAsync(p.density, density, elements * 8, cudaMemcpyHostToDevice, ctx.stream));
    });
    p.density_ready = true;
  });
}

int grid_cuda_run_v1(void* pointer, const double* points, size_t npoint, int features,
                     double* feature_output, double* jet_output, char* error, size_t size) {
  return guarded(error, size, [&] {
    if (!pointer || (features != 0 && features != 1))
      throw std::invalid_argument("invalid CUDA grid execution");
    auto& p = *static_cast<GridPlan*>(pointer);
    auto& ctx = p.context;
    std::lock_guard<std::mutex> lock(ctx.mutex);
    ctx.check_device();
    if (npoint > p.capacity || (npoint && !points) ||
        (features && (!feature_output || p.jets < 4 || !p.density_ready)) ||
        (!features && !jet_output))
      throw std::invalid_argument("invalid grid tile/output");
    if (!npoint) return;
    for (size_t i = 0; i < 3 * npoint; ++i)
      if (!std::isfinite(points[i])) throw std::invalid_argument("nonfinite grid point");
    ctx.section(true, ctx.metrics.input_ms, [&] {
      cuda_check(
          cudaMemcpyAsync(p.points, points, 3 * npoint * 8, cudaMemcpyHostToDevice, ctx.stream));
      cuda_check(cudaMemsetAsync(ctx.error, 0, sizeof(int), ctx.stream));
    });
    ctx.section(true, ctx.metrics.kernel_ms, [&] {
      ao_kernel<<<blocks(p.jets * npoint * p.nao, 128), 128, 0, ctx.stream>>>(
          p.basis, p.natom, p.nprimitive, p.nao, p.points, npoint, p.jets, p.ao, ctx.error);
      cuda_check(cudaGetLastError());
    });
    if (features) {
      const I stride = npoint * p.nao;
      ctx.section(true, ctx.metrics.library_ms, [&] {
        for (int spin = 0; spin < 2; ++spin)
          gemm(ctx, 'N', 'N', static_cast<int>(npoint), static_cast<int>(p.nao),
               static_cast<int>(p.nao), p.ao, p.density + spin * p.nao * p.nao,
               p.work + spin * 4 * stride, stride, 0, stride, 4, 0);
      });
      ctx.section(true, ctx.metrics.packing_ms, [&] {
        feature_kernel<<<blocks(npoint, 128), 128, 0, ctx.stream>>>(p.ao, p.work, npoint, p.nao,
                                                                    p.features, ctx.error);
        cuda_check(cudaGetLastError());
      });
    }
    int failure = 0;
    ctx.section(true, ctx.metrics.output_ms, [&] {
      cuda_check(
          cudaMemcpyAsync(&failure, ctx.error, sizeof(int), cudaMemcpyDeviceToHost, ctx.stream));
      if (features)
        cuda_check(cudaMemcpyAsync(feature_output, p.features, 13 * npoint * 8,
                                   cudaMemcpyDeviceToHost, ctx.stream));
      if (jet_output)
        cuda_check(cudaMemcpyAsync(jet_output, p.ao, p.jets * npoint * p.nao * 8,
                                   cudaMemcpyDeviceToHost, ctx.stream));
    });
    if (failure) throw std::runtime_error("nonfinite CUDA AO/density output");
  });
}
int grid_cuda_metrics_v1(void* pointer, Metrics* metrics, int* versions, char* error, size_t size) {
  return guarded(error, size, [&] {
    if (!pointer || !metrics || !versions) throw std::invalid_argument("null grid metrics");
    auto& ctx = static_cast<GridPlan*>(pointer)->context;
    std::lock_guard<std::mutex> lock(ctx.mutex);
    ctx.check_device();
    *metrics = ctx.metrics;
    metrics->observed_device_delta = ctx.device_delta();
    cuda_check(cudaRuntimeGetVersion(versions));
    cuda_check(cudaDriverGetVersion(versions + 1));
    blas_check(cublasGetVersion(ctx.handle, versions + 2));
  });
}
}
