/** Bounded FP64 AO spatial jets and spin density contractions on CUDA.
 * Host-built grid tiles are explicit inputs. Native normalized shell data,
 * D matrices, cuBLAS handle, stream and reusable arena are owned by the plan.
 */
#include <climits>
#include <cmath>
#include <new>

#include "../tensor/cuda_runtime.cuh"
#include "grid_task_view.cuh"
#include "vibeqc/vibeqc.h"
#include "xc_point.hpp"

namespace {
using namespace vibeqc_tensor;
#if VIBEQC_TEST_HOOKS
thread_local unsigned fail_next_grid_allocation = 0;
thread_local bool fail_next_grid_runtime = false;
#endif
struct GridPlan {
  Context context;
  bool density_ready = false, density_jets_ready = false;
  size_t natom{}, nprimitive{}, nao{}, capacity{}, jets{}, packed_size{};
  size_t active_capacity{}, last_points{}, last_active{};
  std::uint64_t generation{};
  bool local = false, view_ready = false, features_ready = false;
  bool orbital_enabled = false, orbital_ready = false, use_orbitals = false;
  size_t orbital_capacity[2]{}, orbital_count[2]{}, orbital_tile{};
  unsigned feature_mask = 15;
  double *basis{}, *density{}, *points{}, *ao{}, *work{}, *features{};
  double *local_density{}, *local_potential{}, *potential{};
  double *factors[2]{}, *factor_panel{}, *psi{};
  size_t* ao_ids{};
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
  } catch (const std::bad_alloc& e) {
    error_text(error, size, e.what());
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  } catch (const DeviceAllocationError& e) {
    error_text(error, size, e.what());
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  } catch (const DeviceRuntimeError& e) {
    error_text(error, size, e.what());
    return VIBEQC_STATUS_CUDA_ERROR;
  } catch (const std::exception& e) {
    error_text(error, size, e.what());
    return 1;
  } catch (...) {
    error_text(error, size, "unknown CUDA grid failure");
    return 1;
  }
}

// Gather every local matrix element, including all cross-shell terms. A
// sparse AO mask does not imply a sparse global density matrix.
__global__ void gather_density(const double* global, const size_t* ids, I nao, I active,
                               double* local) {
  for (I i = I(blockIdx.x) * blockDim.x + threadIdx.x; i < 2 * active * active;
       i += I(blockDim.x) * gridDim.x) {
    const I spin = i / (active * active), row = i / active % active, col = i % active;
    local[i] = global[(spin * nao + ids[row]) * nao + ids[col]];
  }
}

// Pack the current occupied column tile, retaining every active AO row and
// eventually every supplied orbital column. This is layout work, not screening.
__global__ void gather_factor(const double* global, const size_t* ids, I active, I rank, I begin,
                              I width, double* packed) {
  for (I i = I(blockIdx.x) * blockDim.x + threadIdx.x; i < active * width;
       i += I(blockDim.x) * gridDim.x) {
    const I row = i / width, column = i % width;
    packed[i] = global[(ids ? ids[row] : row) * rank + begin + column];
  }
}

// Tasks execute serially on the owner's stream, and each map is unique. Thus
// each global element has one writer in a launch and needs no floating atomics.
__global__ void scatter_matrix(const double* local, const size_t* ids, I nao, I active,
                               double* global, int* error) {
  for (I i = I(blockIdx.x) * blockDim.x + threadIdx.x; i < 2 * active * active;
       i += I(blockDim.x) * gridDim.x) {
    const I spin = i / (active * active), row = i / active % active, col = i % active;
    const I transpose = (spin * active + col) * active + row;
    const double value = 0.5 * local[i] + 0.5 * local[transpose];
    const I destination = (spin * nao + ids[row]) * nao + ids[col];
    global[destination] = finite(global[destination] + value, error, 2);
  }
}

}  // namespace

extern "C" {
int grid_cuda_create_v3(int device, int major, int minor, const size_t* dimensions,
                        const double* basis, size_t capacity, unsigned order, size_t expected_bytes,
                        size_t active_capacity, const size_t* orbital_capacity, size_t orbital_tile,
                        unsigned feature_mask, void** output, char* error, size_t size) {
  return guarded(error, size, [&] {
    if (!output) throw std::invalid_argument("null CUDA grid output");
    *output = nullptr;
    if (!dimensions || !basis || !capacity || capacity > INT_MAX || order > 3)
      throw std::invalid_argument("invalid CUDA grid plan");
#if VIBEQC_TEST_HOOKS
    if (fail_next_grid_allocation) {
      const auto failure = fail_next_grid_allocation;
      fail_next_grid_allocation = 0;
      if (failure == 2) throw std::bad_alloc();
      throw DeviceAllocationError("injected CUDA grid allocation failure");
    }
#endif
    auto p = std::make_unique<GridPlan>();
    p->natom = dimensions[0];
    p->nprimitive = dimensions[1];
    p->nao = dimensions[2];
    if (!p->natom || !p->nprimitive || !p->nao || p->natom > INT_MAX || p->nprimitive > INT_MAX ||
        p->nao > INT_MAX)
      throw std::invalid_argument("invalid CUDA grid basis dimensions");
    p->capacity = capacity;
    if (active_capacity > p->nao) throw std::invalid_argument("active AO capacity exceeds basis");
    p->local = active_capacity != 0;
    p->active_capacity = active_capacity ? active_capacity : p->nao;
    p->jets = (order + 1) * (order + 2) * (order + 3) / 6;
    if (!feature_mask || feature_mask > 15) throw std::invalid_argument("invalid feature mask");
    p->feature_mask = feature_mask;
    p->orbital_enabled = orbital_capacity != nullptr;
    if (p->orbital_enabled) {
      if (!orbital_tile || orbital_tile > INT_MAX || orbital_capacity[0] > INT_MAX ||
          orbital_capacity[1] > INT_MAX)
        throw std::invalid_argument("invalid orbital tile/capacity");
      p->orbital_tile = orbital_tile;
      for (int s = 0; s < 2; ++s) p->orbital_capacity[s] = orbital_capacity[s];
    }
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
    const size_t tile = mul(capacity, p->active_capacity);
    if (mul(p->jets, tile) > static_cast<size_t>(INT64_MAX))
      throw std::invalid_argument("CUDA grid index overflow");
    size_t elements =
        add(add(p->packed_size, matrices), add(mul(16, capacity), mul(p->jets + 8, tile)));
    // Selected mode retains global D and V explicitly, plus bounded local D/V
    // and one index map. Dense mode keeps its established allocation contract.
    if (p->local)
      elements =
          add(elements,
              add(matrices, add(mul(4, mul(active_capacity, active_capacity)), active_capacity)));
    if (p->orbital_enabled)
      elements = add(elements, add(mul(p->nao, add(p->orbital_capacity[0], p->orbital_capacity[1])),
                                   mul(add(p->active_capacity, mul(4, capacity)), orbital_tile)));
    static_assert(sizeof(size_t) == sizeof(double), "grid map arena requires 64-bit indices");
    const size_t numeric = mul(8, elements), error_offset = mul(add(numeric, 255) / 256, 256);
    const size_t workspace = add(error_offset, 256), bytes = add(workspace, 4U << 20);
    if (expected_bytes && bytes != expected_bytes)
      throw std::invalid_argument("native/Python grid plan mismatch");
    p->context.prepare(device, major, minor, bytes, error_offset, workspace, 4U << 20, 96U << 20,
                       true);
    p->basis = reinterpret_cast<double*>(p->context.arena);
    p->density = p->basis + p->packed_size;
    p->points = p->density + matrices;
    p->ao = p->points + 3 * capacity;
    p->work = p->ao + p->jets * tile;
    p->features = p->work + 8 * tile;
    if (p->local) {
      p->local_density = p->features + 13 * capacity;
      p->local_potential = p->local_density + 2 * active_capacity * active_capacity;
      p->potential = p->local_potential + 2 * active_capacity * active_capacity;
      p->ao_ids = reinterpret_cast<size_t*>(p->potential + matrices);
    }
    if (p->orbital_enabled) {
      p->factors[0] = p->local ? reinterpret_cast<double*>(p->ao_ids + active_capacity)
                               : p->features + 13 * capacity;
      p->factors[1] = p->factors[0] + p->nao * p->orbital_capacity[0];
      p->factor_panel = p->factors[1] + p->nao * p->orbital_capacity[1];
      p->psi = p->factor_panel + p->active_capacity * orbital_tile;
    }
    p->context.section(true, p->context.metrics.input_ms, [&] {
      cuda_check(cudaMemcpyAsync(p->basis, basis, p->packed_size * 8, cudaMemcpyHostToDevice,
                                 p->context.stream));
      if (p->local) cuda_check(cudaMemsetAsync(p->potential, 0, matrices * 8, p->context.stream));
    });
    *output = p.release();
  });
}
#if VIBEQC_TEST_HOOKS
void grid_cuda_fail_next_allocation_for_test_v1() { fail_next_grid_allocation = 1; }
void grid_cuda_fail_next_host_allocation_for_test_v1() { fail_next_grid_allocation = 2; }
void grid_cuda_fail_next_runtime_for_test_v1() { fail_next_grid_runtime = true; }
#endif
int grid_cuda_create_v2(int device, int major, int minor, const size_t* dimensions,
                        const double* basis, size_t capacity, unsigned order, size_t expected_bytes,
                        size_t active_capacity, void** output, char* error, size_t size) {
  return grid_cuda_create_v3(device, major, minor, dimensions, basis, capacity, order,
                             expected_bytes, active_capacity, nullptr, 0, 15, output, error, size);
}
int grid_cuda_create_v1(int device, int major, int minor, const size_t* dimensions,
                        const double* basis, size_t capacity, unsigned order, size_t expected_bytes,
                        void** output, char* error, size_t size) {
  return grid_cuda_create_v2(device, major, minor, dimensions, basis, capacity, order,
                             expected_bytes, 0, output, error, size);
}
void grid_cuda_destroy_v1(void* pointer) { delete static_cast<GridPlan*>(pointer); }

int grid_cuda_centers_v1(void* pointer, const double* centers, size_t elements, char* error,
                         size_t size) {
  return guarded(error, size, [&] {
    if (!pointer || !centers) throw std::invalid_argument("null CUDA grid centers");
    auto& p = *static_cast<GridPlan*>(pointer);
    auto& ctx = p.context;
    std::lock_guard<std::mutex> lock(ctx.mutex);
    ctx.check_device();
    if (elements != 3 * p.natom) throw std::invalid_argument("CUDA grid center size mismatch");
    for (size_t i = 0; i < elements; ++i)
      if (!std::isfinite(centers[i])) throw std::invalid_argument("nonfinite CUDA grid center");
    p.view_ready = p.features_ready = p.density_jets_ready = false;
    p.density_ready = p.orbital_ready = p.use_orbitals = false;
    ++p.generation;
    ctx.section(true, ctx.metrics.input_ms, [&] {
      cuda_check(cudaMemcpyAsync(p.basis, centers, elements * sizeof(double),
                                 cudaMemcpyHostToDevice, ctx.stream));
    });
  });
}

int grid_cuda_density_v1(void* pointer, const double* density, size_t elements, char* error,
                         size_t size) {
  return guarded(error, size, [&] {
    if (!pointer || !density) throw std::invalid_argument("null CUDA grid density");
    auto& p = *static_cast<GridPlan*>(pointer);
    auto& ctx = p.context;
    std::lock_guard<std::mutex> lock(ctx.mutex);
    ctx.check_device();
#if VIBEQC_TEST_HOOKS
    if (fail_next_grid_runtime) {
      fail_next_grid_runtime = false;
      throw DeviceRuntimeError("injected CUDA grid runtime failure");
    }
#endif
    p.view_ready = false;
    ++p.generation;
    if (elements != 2 * p.nao * p.nao) throw std::invalid_argument("density size mismatch");
    for (size_t i = 0; i < elements; ++i)
      if (!std::isfinite(density[i])) throw std::invalid_argument("nonfinite density");
    // A transport failure must not leave either route ready with partial data.
    p.density_ready = p.orbital_ready = p.use_orbitals = false;
    ctx.section(true, ctx.metrics.input_ms, [&] {
      cuda_check(
          cudaMemcpyAsync(p.density, density, elements * 8, cudaMemcpyHostToDevice, ctx.stream));
      // A density upload starts a new execution, even when D is unchanged.
      // Scatter accumulates across its tasks, never across separate executions.
      if (p.local) cuda_check(cudaMemsetAsync(p.potential, 0, elements * 8, ctx.stream));
    });
    p.density_ready = true;
  });
}

/** Private validated-source boundary. DensitySource owns external C/f and
 * content/generation validation; upload D and its checked B snapshots together.
 * This ABI does not infer compatibility from dimensions or diagonalize D.
 */
int grid_cuda_source_v1(void* pointer, const double* density, size_t elements, const double* alpha,
                        const double* beta, const size_t* counts, int use_orbitals, char* error,
                        size_t size) {
  return guarded(error, size, [&] {
    if (!pointer || !density || !counts || (use_orbitals != 0 && use_orbitals != 1))
      throw std::invalid_argument("invalid CUDA density source");
    auto& p = *static_cast<GridPlan*>(pointer);
    auto& ctx = p.context;
    std::lock_guard<std::mutex> lock(ctx.mutex);
    ctx.check_device();
    if (elements != 2 * p.nao * p.nao) throw std::invalid_argument("density size mismatch");
    const double* factors[2] = {alpha, beta};
    for (size_t i = 0; i < elements; ++i)
      if (!std::isfinite(density[i])) throw std::invalid_argument("nonfinite density");
    for (int s = 0; s < 2; ++s) {
      if (counts[s] > p.orbital_capacity[s] || (counts[s] && (!factors[s] || !use_orbitals)))
        throw std::invalid_argument("factor size/capability mismatch");
      for (size_t i = 0; i < p.nao * counts[s]; ++i)
        if (!std::isfinite(factors[s][i])) throw std::invalid_argument("nonfinite orbital factor");
    }
    if (use_orbitals && !p.orbital_enabled)
      throw std::invalid_argument("orbital route not prepared");
    p.view_ready = p.density_ready = p.orbital_ready = p.use_orbitals = false;
    ++p.generation;
    ctx.section(true, ctx.metrics.input_ms, [&] {
      cuda_check(
          cudaMemcpyAsync(p.density, density, elements * 8, cudaMemcpyHostToDevice, ctx.stream));
      for (int s = 0; s < 2; ++s)
        if (counts[s])
          cuda_check(cudaMemcpyAsync(p.factors[s], factors[s], p.nao * counts[s] * 8,
                                     cudaMemcpyHostToDevice, ctx.stream));
      if (p.local) cuda_check(cudaMemsetAsync(p.potential, 0, elements * 8, ctx.stream));
    });
    p.density_ready = true;
    p.orbital_ready = p.use_orbitals = use_orbitals;
    for (int s = 0; s < 2; ++s) p.orbital_count[s] = counts[s];
  });
}

int grid_cuda_run_selected_v1(void* pointer, const double* points, size_t npoint, int features,
                              const size_t* ao_ids, size_t active, double* feature_output,
                              double* jet_output, char* error, size_t size) {
  return guarded(error, size, [&] {
    if (!pointer || (features != 0 && features != 1))
      throw std::invalid_argument("invalid CUDA grid execution");
    auto& p = *static_cast<GridPlan*>(pointer);
    auto& ctx = p.context;
    std::lock_guard<std::mutex> lock(ctx.mutex);
    ctx.check_device();
    p.view_ready = false;
    ++p.generation;
    if (!p.local) {
      if (ao_ids) throw std::invalid_argument("dense plan does not own AO gather buffers");
      active = p.nao;
    } else {
      if (active > p.active_capacity || (active && !ao_ids))
        throw std::invalid_argument("selected AO map exceeds capacity");
      for (size_t i = 0; i < active; ++i)
        if (ao_ids[i] >= p.nao || (i && ao_ids[i] <= ao_ids[i - 1]))
          throw std::invalid_argument("selected AO map must be sorted unique and in range");
    }
    if (npoint > p.capacity || (npoint && !points) ||
        (features && (((p.feature_mask & 14) && p.jets < 4) || !p.density_ready ||
                      (p.use_orbitals && !p.orbital_ready))))
      throw std::invalid_argument("invalid grid tile/output");
    p.last_points = npoint;
    p.last_active = active;
    p.features_ready = features != 0;
    for (size_t i = 0; i < 3 * npoint; ++i)
      if (!std::isfinite(points[i])) throw std::invalid_argument("nonfinite grid point");
    ctx.section(true, ctx.metrics.input_ms, [&] {
      cuda_check(
          cudaMemcpyAsync(p.points, points, 3 * npoint * 8, cudaMemcpyHostToDevice, ctx.stream));
      cuda_check(cudaMemsetAsync(ctx.error, 0, sizeof(int), ctx.stream));
      if (features) cuda_check(cudaMemsetAsync(p.features, 0, 13 * npoint * 8, ctx.stream));
      if (p.local && active)
        cuda_check(cudaMemcpyAsync(p.ao_ids, ao_ids, active * sizeof(size_t),
                                   cudaMemcpyHostToDevice, ctx.stream));
    });
    // Even an empty point tile publishes its new map and clears prior errors;
    // a borrowed view must never expose the previous task's AO labels.
    if (!npoint) {
      p.view_ready = true;
      return;
    }
    if (active)
      ctx.section(true, ctx.metrics.kernel_ms, [&] {
        ao_kernel<<<blocks(p.jets * npoint * active, 128), 128, 0, ctx.stream>>>(
            p.basis, p.natom, p.nprimitive, active, p.points, npoint, p.jets, p.ao, ctx.error,
            p.local ? p.ao_ids : nullptr);
        cuda_check(cudaGetLastError());
      });
    if (features && p.use_orbitals) {
      const I ao_stride = npoint * active;
      for (int spin = 0; spin < 2; ++spin) {
        for (size_t begin = 0; active && begin < p.orbital_count[spin]; begin += p.orbital_tile) {
          const size_t width = std::min(p.orbital_tile, p.orbital_count[spin] - begin);
          ctx.section(true, ctx.metrics.packing_ms, [&] {
            gather_factor<<<blocks(active * width, 128), 128, 0, ctx.stream>>>(
                p.factors[spin], p.local ? p.ao_ids : nullptr, active, p.orbital_count[spin], begin,
                width, p.factor_panel);
            cuda_check(cudaGetLastError());
          });
          const I psi_stride = npoint * width;
          const int first = (p.feature_mask & 7) ? 0 : 1;
          const int count = (p.feature_mask & 14) ? 4 - first : 1;
          ctx.section(true, ctx.metrics.library_ms, [&] {
            gemm(ctx, 'N', 'N', npoint, width, active, p.ao + first * ao_stride, p.factor_panel,
                 p.psi + first * psi_stride, ao_stride, 0, psi_stride, count, 0);
          });
          ctx.section(true, ctx.metrics.packing_ms, [&] {
            orbital_feature_kernel<<<blocks(npoint, 128), 128, 0, ctx.stream>>>(
                p.psi, npoint, width, spin, p.features, ctx.error, p.feature_mask);
            cuda_check(cudaGetLastError());
          });
        }
      }
      if (p.feature_mask & 4)
        ctx.section(true, ctx.metrics.packing_ms, [&] {
          finish_orbital_sigma<<<blocks(npoint, 128), 128, 0, ctx.stream>>>(p.features, npoint,
                                                                            ctx.error);
          cuda_check(cudaGetLastError());
        });
    } else if (features) {
      const I stride = npoint * active;
      if (p.local && active)
        ctx.section(true, ctx.metrics.packing_ms, [&] {
          gather_density<<<blocks(2 * active * active, 128), 128, 0, ctx.stream>>>(
              p.density, p.ao_ids, p.nao, active, p.local_density);
          cuda_check(cudaGetLastError());
        });
      if (active)
        ctx.section(true, ctx.metrics.library_ms, [&] {
          const double* density = p.local ? p.local_density : p.density;
          const int first = (p.feature_mask & 7) ? 0 : 1;
          const int count = (p.feature_mask & 8) ? 4 - first : 1;
          for (int spin = 0; spin < 2; ++spin)
            gemm(ctx, 'N', 'N', static_cast<int>(npoint), static_cast<int>(active),
                 static_cast<int>(active), p.ao + first * stride, density + spin * active * active,
                 p.work + (spin * 4 + first) * stride, stride, 0, stride, count, 0);
        });
      ctx.section(true, ctx.metrics.packing_ms, [&] {
        feature_kernel<<<blocks(npoint, 128), 128, 0, ctx.stream>>>(
            p.ao, p.work, npoint, active, p.features, ctx.error, p.feature_mask);
        cuda_check(cudaGetLastError());
      });
    }
    int failure = 0;
    ctx.section(true, ctx.metrics.output_ms, [&] {
      cuda_check(
          cudaMemcpyAsync(&failure, ctx.error, sizeof(int), cudaMemcpyDeviceToHost, ctx.stream));
      if (features && feature_output)
        cuda_check(cudaMemcpyAsync(feature_output, p.features, 13 * npoint * 8,
                                   cudaMemcpyDeviceToHost, ctx.stream));
      if (jet_output && active)
        cuda_check(cudaMemcpyAsync(jet_output, p.ao, p.jets * npoint * active * 8,
                                   cudaMemcpyDeviceToHost, ctx.stream));
    });
    if (failure) throw std::runtime_error("nonfinite CUDA AO/density output");
    p.density_jets_ready = features && !p.use_orbitals;
    p.view_ready = true;
  });
}
int grid_cuda_run_v1(void* pointer, const double* points, size_t npoint, int features,
                     double* feature_output, double* jet_output, char* error, size_t size) {
  return grid_cuda_run_selected_v1(pointer, points, npoint, features, nullptr, 0, feature_output,
                                   jet_output, error, size);
}

int grid_cuda_view_v1(void* pointer, vibeqc::dft::GridTaskView* output, char* error, size_t size) {
  return guarded(error, size, [&] {
    if (!pointer || !output) throw std::invalid_argument("null grid view");
    auto& p = *static_cast<GridPlan*>(pointer);
    std::lock_guard<std::mutex> lock(p.context.mutex);
    p.context.check_device();
    if (!p.local || !p.view_ready) throw std::invalid_argument("local grid view is not ready");
    *output = {1,
               p.generation,
               p.last_points,
               p.nao,
               p.last_active,
               p.jets,
               p.ao_ids,
               p.points,
               p.ao,
               p.features_ready ? p.features : nullptr,
               p.local_potential,
               p.potential,
               p.context.stream,
               p.context.error};
  });
}

int grid_cuda_basis_v1(void* pointer, vibeqc::dft::GridBasisView* output, char* error,
                       size_t size) {
  return guarded(error, size, [&] {
    if (!pointer || !output) throw std::invalid_argument("null grid basis view");
    auto& p = *static_cast<GridPlan*>(pointer);
    std::lock_guard<std::mutex> lock(p.context.mutex);
    p.context.check_device();
    *output = {1, p.natom, p.nprimitive, p.nao, p.basis, p.context.stream};
  });
}

// The task lease owns the lifetime/stream. This optional extension leaves the
// v1 view ABI intact and refuses orbital tiles or overwritten XC work storage.
int grid_cuda_density_jets_v1(void* pointer, std::uint64_t generation, unsigned jets,
                              const double** output, char* error, size_t size) {
  return guarded(error, size, [&] {
    if (!pointer || !output) throw std::invalid_argument("null contracted AO view");
    auto& p = *static_cast<GridPlan*>(pointer);
    std::lock_guard<std::mutex> lock(p.context.mutex);
    p.context.check_device();
    if (!p.local || !p.view_ready || !p.features_ready || !p.density_jets_ready || p.use_orbitals ||
        generation != p.generation || (jets != 1 && jets != 4) || !(p.feature_mask & 7) ||
        (jets == 4 && !(p.feature_mask & 8)))
      throw std::invalid_argument("contracted AO jets are unavailable");
    *output = p.work;
  });
}

int grid_cuda_xc_v2(void* pointer, std::uint64_t generation, int pbe, int restricted, int interior,
                    const double* weights, size_t npoint, double* integrals, char* error,
                    size_t size) {
  return guarded(error, size, [&] {
    if (!pointer || !integrals || (pbe != 0 && pbe != 1) || (restricted != 0 && restricted != 1) ||
        interior != 1 || (npoint && !weights))
      throw std::invalid_argument("invalid CUDA XC task");
    auto& p = *static_cast<GridPlan*>(pointer);
    auto& ctx = p.context;
    std::lock_guard<std::mutex> lock(ctx.mutex);
    ctx.check_device();
    const unsigned required_features = pbe ? 3U : 1U;
    if (!p.local || !p.view_ready || !p.features_ready || generation != p.generation ||
        npoint != p.last_points || p.jets < (pbe ? 4U : 1U) ||
        (p.feature_mask & required_features) != required_features)
      throw std::invalid_argument("stale or incompatible CUDA XC task");
    // A local plan has active_capacity>=1, so its eight work panels leave at
    // least three doubles after the capacity-sized weight upload.
    p.density_jets_ready = false;
    double* device_weights = p.work;
    double* device_integrals = p.work + p.capacity;
    const size_t matrix_elements = 2 * p.last_active * p.last_active;
    ctx.section(true, ctx.metrics.input_ms, [&] {
      for (size_t i = 0; i < npoint; ++i)
        if (!std::isfinite(weights[i])) throw std::invalid_argument("nonfinite XC weight");
      cuda_check(cudaMemsetAsync(ctx.error, 0, sizeof(int), ctx.stream));
      cuda_check(cudaMemsetAsync(device_integrals, 0, 3 * sizeof(double), ctx.stream));
      if (matrix_elements)
        cuda_check(
            cudaMemsetAsync(p.local_potential, 0, matrix_elements * sizeof(double), ctx.stream));
      if (npoint)
        cuda_check(cudaMemcpyAsync(device_weights, weights, npoint * sizeof(double),
                                   cudaMemcpyHostToDevice, ctx.stream));
    });
    if (npoint) {
      ctx.section(true, ctx.metrics.kernel_ms, [&] {
        xc_integrals_kernel<<<1, 1, 0, ctx.stream>>>(pbe != 0, restricted != 0, p.features,
                                                     device_weights, npoint, device_integrals,
                                                     ctx.error);
        cuda_check(cudaGetLastError());
        if (matrix_elements) {
          xc_local_potential_kernel<<<blocks(matrix_elements, 128), 128, 0, ctx.stream>>>(
              pbe != 0, restricted != 0, p.features, p.ao, device_weights, npoint, p.last_active,
              p.local_potential, ctx.error);
          cuda_check(cudaGetLastError());
        }
      });
    }
    int failure = 0;
    ctx.section(true, ctx.metrics.output_ms, [&] {
      cuda_check(cudaMemcpyAsync(integrals, device_integrals, 3 * sizeof(double),
                                 cudaMemcpyDeviceToHost, ctx.stream));
      cuda_check(
          cudaMemcpyAsync(&failure, ctx.error, sizeof(int), cudaMemcpyDeviceToHost, ctx.stream));
    });
    if (failure) throw std::runtime_error("invalid or nonfinite CUDA XC output");
  });
}

/** Optional host input/output serves diagnostics. A native XC consumer writes
 * local_potential on the borrowed stream and passes null host buffers. */
int grid_cuda_scatter_v1(void* pointer, std::uint64_t generation, const double* host_local,
                         int reset, double* host_global, char* error, size_t size) {
  return guarded(error, size, [&] {
    if (!pointer || (reset != 0 && reset != 1)) throw std::invalid_argument("invalid grid scatter");
    auto& p = *static_cast<GridPlan*>(pointer);
    auto& ctx = p.context;
    std::lock_guard<std::mutex> lock(ctx.mutex);
    ctx.check_device();
    if (!p.local || !p.view_ready || generation != p.generation)
      throw std::invalid_argument("stale local grid view");
    const size_t count = 2 * p.last_active * p.last_active;
    ctx.section(true, ctx.metrics.input_ms, [&] {
      cuda_check(cudaMemsetAsync(ctx.error, 0, sizeof(int), ctx.stream));
      if (reset) cuda_check(cudaMemsetAsync(p.potential, 0, 2 * p.nao * p.nao * 8, ctx.stream));
      if (host_local && count) {
        for (size_t i = 0; i < count; ++i)
          if (!std::isfinite(host_local[i])) throw std::invalid_argument("nonfinite local matrix");
        cuda_check(cudaMemcpyAsync(p.local_potential, host_local, count * 8, cudaMemcpyHostToDevice,
                                   ctx.stream));
      }
    });
    if (count)
      ctx.section(true, ctx.metrics.packing_ms, [&] {
        scatter_matrix<<<blocks(count, 128), 128, 0, ctx.stream>>>(
            p.local_potential, p.ao_ids, p.nao, p.last_active, p.potential, ctx.error);
        cuda_check(cudaGetLastError());
      });
    int failure = 0;
    ctx.section(true, ctx.metrics.output_ms, [&] {
      cuda_check(
          cudaMemcpyAsync(&failure, ctx.error, sizeof(int), cudaMemcpyDeviceToHost, ctx.stream));
      if (host_global)
        cuda_check(cudaMemcpyAsync(host_global, p.potential, 2 * p.nao * p.nao * 8,
                                   cudaMemcpyDeviceToHost, ctx.stream));
    });
    if (failure) throw std::runtime_error("nonfinite local grid consumer output");
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
