#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "api/error.hpp"
#include "generated_ecp_ao.cuh"
#include "integrals/ecp_cuda.hpp"
#include "molecule/basis.hpp"
#include "runtime/resource_cuda.cuh"

namespace vibeqc::integrals {
namespace {
struct Primitive {
  double exponent, coefficient;
};
struct Component {
  unsigned x, y, z;
  double coefficient;
};
struct AO {
  int atom, primitive_offset, primitive_count, term_count;
  double x, y, z;
  Component components[3];
};
struct Four {
  double v[4];
};
__global__ void evaluate_ao(const AO* aos, const Primitive* primitives,
                            const EcpSpherePoint* sphere, const EcpRadialPoint* radial, int n,
                            int nq, int nr, double cx, double cy, double cz, bool derivatives,
                            Four* values) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nr * n * nq) return;
  const int q = i % nq, a = (i / nq) % n, r = i / (nq * n);
  values[i] = generated::ecp_evaluate_ao<Four>(aos[a], primitives, sphere[q], radial[r].r, cx, cy,
                                               cz, derivatives);
}
__global__ void project(const Four* values, const EcpSpherePoint* sphere, int n, int nq, int nr,
                        bool derivatives, Four* projections) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nr * n * 9) return;
  const int m = i % 9, a = (i / 9) % n, r = i / (9 * n);
  projections[i] = generated::ecp_project(values + (r * n + a) * nq, sphere, nq, m, derivatives);
}
__global__ void contract(const AO* aos, const core::EcpTerm* terms, int nt,
                         const EcpSpherePoint* sphere, const EcpRadialPoint* radii,
                         const Four* values, const Four* projections, int n, int nq, int nr,
                         int center, int ncoord, double* output) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nr * n * n) return;
  const int b = i % n, a = (i / n) % n, r = i / (n * n);
  if (b > a) return;
  const int size = n * n, stride = size * (1 + ncoord);
  double parts[2][10];
  generated::ecp_contract(terms, nt, radii[r], sphere, nq, center, values + (r * n + a) * nq,
                          values + (r * n + b) * nq, projections + (r * n + a) * 9,
                          projections + (r * n + b) * 9, ncoord != 0, parts);
  for (int part = 0; part < 2; ++part)
    for (int transpose = 0; transpose < (a == b ? 1 : 2); ++transpose) {
      const int item = transpose ? b * n + a : a * n + b;
      double* out = output + part * stride;
      atomicAdd(out + item, parts[part][0]);
      if (ncoord)
        for (int d = 0; d < 3; ++d) {
          atomicAdd(out + (1 + aos[a].atom * 3 + d) * size + item, parts[part][d + 1]);
          atomicAdd(out + (1 + aos[b].atom * 3 + d) * size + item, parts[part][d + 4]);
          atomicAdd(out + (1 + center * 3 + d) * size + item, parts[part][d + 7]);
        }
    }
}
__global__ void consume(const double* output, int size, int ncoord, double* hcore,
                        const double* density, double* forces) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = size * (1 + ncoord);
  if (hcore && i < size)
    hcore[i] = generated::ecp_add_operator(hcore[i], output[i], output[stride + i]);
  if (forces && i < ncoord) {
    forces[i] += generated::ecp_force_component(output + (1 + i) * size,
                                                output + stride + (1 + i) * size, density, size);
  }
}
struct CudaFailure {
  cudaError_t status;
};
void check(cudaError_t status) {
  if (status != cudaSuccess) throw CudaFailure{status};
}
// ECP callers may own resources on another device. Keep the thread's prior
// selection alive through cleanup, including failures after partial staging.
struct DeviceGuard {
  int previous{};
  DeviceGuard() { check(cudaGetDevice(&previous)); }
  ~DeviceGuard() { (void)cudaSetDevice(previous); }
};
vibeqc_status map_ecp_exception(std::string& detail) {
  try {
    throw;
  } catch (const CudaFailure& error) {
    detail = cudaGetErrorString(error.status);
    return error.status == cudaErrorMemoryAllocation ? VIBEQC_STATUS_OUT_OF_MEMORY
                                                     : VIBEQC_STATUS_CUDA_ERROR;
  } catch (...) {
    return api::map_exception(&detail);
  }
}
__global__ void check_grid(const double* coarse, const double* fine, int size, int stride,
                           int* failed) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 2 * stride) return;
  const double tolerance = (i % stride) < size ? 2e-9 : 2e-8;
  if (!isfinite(fine[i]) || !isfinite(coarse[i]) || fabs(coarse[i] - fine[i]) > tolerance)
    atomicExch(failed, 1);
}
struct Arena {
  cudaStream_t stream;
  std::vector<void*> allocations;
  ~Arena() {
    (void)cudaStreamSynchronize(stream);
    for (auto p : allocations) (void)runtime::resource_cuda_free(p);
  }
  template <class T>
  T* allocate(std::size_t n, const T* source = nullptr) {
    void* pointer = nullptr;
    check(runtime::resource_cuda_malloc(&pointer, n * sizeof(T)));
    try {
      allocations.push_back(pointer);
    } catch (...) {
      (void)runtime::resource_cuda_free(pointer);
      throw;
    }
    if (source)
      check(cudaMemcpyAsync(pointer, source, n * sizeof(T), cudaMemcpyHostToDevice, stream));
    return static_cast<T*>(pointer);
  }
};
void run(const core::System& system, unsigned radial, unsigned polar, bool derivatives,
         cudaStream_t stream, EcpData* export_result, double* hcore, const double* density,
         double* forces, bool convergence = false) {
  const auto n = molecule::ao_count(system);
  if (n == 0 || n > 256 || system.atoms.size() > 128 || system.ecp_terms.empty())
    throw std::invalid_argument("ECP CUDA requires 1..256 AOs, at most 128 atoms and ECP terms");
  std::vector<AO> aos;
  std::vector<Primitive> primitives;
  for (const auto& shell : system.shells) {
    const auto& atom = system.atoms[shell.atom_index];
    const auto offset = primitives.size();
    for (const auto& p : shell.primitives) primitives.push_back({p.exponent, p.coefficient});
    for (const auto& expansion :
         molecule::ao_expansions(shell.angular_momentum, system.basis_representation)) {
      if (expansion.size() > 3)
        throw std::invalid_argument("ECP AO expansion exceeds validated domain");
      AO ao{static_cast<int>(shell.atom_index),
            static_cast<int>(offset),
            static_cast<int>(shell.primitives.size()),
            static_cast<int>(expansion.size()),
            atom.position[0],
            atom.position[1],
            atom.position[2],
            {}};
      for (unsigned t = 0; t < expansion.size(); ++t) {
        const auto& v = expansion[t];
        ao.components[t] = {
            v.component[0], v.component[1], v.component[2],
            v.coefficient * molecule::cartesian_component_normalization(v.component)};
      }
      aos.push_back(ao);
    }
  }
  const int ncoord = derivatives ? system.atoms.size() * 3 : 0;
  const std::size_t size = n * n, stride = size * (1 + ncoord);
  // One radial shell bounds staging independent of the radial grid length.
  Arena arena{stream, {}};
  auto daos = arena.allocate(aos.size(), aos.data());
  auto dprimitives = arena.allocate(primitives.size(), primitives.data());
  auto dterms = arena.allocate(system.ecp_terms.size(), system.ecp_terms.data());
  auto output = arena.allocate<double>(2 * stride);
  auto evaluate_grid = [&](unsigned nr, unsigned na, double* destination) {
    std::vector<EcpRadialPoint> radial_grid;
    std::vector<EcpSpherePoint> sphere;
    ecp_quadrature(nr, na, radial_grid, sphere);
    const int nq = sphere.size();
    Arena grid{stream, {}};
    auto dsphere = grid.allocate(sphere.size(), sphere.data());
    auto dradii = grid.allocate(radial_grid.size(), radial_grid.data());
    auto values = grid.allocate<Four>(n * nq);
    auto projections = grid.allocate<Four>(n * 9);
    check(cudaMemsetAsync(destination, 0, 2 * stride * sizeof(double), stream));
    for (unsigned c = 0; c < system.atoms.size(); ++c) {
      const auto& atom = system.atoms[c];
      if (!atom.ecp_core) continue;
      for (unsigned r = 0; r < nr; ++r) {
        evaluate_ao<<<(n * nq + 127) / 128, 128, 0, stream>>>(
            daos, dprimitives, dsphere, dradii + r, n, nq, 1, atom.position[0], atom.position[1],
            atom.position[2], derivatives, values);
        project<<<(n * 9 + 127) / 128, 128, 0, stream>>>(values, dsphere, n, nq, 1, derivatives,
                                                         projections);
        contract<<<(size + 127) / 128, 128, 0, stream>>>(daos, dterms, system.ecp_terms.size(),
                                                         dsphere, dradii + r, values, projections,
                                                         n, nq, 1, c, ncoord, destination);
        check(cudaPeekAtLastError());
      }
    }
    check(cudaStreamSynchronize(stream));
  };
  if (convergence) {
    auto coarse = arena.allocate<double>(2 * stride);
    evaluate_grid(160, 32, coarse);
    evaluate_grid(224, 44, output);
    auto failed = arena.allocate<int>(1);
    check(cudaMemsetAsync(failed, 0, sizeof(int), stream));
    check_grid<<<(2 * stride + 127) / 128, 128, 0, stream>>>(coarse, output, size, stride, failed);
    check(cudaPeekAtLastError());
    int host_failed = 0;
    check(cudaMemcpyAsync(&host_failed, failed, sizeof(int), cudaMemcpyDeviceToHost, stream));
    check(cudaStreamSynchronize(stream));
    if (host_failed)
      throw std::runtime_error(
          "ECP quadrature convergence gate failed; input outside validated execution domain");
  } else {
    evaluate_grid(radial, polar, output);
  }
  if (hcore || forces) {
    consume<<<(std::max(size, static_cast<std::size_t>(ncoord)) + 127) / 128, 128, 0, stream>>>(
        output, size, ncoord, hcore, density, forces);
    check(cudaPeekAtLastError());
  }
  if (export_result) {
    auto& out = *export_result;
    out = EcpData{n,
                  static_cast<std::size_t>(ncoord),
                  std::vector<double>(size),
                  std::vector<double>(size),
                  std::vector<double>(size * ncoord),
                  std::vector<double>(size * ncoord)};
    check(cudaMemcpyAsync(out.local.data(), output, size * sizeof(double), cudaMemcpyDeviceToHost,
                          stream));
    check(cudaMemcpyAsync(out.nonlocal.data(), output + stride, size * sizeof(double),
                          cudaMemcpyDeviceToHost, stream));
    if (ncoord) {
      check(cudaMemcpyAsync(out.local_derivative.data(), output + size,
                            size * ncoord * sizeof(double), cudaMemcpyDeviceToHost, stream));
      check(cudaMemcpyAsync(out.nonlocal_derivative.data(), output + stride + size,
                            size * ncoord * sizeof(double), cudaMemcpyDeviceToHost, stream));
    }
  }
  check(cudaStreamSynchronize(stream));
}
}  // namespace
vibeqc_status ecp_integrals_cuda(int device, const core::System& system, unsigned radial,
                                 unsigned polar, bool derivatives, EcpData& output,
                                 std::string& detail, bool convergence) {
  try {
    DeviceGuard device_guard;
    check(cudaSetDevice(device));
    run(system, radial, polar, derivatives, nullptr, &output, nullptr, nullptr, nullptr,
        convergence);
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return map_ecp_exception(detail);
  }
}
vibeqc_status add_ecp_cuda(int device, const core::System& system, void* stream, double* hcore,
                           const double* density, double* forces, std::string& detail) {
  if (system.ecp_terms.empty()) return VIBEQC_STATUS_SUCCESS;
  try {
    if (forces && !density) throw std::invalid_argument("ECP force requires density weights");
    DeviceGuard device_guard;
    check(cudaSetDevice(device));
    run(system, 160, 32, forces != nullptr, static_cast<cudaStream_t>(stream), nullptr, hcore,
        density, forces, true);
    return VIBEQC_STATUS_SUCCESS;
  } catch (...) {
    return map_ecp_exception(detail);
  }
}
}  // namespace vibeqc::integrals
