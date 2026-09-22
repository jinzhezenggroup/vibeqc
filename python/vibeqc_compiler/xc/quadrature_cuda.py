"""Bounded CUDA molecular quadrature, using the existing Becke scalar graph.

Geometry distances are produced once, and point-center distances once per tile.
Each (point, atom) worker accumulates logs in reference pair order without a
point-by-atom-by-atom tensor or atom-count-dependent thread-local storage.
Native code owns allocations, quadrature rule inputs, launches and downloads.
"""

from vibeqc_compiler.integral.scalar_c import ScalarCEmitter

from .grid_response_ir import grid_response_program

_LAYOUT = r"""#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace vibeqc::generated::quadrature {
inline size_t product(size_t a, size_t b) {
  if (b && a > std::numeric_limits<size_t>::max() / b)
    throw std::overflow_error("CUDA quadrature size overflow");
  return a * b;
}
inline size_t sum(size_t a, size_t b) {
  if (a > std::numeric_limits<size_t>::max() - b)
    throw std::overflow_error("CUDA quadrature size overflow");
  return a + b;
}
struct Layout {
  size_t atoms, points, tile, rules, geometry, distances, logs, xyz, weights;
  size_t doubles, device_bytes;
};
// 4096 points bounds launch overhead and O(tile * atoms) scratch. The small
// Gauss-Legendre tables use their validated maximum capacity in every plan,
// so shape-only callers need no grid-version-specific ABI or GPU query.
inline Layout layout(size_t atoms, size_t points) {
  if (!atoms || !points || atoms > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("invalid CUDA quadrature shape");
  Layout l{};
  l.atoms = atoms;
  l.points = points;
  l.tile = std::min(points, size_t{4096});
  l.rules = product(4, atoms); // xyz centers plus resolved radii
  l.geometry = sum(l.rules, 2 * (512 + 256));
  l.distances = sum(l.geometry, product(atoms, atoms));
  l.logs = sum(l.distances, product(l.tile, atoms));
  l.xyz = sum(l.logs, product(l.tile, atoms));
  l.weights = sum(l.xyz, product(3, l.tile));
  l.doubles = sum(l.weights, l.tile);
  l.device_bytes = sum(product(l.doubles, sizeof(double)), sizeof(int));
  // Preflight export indexing too, before any CUDA operation/allocation.
  (void)product(points, 3 * sizeof(double));
  return l;
}
inline unsigned blocks(size_t count) {
  return static_cast<unsigned>(std::min(size_t{65535}, 1 + (count - 1) / 128));
}
} // namespace vibeqc::generated::quadrature

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#include <cmath>
namespace vibeqc::generated::quadrature {
"""

_KERNELS = r"""
__device__ inline double distance(const double* a, const double* b) {
  return hypot(hypot(a[0] - b[0], a[1] - b[1]), a[2] - b[2]);
}
// Compute each unordered center distance once, before processing any points.
__global__ void geometry_kernel(const double* centers, size_t na, double* separation) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < na * na;
       i += size_t(blockDim.x) * gridDim.x) {
    const size_t a = i / na, b = i % na;
    if (b > a) continue;
    const double value = distance(centers + 3 * a, centers + 3 * b);
    separation[a * na + b] = separation[b * na + a] = value;
  }
}
// Atom/radial/polar/azimuth order is the public derivative-export contract.
__global__ void points_kernel(size_t begin, size_t count, size_t nr, size_t nz, size_t nphi,
                              const double* centers, const double* radii, const double* rn,
                              const double* rw, const double* zn, const double* zw,
                              double* xyz, double* weights) {
  constexpr double pi = 3.141592653589793238462643383279502884;
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(blockDim.x) * gridDim.x) {
    size_t index = begin + i;
    const size_t phi_index = index % nphi;
    index /= nphi;
    const size_t z = index % nz;
    index /= nz;
    const size_t radial = index % nr, owner = index / nr;
    const double t = 0.5 * (rn[radial] + 1.0);
    const double r = radii[owner] * t / (1.0 - t);
    const double wr = 0.5 * rw[radial] * radii[owner] * r * r / ((1.0 - t) * (1.0 - t));
    const double ring = sqrt(fmax(0.0, 1.0 - zn[z] * zn[z]));
    const double phi = 2.0 * pi * phi_index / nphi;
    xyz[3*i] = centers[3*owner] + r * ring * cos(phi);
    xyz[3*i+1] = centers[3*owner+1] + r * ring * sin(phi);
    xyz[3*i+2] = centers[3*owner+2] + r * zn[z];
    weights[i] = wr * zw[z] * (2.0 * pi / nphi);
  }
}
// Atom-major tiles give adjacent point lanes coalesced scratch access.
__global__ void distances_kernel(size_t count, size_t na, const double* xyz,
                                 const double* centers, double* distances) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count * na;
       i += size_t(blockDim.x) * gridDim.x)
    distances[i] = distance(xyz + 3 * (i % count), centers + 3 * (i / count));
}
template <unsigned Iterations>
__global__ void partition_kernel(size_t count, size_t na, double tolerance,
                                 const double* distances, const double* separation,
                                 double* logs) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count * na;
       i += size_t(blockDim.x) * gridDim.x) {
    const size_t a = i / count, point = i % count;
    double value = 0.0;
    // b<a contributions precede b>a, as in the scalar triangular pair loop.
    // Keep the original orientation for log1p(-pair); reversing mu can lose
    // tails through cancellation and change normalized weights near centers.
    for (size_t b = 0; b < na; ++b) {
      if (a == b) continue;
      const size_t hi = a > b ? a : b, lo = a > b ? b : a;
      const double sep = separation[hi * na + lo];
      const double mu = sep > tolerance
          ? fmin(1.0, fmax(-1.0, (distances[hi * count + point] -
                                  distances[lo * count + point]) / sep)) : 0.0;
      const double pair = fmin(1.0, fmax(0.0, becke<Iterations>(mu)));
      value += a > b ? log(pair) : log1p(-pair);
    }
    logs[i] = value;
  }
}
__global__ void normalize_kernel(size_t begin, size_t count, size_t per_atom, size_t na,
                                 const double* logs, double* weights, int* invalid) {
  for (size_t p = size_t(blockIdx.x) * blockDim.x + threadIdx.x; p < count;
       p += size_t(blockDim.x) * gridDim.x) {
    double maximum = -INFINITY;
    for (size_t a = 0; a < na; ++a) maximum = fmax(maximum, logs[a * count + p]);
    double total = 0.0;
    for (size_t a = 0; a < na; ++a) total += exp(logs[a * count + p] - maximum);
    const size_t owner = (begin + p) / per_atom;
    const double weight = weights[p] * (exp(logs[owner * count + p] - maximum) / total);
    if (!(total > 0.0) || !isfinite(total) || !isfinite(weight)) atomicExch(invalid, 1);
    weights[p] = weight;
  }
}
inline void launch_partition(unsigned iterations, size_t count, size_t na, double tolerance,
                             const double* distances, const double* separation,
                             double* logs, cudaStream_t stream) {
  switch (iterations) {
@PARTITION_CASES@
    default: throw std::invalid_argument("unsupported Becke iteration count");
  }
}
} // namespace vibeqc::generated::quadrature
#endif
"""


def emit_quadrature_cuda() -> str:
    """Emit shape accounting and kernels from the existing scalar Becke IR."""
    lines = [
        _LAYOUT,
        "template <unsigned Iterations> __device__ double becke(double mu);",
    ]
    cases = []
    for iterations in range(1, 6):
        program = grid_response_program("becke", iterations)
        emitter = ScalarCEmitter(program.graph, {"mu": "mu"})
        emitter.emit((program.roots[0],))
        lines.extend(
            [
                f"// Becke graph: {program.identity}",
                f"template <> __device__ inline double becke<{iterations}>(double mu) {{",
                *emitter.lines,
                f"return {emitter.reference(program.roots[0])};",
                "}",
            ]
        )
        cases.append(
            f"    case {iterations}: partition_kernel<{iterations}><<<blocks(count * na), 128, 0, stream>>>"
            "(count, na, tolerance, distances, separation, logs); break;"
        )
    lines.append(_KERNELS.replace("@PARTITION_CASES@", "\n".join(cases)))
    return "\n".join(lines)
