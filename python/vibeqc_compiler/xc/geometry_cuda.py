"""CUDA AO translation pullback and Becke local partials from shared graphs."""

from vibeqc_compiler.dft.ao import jet_indices
from vibeqc_compiler.dft.ao_cuda import emit_grid_policy
from vibeqc_compiler.integral.scalar_c import ScalarCEmitter

from .coefficients import jet_pullback_program
from .grid_native import emit_grid_partials


def emit_geometry_cuda(*, pbe, iterations=3):
    """Lower AO bilinear AD; the caller supplies exact SCF point coefficients."""
    if type(pbe) is not bool:
        raise TypeError("geometry lowering requires a boolean PBE flag")
    program = jet_pullback_program("gga" if pbe else "lda")
    variables = {
        **{f"c{j}": f"c[{j}]" for j in range(len(program.roots))},
        **{f"{leg}{j}": f"w[{j}]" for leg in "xy" for j in range(len(program.roots))},
    }
    emitter = ScalarCEmitter(program.graph, variables)
    emitter.emit(program.roots)
    domain = jet_indices(1 if pbe else 0)
    lookup = jet_indices(2 if pbe else 1)
    shifts = []
    for index in domain:
        row = []
        for k in range(3):
            shifted = list(index)
            shifted[k] += 1
            row.append(lookup.index(tuple(shifted)))
        shifts.append("{" + ",".join(map(str, row)) + "}")
    return "\n".join(
        [
            '#include "dft/grid_response_adjoint.hpp"',
            emit_grid_partials(iterations, device=True),
            f"constexpr bool stationary_pbe = {'true' if pbe else 'false'};",
            f"constexpr unsigned stationary_jets = {len(domain)};",
            f"__device__ __constant__ unsigned stationary_shift[{len(domain)}][3] = {{{','.join(shifts)}}};",
            "__device__ void ao_pullback(const double* c, const double* w, double* out) {",
            *emitter.lines,
            *(
                f"out[{j}] = {emitter.reference(r)};"
                for j, r in enumerate(program.roots)
            ),
            "}",
            "",
        ]
    )


def _emit_pullback_namespace(*, pbe, namespace):
    """Emit one namespaced compiler-owned AO translation pullback."""
    program = jet_pullback_program("gga" if pbe else "lda")
    variables = {
        **{f"c{j}": f"c[{j}]" for j in range(len(program.roots))},
        **{f"{leg}{j}": f"w[{j}]" for leg in "xy" for j in range(len(program.roots))},
    }
    emitter = ScalarCEmitter(program.graph, variables)
    emitter.emit(program.roots)
    domain = jet_indices(1 if pbe else 0)
    lookup = jet_indices(2 if pbe else 1)
    shifts = []
    for index in domain:
        row = []
        for k in range(3):
            shifted = list(index)
            shifted[k] += 1
            row.append(lookup.index(tuple(shifted)))
        shifts.append("{" + ",".join(map(str, row)) + "}")
    return "\n".join(
        [
            f"namespace {namespace} {{",
            f"constexpr unsigned jets = {len(domain)};",
            f"constexpr unsigned ao_jets = {len(lookup)};",
            f"__device__ __constant__ unsigned shift[{len(domain)}][3] = {{{','.join(shifts)}}};",
            "__device__ void apply(const double* c, const double* w, double* out) {",
            *emitter.lines,
            *(
                f"out[{j}] = {emitter.reference(root)};"
                for j, root in enumerate(program.roots)
            ),
            "}",
            "}",
        ]
    )


def emit_native_geometry_cuda():
    """Emit the build-time native XC force TU from shared AO/grid-response graphs."""
    partials = []
    for iterations in range(1, 6):
        partials.extend(
            [
                f"namespace grid_iter_{iterations} {{",
                emit_grid_partials(iterations, device=True),
                "}",
            ]
        )
    runtime = r"""
template <class Pair>
__device__ bool contract_grid_impl(
    const double* point, const double* centers, size_t na, size_t owner, double seed,
    double* gradient, double* logs, double* products, double* bar_product,
    double* bar_distance, size_t* zeros, std::array<double, 4>* distances, Pair pair) {
  return vibeqc_grid_adjoint::contract_point(
      point, centers, na, owner, seed, gradient, logs, products, bar_product,
      bar_distance, zeros, distances, grid_iter_1::local_norm,
      grid_iter_1::local_ratio, grid_iter_1::local_log, pair);
}

__device__ bool contract_grid_runtime(
    unsigned iterations, const double* point, const double* centers, size_t na,
    size_t owner, double seed, double* gradient, double* logs, double* products,
    double* bar_product, double* bar_distance, size_t* zeros,
    std::array<double, 4>* distances) {
  switch (iterations) {
    case 1:
      return contract_grid_impl(point, centers, na, owner, seed, gradient, logs, products,
                                bar_product, bar_distance, zeros, distances,
                                grid_iter_1::local_becke);
    case 2:
      return contract_grid_impl(point, centers, na, owner, seed, gradient, logs, products,
                                bar_product, bar_distance, zeros, distances,
                                grid_iter_2::local_becke);
    case 3:
      return contract_grid_impl(point, centers, na, owner, seed, gradient, logs, products,
                                bar_product, bar_distance, zeros, distances,
                                grid_iter_3::local_becke);
    case 4:
      return contract_grid_impl(point, centers, na, owner, seed, gradient, logs, products,
                                bar_product, bar_distance, zeros, distances,
                                grid_iter_4::local_becke);
    case 5:
      return contract_grid_impl(point, centers, na, owner, seed, gradient, logs, products,
                                bar_product, bar_distance, zeros, distances,
                                grid_iter_5::local_becke);
    default:
      return false;
  }
}

__global__ void force_ao_kernel(
    const double* basis, I natom, I nprimitive, I nao, const double* points,
    I npoint, I jets, double* output, int* error) {
  const double* primitives = basis + 3 * natom;
  const double* records = primitives + 2 * nprimitive;
  for (I index = I(blockIdx.x) * blockDim.x + threadIdx.x;
       index < jets * npoint * nao; index += I(blockDim.x) * gridDim.x) {
    const I ao = index % nao, point = index / nao % npoint;
    const I jet = index / (nao * npoint);
    const double* record = records + 16 * ao;
    const I atom = static_cast<I>(record[0]);
    const double x = points[3 * point] - basis[3 * atom];
    const double y = points[3 * point + 1] - basis[3 * atom + 1];
    const double z = points[3 * point + 2] - basis[3 * atom + 2];
    const double r2 = x * x + y * y + z * z;
    const I first = static_cast<I>(record[1]);
    const I end = first + static_cast<I>(record[2]);
    double value = 0;
    for (I p = first; p < end; ++p) {
      const double alpha = primitives[2 * p];
      const double radial = primitives[2 * p + 1] * exp(-alpha * r2);
      if (radial == 0) continue;
      for (int term = 0; term < static_cast<int>(record[3]); ++term)
        value += radial * record[7 + 4 * term] *
                 vibeqc_grid_policy::axis_jet(
                     static_cast<int>(record[4 + 4 * term]),
                     vibeqc_grid_policy::derivatives[jet][0], alpha, x) *
                 vibeqc_grid_policy::axis_jet(
                     static_cast<int>(record[5 + 4 * term]),
                     vibeqc_grid_policy::derivatives[jet][1], alpha, y) *
                 vibeqc_grid_policy::axis_jet(
                     static_cast<int>(record[6 + 4 * term]),
                     vibeqc_grid_policy::derivatives[jet][2], alpha, z);
    }
    output[index] = finite(value, error, 0);
  }
}

__global__ void force_density_jets(
    const double* density, const double* ao, I n, I count, I spins, I jets,
    double* work, int* error) {
  const I stride = count * n;
  for (I i = I(blockIdx.x) * blockDim.x + threadIdx.x;
       i < spins * jets * stride; i += I(blockDim.x) * gridDim.x) {
    const I spin = i / (jets * stride), jet = i / stride % jets;
    const I local = i % stride, mu = local % n;
    const double* d = density + spin * n * n;
    double value = 0;
    for (I nu = 0; nu < n; ++nu)
      value += (0.5 * d[mu * n + nu] + 0.5 * d[nu * n + mu]) *
               ao[jet * stride + (local / n) * n + nu];
    work[i] = finite(value, error, 1);
  }
}

__global__ void geometry_kernel(
    bool pbe, unsigned iterations, const double* basis, const double* points,
    const double* weights, const double* raw, const std::uint32_t* owners,
    const double* ao, const double* work, I natom, I nprimitive, I n, I count,
    I spins, double* partial, double* scratch, int* error) {
  const size_t lane = threadIdx.x, stride = count * n, sjets = pbe ? 4 : 1;
  double* grad = partial + lane * 9 * natom;
  for (size_t k = 0; k < 9 * natom; ++k) grad[k] = 0;
  double* ws = scratch + lane * 9 * natom;
  auto* distances = reinterpret_cast<std::array<double, 4>*>(ws + 5 * natom);
  auto* zeros = reinterpret_cast<size_t*>(ws + 4 * natom);
  const double* records = basis + 3 * natom + 2 * nprimitive;
  for (size_t p = lane; p < size_t(count); p += workers) {
    const size_t owner = owners[p];
    if (owner >= size_t(natom) || !isfinite(weights[p]) || !isfinite(raw[p])) {
      atomicExch(error, 1);
      return;
    }
    double rho[2]{}, gradient[2][3]{};
    for (I s = 0; s < spins; ++s)
      for (I mu = 0; mu < n; ++mu) {
        const I i = p * n + mu;
        const double phi = ao[i], dphi = work[(s * sjets) * stride + i];
        rho[s] += phi * dphi;
        if (pbe)
          for (I k = 0; k < 3; ++k)
            gradient[s][k] += ao[(k + 1) * stride + i] * dphi +
                              phi * work[(s * sjets + k + 1) * stride + i];
      }
    if (spins == 1) {
      rho[1] = rho[0] * 0.5;
      rho[0] *= 0.5;
      if (pbe)
        for (I k = 0; k < 3; ++k) {
          gradient[1][k] = gradient[0][k] * 0.5;
          gradient[0][k] *= 0.5;
        }
    }
    const auto xc = vibeqc::dft::point::evaluate(pbe, rho, gradient);
    if (!xc.valid) {
      atomicExch(error, 1);
      return;
    }
    for (I mu = 0; mu < n; ++mu) {
      const auto atom = static_cast<size_t>(records[16 * mu]);
      if (atom >= size_t(natom)) {
        atomicExch(error, 1);
        return;
      }
      double pullback[4]{};
      for (I s = 0; s < spins; ++s) {
        double coefficients[4]{}, density_jets[4]{};
        coefficients[0] =
            weights[p] *
            (spins == 1 ? 0.5 * (xc.rho[0] + xc.rho[1]) : xc.rho[s]);
        for (I j = 0; j < sjets; ++j) {
          density_jets[j] = work[(s * sjets + j) * stride + p * n + mu];
          if (j)
            coefficients[j] =
                weights[p] *
                (spins == 1
                     ? 0.5 * (xc.gradient[0][j - 1] + xc.gradient[1][j - 1])
                     : xc.gradient[s][j - 1]);
        }
        double local[4]{};
        if (pbe)
          pbe_pullback::apply(coefficients, density_jets, local);
        else
          lda_pullback::apply(coefficients, density_jets, local);
        for (I j = 0; j < sjets; ++j) pullback[j] += local[j];
      }
      for (I k = 0; k < 3; ++k) {
        double value = 0;
        for (I j = 0; j < sjets; ++j) {
          const unsigned shift =
              pbe ? pbe_pullback::shift[j][k] : lda_pullback::shift[j][k];
          value += pullback[j] * ao[shift * stride + p * n + mu];
        }
        grad[3 * atom + k] -= value;
        grad[3 * natom + 3 * owner + k] += value;
      }
    }
    if (!contract_grid_runtime(
            iterations, points + 3 * p, basis, natom, owner, xc.energy * raw[p],
            grad + 6 * natom, ws, ws + natom, ws + 2 * natom, ws + 3 * natom,
            zeros, distances)) {
      atomicExch(error, 1);
      return;
    }
  }
  for (size_t k = 0; k < 9 * natom; ++k) finite(grad[k], error, 0);
}

__global__ void geometry_reduce(
    const double* partial, I natom, double* output, int* error) {
  if (*error) return;
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 9 * size_t(natom)) return;
  double sum = 0;
  for (size_t lane = 0; lane < workers; ++lane)
    sum += partial[lane * 9 * natom + i];
  output[i] = finite(output[i] + sum, error, 0);
}

void enqueue_gradient(
    const CudaXcLayout& l, cudaStream_t stream, const double* basis,
    const double* points, const double* weights, const double* raw,
    const std::uint32_t* owners, const double* density, double* ao, double* work,
    double* partial, double* scratch, double* output, int* error,
    unsigned iterations) {
  cuda_check(cudaMemsetAsync(error, 0, sizeof(int), stream));
  cuda_check(cudaMemsetAsync(output, 0, 9 * l.natom * sizeof(double), stream));
  const I sjets = l.pbe ? 4 : 1, ajets = l.pbe ? 10 : 4;
  for (size_t begin = 0; begin < l.npoint; begin += l.tile_points) {
    const I count = std::min(l.tile_points, l.npoint - begin);
    const I stride = count * l.nao;
    force_ao_kernel<<<blocks(ajets * stride, 128), 128, 0, stream>>>(
        basis, l.natom, l.nprimitive, l.nao, points + 3 * begin, count, ajets,
        ao, error);
    cuda_check(cudaGetLastError());
    force_density_jets<<<blocks(l.spins * sjets * stride, 128), 128, 0, stream>>>(
        density, ao, l.nao, count, l.spins, sjets, work, error);
    cuda_check(cudaGetLastError());
    geometry_kernel<<<1, workers, 0, stream>>>(
        l.pbe, iterations, basis, points + 3 * begin, weights + begin,
        raw + begin, owners + begin, ao, work, l.natom, l.nprimitive, l.nao,
        count, l.spins, partial, scratch, error);
    cuda_check(cudaGetLastError());
    geometry_reduce<<<blocks(9 * l.natom, 128), 128, 0, stream>>>(
        partial, l.natom, output, error);
    cuda_check(cudaGetLastError());
  }
}
"""
    return "\n".join(
        [
            "#include <algorithm>",
            "#include <array>",
            "#include <cstdint>",
            '#include "dft/cuda_xc.hpp"',
            '#include "dft/grid_response_adjoint.hpp"',
            '#include "dft/xc_point.hpp"',
            '#include "tensor/cuda_runtime.cuh"',
            emit_grid_policy().replace(
                "namespace vibeqc_grid_policy {",
                "namespace vibeqc_xc_gradient_grid_policy {",
                1,
            ),
            "namespace vibeqc::dft::cuda_xc_gradient_detail {",
            "using namespace vibeqc_tensor;",
            "constexpr size_t workers = 32;",
            _emit_pullback_namespace(pbe=False, namespace="lda_pullback"),
            _emit_pullback_namespace(pbe=True, namespace="pbe_pullback"),
            *partials,
            runtime.replace("vibeqc_grid_policy::", "vibeqc_xc_gradient_grid_policy::"),
            "}  // namespace vibeqc::dft::cuda_xc_gradient_detail",
            "",
        ]
    )
