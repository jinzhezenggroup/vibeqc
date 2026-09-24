"""Compiler-owned AO jets and density-invariant arithmetic for grid schedules.

The normalized native basis remains the only source of primitive and public AO
coefficients. Scalar DAGs own Gaussian factors and feature algebra; bounded AO/
feature traversal and local XC contractions are emitted here. Maps, buffers,
matrix calls and host/runtime orchestration remain native.

Rationale: .agents/notes/implemented/architecture/2026-09-20-cuda-grid-emitted-traversal.md
"""

import typing

from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.integral.cuda import CudaEmitter
from vibeqc_compiler.integral.expr import AlgebraForm, Graph

from .ao import jet_indices
from .feature_policy import emit_feature_policy
from .xc_contraction_cuda import emit_native_xc_matrix_schedule

_GRID_SCIENTIFIC_KERNELS = r"""#include "../tensor/cuda_runtime.cuh"
#include "xc_point.hpp"

namespace {
using namespace vibeqc_tensor;
// AO and density arithmetic is emitted by dft/ao_cuda.py.
using vibeqc_grid_policy::axis_jet;
using vibeqc_grid_policy::derivatives;

__global__ void ao_kernel(const double* basis, I natom, I nprimitive, I nao, const double* points,
                          I npoint, I jets, double* output, int* error, const size_t* ao_ids) {
  const double* primitives = basis + 3 * natom;
  const double* records = primitives + 2 * nprimitive;
  for (I index = I(blockIdx.x) * blockDim.x + threadIdx.x; index < jets * npoint * nao;
       index += I(blockDim.x) * gridDim.x) {
    const I ao = index % nao, point = index / nao % npoint, jet = index / (nao * npoint);
    const double* record = records + 16 * (ao_ids ? ao_ids[ao] : ao);
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

__global__ void ao_kernel_fp32(const double* basis, I natom, I nprimitive, I nao,
                               const double* points, I npoint, I jets, double* output,
                               int* error, const size_t* ao_ids) {
  const double* primitives = basis + 3 * natom;
  const double* records = primitives + 2 * nprimitive;
  for (I index = I(blockIdx.x) * blockDim.x + threadIdx.x; index < jets * npoint * nao;
       index += I(blockDim.x) * gridDim.x) {
    const I ao = index % nao, point = index / nao % npoint, jet = index / (nao * npoint);
    const double* record = records + 16 * (ao_ids ? ao_ids[ao] : ao);
    const I atom = static_cast<I>(record[0]);
    // Preserve local displacements before entering the FP32 arithmetic domain.
    // Rounding absolute coordinates first makes AO values depend on translation.
    const float x = static_cast<float>(points[3 * point] - basis[3 * atom]);
    const float y = static_cast<float>(points[3 * point + 1] - basis[3 * atom + 1]);
    const float z = static_cast<float>(points[3 * point + 2] - basis[3 * atom + 2]);
    const float r2 = x * x + y * y + z * z;
    const I first = static_cast<I>(record[1]), end = first + static_cast<I>(record[2]);
    float value = 0.0f;
    for (I p = first; p < end; ++p) {
      const float alpha = static_cast<float>(primitives[2 * p]);
      const float radial = static_cast<float>(primitives[2 * p + 1]) * exp(-alpha * r2);
      if (radial == 0.0f) continue;
      for (int term = 0; term < static_cast<int>(record[3]); ++term) {
        value +=
            radial * static_cast<float>(record[7 + 4 * term]) *
            axis_jet(static_cast<int>(record[4 + 4 * term]), derivatives[jet][0], alpha, x) *
            axis_jet(static_cast<int>(record[5 + 4 * term]), derivatives[jet][1], alpha, y) *
            axis_jet(static_cast<int>(record[6 + 4 * term]), derivatives[jet][2], alpha, z);
      }
    }
    output[index] = finite(static_cast<double>(value), error, 0);
  }
}

__global__ void feature_kernel(const double* ao, const double* work, I npoint, I nao,
                               double* output, int* error, unsigned mask) {
  for (I point = I(blockIdx.x) * blockDim.x + threadIdx.x; point < npoint;
       point += I(blockDim.x) * gridDim.x) {
    double gradients[2][3]{};
    const I stride = npoint * nao;
    for (int spin = 0; spin < 2; ++spin) {
      const double* w = work + 4 * spin * stride;
      double accum[5]{};
      for (I mu = 0; mu < nao; ++mu) {
        const I i = point * nao + mu;
        double derivative[3]{}, panel[4]{};
        if (mask & 7) panel[0] = w[i];
        if (mask & 8)
          for (int k = 1; k < 4; ++k) panel[k] = w[k * stride + i];
        if (mask & 14)
          for (int k = 0; k < 3; ++k) derivative[k] = ao[(k + 1) * stride + i];
        vibeqc_grid_policy::add_features(ao[i], derivative, panel, accum, mask);
      }
      for (int k = 0; k < 5; ++k)
        output[(5 * spin + k) * npoint + point] = finite(accum[k], error, 1);
      for (int k = 0; k < 3; ++k) gradients[spin][k] = accum[k + 1];
    }
    if (mask & 4) {
      double sigma[3];
      vibeqc_grid_policy::sigma(gradients, sigma);
      for (int k = 0; k < 3; ++k) output[(10 + k) * npoint + point] = finite(sigma[k], error, 1);
    }
  }
}

// Reduce one occupied tile on the owner's stream. Partial sums remain per
// spin/point; sigma is formed only after ALL occupied tiles in BOTH spins.
__global__ void orbital_feature_kernel(const double* psi, I npoint, I width, int spin,
                                       double* output, int* error, unsigned mask) {
  const I stride = npoint * width;
  for (I point = I(blockIdx.x) * blockDim.x + threadIdx.x; point < npoint;
       point += I(blockDim.x) * gridDim.x) {
    double accum[5]{};
    for (I orbital = 0; orbital < width; ++orbital) {
      const I i = point * width + orbital;
      double panel[4]{}, derivative[3]{};
      if (mask & 7) panel[0] = psi[i];
      if (mask & 14)
        for (int k = 0; k < 3; ++k) derivative[k] = panel[k + 1] = psi[(k + 1) * stride + i];
      vibeqc_grid_policy::add_features(panel[0], derivative, panel, accum, mask);
    }
    for (int k = 0; k < 5; ++k) {
      const I destination = (5 * spin + k) * npoint + point;
      output[destination] = finite(output[destination] + accum[k], error, 1);
    }
  }
}

__global__ void finish_orbital_sigma(double* output, I npoint, int* error) {
  for (I point = I(blockIdx.x) * blockDim.x + threadIdx.x; point < npoint;
       point += I(blockDim.x) * gridDim.x) {
    double gradients[2][3], sigma[3];
    for (int s = 0; s < 2; ++s)
      for (int k = 0; k < 3; ++k) gradients[s][k] = output[(5 * s + k + 1) * npoint + point];
    vibeqc_grid_policy::sigma(gradients, sigma);
    for (int k = 0; k < 3; ++k) output[(10 + k) * npoint + point] = finite(sigma[k], error, 1);
  }
}

__device__ vibeqc::dft::point::Value evaluate_xc_point(bool pbe, bool restricted,
                                                       const double* features, I npoint, I point) {
  const double rho[2]{features[point], features[5 * npoint + point]};
  double gradient[2][3]{};
  if (pbe)
    for (int spin = 0; spin < 2; ++spin)
      for (int axis = 0; axis < 3; ++axis)
        gradient[spin][axis] = features[(5 * spin + axis + 1) * npoint + point];
  // This spatial consumer implements the compiler's interior-v1 contract.
  // Native KS uses its own resident consumer of the same point algebra.
  if (restricted) {
    vibeqc::dft::point::Value invalid;
    if (rho[0] != rho[1]) {
      invalid.valid = false;
      return invalid;
    }
    for (int axis = 0; axis < 3; ++axis)
      if (gradient[0][axis] != gradient[1][axis]) {
        invalid.valid = false;
        return invalid;
      }
  }
  return vibeqc::dft::point::evaluate_interior(pbe, rho, gradient);
}

/** Deterministic scalar reduction. This correctness baseline intentionally
 * uses one device thread; matrix assembly remains parallel and later tuning
 * may replace only this reduction after endpoint-equivalence evidence.
 */
__global__ void xc_integrals_kernel(bool pbe, bool restricted, const double* features,
                                    const double* weights, I npoint, double* integrals,
                                    int* error) {
  if (blockIdx.x || threadIdx.x) return;
  double energy = 0.0, electrons[2]{};
  for (I point = 0; point < npoint; ++point) {
    const auto xc = evaluate_xc_point(pbe, restricted, features, npoint, point);
    if (!xc.valid) {
      atomicCAS(error, 0, 3);
      return;
    }
    const double weight = weights[point];
    energy += weight * xc.energy;
    electrons[0] += weight * features[point];
    electrons[1] += weight * features[5 * npoint + point];
  }
  integrals[0] = finite(energy, error, 3);
  integrals[1] = finite(electrons[0], error, 3);
  integrals[2] = finite(electrons[1], error, 3);
}

__global__ void xc_local_potential_kernel(bool pbe, bool restricted, const double* features,
                                          const double* ao, const double* weights, I npoint,
                                          I active, double* potential, int* error) {
  for (I index = I(blockIdx.x) * blockDim.x + threadIdx.x; index < 2 * active * active;
       index += I(blockDim.x) * gridDim.x) {
    const I spin = index / (active * active), row = index / active % active, col = index % active;
    double value = 0.0;
    for (I point = 0; point < npoint; ++point) {
      const auto xc = evaluate_xc_point(pbe, restricted, features, npoint, point);
      if (!xc.valid) {
        atomicCAS(error, 0, 3);
        return;
      }
      const I base = point * active;
      const double phi_row = ao[base + row], phi_col = ao[base + col];
      double contribution = xc.rho[spin] * phi_row * phi_col;
      if (pbe) {
        const I stride = npoint * active;
        for (int axis = 0; axis < 3; ++axis) {
          const double derivative_row = ao[(axis + 1) * stride + base + row];
          const double derivative_col = ao[(axis + 1) * stride + base + col];
          contribution +=
              xc.gradient[spin][axis] * (derivative_row * phi_col + phi_row * derivative_col);
        }
      }
      value += weights[point] * contribution;
    }
    potential[index] = finite(value, error, 3);
  }
}
}  // namespace
"""

_NATIVE_XC_CONTRACTION_KERNELS = r"""// Generated by vibeqc_compiler.dft.ao_cuda; do not edit.
#include "dft/cuda_xc.hpp"
#include "dft/xc_point_response.hpp"
#include "generated_r2scan_device.cuh"
namespace vibeqc::dft::cuda_xc_detail {
namespace {
using vibeqc_tensor::finite;
using vibeqc_tensor::I;

// r2SCAN additionally keeps D*grad(phi) so tau is formed from the same
// density matrix as rho/gradient; LDA/GGA retain the one-panel fast path.
template <bool Mixed>
__global__ void density_product(const double* density, const double* ao, I n, I count, I spins,
                                I work_jets, double* work, int* error) {
  const I panel = count * n;
  for (I i = I(blockIdx.x) * blockDim.x + threadIdx.x; i < spins * work_jets * panel;
       i += I(blockDim.x) * gridDim.x) {
    const I spin = i / (work_jets * panel), jet = i / panel % work_jets;
    const I point = i / n % count, mu = i % n;
    const double* d = density + spin * n * n;
    const double* source = ao + jet * panel;
    double value = 0.0;
    for (I nu = 0; nu < n; ++nu) {
      if constexpr (Mixed) {
        // AUTO uses binary32 products while retaining the long AO reduction in binary64.
        const float left = __double2float_rn(d[mu * n + nu]);
        const float right = __double2float_rn(d[nu * n + mu]);
        const float symmetric = __fadd_rn(__fmul_rn(0.5f, left), __fmul_rn(0.5f, right));
        const float orbital = __double2float_rn(source[point * n + nu]);
        value = __dadd_rn(value, static_cast<double>(__fmul_rn(symmetric, orbital)));
      } else {
        value += (0.5 * d[mu * n + nu] + 0.5 * d[nu * n + mu]) * source[point * n + nu];
      }
    }
    work[i] = finite(value, error, 1);
  }
}

template<bool Cooperative>
__global__ void density_features(const double* ao, const double* work, I n, I count, I spins,
                                 I ao_jets, I work_jets, I feature_terms, I functional,
                                 double* features, int* error) {
  const I stride = count * n;
  // A warp owns one point so adjacent lanes read adjacent AO components.
  // Every lane follows the same point loop, including a partial final warp.
  constexpr I width = Cooperative ? 32 : 1;
  const I lane = threadIdx.x % width;
  const unsigned ingredient_mask = functional == 0 ? 1U : (functional == 1 ? 7U : 15U);
  for (I i = (I(blockIdx.x) * blockDim.x + threadIdx.x) / width; i < spins * count;
       i += I(blockDim.x) * gridDim.x / width) {
    const I spin = i / count, point = i % count;
    double accum[5]{};
    for (I mu = lane; mu < n; mu += width) {
      const I index = point * n + mu;
      double derivatives[3]{};
      double panel[4]{work[(spin * work_jets) * stride + index], 0.0, 0.0, 0.0};
      if (ao_jets == 4)
        for (unsigned k = 0; k < 3; ++k) derivatives[k] = ao[(k + 1) * stride + index];
      if (work_jets == 4)
        for (unsigned k = 0; k < 3; ++k)
          panel[k + 1] = work[(spin * work_jets + k + 1) * stride + index];
      vibeqc_grid_policy::add_features(ao[index], derivatives, panel, accum, ingredient_mask);
    }
    for (I k = 0; k < feature_terms; ++k) {
      if constexpr (Cooperative)
        for (int offset = 16; offset > 0; offset >>= 1)
          accum[k] += __shfl_down_sync(0xffffffffu, accum[k], offset);
      if (lane == 0)
        features[(spin * feature_terms + k) * count + point] = finite(accum[k], error, 1);
    }
  }
}

// Native binds buffers; this compiler owner selects the bounded reduction.
// Preserve scalar order for small AO spaces, with no additional storage.
inline void scheduled_density_features(cudaStream_t stream, const double* ao, const double* work,
    I n, I count, I spins, I ao_jets, I work_jets, I feature_terms, I functional,
    double* features, int* error) {
  if (n >= 32) {
    density_features<true><<<vibeqc_tensor::blocks(spins*count*32,128),128,0,stream>>>(
        ao,work,n,count,spins,ao_jets,work_jets,feature_terms,functional,features,error);
  } else {
    density_features<false><<<vibeqc_tensor::blocks(spins*count,128),128,0,stream>>>(
        ao,work,n,count,spins,ao_jets,work_jets,feature_terms,functional,features,error);
  }
}

struct DevicePointValue {
  double energy{}, rho[2]{}, gradient[2][3]{}, kinetic[2]{};
  bool valid{true};
};

__device__ inline DevicePointValue from_point(const point::Value& value) {
  DevicePointValue out;
  out.energy = value.energy;
  out.valid = value.valid;
  for (I s = 0; s < 2; ++s) {
    out.rho[s] = value.rho[s];
    for (I k = 0; k < 3; ++k) out.gradient[s][k] = value.gradient[s][k];
  }
  return out;
}

// Layout adaptation only: response differentiates the same scaled point
// expression as the CPU consumer, including its vacuum/empty-spin policy.
__device__ inline DevicePointValue response_point(const double* features, const double* delta, I p,
                                                  I count, I spins, I terms, I functional) {
  double rho[2]{}, gradient[2][3]{}, drho[2]{}, dgradient[2][3]{};
  for (I s = 0; s < spins; ++s) {
    rho[s] = features[s * terms * count + p];
    drho[s] = delta[s * terms * count + p];
    if (terms >= 4)
      for (I k = 0; k < 3; ++k) {
        gradient[s][k] = features[(s * terms + k + 1) * count + p];
        dgradient[s][k] = delta[(s * terms + k + 1) * count + p];
      }
  }
  return from_point(
      spins == 1
          ? point::restricted_response(functional == 1, rho[0], gradient[0], drho[0], dgradient[0])
          : point::unrestricted_response(functional == 1, rho, gradient, drho, dgradient));
}

__device__ inline DevicePointValue evaluate_semilocal_point(I functional, const double rho[2],
                                                            const double gradient[2][3],
                                                            const double tau[2]) {
  DevicePointValue out;
  if (functional < 2) {
    return from_point(point::evaluate(functional == 1, rho, gradient));
  }
  // Match the host r2SCAN physical-domain gate before either the vacuum
  // shortcut or Libxc work-input continuation can hide an invalid density.
  for (I spin = 0; spin < 2; ++spin) {
    if (!isfinite(rho[spin]) || rho[spin] < 0.0 ||
        !isfinite(tau[spin]) || tau[spin] < 0.0) {
      out.valid = false;
      return out;
    }
    for (I k = 0; k < 3; ++k)
      if (!isfinite(gradient[spin][k])) {
        out.valid = false;
        return out;
      }
  }
  const double total = rho[0] + rho[1];
  constexpr double tail_low = 1.0e-56, tail_high = 1.0e-52;
  if (total <= tail_low) return out;
  double sigma[3]{};
  for (I k = 0; k < 3; ++k) {
    sigma[0] += gradient[0][k] * gradient[0][k];
    sigma[1] += gradient[0][k] * gradient[1][k];
    sigma[2] += gradient[1][k] * gradient[1][k];
  }
  auto raw = generated::r2scan_device(rho[0], rho[1], sigma[0], sigma[1], sigma[2], tau[0], tau[1]);
  out.valid = isfinite(raw.energy_density);
  for (double derivative : raw.feature_derivative) out.valid = out.valid && isfinite(derivative);
  if (!out.valid) return out;
  if (total < tail_high) {
    const double width = tail_high - tail_low;
    const double x = (total - tail_low) / width;
    const double x2 = x * x, x3 = x2 * x;
    const double scale = x3 * (10.0 + x * (-15.0 + 6.0 * x));
    const double dscale = 30.0 * x2 * (1.0 - x) * (1.0 - x) / width;
    const double energy = raw.energy_density;
    raw.energy_density *= scale;
    raw.feature_derivative[0] = scale * raw.feature_derivative[0] + dscale * energy;
    raw.feature_derivative[1] = scale * raw.feature_derivative[1] + dscale * energy;
    for (I i = 2; i < 7; ++i) raw.feature_derivative[i] *= scale;
  }
  out.energy = raw.energy_density;
  out.rho[0] = raw.feature_derivative[0];
  out.rho[1] = raw.feature_derivative[1];
  for (I k = 0; k < 3; ++k) {
    out.gradient[0][k] = 2.0 * raw.feature_derivative[2] * gradient[0][k] +
                         raw.feature_derivative[3] * gradient[1][k];
    out.gradient[1][k] = raw.feature_derivative[3] * gradient[0][k] +
                         2.0 * raw.feature_derivative[4] * gradient[1][k];
  }
  out.kinetic[0] = 0.5 * raw.feature_derivative[5];
  out.kinetic[1] = 0.5 * raw.feature_derivative[6];
  return out;
}

// The immutable plan's admitted consumer is a compile-time fact. Keep spin as
// a layout argument to bound AOT growth while pruning unrelated point algebra
// before register allocation. All variants reuse the canonical point source.
template <I functional, bool response>
__global__ void evaluate_points(const double* features, const double* weights, I count, I spins,
                                double* coefficients,
                                double* point_totals, int* error, const double* delta) {
  static_assert(functional >= 0 && functional <= 2 && (!response || functional < 2));
  constexpr I feature_terms = functional == 0 ? 1 : (functional == 1 ? 4 : 5);
  for (I p = I(blockIdx.x) * blockDim.x + threadIdx.x; p < count; p += I(blockDim.x) * gridDim.x) {
    double rho[2]{}, gradient[2][3]{}, tau[2]{};
    for (I s = 0; s < 2; ++s) {
      const I source = spins == 1 ? 0 : s;
      const double scale = spins == 1 ? 0.5 : 1.0;
      rho[s] = scale * features[source * feature_terms * count + p];
      if (feature_terms >= 4)
        for (I k = 0; k < 3; ++k)
          gradient[s][k] = scale * features[(source * feature_terms + k + 1) * count + p];
      if (feature_terms == 5) tau[s] = scale * features[(source * feature_terms + 4) * count + p];
    }
    DevicePointValue xc;
    if constexpr (response)
      xc = response_point(features, delta, p, count, spins, feature_terms, functional);
    else
      xc = evaluate_semilocal_point(functional, rho, gradient, tau);
    if (!xc.valid) atomicCAS(error, 0, 3);
    point_totals[p] = finite(weights[p] * xc.energy, error, 2);
    for (I s = 0; s < 2; ++s)
      point_totals[(s + 1) * count + p] = finite(weights[p] * rho[s], error, 2);
    for (I s = 0; s < spins; ++s) {
      coefficients[s * feature_terms * count + p] =
          spins == 1 ? 0.5 * (xc.rho[0] + xc.rho[1]) : xc.rho[s];
      if (feature_terms >= 4)
        for (I k = 0; k < 3; ++k)
          coefficients[(s * feature_terms + k + 1) * count + p] =
              spins == 1 ? 0.5 * (xc.gradient[0][k] + xc.gradient[1][k]) : xc.gradient[s][k];
      if (feature_terms == 5)
        coefficients[(s * feature_terms + 4) * count + p] =
            spins == 1 ? 0.5 * (xc.kinetic[0] + xc.kinetic[1]) : xc.kinetic[s];
    }
  }
}

// The qualified physical PBE schedule spreads a bounded tile over more SMs.
// Other consumers retain their original launch width until separately measured;
// no extra kernel variant, point work or device workspace is introduced.
template <I functional, bool response>
void launch_points(cudaStream_t stream, const double* features, const double* weights,
                    std::size_t count, std::size_t spins, double* coefficients,
                    double* point_totals, int* error, const double* delta) {
  constexpr I threads = functional == 1 && !response ? 32 : 128;
  evaluate_points<functional, response><<<vibeqc_tensor::blocks(count, threads), threads, 0, stream>>>(
      features, weights, count, spins, coefficients, point_totals, error, delta);
}

__global__ void assemble_potential(const double* ao, const double* coefficients,
                                   const double* weights, I n, I count, I spins, I feature_terms,
                                   double* potential, int* error) {
  const I stride = count * n;
  for (I i = I(blockIdx.x) * blockDim.x + threadIdx.x; i < spins * n * n;
       i += I(blockDim.x) * gridDim.x) {
    const I spin = i / (n * n), mu = i / n % n, nu = i % n;
    if (mu > nu) continue;
    double value = 0.0;
    for (I p = 0; p < count; ++p) {
      const double a = ao[p * n + mu], b = ao[p * n + nu];
      double integrand = coefficients[spin * feature_terms * count + p] * a * b;
      if (feature_terms >= 4)
        for (I k = 0; k < 3; ++k)
          integrand +=
              coefficients[(spin * feature_terms + k + 1) * count + p] *
              (ao[(k + 1) * stride + p * n + mu] * b + a * ao[(k + 1) * stride + p * n + nu]);
      if (feature_terms == 5)
        for (I k = 0; k < 3; ++k)
          integrand += coefficients[(spin * feature_terms + 4) * count + p] *
                       ao[(k + 1) * stride + p * n + mu] * ao[(k + 1) * stride + p * n + nu];
      value += weights[p] * integrand;
    }
    value = finite(potential[i] + value, error, 3);
    potential[i] = value;
    potential[(spin * n + nu) * n + mu] = value;
  }
}

__global__ void accumulate_totals(const double* point_totals, I count, double* totals, int* error) {
  const I channel = threadIdx.x;
  if (channel >= 3) return;
  double value = 0.0;
  for (I p = 0; p < count; ++p) value += point_totals[channel * count + p];
  totals[channel] = finite(totals[channel] + value, error, 3);
}
}  // namespace
@POINT_DISPATCH@
}  // namespace vibeqc::dft::cuda_xc_detail
"""


def emit_native_xc_point_dispatch() -> str:
    """Resolve the finite native admission key without device work or allocation.

    Physical LDA/PBE/r²SCAN and signed LDA/PBE response share the existing point
    equations. Spin and AO precision do not multiply the FP64 point entry set.
    The native plan retains this launcher; unsupported keys fail before enqueue.
    """
    lines = [
        "CudaXcPointLauncher resolve_point_launcher(std::uint32_t functional, bool response) {"
    ]
    for functional, response in (
        (0, False),
        (1, False),
        (2, False),
        (0, True),
        (1, True),
    ):
        consumer = "true" if response else "false"
        lines.append(
            f"  if (functional == {functional}U && response == {consumer}) "
            f"return &launch_points<{functional}, {consumer}>;"
        )
    lines.extend(
        (
            '  throw std::invalid_argument("unsupported CUDA XC point consumer");',
            "}",
        )
    )
    return "\n".join(lines) + "\n"


def emit_native_xc_contraction_kernels() -> str:
    """Emit resident XC features, point algebra, potential and scalar reductions."""

    return (
        _NATIVE_XC_CONTRACTION_KERNELS.replace(
            "@POINT_DISPATCH@", emit_native_xc_point_dispatch()
        )
        + emit_native_xc_matrix_schedule()
    )


def emit_grid_scientific_kernels() -> str:
    """Emit AO/feature traversal and local XC contractions for the grid runtime."""
    return _GRID_SCIENTIFIC_KERNELS


def axis_expression(power: typing.Any, derivative: typing.Any) -> typing.Any:
    """Return exp(+a*x*x) d^d[x^l exp(-a*x*x)] as a scalar polynomial.

    Keeping the exponential outside avoids division by an underflowed radial
    factor. Derivatives are ordinary spatial derivatives, without factorials.
    """
    if (
        type(power) is not int
        or type(derivative) is not int
        or not (0 <= power <= 3 and 0 <= derivative <= 3)
    ):
        raise ValueError("AO axis domain is l,d in [0,3]")
    graph = Graph()
    x, a = graph.variable("x"), graph.variable("a")
    root = graph.constant(1)
    for _ in range(power):
        root = root * x
    for _ in range(derivative):
        root = graph.differentiate(root, x) - 2 * a * x * root
    return graph, root


def emit_grid_policy() -> typing.Any:
    """Emit through-f/order-three AO factors and spin feature contractions.

    The same policy serves dense and selected-column execution. Adding an AO
    domain needs a compiler change rather than another native scientific body.
    """
    lines = [
        "// Generated by vibeqc_compiler.dft.ao_cuda; do not edit.",
        "#include <cmath>",
        "namespace vibeqc_grid_policy {",
    ]
    # Both precision variants lower the same symbolic AO DAG.  The explicit
    # float emitter keeps literals and temporaries in FP32 instead of computing
    # the strict expression in FP64 and truncating only the stored panel.
    for scalar_type in ("double", "float"):
        arguments = (
            "int l, int d, double a, double x"
            if scalar_type == "double"
            else "int l, int d, float a, float x"
        )
        lines.extend(
            (
                # This common order-three evaluator otherwise inlines three
                # copies of its branch temporaries into AO traversal.
                f"__device__ __noinline__ {scalar_type} axis_jet({arguments}) {{",
                "  switch (4*l+d) {",
            )
        )
        for power in range(4):
            for derivative in range(4):
                graph, root = axis_expression(power, derivative)
                graph, (root,) = graph.apply_algebra_form(
                    (root,), AlgebraForm.FACTORED_NARY
                )
                emitter = CudaEmitter(graph, {}, scalar_type=scalar_type)
                emitter.emit((root,))
                lines.append(f"  case {4 * power + derivative}: {{")
                lines.extend(emitter.lines)
                lines.extend((f"    return {emitter.reference(root)};", "  }"))
        lines.extend(("  }", "  return NAN;", "}"))
    domain = ",".join("{" + ",".join(map(str, d)) + "}" for d in jet_indices(3))
    lines.append(f"__constant__ int derivatives[20][3] = {{{domain}}};")
    lines.extend((emit_feature_policy(device=True).rstrip("\n"), "}", ""))
    return "\n".join(lines)


def emit_grid_source(*, native_ks: typing.Any = False) -> typing.Any:
    """Compose one AO policy with the grid runtime and optional resident KS glue.

    The native library additionally instantiates its borrowed-buffer XC kernels.
    JIT grid owners retain their own ABI and arena without that native extension.
    """
    policy = emit_grid_policy()
    source = policy + emit_grid_scientific_kernels() + '#include "cuda_grid.cu"\n'
    if native_ks:
        source += emit_native_xc_contraction_kernels()
        source += '#include "cuda_xc_kernels.cuh"\n'
    return (
        source,
        canonical_hash({"schema": "vibeqc.grid-policy.v1", "source": source}),
        (
            asset_path("src/dft/cuda_grid.cu"),
            asset_path("src/tensor/cuda_runtime.cuh"),
            asset_path("src/runtime/bounded_workspace.hpp"),
            asset_path("src/runtime/cuda_resources.cuh"),
            asset_path("src/runtime/resource_cuda.cuh"),
            asset_path("src/runtime/resource_ledger.hpp"),
            asset_path("src/dft/grid_task_view.cuh"),
            asset_path("src/dft/xc_point.hpp"),
            asset_path("src/tensor/cuda_error.hpp"),
            asset_path("include/vibeqc/vibeqc.h"),
        ),
    )
