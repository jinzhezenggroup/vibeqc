#pragma once

// Included only by the native generated_grid_policy.cu, after cuda_grid.cu.
// Reuse its AO traversal and the exact generated D/C ingredient bilinears.
#include "dft/cuda_xc.hpp"
#include "dft/xc_point.hpp"
#include "dft/xc_point_response.hpp"
#include "generated_r2scan_device.cuh"

namespace vibeqc::dft::cuda_xc_detail {
namespace {
using vibeqc_tensor::blocks;
using vibeqc_tensor::cuda_check;
using vibeqc_tensor::finite;
using vibeqc_tensor::I;

__global__ void validate_density(const double* density, I n, I spins, int* error) {
  for (I i = I(blockIdx.x) * blockDim.x + threadIdx.x; i < spins * n * n;
       i += I(blockDim.x) * gridDim.x) {
    const I transpose = (i / (n * n) * n + i % n) * n + i / n % n;
    const double a = density[i], b = density[transpose];
    if (!isfinite(a) || fabs(a - b) > 1e-12 + 1e-10 * fmax(fabs(a), fabs(b)))
      atomicCAS(error, 0, 5);
  }
}

// r2SCAN additionally keeps D*grad(phi) so tau is formed from the same
// density matrix as rho/gradient; LDA/GGA retain the one-panel fast path.
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
    for (I nu = 0; nu < n; ++nu)
      value += (0.5 * d[mu * n + nu] + 0.5 * d[nu * n + mu]) * source[point * n + nu];
    work[i] = finite(value, error, 1);
  }
}

__global__ void density_features(const double* ao, const double* work, I n, I count, I spins,
                                 I ao_jets, I work_jets, I feature_terms, I functional,
                                 double* features, int* error) {
  const I stride = count * n;
  const unsigned ingredient_mask = functional == 0 ? 1U : (functional == 1 ? 7U : 15U);
  for (I i = I(blockIdx.x) * blockDim.x + threadIdx.x; i < spins * count;
       i += I(blockDim.x) * gridDim.x) {
    const I spin = i / count, point = i % count;
    double accum[5]{};
    for (I mu = 0; mu < n; ++mu) {
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
    for (I k = 0; k < feature_terms; ++k)
      features[(spin * feature_terms + k) * count + point] = finite(accum[k], error, 1);
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

__global__ void evaluate_points(const double* features, const double* weights, I count, I spins,
                                I feature_terms, I functional, double* coefficients,
                                double* point_totals, int* error, const double* delta) {
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
    const auto xc =
        delta ? response_point(features, delta, p, count, spins, feature_terms, functional)
              : evaluate_semilocal_point(functional, rho, gradient, tau);
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

void enqueue(const CudaXcLayout& l, cudaStream_t stream, const double* basis, const double* points,
             const double* weights, const double* density, double* ao, double* work,
             double* features, double* coefficients, double* point_totals, double* potential,
             double* totals, int* error, const double* direction, double* delta_features) {
  const I matrices = l.spins * l.nao * l.nao;
  cuda_check(cudaMemsetAsync(error, 0, sizeof(int), stream));
  cuda_check(cudaMemsetAsync(totals, 0, 3 * sizeof(double), stream));
  cuda_check(cudaMemsetAsync(potential, 0, matrices * sizeof(double), stream));
  validate_density<<<blocks(matrices, 128), 128, 0, stream>>>(density, l.nao, l.spins, error);
  cuda_check(cudaGetLastError());
  if (direction) {
    validate_density<<<blocks(matrices, 128), 128, 0, stream>>>(direction, l.nao, l.spins, error);
    cuda_check(cudaGetLastError());
  }
  for (std::size_t begin = 0; begin < l.npoint; begin += l.tile_points) {
    const I count = std::min(l.tile_points, l.npoint - begin);
    // The existing through-f AO kernel is compiled earlier in this same TU.
    ao_kernel<<<blocks(l.jets * count * l.nao, 128), 128, 0, stream>>>(
        basis, l.natom, l.nprimitive, l.nao, points + 3 * begin, count, l.jets, ao, error, nullptr);
    cuda_check(cudaGetLastError());
    density_product<<<blocks(l.spins * l.work_jets * count * l.nao, 128), 128, 0, stream>>>(
        density, ao, l.nao, count, l.spins, l.work_jets, work, error);
    cuda_check(cudaGetLastError());
    density_features<<<blocks(l.spins * count, 128), 128, 0, stream>>>(
        ao, work, l.nao, count, l.spins, l.jets, l.work_jets, l.feature_terms, l.functional,
        features, error);
    cuda_check(cudaGetLastError());
    if (direction) {
      // AO panels are shared; work is scratch and can be reused after the
      // reference features are retained. No host AO/feature staging occurs.
      density_product<<<blocks(l.spins * count * l.nao, 128), 128, 0, stream>>>(
          direction, ao, l.nao, count, l.spins, l.work_jets, work, error);
      cuda_check(cudaGetLastError());
      density_features<<<blocks(l.spins * count, 128), 128, 0, stream>>>(
          ao, work, l.nao, count, l.spins, l.jets, l.work_jets, l.feature_terms, l.functional,
          delta_features, error);
      cuda_check(cudaGetLastError());
    }
    evaluate_points<<<blocks(count, 128), 128, 0, stream>>>(
        features, weights + begin, count, l.spins, l.feature_terms, l.functional, coefficients,
        point_totals, error, delta_features);
    cuda_check(cudaGetLastError());
    assemble_potential<<<blocks(matrices, 128), 128, 0, stream>>>(
        ao, coefficients, weights + begin, l.nao, count, l.spins, l.feature_terms, potential,
        error);
    cuda_check(cudaGetLastError());
    accumulate_totals<<<1, 32, 0, stream>>>(point_totals, count, totals, error);
    cuda_check(cudaGetLastError());
  }
}
}  // namespace vibeqc::dft::cuda_xc_detail
