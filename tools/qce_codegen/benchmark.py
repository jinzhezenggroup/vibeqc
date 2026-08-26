"""Compile and time the generated fused ``dppp`` kernel in isolation.

The comparison kernel uses the same generated dppp recurrence and contraction
math but recomputes primitive geometry, Boys values, and Cartesian Coulomb
states independently in every component lane.  This isolates the scheduling
benefit of shell-class fusion without involving production CUDA dispatch.

Run, for example::

    python -m tools.qce_codegen.benchmark \
      --nvcc /group/software/cuda-12.9.1/bin/nvcc --architecture sm_120 \
      --partition main --gres gpu:5090:1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .dppp_dispatch import (
    _emitted_component_names,
    _generic_task_component_setup,
    _specialize_dppp_identifiers,
    emit_shell_class_fused_cuda,
    emit_uncached_primitive_geometry_cuda,
)
from .fused_schedule import FusedShellPlan, build_fused_shell_plan
from .ir import KernelConsumer, ScheduleIR, ScheduleKind
from .shell_spec import DDPS_SPEC, DPDS_SPEC, DPPP_SPEC, ShellClassSpec

_CUDA_PRELUDE = r"""
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

template <unsigned MaximumOrder>
__device__ __forceinline__ void boys_values(double argument, double* values) {
  for (unsigned order = 0; order <= MaximumOrder; ++order) values[order] = 0.0;
  if (argument < 6.0) {
    double term = 1.0;
    double sum = 0.0;
    for (unsigned k = 0; k < 80U; ++k) {
      sum += term /
          static_cast<double>(2U * MaximumOrder + 2U * k + 1U);
      term *= -argument / static_cast<double>(k + 1U);
      if (fabs(term) < 1.0e-18) break;
    }
    values[MaximumOrder] = sum;
    const double exponential = exp(-argument);
    for (unsigned order = MaximumOrder; order > 0U; --order) {
      values[order - 1U] =
          (2.0 * argument * values[order] + exponential) /
          static_cast<double>(2U * order - 1U);
    }
    return;
  }
  values[0] = 0.5 * sqrt(3.14159265358979323846 / argument) *
      erf(sqrt(argument));
  const double exponential = exp(-argument);
  for (unsigned order = 1; order <= MaximumOrder; ++order) {
    values[order] =
        ((2.0 * static_cast<double>(order) - 1.0) * values[order - 1U] -
         exponential) /
        (2.0 * argument);
  }
}
"""


_UNFUSED_KERNEL = r"""
/** Per-component recurrence baseline with no primitive/Coulomb sharing. */
extern "C" __global__ __launch_bounds__(kGeneratedDpppBlockThreads)
void generated_dppp_component_recompute_rhf_kernel(
    const GeneratedDpppShellTask* tasks,
    const double* primitive_exponents,
    const double* primitive_coefficients,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    const double* density,
    double* forces,
    std::size_t task_count) {
  struct Shared {
    GeneratedDpppShellTask task;
    GeneratedDpppVec3 positions[4];
    double warp_sums[kGeneratedDpppWarpCount][12];
  };
  __shared__ Shared shared;
  const unsigned lane = threadIdx.x;
  if (blockDim.x != kGeneratedDpppBlockThreads || blockIdx.x >= task_count) return;
  if (lane == 0U) {
    shared.task = tasks[blockIdx.x];
#pragma unroll
    for (unsigned center = 0; center < 4U; ++center) {
      shared.positions[center] = atom_positions[shared.task.atom[center]];
    }
  }
  __syncthreads();

QCE_COMPONENT_SCHEDULE_SETUP
QCE_COMPONENT_SETUP
  const double density_coefficient =
      component_lane && unique_ket_component
      ? generated_dppp_density_coefficient<false>(
            shared.task, i, j, k, l, density)
      : 0.0;
QCE_ANGULAR_COEFFICIENT
  double component_force[12]{};

  if (density_coefficient != 0.0) {
    for (std::uint64_t a = shared.task.primitive_begin[0];
         a < shared.task.primitive_end[0]; ++a) {
      for (std::uint64_t b = shared.task.primitive_begin[1];
           b < shared.task.primitive_end[1]; ++b) {
        for (std::uint64_t c = shared.task.primitive_begin[2];
             c < shared.task.primitive_end[2]; ++c) {
          for (std::uint64_t d = shared.task.primitive_begin[3];
               d < shared.task.primitive_end[3]; ++d) {
            GeneratedDpppPrimitiveGeometry primitive;
            generated_dppp_make_primitive_geometry_uncached(
                primitive_exponents[a], shared.positions[0],
                primitive_exponents[b], shared.positions[1],
                primitive_exponents[c], shared.positions[2],
                primitive_exponents[d], shared.positions[3],
                primitive_coefficients[a] * primitive_coefficients[b] *
                    primitive_coefficients[c] * primitive_coefficients[d],
                primitive);
            double primitive_gradient[4][3];
            generated_dppp_component_gradient<false>(
                component, primitive, nullptr, primitive_gradient);
            const double scale = -density_coefficient * angular_coefficient *
                primitive.primitive_coefficient;
#pragma unroll
            for (unsigned center = 0; center < 4U; ++center) {
#pragma unroll
              for (unsigned coordinate = 0; coordinate < 3U; ++coordinate) {
                component_force[center * 3U + coordinate] +=
                    scale * primitive_gradient[center][coordinate];
              }
            }
          }
        }
      }
    }
  }

  const unsigned warp = lane / 32U;
  const unsigned warp_lane = lane % 32U;
#pragma unroll
  for (unsigned slot = 0; slot < 12U; ++slot) {
    double value = component_force[slot];
#pragma unroll
    for (unsigned offset = 16U; offset != 0U; offset /= 2U) {
      value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    if (warp_lane == 0U) shared.warp_sums[warp][slot] = value;
  }
  __syncthreads();
  if (lane < 12U) {
    double value = 0.0;
#pragma unroll
    for (unsigned source_warp = 0; source_warp < kGeneratedDpppWarpCount;
         ++source_warp) {
      value += shared.warp_sums[source_warp][lane];
    }
    if (value != 0.0) {
      const unsigned center = lane / 3U;
      const unsigned coordinate = lane % 3U;
      atomicAdd(forces + static_cast<std::size_t>(shared.task.atom[center]) * 3U +
                    coordinate,
                value);
    }
  }
QCE_COMPONENT_SCHEDULE_CLOSE
}
"""


_UNFUSED_FOCK_KERNEL = r"""
/** Per-component Fock baseline with no primitive/Coulomb sharing. */
extern "C" __global__ __launch_bounds__(kGeneratedDpppFockBlockThreads)
void generated_dppp_component_recompute_fock_rhf_kernel(
    const GeneratedDpppShellTask* tasks,
    const double* primitive_exponents,
    const double* primitive_coefficients,
    const double* ao_coefficients,
    const GeneratedDpppVec3* atom_positions,
    const double* density,
    double* fock,
    std::size_t task_count) {
  struct Shared {
    GeneratedDpppShellTask task;
    GeneratedDpppVec3 positions[4];
  };
  __shared__ Shared shared;
  const unsigned lane = threadIdx.x;
  if (blockDim.x != kGeneratedDpppFockBlockThreads ||
      blockIdx.x >= task_count) return;
  if (lane == 0U) {
    shared.task = tasks[blockIdx.x];
#pragma unroll
    for (unsigned center = 0; center < 4U; ++center) {
      shared.positions[center] = atom_positions[shared.task.atom[center]];
    }
  }
  __syncthreads();

QCE_COMPONENT_SCHEDULE_SETUP
QCE_COMPONENT_SETUP
  const bool evaluate_component = component_lane && unique_ket_component;
QCE_ANGULAR_COEFFICIENT
  double component_integral = 0.0;
  if (evaluate_component) {
    for (std::uint64_t a = shared.task.primitive_begin[0];
         a < shared.task.primitive_end[0]; ++a) {
      for (std::uint64_t b = shared.task.primitive_begin[1];
           b < shared.task.primitive_end[1]; ++b) {
        for (std::uint64_t c = shared.task.primitive_begin[2];
             c < shared.task.primitive_end[2]; ++c) {
          for (std::uint64_t d = shared.task.primitive_begin[3];
               d < shared.task.primitive_end[3]; ++d) {
            GeneratedDpppPrimitiveGeometry primitive;
            generated_dppp_make_primitive_geometry_uncached(
                primitive_exponents[a], shared.positions[0],
                primitive_exponents[b], shared.positions[1],
                primitive_exponents[c], shared.positions[2],
                primitive_exponents[d], shared.positions[3],
                primitive_coefficients[a] * primitive_coefficients[b] *
                    primitive_coefficients[c] * primitive_coefficients[d],
                primitive);
            component_integral += angular_coefficient *
                primitive.primitive_coefficient *
                generated_dppp_component_value<false>(
                    component, primitive, nullptr);
          }
        }
      }
    }
  }
  if (evaluate_component && component_integral != 0.0) {
    generated_dppp_accumulate_fock<false>(
        shared.task, density, fock, i, j, k, l, component_integral);
  }
QCE_COMPONENT_SCHEDULE_CLOSE
}
"""


_HOST_HARNESS = r"""
#define QCE_CUDA_CHECK(call) do { \
  const cudaError_t error = (call); \
  if (error != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, \
                 cudaGetErrorString(error)); \
    std::exit(2); \
  } \
} while (false)

constexpr unsigned kTaskCount = QCE_TASK_COUNT;
constexpr unsigned kPrimitiveCount = QCE_PRIMITIVE_COUNT;
constexpr unsigned kWarmups = QCE_WARMUPS;
constexpr unsigned kIterations = QCE_ITERATIONS;
constexpr unsigned kSamples = QCE_SAMPLES;

template <typename Launch>
float benchmark_kernel(Launch launch, double* forces, std::size_t force_bytes) {
  std::vector<float> samples;
  samples.reserve(kSamples);
  for (unsigned sample = 0; sample < kSamples; ++sample) {
    QCE_CUDA_CHECK(cudaMemset(forces, 0, force_bytes));
    for (unsigned warmup = 0; warmup < kWarmups; ++warmup) launch();
    QCE_CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t begin;
    cudaEvent_t end;
    QCE_CUDA_CHECK(cudaEventCreate(&begin));
    QCE_CUDA_CHECK(cudaEventCreate(&end));
    QCE_CUDA_CHECK(cudaEventRecord(begin));
    for (unsigned iteration = 0; iteration < kIterations; ++iteration) launch();
    QCE_CUDA_CHECK(cudaEventRecord(end));
    QCE_CUDA_CHECK(cudaEventSynchronize(end));
    float milliseconds = 0.0f;
    QCE_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, begin, end));
    QCE_CUDA_CHECK(cudaEventDestroy(begin));
    QCE_CUDA_CHECK(cudaEventDestroy(end));
    samples.push_back(milliseconds / static_cast<float>(kIterations));
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2U];
}

int main() {
  constexpr std::size_t n = QCE_MATRIX_ORDER;
  constexpr std::size_t matrix_size = n * n;
  std::vector<GeneratedDpppShellTask> tasks(kTaskCount);
  std::vector<GeneratedDpppVec3> positions(kTaskCount * 4U);
  std::vector<double> exponents(kPrimitiveCount * 4U);
  std::vector<double> primitive_coefficients(kPrimitiveCount * 4U);
  std::vector<double> ao_coefficients(QCE_AO_COUNT, 1.0);
  std::vector<double> density(matrix_size);
  for (unsigned primitive = 0; primitive < kPrimitiveCount * 4U; ++primitive) {
    exponents[primitive] = 0.45 + 0.07 * static_cast<double>(primitive % 7U);
    primitive_coefficients[primitive] =
        0.8 / (1.0 + 0.1 * static_cast<double>(primitive % kPrimitiveCount));
  }
  for (std::size_t column = 0; column < n; ++column) {
    for (std::size_t row = 0; row < n; ++row) {
      density[row + column * n] =
          0.03 / (1.0 + static_cast<double>(row > column ? row - column : column - row));
    }
  }
  const GeneratedDpppVec3 base_positions[4] = {
      {0.1, -0.3, 0.2}, {-0.4, 0.2, 0.5},
      {0.6, -0.1, -0.2}, {-0.2, 0.4, -0.6}};
  const std::size_t primitive_pairs_per_shell_pair =
      kPrimitiveCount * kPrimitiveCount;
  std::vector<std::int64_t> primitive_pair_offsets = {
      0,
      static_cast<std::int64_t>(primitive_pairs_per_shell_pair),
      static_cast<std::int64_t>(2U * primitive_pairs_per_shell_pair),
  };
  std::vector<GeneratedDpppPrimitivePairData> primitive_pairs(
      2U * primitive_pairs_per_shell_pair);
  for (unsigned shell_pair = 0; shell_pair < 2U; ++shell_pair) {
    const unsigned first_center = shell_pair * 2U;
    const unsigned second_center = first_center + 1U;
    for (unsigned first_primitive = 0; first_primitive < kPrimitiveCount;
         ++first_primitive) {
      for (unsigned second_primitive = 0; second_primitive < kPrimitiveCount;
           ++second_primitive) {
        const double alpha =
            exponents[first_center * kPrimitiveCount + first_primitive];
        const double beta =
            exponents[second_center * kPrimitiveCount + second_primitive];
        const double exponent_sum = alpha + beta;
        const double reduced_exponent = alpha * beta / exponent_sum;
        const double dx = base_positions[first_center].x -
            base_positions[second_center].x;
        const double dy = base_positions[first_center].y -
            base_positions[second_center].y;
        const double dz = base_positions[first_center].z -
            base_positions[second_center].z;
        const std::size_t ordinal =
            shell_pair * primitive_pairs_per_shell_pair +
            first_primitive * kPrimitiveCount + second_primitive;
        GeneratedDpppPrimitivePairData& pair = primitive_pairs[ordinal];
        pair.exponent_sum = exponent_sum;
        pair.reduced_exponent = reduced_exponent;
        pair.product_center = {
            (alpha * base_positions[first_center].x +
             beta * base_positions[second_center].x) / exponent_sum,
            (alpha * base_positions[first_center].y +
             beta * base_positions[second_center].y) / exponent_sum,
            (alpha * base_positions[first_center].z +
             beta * base_positions[second_center].z) / exponent_sum,
        };
        pair.weighted_coefficient =
            primitive_coefficients[
                first_center * kPrimitiveCount + first_primitive] *
            primitive_coefficients[
                second_center * kPrimitiveCount + second_primitive] *
            exp(-reduced_exponent * (dx * dx + dy * dy + dz * dz));
        pair.first_product_scale = alpha / exponent_sum;
        pair.second_product_scale = beta / exponent_sum;
      }
    }
  }
  for (unsigned task_index = 0; task_index < kTaskCount; ++task_index) {
    GeneratedDpppShellTask& task = tasks[task_index];
    for (unsigned center = 0; center < 4U; ++center) {
      task.primitive_begin[center] = center * kPrimitiveCount;
      task.primitive_end[center] = (center + 1U) * kPrimitiveCount;
      task.atom[center] = QCE_ATOM_INDEX;
      task.shell[center] = task_index * 4U + center;
      positions[task.atom[center]] = base_positions[center];
    }
QCE_AO_OFFSETS
    task.density_offset = 0U;
    task.spin_offset = 0U;
    task.matrix_order = static_cast<std::uint32_t>(n);
    task.shell_pair[0] = 0U;
    task.shell_pair[1] = 1U;
    task.reversed_shell_pair_mask = 0U;
  }

  GeneratedDpppShellTask* device_tasks = nullptr;
  GeneratedDpppVec3* device_positions = nullptr;
  double* device_exponents = nullptr;
  double* device_primitive_coefficients = nullptr;
  std::int64_t* device_primitive_pair_offsets = nullptr;
  GeneratedDpppPrimitivePairData* device_primitive_pairs = nullptr;
  double* device_ao_coefficients = nullptr;
  double* device_density = nullptr;
  double* device_forces = nullptr;
QCE_PERSISTENT_DECLARATIONS
  const std::size_t force_count = QCE_FORCE_COUNT;
  QCE_CUDA_CHECK(cudaMalloc(&device_tasks, tasks.size() * sizeof(tasks[0])));
  QCE_CUDA_CHECK(cudaMalloc(&device_positions, positions.size() * sizeof(positions[0])));
  QCE_CUDA_CHECK(cudaMalloc(&device_exponents, exponents.size() * sizeof(double)));
  QCE_CUDA_CHECK(cudaMalloc(&device_primitive_coefficients,
                            primitive_coefficients.size() * sizeof(double)));
  QCE_CUDA_CHECK(cudaMalloc(
      &device_primitive_pair_offsets,
      primitive_pair_offsets.size() * sizeof(primitive_pair_offsets[0])));
  QCE_CUDA_CHECK(cudaMalloc(
      &device_primitive_pairs,
      primitive_pairs.size() * sizeof(primitive_pairs[0])));
  QCE_CUDA_CHECK(cudaMalloc(&device_ao_coefficients,
                            ao_coefficients.size() * sizeof(double)));
  QCE_CUDA_CHECK(cudaMalloc(&device_density, density.size() * sizeof(double)));
  QCE_CUDA_CHECK(cudaMalloc(&device_forces, force_count * sizeof(double)));
  QCE_CUDA_CHECK(cudaMemcpy(device_tasks, tasks.data(),
                            tasks.size() * sizeof(tasks[0]), cudaMemcpyHostToDevice));
  QCE_CUDA_CHECK(cudaMemcpy(device_positions, positions.data(),
                            positions.size() * sizeof(positions[0]), cudaMemcpyHostToDevice));
  QCE_CUDA_CHECK(cudaMemcpy(device_exponents, exponents.data(),
                            exponents.size() * sizeof(double), cudaMemcpyHostToDevice));
  QCE_CUDA_CHECK(cudaMemcpy(device_primitive_coefficients,
                            primitive_coefficients.data(),
                            primitive_coefficients.size() * sizeof(double),
                            cudaMemcpyHostToDevice));
  QCE_CUDA_CHECK(cudaMemcpy(
      device_primitive_pair_offsets, primitive_pair_offsets.data(),
      primitive_pair_offsets.size() * sizeof(primitive_pair_offsets[0]),
      cudaMemcpyHostToDevice));
  QCE_CUDA_CHECK(cudaMemcpy(
      device_primitive_pairs, primitive_pairs.data(),
      primitive_pairs.size() * sizeof(primitive_pairs[0]),
      cudaMemcpyHostToDevice));
  QCE_CUDA_CHECK(cudaMemcpy(device_ao_coefficients, ao_coefficients.data(),
                            ao_coefficients.size() * sizeof(double), cudaMemcpyHostToDevice));
  QCE_CUDA_CHECK(cudaMemcpy(device_density, density.data(),
                            density.size() * sizeof(double), cudaMemcpyHostToDevice));
QCE_PERSISTENT_SETUP

  auto launch_fused = [&]() {
QCE_FUSED_LAUNCH
  };
  auto launch_recompute = [&]() {
    generated_dppp_component_recompute_rhf_kernel<<<kTaskCount,
        kGeneratedDpppBlockThreads>>>(
        device_tasks, device_exponents, device_primitive_coefficients,
        device_ao_coefficients, device_positions, device_density,
        device_forces, kTaskCount);
  };

  std::vector<double> fused_forces(force_count);
  std::vector<double> recompute_forces(force_count);
  QCE_CUDA_CHECK(cudaMemset(device_forces, 0, force_count * sizeof(double)));
  launch_fused();
  QCE_CUDA_CHECK(cudaGetLastError());
  QCE_CUDA_CHECK(cudaMemcpy(fused_forces.data(), device_forces,
                            force_count * sizeof(double), cudaMemcpyDeviceToHost));
  QCE_CUDA_CHECK(cudaMemset(device_forces, 0, force_count * sizeof(double)));
  launch_recompute();
  QCE_CUDA_CHECK(cudaGetLastError());
  QCE_CUDA_CHECK(cudaMemcpy(recompute_forces.data(), device_forces,
                            force_count * sizeof(double), cudaMemcpyDeviceToHost));
  double maximum_error = 0.0;
  double maximum_force = 0.0;
  for (std::size_t item = 0; item < force_count; ++item) {
    maximum_error = fmax(maximum_error,
                         fabs(fused_forces[item] - recompute_forces[item]));
    maximum_force = fmax(maximum_force, fabs(recompute_forces[item]));
  }

  const float fused_ms = benchmark_kernel(
      launch_fused, device_forces, force_count * sizeof(double));
  const float recompute_ms = benchmark_kernel(
      launch_recompute, device_forces, force_count * sizeof(double));
  std::printf(
      "{\"task_count\":%u,\"primitive_count_per_shell\":%u,"
      "\"primitive_quartets_per_task\":%u,\"consumer\":\"force\","
      "\"topology\":\"QCE_BENCHMARK_TOPOLOGY\","
      "\"fused_ms\":%.9g,"
      "\"recompute_ms\":%.9g,\"speedup\":%.9g,"
      "\"maximum_force_error\":%.17g,\"maximum_force\":%.17g}\n",
      kTaskCount, kPrimitiveCount,
      kPrimitiveCount * kPrimitiveCount * kPrimitiveCount * kPrimitiveCount,
      fused_ms, recompute_ms, recompute_ms / fused_ms,
      maximum_error, maximum_force);

  cudaFree(device_forces);
  cudaFree(device_density);
  cudaFree(device_ao_coefficients);
  cudaFree(device_primitive_pairs);
  cudaFree(device_primitive_pair_offsets);
  cudaFree(device_primitive_coefficients);
  cudaFree(device_exponents);
  cudaFree(device_positions);
  cudaFree(device_tasks);
QCE_PERSISTENT_FREE
  return maximum_error <= 2.0e-10 * fmax(1.0, maximum_force) ? 0 : 3;
}
"""


_FOCK_HOST_HARNESS = r"""
#define QCE_CUDA_CHECK(call) do { \
  const cudaError_t error = (call); \
  if (error != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, \
                 cudaGetErrorString(error)); \
    std::exit(2); \
  } \
} while (false)

constexpr unsigned kTaskCount = QCE_TASK_COUNT;
constexpr unsigned kPrimitiveCount = QCE_PRIMITIVE_COUNT;
constexpr unsigned kWarmups = QCE_WARMUPS;
constexpr unsigned kIterations = QCE_ITERATIONS;
constexpr unsigned kSamples = QCE_SAMPLES;

template <typename Launch>
float benchmark_kernel(Launch launch, double* output, std::size_t output_bytes) {
  std::vector<float> samples;
  samples.reserve(kSamples);
  for (unsigned sample = 0; sample < kSamples; ++sample) {
    QCE_CUDA_CHECK(cudaMemset(output, 0, output_bytes));
    for (unsigned warmup = 0; warmup < kWarmups; ++warmup) launch();
    QCE_CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t begin;
    cudaEvent_t end;
    QCE_CUDA_CHECK(cudaEventCreate(&begin));
    QCE_CUDA_CHECK(cudaEventCreate(&end));
    QCE_CUDA_CHECK(cudaEventRecord(begin));
    for (unsigned iteration = 0; iteration < kIterations; ++iteration) launch();
    QCE_CUDA_CHECK(cudaEventRecord(end));
    QCE_CUDA_CHECK(cudaEventSynchronize(end));
    float milliseconds = 0.0f;
    QCE_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, begin, end));
    QCE_CUDA_CHECK(cudaEventDestroy(begin));
    QCE_CUDA_CHECK(cudaEventDestroy(end));
    samples.push_back(milliseconds / static_cast<float>(kIterations));
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2U];
}

int main() {
  constexpr std::size_t n = QCE_MATRIX_ORDER;
  constexpr std::size_t matrix_size = n * n;
  std::vector<GeneratedDpppShellTask> tasks(kTaskCount);
  std::vector<GeneratedDpppVec3> positions(kTaskCount * 4U);
  std::vector<double> exponents(kPrimitiveCount * 4U);
  std::vector<double> primitive_coefficients(kPrimitiveCount * 4U);
  std::vector<double> ao_coefficients(QCE_AO_COUNT, 1.0);
  std::vector<double> density(kTaskCount * matrix_size);
  for (unsigned primitive = 0; primitive < kPrimitiveCount * 4U; ++primitive) {
    exponents[primitive] = 0.45 + 0.07 * static_cast<double>(primitive % 7U);
    primitive_coefficients[primitive] =
        0.8 / (1.0 + 0.1 * static_cast<double>(primitive % kPrimitiveCount));
  }
  for (unsigned task_index = 0; task_index < kTaskCount; ++task_index) {
    const std::size_t density_begin =
        static_cast<std::size_t>(task_index) * matrix_size;
    for (std::size_t column = 0; column < n; ++column) {
      for (std::size_t row = 0; row < n; ++row) {
        density[density_begin + row + column * n] =
            0.03 /
            (1.0 + static_cast<double>(
                row > column ? row - column : column - row));
      }
    }
  }
  const GeneratedDpppVec3 base_positions[4] = {
      {0.1, -0.3, 0.2}, {-0.4, 0.2, 0.5},
      {0.6, -0.1, -0.2}, {-0.2, 0.4, -0.6}};
  const std::size_t primitive_pairs_per_shell_pair =
      kPrimitiveCount * kPrimitiveCount;
  std::vector<std::int64_t> primitive_pair_offsets = {
      0,
      static_cast<std::int64_t>(primitive_pairs_per_shell_pair),
      static_cast<std::int64_t>(2U * primitive_pairs_per_shell_pair),
  };
  std::vector<GeneratedDpppPrimitivePairData> primitive_pairs(
      2U * primitive_pairs_per_shell_pair);
  for (unsigned shell_pair = 0; shell_pair < 2U; ++shell_pair) {
    const unsigned first_center = shell_pair * 2U;
    const unsigned second_center = first_center + 1U;
    for (unsigned first_primitive = 0; first_primitive < kPrimitiveCount;
         ++first_primitive) {
      for (unsigned second_primitive = 0; second_primitive < kPrimitiveCount;
           ++second_primitive) {
        const double alpha =
            exponents[first_center * kPrimitiveCount + first_primitive];
        const double beta =
            exponents[second_center * kPrimitiveCount + second_primitive];
        const double exponent_sum = alpha + beta;
        const double reduced_exponent = alpha * beta / exponent_sum;
        const double dx = base_positions[first_center].x -
            base_positions[second_center].x;
        const double dy = base_positions[first_center].y -
            base_positions[second_center].y;
        const double dz = base_positions[first_center].z -
            base_positions[second_center].z;
        const std::size_t ordinal =
            shell_pair * primitive_pairs_per_shell_pair +
            first_primitive * kPrimitiveCount + second_primitive;
        GeneratedDpppPrimitivePairData& pair = primitive_pairs[ordinal];
        pair.exponent_sum = exponent_sum;
        pair.reduced_exponent = reduced_exponent;
        pair.product_center = {
            (alpha * base_positions[first_center].x +
             beta * base_positions[second_center].x) / exponent_sum,
            (alpha * base_positions[first_center].y +
             beta * base_positions[second_center].y) / exponent_sum,
            (alpha * base_positions[first_center].z +
             beta * base_positions[second_center].z) / exponent_sum,
        };
        pair.weighted_coefficient =
            primitive_coefficients[
                first_center * kPrimitiveCount + first_primitive] *
            primitive_coefficients[
                second_center * kPrimitiveCount + second_primitive] *
            exp(-reduced_exponent * (dx * dx + dy * dy + dz * dz));
        pair.first_product_scale = alpha / exponent_sum;
        pair.second_product_scale = beta / exponent_sum;
      }
    }
  }
  for (unsigned task_index = 0; task_index < kTaskCount; ++task_index) {
    GeneratedDpppShellTask& task = tasks[task_index];
    for (unsigned center = 0; center < 4U; ++center) {
      task.primitive_begin[center] = center * kPrimitiveCount;
      task.primitive_end[center] = (center + 1U) * kPrimitiveCount;
      task.atom[center] = task_index * 4U + center;
      task.shell[center] = task_index * 4U + center;
      positions[task.atom[center]] = base_positions[center];
    }
QCE_AO_OFFSETS
    task.density_offset = QCE_DENSITY_OFFSET;
    task.spin_offset = 0U;
    task.matrix_order = static_cast<std::uint32_t>(n);
    task.shell_pair[0] = 0U;
    task.shell_pair[1] = 1U;
    task.reversed_shell_pair_mask = 0U;
  }

  GeneratedDpppShellTask* device_tasks = nullptr;
  GeneratedDpppVec3* device_positions = nullptr;
  double* device_exponents = nullptr;
  double* device_primitive_coefficients = nullptr;
  std::int64_t* device_primitive_pair_offsets = nullptr;
  GeneratedDpppPrimitivePairData* device_primitive_pairs = nullptr;
  double* device_ao_coefficients = nullptr;
  double* device_density = nullptr;
  double* device_fock = nullptr;
QCE_PERSISTENT_DECLARATIONS
  const std::size_t fock_count = density.size();
  QCE_CUDA_CHECK(cudaMalloc(&device_tasks, tasks.size() * sizeof(tasks[0])));
  QCE_CUDA_CHECK(cudaMalloc(&device_positions, positions.size() * sizeof(positions[0])));
  QCE_CUDA_CHECK(cudaMalloc(&device_exponents, exponents.size() * sizeof(double)));
  QCE_CUDA_CHECK(cudaMalloc(&device_primitive_coefficients,
                            primitive_coefficients.size() * sizeof(double)));
  QCE_CUDA_CHECK(cudaMalloc(
      &device_primitive_pair_offsets,
      primitive_pair_offsets.size() * sizeof(primitive_pair_offsets[0])));
  QCE_CUDA_CHECK(cudaMalloc(
      &device_primitive_pairs,
      primitive_pairs.size() * sizeof(primitive_pairs[0])));
  QCE_CUDA_CHECK(cudaMalloc(&device_ao_coefficients,
                            ao_coefficients.size() * sizeof(double)));
  QCE_CUDA_CHECK(cudaMalloc(&device_density, density.size() * sizeof(double)));
  QCE_CUDA_CHECK(cudaMalloc(&device_fock, fock_count * sizeof(double)));
  QCE_CUDA_CHECK(cudaMemcpy(device_tasks, tasks.data(),
                            tasks.size() * sizeof(tasks[0]), cudaMemcpyHostToDevice));
  QCE_CUDA_CHECK(cudaMemcpy(device_positions, positions.data(),
                            positions.size() * sizeof(positions[0]), cudaMemcpyHostToDevice));
  QCE_CUDA_CHECK(cudaMemcpy(device_exponents, exponents.data(),
                            exponents.size() * sizeof(double), cudaMemcpyHostToDevice));
  QCE_CUDA_CHECK(cudaMemcpy(device_primitive_coefficients,
                            primitive_coefficients.data(),
                            primitive_coefficients.size() * sizeof(double),
                            cudaMemcpyHostToDevice));
  QCE_CUDA_CHECK(cudaMemcpy(
      device_primitive_pair_offsets, primitive_pair_offsets.data(),
      primitive_pair_offsets.size() * sizeof(primitive_pair_offsets[0]),
      cudaMemcpyHostToDevice));
  QCE_CUDA_CHECK(cudaMemcpy(
      device_primitive_pairs, primitive_pairs.data(),
      primitive_pairs.size() * sizeof(primitive_pairs[0]),
      cudaMemcpyHostToDevice));
  QCE_CUDA_CHECK(cudaMemcpy(device_ao_coefficients, ao_coefficients.data(),
                            ao_coefficients.size() * sizeof(double), cudaMemcpyHostToDevice));
  QCE_CUDA_CHECK(cudaMemcpy(device_density, density.data(),
                            density.size() * sizeof(double), cudaMemcpyHostToDevice));
QCE_PERSISTENT_SETUP

  auto launch_fused = [&]() {
QCE_FUSED_LAUNCH
  };
  auto launch_recompute = [&]() {
    generated_dppp_component_recompute_fock_rhf_kernel<<<kTaskCount,
        kGeneratedDpppFockBlockThreads>>>(
        device_tasks, device_exponents, device_primitive_coefficients,
        device_ao_coefficients, device_positions, device_density,
        device_fock, kTaskCount);
  };

  std::vector<double> fused_fock(fock_count);
  std::vector<double> recompute_fock(fock_count);
  QCE_CUDA_CHECK(cudaMemset(device_fock, 0, fock_count * sizeof(double)));
  launch_fused();
  QCE_CUDA_CHECK(cudaGetLastError());
  QCE_CUDA_CHECK(cudaMemcpy(fused_fock.data(), device_fock,
                            fock_count * sizeof(double), cudaMemcpyDeviceToHost));
  QCE_CUDA_CHECK(cudaMemset(device_fock, 0, fock_count * sizeof(double)));
  launch_recompute();
  QCE_CUDA_CHECK(cudaGetLastError());
  QCE_CUDA_CHECK(cudaMemcpy(recompute_fock.data(), device_fock,
                            fock_count * sizeof(double), cudaMemcpyDeviceToHost));
  double maximum_error = 0.0;
  double maximum_fock = 0.0;
  for (std::size_t item = 0; item < fock_count; ++item) {
    maximum_error = fmax(maximum_error,
                         fabs(fused_fock[item] - recompute_fock[item]));
    maximum_fock = fmax(maximum_fock, fabs(recompute_fock[item]));
  }

  const float fused_ms = benchmark_kernel(
      launch_fused, device_fock, fock_count * sizeof(double));
  const float recompute_ms = benchmark_kernel(
      launch_recompute, device_fock, fock_count * sizeof(double));
  std::printf(
      "{\"task_count\":%u,\"primitive_count_per_shell\":%u,"
      "\"primitive_quartets_per_task\":%u,\"consumer\":\"fock\","
      "\"topology\":\"QCE_BENCHMARK_TOPOLOGY\","
      "\"fused_ms\":%.9g,\"recompute_ms\":%.9g,\"speedup\":%.9g,"
      "\"maximum_fock_error\":%.17g,\"maximum_fock\":%.17g}\n",
      kTaskCount, kPrimitiveCount,
      kPrimitiveCount * kPrimitiveCount * kPrimitiveCount * kPrimitiveCount,
      fused_ms, recompute_ms, recompute_ms / fused_ms,
      maximum_error, maximum_fock);

  cudaFree(device_fock);
  cudaFree(device_density);
  cudaFree(device_ao_coefficients);
  cudaFree(device_primitive_pairs);
  cudaFree(device_primitive_pair_offsets);
  cudaFree(device_primitive_coefficients);
  cudaFree(device_exponents);
  cudaFree(device_positions);
  cudaFree(device_tasks);
QCE_PERSISTENT_FREE
  return maximum_error <= 2.0e-10 * fmax(1.0, maximum_fock) ? 0 : 3;
}
"""


def _benchmark_unfused_kernel(
    spec: ShellClassSpec, plan: FusedShellPlan
) -> str:
    """Specialize the independent per-component baseline for one shell class."""

    names = _emitted_component_names(spec)
    angular_lines = [
        "  const double angular_coefficient = component_lane",
        f"      ? ao_coefficients[shared.task.ao_coefficient_begin[0] + {names[0]}] *",
        f"        ao_coefficients[shared.task.ao_coefficient_begin[1] + {names[1]}] *",
        f"        ao_coefficients[shared.task.ao_coefficient_begin[2] + {names[2]}] *",
        f"        ao_coefficients[shared.task.ao_coefficient_begin[3] + {names[3]}]",
        "      : 0.0;",
    ]
    if plan.schedule.kind == ScheduleKind.TILED_COMPONENTS:
        schedule_setup = f"""  for (unsigned component_tile_begin = 0U;
       component_tile_begin < kGeneratedDpppComponentCount;
       component_tile_begin += {plan.schedule.component_tile}U) {{
  const unsigned tile_component = component_tile_begin + lane;
  const bool component_lane = tile_component < kGeneratedDpppComponentCount;
  const unsigned component = component_lane ? tile_component : 0U;"""
        schedule_close = "  __syncthreads();\n  }"
    else:
        schedule_setup = """  const bool component_lane = lane < kGeneratedDpppComponentCount;
  const unsigned component = component_lane ? lane : 0U;"""
        schedule_close = ""
    source = _UNFUSED_KERNEL.replace(
        "QCE_COMPONENT_SCHEDULE_SETUP", schedule_setup
    ).replace("QCE_COMPONENT_SCHEDULE_CLOSE", schedule_close)
    source = source.replace(
        "QCE_COMPONENT_SETUP", _generic_task_component_setup(spec)
    ).replace("QCE_ANGULAR_COEFFICIENT", "\n".join(angular_lines))
    return _specialize_dppp_identifiers(source, spec)


def _benchmark_unfused_fock_kernel(
    spec: ShellClassSpec, plan: FusedShellPlan
) -> str:
    """Specialize the independent value/Fock baseline for one shell class."""

    names = _emitted_component_names(spec)
    angular_lines = [
        "  const double angular_coefficient = evaluate_component",
        f"      ? ao_coefficients[shared.task.ao_coefficient_begin[0] + {names[0]}] *",
        f"        ao_coefficients[shared.task.ao_coefficient_begin[1] + {names[1]}] *",
        f"        ao_coefficients[shared.task.ao_coefficient_begin[2] + {names[2]}] *",
        f"        ao_coefficients[shared.task.ao_coefficient_begin[3] + {names[3]}]",
        "      : 0.0;",
    ]
    if plan.schedule.kind == ScheduleKind.TILED_COMPONENTS:
        schedule_setup = f"""  for (unsigned component_tile_begin = 0U;
       component_tile_begin < kGeneratedDpppComponentCount;
       component_tile_begin += {plan.schedule.component_tile}U) {{
  const unsigned tile_component = component_tile_begin + lane;
  const bool component_lane = tile_component < kGeneratedDpppComponentCount;
  const unsigned component = component_lane ? tile_component : 0U;"""
        schedule_close = "  __syncthreads();\n  }"
    else:
        schedule_setup = """  const bool component_lane =
      lane < kGeneratedDpppComponentCount;
  const unsigned component = component_lane ? lane : 0U;"""
        schedule_close = ""
    source = _UNFUSED_FOCK_KERNEL.replace(
        "QCE_COMPONENT_SCHEDULE_SETUP", schedule_setup
    ).replace("QCE_COMPONENT_SCHEDULE_CLOSE", schedule_close)
    source = source.replace(
        "QCE_COMPONENT_SETUP", _generic_task_component_setup(spec)
    ).replace("QCE_ANGULAR_COEFFICIENT", "\n".join(angular_lines))
    return _specialize_dppp_identifiers(source, spec)


def _persistent_benchmark_snippets(
    consumer: KernelConsumer,
) -> dict[str, str]:
    """Return host snippets that exercise the production persistent ABI.

    Production launches eight worker blocks per SM and all workers contend on
    one queue head.  Keeping that topology in schedule timing is important:
    ordinary one-shot kernels hide terminal queue claims and greatly
    understate scatter contention for multi-task blocks.
    """

    output = "device_fock" if consumer == KernelConsumer.FOCK else "device_forces"
    kernel = (
        "generated_dppp_shell_class_fock_rhf_persistent_kernel"
        if consumer == KernelConsumer.FOCK
        else "generated_dppp_shell_class_force_rhf_persistent_kernel"
    )
    block_threads = (
        "kGeneratedDpppFockBlockThreads"
        if consumer == KernelConsumer.FOCK
        else "kGeneratedDpppBlockThreads"
    )
    return {
        "QCE_PERSISTENT_DECLARATIONS": """  std::uint32_t* device_task_offset = nullptr;
  std::uint32_t* device_task_count = nullptr;
  std::uint32_t* device_task_head = nullptr;""",
        "QCE_PERSISTENT_SETUP": """  const std::uint32_t host_task_offset = 0U;
  const std::uint32_t host_task_count = kTaskCount;
  cudaDeviceProp device_properties{};
  QCE_CUDA_CHECK(cudaGetDeviceProperties(&device_properties, 0));
  const unsigned persistent_grid_count =
      static_cast<unsigned>(device_properties.multiProcessorCount) * 8U;
  QCE_CUDA_CHECK(cudaMalloc(&device_task_offset, sizeof(std::uint32_t)));
  QCE_CUDA_CHECK(cudaMalloc(&device_task_count, sizeof(std::uint32_t)));
  QCE_CUDA_CHECK(cudaMalloc(&device_task_head, sizeof(std::uint32_t)));
  QCE_CUDA_CHECK(cudaMemcpy(device_task_offset, &host_task_offset,
                            sizeof(std::uint32_t), cudaMemcpyHostToDevice));
  QCE_CUDA_CHECK(cudaMemcpy(device_task_count, &host_task_count,
                            sizeof(std::uint32_t), cudaMemcpyHostToDevice));""",
        "QCE_FUSED_LAUNCH": f"""    QCE_CUDA_CHECK(cudaMemsetAsync(
        device_task_head, 0, sizeof(std::uint32_t)));
    {kernel}<<<persistent_grid_count, {block_threads}>>>(
        device_tasks, device_primitive_pairs, device_primitive_pair_offsets,
        device_ao_coefficients, device_positions, 0.0, nullptr,
        device_density, {output}, device_task_offset, device_task_count,
        device_task_head);""",
        "QCE_PERSISTENT_FREE": """  cudaFree(device_task_head);
  cudaFree(device_task_count);
  cudaFree(device_task_offset);""",
        "QCE_BENCHMARK_TOPOLOGY": "persistent_shared",
    }


def _ordinary_benchmark_snippets(
    consumer: KernelConsumer,
) -> dict[str, str]:
    """Return the legacy isolated one-shot benchmark launch snippets."""

    if consumer == KernelConsumer.FOCK:
        launch = """    generated_dppp_shell_class_fock_rhf_kernel<<<QCE_FUSED_GRID_COUNT,
        kGeneratedDpppFockBlockThreads>>>(
        device_tasks, device_primitive_pairs, device_primitive_pair_offsets,
        device_ao_coefficients, device_positions, 0.0, nullptr,
        device_density, device_fock, kTaskCount);"""
    else:
        launch = """    generated_dppp_shell_class_force_rhf_kernel<<<QCE_FUSED_GRID_COUNT,
        kGeneratedDpppBlockThreads>>>(
        device_tasks, device_primitive_pairs, device_primitive_pair_offsets,
        device_ao_coefficients, device_positions, 0.0, nullptr,
        device_density, device_forces, kTaskCount);"""
    return {
        "QCE_PERSISTENT_DECLARATIONS": "",
        "QCE_PERSISTENT_SETUP": "",
        "QCE_FUSED_LAUNCH": launch,
        "QCE_PERSISTENT_FREE": "",
        "QCE_BENCHMARK_TOPOLOGY": "ordinary_isolated",
    }


def _apply_benchmark_topology(
    source: str,
    consumer: KernelConsumer,
    *,
    persistent_kernel: bool,
) -> str:
    """Specialize host task ownership, scatter targets, and launch ABI."""

    snippets = (
        _persistent_benchmark_snippets(consumer)
        if persistent_kernel
        else _ordinary_benchmark_snippets(consumer)
    )
    if consumer == KernelConsumer.FORCE:
        snippets.update(
            {
                "QCE_ATOM_INDEX": (
                    "center * 6U + ((task_index / "
                    "(center < 2U ? 1024U : 16U)) % 6U)"
                    if persistent_kernel
                    else "task_index * 4U + center"
                ),
                "QCE_FORCE_COUNT": (
                    "24U * 3U" if persistent_kernel else "kTaskCount * 12U"
                ),
            }
        )
    else:
        snippets["QCE_DENSITY_OFFSET"] = (
            "0U"
            if persistent_kernel
            else "static_cast<std::uint64_t>(task_index) * matrix_size"
        )
    for marker, replacement in snippets.items():
        if marker not in source:
            raise RuntimeError(f"benchmark topology marker {marker} is missing")
        source = source.replace(marker, replacement)
    return source


def _benchmark_host_harness(
    spec: ShellClassSpec,
    plan: FusedShellPlan,
    *,
    persistent_kernel: bool,
) -> str:
    """Generate dense AO storage and offsets for a synthetic shell quartet."""

    component_counts = tuple(map(len, spec.center_components))
    offsets = tuple(
        sum(component_counts[:center]) for center in range(4)
    )
    offset_lines = []
    for field in ("ao_begin", "ao_coefficient_begin"):
        offset_lines.extend(
            f"    task.{field}[{center}] = {offset}U;"
            for center, offset in enumerate(offsets)
        )
    ao_count = sum(component_counts)
    source = _HOST_HARNESS.replace("QCE_MATRIX_ORDER", f"{ao_count}U")
    source = source.replace("QCE_AO_COUNT", f"{ao_count}U")
    source = source.replace("QCE_AO_OFFSETS", "\n".join(offset_lines))
    tasks_per_block = plan.schedule.tasks_per_block
    fused_grid_count = (
        f"(kTaskCount + {tasks_per_block - 1}U) / {tasks_per_block}U"
        if tasks_per_block > 1
        else "kTaskCount"
    )
    source = _apply_benchmark_topology(
        source,
        KernelConsumer.FORCE,
        persistent_kernel=persistent_kernel,
    )
    source = source.replace("QCE_FUSED_GRID_COUNT", fused_grid_count)
    return _specialize_dppp_identifiers(source, spec)


def _benchmark_fock_host_harness(
    spec: ShellClassSpec,
    plan: FusedShellPlan,
    *,
    persistent_kernel: bool,
) -> str:
    """Generate isolated per-task density/Fock matrices for value timing."""

    component_counts = tuple(map(len, spec.center_components))
    offsets = tuple(sum(component_counts[:center]) for center in range(4))
    offset_lines = []
    for field in ("ao_begin", "ao_coefficient_begin"):
        offset_lines.extend(
            f"    task.{field}[{center}] = {offset}U;"
            for center, offset in enumerate(offsets)
        )
    ao_count = sum(component_counts)
    source = _FOCK_HOST_HARNESS.replace("QCE_MATRIX_ORDER", f"{ao_count}U")
    source = source.replace("QCE_AO_COUNT", f"{ao_count}U")
    source = source.replace("QCE_AO_OFFSETS", "\n".join(offset_lines))
    tasks_per_block = plan.schedule.tasks_per_block
    fused_grid_count = (
        f"(kTaskCount + {tasks_per_block - 1}U) / {tasks_per_block}U"
        if tasks_per_block > 1
        else "kTaskCount"
    )
    source = _apply_benchmark_topology(
        source,
        KernelConsumer.FOCK,
        persistent_kernel=persistent_kernel,
    )
    source = source.replace("QCE_FUSED_GRID_COUNT", fused_grid_count)
    return _specialize_dppp_identifiers(source, spec)


def _remove_cuda_kernel_definition(source: str, signature: str) -> str:
    """Remove one complete extern CUDA wrapper identified by its signature."""

    signature_begin = source.find(signature)
    if signature_begin < 0:
        raise RuntimeError(f"benchmark wrapper {signature} changed unexpectedly")
    function_begin = source.rfind('extern "C" __global__', 0, signature_begin)
    if function_begin < 0:
        raise RuntimeError(f"benchmark wrapper {signature} has no declaration")
    brace_begin = source.find("{", signature_begin)
    if brace_begin < 0:
        raise RuntimeError(f"benchmark wrapper {signature} has no body")
    depth = 0
    function_end = -1
    for position in range(brace_begin, len(source)):
        character = source[position]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                function_end = position + 1
                break
    if function_end < 0:
        raise RuntimeError(f"benchmark wrapper {signature} has an open body")
    while function_end < len(source) and source[function_end] in "\r\n":
        function_end += 1
    return source[:function_begin] + source[function_end:]


def _retain_selected_persistent_benchmark_kernel(
    source: str,
    spec: ShellClassSpec,
    consumer: KernelConsumer,
) -> str:
    """Keep the production RHF persistent wrapper and its device helpers."""

    prefix = f"generated_{spec.name}"
    fock_section = "/** Coefficient-only pair term used by the SCF Fock recurrence. */"
    if consumer == KernelConsumer.FOCK:
        force_task_signature = source.find(
            f"__device__ __forceinline__ void {prefix}_shell_class_force_task("
        )
        if force_task_signature >= 0:
            force_begin = source.rfind(
                "template <bool Unrestricted>", 0, force_task_signature
            )
        else:
            force_rhf = source.find(
                f"void {prefix}_shell_class_force_rhf_kernel("
            )
            force_begin = source.rfind('extern "C" __global__', 0, force_rhf)
        fock_begin = source.find(fock_section)
        if force_begin < 0 or fock_begin < 0 or force_begin >= fock_begin:
            raise RuntimeError(
                "persistent Fock benchmark force-section markers changed "
                "unexpectedly"
            )
        source = source[:force_begin] + source[fock_begin:]

    stem = f"{prefix}_shell_class_{consumer.value}"
    for signature in (
        f"void {stem}_rhf_kernel(",
        f"void {stem}_uhf_kernel(",
        f"void {stem}_uhf_persistent_kernel(",
    ):
        source = _remove_cuda_kernel_definition(source, signature)
    if f"void {stem}_rhf_persistent_kernel(" not in source:
        raise RuntimeError("persistent RHF benchmark wrapper is missing")
    return source


def _retain_selected_benchmark_kernel(
    source: str,
    spec: ShellClassSpec,
    consumer: KernelConsumer,
    *,
    persistent_kernel: bool = False,
) -> str:
    """Remove production-only wrappers from a timing translation unit.

    Autotuning keeps exactly one RHF entry point so unused UHF companions do
    not dominate NVCC time.  Production-like timing retains the persistent
    wrapper because queue and scatter topology materially affect schedule
    ranking; the standalone diagnostic benchmark may still request the
    ordinary one-shot wrapper.
    """

    if persistent_kernel:
        return _retain_selected_persistent_benchmark_kernel(
            source,
            spec,
            consumer,
        )

    prefix = f"generated_{spec.name}"
    force_uhf = f"void {prefix}_shell_class_force_uhf_kernel("
    fock_section = "/** Coefficient-only pair term used by the SCF Fock recurrence. */"
    fock_uhf = f"void {prefix}_shell_class_fock_uhf_kernel("

    if consumer == KernelConsumer.FORCE:
        unwanted = source.find(force_uhf)
        if unwanted < 0:
            raise RuntimeError("force benchmark wrapper markers changed unexpectedly")
        unwanted = source.rfind('extern "C" __global__', 0, unwanted)
        if unwanted < 0:
            raise RuntimeError("force UHF wrapper declaration changed unexpectedly")
        return source[:unwanted]

    force_task_signature = source.find(
        f"__device__ __forceinline__ void {prefix}_shell_class_force_task("
    )
    if force_task_signature >= 0:
        force_task = source.rfind(
            "template <bool Unrestricted>", 0, force_task_signature
        )
    else:
        # Packed low-order schedules inline their force work directly in the
        # wrapper and therefore have no shared force-task helper to anchor.
        force_rhf = source.find(
            f"void {prefix}_shell_class_force_rhf_kernel("
        )
        force_task = source.rfind(
            'extern "C" __global__', 0, force_rhf
        )
    fock_begin = source.find(fock_section)
    if force_task < 0 or fock_begin < 0 or force_task >= fock_begin:
        raise RuntimeError("Fock benchmark force-section markers changed unexpectedly")
    source = source[:force_task] + source[fock_begin:]
    unwanted = source.find(fock_uhf)
    if unwanted < 0:
        raise RuntimeError("Fock benchmark wrapper markers changed unexpectedly")
    unwanted = source.rfind('extern "C" __global__', 0, unwanted)
    if unwanted < 0:
        raise RuntimeError("Fock UHF wrapper declaration changed unexpectedly")
    return source[:unwanted]


def emit_shell_class_resource_cuda(
    spec: ShellClassSpec,
    plan: FusedShellPlan,
) -> str:
    """Emit every production wrapper without the synthetic benchmark oracle."""

    if plan.spec != spec:
        raise ValueError("resource plan and shell specification do not match")
    return _CUDA_PRELUDE + emit_shell_class_fused_cuda(spec, plan)


def _oracle_kernel_declaration(
    spec: ShellClassSpec,
    consumer: KernelConsumer,
    symbol_prefix: str,
) -> str:
    """Declare a separately compiled oracle against the candidate task ABI."""

    class_name = spec.name[0].upper() + spec.name[1:]
    kernel_suffix = (
        "component_recompute_fock_rhf_kernel"
        if consumer == KernelConsumer.FOCK
        else "component_recompute_rhf_kernel"
    )
    output_name = "fock" if consumer == KernelConsumer.FOCK else "forces"
    return f"""
extern "C" __global__ void {symbol_prefix}_{kernel_suffix}(
    const Generated{class_name}ShellTask* tasks,
    const double* primitive_exponents,
    const double* primitive_coefficients,
    const double* ao_coefficients,
    const Generated{class_name}Vec3* atom_positions,
    const double* density,
    double* {output_name},
    std::size_t task_count);
"""


def emit_shell_class_oracle_cuda(
    spec: ShellClassSpec,
    plan: FusedShellPlan,
    consumer: KernelConsumer | str,
) -> str:
    """Emit one reusable recompute oracle for a structural schedule family."""

    selected_consumer = KernelConsumer(consumer)
    if plan.spec != spec:
        raise ValueError("oracle plan and shell specification do not match")
    fused = _retain_selected_benchmark_kernel(
        emit_shell_class_fused_cuda(spec, plan),
        spec,
        selected_consumer,
    )
    baseline = (
        _benchmark_unfused_fock_kernel(spec, plan)
        if selected_consumer == KernelConsumer.FOCK
        else _benchmark_unfused_kernel(spec, plan)
    )
    source = (
        _CUDA_PRELUDE
        + fused
        + emit_uncached_primitive_geometry_cuda(spec)
        + baseline
    )
    if plan.schedule.kind == ScheduleKind.TILED_COMPONENTS:
        function = (
            "component_value"
            if selected_consumer == KernelConsumer.FOCK
            else "component_gradient"
        )
        needle = (
            f"__device__ __forceinline__ "
            f"{'double' if function == 'component_value' else 'void'} "
            f"generated_{spec.name}_{function}("
        )
        replacement = needle.replace("__forceinline__", "__noinline__")
        if source.count(needle) != 1:
            raise RuntimeError("oracle recurrence marker changed unexpectedly")
        source = source.replace(needle, replacement)
    return source


def emit_shell_class_benchmark_cuda(
    spec: ShellClassSpec,
    task_count: int,
    primitive_count: int,
    warmups: int,
    iterations: int,
    samples: int,
    *,
    plan: FusedShellPlan | None = None,
    schedule: ScheduleIR | None = None,
    consumer: KernelConsumer | str = KernelConsumer.FORCE,
    benchmark_kernel_only: bool = False,
    persistent_kernel: bool = False,
    oracle_symbol_prefix: str | None = None,
) -> str:
    """Return a self-contained benchmark for one explicit schedule.

    ``plan`` is useful when callers already lowered a multi-consumer
    ``KernelIR``. ``schedule`` is the lightweight autotuning entry point.
    Fock timing lowers a joint Fock/force plan because the production source
    deliberately shares one canonical task ABI with the force companion.
    """

    positive_values = (task_count, primitive_count, iterations, samples)
    if any(value <= 0 for value in positive_values) or warmups < 0:
        raise ValueError("benchmark sizes must be positive and warmups non-negative")
    if plan is not None and schedule is not None:
        raise ValueError("pass either a fused plan or a schedule, not both")
    selected_consumer = KernelConsumer(consumer)
    consumers = (
        (KernelConsumer.FOCK, KernelConsumer.FORCE)
        if selected_consumer == KernelConsumer.FOCK
        else (KernelConsumer.FORCE,)
    )
    selected_plan = plan or build_fused_shell_plan(
        spec,
        consumers=consumers,
        schedule=schedule,
    )
    if selected_plan.spec != spec:
        raise ValueError("benchmark plan and shell specification do not match")
    if selected_consumer not in selected_plan.kernel.integral.consumers:
        raise ValueError(
            f"{selected_consumer.value} benchmark requires its consumer"
        )
    if selected_consumer == KernelConsumer.FOCK:
        host = _benchmark_fock_host_harness(
            spec,
            selected_plan,
            persistent_kernel=persistent_kernel,
        )
        baseline = _benchmark_unfused_fock_kernel(spec, selected_plan)
    else:
        host = _benchmark_host_harness(
            spec,
            selected_plan,
            persistent_kernel=persistent_kernel,
        )
        baseline = _benchmark_unfused_kernel(spec, selected_plan)
    replacements = {
        "QCE_TASK_COUNT": str(task_count),
        "QCE_PRIMITIVE_COUNT": str(primitive_count),
        "QCE_WARMUPS": str(warmups),
        "QCE_ITERATIONS": str(iterations),
        "QCE_SAMPLES": str(samples),
    }
    for marker, value in replacements.items():
        host = host.replace(marker, value)
    fused = emit_shell_class_fused_cuda(spec, selected_plan)
    if benchmark_kernel_only:
        fused = _retain_selected_benchmark_kernel(
            fused,
            spec,
            selected_consumer,
            persistent_kernel=persistent_kernel,
        )
    oracle_declaration = ""
    if oracle_symbol_prefix is not None:
        kernel_suffix = (
            "component_recompute_fock_rhf_kernel"
            if selected_consumer == KernelConsumer.FOCK
            else "component_recompute_rhf_kernel"
        )
        local_name = f"generated_{spec.name}_{kernel_suffix}"
        oracle_name = f"{oracle_symbol_prefix}_{kernel_suffix}"
        if host.count(local_name) != 1:
            raise RuntimeError("benchmark oracle launch marker changed unexpectedly")
        host = host.replace(local_name, oracle_name)
        oracle_declaration = _oracle_kernel_declaration(
            spec,
            selected_consumer,
            oracle_symbol_prefix,
        )
        baseline = ""
    return (
        _CUDA_PRELUDE
        + fused
        + emit_uncached_primitive_geometry_cuda(spec)
        + oracle_declaration
        + baseline
        + host
    )


def emit_dppp_benchmark_cuda(
    task_count: int,
    primitive_count: int,
    warmups: int,
    iterations: int,
    samples: int,
    *,
    schedule: ScheduleIR | None = None,
    consumer: KernelConsumer | str = KernelConsumer.FORCE,
) -> str:
    """Return the production-golden dppp standalone benchmark."""

    return emit_shell_class_benchmark_cuda(
        DPPP_SPEC,
        task_count,
        primitive_count,
        warmups,
        iterations,
        samples,
        schedule=schedule,
        consumer=consumer,
    )


def _runtime_environment(nvcc: Path) -> dict[str, str]:
    """Make the selected toolkit's CUDA runtime visible to the executable."""

    environment = dict(os.environ)
    # The shared development shell exports an empty value to keep ordinary
    # tests off the GPU. Treat that sentinel like ``env -u`` for this explicit
    # benchmark command; non-empty user selections remain authoritative.
    if environment.get("CUDA_VISIBLE_DEVICES") == "":
        environment.pop("CUDA_VISIBLE_DEVICES")
    library = nvcc.parent.parent / "lib64"
    previous = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = (
        str(library) if not previous else f"{library}:{previous}"
    )
    return environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--architecture", default="sm_120")
    parser.add_argument("--srun", default="srun")
    parser.add_argument("--partition", default="main")
    parser.add_argument("--gres", default="gpu:5090:1")
    parser.add_argument(
        "--shell-class", choices=("dppp", "dpds", "ddps"), default="dppp"
    )
    parser.add_argument(
        "--consumer",
        choices=tuple(item.value for item in KernelConsumer),
        default=KernelConsumer.FORCE.value,
    )
    parser.add_argument("--tasks", type=int, default=512)
    parser.add_argument("--primitives", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--keep-source",
        type=Path,
        help="also write the generated standalone CUDA source to this path",
    )
    arguments = parser.parse_args()
    specification = {
        "dppp": DPPP_SPEC,
        "dpds": DPDS_SPEC,
        "ddps": DDPS_SPEC,
    }[arguments.shell_class]
    source = emit_shell_class_benchmark_cuda(
        specification,
        arguments.tasks,
        arguments.primitives,
        arguments.warmups,
        arguments.iterations,
        arguments.samples,
        consumer=arguments.consumer,
    )
    if arguments.keep_source is not None:
        arguments.keep_source.parent.mkdir(parents=True, exist_ok=True)
        arguments.keep_source.write_text(source, encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="qce-dppp-benchmark-") as temporary:
        directory = Path(temporary)
        cuda_source = directory / "dppp_benchmark.cu"
        executable = directory / "dppp_benchmark"
        cuda_source.write_text(source, encoding="utf-8")
        compile_result = subprocess.run(
            [
                str(arguments.nvcc),
                "-std=c++17",
                f"-arch={arguments.architecture}",
                "-O3",
                "-Xptxas=-v",
                str(cuda_source),
                "-o",
                str(executable),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
        )
        if compile_result.stdout:
            print(compile_result.stdout, file=sys.stderr, end="")
        if compile_result.stderr:
            print(compile_result.stderr, file=sys.stderr, end="")
        if compile_result.returncode != 0:
            raise SystemExit(compile_result.returncode)
        run_result = subprocess.run(
            [
                arguments.srun,
                f"--partition={arguments.partition}",
                f"--gres={arguments.gres}",
                "--nodes=1",
                "--ntasks=1",
                str(executable),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=180,
            env=_runtime_environment(arguments.nvcc),
        )
        if run_result.stderr:
            print(run_result.stderr, file=sys.stderr, end="")
        if run_result.stdout:
            print(run_result.stdout, end="")
        if run_result.returncode != 0:
            raise SystemExit(run_result.returncode)


if __name__ == "__main__":
    main()
