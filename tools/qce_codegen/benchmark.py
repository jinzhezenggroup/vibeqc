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

  const bool component_lane = lane < kGeneratedDpppComponentCount;
  const unsigned component = component_lane ? lane : 0U;
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
      task.atom[center] = task_index * 4U + center;
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
  const std::size_t force_count = kTaskCount * 12U;
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

  auto launch_fused = [&]() {
    generated_dppp_shell_class_force_rhf_kernel<<<kTaskCount,
        kGeneratedDpppBlockThreads>>>(
        device_tasks, device_primitive_pairs, device_primitive_pair_offsets,
        device_ao_coefficients, device_positions, 0.0, nullptr,
        device_density, device_forces, kTaskCount);
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
      "\"primitive_quartets_per_task\":%u,\"fused_ms\":%.9g,"
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
  return maximum_error <= 2.0e-10 * fmax(1.0, maximum_force) ? 0 : 3;
}
"""


def _benchmark_unfused_kernel(spec: ShellClassSpec) -> str:
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
    source = _UNFUSED_KERNEL.replace(
        "QCE_COMPONENT_SETUP", _generic_task_component_setup(spec)
    ).replace("QCE_ANGULAR_COEFFICIENT", "\n".join(angular_lines))
    return _specialize_dppp_identifiers(source, spec)


def _benchmark_host_harness(spec: ShellClassSpec) -> str:
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
    return _specialize_dppp_identifiers(source, spec)


def emit_shell_class_benchmark_cuda(
    spec: ShellClassSpec,
    task_count: int,
    primitive_count: int,
    warmups: int,
    iterations: int,
    samples: int,
) -> str:
    """Return a self-contained fused-vs-recomputed shell benchmark."""

    positive_values = (task_count, primitive_count, iterations, samples)
    if any(value <= 0 for value in positive_values) or warmups < 0:
        raise ValueError("benchmark sizes must be positive and warmups non-negative")
    host = _benchmark_host_harness(spec)
    replacements = {
        "QCE_TASK_COUNT": str(task_count),
        "QCE_PRIMITIVE_COUNT": str(primitive_count),
        "QCE_WARMUPS": str(warmups),
        "QCE_ITERATIONS": str(iterations),
        "QCE_SAMPLES": str(samples),
    }
    for marker, value in replacements.items():
        host = host.replace(marker, value)
    return (
        _CUDA_PRELUDE
        + emit_shell_class_fused_cuda(spec)
        + emit_uncached_primitive_geometry_cuda(spec)
        + _benchmark_unfused_kernel(spec)
        + host
    )


def emit_dppp_benchmark_cuda(
    task_count: int,
    primitive_count: int,
    warmups: int,
    iterations: int,
    samples: int,
) -> str:
    """Return the production-golden dppp standalone benchmark."""

    return emit_shell_class_benchmark_cuda(
        DPPP_SPEC,
        task_count,
        primitive_count,
        warmups,
        iterations,
        samples,
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
