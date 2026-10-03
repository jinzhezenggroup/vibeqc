"""CUDA lowering composition for the bounded stationary RKS diagnostic.

Primitive recurrences, Becke local AD and AO bilinear AD remain their existing
compiler programs; their bounded primitive/geometry contractions are emitted here.
Native code owns allocation, validation, transfers, launches and ABI only.
Generation is host-only and does not import the public runtime or probe CUDA.

Rationale: .agents/notes/implemented/architecture/2026-09-20-stationary-cuda-emitted-contractions.md
"""

import json
import os
import typing
from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from pathlib import Path

from generativeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from generativeqc_compiler.common.cuda_runtime import CudaArtifact
from generativeqc_compiler.common.native_runtime import (
    compile_cuda_object,
    link_cuda_objects,
)
from generativeqc_compiler.common.paths import asset_path, source_hashes
from generativeqc_compiler.common.provenance import canonical_hash, file_hash
from generativeqc_compiler.common.source_cache import cache_source
from generativeqc_compiler.method.stationary_resources import (
    BECKE_COOPERATIVE_CONTROL_BYTES,
    BECKE_COOPERATIVE_MAX_ATOMS,
    BECKE_COOPERATIVE_THREADS,
    BECKE_PAIR_TILE_ROWS,
    BECKE_RETAINED_MAX_ATOMS,
    GEOMETRY_MAX_LANES,
    GEOMETRY_MAX_SCRATCH_BYTES,
    GEOMETRY_THREADS,
)
from generativeqc_compiler.tensor.cuda_inline import (
    InlineCudaOutput,
    exact_cuda_literal,
    lower_inline_cuda_output,
)
from generativeqc_compiler.xc.geometry_cuda import emit_geometry_cuda

from .spec import SemilocalXCPrimitive, resolve_method
from .stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)

STATIONARY_RUNTIME_SOURCE_NAMES = (
    "one_electron",
    "coulomb",
    "xc_ao",
    "xc_grid",
    "xc_weight",
    "overlap_pulay",
    "nuclear",
)
_FUSED_WEIGHT_SOURCES = ("one_electron", "coulomb", "overlap_pulay")
_SPLIT_COMPILE_THREADS_ENV = "GENERATIVEQC_STATIONARY_CUDA_SPLIT_COMPILE_THREADS"


def stationary_runtime_sources(plan: StationaryGradientPlan) -> tuple[str, ...]:
    """Retain plan order for contributions owned by the bounded CUDA arena.

    ECP and range/nonlocal providers retain separate ownership. Full-range
    exchange uses the same ERI derivative provider as Coulomb, with its own
    same-spin density pairing and compiler-derived weight.
    """
    return tuple(
        source
        for source in plan.source_names
        if source in (*STATIONARY_RUNTIME_SOURCE_NAMES, "exact_exchange")
    )


_NATIVE_EXTERNAL_PROVIDER_SOURCES = frozenset(("ecp_local", "ecp_nonlocal"))


def stationary_external_provider_sources(
    plan: StationaryGradientPlan,
) -> tuple[str, ...]:
    """Return plan sources intentionally owned outside the shared CUDA arena."""
    if not isinstance(plan, StationaryGradientPlan):
        raise TypeError("stationary source inventory requires StationaryGradientPlan")
    runtime = set(stationary_runtime_sources(plan))
    return tuple(
        source
        for source in plan.source_names
        if source not in runtime and source not in _NATIVE_EXTERNAL_PROVIDER_SOURCES
    )


def _runtime_layout_cuda(plan: StationaryGradientPlan) -> str:
    sources = stationary_runtime_sources(plan)
    integral_slots = [
        sources.index(s)
        for s in (*_FUSED_WEIGHT_SOURCES, "exact_exchange")
        if s in sources
    ]
    return "\n".join(
        (
            "namespace generativeqc_stationary_cuda {",
            f"constexpr unsigned stationary_source_count = {len(sources)};",
            f"constexpr size_t stationary_becke_max_atoms = {BECKE_COOPERATIVE_MAX_ATOMS};",
            f"constexpr size_t stationary_becke_retained_max_atoms = {BECKE_RETAINED_MAX_ATOMS};",
            f"constexpr size_t stationary_becke_pair_tile_rows = {BECKE_PAIR_TILE_ROWS};",
            f"constexpr size_t stationary_becke_threads = {BECKE_COOPERATIVE_THREADS};",
            f"constexpr size_t stationary_becke_control_bytes = {BECKE_COOPERATIVE_CONTROL_BYTES};",
            f"constexpr size_t stationary_geometry_max_lanes = {GEOMETRY_MAX_LANES};",
            f"constexpr size_t stationary_geometry_max_scratch_bytes = {GEOMETRY_MAX_SCRATCH_BYTES};",
            f"constexpr size_t stationary_geometry_max_threads = {GEOMETRY_THREADS};",
            f"constexpr unsigned stationary_nuclear_source = {sources.index('nuclear')};",
            f"constexpr unsigned stationary_xc_source = {sources.index('xc_ao')};",
            "__host__ __device__ inline bool stationary_integral_source(int64_t source) {",
            "  return "
            + " || ".join(f"source == {slot}" for slot in integral_slots)
            + ";",
            "}",
            "}",
            "",
        )
    )


def _split_compile_options(
    environment: typing.Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Return explicit NVCC split-compilation flags for this oversized runtime."""

    env = os.environ if environment is None else environment
    raw = env.get(_SPLIT_COMPILE_THREADS_ENV, "1")
    try:
        threads = int(raw)
    except ValueError as error:
        raise ValueError(f"{_SPLIT_COMPILE_THREADS_ENV} must be an integer") from error
    if not 1 <= threads <= 32:
        raise ValueError(f"{_SPLIT_COMPILE_THREADS_ENV} must be in [1,32]")
    return () if threads == 1 else (f"--split-compile={threads}",)


def _weight_expression(
    plan: StationaryGradientPlan, source: str
) -> tuple[InlineCudaOutput, int]:
    """Specialize one generated weight through the shared inline-consumer path."""
    program = plan.integral_block(source, terms=1).weights
    locations = {
        "density_left": ("density", 0, 1),
        "density_right": ("density", 2, 3),
        "weighted_density": ("weighted_density", 0, 1),
    }
    if source == "exact_exchange":
        # Ordered (ab|cd) derivatives contract D[a,c] D[b,d], independently
        # in each spin channel. Coulomb instead uses D[a,b] D[c,d].
        locations["density_left"] = ("density", 0, 2)
        locations["density_right"] = ("density", 1, 3)
    bindings = {
        name: tuple(
            f"{pointer}[{spin} * n * n + size_t(ao[{left}]) * n + size_t(ao[{right}])]"
            for spin in range(plan.spin_blocks)
        )
        for name, (pointer, left, right) in locations.items()
    }
    lowered = lower_inline_cuda_output(program, output="weights", bindings=bindings)
    required = set(lowered.required_inputs)
    arity = max(
        (right + 1 for name, (_, _, right) in locations.items() if name in required),
        default=0,
    )
    return lowered, arity


def emit_stationary_weight_cuda(plan: typing.Any) -> str:
    """Emit pointwise device weights directly from StationaryGradientPlan TensorIR."""
    if not isinstance(plan, StationaryGradientPlan):
        raise TypeError(
            "stationary CUDA weight lowering requires StationaryGradientPlan"
        )
    functions = [
        "namespace generativeqc_stationary_cuda {",
        f"// stationary-plan: {plan.identity}",
    ]
    dispatch: list[str] = []
    sources = stationary_runtime_sources(plan)
    for source in (
        *_FUSED_WEIGHT_SOURCES,
        *(("exact_exchange",) if plan.exchange is not None else ()),
    ):
        lowered, arity = _weight_expression(plan, source)
        symbol = f"stationary_weight_{source}"
        functions.extend(
            (
                f"// stationary-weight-program-{source}: {lowered.original_logical_hash}",
                f"// stationary-weight-specialization-{source}: {lowered.specialization_logical_hash}",
                f"// stationary-weight-lowered-{source}: {lowered.optimized_logical_hash}",
                f"// stationary-weight-optimizer-{source}: {lowered.optimizer_identity}",
                f"__device__ inline double {symbol}(const double* density, const double* weighted_density, size_t n, const int64_t* ao) {{",
                f"  return {lowered.expression};",
                "}",
            )
        )
        slot = sources.index(source)
        checks = " || ".join(
            f"ao[{i}] < 0 || ao[{i}] >= int64_t(n)" for i in range(arity)
        )
        dispatch.extend(
            (
                f"    case {slot}:",
                f"      if ({checks}) return false;",
                f"      value = {symbol}(density, weighted_density, n, ao);",
                "      return isfinite(value);",
            )
        )
    functions.extend(
        (
            "__device__ inline bool stationary_source_weight(unsigned source, const double* density, const double* weighted_density, size_t n, const int64_t* ao, double& value) {",
            "  switch (source) {",
            *dispatch,
            "    default: return false;",
            "  }",
            "}",
            "}  // namespace generativeqc_stationary_cuda",
            "",
        )
    )
    return "\n".join(functions)


_STATIONARY_SCIENTIFIC_KERNELS = r"""namespace generativeqc_stationary_cuda {
__global__ void task_kernel(
    const int64_t* tasks, const double* charges, size_t count, const double* primitives,
    size_t nprimitive, const int64_t* ao_ranges, const double* ao_norms,
    const int64_t* ao_atoms, const double* centers, const double* density,
    const double* weighted_density, size_t nao, size_t na, double* output, int* error) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += blockDim.x * gridDim.x) {
    const int64_t* task = tasks + task_stride * i;
    const auto kind = unsigned(task[0]);
    const auto source = task[1];
    const auto rank = task[2];
    const auto nucleus = task[3];
    if (!stationary_integral_source(source) || (rank != 2 && rank != 4) ||
        (nucleus >= 0 && (rank != 2 || nucleus >= int64_t(na)))) {
      atomicExch(error, 1);
      return;
    }
    double source_weight = 1.0;
    if (!stationary_source_weight(unsigned(source), density, weighted_density, nao,
                                  task + 4, source_weight) || !isfinite(charges[i])) {
      atomicExch(error, 1);
      return;
    }
    size_t starts[4]{}, counts[4]{};
    size_t primitive_work = 1;
    for (size_t center = 0; center < size_t(rank); ++center) {
      const auto ao = task[4 + center];
      if (ao < 0 || ao >= int64_t(nao)) {
        atomicExch(error, 1);
        return;
      }
      const auto begin = ao_ranges[2 * ao];
      const auto extent = ao_ranges[2 * ao + 1];
      if (begin < 0 || extent <= 0 || begin > int64_t(nprimitive) ||
          extent > int64_t(nprimitive) - begin) {
        atomicExch(error, 1);
        return;
      }
      starts[center] = size_t(begin);
      counts[center] = size_t(extent);
      primitive_work *= counts[center];
    }
    if (task[8] <= 0 || uint64_t(task[8]) != primitive_work) {
      atomicExch(error, 1);
      return;
    }
    double accumulated[12]{};
    for (size_t linear = 0; linear < primitive_work; ++linear) {
      double r[record_stride];
      for (size_t j = 0; j < record_stride; ++j) r[j] = 1.0;
      size_t cursor = linear;
      for (size_t center = size_t(rank); center-- > 0;) {
        const auto ao = task[4 + center];
        const size_t primitive = starts[center] + cursor % counts[center];
        cursor /= counts[center];
        const auto atom = ao_atoms[ao];
        if (atom < 0 || atom >= int64_t(na) || !isfinite(ao_norms[ao])) {
          atomicExch(error, 1);
          return;
        }
        r[center] = primitives[2 * primitive];
        r[16 + center] = primitives[2 * primitive + 1];
        r[20 + center] = ao_norms[ao];
        for (size_t k = 0; k < 3; ++k) r[4 + 3 * center + k] = centers[3 * atom + k];
      }
      if (nucleus >= 0)
        for (size_t k = 0; k < 3; ++k)
          r[4 + 3 * size_t(rank) + k] = centers[3 * size_t(nucleus) + k];
      r[25] = charges[i];
      double v[12]{};
      for (size_t j = 0; j < record_stride; ++j)
        if (!isfinite(r[j])) {
          atomicExch(error, 1);
          return;
        }
      for (size_t j = 0; j < 4; ++j)
        if (!(r[j] > 0)) {
          atomicExch(error, 1);
          return;
        }
      if (!first_derivative(kind, r, r + 4, v)) {
        atomicExch(error, 1);
        return;
      }
      double weight = source_weight * r[25];
      for (size_t j = 0; j < 4; ++j) weight *= r[16 + j] * r[20 + j];
      for (size_t j = 0; j < 12; ++j)
        accumulated[j] += finite(weight * v[j], error, 0);
    }
    for (size_t j = 0; j < 12; ++j)
      output[12 * i + j] = finite(accumulated[j], error, 0);
  }
}
__global__ void task_reduce(const double* input, const int64_t* tasks, size_t count,
                            const int64_t* ao_atoms, size_t na, double* output, int* error) {
  if (*error) return;
  const size_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= 3 * stationary_source_count * na) return;
  const size_t source = slot / (3 * na);
  if (!stationary_integral_source(int64_t(source))) return;
  const size_t coord = slot % (3 * na);
  double sum = 0;
  for (size_t i = 0; i < count; ++i) {
    const int64_t* task = tasks + task_stride * i;
    if (task[1] != int64_t(source)) continue;
    const size_t rank = size_t(task[2]);
    for (size_t center = 0; center < rank; ++center) {
      const auto atom = ao_atoms[task[4 + center]];
      if (atom == int64_t(coord / 3))
        sum += input[12 * i + 3 * center + coord % 3];
    }
    if (task[3] == int64_t(coord / 3))
      sum += input[12 * i + 3 * rank + coord % 3];
  }
  output[slot] = finite(output[slot] + sum, error, 0);
}
__device__ bool nuclear_pair(unsigned kind, int64_t a, int64_t b, double za, double zb,
                             const double* centers, size_t na, double* output, int* error) {
  if (a < 0 || b < 0 || a >= int64_t(na) || b >= int64_t(na) || a == b ||
      !isfinite(za) || !isfinite(zb) || !(za > 0) || !(zb > 0)) {
    atomicExch(error, 1);
    return false;
  }
  double r[record_stride];
  for (size_t j = 0; j < record_stride; ++j) r[j] = 1.0;
  r[0] = za;
  r[1] = zb;
  for (size_t k = 0; k < 3; ++k) {
    r[4 + k] = centers[3 * a + k];
    r[7 + k] = centers[3 * b + k];
  }
  double v[12]{};
  if (!first_derivative(kind, r, r + 4, v)) {
    atomicExch(error, 1);
    return false;
  }
  for (size_t k = 0; k < 3; ++k) {
    output[3 * stationary_nuclear_source * na + 3 * a + k] =
        finite(output[3 * stationary_nuclear_source * na + 3 * a + k] + v[k], error, 0);
    output[3 * stationary_nuclear_source * na + 3 * b + k] =
        finite(output[3 * stationary_nuclear_source * na + 3 * b + k] + v[3 + k], error, 0);
  }
  return !*error;
}
__global__ void nuclear_kernel(unsigned kind, int64_t a, int64_t b, double za, double zb,
                               const double* centers, size_t na, double* output, int* error) {
  nuclear_pair(kind, a, b, za, zb, centers, na, output, error);
}
__global__ void nuclear_all_kernel(unsigned kind, const double* charges, const double* centers,
                                   size_t na, double* output, int* error) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  for (size_t a = 0; a < na; ++a)
    for (size_t b = 0; b < a; ++b)
      if (!nuclear_pair(kind, int64_t(a), int64_t(b), charges[a], charges[b], centers, na, output,
                        error))
        return;
}
__global__ void validate_centers(const double* centers, size_t na, double tolerance,
                                generativeqc_grid_adjoint::CenterPair* center_pairs, int* error) {
  if (!generativeqc_grid_adjoint::prepare_center_geometry(
          centers, na, tolerance, center_pairs, local_norm, local_ratio_geometry))
    atomicExch(error, 1);
}
__device__ bool geometry_point_setup(generativeqc::dft::GridTaskView view,
    size_t p, size_t owner, size_t na,
    const double* raw, const double* external, size_t external_stride, size_t external_offset,
    double* grad, StationaryPointValue& xc, double& becke_seed, int* error) {
  const size_t np = view.npoint;
  double rho[2]{view.features[p], view.features[5 * np + p]}, g[2][3]{}, tau[2]{};
  if (stationary_functional != 0)
    for (size_t s = 0; s < 2; ++s)
      for (size_t k = 0; k < 3; ++k) g[s][k] = view.features[(5 * s + k + 1) * np + p];
  if (stationary_coefficients == 5)
    for (size_t s = 0; s < 2; ++s) tau[s] = view.features[(5 * s + 4) * np + p];
  // The exact shared SCF point model, including vacuum/spin boundaries.
  xc = StationaryPointValue{};
  if (external) {
    // Nonlocal E supplies partials in total rho/sigma, explicit pair
    // coordinates and both weight legs. Device-resident callers may lend a
    // full-grid [6,stride] seed owner and select one tile by offset, avoiding
    // any host or device repack. Validate all six borrowed values before use.
    if (external_stride < external_offset ||
        np > external_stride - external_offset) {
      atomicExch(error, 1);
      return false;
    }
    const size_t ep = external_offset + p;
    double seed[6];
    for (size_t k = 0; k < 6; ++k) {
      seed[k] = external[k * external_stride + ep];
      if (!isfinite(seed[k])) {
        atomicExch(error, 1);
        return false;
      }
    }
    xc.energy = seed[5];
    for (size_t s = 0; s < 2; ++s) {
      xc.rho[s] = seed[0];
      for (size_t k = 0; k < 3; ++k)
        xc.gradient[s][k] = 2.0 * seed[1] * (g[0][k] + g[1][k]);
    }
    for (size_t k = 0; k < 3; ++k)
      grad[3 * na + 3 * owner + k] += seed[2 + k];
  } else {
    xc = stationary_evaluate_point(rho, g, tau);
  }
  if (!xc.valid) {
    atomicExch(error, 1);
    return false;
  }
  becke_seed = xc.energy * raw[p];
  return true;
}

__device__ void geometry_ao_gradient(generativeqc::dft::GridTaskView view, const double* work,
    size_t point, size_t ao_index, double weight, const StationaryPointValue& xc,
    double* gradient) {
  const size_t stride = view.npoint * view.nactive;
  const size_t offset = point * view.nactive + ao_index;
  double pullback[4]{};
  for (size_t spin = 0; spin < 2; ++spin) {
    double coefficients[5]{weight * xc.rho[spin]}, products[4]{};
    for (size_t jet = 0; jet < stationary_jets; ++jet) {
      products[jet] = work[(4 * spin + jet) * stride + offset];
      if (jet) coefficients[jet] = weight * xc.gradient[spin][jet - 1];
    }
    if (stationary_coefficients == 5) coefficients[4] = weight * xc.kinetic[spin];
    double local[4]{};
    ao_pullback(coefficients, products, local);
    for (size_t jet = 0; jet < stationary_jets; ++jet) pullback[jet] += local[jet];
  }
  for (size_t axis = 0; axis < 3; ++axis) {
    double value = 0;
    for (size_t jet = 0; jet < stationary_jets; ++jet)
      value += pullback[jet] * view.ao[stationary_shift[jet][axis] * stride + offset];
    gradient[axis] = value;
  }
}

__device__ bool geometry_point_ao(generativeqc::dft::GridTaskView view, const double* work,
    const int64_t* ao_atoms, size_t p, size_t owner, size_t na, const double* weights,
    const double* raw, const double* external, size_t external_stride, size_t external_offset,
    double* grad, double& becke_seed, int* error) {
  StationaryPointValue xc;
  if (!geometry_point_setup(view, p, owner, na, raw, external, external_stride, external_offset,
                            grad, xc, becke_seed, error)) return false;
  const size_t n = view.nactive;
  for (size_t mu = 0; mu < n; ++mu) {
    const size_t global_ao = view.ao_ids ? view.ao_ids[mu] : mu;
    if (global_ao >= view.nao) {
      atomicExch(error, 1);
      return false;
    }
    const auto atom = ao_atoms[global_ao];
    if (atom < 0 || atom >= int64_t(na)) {
      atomicExch(error, 1);
      return false;
    }
    double gradient[3];
    geometry_ao_gradient(view, work, p, mu, weights[p], xc, gradient);
    for (size_t k = 0; k < 3; ++k) {
      const double value = gradient[k];
      grad[3 * atom + k] -= value;
      grad[3 * na + 3 * owner + k] += value;
    }
  }
  return true;
}

struct GeometryBlockControl {
  double seed;
  int valid;
  int collective_valid;
};
static_assert(sizeof(GeometryBlockControl) == 16);
struct GeometryBlockTeam {
  int* valid = nullptr;
  __device__ size_t rank() const { return threadIdx.x; }
  __device__ size_t size() const { return blockDim.x; }
  __device__ void sync() const { __syncthreads(); }
  __device__ bool all(bool value) const {
    if (!value) atomicExch(valid, 0);
    sync();
    const bool result = *valid != 0;
    sync();  // all readers finish before a later vote can clear the shared flag
    return result;
  }
};
__global__ void geometry_cooperative_kernel(generativeqc::dft::GridTaskView view, const double* work,
                                const int64_t* ao_atoms, const int64_t* owners,
                                size_t owner_offset, size_t points_per_atom,
                                const double* centers, size_t na, const double* weights,
                                const double* raw, const double* external,
                                size_t external_stride, size_t external_offset,
                                size_t geometry_lanes, double* partial, double* scratch,
                                const generativeqc_grid_adjoint::CenterPair* center_pairs, int* error) {
  // Lanes remain point workers. A whole block cooperates on one worker's panel.
  const size_t lane = blockIdx.x;
  if (lane >= geometry_lanes) return;
  if (view.error && *view.error) {
    if (threadIdx.x == 0) atomicExch(error, 1);
    return;
  }
  extern __shared__ double geometry_pair_storage[];
  auto* states = reinterpret_cast<generativeqc_grid_adjoint::PointPair*>(geometry_pair_storage);
  // Pair state is dead until AO/grid-motion publication completes. Reuse it
  // for one XC value and one ordered AO panel, without increasing admission.
  constexpr size_t point_words = (sizeof(StationaryPointValue) + sizeof(double) - 1) / sizeof(double);
  static_assert(alignof(StationaryPointValue) <= alignof(double));
  const size_t pair_rows = na - 1 < stationary_becke_pair_tile_rows
      ? na - 1 : stationary_becke_pair_tile_rows;
  const size_t pair_capacity = na <= stationary_becke_retained_max_atoms
      ? na * (na - 1) / 2 : pair_rows * (2 * na - pair_rows - 1) / 2;
  const bool cooperative_ao = (point_words + 3 * view.nactive) * sizeof(double)
      <= pair_capacity * sizeof(generativeqc_grid_adjoint::PointPair);
  auto* point_value = reinterpret_cast<StationaryPointValue*>(geometry_pair_storage);
  double* ao_gradient = geometry_pair_storage + point_words;
  __shared__ GeometryBlockControl control;
  double* grad = partial + lane * 9 * na;
  double* ws = scratch + lane * 9 * na;
  auto* distances = reinterpret_cast<std::array<double, 4>*>(ws + 5 * na);
  auto* zeros = reinterpret_cast<size_t*>(ws + 4 * na);
  for (size_t k = threadIdx.x; k < 9 * na; k += blockDim.x) grad[k] = 0;
  __syncthreads();
  for (size_t p = lane; p < view.npoint; p += geometry_lanes) {
    const int64_t owner = owners ? owners[p]
        : (points_per_atom ? int64_t((owner_offset + p) / points_per_atom) : int64_t{-1});
    if (threadIdx.x == 0) {
      // Bit zero is the sticky validity vote; bit one admits ordered AO owners.
      // Clearing either bit cannot undo another thread's invalid-input vote.
      control.collective_valid = 3;
      control.valid = owner >= 0 && owner < int64_t(na) && isfinite(weights[p]) && isfinite(raw[p]);
      if (control.valid) {
        if (cooperative_ao)
          control.valid = geometry_point_setup(view, p, owner, na, raw, external,
              external_stride, external_offset, grad, *point_value, control.seed, error);
        else
          control.valid = geometry_point_ao(view, work, ao_atoms, p, owner, na, weights, raw, external,
                                          external_stride, external_offset, grad, control.seed, error);
      }
    }
    __syncthreads();
    if (!control.valid) {
      if (threadIdx.x == 0) atomicExch(error, 1);
      return;
    }
    if (cooperative_ao) {
      for (size_t ao_index = threadIdx.x; ao_index < view.nactive; ao_index += blockDim.x) {
        const size_t global_ao = view.ao_ids ? view.ao_ids[ao_index] : ao_index;
        if (global_ao >= view.nao) {
          atomicExch(&control.collective_valid, 0);
          continue;
        }
        const int64_t atom = ao_atoms[global_ao];
        if (atom < 0 || atom >= int64_t(na)) {
          atomicExch(&control.collective_valid, 0);
          continue;
        }
        if (ao_index) {
          const size_t previous_global_ao = view.ao_ids ? view.ao_ids[ao_index - 1] : ao_index - 1;
          if (previous_global_ao >= view.nao) {
            atomicExch(&control.collective_valid, 0);
            continue;
          }
          if (ao_atoms[previous_global_ao] > atom)
            atomicAnd(&control.collective_valid, 1);
        }
        geometry_ao_gradient(view, work, p, ao_index, weights[p], *point_value,
                             ao_gradient + 3 * ao_index);
      }
      __syncthreads();
      if (!control.collective_valid) {
        if (threadIdx.x == 0) atomicExch(error, 1);
        return;
      }
      // Monotone owner labels admit one contiguous range per atom, even when
      // AO IDs are permuted within that atom. Other maps keep the full scan.
      // Each atom still has one writer and its original AO summation order.
      const bool ordered_ao_owners = (control.collective_valid & 2) != 0;
      for (size_t atom = threadIdx.x; atom < na; atom += blockDim.x) {
        size_t ao_begin = 0;
        if (ordered_ao_owners) {
          size_t ao_end = view.nactive;
          while (ao_begin < ao_end) {
            const size_t middle = ao_begin + (ao_end - ao_begin) / 2;
            const size_t global_ao = view.ao_ids ? view.ao_ids[middle] : middle;
            if (ao_atoms[global_ao] < int64_t(atom)) ao_begin = middle + 1;
            else ao_end = middle;
          }
        }
        for (size_t ao_index = ao_begin; ao_index < view.nactive; ++ao_index) {
          const size_t global_ao = view.ao_ids ? view.ao_ids[ao_index] : ao_index;
          if (ao_atoms[global_ao] == int64_t(atom))
            for (size_t axis = 0; axis < 3; ++axis)
              grad[3 * atom + axis] -= ao_gradient[3 * ao_index + axis];
          else if (ordered_ao_owners) break;
        }
      }
      if (threadIdx.x == 0)
        for (size_t ao_index = 0; ao_index < view.nactive; ++ao_index)
          for (size_t axis = 0; axis < 3; ++axis)
            grad[3 * na + 3 * owner + axis] += ao_gradient[3 * ao_index + axis];
      // Every reader must finish before Becke overwrites the aliased panel.
      __syncthreads();
    }
    const bool valid = na <= stationary_becke_retained_max_atoms
        ? generativeqc_grid_adjoint::contract_point_cooperative(
            view.points + 3 * p, centers, na, owner, control.seed, grad + 6 * na, ws, ws + na,
            ws + 2 * na, ws + 3 * na, zeros, distances, states, GeometryBlockTeam{},
            local_norm, local_ratio, local_log, local_becke, center_pairs, local_ratio_prepared)
        : generativeqc_grid_adjoint::contract_point_tiled_cooperative(
            view.points + 3 * p, centers, na, owner, control.seed, grad + 6 * na, ws, ws + na,
            ws + 2 * na, ws + 3 * na, zeros, distances, states, stationary_becke_pair_tile_rows,
            GeometryBlockTeam{&control.collective_valid}, local_norm, local_ratio, local_log, local_becke,
            center_pairs, local_ratio_prepared);
    if (!valid) {
      if (threadIdx.x == 0) atomicExch(error, 1);
      return;
    }
  }
  for (size_t k = threadIdx.x; k < 9 * na; k += blockDim.x) finite(grad[k], error, 0);
}
__global__ void geometry_kernel(generativeqc::dft::GridTaskView view, const double* work,
                                const int64_t* ao_atoms, const int64_t* owners,
                                size_t owner_offset, size_t points_per_atom,
                                const double* centers, size_t na, const double* weights,
                                const double* raw, const double* external,
                                size_t external_stride, size_t external_offset,
                                size_t geometry_lanes, double* partial, double* scratch,
                                const generativeqc_grid_adjoint::CenterPair* center_pairs, int* error) {
  const size_t lane = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (lane >= geometry_lanes) return;
  // Same-stream consumers may receive a resident grid tile before a host
  // error publication gate. Propagate the producer's sticky device status into
  // the stationary owner before reading AO/features so one later drain can
  // validate both stages without duplicating scientific arithmetic.
  if (view.error && *view.error) {
    if (lane == 0) atomicExch(error, 1);
    return;
  }
  const size_t np = view.npoint;
  double* grad = partial + lane * 9 * na;
  for (size_t k = 0; k < 9 * na; ++k) grad[k] = 0;
  double* ws = scratch + lane * 9 * na;
  auto* distances = reinterpret_cast<std::array<double, 4>*>(ws + 5 * na);
  auto* zeros = reinterpret_cast<size_t*>(ws + 4 * na);
  for (size_t p = lane; p < np; p += geometry_lanes) {
    const int64_t owner =
        owners ? owners[p]
               : (points_per_atom ? int64_t((owner_offset + p) / points_per_atom) : int64_t{-1});
    if (owner < 0 || owner >= int64_t(na) || !isfinite(weights[p]) || !isfinite(raw[p])) {
      atomicExch(error, 1);
      return;
    }
    double becke_seed = 0;
    if (!geometry_point_ao(view, work, ao_atoms, p, owner, na, weights, raw, external,
                           external_stride, external_offset, grad, becke_seed, error)) return;
    if (!generativeqc_grid_adjoint::contract_point_prepared(view.points + 3 * p, centers, na, owner,
                                             becke_seed, grad + 6 * na, ws, ws + na,
                                             ws + 2 * na, ws + 3 * na, zeros, distances, local_norm,
                                             local_ratio, local_log, local_becke, center_pairs, local_ratio_prepared)) {
      atomicExch(error, 1);
      return;
    }
  }
  for (size_t k = 0; k < 9 * na; ++k) finite(grad[k], error, 0);
}
__global__ void geometry_reduce(const double* partial, size_t na, size_t geometry_lanes, double* output, int* error) {
  if (*error) return;
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 9 * na) return;
  double sum = 0;
  for (size_t lane = 0; lane < geometry_lanes; ++lane) sum += partial[lane * 9 * na + i];
  output[i] = finite(output[i] + sum, error, 0);
}

}  // namespace generativeqc_stationary_cuda
"""


def emit_stationary_reduction_cuda(plan: StationaryGradientPlan) -> str:
    """Specialize the authoritative ordered TensorIR sum for native source storage."""
    lines = [
        "namespace generativeqc_stationary_cuda {",
        "__global__ void source_reduce(const double* input, size_t na, double* output, int* error) {",
        "  if (*error) return;",
    ]
    sources = stationary_runtime_sources(plan)
    if plan.source_names != sources:
        # ECP/range/nonlocal sources still have separately owned providers.
        lines += ["  atomicExch(error, 1);", "}", "}", ""]
        return "\n".join(lines)
    program = plan.reduction_program(atoms=1)
    result = program.outputs["gradient"]
    if (
        result.op != "add"
        or result.spec.shape != (1, 3)
        or result.spec.dtype != "float64"
        or tuple(node.attrs.get("name") for node in result.inputs) != sources
        or any(
            node.op != "input" or node.spec.shape != (1, 3) for node in result.inputs
        )
    ):
        raise ValueError("unsupported stationary native reduction program")
    lines += [
        f"  // stationary-reduction-program: {program.logical_hash}",
        "  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;",
        "  if (i >= 3 * na) return;",
        "  double sum = 0;",
    ]
    for node, coefficient in zip(
        result.inputs, result.attrs["coefficients"], strict=True
    ):
        slot = sources.index(node.attrs["name"])
        lines.append(
            f"  sum += {exact_cuda_literal(coefficient)} * input[{slot} * 3 * na + i];"
        )
    lines += ["  output[i] = finite(sum, error, 0);", "}", "}", ""]
    return "\n".join(lines)


def emit_stationary_scientific_kernels(plan: typing.Any) -> str:
    """Emit bounded task/primitive and XC geometry contractions for the runtime owner."""
    return (
        emit_stationary_weight_cuda(plan)
        + _STATIONARY_SCIENTIFIC_KERNELS
        + emit_stationary_reduction_cuda(plan)
    )


QUALIFIED_SP_COMPONENTS = ("", "x", "y", "z")
QUALIFIED_SPD_COMPONENTS = ("", "x", "xx", "xy", "xz", "y", "yy", "yz", "z", "zz")
QUALIFIED_SPD_AOT_SHARD_WIDTH = 16
QUALIFIED_SPD_AOT_SHARDS = 23
QUALIFIED_FUNCTIONALS = (0, 1, 2)
QUALIFIED_SPINS = ("unpolarized", "polarized")
QUALIFIED_PARTITION_ITERATIONS = 3


@dataclass(frozen=True, slots=True)
class StationaryAotProfile:
    """Build/package profile selected at runtime only by exact plan identity."""

    name: str
    functional: int
    method: str
    spin: str

    @property
    def plan(self) -> StationaryGradientPlan:
        return StationaryGradientPlan(
            resolve_method(self.method, spin=self.spin),
            StationaryMeanField(SCF_POINT_MODEL),
        )


QUALIFIED_STATIONARY_AOT_PROFILES = tuple(
    StationaryAotProfile(name, functional, method, spin)
    for name, functional, method in (
        ("lda", 0, "LDA_XC_PW"),
        ("pbe", 1, "PBE"),
        ("r2scan", 2, "R2SCAN"),
        ("pbe0", 1, "PBE0"),
        ("b3lyp", 3, "B3LYP"),
    )
    for spin in QUALIFIED_SPINS
)
QUALIFIED_STATIONARY_AOT_PROFILE_NAMES = tuple(
    f"{profile.name}_{'rks' if profile.spin == 'unpolarized' else 'uks'}"
    for profile in QUALIFIED_STATIONARY_AOT_PROFILES
)
STATIONARY_AOT_ASSETS = (
    "src/dft/stationary_gradient_cuda.cuh",
    "src/dft/grid_task_view.cuh",
    "src/dft/xc_point.hpp",
    "src/integrals/eri_geometry.hpp",
    "src/integrals/range_moments.hpp",
    "src/tensor/cuda_runtime.cuh",
    "src/runtime/bounded_workspace.hpp",
    "src/runtime/cuda_resources.cuh",
    "src/runtime/resource_cuda.cuh",
    "src/runtime/resource_ledger.hpp",
    "src/tensor/cuda_error.hpp",
    "src/tensor/metrics.hpp",
    "src/runtime/allocation_measurement.hpp",
)


def qualified_sp_requests() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return the molecule-independent stationary s/p derivative inventory."""
    from itertools import product

    domain = QUALIFIED_SP_COMPONENTS
    return tuple(
        [
            (operator, components)
            for operator in ("overlap", "kinetic", "nuclear_attraction")
            for components in product(domain, repeat=2)
        ]
        + [("four_center_eri", components) for components in product(domain, repeat=4)]
        + [("nuclear", ())]
    )


def _qualified_component_aot_domain(
    component_domain: typing.Iterable[str] | None,
) -> tuple[str, ...] | None:
    if component_domain is None:
        return None
    domain = tuple(component_domain)
    if domain != QUALIFIED_SPD_COMPONENTS:
        raise NotImplementedError(
            "packaged stationary component CUDA currently qualifies the full s/p/d domain only"
        )
    return domain


def _profile_stem(profile: StationaryAotProfile) -> str:
    return f"{profile.name}_{'rks' if profile.spin == 'unpolarized' else 'uks'}"


@lru_cache(maxsize=16)
def _qualified_aot_profile(name: str) -> StationaryAotProfile:
    matches = tuple(
        profile
        for profile in QUALIFIED_STATIONARY_AOT_PROFILES
        if _profile_stem(profile) == name
    )
    if len(matches) != 1:
        raise ValueError(f"unknown stationary AOT profile {name!r}")
    return matches[0]


def _qualified_aot_profile_for_plan(
    functional: int, spin: str, plan: StationaryGradientPlan
) -> StationaryAotProfile:
    """Select a package profile by point program and exact plan, never method name."""

    if type(functional) is not int:
        raise TypeError("AOT stationary functional code must be an integer")
    if spin not in QUALIFIED_SPINS:
        raise ValueError("AOT stationary spin must be unpolarized or polarized")
    if not isinstance(plan, StationaryGradientPlan):
        raise TypeError("stationary CUDA AOT requires StationaryGradientPlan")
    matches = tuple(
        profile
        for profile in QUALIFIED_STATIONARY_AOT_PROFILES
        if profile.functional == functional
        and profile.spin == spin
        and profile.plan.identity == plan.identity
    )
    if len(matches) != 1:
        raise ValueError("stationary CUDA AOT plan identity is not packaged")
    return matches[0]


def _legacy_profile(functional: int, spin: str) -> StationaryAotProfile:
    if type(functional) is not int or functional not in QUALIFIED_FUNCTIONALS:
        raise ValueError("AOT stationary functional must be 0, 1, or 2")
    if spin not in QUALIFIED_SPINS:
        raise ValueError("AOT stationary spin must be unpolarized or polarized")
    name = ("lda", "pbe", "r2scan")[functional]
    return _qualified_aot_profile(f"{name}_{'rks' if spin == 'unpolarized' else 'uks'}")


def _qualified_aot_plan(functional: int, spin: str) -> StationaryGradientPlan:
    """Legacy semilocal profile helper retained for existing package tooling."""

    return _legacy_profile(functional, spin).plan


def _stationary_aot_name(
    functional: int,
    spin: str,
    *,
    component_domain: typing.Iterable[str] | None = None,
    plan: StationaryGradientPlan | None = None,
) -> str:
    domain = _qualified_component_aot_domain(component_domain)
    profile = (
        _legacy_profile(functional, spin)
        if plan is None
        else _qualified_aot_profile_for_plan(functional, spin, plan)
    )
    base = _profile_stem(profile)
    return base if domain is None else f"{base}_spd"


def stationary_aot_profile_plan_identity(profile: str) -> str:
    """Return the exact plan identity of one named build/package profile."""

    return _qualified_aot_profile(profile).plan.identity


def stationary_aot_plan_identity(functional: int, *, spin: str) -> str:
    """Return the legacy semilocal generated-plan identity."""

    return _qualified_aot_plan(functional, spin).identity


def emit_stationary_profile_aot_cuda(
    profile: str,
    *,
    primitive_source: str,
    iterations: int = QUALIFIED_PARTITION_ITERATIONS,
) -> str:
    """Emit one qualified profile-bound all-electron stationary CUDA s/p artifact."""

    if type(iterations) is not int or iterations != QUALIFIED_PARTITION_ITERATIONS:
        raise ValueError(
            "AOT stationary CUDA currently qualifies partition_iterations=3 only"
        )
    if not isinstance(primitive_source, str) or not primitive_source:
        raise ValueError("AOT stationary CUDA requires generated primitive source")
    selected = _qualified_aot_profile(profile)
    return emit_stationary_cuda(
        primitive_source,
        functional=selected.functional,
        plan=selected.plan,
        iterations=iterations,
    )


def emit_stationary_profile_component_aot_wrapper_cuda(
    profile: str,
    *,
    iterations: int = QUALIFIED_PARTITION_ITERATIONS,
) -> str:
    """Emit a profile wrapper linked against the shared packaged s/p/d shards."""

    if type(iterations) is not int or iterations != QUALIFIED_PARTITION_ITERATIONS:
        raise ValueError(
            "AOT stationary CUDA currently qualifies partition_iterations=3 only"
        )
    selected = _qualified_aot_profile(profile)
    return emit_stationary_wrapper_cuda(
        functional=selected.functional,
        plan=selected.plan,
        iterations=iterations,
        primitive_shards=QUALIFIED_SPD_AOT_SHARDS,
        primitive_shard_width=QUALIFIED_SPD_AOT_SHARD_WIDTH,
    )


def emit_stationary_aot_cuda(
    functional: int,
    *,
    primitive_source: str,
    spin: str = "unpolarized",
    iterations: int = QUALIFIED_PARTITION_ITERATIONS,
) -> str:
    """Emit one qualified all-electron stationary CUDA s/p artifact."""
    if type(iterations) is not int or iterations != QUALIFIED_PARTITION_ITERATIONS:
        raise ValueError(
            "AOT stationary CUDA currently qualifies partition_iterations=3 only"
        )
    if not isinstance(primitive_source, str) or not primitive_source:
        raise ValueError("AOT stationary CUDA requires generated primitive source")
    profile = _legacy_profile(functional, spin)
    return emit_stationary_profile_aot_cuda(
        _profile_stem(profile),
        primitive_source=primitive_source,
        iterations=iterations,
    )


def emit_stationary_component_aot_wrapper_cuda(
    functional: int,
    *,
    spin: str = "unpolarized",
    iterations: int = QUALIFIED_PARTITION_ITERATIONS,
) -> str:
    """Emit the small wrapper linked against the fixed packaged s/p/d primitive shards."""
    if type(iterations) is not int or iterations != QUALIFIED_PARTITION_ITERATIONS:
        raise ValueError(
            "AOT stationary CUDA currently qualifies partition_iterations=3 only"
        )
    profile = _legacy_profile(functional, spin)
    return emit_stationary_profile_component_aot_wrapper_cuda(
        _profile_stem(profile), iterations=iterations
    )


def stationary_aot_source_identity(
    functional: int,
    *,
    primitive_source: str,
    spin: str = "unpolarized",
    iterations: int = QUALIFIED_PARTITION_ITERATIONS,
) -> str:
    """Content identity shared by checkout builds and installed s/p artifacts."""
    return canonical_hash(
        emit_stationary_aot_cuda(
            functional,
            primitive_source=primitive_source,
            spin=spin,
            iterations=iterations,
        )
    )


def _stationary_aot_contract_identity(
    profile: StationaryAotProfile,
    *,
    iterations: int,
    component_domain: tuple[str, ...] | None,
) -> str:
    if iterations != QUALIFIED_PARTITION_ITERATIONS:
        raise ValueError(
            "AOT stationary CUDA currently qualifies partition_iterations=3 only"
        )
    domain = _qualified_component_aot_domain(component_domain)
    plan = profile.plan
    component = (
        {}
        if domain is None
        else {
            "component_domain": domain,
            "primitive_shard_width": QUALIFIED_SPD_AOT_SHARD_WIDTH,
            "primitive_shards": QUALIFIED_SPD_AOT_SHARDS,
            "primitive_schedule": "first_derivative_schedule.s/p/d",
        }
    )
    weight_sources = tuple(
        source
        for source in (*_FUSED_WEIGHT_SOURCES, "exact_exchange")
        if source in stationary_runtime_sources(plan)
    )
    return canonical_hash(
        {
            "schema": (
                "generativeqc.stationary-cuda-aot.contract.v2"
                if domain is None
                else "generativeqc.stationary-cuda-aot.contract.v3"
            ),
            "functional": profile.functional,
            "spin": profile.spin,
            "plan_identity": plan.identity,
            "weight_programs": {
                source: plan.integral_block(source, terms=1).weights.logical_hash
                for source in weight_sources
            },
            "partition_iterations": iterations,
            "requests": (
                qualified_sp_requests()
                if domain is None
                else {
                    "component_domain": domain,
                    "canonicalization": "first_derivative_schedule",
                }
            ),
            **component,
            "method_module": file_hash(Path(__file__)),
            "resource_module": file_hash(
                Path(__file__).with_name("stationary_resources.py")
            ),
            "compiler_sources": source_hashes(
                "common", "integral", "xc", "dft", assets=STATIONARY_AOT_ASSETS
            ),
        }
    )


@lru_cache(maxsize=24)
def stationary_aot_profile_contract_identity(
    profile: str,
    *,
    iterations: int = QUALIFIED_PARTITION_ITERATIONS,
    component_domain: tuple[str, ...] | None = None,
) -> str:
    """Identity compiler inputs for one exact package profile."""

    return _stationary_aot_contract_identity(
        _qualified_aot_profile(profile),
        iterations=iterations,
        component_domain=component_domain,
    )


@lru_cache(maxsize=12)
def stationary_aot_contract_identity(
    functional: int,
    *,
    spin: str = "unpolarized",
    iterations: int = QUALIFIED_PARTITION_ITERATIONS,
    component_domain: tuple[str, ...] | None = None,
) -> str:
    """Identity every non-numeric compiler input affecting a legacy package."""

    return _stationary_aot_contract_identity(
        _legacy_profile(functional, spin),
        iterations=iterations,
        component_domain=component_domain,
    )


def load_stationary_aot_artifact(
    directory: typing.Any,
    *,
    functional: int,
    spin: str,
    plan: StationaryGradientPlan,
    architecture: str,
    iterations: int = QUALIFIED_PARTITION_ITERATIONS,
    component_domain: tuple[str, ...] | None = None,
) -> CudaArtifact:
    """Load one packaged artifact after checking its plan, domain, and binary identity."""
    domain = _qualified_component_aot_domain(component_domain)
    profile = _qualified_aot_profile_for_plan(functional, spin, plan)
    if type(architecture) is not str or not architecture.startswith("sm_"):
        raise ValueError("stationary AOT architecture must be an sm_XX identity")
    if iterations != QUALIFIED_PARTITION_ITERATIONS:
        raise NotImplementedError(
            "packaged stationary CUDA currently qualifies partition_iterations=3 only"
        )
    name = _stationary_aot_name(functional, spin, component_domain=domain, plan=plan)
    directory = Path(directory).resolve()
    manifest_path = directory / f"generativeqc_stationary_{name}.json"
    candidates = (
        directory / f"libgenerativeqc_stationary_{name}.so",
        directory / f"libgenerativeqc_stationary_{name}.dylib",
        directory / f"generativeqc_stationary_{name}.dll",
    )
    library = next((path for path in candidates if path.is_file()), None)
    if library is None or not manifest_path.is_file():
        raise FileNotFoundError(f"missing packaged stationary CUDA artifact for {name}")
    metadata = json.loads(manifest_path.read_text())
    expected = {
        "schema": (
            "generativeqc.stationary-cuda-aot.v2"
            if domain is None
            else "generativeqc.stationary-cuda-aot.v3"
        ),
        "functional": functional,
        "spin": spin,
        "plan_identity": plan.identity,
        "partition_iterations": iterations,
        "contract_identity": stationary_aot_profile_contract_identity(
            _profile_stem(profile),
            iterations=iterations,
            component_domain=domain,
        ),
    }
    if domain is not None:
        expected.update(
            {
                "component_domain": list(domain),
                "primitive_shard_width": QUALIFIED_SPD_AOT_SHARD_WIDTH,
                "primitive_shards": QUALIFIED_SPD_AOT_SHARDS,
            }
        )
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"stationary CUDA AOT {key} identity mismatch")
    compilation = metadata.get("compile_contract")
    if (
        not isinstance(compilation, dict)
        or compilation.get("fp64") is not True
        or compilation.get("fmad") is not False
    ):
        raise ValueError("stationary CUDA AOT precision contract mismatch")
    architectures = metadata.get("architectures")
    if (
        not isinstance(architectures, list)
        or any(type(value) is not str for value in architectures)
        or architecture not in architectures
    ):
        raise NotImplementedError(
            f"stationary CUDA AOT artifact does not package {architecture}"
        )
    code_objects = metadata.get("code_objects")
    if not isinstance(code_objects, list) or any(
        not isinstance(item, dict)
        or set(item) != {"architecture", "kind"}
        or item["kind"] not in ("cubin", "ptx")
        for item in code_objects
    ):
        raise ValueError("stationary CUDA AOT code-object metadata is invalid")
    code_kinds = sorted(
        {item["kind"] for item in code_objects if item["architecture"] == architecture}
    )
    if not code_kinds:
        raise ValueError("stationary CUDA AOT target has no code object")
    digest = file_hash(library)
    if (
        metadata.get("binary_sha256") != digest
        or metadata.get("binary_bytes") != library.stat().st_size
    ):
        raise ValueError("stationary CUDA AOT binary integrity mismatch")
    identity = {
        "schema": metadata["schema"],
        "source": metadata["source_identity"],
        "contract": metadata["contract_identity"],
        "functional": functional,
        "spin": spin,
        "plan": plan.identity,
        "partition_iterations": iterations,
        **({"component_domain": list(domain)} if domain is not None else {}),
        "target": {"architecture": architecture, "code_kinds": code_kinds},
    }
    return CudaArtifact(
        library,
        {
            **metadata,
            "identity": identity,
            "key": canonical_hash(
                {"identity": identity, "binary_sha256": metadata["binary_sha256"]}
            ),
            "artifact_kind": "packaged-aot",
            "driver_ptx_jit_possible": "ptx" in code_kinds,
            "driver_ptx_jit_required": "cubin" not in code_kinds,
        },
    )


_FIRST_DERIVATIVE_DECLARATION = """#include <cuda_runtime.h>
extern __device__ bool first_derivative(
    unsigned kind, const double* e, const double* c, double* out);
"""


_DERIVATIVE_AXIS_BITS = 3
_DERIVATIVE_CENTER_BITS = 8
_DERIVATIVE_NUCLEUS_BIT = 11
_DERIVATIVE_RANK4_BIT = 12
_DERIVATIVE_KIND_SHIFT = 13
_DERIVATIVE_AXES = tuple(permutations(range(3)))


def encode_stationary_derivative_kind(
    kind: int,
    binding: typing.Any,
    *,
    rank: int,
    has_nucleus: bool,
) -> int:
    """Pack canonical derivative binding metadata into the existing task kind."""

    if type(kind) is not int or kind < 0:
        raise ValueError("stationary derivative kind must be a nonnegative integer")
    if rank not in (2, 4) or type(has_nucleus) is not bool:
        raise ValueError("stationary derivative binding requires rank two or four")
    owners = rank + int(has_nucleus)
    centers = tuple(binding.centers)
    axes = tuple(binding.axes)
    if (
        len(centers) != owners
        or sorted(centers) != list(range(owners))
        or axes not in _DERIVATIVE_AXES
    ):
        raise ValueError(
            "stationary derivative binding is not a center/axis permutation"
        )
    center_code = sum(center << (2 * slot) for slot, center in enumerate(centers))
    return (
        (kind << _DERIVATIVE_KIND_SHIFT)
        | ((rank == 4) << _DERIVATIVE_RANK4_BIT)
        | (has_nucleus << _DERIVATIVE_NUCLEUS_BIT)
        | (center_code << _DERIVATIVE_AXIS_BITS)
        | _DERIVATIVE_AXES.index(axes)
    )


def _sharded_first_derivative_adapter(shards: int, shard_width: int) -> str:
    """Dispatch bounded derivative objects while restoring public center/axis order."""

    if (
        type(shards) is not int
        or shards < 1
        or type(shard_width) is not int
        or shard_width < 1
    ):
        raise ValueError("stationary CUDA derivative shard dimensions must be positive")
    declarations = [
        "#include <cuda_runtime.h>",
        *(
            f"extern __device__ bool first_derivative_shard_{unit}("
            "unsigned kind, const double* e, const double* c, double* out);"
            for unit in range(shards)
        ),
    ]
    dispatch = [
        "  bool ok = false;",
        f"  switch (kind / {shard_width}u) {{",
        *(
            f"    case {unit}: ok = first_derivative_shard_{unit}("
            f"kind % {shard_width}u, exponents, centers, canonical); break;"
            for unit in range(shards)
        ),
        "    default: return false;",
        "  }",
        "  if (!ok) return false;",
    ]
    return "\n".join(
        [
            *declarations,
            (
                "__device__ bool first_derivative(unsigned encoded, const double* e, "
                "const double* c, double* out) {"
            ),
            f"  const unsigned axis_index = encoded & {(1 << _DERIVATIVE_AXIS_BITS) - 1}u;",
            (
                f"  const unsigned center_code = (encoded >> {_DERIVATIVE_AXIS_BITS}) & "
                f"{(1 << _DERIVATIVE_CENTER_BITS) - 1}u;"
            ),
            f"  const bool has_nucleus = ((encoded >> {_DERIVATIVE_NUCLEUS_BIT}) & 1u) != 0;",
            f"  const unsigned rank = ((encoded >> {_DERIVATIVE_RANK4_BIT}) & 1u) ? 4u : 2u;",
            f"  const unsigned kind = encoded >> {_DERIVATIVE_KIND_SHIFT};",
            "  unsigned axes[3]{};",
            "  switch (axis_index) {",
            "    case 0: axes[0]=0; axes[1]=1; axes[2]=2; break;",
            "    case 1: axes[0]=0; axes[1]=2; axes[2]=1; break;",
            "    case 2: axes[0]=1; axes[1]=0; axes[2]=2; break;",
            "    case 3: axes[0]=1; axes[1]=2; axes[2]=0; break;",
            "    case 4: axes[0]=2; axes[1]=0; axes[2]=1; break;",
            "    case 5: axes[0]=2; axes[1]=1; axes[2]=0; break;",
            "    default: return false;",
            "  }",
            "  const unsigned owners = rank + unsigned(has_nucleus);",
            "  double exponents[4]{1.0,1.0,1.0,1.0};",
            "  double centers[12]{};",
            "  double canonical[12]{};",
            "  for (unsigned center = 0; center < owners; ++center) {",
            "    const unsigned original = (center_code >> (2 * center)) & 3u;",
            "    if (original >= owners) return false;",
            "    if (center < rank) {",
            "      if (original >= rank) return false;",
            "      exponents[center] = e[original];",
            "    }",
            "    for (unsigned axis = 0; axis < 3; ++axis)",
            "      centers[3 * center + axis] = c[3 * original + axes[axis]];",
            "  }",
            *dispatch,
            "  for (unsigned j = 0; j < 12; ++j) out[j] = 0.0;",
            "  for (unsigned center = 0; center < owners; ++center) {",
            "    const unsigned original = (center_code >> (2 * center)) & 3u;",
            "    for (unsigned axis = 0; axis < 3; ++axis)",
            "      out[3 * original + axes[axis]] = canonical[3 * center + axis];",
            "  }",
            "  return true;",
            "}",
            "",
        ]
    )


def emit_stationary_wrapper_cuda(
    *,
    functional: typing.Any = None,
    pbe: typing.Any = None,
    plan: typing.Any,
    iterations: typing.Any = 3,
    declare_primitive: bool = True,
    primitive_shards: int | None = None,
    primitive_shard_width: int | None = None,
) -> typing.Any:
    """Emit the small method-specific TU linked against cached primitive code."""

    if not isinstance(plan, StationaryGradientPlan):
        raise TypeError("stationary CUDA requires StationaryGradientPlan")
    if (primitive_shards is None) != (primitive_shard_width is None):
        raise ValueError("sharded stationary primitive metadata must be complete")
    if primitive_shards is not None and not declare_primitive:
        raise ValueError("sharded stationary primitive adapter owns its declaration")
    if primitive_shards is not None:
        if primitive_shard_width is None:
            raise ValueError("sharded stationary primitive metadata must be complete")
        primitive_declaration = _sharded_first_derivative_adapter(
            primitive_shards, primitive_shard_width
        )
    else:
        primitive_declaration = (
            _FIRST_DERIVATIVE_DECLARATION if declare_primitive else ""
        )
    semilocal = next(
        (
            primitive.functional
            for primitive in plan.method.primitives
            if type(primitive) is SemilocalXCPrimitive
        ),
        None,
    )
    return (
        primitive_declaration
        + emit_geometry_cuda(
            functional=functional,
            pbe=pbe,
            iterations=iterations,
            semilocal=semilocal,
        )
        + "namespace generativeqc_stationary_cuda {\n"
        + f"constexpr unsigned stationary_spin_blocks = {plan.spin_blocks};\n"
        + "constexpr bool stationary_native_reduction_supported = "
        + str(plan.source_names == stationary_runtime_sources(plan)).lower()
        + ";\n"
        + "}\n"
        + _runtime_layout_cuda(plan)
        + '#include "dft/stationary_gradient_cuda.cuh"\n'
        + emit_stationary_scientific_kernels(plan)
    )


def emit_stationary_cuda(
    primitive_source: typing.Any,
    *,
    functional: typing.Any = None,
    pbe: typing.Any = None,
    plan: typing.Any,
    iterations: typing.Any = 3,
) -> typing.Any:
    """Compose the legacy single-TU source for inspection and provenance tests.

    Runtime compilation uses separable CUDA objects so the large primitive
    lowering is cached independently of functional and spin specialization.
    """

    return primitive_source + emit_stationary_wrapper_cuda(
        functional=functional,
        pbe=pbe,
        plan=plan,
        iterations=iterations,
        declare_primitive=False,
    )


def compile_stationary_cuda(
    primitive_source: typing.Any,
    *,
    functional: typing.Any = None,
    pbe: typing.Any = None,
    plan: typing.Any,
    iterations: typing.Any,
    compiler: typing.Any,
    cache: typing.Any,
    primitive_shard_width: int | None = None,
) -> typing.Any:
    """Compile strict-FP64 primitive and wrapper objects, then device-link them."""

    if not isinstance(compiler, CudaCompilerAdapter):
        raise TypeError("stationary CUDA requires an explicit CUDA compiler adapter")
    if os.environ.get("NVCC_PREPEND_FLAGS") or os.environ.get("NVCC_APPEND_FLAGS"):
        raise ValueError("stationary strict CUDA rejects NVCC flag overrides")

    sharded = not isinstance(primitive_source, str)
    primitive_sources = (
        tuple(primitive_source) if sharded else (typing.cast("str", primitive_source),)
    )
    if not primitive_sources or any(
        not isinstance(source, str) or not source for source in primitive_sources
    ):
        raise ValueError("stationary CUDA requires nonempty primitive source")
    if sharded != (primitive_shard_width is not None):
        raise ValueError("stationary CUDA shard width must match primitive sources")
    wrapper_source = emit_stationary_wrapper_cuda(
        functional=functional,
        pbe=pbe,
        plan=plan,
        iterations=iterations,
        primitive_shards=len(primitive_sources) if sharded else None,
        primitive_shard_width=primitive_shard_width,
    )
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    wrapper_path = cache / (canonical_hash(wrapper_source) + ".stationary.cu")
    cache_source(wrapper_path, wrapper_source)

    header = asset_path("src/dft/stationary_gradient_cuda.cuh")
    include = f"-I{header.parents[1]}"
    primitive_headers = tuple(
        asset_path(name)
        for name in (
            "src/integrals/eri_geometry.hpp",
            "src/integrals/range_moments.hpp",
        )
    )
    wrapper_headers = tuple(
        asset_path(name)
        for name in (
            "src/dft/stationary_gradient_cuda.cuh",
            "src/dft/grid_task_view.cuh",
            "src/dft/xc_point.hpp",
            "src/tensor/cuda_runtime.cuh",
            "src/runtime/bounded_workspace.hpp",
            "src/runtime/cuda_resources.cuh",
            "src/runtime/resource_cuda.cuh",
            "src/runtime/resource_ledger.hpp",
            "src/tensor/cuda_error.hpp",
            "src/tensor/metrics.hpp",
            "src/runtime/allocation_measurement.hpp",
        )
    )
    primitives = []
    for source in primitive_sources:
        primitive_path = cache / (canonical_hash(source) + ".primitive.cu")
        cache_source(primitive_path, source)
        primitives.append(
            compile_cuda_object(
                compiler,
                cache,
                primitive_path,
                headers=primitive_headers,
                options=(
                    "--fmad=false",
                    "--expt-relaxed-constexpr",
                    include,
                    *_split_compile_options(),
                ),
            )
        )
    wrapper = compile_cuda_object(
        compiler,
        cache,
        wrapper_path,
        headers=wrapper_headers,
        options=("--fmad=false", "--expt-relaxed-constexpr", include),
    )
    return link_cuda_objects(
        compiler,
        cache,
        (*primitives, wrapper),
        libraries=("cublas",),
    )
