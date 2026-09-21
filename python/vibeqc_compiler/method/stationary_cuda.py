"""CUDA lowering composition for the bounded stationary RKS diagnostic.

Primitive recurrences, Becke local AD and AO bilinear AD remain their existing
compiler programs; their bounded primitive/geometry contractions are emitted here.
Native code owns allocation, validation, transfers, launches and ABI only.
Generation is host-only and does not import the public runtime or probe CUDA.

Rationale: .agents/notes/implemented/architecture/2026-09-20-stationary-cuda-emitted-contractions.md
"""

import os
import typing
from fractions import Fraction
from pathlib import Path

from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.native_runtime import compile_runtime
from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.common.source_cache import cache_source
from vibeqc_compiler.xc.geometry_cuda import emit_geometry_cuda

from .stationary_gradient import StationaryGradientPlan

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
_SPLIT_COMPILE_THREADS_ENV = "VIBEQC_STATIONARY_CUDA_SPLIT_COMPILE_THREADS"


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


def _fraction(value: typing.Any) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if type(value) is int:
        return Fraction(value, 1)
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(type(v) is int for v in value)
    ):
        return Fraction(*value)
    raise TypeError("stationary CUDA weight lowering requires exact rational constants")


def _literal(value: typing.Any) -> str:
    value = _fraction(value)
    if value.denominator == 1:
        return f"{value.numerator}.0"
    return f"({value.numerator}.0/{value.denominator}.0)"


def _weight_expression(
    plan: StationaryGradientPlan, source: str
) -> tuple[str, str, int]:
    """Lower generated one-term TensorIR weight to a scalar CUDA expression."""
    block = plan.integral_block(source, terms=1)
    program = block.weights
    values: dict[typing.Any, str | tuple[str, ...]] = {}
    arity = 0
    bindings = {
        "density_left": ("density", 0, 1),
        "density_right": ("density", 2, 3),
        "weighted_density": ("weighted_density", 0, 1),
    }
    for node in program.live_nodes:
        if node.op == "input":
            name = node.attrs["name"]
            if name not in bindings or node.spec.shape != (plan.spin_blocks, 1):
                raise ValueError("unsupported stationary CUDA weight input contract")
            pointer, left, right = bindings[name]
            arity = max(arity, left + 1, right + 1)
            values[node] = tuple(
                f"{pointer}[{spin} * n * n + size_t(ao[{left}]) * n + size_t(ao[{right}])]"
                for spin in range(plan.spin_blocks)
            )
        elif node.op == "constant":
            raw = node.attrs["values"]
            if node.spec.shape or len(raw) != 1:
                raise ValueError(
                    "stationary CUDA weight lowering requires scalar constants"
                )
            values[node] = _literal(raw[0])
        elif node.op == "reduce":
            operand = values[node.inputs[0]]
            if node.attrs["axes"] != (0,) or not isinstance(operand, tuple):
                raise ValueError("unsupported stationary CUDA weight reduction")
            values[node] = "(" + " + ".join(operand) + ")"
        elif node.op == "einsum":
            operands = [values[item] for item in node.inputs]
            if node.spec.shape != (1,) or any(
                not isinstance(item, str) for item in operands
            ):
                raise ValueError("unsupported stationary CUDA weight einsum")
            coefficient = _fraction(node.attrs["coefficient"])
            factors = [typing.cast("str", item) for item in operands]
            if coefficient != 1:
                factors.insert(0, _literal(coefficient))
            values[node] = "(" + " * ".join(factors) + ")"
        else:
            raise ValueError(f"unsupported stationary CUDA weight op: {node.op}")
    output = values[program.outputs["weights"]]
    if not isinstance(output, str):
        raise TypeError("stationary CUDA weight output did not lower to a scalar")
    return output, program.logical_hash, arity


def emit_stationary_weight_cuda(plan: typing.Any) -> str:
    """Emit pointwise device weights directly from StationaryGradientPlan TensorIR."""
    if not isinstance(plan, StationaryGradientPlan):
        raise TypeError(
            "stationary CUDA weight lowering requires StationaryGradientPlan"
        )
    functions = [
        "namespace vibeqc_stationary_cuda {",
        f"// stationary-plan: {plan.identity}",
        f"constexpr unsigned stationary_nuclear_source = {STATIONARY_RUNTIME_SOURCE_NAMES.index('nuclear')};",
    ]
    dispatch: list[str] = []
    for source in _FUSED_WEIGHT_SOURCES:
        expression, identity, arity = _weight_expression(plan, source)
        symbol = f"stationary_weight_{source}"
        functions.extend(
            (
                f"// stationary-weight-program-{source}: {identity}",
                f"__device__ inline double {symbol}(const double* density, const double* weighted_density, size_t n, const int64_t* ao) {{",
                f"  return {expression};",
                "}",
            )
        )
        slot = STATIONARY_RUNTIME_SOURCE_NAMES.index(source)
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
            "}  // namespace vibeqc_stationary_cuda",
            "",
        )
    )
    return "\n".join(functions)


_STATIONARY_SCIENTIFIC_KERNELS = r"""namespace vibeqc_stationary_cuda {
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
    if ((source != 0 && source != 1 && source != 5) || (rank != 2 && rank != 4) ||
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
  if (slot >= 21 * na) return;
  const size_t source = slot / (3 * na);
  if (source != 0 && source != 1 && source != 5) return;
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
__global__ void nuclear_kernel(unsigned kind, int64_t a, int64_t b, double za, double zb,
                               const double* centers, size_t na, double* output, int* error) {
  if (a < 0 || b < 0 || a >= int64_t(na) || b >= int64_t(na) || a == b ||
      !isfinite(za) || !isfinite(zb) || !(za > 0) || !(zb > 0)) {
    atomicExch(error, 1);
    return;
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
    return;
  }
  for (size_t k = 0; k < 3; ++k) {
    output[18 * na + 3 * a + k] =
        finite(output[18 * na + 3 * a + k] + v[k], error, 0);
    output[18 * na + 3 * b + k] =
        finite(output[18 * na + 3 * b + k] + v[3 + k], error, 0);
  }
}
__global__ void validate_centers(const double* centers, size_t na, double tolerance, int* error) {
  bool valid = true;
  for (size_t a = 0; a < na; ++a) {
    for (size_t k = 0; k < 3; ++k)
      if (!isfinite(centers[3 * a + k])) valid = false;
    for (size_t b = 0; b < a; ++b)
      if (vibeqc_grid_adjoint::distance(centers + 3 * a, centers + 3 * b, local_norm, valid)[0] <=
          tolerance)
        valid = false;
  }
  if (!valid) atomicExch(error, 1);
}
__global__ void geometry_kernel(vibeqc::dft::GridTaskView view, const double* work,
                                const int64_t* ao_atoms, const int64_t* owners,
                                const double* centers, size_t na, const double* weights,
                                const double* raw, double* partial, double* scratch, int* error) {
  const size_t lane = threadIdx.x;
  const size_t np = view.npoint, n = view.nactive, stride = np * n;
  double* grad = partial + lane * 9 * na;
  for (size_t k = 0; k < 9 * na; ++k) grad[k] = 0;
  double* ws = scratch + lane * 9 * na;
  auto* distances = reinterpret_cast<std::array<double, 4>*>(ws + 5 * na);
  auto* zeros = reinterpret_cast<size_t*>(ws + 4 * na);
  for (size_t p = lane; p < np; p += workers) {
    if (owners[p] < 0 || owners[p] >= int64_t(na) || !isfinite(weights[p]) || !isfinite(raw[p])) {
      atomicExch(error, 1);
      return;
    }
    double rho[2]{view.features[p], view.features[5 * np + p]}, g[2][3]{}, tau[2]{};
    if (stationary_functional != 0)
      for (size_t s = 0; s < 2; ++s)
        for (size_t k = 0; k < 3; ++k) g[s][k] = view.features[(5 * s + k + 1) * np + p];
    if (stationary_functional == 2)
      for (size_t s = 0; s < 2; ++s) tau[s] = view.features[(5 * s + 4) * np + p];
    // The exact shared SCF point model, including vacuum/spin boundaries.
    const auto xc = stationary_evaluate_point(rho, g, tau);
    if (!xc.valid) {
      atomicExch(error, 1);
      return;
    }
    for (size_t mu = 0; mu < n; ++mu) {
      if (view.ao_ids[mu] >= view.nao) {
        atomicExch(error, 1);
        return;
      }
      const auto atom = ao_atoms[view.ao_ids[mu]];
      if (atom < 0 || atom >= int64_t(na)) {
        atomicExch(error, 1);
        return;
      }
      double pullback[4]{};
      for (size_t s = 0; s < 2; ++s) {
        double c[5]{weights[p] * xc.rho[s]}, w[4]{};
        for (size_t j = 0; j < stationary_jets; ++j) {
          w[j] = work[(4 * s + j) * stride + p * n + mu];
          if (j) c[j] = weights[p] * xc.gradient[s][j - 1];
        }
        if (stationary_functional == 2) c[4] = weights[p] * xc.kinetic[s];
        double local[4]{};
        ao_pullback(c, w, local);
        for (size_t j = 0; j < stationary_jets; ++j) pullback[j] += local[j];
      }
      for (size_t k = 0; k < 3; ++k) {
        double value = 0;
        for (size_t j = 0; j < stationary_jets; ++j)
          value += pullback[j] * view.ao[stationary_shift[j][k] * stride + p * n + mu];
        grad[3 * atom + k] -= value;
        grad[3 * na + 3 * owners[p] + k] += value;
      }
    }
    if (!vibeqc_grid_adjoint::contract_point(view.points + 3 * p, centers, na, owners[p],
                                             xc.energy * raw[p], grad + 6 * na, ws, ws + na,
                                             ws + 2 * na, ws + 3 * na, zeros, distances, local_norm,
                                             local_ratio, local_log, local_becke)) {
      atomicExch(error, 1);
      return;
    }
  }
  for (size_t k = 0; k < 9 * na; ++k) finite(grad[k], error, 0);
}
__global__ void geometry_reduce(const double* partial, size_t na, double* output, int* error) {
  if (*error) return;
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 9 * na) return;
  double sum = 0;
  for (size_t lane = 0; lane < workers; ++lane) sum += partial[lane * 9 * na + i];
  output[i] = finite(output[i] + sum, error, 0);
}

}  // namespace vibeqc_stationary_cuda
"""


def emit_stationary_reduction_cuda(plan: StationaryGradientPlan) -> str:
    """Specialize the authoritative ordered TensorIR sum for native source storage."""
    lines = [
        "namespace vibeqc_stationary_cuda {",
        "__global__ void source_reduce(const double* input, size_t na, double* output, int* error) {",
        "  if (*error) return;",
    ]
    if plan.source_names != STATIONARY_RUNTIME_SOURCE_NAMES:
        # ECP/hybrid/nonlocal sources are not all owned by this seven-source arena.
        lines += ["  atomicExch(error, 1);", "}", "}", ""]
        return "\n".join(lines)
    program = plan.reduction_program(atoms=1)
    result = program.outputs["gradient"]
    if (
        result.op != "add"
        or result.spec.shape != (1, 3)
        or result.spec.dtype != "float64"
        or tuple(node.attrs.get("name") for node in result.inputs)
        != STATIONARY_RUNTIME_SOURCE_NAMES
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
        slot = STATIONARY_RUNTIME_SOURCE_NAMES.index(node.attrs["name"])
        lines.append(f"  sum += {_literal(coefficient)} * input[{slot} * 3 * na + i];")
    lines += ["  output[i] = finite(sum, error, 0);", "}", "}", ""]
    return "\n".join(lines)


def emit_stationary_scientific_kernels(plan: typing.Any) -> str:
    """Emit bounded task/primitive and XC geometry contractions for the runtime owner."""
    return (
        emit_stationary_weight_cuda(plan)
        + _STATIONARY_SCIENTIFIC_KERNELS
        + emit_stationary_reduction_cuda(plan)
    )


def emit_stationary_cuda(
    primitive_source: typing.Any,
    *,
    functional: typing.Any = None,
    pbe: typing.Any = None,
    plan: typing.Any,
    iterations: typing.Any = 3,
) -> typing.Any:
    """Compose explicit primitive lowering and shared XC geometric lowering.

    ``pbe`` remains a compatibility spelling for historical LDA/PBE callers.
    New method-owned lowering passes 0=LDA, 1=PBE, or 2=r2SCAN explicitly.
    """
    if not isinstance(plan, StationaryGradientPlan):
        raise TypeError("stationary CUDA requires StationaryGradientPlan")
    return (
        primitive_source
        + emit_geometry_cuda(functional=functional, pbe=pbe, iterations=iterations)
        + "namespace vibeqc_stationary_cuda {\n"
        + f"constexpr unsigned stationary_spin_blocks = {plan.spin_blocks};\n"
        + "constexpr bool stationary_native_reduction_supported = "
        + str(plan.source_names == STATIONARY_RUNTIME_SOURCE_NAMES).lower()
        + ";\n"
        + "}\n"
        + '#include "dft/stationary_gradient_cuda.cuh"\n'
        + emit_stationary_scientific_kernels(plan)
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
) -> typing.Any:
    """Compile a finite strict-FP64 artifact with transitive header identities."""
    if not isinstance(compiler, CudaCompilerAdapter):
        raise TypeError("stationary CUDA requires an explicit CUDA compiler adapter")
    if os.environ.get("NVCC_PREPEND_FLAGS") or os.environ.get("NVCC_APPEND_FLAGS"):
        raise ValueError("stationary strict CUDA rejects NVCC flag overrides")
    source = emit_stationary_cuda(
        primitive_source,
        functional=functional,
        pbe=pbe,
        plan=plan,
        iterations=iterations,
    )
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / (canonical_hash(source) + ".cu")
    cache_source(path, source)
    header = asset_path("src/dft/stationary_gradient_cuda.cuh")
    return compile_runtime(
        compiler,
        cache,
        path,
        headers=tuple(
            asset_path(name)
            for name in (
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
        ),
        libraries=("cublas",),
        options=(
            "--fmad=false",
            "--expt-relaxed-constexpr",
            f"-I{header.parents[1]}",
            *_split_compile_options(),
        ),
    )
