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
from functools import lru_cache
from itertools import permutations
from pathlib import Path

from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_runtime import CudaArtifact
from vibeqc_compiler.common.native_runtime import (
    compile_cuda_object,
    link_cuda_objects,
)
from vibeqc_compiler.common.paths import asset_path, source_hashes
from vibeqc_compiler.common.provenance import canonical_hash, file_hash
from vibeqc_compiler.common.source_cache import cache_source
from vibeqc_compiler.tensor.cuda_inline import (
    InlineCudaOutput,
    exact_cuda_literal,
    lower_inline_cuda_output,
)
from vibeqc_compiler.xc.geometry_cuda import emit_geometry_cuda

from .spec import resolve_method
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
        "namespace vibeqc_stationary_cuda {",
        f"// stationary-plan: {plan.identity}",
        f"constexpr unsigned stationary_nuclear_source = {STATIONARY_RUNTIME_SOURCE_NAMES.index('nuclear')};",
    ]
    dispatch: list[str] = []
    for source in _FUSED_WEIGHT_SOURCES:
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
QUALIFIED_FUNCTIONALS = (0, 1, 2)
QUALIFIED_SPINS = ("unpolarized", "polarized")
QUALIFIED_PARTITION_ITERATIONS = 3
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


def _qualified_aot_plan(functional: int, spin: str) -> StationaryGradientPlan:
    if type(functional) is not int or functional not in QUALIFIED_FUNCTIONALS:
        raise ValueError("AOT stationary functional must be 0, 1, or 2")
    if spin not in QUALIFIED_SPINS:
        raise ValueError("AOT stationary spin must be unpolarized or polarized")
    method_name = ("LDA_XC_PW", "PBE", "R2SCAN")[functional]
    return StationaryGradientPlan(
        resolve_method(method_name, spin=spin),
        StationaryMeanField(SCF_POINT_MODEL),
    )


def _stationary_aot_name(functional: int, spin: str) -> str:
    _qualified_aot_plan(functional, spin)
    return f"{('lda', 'pbe', 'r2scan')[functional]}_{'rks' if spin == 'unpolarized' else 'uks'}"


def stationary_aot_plan_identity(functional: int, *, spin: str) -> str:
    """Return the exact generated-plan identity encoded by one AOT artifact."""
    return _qualified_aot_plan(functional, spin).identity


def emit_stationary_aot_cuda(
    functional: int,
    *,
    primitive_source: str,
    spin: str = "unpolarized",
    iterations: int = QUALIFIED_PARTITION_ITERATIONS,
) -> str:
    """Emit one qualified all-electron stationary CUDA artifact."""
    if type(iterations) is not int or iterations != QUALIFIED_PARTITION_ITERATIONS:
        raise ValueError(
            "AOT stationary CUDA currently qualifies partition_iterations=3 only"
        )
    if not isinstance(primitive_source, str) or not primitive_source:
        raise ValueError("AOT stationary CUDA requires generated primitive source")
    plan = _qualified_aot_plan(functional, spin)
    return emit_stationary_cuda(
        primitive_source,
        functional=functional,
        plan=plan,
        iterations=iterations,
    )


def stationary_aot_source_identity(
    functional: int,
    *,
    primitive_source: str,
    spin: str = "unpolarized",
    iterations: int = QUALIFIED_PARTITION_ITERATIONS,
) -> str:
    """Content identity shared by checkout builds and installed artifacts."""
    return canonical_hash(
        emit_stationary_aot_cuda(
            functional,
            primitive_source=primitive_source,
            spin=spin,
            iterations=iterations,
        )
    )


@lru_cache(maxsize=6)
def stationary_aot_contract_identity(
    functional: int,
    *,
    spin: str = "unpolarized",
    iterations: int = QUALIFIED_PARTITION_ITERATIONS,
) -> str:
    """Identity every non-numeric compiler input affecting a packaged artifact."""
    if iterations != QUALIFIED_PARTITION_ITERATIONS:
        raise ValueError(
            "AOT stationary CUDA currently qualifies partition_iterations=3 only"
        )
    plan = _qualified_aot_plan(functional, spin)
    return canonical_hash(
        {
            "schema": "vibeqc.stationary-cuda-aot.contract.v2",
            "functional": functional,
            "spin": spin,
            "plan_identity": plan.identity,
            "weight_programs": {
                source: plan.integral_block(source, terms=1).weights.logical_hash
                for source in _FUSED_WEIGHT_SOURCES
            },
            "partition_iterations": iterations,
            "requests": qualified_sp_requests(),
            "method_module": file_hash(Path(__file__)),
            "compiler_sources": source_hashes(
                "common", "integral", "xc", "dft", assets=STATIONARY_AOT_ASSETS
            ),
        }
    )


def load_stationary_aot_artifact(
    directory: typing.Any,
    *,
    functional: int,
    spin: str,
    plan: StationaryGradientPlan,
    architecture: str,
    iterations: int = QUALIFIED_PARTITION_ITERATIONS,
) -> CudaArtifact:
    """Load one packaged artifact after checking its plan and binary identity."""
    expected_plan = _qualified_aot_plan(functional, spin)
    if (
        not isinstance(plan, StationaryGradientPlan)
        or plan.identity != expected_plan.identity
    ):
        raise ValueError("stationary CUDA AOT plan identity mismatch")
    if type(architecture) is not str or not architecture.startswith("sm_"):
        raise ValueError("stationary AOT architecture must be an sm_XX identity")
    if iterations != QUALIFIED_PARTITION_ITERATIONS:
        raise NotImplementedError(
            "packaged stationary CUDA currently qualifies partition_iterations=3 only"
        )
    name = _stationary_aot_name(functional, spin)
    directory = Path(directory).resolve()
    manifest_path = directory / f"vibeqc_stationary_{name}.json"
    candidates = (
        directory / f"libvibeqc_stationary_{name}.so",
        directory / f"libvibeqc_stationary_{name}.dylib",
        directory / f"vibeqc_stationary_{name}.dll",
    )
    library = next((path for path in candidates if path.is_file()), None)
    if library is None or not manifest_path.is_file():
        raise FileNotFoundError(f"missing packaged stationary CUDA artifact for {name}")
    metadata = json.loads(manifest_path.read_text())
    expected = {
        "schema": "vibeqc.stationary-cuda-aot.v2",
        "functional": functional,
        "spin": spin,
        "plan_identity": plan.identity,
        "partition_iterations": iterations,
        "contract_identity": stationary_aot_contract_identity(
            functional, spin=spin, iterations=iterations
        ),
    }
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
        "schema": "vibeqc.stationary-cuda-aot.v2",
        "source": metadata["source_identity"],
        "contract": metadata["contract_identity"],
        "functional": functional,
        "spin": spin,
        "plan": plan.identity,
        "partition_iterations": iterations,
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
    return (
        primitive_declaration
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
