"""Native scalar emission and shared bounded runtime for partial Hessian tiles.

CPU/CUDA consume the same generated arithmetic and fixed packed weights. The
native owner is the established weighted-integral Plan with a coordinate-tile
result policy; this module adds no allocator, stream or compilation cache.
"""

from vibeqc_compiler.common.provenance import canonical_hash

from .cuda import CudaEmitter
from .expr import AlgebraForm, AlgebraFusion, AlgebraOrdering, RematerializationPolicy
from .ir_serialization import integral_to_payload
from .second_derivatives import SecondDerivativeKernel, require_second_consumer

SECOND_RECORD_TAG = 0x32445648


def second_program_identity(integral, component_indices, output_indices, backend):
    """Bind immutable operator, derivative, subset and coordinate-output semantics."""
    if backend not in ("cpu", "cuda"):
        raise ValueError("second derivative native backend must be cpu or cuda")
    return canonical_hash(
        {
            "schema": "vibeqc.second-native.v1",
            "backend": backend,
            "integral": integral_to_payload(integral),
            "component_indices": tuple(component_indices),
            "output_indices": tuple(output_indices),
        }
    )


def emit_second_derivative_primitive(kernel: SecondDerivativeKernel, *, backend="cuda"):
    """Emit one to twelve output coordinates using the common scalar C emitter.

    The callable takes primitive exponents, mathematical-center xyz positions,
    weights in selected-component order and a direction in requested-center
    order. Raw kernels require unit component weights. Coefficients and basis
    normalization can be applied as a fixed record scale by the runtime.
    All outputs are transactional; nonfinite geometry/arithmetic returns false.
    """
    if backend not in ("cpu", "cuda"):
        raise ValueError("second derivative native backend must be cpu or cuda")
    if not 1 <= len(kernel.outputs) <= 12:
        raise ValueError(
            "native second derivatives require one to twelve coordinate outputs per tile"
        )
    consumer = require_second_consumer(kernel.integral)
    is_eri = len(kernel.integral.signature.shells) == 4
    count = len(kernel.integral.operator.centers)
    raw = consumer.weights is None
    hvp = consumer.output == "weighted_hvp"
    qualifier = "__device__ __forceinline__" if backend == "cuda" else "inline"
    lines = [
        "#include <cmath>",
        '#include "integrals/eri_geometry.hpp"',
        "namespace vibeqc::scf::generated_second {",
    ]
    if is_eri:
        # Fifteen moments belong to this new type. The first-gradient geometry
        # and its fourteen-double Boys buffer retain their exact declaration.
        lines += [
            "struct Geometry {",
            "  double inverse_two_p, inverse_two_q, rho;",
            "  double difference[3], shifts[4][3], product_scales[4], decay[4][3];",
            "  double prefactor, boys[15];",
            "};",
        ]
    lines += [
        f"{qualifier} bool primitive(const double* exponents, const double* centers,",
        "    const double* weights, const double* direction, double* output) {",
        "  if (!exponents || !centers || !weights || !direction || !output) return false;",
        f"  for (unsigned i = 0; i < {len(kernel.integral.signature.shells)}; ++i)",
        "    if (!std::isfinite(exponents[i]) || exponents[i] <= 0) return false;",
        f"  for (unsigned i = 0; i < {count * 3}; ++i)",
        "    if (!std::isfinite(centers[i])) return false;",
        f"  for (unsigned i = 0; i < {len(kernel.component_indices)}; ++i)",
        "    if (!std::isfinite(weights[i])"
        + (" || weights[i] != 1" if raw else "")
        + ") return false;",
    ]
    if hvp:
        lines += [
            f"  for (unsigned i = 0; i < {len(kernel.recovery.centers) * 3}; ++i)",
            "    if (!std::isfinite(direction[i])) return false;",
        ]
    variables = {
        f"component_weight_{i}": "1.0" if raw else f"weights[{packed}]"
        for packed, i in enumerate(kernel.component_indices)
    }
    variables.update(
        {
            f"direction_{i}_{axis}": f"direction[{3 * i + a}]"
            for i in range(len(kernel.recovery.centers))
            for a, axis in enumerate("xyz")
        }
    )
    if is_eri:
        lines += [
            "  Geometry geometry{};",
            "  if (!vibeqc::integrals::make_eri_geometry<Geometry, 14>(exponents, centers,",
            f"      {kernel.boys_count - 1}, vibeqc::integrals::CoulombRange::Full, 0, geometry)) return false;",
        ]
        variables.update(
            {
                name: f"geometry.{name}"
                for name in ("inverse_two_p", "inverse_two_q", "rho", "prefactor")
            }
        )
        for a, axis in enumerate("xyz"):
            variables[f"difference_{axis}"] = f"geometry.difference[{a}]"
            for i, prefix in enumerate(("pa", "pb", "qc", "qd")):
                variables[f"{prefix}_{axis}"] = f"geometry.shifts[{i}][{a}]"
            for i, prefix in enumerate(("first", "second", "third", "fourth")):
                variables[f"decay_{prefix}_{axis}"] = f"geometry.decay[{i}][{a}]"
                variables[f"{prefix}_product_scale"] = f"geometry.product_scales[{i}]"
        variables.update(
            {f"boys_{i}": f"geometry.boys[{i}]" for i in range(kernel.boys_count)}
        )
    else:
        lines += ["  if (!std::isfinite(exponents[0] + exponents[1])) return false;"]
        variables.update(alpha="exponents[0]", beta="exponents[1]")
        variables.update(
            {
                f"{'abc'[i]}_{axis}": f"centers[{3 * i + a}]"
                for i in range(count)
                for a, axis in enumerate("xyz")
            }
        )
        if kernel.boys_argument is not None:
            argument = CudaEmitter(kernel.graph, variables)
            argument.emit((kernel.boys_argument,))
            lines += [
                f"  double boys[{kernel.boys_count}]{{}};",
                "  {",
                *argument.lines,
                f"    if (!vibeqc::integrals::range_moments({kernel.boys_count - 1},",
                f"        {argument.reference(kernel.boys_argument)}, 1, vibeqc::integrals::CoulombRange::Full, 0, boys)) return false;",
                "  }",
            ]
            variables.update(
                {f"boys_{i}": f"boys[{i}]" for i in range(kernel.boys_count)}
            )
    graph, roots = kernel.graph.apply_algebra_form(
        kernel.outputs, AlgebraForm.FACTORED_NARY
    )
    plan = graph.materialization_plan(
        roots,
        RematerializationPolicy(name="second_coordinate_tile", inline_single_use=True),
        AlgebraOrdering.TOPOLOGICAL,
        AlgebraFusion.SEPARATE,
    )
    emitter = CudaEmitter(graph, variables, plan)
    emitter.emit(roots)
    lines += [
        *emitter.lines,
        "  const double candidate[] = {"
        + ", ".join(emitter.reference(root) for root in roots)
        + "};",
        "  for (double value : candidate) if (!std::isfinite(value)) return false;",
        f"  for (unsigned i = 0; i < {len(roots)}; ++i) output[i] = candidate[i];",
        "  return true;",
        "}",
        "}  // namespace vibeqc::scf::generated_second",
    ]
    return "\n".join(lines) + "\n"


def emit_second_derivative_runtime(kernel, *, backend="cuda"):
    """Bind a tagged packed primitive record to shared CPU/CUDA Plan storage.

    Records retain the existing primitive prefix for common finite/center/tile
    checks. Its first weight is a fixed normalization scale; other prefix
    weights and angular fields are zero. The suffix owns packed component
    weights and twelve direction doubles, with all unused directions zero.
    The new entry points verify record stride and tag independently of content.
    """
    source = emit_second_derivative_primitive(kernel, backend=backend)
    identity = second_program_identity(
        kernel.integral, kernel.component_indices, kernel.output_indices, backend
    )
    source += r"""
#include <cstdio>
#include <memory>
#include "scf/weighted_eri_runtime.hpp"
#include "vibeqc/vibeqc.h"
namespace {
using namespace vibeqc::scf;
using namespace vibeqc::scf::weighted_runtime;
using Output = CoordinateOutput<@OUTPUTS@>;
using SecondResult = Output::Result;
struct SecondRecord {
  CudaWeightedEriPrimitive primitive;
  double weights[@WEIGHTS@];
  double direction[12];
};
#ifdef __CUDACC__
#define SECOND_HD __host__ __device__
#define SECOND_EVAL __device__
#else
#define SECOND_HD
#define SECOND_EVAL
#endif
struct Program {
  using Record = SecondRecord;
  SECOND_HD static const CudaWeightedEriPrimitive& base(const Record& r) { return r.primitive; }
  static bool validate(const Record& r) {
    const auto& b = base(r);
    if (b.kind != @TAG@U || b.weights[1] != 0 || b.weights[2] != 0) return false;
    for (const auto& powers : b.angular) for (unsigned p : powers) if (p) return false;
    for (unsigned i = @SHELLS@; i < 4; ++i) if (b.exponents[i] != 1) return false;
    for (unsigned i = @CENTERS@; i < 4; ++i)
      for (double coordinate : b.centers[i]) if (coordinate != 0) return false;
    for (double weight : r.weights)
      if (!std::isfinite(weight) || (@RAW@ && weight != 1)) return false;
    for (unsigned i = 0; i < 12; ++i)
      if (!std::isfinite(r.direction[i]) || (i >= @DIRECTIONS@ && r.direction[i] != 0)) return false;
    return true;
  }
  SECOND_EVAL static bool evaluate(const Record& r, SecondResult& result) {
    if (!generated_second::primitive(r.primitive.exponents, &r.primitive.centers[0][0],
                                      r.weights, r.direction, result.values)) return false;
    for (double& value : result.values) {
      value *= r.primitive.weights[0];
      if (!std::isfinite(value)) return false;
    }
    return true;
  }
};
using NativePlan = Plan<Program, Output>;
#undef SECOND_HD
#undef SECOND_EVAL

template <class F> int boundary(F operation, char* detail, std::size_t size) {
  if (detail && size) detail[0] = '\0';
  try { operation(); return VIBEQC_STATUS_SUCCESS; }
  catch (const NumericalFailure& error) {
    if (detail && size) std::snprintf(detail, size, "%s", error.what());
    return VIBEQC_STATUS_NUMERICAL_FAILURE;
  }
#ifdef __CUDACC__
  catch (const vibeqc_tensor::DeviceAllocationError& error) {
    if (detail && size) std::snprintf(detail, size, "%s", error.what());
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
#endif
  catch (const std::bad_alloc& error) {
    if (detail && size) std::snprintf(detail, size, "%s", error.what());
    return VIBEQC_STATUS_OUT_OF_MEMORY;
  }
  catch (const std::invalid_argument& error) {
    if (detail && size) std::snprintf(detail, size, "%s", error.what());
    return VIBEQC_STATUS_INVALID_ARGUMENT;
  }
  catch (const std::exception& error) {
    if (detail && size) std::snprintf(detail, size, "%s", error.what());
#ifdef __CUDACC__
    return VIBEQC_STATUS_CUDA_ERROR;
#else
    return VIBEQC_STATUS_INTERNAL_ERROR;
#endif
  }
}
}  // namespace

extern "C" const char* vibeqc_second_identity_v1() { return "@IDENTITY@"; }
extern "C" std::size_t vibeqc_second_stride_v1() { return sizeof(SecondRecord); }
extern "C" int vibeqc_second_create_v1(int device, int major, int minor,
    std::size_t capacity, std::size_t tiles, std::size_t budget, void** output,
    char* detail, std::size_t size) {
  if (output) *output = nullptr;
  return boundary([&] {
    if (!output) throw std::invalid_argument("null second derivative plan output");
    auto candidate = std::make_unique<NativePlan>(device, major, minor, capacity, tiles, budget);
    *output = candidate.release();
  }, detail, size);
}
extern "C" int vibeqc_second_run_v1(void* handle, const void* records,
    std::size_t count, std::size_t stride, std::size_t tiles, void* output,
    int profile, char* detail, std::size_t size) {
  return boundary([&] {
    if (!handle || stride != sizeof(SecondRecord) || (profile != 0 && profile != 1))
      throw std::invalid_argument("second derivative handle, record stride or profile flag");
    static_cast<NativePlan*>(handle)->run(static_cast<const SecondRecord*>(records), count,
                                         tiles, static_cast<SecondResult*>(output), profile);
  }, detail, size);
}
extern "C" int vibeqc_second_storage_v1(void* handle, std::uint64_t* amounts,
    char* detail, std::size_t size) {
  return boundary([&] {
    if (!handle || !amounts) throw std::invalid_argument("null second derivative storage query");
    auto* plan = static_cast<NativePlan*>(handle);
    amounts[0] = plan->host_bytes(); amounts[1] = plan->device_bytes();
  }, detail, size);
}
#ifdef __CUDACC__
extern "C" int vibeqc_second_metrics_v1(void* handle, vibeqc_tensor::Metrics* result,
    char* detail, std::size_t size) {
  return boundary([&] {
    if (!handle || !result) throw std::invalid_argument("null second derivative metrics");
    *result = static_cast<NativePlan*>(handle)->metrics();
  }, detail, size);
}
#endif
extern "C" void vibeqc_second_destroy_v1(void* handle) { delete static_cast<NativePlan*>(handle); }
"""
    for token, value in {
        "@OUTPUTS@": len(kernel.outputs),
        "@WEIGHTS@": len(kernel.component_indices),
        "@SHELLS@": len(kernel.integral.signature.shells),
        "@CENTERS@": len(kernel.integral.operator.centers),
        "@TAG@": SECOND_RECORD_TAG,
        "@RAW@": int(kernel.integral.contractions[0].weights is None),
        "@DIRECTIONS@": len(kernel.recovery.centers) * 3
        if kernel.integral.contractions[0].output == "weighted_hvp"
        else 0,
        "@IDENTITY@": identity,
    }.items():
        source = source.replace(token, str(value))
    return source
