"""Complete native primitive binding for the existing weighted Hermite/AD DAG.

This module emits scalar callables only. Bounded record uploads, public-weight
pullbacks, contraction/scatter and compilation caches remain in their existing
consumer/runtime owners. Explicit radial inputs are frozen in every callable.
"""

from __future__ import annotations

from vibeqc_compiler.common.provenance import canonical_hash

from .ir import IntegralIR
from .ir_serialization import integral_to_payload
from .range_separation import CoulombKernelFamily
from .weighted_eri import WeightedEriKernel
from .weighted_eri_cuda import emit_weighted_eri_header


def emit_weighted_eri_primitive_header(
    functions: tuple[tuple[WeightedEriKernel, str], ...], *, backend="cuda"
) -> str:
    """Emit CPU/CUDA values and all twelve derivatives for packed weight subsets.

    Each ``name_primitive`` consumes four positive exponents, twelve Bohr
    coordinates, and weights in the kernel's component_indices order. Coefficients
    and Cartesian/public normalization are already in the weights. A failed
    geometry or nonfinite result leaves the caller's output unchanged.

    The operator family and exact normalized omega are embedded in the source;
    they cannot be changed by replaying a compiled callable with new geometry.
    No screening, omega derivative, Hessian or range-separated DF is implied.
    """
    source = emit_weighted_eri_header(functions, backend=backend, packed_weights=True)
    qualifier = "__device__ __forceinline__" if backend == "cuda" else "inline"
    lines = [
        "#ifndef VIBEQC_GENERATED_WEIGHTED_ERI_PRIMITIVE_HPP",
        "#define VIBEQC_GENERATED_WEIGHTED_ERI_PRIMITIVE_HPP",
        source,
        '#include "integrals/eri_geometry.hpp"',
        "namespace vibeqc::scf::generated_weighted_eri {",
    ]
    native_ranges = {
        CoulombKernelFamily.FULL_RANGE: "Full",
        CoulombKernelFamily.LONG_RANGE: "Long",
        CoulombKernelFamily.SHORT_RANGE: "Short",
    }
    for kernel, name in functions:
        radial = kernel.integral.operator.coulomb_kernel
        lines.extend(
            [
                f"{qualifier} bool {name}_primitive(const double* exponents, const double* centers,",
                "    const double* component_weights, Gradient& output) {",
                "  if (!component_weights) return false;",
                f"  for (unsigned i = 0; i < {len(kernel.component_indices)}; ++i)",
                "    if (!std::isfinite(component_weights[i])) return false;",
                "  Geometry geometry{};",
                "  if (!vibeqc::integrals::make_eri_geometry(exponents, centers,",
                f"      {kernel.integral.maximum_coulomb_order}, vibeqc::integrals::CoulombRange::{native_ranges[radial.family]},",
                f"      {radial.omega.hex()}, geometry)) return false;",
                f"  const auto candidate = {name}(geometry, component_weights);",
                "  if (!std::isfinite(candidate.value)) return false;",
                "  for (unsigned center = 0; center < 4; ++center)",
                "    for (unsigned axis = 0; axis < 3; ++axis)",
                "      if (!std::isfinite(candidate.center[center][axis])) return false;",
                "  output = candidate;",
                "  return true;",
                "}",
            ]
        )
    lines.append("}  // namespace vibeqc::scf::generated_weighted_eri")
    lines.append("#endif")
    return "\n".join(lines) + "\n"


def weighted_eri_program_identity(kernel: WeightedEriKernel, backend: str) -> str:
    """Identify a compiled packed subset independently of mutable geometry."""
    return weighted_eri_metadata_identity(
        kernel.integral, kernel.component_indices, backend
    )


def weighted_eri_metadata_identity(
    integral: IntegralIR, component_indices, backend: str
) -> str:
    """Revalidate exported program metadata without rebuilding the arithmetic DAG."""
    if backend not in ("cpu", "cuda"):
        raise ValueError("weighted native execution supports cpu or cuda")
    return canonical_hash(
        {
            "schema": "vibeqc.weighted-native.v2",
            "backend": backend,
            "integral": integral_to_payload(integral),
            "component_indices": component_indices,
        }
    )


def emit_weighted_eri_runtime(kernel: WeightedEriKernel, *, backend="cuda") -> str:
    """Bind tagged weighted records to the shared bounded native runtime.

    Record weights already contain normalization and all consumer factors.
    This experimental v2 program therefore requires canonical unit factors
    and a range operator, and validates its frozen omega/ABI/subset per record.
    Compilation/preparation do not authorize a performance-profile promotion.
    """
    identity = weighted_eri_program_identity(kernel, backend)
    if not kernel.integral.operator.range_separated:
        raise ValueError("the v2 generated runtime requires an explicit range operator")
    consumer = kernel.integral.contractions[0]
    if (consumer.output_sign, consumer.weights.sign, consumer.weights.prefactor) != (
        1,
        1,
        1,
    ):
        raise ValueError(
            "native program requires unit factors; the stream already applies them"
        )
    if kernel.integral.requested_derivative_centers != (0, 1, 2, 3):
        raise ValueError(
            "native program returns all four independent shell-center slots"
        )
    radial = kernel.integral.operator.coulomb_kernel
    tag = 1 if radial.family == CoulombKernelFamily.LONG_RANGE else 2
    codes = {}
    for packed, index in enumerate(kernel.component_indices):
        powers = tuple(
            component.count(axis)
            for component in kernel.spec.components[index]
            for axis in "xyz"
        )
        codes[sum(power << (2 * i) for i, power in enumerate(powers))] = packed
    source = emit_weighted_eri_primitive_header(
        ((kernel, "weighted"),), backend=backend
    )
    source += r"""
#include <cstdio>
#include <memory>
#include "scf/weighted_eri_runtime.hpp"
namespace {
using namespace vibeqc::scf;
using namespace vibeqc::scf::weighted_runtime;
#ifdef __CUDACC__
#define WEIGHTED_HD __host__ __device__
#define WEIGHTED_EVAL __device__
#else
#define WEIGHTED_HD
#define WEIGHTED_EVAL
#endif
struct Program {
  using Record = CudaWeightedEriRangePrimitive;
  WEIGHTED_HD static const CudaWeightedEriPrimitive& base(const Record& r) { return r.primitive; }
  WEIGHTED_HD static unsigned code(const CudaWeightedEriPrimitive& r) {
    unsigned result = 0;
    for (unsigned c = 0; c < 4; ++c)
      for (unsigned a = 0; a < 3; ++a) result |= r.angular[c][a] << (2 * (3*c + a));
    return result;
  }
  WEIGHTED_HD static int index(unsigned code) {
    switch (code) {
@CASES@
      default: return -1;
    }
  }
  static bool validate(const Record& r) {
    const auto& b = base(r);
    if (r.abi_version != 2 || r.omega != @OMEGA@ || (b.kind & ~1U) != @TAG@U) return false;
    if (b.kind & 1U) {
      if (!@PSSS@ || code(b) != 1) return false;
      for (unsigned a = 0; a < 3; ++a)
        if (index(1U << (2*a)) < 0 && b.weights[a] != 0) return false;
      return true;
    }
    return b.weights[1] == 0 && b.weights[2] == 0 && index(code(b)) >= 0;
  }
  WEIGHTED_EVAL static bool evaluate(const Record& r, Result& output) {
    const auto& b = base(r);
    double weights[@COUNT@]{};
    if (b.kind & 1U) {
      for (unsigned a = 0; a < 3; ++a) {
        const int packed = index(1U << (2*a));
        if (packed >= 0) weights[packed] = b.weights[a];
      }
    } else {
      const int packed = index(code(b));
      if (packed < 0) return false;
      weights[packed] = b.weights[0];
    }
    generated_weighted_eri::Gradient candidate{};
    if (!generated_weighted_eri::weighted_primitive(b.exponents, &b.centers[0][0], weights, candidate))
      return false;
    output.value = candidate.value;
    for (unsigned c = 0; c < 4; ++c)
      for (unsigned a = 0; a < 3; ++a) output.center[c][a] = candidate.center[c][a];
    return true;
  }
};
using NativePlan = Plan<Program>;
#undef WEIGHTED_HD
#undef WEIGHTED_EVAL

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

extern "C" const char* vibeqc_weighted_identity_v2() { return "@IDENTITY@"; }
extern "C" int vibeqc_weighted_create_v2(int device, int major, int minor,
    std::size_t capacity, std::size_t tiles, std::size_t budget, void** output,
    char* detail, std::size_t size) {
  if (output) *output = nullptr;
  return boundary([&] {
    if (!output) throw std::invalid_argument("null weighted ERI plan output");
    auto candidate = std::make_unique<NativePlan>(device, major, minor, capacity, tiles, budget);
    *output = candidate.release();
  }, detail, size);
}
extern "C" void vibeqc_weighted_destroy_v2(void* handle) { delete static_cast<NativePlan*>(handle); }
extern "C" int vibeqc_weighted_run_v2(void* handle, const Program::Record* records,
    std::size_t count, std::size_t tiles, Result* output, int profile, char* detail, std::size_t size) {
  return boundary([&] {
    if (!handle || (profile != 0 && profile != 1)) throw std::invalid_argument("invalid weighted ERI handle/profile");
    static_cast<NativePlan*>(handle)->run(records, count, tiles, output, profile != 0);
  }, detail, size);
}
extern "C" int vibeqc_weighted_storage_v2(void* handle, std::uint64_t* output,
    char* detail, std::size_t size) {
  return boundary([&] {
    if (!handle || !output) throw std::invalid_argument("null weighted ERI storage argument");
    const auto& plan = *static_cast<NativePlan*>(handle);
    output[0] = plan.host_bytes(); output[1] = plan.device_bytes();
  }, detail, size);
}
#ifdef __CUDACC__
extern "C" int vibeqc_weighted_metrics_v2(void* handle, vibeqc_tensor::Metrics* output,
    char* detail, std::size_t size) {
  return boundary([&] {
    if (!handle || !output) throw std::invalid_argument("null weighted ERI metrics argument");
    *output = static_cast<NativePlan*>(handle)->metrics();
  }, detail, size);
}
#endif
"""
    for marker, replacement in {
        "@CASES@": "\n".join(
            f"      case {code}U: return {packed};" for code, packed in codes.items()
        ),
        "@OMEGA@": radial.omega.hex(),
        "@TAG@": str(tag << 8),
        "@PSSS@": "true" if kernel.spec.angular == (1, 0, 0, 0) else "false",
        "@COUNT@": str(len(kernel.component_indices)),
        "@IDENTITY@": identity,
    }.items():
        source = source.replace(marker, replacement)
    return source
