"""CPU adjoint contraction of native atomic measures through the Becke graphs.

One forward and one reverse pair pass per point replace coordinate-wise JVP
traversals. Local scalar partials are differentiated from Graph roots and the
normalized-product reverse traversal is emitted by the compiler for CPU/CUDA.

Rationale: .agents/notes/implemented/architecture/2026-09-20-becke-adjoint-compiler-owner.md
"""

import ctypes as ct
import typing
from pathlib import Path

import numpy as np

from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.native_runtime import compile_runtime
from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.common.source_cache import cache_source
from vibeqc_compiler.dft.grid import checked_int
from vibeqc_compiler.integral.scalar_c import ScalarCEmitter

from .grid_response import grid_response_program

_GRID_ADJOINT_SOURCE = r"""#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

#if defined(__CUDACC__)
#define VIBEQC_GRID_HD __host__ __device__
#else
#define VIBEQC_GRID_HD
#endif

namespace vibeqc_grid_adjoint {
VIBEQC_GRID_HD inline double portable_abs(double value) { return value < 0.0 ? -value : value; }

VIBEQC_GRID_HD inline double portable_exp(double value) {
#if defined(__CUDA_ARCH__)
  return ::exp(value);
#else
  return std::exp(value);
#endif
}

// Shared two-pass Becke reverse composition. Runtime owners supply O(natom)
// scratch per worker and transactional reduction storage. Local mathematical
// partials come exclusively from the compiler's grid_response Graphs.
template <class Norm>
VIBEQC_GRID_HD std::array<double, 4> distance(const double* a, const double* b, Norm norm,
                                              bool& valid) {
  double delta[3], scale = 0;
  for (size_t k = 0; k < 3; ++k) {
    delta[k] = a[k] - b[k];
    scale = std::max(scale, portable_abs(delta[k]));
  }
  if (!(scale > 0) || !std::isfinite(scale)) {
    valid = false;
    return {};
  }
  auto result = norm(delta[0] / scale, delta[1] / scale, delta[2] / scale);
  result[0] *= scale;
  for (double v : result)
    if (!std::isfinite(v)) {
      valid = false;
      return {};
    }
  return result;
}

template <class Norm, class Ratio, class Log, class Pair>
VIBEQC_GRID_HD bool contract_point(const double* point, const double* centers, size_t na,
                                   size_t owner, double seed, double* gradient, double* logs,
                                   double* products, double* bar_product, double* bar_distance,
                                   size_t* zeros, std::array<double, 4>* distances, Norm norm,
                                   Ratio ratio, Log logarithm, Pair pair) {
  bool valid = true;
  for (size_t a = 0; a < na; ++a) distances[a] = distance(point, centers + 3 * a, norm, valid);
  if (!valid) return false;
  if (na == 1) return true;
  for (size_t a = 0; a < na; ++a) logs[a] = 0;
  for (size_t a = 0; a < na; ++a) zeros[a] = 0;
  for (size_t a = 0; a < na; ++a) bar_distance[a] = 0;
  auto factor = [&](size_t a, size_t b, double separation) {
    auto r = ratio(distances[a][0] - distances[b][0], separation);
    const bool clipped = portable_abs(r[0]) >= 1;
    auto f = pair(std::clamp(r[0], -1.0, 1.0));
    if (clipped || f[0] < 0 || f[0] > 1) f[1] = 0;
    f[0] = std::clamp(f[0], 0.0, 1.0);
    return f;
  };
  for (size_t a = 0; a < na; ++a)
    for (size_t b = 0; b < a; ++b) {
      const double separation = distance(centers + 3 * a, centers + 3 * b, norm, valid)[0];
      const auto f = factor(a, b, separation);
      for (size_t side = 0; side < 2; ++side) {
        const size_t atom = side ? b : a;
        const double v = side ? 1 - f[0] : f[0];
        if (v > 0)
          logs[atom] += logarithm(v)[0];
        else
          ++zeros[atom];
      }
    }
  double maximum = -std::numeric_limits<double>::infinity();
  for (size_t a = 0; a < na; ++a)
    if (!zeros[a]) maximum = std::max(maximum, logs[a]);
  if (!std::isfinite(maximum)) return false;
  double total = 0;
  for (size_t a = 0; a < na; ++a) {
    products[a] = zeros[a] ? 0 : portable_exp(logs[a] - maximum);
    total += products[a];
  }
  // The selected normalized-product objective uses the SAME ratio
  // graph. A frozen log scale cancels between numerator/denominator.
  auto objective = ratio(products[owner], total);
  for (size_t a = 0; a < na; ++a)
    bar_product[a] = seed * (objective[2] + (a == size_t(owner) ? objective[1] : 0));
  for (size_t a = 0; a < na; ++a)
    for (size_t b = 0; b < a; ++b) {
      const auto separation = distance(centers + 3 * a, centers + 3 * b, norm, valid);
      const auto r = ratio(distances[a][0] - distances[b][0], separation[0]);
      const auto f = factor(a, b, separation[0]);
      // Saturated branches have exactly zero pullback. Skip before
      // exponentiation to avoid the undefined numerical form inf*0.
      if (f[1] == 0) continue;
      double bar_mu = 0;
      for (size_t side = 0; side < 2; ++side) {
        const size_t atom = side ? b : a;
        const double v = side ? 1 - f[0] : f[0];
        double derivative = 0;
        if (!zeros[atom]) derivative = products[atom] * logarithm(v)[1];
        // ONE exact zero leaves the product of all other factors;
        // two zeros kill the first derivative. Never divide by zero.
        else if (zeros[atom] == 1 && v == 0)
          derivative = portable_exp(logs[atom] - maximum);
        bar_mu += (side ? -1 : 1) * bar_product[atom] * derivative * f[1];
      }
      bar_distance[a] += bar_mu * r[1];
      bar_distance[b] -= bar_mu * r[1];
      for (size_t k = 0; k < 3; ++k) {
        const double value = bar_mu * r[2] * separation[k + 1];
        gradient[3 * a + k] += value;
        gradient[3 * b + k] -= value;
      }
    }
  for (size_t a = 0; a < na; ++a)
    for (size_t k = 0; k < 3; ++k) {
      const double value = bar_distance[a] * distances[a][k + 1];
      gradient[3 * a + k] -= value;
      gradient[3 * owner + k] += value;
    }
  return valid;
}
}  // namespace vibeqc_grid_adjoint
#undef VIBEQC_GRID_HD
"""


def emit_grid_adjoint() -> str:
    """Emit the shared bounded Becke normalized-product reverse traversal."""
    return _GRID_ADJOINT_SOURCE


def emit_grid_partials(
    iterations: typing.Any = 3, *, device: typing.Any = False
) -> typing.Any:
    """Shared local AD construction for CPU and CUDA traversal owners."""
    lines = []
    identities = {}
    for kind, names in (
        ("norm", ("x", "y", "z")),
        ("ratio", ("a", "b")),
        ("log", ("p",)),
        ("becke", ("mu",)),
    ):
        program = grid_response_program(kind, iterations)
        graph, primal = program.graph, program.roots[0]
        roots = (
            primal,
            *(graph.differentiate(primal, graph.variable(name)) for name in names),
        )
        emitter = ScalarCEmitter(graph, {name: name for name in names})
        emitter.emit(roots)
        lines += [
            f"{'__device__' if device else 'static'} std::array<double, {len(roots)}> local_{kind}({', '.join('double ' + name for name in names)}) {{",
            *emitter.lines,
            "return {" + ", ".join(emitter.reference(root) for root in roots) + "};",
            "}",
        ]
        identities[kind] = program.identity
    identity = canonical_hash({"schema": "grid-cpu-adjoint-v1", "programs": identities})
    lines.append(f"// Shared grid graphs: {identity}")
    return "\n".join(lines) + "\n"


def emit_grid_contraction(iterations: typing.Any = 3) -> typing.Any:
    """Generate the bounded Becke adjoint from compiler-owned composition."""
    lines = [
        emit_grid_adjoint(),
        '#include "grid_response_cpu.hpp"',
        emit_grid_partials(iterations),
    ]
    lines += [
        'extern "C" int grid_contract(const double* points, size_t np, const double* centers, size_t na, const int64_t* owners, const double* seeds, double* output, size_t no, size_t budget, size_t max_pairs, double tolerance) noexcept {',
        "return vibeqc_grid_cpu::contract(points, np, centers, na, owners, seeds, output, no, budget, max_pairs, tolerance, local_norm, local_ratio, local_log, local_becke);",
        "}",
    ]
    return "\n".join(lines) + "\n"


class NativeGridContraction:
    """Strict FP64 CPU ABI; seeds are energy-density times raw atomic measure.

    Points move with their integer owner. The result contracts into ALL nuclear
    coordinates, with no point/atom/coordinate Jacobian and no scalar interpreter.
    max_bytes covers native scratch plus conservative adapter staging; it is not
    a simultaneous endpoint/SCF resource reservation.
    """

    def __init__(
        self,
        *,
        compiler: typing.Any,
        cache: typing.Any,
        iterations: typing.Any = 3,
        max_bytes: typing.Any = 8 * 1024 * 1024,
        max_pair_visits: typing.Any = 100_000_000,
    ) -> None:
        if not isinstance(compiler, CppCompilerAdapter):
            raise TypeError("native grid requires a CPU compiler adapter")
        checked_int(max_bytes, "grid byte budget", high=(1 << 63) - 1)
        checked_int(max_pair_visits, "grid work budget", low=0, high=(1 << 63) - 1)
        self.max_pair_visits = max_pair_visits
        self.max_bytes = max_bytes
        source = emit_grid_contraction(iterations)
        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / (canonical_hash(source) + ".cpp")
        cache_source(path, source)
        header = asset_path("src/dft/grid_response_cpu.hpp")
        self.artifact = compile_runtime(
            compiler,
            cache,
            path,
            headers=(header,),
            options=("-ffp-contract=off", f"-I{header.parent}"),
        )
        self.identity = self.artifact.metadata["key"]
        self.library = ct.CDLL(str(self.artifact.library))
        self.call = self.library.grid_contract
        self.call.argtypes = [
            ct.POINTER(ct.c_double),
            ct.c_size_t,
            ct.POINTER(ct.c_double),
            ct.c_size_t,
            ct.POINTER(ct.c_int64),
            ct.POINTER(ct.c_double),
            ct.POINTER(ct.c_double),
            ct.c_size_t,
            ct.c_size_t,
            ct.c_size_t,
            ct.c_double,
        ]
        self.call.restype = ct.c_int

    def contract(
        self,
        points: typing.Any,
        centers: typing.Any,
        owners: typing.Any,
        seeds: typing.Any,
        *,
        coincident_tolerance: typing.Any = 1e-12,
    ) -> typing.Any:
        """Return a detached gradient, publishing nothing on a late native error."""
        points, centers, owners, seeds = map(
            np.asarray, (points, centers, owners, seeds)
        )
        if (
            points.ndim != 2
            or points.shape[1:] != (3,)
            or centers.ndim != 2
            or centers.shape[1:] != (3,)
            or not len(centers)
        ):
            raise ValueError("grid points/centers require (n,3) and nonempty centers")
        npnt, natom = len(points), len(centers)
        if (1 + 2 * npnt) * natom * (natom - 1) // 2 > self.max_pair_visits:
            raise ValueError("grid work budget exceeded")
        if 8 * (10 * npnt + 30 * natom) > self.max_bytes:
            raise ValueError("grid byte budget exceeded")
        if (
            owners.shape != (npnt,)
            or owners.dtype != np.int64
            or seeds.shape != (npnt,)
        ):
            raise ValueError("grid owners require int64 and seeds require point shape")
        if any(
            v.dtype != np.float64 or not np.isfinite(v).all()
            for v in (points, centers, seeds)
        ):
            raise ValueError("grid inputs require finite float64")
        points, centers, owners, seeds = map(
            np.ascontiguousarray, (points, centers, owners, seeds)
        )
        output = np.empty((natom, 3))
        ptr = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_double))
        code = self.call(
            ptr(points),
            npnt,
            ptr(centers),
            natom,
            owners.ctypes.data_as(ct.POINTER(ct.c_int64)),
            ptr(seeds),
            ptr(output),
            output.size,
            self.max_bytes,
            self.max_pair_visits,
            coincident_tolerance,
        )
        if code:
            raise ValueError(
                f"native grid contraction failed ({code}); invalid or nonsmooth inputs"
            )
        return immutable(output)
