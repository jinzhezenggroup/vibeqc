"""CPU adjoint contraction of native atomic measures through the Becke graphs.

One forward and one reverse pair pass per point replace coordinate-wise JVP
traversals. Only local scalar partials are differentiated; normalized products
compose their adjoints in the bounded native scheduling template.
"""

import ctypes as ct
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


def emit_grid_contraction(iterations=3):
    """Generate primal and local VJP entries from existing Graph roots only."""
    lines = ['#include "grid_response_cpu.hpp"']
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
            f"static std::array<double, {len(roots)}> local_{kind}({', '.join('double ' + name for name in names)}) {{",
            *emitter.lines,
            "return {" + ", ".join(emitter.reference(root) for root in roots) + "};",
            "}",
        ]
        identities[kind] = program.identity
    identity = canonical_hash({"schema": "grid-cpu-adjoint-v1", "programs": identities})
    lines += [
        f"// Shared grid graphs: {identity}",
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
        compiler,
        cache,
        iterations=3,
        max_bytes=8 * 1024 * 1024,
        max_pair_visits=100_000_000,
    ):
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

    def contract(self, points, centers, owners, seeds, *, coincident_tolerance=1e-12):
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
