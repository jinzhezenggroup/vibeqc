"""Native CPU execution of the existing scalar and compact coefficient DAGs.

This adapter reuses ScalarCEmitter, CppCompilerAdapter and the shared compiled
artifact cache. Scientific validation and AO contractions remain in the common
XC program; no external functional callback or second differentiation system
is introduced. CPU BLAS consumes the resulting bounded point coefficients.
"""

import ctypes as ct
import os
import tempfile
from hashlib import sha256
from pathlib import Path

import numpy as np

from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.native_runtime import compile_runtime
from vibeqc_compiler.common.paths import source_hashes
from vibeqc_compiler.common.provenance import canonical_hash, file_hash
from vibeqc_compiler.integral.scalar_c import ScalarCEmitter

from .contractions import ContractionProgram, _pack
from .program import validate_features


def _cache_source(path, source):
    """Publish complete compiler input without rewriting a cache hit.

    Concurrent misses publish identical bytes via same-directory replacement;
    another reader sees either no file or the complete immutable source. An
    existing mismatch remains an error rather than silently repairing a cache.
    """
    if path.exists():
        if path.read_text() != source:
            raise ValueError("native XC source identity mismatch")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".xc-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(source)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _variables(graph, roots):
    """Order only reachable scalar leaves; inactive tau slots need no loads."""
    return tuple(
        sorted(
            {
                str(graph.nodes[i].payload)
                for i in graph.topological_order(roots)
                if graph.nodes[i].operation == "variable"
            }
        )
    )


def emit_native(program):
    """Emit checked point loops around shared scalar C lowering, with no AO expansion."""
    graphs = {"xc_scalar": (program.program.graph, program.program.roots)}
    if program.contract.request.observable in ("potential", "geometry"):
        graphs["xc_coefficients"] = (
            program.coefficients.graph,
            program.coefficients.roots,
        )
    if program.response_coefficients is not None:
        graphs["xc_response_coefficients"] = (
            program.response_coefficients.graph,
            program.response_coefficients.roots,
        )
    if program.jet_pullback is not None:
        graphs["xc_jet_pullback"] = (
            program.jet_pullback.graph,
            program.jet_pullback.roots,
        )
    lines = [
        "// Generated XC roots; scalar functional provenance: external/libxc-7.0.0.",
        "#include <cmath>",
        "#include <cstddef>",
        "#include <cstdint>",
    ]
    layouts = {}
    for name, (graph, roots) in graphs.items():
        variables = _variables(graph, roots)
        emitter = ScalarCEmitter(
            graph,
            {key: f"input[{i} * npoint + point]" for i, key in enumerate(variables)},
        )
        emitter.emit(roots)
        lines.extend(
            [
                f'extern "C" int {name}(const double* input, size_t input_count, double* output, size_t output_count, size_t npoint) noexcept {{',
                f"  if (npoint > SIZE_MAX / {max(1, len(variables), len(roots))} || input_count != {len(variables)} * npoint || output_count != {len(roots)} * npoint) return -1;",
                "  if (!npoint) return 0;",
                "  if (!output || (input_count && !input)) return -1;",
                "  for (size_t point = 0; point < npoint; ++point) {",
                *emitter.lines,
            ]
        )
        for i, root in enumerate(roots):
            target = f"output[{i} * npoint + point]"
            lines.extend(
                [
                    f"    {target} = {emitter.reference(root)};",
                    f"    if (!std::isfinite({target})) return {i + 1};",
                ]
            )
        lines.extend(["  }", "  return 0;", "}"])
        layouts[name] = {
            "variables": variables,
            "outputs": len(roots),
            "ssa": graph.analyze_ssa(roots).to_payload(),
        }
    source = "\n".join(lines) + "\n"
    return source, {
        "schema": "vibeqc.xc-native-contractions.v1",
        "contract": program.contract.to_payload(),
        "layouts": layouts,
        "source_sha256": sha256(source.encode()).hexdigest(),
        "source_bytes": len(source.encode()),
        # Point-source cache reuse is valid for byte-identical native kernels,
        # but an endpoint also depends on host reductions, maps and assembly.
        "host_source_hashes": source_hashes("common", "xc", "dft"),
    }


class _PointFunction:
    """Checked contiguous arrays for a generated native point function."""

    def __init__(self, library, name, layout):
        self.variables = tuple(layout["variables"])
        self.outputs = layout["outputs"]
        self.function = getattr(library, name)
        self.function.argtypes = [
            ct.POINTER(ct.c_double),
            ct.c_size_t,
            ct.POINTER(ct.c_double),
            ct.c_size_t,
            ct.c_size_t,
        ]
        self.function.restype = ct.c_int

    def evaluate(self, variables, npoint):
        values = (
            np.stack(
                [np.broadcast_to(variables[name], (npoint,)) for name in self.variables]
            )
            if self.variables
            else np.empty((0, npoint))
        )
        values = immutable(values)
        output = np.empty((self.outputs, npoint))
        code = self.function(
            values.ctypes.data_as(ct.POINTER(ct.c_double)),
            values.size,
            output.ctypes.data_as(ct.POINTER(ct.c_double)),
            output.size,
            npoint,
        )
        if code:
            raise ArithmeticError(f"native XC point evaluation failed at output {code}")
        return output


class _NativeCoefficients:
    """Keep diagnostic/native binding and root-label interpretation identical."""

    def __init__(self, program, function):
        self.program, self.function = program, function

    def evaluate(self, gradient, v, *, delta_gradient=None, delta_v=None):
        variables, npoint = self.program.bind(
            gradient, v, delta_gradient=delta_gradient, delta_v=delta_v
        )
        return self.program.unpack(self.function.evaluate(variables, npoint), npoint)


class _NativeJetPullback:
    """Native execution of the identical generated two-leg AO pullback."""

    def __init__(self, program, function):
        self.program, self.function = program, function

    def evaluate(self, coefficients, work):
        variables, shape = self.program.bind(coefficients, work)
        return self.program.unpack(
            self.function.evaluate(variables, shape[0] * shape[1]), shape
        )


class NativeContractionProgram(ContractionProgram):
    """Explicit native CPU candidate using the same minimal scientific roots.

    Compilation is explicit and CPU-only. Repeated tile calls reuse verified
    code, not previous density/features. Use PreparedXCContractions to compose
    a complete bounded collocation/response/geometry endpoint.
    """

    def __init__(self, spec, observable="potential", *, compiler, cache):
        if not isinstance(compiler, CppCompilerAdapter):
            raise TypeError("native XC currently requires CppCompilerAdapter")
        super().__init__(spec, observable)
        source, self.metadata = emit_native(self)
        cache = Path(cache).resolve()
        identity = canonical_hash(self.metadata)
        directory = cache / "source" / identity
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "xc.cpp"
        _cache_source(path, source)
        self.artifact = compile_runtime(compiler, cache, path)
        if (
            self.artifact.metadata["identity"]["source"]
            != self.metadata["source_sha256"]
        ):
            raise ValueError("native XC compiled source identity mismatch")
        if file_hash(self.artifact.library) != self.artifact.metadata["binary_sha256"]:
            raise ValueError("native XC binary identity mismatch")
        self._library = ct.CDLL(str(self.artifact.library))
        functions = {
            name: _PointFunction(self._library, name, layout)
            for name, layout in self.metadata["layouts"].items()
        }
        self._scalar = functions["xc_scalar"]
        if "xc_coefficients" in functions:
            self.coefficients = _NativeCoefficients(
                self.coefficients, functions["xc_coefficients"]
            )
        if self.response_coefficients is not None:
            self.response_coefficients = _NativeCoefficients(
                self.response_coefficients, functions["xc_response_coefficients"]
            )
        if self.jet_pullback is not None:
            self.jet_pullback = _NativeJetPullback(
                self.jet_pullback, functions["xc_jet_pullback"]
            )

    def scalar_values(self, features):
        """Preserve audited physical-domain checks before native scalar evaluation."""
        x, active = validate_features(
            self.spec, _pack(self.spec, features), order=self.program.order
        )
        result = np.zeros((len(self.program.outputs), x.shape[1]))
        if np.any(active):
            variables = dict(zip(self.spec.features, x[:, active], strict=True))
            result[:, active] = self._scalar.evaluate(variables, int(active.sum()))
        return dict(zip(self.program.outputs, result, strict=True))
