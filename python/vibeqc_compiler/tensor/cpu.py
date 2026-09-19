"""Explicit bounded FP64 CPU lowering of ordinary TensorIR primitives.

This intentionally small materializing executor accepts input/constant, add,
multiply, reduce and einsum. It never dispatches on method or source names.
Unsupported semantics fail at generation, before any allocation or compilation.
"""

import ctypes as ct
from collections.abc import Mapping
from fractions import Fraction
from math import isfinite, prod
from pathlib import Path

import numpy as np

from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.native_runtime import compile_runtime
from vibeqc_compiler.common.paths import asset_path
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.common.source_cache import cache_source

from .cuda_emit import _coordinate, _flat
from .program import Program
from .types import checked_size


def _literal(pair):
    value = float(Fraction(*pair))
    if not isfinite(value):
        raise ValueError("CPU coefficient is not finite FP64")
    return value.hex()


def emit_cpu(program, *, max_bytes=8 * 1024 * 1024, max_work=100_000_000):
    """Return source and exact bounded storage/work requirements without runtime imports.

    No packed/symmetric semantics or implicit dtype conversion are admitted.
    Index expressions are shared with CUDA; CUDA emission bytes are unchanged.
    """
    if not isinstance(program, Program):
        raise TypeError("CPU lowering requires a TensorIR Program")
    checked_size(max_bytes, "CPU byte budget")
    checked_size(max_work, "CPU work budget")
    nodes = program.live_nodes
    if len(nodes) > 4096:
        raise ValueError("CPU program exceeds node budget")
    offsets, cursor, inputs, work = {}, 0, [], 0
    for node in nodes:
        if node.op not in {"input", "constant", "add", "multiply", "reduce", "einsum"}:
            raise ValueError(f"unsupported CPU primitive: {node.op}")
        if node.spec.dtype != "float64" or node.spec.symmetries:
            raise ValueError("CPU lowering requires plain real float64 semantics")
        offsets[node], cursor = cursor, cursor + node.spec.size
        if node.op == "input":
            inputs.append(node)
        reduction = 1
        if node.op == "reduce":
            reduction = prod(node.inputs[0].spec.shape[i] for i in node.attrs["axes"])
        if node.op == "einsum":
            domains = {}
            for child, labels in zip(node.inputs, node.attrs["labels"], strict=True):
                domains.update(zip(labels, child.spec.shape, strict=True))
            reduction = prod(
                v for k, v in domains.items() if k not in node.attrs["output"]
            )
        work += node.spec.size * reduction * max(1, len(node.inputs))
    ni = sum(n.spec.size for n in inputs)
    no = sum(n.spec.size for n in program.outputs.values())
    arena = cursor + no
    required = 8 * (arena + 2 * ni + 3 * no)
    if required > min(max_bytes, (1 << 63) - 1) or work > max_work:
        raise ValueError("CPU program exceeds byte/work budget")
    lines = [
        '#include "cpu_runtime.hpp"',
        f"// TensorIR {program.logical_hash}; provenance {canonical_hash(program.provenance)}",
        'extern "C" int tensor_cpu(const double* input, size_t ni, double* output, size_t no, size_t budget) noexcept {',
        f"return vibeqc_tensor_cpu::run(input, ni, output, no, budget, {ni}ULL, {no}ULL, {arena}ULL, {required}ULL,",
        "[](const double* input, double* p) {",
    ]
    input_cursor = 0
    for i, node in enumerate(nodes):
        a, shape = node.attrs, node.spec.shape

        def read(child, index="z"):
            return f"p[{offsets[child]}ULL + ({index})]"

        c = [_coordinate("z", shape, axis) for axis in range(len(shape))]
        body = []
        if node.op == "input":
            expression = f"input[{input_cursor}ULL + z]"
            input_cursor += node.spec.size
        elif node.op == "constant":
            values = ", ".join(_literal(v) for v in a["values"]) or "0.0"
            lines.append(f"static const double constant_{i}[] = {{{values}}};")
            expression = f"constant_{i}[z]"
        elif node.op == "add":
            body.append("double value = 0.0;")
            for child, factor in zip(node.inputs, a["coefficients"], strict=True):
                body.append(f"value += {_literal(factor)} * {read(child)};")
            expression = "value"
        elif node.op == "multiply":
            expression = f"{read(node.inputs[0])} * {read(node.inputs[1])}"
        else:
            if node.op == "reduce":
                child = node.inputs[0]
                rs = tuple(child.spec.shape[axis] for axis in a["axes"])
                coords, kept = [], iter(c)
                for axis in range(len(child.spec.shape)):
                    coords.append(
                        _coordinate("r", rs, a["axes"].index(axis))
                        if axis in a["axes"]
                        else next(kept)
                    )
                term = read(child, _flat(coords, child.spec.shape))
                factor = "1.0"
            else:
                domains = {}
                for child, labels in zip(node.inputs, a["labels"], strict=True):
                    domains.update(zip(labels, child.spec.shape, strict=True))
                reduced = tuple(k for k in sorted(domains) if k not in a["output"])
                rs = tuple(domains[k] for k in reduced)
                mapping = dict(zip(a["output"], c, strict=True))
                mapping.update(
                    (k, _coordinate("r", rs, axis)) for axis, k in enumerate(reduced)
                )
                term = " * ".join(
                    read(child, _flat([mapping[k] for k in labels], child.spec.shape))
                    for child, labels in zip(node.inputs, a["labels"], strict=True)
                )
                factor = _literal(a["coefficient"])
            body += [
                "double value = 0.0;",
                f"for (I r = 0; r < {prod(rs)}LL; ++r) value += {term};",
                "if (!std::isfinite(value)) return false;",
            ]
            expression = f"value * {factor}"
        lines += [
            f"for (I z = 0; z < {node.spec.size}LL; ++z) {{",
            *body,
            f"{read(node)} = {expression};",
            f"if (!std::isfinite({read(node)})) return false;",
            "}",
        ]
    for node in program.outputs.values():
        if node.spec.size:
            lines.append(
                f"std::copy_n(p + {offsets[node]}, {node.spec.size}, p + {cursor});"
            )
        cursor += node.spec.size
    lines += ["return true; });", "}"]
    source = (
        "\n".join(lines).replace(
            '#include "cpu_runtime.hpp"',
            '#include "cpu_runtime.hpp"\nusing I = long long;',
        )
        + "\n"
    )
    return source, {
        "input_count": ni,
        "output_count": no,
        "required_bytes": required,
        "scalar_work": work,
    }


class NativeTensorProgram:
    """Compiled CPU-only program, with checked feeds and transactional outputs."""

    def __init__(
        self,
        program,
        *,
        compiler,
        cache,
        max_bytes=8 * 1024 * 1024,
        max_work=100_000_000,
    ):
        if not isinstance(compiler, CppCompilerAdapter):
            raise TypeError("native TensorIR requires a CPU compiler adapter")
        source, self.resources = emit_cpu(
            program, max_bytes=max_bytes, max_work=max_work
        )
        self.program, self.max_bytes = program, max_bytes
        self.inputs = tuple(n for n in program.live_nodes if n.op == "input")
        self.identity = canonical_hash(
            {"program": program.to_payload(), "source": source}
        )
        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / (self.identity + ".cpp")
        cache_source(path, source)
        header = asset_path("src/tensor/cpu_runtime.hpp")
        self.artifact = compile_runtime(
            compiler,
            cache,
            path,
            headers=(header,),
            options=("-ffp-contract=off", f"-I{header.parent}"),
        )
        self.library = ct.CDLL(str(self.artifact.library))
        self.call = self.library.tensor_cpu
        self.call.argtypes = [
            ct.POINTER(ct.c_double),
            ct.c_size_t,
            ct.POINTER(ct.c_double),
            ct.c_size_t,
            ct.c_size_t,
        ]
        self.call.restype = ct.c_int

    def execute(self, feeds):
        """Reject wrong shapes/dtypes/NaNs before calling the checked native ABI."""
        if not isinstance(feeds, Mapping):
            raise TypeError("CPU tensor feeds must be a mapping")
        values = []
        for node in self.inputs:
            name = node.attrs["name"]
            if name not in feeds:
                raise ValueError(f"missing tensor input: {name}")
            value = np.asarray(feeds[name])
            if (
                value.shape != node.spec.shape
                or value.dtype != np.float64
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"invalid float64 tensor input: {name}")
            values.append(value.reshape(-1))
        packed = np.concatenate(values) if values else np.empty(0)
        output = np.empty(self.resources["output_count"])
        code = self.call(
            packed.ctypes.data_as(ct.POINTER(ct.c_double)),
            packed.size,
            output.ctypes.data_as(ct.POINTER(ct.c_double)),
            output.size,
            self.max_bytes,
        )
        if code:
            raise ValueError(f"native CPU tensor evaluation failed ({code})")
        result, cursor = {}, 0
        for name, node in self.program.outputs.items():
            result[name] = immutable(
                output[cursor : cursor + node.spec.size].reshape(node.spec.shape)
            )
            cursor += node.spec.size
        return result
